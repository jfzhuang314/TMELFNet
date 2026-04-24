import json
import os
from datetime import datetime
from pathlib import Path

import torch
from hydra import initialize_config_dir, compose
from omegaconf import DictConfig
import torch.nn as nn
import pytorch_lightning as pl
from torch_geometric.data import Data, DataLoader, Batch
from common.BaseTrainer import BaseTrainer
from common.interfaces import generate_exp_name
from configs.hydra_yaml_config import instantiate_from_yaml_config, print_yaml_config, instantiate_from_config, \
    get_yaml_config, print_yaml_cfg_dict

from datasets import T3DSPlusSegDataset
from datasets import TADPMSegDataset
from datasets.T3DSPlusSegDataset import T3DSPlusSegReader
from datasets.transform.TopoMorphCluster import TopoMorphCluster, AlphaShape
from evaluation.eval_TMC import batch_tmc_pred

from metrics.dice_loss import DiceLoss
from model._BinDTSegNet.BinDTSegNet import BinDTSegNet
from thop import profile
from metrics.CrossToothSegMetricsManager import CrossToothSegMetricsManager as SegMetricsManager, \
    DynamicRadiusNNEdgeMaskWithPos
from model._DilatedToothSegNet.cbl import CBLLoss
from external_libs import gen_utils as gu
from model._TMELFNet.tmelf_seg_module import TMELFSegNetTrainer

MODEL_REGISTRY = {
    "BinDTSegNet": BinDTSegNet,     # dialated tooth segmentation network
}

class BinDTSegModule(pl.LightningModule):
    """ 牙齿分割网络训练模块 - 采用TripleBinClsModule的逻辑结构重构"""
    def __init__(self, model_cfg: DictConfig, num_classes=15):
        super().__init__()

        self.config = model_cfg
        self.num_classes = num_classes
        self.input_dim = 6
        self.num_epochs = model_cfg.get("num_epochs", 30)
        self.lr = model_cfg.get("lr", 0.001)

        # CBL相关参数
        model_cfg.add_cbl = True if "_CBL" in model_cfg.loss_type else False

        self.add_cbl = model_cfg.add_cbl
        self.k = model_cfg.get("k", 16)  # 用于边界mask计算
        self.lam = model_cfg.get("lam", 0.1)  # CBL损失权重
        self.compute_edge_binary_iou = model_cfg.get("compute_edge_binary_iou", False)

        self.class_weights = torch.ones(self.num_classes, dtype=torch.float)

        # 模型
        self.net_params = model_cfg.net_params
        self.model = self.build_model(model_cfg)

        # 损失函数
        self.xe_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        self.criterion_cbl = CBLLoss().to(device)
        self.loss_type = model_cfg.loss_type

        # 指标管理器
        self.metrics_mgr = SegMetricsManager(
            task_type="segmentation",
            device=device,
            num_classes=self.num_classes,
            num_epochs=self.num_epochs,
            compute_edge_binary_iou = self.compute_edge_binary_iou,
        )

    def build_model(self, model_cfg):

         # 提取网络参数
        net_params = model_cfg.net_params
        kwargs = {
            **{k: net_params[k] for k in net_params.keys()}
        }

        model_class = MODEL_REGISTRY.get(model_cfg.net_name)
        if model_class is None:
            raise ValueError(f"Invalid model type: {model_cfg.net_name}. "
                             f"Available models: {list(MODEL_REGISTRY.keys())}")

        self.model = model_class(net_params)
        self.model = nn.DataParallel(self.model).to(device)

        return self.model

    def _forward_pass(self, inputs):
        """专门负责模型前向传播"""
        outputs = self.model(inputs)

        # 从二分类 logits 构建 15 分类 logits
        bg_logits = outputs['tg_seg_result']  # (B, 2, N)
        # 构建 seg_result: [背景类，牙齿类 1, 牙齿类 2, ..., 牙齿类 14]
        # 其中所有"牙齿类"都使用 bg_logits 的第 1 类
        bg_class = bg_logits[:, 0:1, :]  # (B, 1, N) - 背景类
        tooth_class = bg_logits[:, 1:2, :].repeat(1, 14, 1)  # (B, 14, N) - 14 个牙齿类

        outputs['seg_result'] = torch.cat([bg_class, tooth_class], dim=1)

        return {
            'seg_result': outputs['seg_result'].permute([0, 2, 1]),  # BNC
            'tg_seg_result': outputs['tg_seg_result'],  # B2N
            'edge_seg_result': outputs['edge_seg_result'],  # B2N
            'tte_seg_result': outputs['tte_seg_result'],  # B2N
            'cbl_inputs': outputs['cbl_inputs'] if 'cbl_inputs' in outputs else None  # PXO
        }

    def _generate_gt_labels(self, batch):
        """专门负责生成Ground Truth标签"""
        batch_size = batch.batch.max().item() + 1 if hasattr(batch, 'batch') else 1
        num_points = batch.pos.shape[0] // batch_size
        gt_labels = batch.y.to(device, dtype=torch.long)  # BN,

        one_hot_labels = (torch.nn.functional.one_hot(gt_labels, num_classes=self.num_classes).
                          reshape(batch_size, num_points, -1).to(device, dtype=torch.long))  # [B,N, 15]


        ground_truth = {
            'gt_labels': gt_labels,  # BN,
            'onehot_gt_labels': one_hot_labels,  # BNC
        }

        if hasattr(batch, 'gt_edge_mask'):
            gt_edge_mask = batch.gt_edge_mask.to(gt_labels.device, dtype=torch.float32)
            ground_truth['gt_edge_mask'] = gt_edge_mask.long()  # BN,

        if hasattr(batch, 'gt_tte_mask'):
            gt_tte_mask = batch.gt_tte_mask.to(gt_labels.device, dtype=torch.float32)
            ground_truth['gt_tte_mask'] = gt_tte_mask.long()  # BN,

        if 'gt_edge_mask' not in ground_truth or 'gt_tte_mask' not in ground_truth:
            gt_edge_mask, _ = DynamicRadiusNNEdgeMaskWithPos().generate_edge_masks(batch.pos, gt_labels)
            ground_truth['gt_edge_mask'] = gt_edge_mask.to(gt_labels.device, dtype=torch.float32).long()  # BN,
            ground_truth['gt_tte_mask'] = gt_tte_mask.to(gt_labels.device, dtype=torch.float32).long()  # BN,

        return ground_truth

    def _compute_losses(self, outputs, ground_truth, gt_edge_mask):
        """专门负责损失计算"""
        # 语义分割损失
        seg_outputs = outputs['seg_result'].reshape([-1, self.num_classes])  # BNC -> (B*N, C)
        gt_labels = ground_truth['gt_labels'].view(-1)  # BN -> (B*N,)
        gt_bg_mask = (gt_labels>0).long()
        gt_edge_mask = ground_truth.get('gt_edge_mask', None)
        gt_tte_mask = ground_truth.get('gt_tte_mask', None)

        seg_loss = self.xe_loss(seg_outputs, gt_labels)
        dice_loss = self.dice_loss(seg_outputs, gt_labels)

        # 背景分割损失
        bg_loss = torch.tensor(0.0, device=self.device)
        if 'tg_seg_result' in outputs and outputs['tg_seg_result'] is not None:
            tg_seg_result = outputs['tg_seg_result']  # 假设类别 0 是背景
            B, _, N = tg_seg_result.shape

            # 计算bg分割loss
            bg_logits = tg_seg_result.permute(0, 2, 1).contiguous().view(B * N, 2)  # (B*N, 2)
            gt_bg_labels = gt_bg_mask.view(B * N)
            bg_loss = self.xe_loss(bg_logits, gt_bg_labels)

        # 边界分割损失
        edge_loss = torch.tensor(0.0, device=self.device)
        if 'edge_seg_result' in outputs and outputs['edge_seg_result'] is not None:
            edge_seg_result = outputs['edge_seg_result']  # BCN
            B, _, N = edge_seg_result.shape

            # 计算边界分割loss
            edge_logits = edge_seg_result.permute(0, 2, 1).contiguous().view(B * N, 2)  # (B*N, 2)
            edge_labels = gt_edge_mask.view(B * N)  # (B*N,)
            # edge_loss = self.xe_loss(edge_logits, edge_labels)
            edge_loss = self.dice_loss(edge_logits, edge_labels)

        tte_loss = torch.tensor(0.0, device=self.device)
        if 'tte_seg_result' in outputs and outputs['tte_seg_result'] is not None:
            tte_seg_result = outputs['tte_seg_result']  # BCN
            B, _, N = tte_seg_result.shape

            # 计算TTE分割loss
            tte_logits = tte_seg_result.permute(0, 2, 1).contiguous().view(B * N, -1)  # (B*N, 15)
            tte_labels = gt_tte_mask.view(B * N)  # (B*N,)
            # tte_loss = self.xe_loss(tte_logits, tte_labels)
            tte_loss = self.dice_loss(tte_logits, tte_labels)

        # CBL损失（如果存在）
        cbl_loss = torch.tensor(0.0, device=self.device)
        if outputs.get('cbl_inputs') is not None and self.add_cbl:
            p1, x1, o1 = outputs['cbl_inputs']
            p1 = p1.contiguous()
            x1 = x1.contiguous()
            o1 = o1.contiguous()

            labels_flat = gt_labels.view(-1)
            cbl_loss = self.criterion_cbl([p1,x1,o1], labels_flat)

        # L2正则化损失
        l2_loss = self.regularization_losses()

        return {
            'seg_loss': seg_loss,
            'bg_loss': bg_loss,
            'dice_loss': dice_loss,
            'edge_loss': edge_loss,
            'tte_loss': tte_loss,
            'cbl_loss': cbl_loss,
            'l2_loss': l2_loss
        }

    def _aggregate_losses(self, losses):
        """聚合各项损失"""
        seg_loss = losses['seg_loss']
        bg_loss = losses['bg_loss']
        dice_loss = losses['dice_loss']
        edge_loss = losses['edge_loss']
        tte_loss = losses['tte_loss']
        cbl_loss = losses['cbl_loss']
        l2_loss = losses['l2_loss']

        lam_list = self.get_lam_list()

        loss_list = [seg_loss, dice_loss, l2_loss, cbl_loss, bg_loss,edge_loss, tte_loss]

        total_loss = 0.0
        for lam, loss in zip(lam_list, loss_list):
            total_loss += lam * loss

        return total_loss

    def get_lam_list(self):
        lam_bg = 1.0
        lam_edge = 0.2
        lam_dice = 0.5
        lam_cbl = 0.3
        if self.loss_type == 'SEG_TAE_TTE_CBL':
            lam_list = [1.0, lam_dice, 0.00001, lam_cbl, 0.0, lam_edge, lam_edge]
        elif self.loss_type == 'SEG_TAE_TTE':
            lam_list = [1.0, lam_dice, 0.00001, 0.0, 0.0, lam_edge, lam_edge]
        elif self.loss_type == 'SEG_CBL':
            lam_list = [1.0, lam_dice, 0.00001, lam_cbl,  0.0, 0.0, 0.0]
        elif self.loss_type == 'SEG':
            lam_list = [1.0, lam_dice, 0.00001, 0.0,  0.0, 0.0, 0.0]
        elif self.loss_type == 'BG_TAE_TTE_CBL':
            lam_list = [0.0, 0.0, 0.00001, lam_cbl,  lam_bg, 1.0, 1.0]
        elif self.loss_type == 'BG_TAE_TTE':
            lam_list = [0.0, 0.0, 0.00001, 0.0, lam_bg, 1.0, 1.0]
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}. "
                             f"Available loss types: TAE_TTE, SEG_TAE_TTE, SEG")
        return lam_list

    def _compute_metrics(self, predictions, ground_truth, **kwargs):
        """专门负责指标计算"""

        return self.metrics_mgr.compute_step_metrics(
            predictions,
            ground_truth,
            **kwargs
        )

    def _perform_step_forward(self, batch):
        """执行完整的前向传播流程"""
        # 1. 数据预处理
        B = batch.batch.max().item() + 1 if hasattr(batch, 'batch') else 1
        N = batch.pos.shape[0]
        x0 = batch.x.to(self.device, dtype=torch.float32)  # NC
        gt_labels = batch.y.to(self.device, dtype=torch.long)  # N,
        pos = batch.pos.to(self.device, dtype=torch.float32)  # N, 3

        # 转换为BCN格式
        inputs_bcn = x0.permute([1, 0]).reshape(B, self.input_dim, -1)  # BCN格式
        points_bnc = pos.reshape(B, -1, 3)  # B, N, 3

        # 2. 模型前向传播
        outputs = self._forward_pass(inputs_bcn)

        # 3. 生成真实标签
        ground_truth = self._generate_gt_labels(batch)

        # 4. 计算损失
        loss_components = self._compute_losses(outputs, ground_truth, points_bnc)
        total_loss = self._aggregate_losses(loss_components)

        # 5. 计算评估指标（step级别）
        # c2f_idx = batch.c2f_idx.squeeze().to(self.device)
        # outputs_ori, gt_ori = self.map_to_ori(batch, outputs, ground_truth)
        # metrics = self._compute_metrics(outputs_ori, gt_ori, pos=pos[c2f_idx,:])

        metrics = self._compute_metrics(outputs, ground_truth, pos=pos)


        return {
            'loss': total_loss,
            'outputs': outputs,
            'ground_truth': ground_truth,
            'loss_components': loss_components,
            'metrics': metrics
        }

    def map_to_ori(self, batch, outputs, ground_truth):
        B = batch.batch.max().item() + 1 if hasattr(batch, 'batch') else 1
        N = batch.pos.shape[0]

        seg_result = outputs['seg_result']
        edge_seg_result = outputs['edge_seg_result']
        tte_seg_result = outputs['tte_seg_result']

        c2f_idx = batch.c2f_idx.squeeze().to(seg_result.device)
        seg_result_onehot = torch.zeros(B, c2f_idx.shape[0], self.num_classes).to(seg_result.device)
        y_pred = seg_result.argmax(dim=-1, keepdim=True)  # [B, N, 1]
        y_pred_ori = y_pred[:,c2f_idx,:]
        seg_result_onehot.scatter_(2, y_pred_ori, 1)  # [B, N, C]

        outputs_ori={}
        outputs_ori['seg_result'] = seg_result_onehot
        outputs_ori['edge_seg_result'] = edge_seg_result[:, :, c2f_idx]
        outputs_ori['tte_seg_result'] = tte_seg_result[:, :, c2f_idx]

        y_ori = batch.y_ori.to(device, dtype=torch.long)  # BN,
        onehot_y_ori = (torch.nn.functional.one_hot(y_ori, num_classes=self.num_classes).
                        reshape(B, -1, self.num_classes).to(device, dtype=torch.long))  # [B,N, 15]
        gt_edge_mask = ground_truth['gt_edge_mask']
        gt_tte_mask = ground_truth['gt_tte_mask']
        gt_edge_mask_ori = gt_edge_mask[c2f_idx]
        gt_tte_mask_ori = gt_tte_mask[c2f_idx]

        gt_ori = {}
        gt_ori['gt_labels'] = y_ori
        gt_ori['onehot_gt_labels'] = onehot_y_ori
        gt_ori['gt_edge_mask'] = gt_edge_mask_ori
        gt_ori['gt_tte_mask'] = gt_tte_mask_ori

        return outputs_ori, gt_ori
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        outputs = self._perform_step_forward(batch)

        loss = outputs['loss']
        metrics_dict = outputs['metrics']

        # 更新训练指标
        self.metrics_mgr.update(
            loss.item(), metrics_dict, is_train=True
        )

        # 记录日志
        metrics = self.metrics_mgr.get_metrics_dict(
            loss.item(), metrics_dict, is_train=True
        )

        self.log_dict(metrics)
        return loss

    def validation_step(self, batch, batch_idx=0):
        outputs = self._perform_step_forward(batch)

        loss = outputs['loss']
        loss_components = outputs['loss_components']
        seg_loss = loss_components['seg_loss']
        dice_loss = loss_components['dice_loss']
        edge_loss = loss_components['edge_loss']
        tte_loss = loss_components['tte_loss']
        bg_loss = loss_components['bg_loss']
        cbl_loss = loss_components['cbl_loss']
        l2_loss = loss_components['l2_loss']

        metrics_dict = outputs['metrics']

        # 打印详细信息
        print(f"epoch:{self.current_epoch}-{batch.filename[0]}, "
              f"loss:{loss:.4f}=Lseg:{seg_loss:.4f} + Ldice:{dice_loss:.4f} "
              f"+ Lbg:{bg_loss:.4f} "
              f"+ Ledge:{edge_loss:.4f} + Ltte:{tte_loss:.4f} "
              f"+ Lcbl:{cbl_loss:.4f} + L2:{l2_loss:.4f}", end=' ')

        # 更新验证指标
        self.metrics_mgr.update(
            loss.item(), metrics_dict, is_train=False
        )

        # 记录日志
        metrics = self.metrics_mgr.get_metrics_dict(
            loss.item(), metrics_dict, is_train=False
        )

        self.log_dict(metrics)
        return metrics

    def predict_step(self, batch, batch_idx=0):
        outputs = self._perform_step_forward(batch)

        batch.outputs = outputs['outputs']
        batch.metrics = outputs['metrics']

        return batch

    def predict_step_forward(self, batch):
        B = batch.batch.max().item() + 1 if hasattr(batch, 'batch') else 1
        x0 = batch.x.to(self.device, dtype=torch.float32)  # NC

        # 转换为BCN格式
        inputs_bcn = x0.permute([1, 0]).reshape(B, self.input_dim, -1)  # BCN格式

        # 2. 模型前向传播
        outputs = self._forward_pass(inputs_bcn)

        batch.outputs = outputs

        return  batch

    def regularization_losses(self):
        """计算L2正则化损失"""
        l2_lambda = 0.0001
        l2_reg = torch.tensor(0., requires_grad=True).to(self.device)

        for name, param in self.model.named_parameters():
            if 'weight' in name:
                l2_reg = l2_reg + torch.norm(param, 2)

        return l2_lambda * l2_reg

    def on_train_epoch_end(self):
        print("Training epoch ended.")
        optimizer = self.trainer.optimizers[0]
        current_lr = optimizer.param_groups[0]['lr']
        # 打印epoch级别的学习率
        print(f"===== Epoch {self.current_epoch} End, Current LR: {current_lr:.6f} =====")


        torch.cuda.empty_cache()
        print(f"Train Epoch end, GPU内存清理完成: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")

    def on_validation_epoch_end(self):
        self.metrics_mgr.compute(is_train=True)
        self.metrics_mgr.compute(is_train=False)
        self.metrics_mgr._print_epoch_info(self.current_epoch)

        self.metrics_mgr.reset(is_train=True)
        self.metrics_mgr.reset(is_train=False)

        torch.cuda.empty_cache()
        print(f"Val Epoch end, GPU内存清理完成: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")

    def on_train_batch_start(self, batch, batch_idx):
        if batch_idx % 60 == 0:
            torch.cuda.empty_cache()
            print(f"Batch start, GPU内存清理完成: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")

    def on_validation_batch_start(self, batch, batch_idx):
        if batch_idx % 60 == 0:
            torch.cuda.empty_cache()
            print(f"Validation batch start, GPU内存清理完成: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), amsgrad=True, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=5e-5)
        return [optimizer], [scheduler]



class BinDTSegTrainer(BaseTrainer):
    """DilatedToothSegNet训练器，继承自BaseTrainer"""
    def __init__(self, config_path="../../configs", config_name="Module_DTSegNet"):
        super().__init__(config_path, config_name)

    # 获得模型实例
    def build_module(self, model_obj: DictConfig, ckpt_path=None):
        """获取DilatedToothSegNet模型实例"""
        # 如果指定了检查点路径，则从检查点中加载模型
        if ckpt_path is not None and os.path.exists(ckpt_path):
            try:
                print(f"\n开始加载检查点...")

                module =BinDTSegModule.load_from_checkpoint(
                    checkpoint_path=ckpt_path,
                    model_cfg=model_obj,  # 传递配置
                    num_classes=15,
                    strict=False,
                    map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                )
                print(f"✅ 成功加载检查点{ckpt_path}")

                return module
            except Exception as e:
                print(f"\n❌ 未能加载检查点{ckpt_path}: {e}")

        # 否则，创建新的模型实例
        name = model_obj["name"]
        if name == "BinDTSegNetModule":
            module = BinDTSegModule(model_obj, num_classes=15).to(device)
        else:
            raise ValueError(f"Invalid model name: {name}")

        return module

    # 准备数据
    def prepare_data(self, cfg_obj: DictConfig):
        if cfg_obj.dataset.name == "T3DSPlusSegDataset":
            _, self.train_loader, self.test_loader \
                = T3DSPlusSegDataset.get_dataloaders_from_cfg_obj(cfg_obj)
        elif cfg_obj.dataset.name == "TADPMSegDataset":
            _, self.train_loader, self.test_loader \
                = TADPMSegDataset.get_dataloaders_from_cfg_obj(cfg_obj)


def load_patchnet_model(config_path, config_name, model_path, model_name,net_name,net_para_name):

    trainer = TMELFSegNetTrainer(config_path=config_path, config_name=config_name)
    cfg_obj = trainer.cfg_obj

    # 2. 获取模型实例
    # model_path = cfg_obj.prediction.model_path
    # model_name = cfg_obj.prediction.model_name
    ckpt_path = os.path.join(model_path, model_name)

    cfg_obj.dataset.sample_ratio = 1.0
    cfg_obj.model.net_name = net_name
    cfg_obj.model.net_para_name = net_para_name

    # 3. 加载模型
    trainer.prepare_trainer(ckpt_path) if not trainer.trainer else None

    # 4. 获取模型实例
    module = trainer.module.to(device)  # 获取模型实例

    return module


def save_tmelf_batch_info_to_json(batch, save_dir, only_save_pred=False):
    """
    保存 TMELFNet batch 的关键信息到 JSON 文件

    Args:
        batch: 输入的 batch 数据
        save_dir: 保存目录
        only_save_pred: 是否仅保存预测结果
    """
    # 获取文件名
    filename = batch.filename[0] if isinstance(batch.filename, list) else batch.filename

    if only_save_pred:
        batch_data = {
            'filename': filename,
            'pos': batch.pos.cpu().numpy().tolist() if hasattr(batch, 'pos') and batch.pos is not None else None,
            'y': batch.y.cpu().numpy().tolist() if hasattr(batch, 'y') and batch.y is not None else None,
        }
    else:
        # 构建保存字典 - Ground Truth 数据
        batch_data = {
            'filename': filename,
            # 基本几何数据
            'y': batch.y.cpu().numpy().tolist() if hasattr(batch, 'y') and batch.y is not None else None,
            'face': batch.face.cpu().numpy().tolist() if hasattr(batch, 'face') and batch.face is not None else None,
            'pos': batch.pos.cpu().numpy().tolist() if hasattr(batch, 'pos') and batch.pos is not None else None,
            'normals': batch.normals.cpu().numpy().tolist() if hasattr(batch, 'normals') and batch.normals is not None else None,
            'x': batch.x.cpu().numpy().tolist() if hasattr(batch, 'x') and batch.x is not None else None,
            # 原始标签
            'y_ori': batch.y_ori.cpu().numpy().tolist() if hasattr(batch, 'y_ori') and batch.y_ori is not None else None,
            # 索引映射
            'f2c_idx': batch.f2c_idx.cpu().numpy().tolist() if hasattr(batch, 'f2c_idx') and batch.f2c_idx is not None else None,
            'c2f_idx': batch.c2f_idx.cpu().numpy().tolist() if hasattr(batch, 'c2f_idx') and batch.c2f_idx is not None else None,
            # 掩码标签
            'gt_edge_mask': batch.gt_edge_mask.cpu().numpy().tolist() if hasattr(batch, 'gt_edge_mask') and batch.gt_edge_mask is not None else None,
            'gt_tte_mask': batch.gt_tte_mask.cpu().numpy().tolist() if hasattr(batch, 'gt_tte_mask') and batch.gt_tte_mask is not None else None,
            # 实例信息
            'instances': batch.instances.cpu().numpy().tolist() if hasattr(batch, 'instances') and batch.instances is not None else None,
            'instance_centroids': batch.instance_centroids.cpu().numpy().tolist() if hasattr(batch, 'instance_centroids') and batch.instance_centroids is not None else None,
            'instance_labels': batch.instance_labels.cpu().numpy().tolist() if hasattr(batch, 'instance_labels') and batch.instance_labels is not None else None,
            'per_instance_count': batch.per_instance_count.cpu().numpy().tolist() if hasattr(batch, 'per_instance_count') and batch.per_instance_count is not None else None,
            'instance_count': batch.instance_count.cpu().numpy().tolist() if hasattr(batch, 'instance_count') and batch.instance_count is not None else None,
        }


    # 提取二值网络的输出结果
    if hasattr(batch, 'tmc_proposal'):
        # 获取最终分割结果 (BNC)
        # seg_result = outputs.get('seg_result', None)
        # if seg_result is not None:
        #     batch_data['pred_seg'] = torch.argmax(seg_result, dim=1).squeeze().cpu().numpy().tolist()
        tmc_proposal = batch.tmc_proposal
        outputs = tmc_proposal.get('outputs',None)
        # 获取 crops 字典
        if outputs is not None:
            batch_data['bin_nets'] = {
                # crop 索引映射 (1, clu, crop)
                'tg_seg_result': outputs['tg_seg_result'].cpu().numpy().tolist() if  outputs['tg_seg_result']is not None else None,
                'edge_seg_result': outputs['edge_seg_result'].cpu().numpy().tolist() if outputs['edge_seg_result'] is not None else None,
                'tte_seg_result': outputs['tte_seg_result'].cpu().numpy().tolist() if outputs['tte_seg_result'] is not None else None,
            }

    # 安全获取预测结果并进行存在性判断
    if hasattr(batch, 'outputs'):
        # 获取最终分割结果 (BNC)
        # seg_result = outputs.get('seg_result', None)
        # if seg_result is not None:
        #     batch_data['pred_seg'] = torch.argmax(seg_result, dim=1).squeeze().cpu().numpy().tolist()
        outputs = batch.outputs

        seg_result = outputs.get('seg_result', None)
        if seg_result is not None:
            batch_data['pred_seg'] = torch.argmax(seg_result, dim=1).squeeze().cpu().numpy().tolist()

        # 获取 crops 字典
        crops = outputs.get('crops', None)
        if crops is not None:
            batch_data['crops'] = {
                # crop 索引映射 (1, clu, crop)
                'cropped_nn_indexes': crops['cropped_nn_indexes'].cpu().numpy().tolist() if crops['cropped_nn_indexes'] is not None else None,
                # crop 位置 (clu, 3, crop)
                'cropped_pos': crops['cropped_pos'].cpu().numpy().tolist() if crops['cropped_pos'] is not None else None,
                # crop 的 GT 分割标签 (clu, 1, crop)
                'crop_gt_seg_label': crops['crop_gt_seg_label'].cpu().numpy().tolist() if crops['crop_gt_seg_label'] is not None else None,
                # crop 的 GT 二分类标签 (clu, 1, crop)
                'crop_gt_binseg_label': crops['crop_gt_binseg_label'].cpu().numpy().tolist() if crops['crop_gt_binseg_label'] is not None else None,
                # crop 的预测二分类 logits (clu, 2, crop)
                'crop_pred_seg_labels': crops['crop_pred_seg_labels'].cpu().numpy().tolist() if crops['crop_pred_seg_labels'] is not None else None,
                # 聚类 GT 标签 (clu,)
                'cluster_gt_seg_labels': crops['cluster_gt_seg_labels'].cpu().numpy().tolist() if crops['cluster_gt_seg_labels'] is not None else None,
            }


    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 保存为 JSON 文件
    json_path = os.path.join(save_dir, f"{filename}.json")

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(batch_data, f, ensure_ascii=False, indent=2)

    return json_path

def batch_tmc_patchnet_pred(ckpt_path,tmc, patchnet_module, test_loader,  save_json=True):
    """
    批量预测并评估 TMC 模型性能

    Args:
        ckpt_path: 模型检查点路径
        version_name: 版本名称
        tmc: TopoMorphCluster 实例
        patchnet: PatchNet
        test_loader: 测试数据加载器
        thre_dist: 距离阈值
        save_excel: 是否保存 Excel 结果
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_save_path = Path(ckpt_path).parent
    last_folder_name = Path(base_save_path).name

    json_save_dir = base_save_path / f"batch_tmelf_json_{timestamp}"
    print(f"\n📂 JSON 保存目录：{json_save_dir}")


    error_list = []

    for batch_idx, batch in enumerate(test_loader):
        print(f"评估样本 {batch_idx + 1}/{len(test_loader)}: {batch.filename}")

        try:
            data = tmc(batch)

            if not hasattr(data, 'tmc_proposal'):
                error_list.append(f"{batch.filename} 缺少 tmc_proposal")
                continue
            # tmc_proposal = data.tmc_proposal

            with torch.no_grad():
                tmc_batch = Batch.from_data_list([data])
                pred_data = patchnet_module.predict_step_forward(tmc_batch)

            if save_json:
                try:
                    json_path = save_tmelf_batch_info_to_json(pred_data,
                                                              json_save_dir,
                                                              only_save_pred=False)
                    print(f"✅ JSON 已保存：{os.path.basename(json_path)}")
                except Exception as e:
                    print(f"❌ JSON 保存失败：{e}")

        except Exception as e:
            error_list.append(batch.filename if hasattr(batch, 'filename') else f"Batch {batch_idx}")
            print(f"❌ Error in batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()

    for error_file in error_list:
        print(f"Error in {error_file}")


############################################################################
## Training Testing Prediction
############################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config_path = "../../configs"
config_name = "Module_BinDTSegNet"
def do_train():
    config_dir = os.path.abspath(config_path)

    # 可变参数
    # loss_type_list = ["SEG_CBL", "SEG", "SEG_TAE_TTE_CBL", "SEG_TAE_TTE"]
    # loss_type_list = ["SEG_TAE_TTE_CBL","SEG_TAE_TTE", "SEG_CBL","SEG"]
    loss_type_list = ["BG_TAE_TTE_CBL" ] # BG_TAE_TTE_CBL、BG_TAE_TTE


    for loss_type in loss_type_list:
        # 使用 Hydra compose API 动态覆盖配置
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            # 动态覆盖 loss_type
            cfg = compose(
                config_name=config_name,
                overrides=[
                    f"loss_type={loss_type}",
                ]
            )

            cfg_obj = instantiate_from_config(cfg)

            # 创建 trainer
            trainer = BinDTSegTrainer(config_path=config_path, config_name=config_name)
            trainer.set_cfg_obj(cfg_obj)    # 要用函数更新，否则会报错
            trainer.fit()

            # 验证
            # print(f"\n配置检查 (loss_type={loss_type}):")
            # print(f"  loss_type: {trainer.cfg_obj.loss_type}")
            # print(f"  model.loss_type: {trainer.cfg_obj.model.loss_type}")
            # print(f"  ckpt.exp_name: {trainer.cfg_obj.ckpt.exp_name}")
            # print(f"  logger.exp_args.lt: {trainer.cfg_obj.logger.exp_args.lt}")
            print(f"\n✅ 完成实验：{loss_type}-{trainer.cfg_obj.net_name}-{trainer.cfg_obj.net_para_name}\n")

def do_eval_TMC():
    trainer = BinDTSegTrainer(config_path=config_path, config_name=config_name)

    cfg_obj = trainer.cfg_obj

    # 2. 获取模型实例
    model_path = cfg_obj.prediction.tmc.model_path
    model_name = cfg_obj.prediction.tmc.model_name
    ckpt_path = os.path.join(model_path, model_name)
    version_name = generate_exp_name(cfg_obj.prediction.tmc.exp_args, add_timestamp=False, only_vals=True)

    # 更改数据集的采样比例
    cfg_obj.dataset.sample_ratio = 1.0
    # 3. 加载模型
    trainer.prepare_trainer(ckpt_path) if not trainer.trainer else None

    # 4. 获取模型实例
    module = trainer.module.to(device) # 获取模型实例

    # 5. TMC聚类模型，手动配置seg_predictor
    tmc = TopoMorphCluster(num_classes=cfg_obj.tmc_cluster.num_classes,
                 seg_predictor= module,
                 filter_type=cfg_obj.tmc_cluster.filter_type,  # bg, tae, tae_tte, tae_tte_gtte
                 mounted_segnet_outputs= cfg_obj.tmc_cluster.mounted_segnet_outputs,
                 mounted_mask_outputs=cfg_obj.tmc_cluster.mounted_mask_outputs,
                 mounted_knn_ica_meshes=cfg_obj.tmc_cluster.mounted_knn_ica_meshes,
                 aggr_type=cfg_obj.tmc_cluster.aggr_type,  # boolean, add
                 small_cluster_size_thre = cfg_obj.tmc_cluster.small_cluster_size_thre,
                 alpha = cfg_obj.tmc_cluster.alpha,
                 k = cfg_obj.tmc_cluster.k,
                 max_iter = cfg_obj.tmc_cluster.max_iter,
                 iter_thred_len_ratio = cfg_obj.tmc_cluster.iter_thred_len_ratio,
                 field_bound_ratio = cfg_obj.tmc_cluster.field_bound_ratio,
                 field_sigma_ratio = cfg_obj.tmc_cluster.field_sigma_ratio)


    # 5. 测试集数据加载
    test_loader = trainer.test_loader

    AlphaShape.build_alpha_shape.reset_count()
    # 6. 预测并保存结果
    batch_tmc_pred(ckpt_path, version_name, tmc, test_loader, thre_dist=0.10, save_excel=True)

    print("alpha shape call count",AlphaShape.build_alpha_shape.get_count())




def do_eval_tmc_patchnet():
    trainer = BinDTSegTrainer(config_path=config_path, config_name=config_name)

    cfg_obj = trainer.cfg_obj

    # 2. 获取模型实例
    tmc_model_path = cfg_obj.prediction.tmc.model_path
    tmc_model_name = cfg_obj.prediction.tmc.model_name
    tmc_ckpt_path = os.path.join(tmc_model_path, tmc_model_name)
    version_name = generate_exp_name(cfg_obj.prediction.tmc.exp_args, add_timestamp=False, only_vals=True)

    # 更改数据集的采样比例
    cfg_obj.dataset.sample_ratio = 1.0
    # 3. 加载模型
    trainer.prepare_trainer(tmc_ckpt_path) if not trainer.trainer else None

    # 4. 获取模型实例
    bin_dtseg_module = trainer.module.to(device) # 获取模型实例

    # 5. TMC聚类模型，手动配置seg_predictor
    tmc = TopoMorphCluster(num_classes=cfg_obj.tmc_cluster.num_classes,
                 seg_predictor= bin_dtseg_module,
                 filter_type=cfg_obj.tmc_cluster.filter_type,  # bg, tae, tae_tte, tae_tte_gtte
                 mounted_segnet_outputs= cfg_obj.tmc_cluster.mounted_segnet_outputs,
                 mounted_mask_outputs=cfg_obj.tmc_cluster.mounted_mask_outputs,
                 mounted_knn_ica_meshes=cfg_obj.tmc_cluster.mounted_knn_ica_meshes,
                 aggr_type=cfg_obj.tmc_cluster.aggr_type,  # boolean, add
                 small_cluster_size_thre = cfg_obj.tmc_cluster.small_cluster_size_thre,
                 alpha = cfg_obj.tmc_cluster.alpha,
                 k = cfg_obj.tmc_cluster.k,
                 max_iter = cfg_obj.tmc_cluster.max_iter,
                 iter_thred_len_ratio = cfg_obj.tmc_cluster.iter_thred_len_ratio,
                 field_bound_ratio = cfg_obj.tmc_cluster.field_bound_ratio,
                 field_sigma_ratio = cfg_obj.tmc_cluster.field_sigma_ratio)

    # 加载patch_based网络模型
    patch_net_obj = cfg_obj.prediction.patch_net
    patchnet_module = load_patchnet_model( patch_net_obj.config_path,
                                          patch_net_obj.config_name,
                                          patch_net_obj.model_path,
                                          patch_net_obj.model_name,
                                          patch_net_obj.net_name,
                                          patch_net_obj.net_para_name)

    # print(patchnet_module)

    # 5. 测试集数据加载
    test_loader = trainer.test_loader

    batch_tmc_patchnet_pred(tmc_ckpt_path, tmc, patchnet_module, test_loader, save_json=True)

def main():

    # 打印配置信息
    print_yaml_config(config_path="../../configs", config_name="Module_BinDTSegNet")
    cfg_obj = instantiate_from_yaml_config(config_path="../../configs", config_name="Module_BinDTSegNet")
    print(cfg_obj)

    # 测试模型
    # test_get_model()
    # do_train()
    do_eval_TMC()
    # do_eval_tmc_patchnet()

    # test_predict_1()

    # test_reader()
    # test_reader_2()

if __name__ == '__main__':
    main()
