import os

from hydra import initialize_config_dir, compose
from omegaconf import DictConfig

import torch.nn as nn
import pytorch_lightning as pl

from common.BaseTrainer import BaseTrainer
from configs.hydra_yaml_config import instantiate_from_yaml_config, instantiate_from_config
from datasets import T3DSPlusSegDataset
from datasets import TADPMSegDataset
# from datasets.T3DSPlusSegDataset import load_datasets_from_cfg_obj
from metrics.dice_loss import DiceLoss

import torch

from model._cross_tooth.cbl import CBLLoss
from metrics.CrossToothSegMetricsManager import CrossToothSegMetricsManager as SegMetricsManager, \
    DynamicRadiusNNEdgeMaskWithPos

# 不同模型的加载
from model._cross_tooth.point_transformer_segnet import PointTransformerSeg50, PointTransformerSeg, Bottleneck

MODEL_REGISTRY = {
    "PTSegNet": PointTransformerSeg,
}


class PTSegNetModule(pl.LightningModule):
    """ Point Transformer实例分割模块 - 完整实现 """

    def __init__(self, model_cfg, num_classes=15):
        super().__init__()

        self.config = model_cfg
        self.num_classes = model_cfg.num_classes
        self.input_dim = model_cfg.feature_dim
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
        self.dice_loss = DiceLoss(ignore_index=0)
        self.criterion_cbl = CBLLoss().to(device)
        self.loss_type = model_cfg.loss_type

        # 指标管理器
        self.metrics_mgr = SegMetricsManager(
            task_type="segmentation",
            device=device,
            num_classes=self.num_classes,
            num_epochs=self.num_epochs,
            compute_edge_binary_iou=self.compute_edge_binary_iou,
        )

    def build_model(self, model_cfg):

        keys = ["planes", "stride", "nsample", "share_planes", "in_channels",
                "num_classes", "add_cbl", "enable_pic_feat", "pretrain"]

        # 提取网络参数
        net_params = model_cfg.net_params
        kwargs = {
            "block_net": Bottleneck,
            **{k: net_params[k] for k in keys}
        }

        model_class = MODEL_REGISTRY.get(model_cfg.net_name)
        if model_class is None:
            raise ValueError(f"Invalid model type: {model_cfg.net_name}. "
                             f"Available models: {list(MODEL_REGISTRY.keys())}")

        self.model = model_class(**kwargs)
        self.model = nn.DataParallel(self.model).to(device)

        return self.model

    def _forward_pass(self, inputs):
        """
        Args:
            inputs: BCN格式, where B is batch size, C is feature dimension
        Returns:
        """

        outputs = self.model(inputs)

        return {
            'seg_result': outputs['seg_result'].permute([0, 2, 1]),  # BNC
            'edge_seg_result': outputs['edge_seg_result'],  # B2N
            'tte_seg_result': outputs['tte_seg_result'],  # B2N
            'cbl_inputs': outputs['cbl_inputs']  # PXO
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
            'onehot_gt_labels': one_hot_labels  # BNC
        }

        if hasattr(batch, 'gt_edge_mask'):
            gt_edge_mask = batch.gt_edge_mask.to(gt_labels.device, dtype=torch.float32)
            ground_truth['gt_edge_mask'] = gt_edge_mask.long()  # BN,

        if hasattr(batch, 'gt_tte_mask'):
            gt_tte_mask = batch.gt_tte_mask.to(gt_labels.device, dtype=torch.float32)
            ground_truth['gt_tte_mask'] = gt_tte_mask.long()  # BN,

        if 'gt_edge_mask' not in ground_truth or 'gt_tte_mask' not in ground_truth:
            gt_edge_mask, gt_tte_mask = DynamicRadiusNNEdgeMaskWithPos().generate_edge_masks(batch.pos, gt_labels)
            ground_truth['gt_edge_mask'] = gt_edge_mask.squeeze().to(gt_labels.device, dtype=torch.float32).long()  # BN,
            ground_truth['gt_tte_mask'] = gt_tte_mask.squeeze().to(gt_labels.device, dtype=torch.float32).long()  # BN,

        return ground_truth

    def _compute_losses(self, outputs, ground_truth, gt_edge_mask):
        """专门负责损失计算"""
        # 语义分割损失
        seg_outputs = outputs['seg_result'].reshape([-1, self.num_classes])  # BNC -> (B*N, C)
        gt_labels = ground_truth['gt_labels'].view(-1)  # BN -> (B*N,)
        gt_edge_mask = ground_truth.get('gt_edge_mask', None)
        gt_tte_mask = ground_truth.get('gt_tte_mask', None)

        seg_loss = self.xe_loss(seg_outputs, gt_labels)
        dice_loss = self.dice_loss(seg_outputs, gt_labels)

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
            labels_flat = gt_labels.view(-1)
            # pos_mask = gt_edge_mask.view(B * N) if gt_edge_mask is not None else None
            cbl_loss = self.criterion_cbl([p1, x1, o1], labels_flat,None)

        # L2正则化损失
        # l2_loss = self.regularization_losses()
        l2_loss = torch.tensor(0.0, device=self.device)

        return {
            'seg_loss': seg_loss,
            'dice_loss': dice_loss,
            'edge_loss': edge_loss,
            'tte_loss': tte_loss,
            'cbl_loss': cbl_loss,
            'l2_loss': l2_loss
        }

    def _aggregate_losses(self, losses):
        """聚合各项损失"""
        seg_loss = losses['seg_loss']
        dice_loss = losses['dice_loss']
        edge_loss = losses['edge_loss']
        tte_loss = losses['tte_loss']
        cbl_loss = losses['cbl_loss']
        l2_loss = losses['l2_loss']

        # if self.current_epoch < 10:
        #     lam_list = [1.0, 0.0, 0.0001, self.lam, 0.0, 0.0]
        # else:
        #     lam_progress = min(1.0, self.current_epoch/10.0-1.0)
        #     lam_list = [1.0, 0.0, 0.0001, self.lam, lam_progress, 0.1*lam_progress]
        lam_list = self.get_lam_list()

        loss_list = [seg_loss, dice_loss, l2_loss, cbl_loss, edge_loss, tte_loss]

        total_loss = 0.0
        for lam, loss in zip(lam_list, loss_list):
            total_loss += lam * loss

        return total_loss

    def get_lam_list(self):
        if self.loss_type == 'SEG_TAE_TTE_CBL':
            lam_list = [1.0, 1.0, 0.0, 0.2, 0.5, 0.5]
        elif self.loss_type == 'SEG_TAE_TTE':
            lam_list = [1.0, 1.0, 0.00001, 0.0, 0.5, 0.5]
        elif self.loss_type == 'SEG_CBL':
            lam_list = [1.0, 0.0, 0.00001, 0.2, 0.0, 0.0]
        elif self.loss_type == 'SEG':
            lam_list = [1.0, 0.0, 0.00001, 0.0, 0.0, 0.0]
        elif self.loss_type == 'TAE_TTE_CBL':
            lam_list = [0.0, 0.0, 0.00001, 0.1, 0.5, 0.5]
        elif self.loss_type == 'TAE_TTE':
            lam_list = [0.0, 0.0, 0.00001, 0.0, 0.5, 0.5]
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}. "
                             f"Available loss types: SEG_TAE_TTE_CBL, SEG_TAE_TTE, SEG_CBL, SEG, TAE_TTE_CBL, TAE_TTE")
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
        x0 = batch.x.to(self.device, dtype=torch.float32)  # NC
        gt_labels = batch.y.to(self.device, dtype=torch.long)  # N,
        pos = batch.pos.to(self.device, dtype=torch.float32)  # N, 3

        # 转换为BCN格式
        inputs_bcn = x0.reshape(B, -1, self.input_dim).permute(0, 2, 1)  # BCN格式
        points_bnc = pos.reshape(B, -1, 3)  # B, N, 3

        # 2. 模型前向传播
        outputs = self._forward_pass(inputs_bcn)

        # 3. 生成真实标签
        ground_truth = self._generate_gt_labels(batch)

        # 4. 计算损失
        loss_components = self._compute_losses(outputs, ground_truth, points_bnc)
        total_loss = self._aggregate_losses(loss_components)

        # 5. 计算评估指标
        metrics = self._compute_metrics(outputs, ground_truth, pos=pos)

        return {
            'loss': total_loss,
            'outputs': outputs,
            'ground_truth': ground_truth,
            'loss_components': loss_components,
            'metrics': metrics
        }

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx=0):
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
        cbl_loss = loss_components['cbl_loss']
        l2_loss = loss_components['l2_loss']

        metrics_dict = outputs['metrics']

        # 打印详细信息
        print(f"epoch:{self.current_epoch}-{batch.filename[0]}, "
              f"loss:{loss:.4f}=Lseg:{seg_loss:.4f} + Ldice:{dice_loss:.4f} "
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

        # return {
        #     'loss': total_loss,
        #     'outputs': outputs,
        #     'ground_truth': ground_truth,
        #     'loss_components': loss_components,
        #     'metrics': metrics
        # }

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

    # def regularization_losses(self):
    #     """计算L2正则化损失"""
    #     l2_lambda = 0.0001
    #     l2_reg = torch.tensor(0., requires_grad=True).to(self.device)
    #
    #     for name, param in self.model.named_parameters():
    #         if 'weight' in name:
    #             l2_reg = l2_reg + torch.norm(param, 2)
    #
    #     return l2_lambda * l2_reg

    def on_train_epoch_end(self):
        print("Training epoch ended.")
        self.metrics_mgr.compute(is_train=True)

        torch.cuda.empty_cache()
        print(f"Train Epoch end, GPU内存清理完成: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")

    def on_validation_epoch_end(self):
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
        # 按照CrossTooth的配置，使用Adam优化器和余弦退火调度器
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=1e-6)


        return [optimizer], [scheduler]


class PTSegNetTrainer(BaseTrainer):
    """PTInsSeg训练器，继承自BaseTrainer"""

    def __init__(self, config_path="../../configs", config_name="Module_CrossTooth"):
        super().__init__(config_path, config_name)

    # 获得模型实例
    def build_module(self, model_obj: DictConfig, ckpt_path=None):
        """获取PTSegNetModule模型实例"""
        # 如果指定了检查点路径，则从检查点中加载模型
        if ckpt_path is not None and os.path.exists(ckpt_path):
            try:
                print(f"\n开始加载检查点...")

                module = PTSegNetModule.load_from_checkpoint(
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
        if name == "PTSegNetModule":
            module = PTSegNetModule(model_obj, num_classes=15).to(device)
        else:
            raise ValueError(f"Invalid model name: {name}")

        return module

    # 准备数据
    def prepare_data(self, cfg_obj: DictConfig):
        if cfg_obj.dataset.name == "T3DSPlusSegDataset":
            self.data_module, self.train_loader, self.test_loader \
                = T3DSPlusSegDataset.get_dataloaders_from_cfg_obj(cfg_obj)
        elif cfg_obj.dataset.name == "T3DSSegDataset":
            self.data_module, self.train_loader, self.test_loader \
                = T3DSPlusSegDataset.get_dataloaders_from_cfg_obj(cfg_obj)
        elif cfg_obj.dataset.name == "TADPMSegDataset":
            self.data_module, self.train_loader, self.test_loader \
                = TADPMSegDataset.get_dataloaders_from_cfg_obj(cfg_obj)
        else:
            raise ValueError(f"Invalid dataset name: {cfg_obj.dataset.name}")

############################################################################
## Training Testing Prediction
############################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config_path = "../../configs"
config_name = "Module_CrossTooth"


# def do_train():
#     loss_type_list = ["SEG_TAE_TTE", "SEG_CBL", "SEG", "TAE_TTE_CBL", "TAE_TTE"]
#
#     for loss_type in loss_type_list:
#         trainer = PTSegNetTrainer(config_path=config_path, config_name=config_name)
#         trainer.cfg_obj.loss_type = loss_type
#         trainer.fit()
def do_train():
    """
    依次使用不同的 loss_type 进行训练
    关键：每次都要重新加载配置，确保所有引用都正确更新
    """
    config_dir = os.path.abspath(config_path)

    # 可变参数
    # loss_type_list = ["SEG_TAE_TTE_CBL","SEG_TAE_TTE","TAE_TTE_CBL", "TAE_TTE",  "SEG_CBL", "SEG"]
    # loss_type_list = ["SEG_TAE_TTE_CBL","SEG_TAE_TTE","TAE_TTE_CBL", "TAE_TTE",  "SEG_CBL", "SEG"]
    loss_type_list = ["SEG_TAE_TTE_CBL"]


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
            trainer = PTSegNetTrainer(config_path=config_path, config_name=config_name)
            trainer.set_cfg_obj(cfg_obj)    # 要用函数更新，否则会报错
            print(trainer.cfg_obj)
            trainer.fit()

            # 验证
            # print(f"\n配置检查 (loss_type={loss_type}):")
            # print(f"  loss_type: {trainer.cfg_obj.loss_type}")
            # print(f"  model.loss_type: {trainer.cfg_obj.model.loss_type}")
            # print(f"  ckpt.exp_name: {trainer.cfg_obj.ckpt.exp_name}")
            # print(f"  logger.exp_args.lt: {trainer.cfg_obj.logger.exp_args.lt}")
            print(f"\n✅ 完成实验：{loss_type}-{trainer.cfg_obj.net_name}-{trainer.cfg_obj.net_para_name}\n")

def do_pred():
    trainer = PTSegNetTrainer(config_path=config_path, config_name=config_name)

    ckpt_path = ("C:\\Users\\win10\\DualMesh\\pynb\\paraTeeth\\model\\checkpoints\\"
                 r"exp_CrossTooth_fps_16000\\"
                 r"0.001lr_0.1ratio_SEG_TAE_TTE_T3DSPlusSegDataset_PTSegNetModule_PTSegNet_seg50\\"
                 r"model-22-0.5490.ckpt"
                 )

    print(ckpt_path)
    trainer.prepare_trainer(ckpt_path=ckpt_path)

    # predictions = trainer.trainer.predict(
    #     trainer.module,
    #     dataloaders=trainer.test_loader,
    #     ckpt_path=trainer.loaded_ckpt
    # )
    predictions = trainer.predict(
        predict_loader=trainer.test_loader,
        ckpt_path=trainer.loaded_ckpt
    )

    # for pred in predictions:
    #     print(pred['metrics'])

    cfg_obj = trainer.get_cfg_obj()
    # pred_label_saver = cfg_obj.vis.fps_16000.pred_label_saver
    # pred_edge_saver = cfg_obj.vis.fps_16000.pred_edge_saver
    # pred_tte_saver = cfg_obj.vis.fps_16000.pred_tte_saver
    pred_label_saver = cfg_obj.vis.qslim_10000.pred_label_saver
    pred_edge_saver = cfg_obj.vis.qslim_10000.pred_edge_saver
    pred_tte_saver = cfg_obj.vis.qslim_10000.pred_tte_saver
    pred_bg_saver = cfg_obj.vis.qslim_10000.pred_bg_saver

    for pred_data in predictions:
        bg = (pred_data.outputs['seg_result'].argmax(dim=-1) > 0).squeeze()
        tae_lf = pred_data.outputs['edge_seg_result']
        tte_lf = pred_data.outputs['tte_seg_result']
        pred_data.y = bg.long()
        # pred_data.gt_edge_mask = pred_data.outputs['edge_seg_result'].argmax(dim=1).squeeze()
        # pred_data.gt_tte_mask = pred_data.outputs['tte_seg_result'].argmax(dim=1).squeeze()

        pred_data.filename = pred_data.filename[0] if isinstance(pred_data.filename, list) else pred_data.filename
        # pred_bg_saver(pred_data)

        pred_data.y = ((tae_lf + tte_lf).argmax(dim=1).squeeze() | bg < 1).long()
        pred_edge_saver(pred_data)

        # pred_edge_saver(pred_data)
        # pred_tte_saver(pred_data)


def main():
    # 打印配置信息
    # print_yaml_config(config_path="../../configs", config_name="Module_CrossTooth")
    # cfg_obj = instantiate_from_yaml_config(config_path=config_path, config_name=config_name)
    # print(cfg_obj)

    # 测试模型
    # test_get_model()
    do_train()
    # do_pred()


if __name__ == '__main__':
    main()
