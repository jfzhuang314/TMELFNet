"""
@File    :   skip_mask.py
这一版本主要简化了输入特征的提取，仅仅使用两个edgeconv层提取特征；
"""

from omegaconf import DictConfig, OmegaConf
from thop import profile

import torch.nn as nn
import pytorch_lightning as pl

from torch_geometric.data import Data, DataLoader

from common.BaseTrainer import BaseTrainer
from configs.hydra_yaml_config import instantiate_from_yaml_config
from datasets import TADPMSegDataset
from datasets import T3DSPlusSegDataset
from datasets.transform.t3dsPlus_transforms import DynamicRadiusNNEdgeMask
from metrics.dice_loss import DiceLoss

import torch
from model2._PT.cbl_point_transformer.cbl_point_transformer_module import get_model, get_model_by_cfg_dict
from metrics.CrossToothSegMetricsManager import CrossToothSegMetricsManager as SegMetricsManager

class PointTransformerNet(torch.nn.Module):
    def __init__(self, model_cfg: DictConfig=None, num_classes=15, num_feat=6):
        super().__init__()
        self.input_dim = num_feat
        self.output_dim = num_classes
        self.model_cfg = model_cfg

        if self.model_cfg is not None:
            model_dict = OmegaConf.to_container(self.model_cfg, resolve=True)
            self.PT_model = get_model_by_cfg_dict(**model_dict)
        else:
            self.PT_model = self.build_default_model()

    def build_default_model(self):
        model_parameter = {
            "input_feat": 6,
            "stride": [1, 4, 4, 4, 4],
            "nstride": [2, 2, 2, 2],
            "nsample": [36, 24, 24, 24, 24],
            "blocks": [2, 3, 4, 6, 3],
            "block_num": 5,
            "planes": [32, 64, 128, 256, 512],
            "contain_weight": False,
            "crop_sample_size": 3072,
        }

        return get_model(c=self.input_dim, k=self.output_dim, **model_parameter)

    def forward(self, inputs):
        """
        inputs: list
            inputs[0] => B, 6, 24000 : point features
            inputs[1]=> B, 1, 24000 : ground truth segmentation (optional)
        """
        outputs = {}

        if len(inputs) >= 2:
            cbl_loss, sem_1, offset_1, mask_1, first_features = self.PT_model([inputs[0], inputs[1]])
            outputs.update({
                "cbl_loss": cbl_loss,
                "sem_1": sem_1,    # BCN  [1,15,24000]
                "offset_1": offset_1,
                "mask_1": mask_1,
                "first_features": first_features,
                "cls_pred": sem_1
            })
        else:
            sem_1, offset_1, mask_1, first_features = self.PT_model([inputs[0]])
            outputs.update({
                "sem_1": sem_1,
                "offset_1": offset_1,
                "mask_1": mask_1,
                "first_features": first_features,
                "cls_pred": sem_1
            })

        return outputs


class PTSegNetModule(pl.LightningModule):
    """ Point Transformer实例分割模块 - 参照DilatedToothSegModule重构 """

    def __init__(self, model_cfg, num_classes=15):
        super().__init__()

        self.config = model_cfg
        self.num_classes = model_cfg.num_classes
        self.input_dim = model_cfg.feature_dim
        self.num_epochs = model_cfg.get("num_epochs", 30)
        self.lr = model_cfg.get("lr", 0.001)

        self.class_weights = torch.ones(self.num_classes, dtype=torch.float)

        # 模型
        self.model = PointTransformerNet(model_cfg, num_classes=self.num_classes, num_feat=self.input_dim)
        self.model = nn.DataParallel(self.model).to(device)

        # 损失函数
        self.xe_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()

        # 指标管理器
        self.metrics_mgr = SegMetricsManager(
            task_type="segmentation",
            device=device,
            num_classes=self.num_classes,
            num_epochs=self.num_epochs
        )

    def _forward_pass(self, inputs):
        """专门负责模型前向传播"""
        outputs = self.model(inputs)  # BCN格式

        # 添加一个键
        outputs['seg_result'] = outputs['sem_1'].permute([0, 2, 1])  # BNC

        return outputs

    def _generate_gt_labels(self, batch):
        batch_size = batch.batch.max().item() + 1 if hasattr(batch, 'batch') else 1
        num_points = batch.pos.shape[0] // batch_size
        gt_labels = batch.y.to(device, dtype=torch.long)  # BN,

        one_hot_labels = (torch.nn.functional.one_hot(gt_labels, num_classes=self.num_classes).
                          reshape(batch_size, num_points,-1).to(device, dtype=torch.long))  # [B,N, 15]

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
            gt_edge_mask, gt_tte_mask = DynamicRadiusNNEdgeMask().generate_edge_masks(batch.pos, gt_labels)
            ground_truth['gt_edge_mask'] = gt_edge_mask.squeeze().to(gt_labels.device, dtype=torch.float32).long()  # BN,
            ground_truth['gt_tte_mask'] = gt_tte_mask.squeeze().to(gt_labels.device, dtype=torch.float32).long()  # BN,

        return ground_truth

    def _compute_losses(self, outputs, ground_truth):
        """专门负责损失计算"""
        # 语义分割损失
        outputs1 = outputs['sem_1'].permute([0, 2, 1]).reshape([-1, self.num_classes])
        gt_labels = ground_truth['gt_labels'].view(-1)

        seg_loss = self.xe_loss(outputs1, gt_labels)
        dice_loss = self.dice_loss(outputs1, gt_labels)

        # CBL损失（如果存在）
        cbl_loss = outputs.get('cbl_loss', torch.tensor(0.0, device=self.device))
        if isinstance(cbl_loss, torch.Tensor):
            cbl_loss = cbl_loss.sum()
        else:
            cbl_loss = torch.tensor(0.0, device=self.device)

        # L2正则化损失
        l2_loss = self.regularization_losses()

        return {
            'seg_loss': seg_loss,
            'dice_loss': dice_loss,
            'cbl_loss': cbl_loss,
            'l2_loss': l2_loss
        }

    def _aggregate_losses(self, losses):
        """聚合各项损失"""
        seg_loss = losses['seg_loss']
        dice_loss = losses['dice_loss']
        cbl_loss = losses['cbl_loss']
        l2_loss = losses['l2_loss']

        # 权重参数
        alpha, beta = 0.8, 0.2  # 可根据需要调整

        lambda0, lambda1, lambda2 = 1.0, 0.0001, 0.1  # 主损失、正则化损失和CBL损失的权重
        total_loss = lambda0 * (alpha * seg_loss + beta * dice_loss) + lambda1 * l2_loss + lambda2 * cbl_loss

        return total_loss

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
        gt_labels = batch.y.to(self.device, dtype=torch.long)  # N,1
        pos = batch.pos.to(device, dtype=torch.float32)  # BN, 3

        # inputs = [x0.permute([1, 0]).reshape(B, self.input_dim, -1)]  # BCN格式
        inputs_with_gt = [x0.permute([1, 0]).reshape(B, self.input_dim, -1), gt_labels.reshape([B, 1, -1])]

        # 2. 模型前向传播
        outputs = self._forward_pass(inputs_with_gt)

        # 3. 生成真实标签
        ground_truth = self._generate_gt_labels(batch)

        # 4. 计算损失
        loss_components = self._compute_losses(outputs, ground_truth)
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

    def validation_step(self, batch, batch_idx):
        outputs = self._perform_step_forward(batch)

        loss = outputs['loss']
        loss_components = outputs['loss_components']
        seg_loss, dice_loss = loss_components['seg_loss'], loss_components['dice_loss']

        metrics_dict = outputs['metrics']

        # 打印详细信息
        print(f"epoch:{self.current_epoch}-{batch.filename[0]}, "
              f"loss:{loss:.4f}=Lseg:{seg_loss:.4f} + Ldice:{dice_loss:.4f}", end=' ')

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

    def predict_step(self, batch, batch_idx):
        outputs = self._perform_step_forward(batch)

        loss = outputs['loss']
        metrics_dict = outputs['metrics']

        # 获取预测结果
        predictions = outputs['predictions']
        outputs1 = predictions.permute([0, 2, 1]).reshape([-1, self.num_classes])
        _, pred_labels = torch.max(outputs1, 1)

        filename = batch.filename
        pts_normals = batch.x
        gt_labels = batch.y.to(self.device, dtype=torch.long)

        metrics_dict = self.metrics_mgr.get_metrics_dict(
            loss.item(), metrics_dict, is_train=False
        )

        metrics = {
            **metrics_dict,
            'pts_normals': pts_normals,
            'gt_labels': gt_labels,
            'pred_labels': pred_labels,
            'mesh_file': filename,
            'predictions': outputs1
        }

        return metrics
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
        optimizer = torch.optim.Adam(self.model.parameters(), amsgrad=True, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        return [optimizer], [scheduler]


class PTSegNetTrainer(BaseTrainer):
    """PTInsSeg训练器，继承自BaseTrainer"""
    def __init__(self, config_path="../../configs", config_name="Module_PTSegNet"):
        super().__init__(config_path, config_name)

    # 获得模型实例
    def build_module(self, model_obj: DictConfig, ckpt = None):
        """获取PTSegNet模型实例"""
        return PTSegNetModule(model_obj, num_classes=15).to(device)

    # 准备数据
    def prepare_data(self, cfg_obj: DictConfig):
        if cfg_obj.dataset.name == "T3DSPlusSegDataset":
            self.data_module, self.train_loader, self.test_loader \
                = T3DSPlusSegDataset.get_dataloaders_from_cfg_obj(cfg_obj)
        elif cfg_obj.dataset.name == "TADPMSegDataset":
            self.data_module, self.train_loader, self.test_loader \
                = TADPMSegDataset.get_dataloaders_from_cfg_obj(cfg_obj)


############################################################################
## Training Testing Prediction
############################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def do_train():
    trainer = PTSegNetTrainer(config_path="../../configs", config_name="Module_PTSegNet")

    # trainer.prepare_data(trainer.cfg_obj)
    # trainer_loader,_ = trainer.get_data_loaders()
    # data0 = next(iter(trainer_loader))
    # print(data0)

    trainer.fit()


def test_get_model():
    cfg_obj = instantiate_from_yaml_config(config_path="../../configs/nets", config_name="PTSegNet")

    model = PointTransformerNet(cfg_obj.model, num_classes=15, num_feat=6)
    model.cuda()
    xyz = torch.rand(16000, 6).cuda()
    data = Data(x=xyz, pos=xyz[:, :3])
    data_list = [data]
    # 使用 DataLoader 批量处理
    loader = DataLoader(data_list, batch_size=1, shuffle=False)

    for batch in loader:
        B, C, N = batch.batch_size, batch.x.size(1), batch.x.size(0) // batch.batch_size

        x_bcn = batch.x.view(B, N, C).permute(0, 2, 1)  # 转换为 (B, C, N)
        pos_bcn = batch.pos.view(B, N, 3).permute(0, 2, 1)  # 转换为 (B, 3, N)

        inputs = [x_bcn]
        outputs = model(inputs)

        # output is B, C, N order
        print(outputs['sem_1'].shape)

    x_bcn = data.x.unsqueeze(0).permute(0, 2, 1)
    flops, params = profile(model, inputs=([x_bcn],))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f G, params: %.2f M' % (flops / 1.0e9, params / 1.0e6))

def main():
    # 打印配置信息
    # print_yaml_config(config_path="../../configs", config_name="Module_PTSegNet")
    cfg_obj = instantiate_from_yaml_config(config_path="../../configs", config_name="Module_PTSegNet")
    print(cfg_obj)

    # 测试模型
    # test_get_model()
    do_train()


if __name__ == '__main__':
    main()

