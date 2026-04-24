import numpy as np
import torch
from torchmetrics import Dice, Accuracy, Precision, Recall, JaccardIndex
from evaluation.seg_evaluation import *


class SegMetricsManager:
    def __init__(self, task_type: str, device: torch.device = None, num_classes: int = 15, num_epochs: int = 30):

        # super().__init__(task_type, device)
        self.task_type = task_type
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_epochs = num_epochs
        self.num_classes = num_classes
        self.ignore_index = 0

        self._init_metrics()

    def _init_metrics(self):
        self.train_metrics = {
            'losses': [],
            'mdsc': [], 'msen': [], 'mppv': [], 'macc': [], 'miou': []
        }
        self.val_metrics = {
            'test_losses': [],
            'mdsc': [], 'msen': [], 'mppv': [], 'macc': [], 'miou': []
        }

        # 精度通常要保留背景类
        self.accuracy = Accuracy(task='multiclass', num_classes=self.num_classes).to(self.device)

        # 排除了背景类
        self.precision = Precision(task='multiclass', num_classes=self.num_classes, average='macro', ignore_index=0).to(self.device)
        self.recall = Recall(task='multiclass', num_classes=self.num_classes, average='macro', ignore_index=0).to(self.device)
        self.dice = Dice(num_classes=self.num_classes, average='macro', ignore_index=0).to(self.device)
        self.iou = JaccardIndex(task='multiclass', num_classes=self.num_classes, average='macro', ignore_index=0).to(self.device)

    # 在_compute_groups_metrics中使用
    def compute_step_metrics(self, predictions, ground_truth, **kwargs):
        """
        Args:
            predictions: dict, 包含模型输出的预测结果，例如 {'seg_result': BCN}
            ground_truth: dict, 包含真实标签的字典，例如 {'onehot_gt_labels': BNC}
            **kwargs: 其他可能需要的参数，例如 pos: BN, 3

        Returns:
            tuple: (dsc, sen, ppv, acc, iou)
        """""
        # [B, N, 15 ]
        pred = predictions['seg_result']
        gt_labels = ground_truth['onehot_gt_labels']

        gt_labels = torch.argmax(gt_labels, dim=-1)
        pred_classes = torch.argmax(pred, dim=-1)

        # 确保数据类型
        pred_classes = pred_classes.long().to(self.device)
        gt_labels = gt_labels.long().to(self.device)

        # 计算指标
        # dsc = self.dice(pred_classes, gt_labels)
        # sen = self.recall(pred_classes, gt_labels)
        # ppv = self.precision(pred_classes, gt_labels)
        # acc = self.accuracy(pred_classes, gt_labels)
        # iou = self.iou(pred_classes, gt_labels)

        outs = cal_multi_cls_metric(gt_labels.cpu().numpy(), pred_classes.cpu().numpy(), pred_classes.cpu().numpy())
        # ACC, IOU, F1, PRECISION, RECALL, SEM_ACC, IOU_arr = outs
        acc, iou, dsc, ppv, sen, _, _ = outs
        # # ========== 打印两组指标对比 ==========
        # print("\n" + "="*80)
        # print("📊 两组指标对比")
        # print("="*80)
        # print(f"{'指标':<20} | {'第一组 (cal_multi_cls_metric)':<25} | {'第二组 (torchmetrics)':<25} | {'差异':<10}")
        # print("-"*80)
        # print(f"{'ACC vs acc':<20} | {ACC:<25.6f} | {acc:<25.6f} | {abs(ACC-acc):<10.6e}")
        # print(f"{'IOU vs iou':<20} | {IOU:<25.6f} | {iou:<25.6f} | {abs(IOU-iou):<10.6e}")
        # print(f"{'F1 vs dsc':<20} | {F1:<25.6f} | {dsc:<25.6f} | {abs(F1-dsc):<10.6e}")
        # print(f"{'PRECISION vs ppv':<20} | {PRECISION:<25.6f} | {ppv:<25.6f} | {abs(PRECISION-ppv):<10.6e}")
        # print(f"{'RECALL vs sen':<20} | {RECALL:<25.6f} | {sen:<25.6f} | {abs(RECALL-sen):<10.6e}")
        # print("="*80)

        return dsc, sen, ppv, acc, iou


    def update(self, loss, metrics_dict, is_train=True):
        """分组传参方式"""

        dsc, sen, ppv, acc, iou = (metrics_dict['dsc'], metrics_dict['sen'], metrics_dict['ppv'],
                                   metrics_dict['acc'], metrics_dict['iou'])

        metrics = self.train_metrics if is_train else self.val_metrics
        metrics['losses' if is_train else 'test_losses'].append(loss)

        metrics['mdsc'].append(dsc)
        metrics['msen'].append(sen)
        metrics['mppv'].append(ppv)
        metrics['macc'].append(acc)
        metrics['miou'].append(iou)


    def compute(self, is_train=True):
        metrics = self.train_metrics if is_train else self.val_metrics
        for metric in ['mdsc', 'msen', 'mppv', 'macc', 'miou']:
            values = metrics[metric]
            # 存储为列表，便于后续的打印、重置等操作
            metrics[metric] = [sum(values) / len(values) if values else 0.0]

        return metrics

    def reset(self, is_train=True):
        metrics = self.train_metrics if is_train else self.val_metrics
        for metric in ['mdsc', 'msen', 'mppv', 'macc', 'miou']:
            metrics[metric].clear()
        metrics['losses' if is_train else 'test_losses'].clear()

    def get_metrics_dict(self, loss, metrics_dict, is_train=True):
        """构造并返回标准格式的指标字典 - 分组传参版本"""
        prefix = "train/" if is_train else "test/"

        metrics = {
            f"{prefix}loss": loss,
            f"{prefix}dsc": metrics_dict['dsc'],
            f"{prefix}sen": metrics_dict['sen'],
            f"{prefix}ppv": metrics_dict['ppv'],
            f"{prefix}acc": metrics_dict['acc'],
            f"{prefix}iou": metrics_dict['iou'],

        }
        return metrics

    def _prepare_epoch_metrics(self, current_epoch):
        """准备epoch指标数据，供打印和保存CSV使用"""
        if len(self.train_metrics['losses']) > 0 and len(self.val_metrics['test_losses']) > 0:
            pd_dict = {
                'epoch': current_epoch + 1,
                'train_loss': self.train_metrics['losses'][-1],
                'train_DSC': self.train_metrics['mdsc'][-1],
                'train_SEN': self.train_metrics['msen'][-1],
                'train_PPV': self.train_metrics['mppv'][-1],
                'train_ACC': self.train_metrics['macc'][-1],
                'train_IOU': self.train_metrics['miou'][-1],
                'test_loss': self.val_metrics['test_losses'][-1],
                'test_DSC': self.val_metrics['mdsc'][-1],
                'test_SEN': self.val_metrics['msen'][-1],
                'test_PPV': self.val_metrics['mppv'][-1],
                'test_ACC': self.val_metrics['macc'][-1],
                'test_IOU': self.val_metrics['miou'][-1],
            }
            return pd_dict
        return None

    def _print_epoch_info(self, current_epoch):
        # 从train_metrics中获取最新指标
        pd_dict = self._prepare_epoch_metrics(current_epoch)
        if pd_dict:
            # 直接使用pd_dict中的数据进行打印
            print(f"\nEpoch: {pd_dict['epoch']}/{self.num_epochs}")
            print("=" * 80)
            print(f"{'Metric':<20} | {'Train':<15} | {'Test':<15}")
            print("-" * 80)
            print(f"{'Loss':<20} | {pd_dict['train_loss']:<15.4f} | {pd_dict['test_loss']:<15.4f}")
            print(f"{'DSC':<20} | {pd_dict['train_DSC']:<15.4f} | {pd_dict['test_DSC']:<15.4f}")
            print(f"{'SEN':<20} | {pd_dict['train_SEN']:<15.4f} | {pd_dict['test_SEN']:<15.4f}")
            print(f"{'PPV':<20} | {pd_dict['train_PPV']:<15.4f} | {pd_dict['test_PPV']:<15.4f}")
            print(f"{'ACC':<20} | {pd_dict['train_ACC']:<15.4f} | {pd_dict['test_ACC']:<15.4f}")
            print(f"{'IOU':<20} | {pd_dict['train_IOU']:<15.4f} | {pd_dict['test_IOU']:<15.4f}")
            print("=" * 80)


if __name__ == "__main__":
    """
    SegMetricsManager 调用示例
    演示如何在训练流程中使用指标管理器
    """

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    print("="*80)
    print("🧪 SegMetricsManager 调用示例")
    print("="*80)

    # ========== 1. 初始化配置 ==========
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 15
    num_epochs = 10
    batch_size = 4

    print(f"\n📋 配置参数:")
    print(f"   - Device: {device}")
    print(f"   - Num Classes: {num_classes}")
    print(f"   - Num Epochs: {num_epochs}")
    print(f"   - Batch Size: {batch_size}")

    # ========== 2. 创建指标管理器 ==========
    metrics_manager = SegMetricsManager(
        task_type="segmentation",
        device=device,
        num_classes=num_classes,
        num_epochs=num_epochs
    )

    print(f"\n✅ 指标管理器已初始化")
    print(f"   - Train Metrics Keys: {list(metrics_manager.train_metrics.keys())}")
    print(f"   - Val Metrics Keys: {list(metrics_manager.val_metrics.keys())}")

    # ========== 3. 模拟数据生成 ==========
    def generate_batch_data(batch_size, num_points, num_classes, device):
        """生成一批模拟的点云分割数据"""
        # seg_result: [B, N, C] 模型输出
        seg_result = torch.randn(batch_size, num_points, num_classes, device=device)

        # onehot_gt_labels: [B, N, C] 真实标签的 one-hot 编码
        gt_labels = torch.randint(0, num_classes, (batch_size, num_points), device=device)
        onehot_gt_labels = torch.zeros(batch_size, num_points, num_classes, device=device)
        onehot_gt_labels.scatter_(2, gt_labels.unsqueeze(-1), 1)

        return {
            'seg_result': seg_result,
            'onehot_gt_labels': onehot_gt_labels
        }

    num_points = 1024
    print(f"\n📊 模拟数据:")
    print(f"   - Points per sample: {num_points}")

    # ========== 4. 模拟训练循环 ==========
    print("\n" + "="*80)
    print("🚀 开始模拟训练...")
    print("="*80)

    for epoch in range(num_epochs):
        print(f"\n{'='*80}")
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"{'='*80}")

        # --- 训练阶段 ---
        # metrics_manager.train()  # 切换到训练模式

        train_loss = 0.0
        num_train_batches = 5

        # --- 验证阶段 ---
        # metrics_manager.eval()  # 切换到评估模式

        val_loss = 0.0
        num_val_batches = 2

        with torch.no_grad():
            for batch_idx in range(num_val_batches):
                # 生成模拟数据
                data = generate_batch_data(batch_size, num_points, num_classes, device)

                # 模拟损失
                loss = torch.tensor(0.4 + torch.rand(1).item() * 0.2, device=device)

                # 计算指标
                dsc, sen, ppv, acc, iou = metrics_manager.compute_step_metrics(
                    predictions={'seg_result': data['seg_result']},
                    ground_truth={'onehot_gt_labels': data['onehot_gt_labels']}
                )

                # 更新指标
                metrics_dict = {'dsc': dsc, 'sen': sen, 'ppv': ppv, 'acc': acc, 'iou': iou}
                metrics_manager.update(loss.item(), metrics_dict, is_train=False)


                val_loss += loss.item()

        avg_val_loss = val_loss / num_val_batches

        # --- 计算并打印 epoch 指标 ---
        val_metrics = metrics_manager.compute(is_train=False)

        # 使用内置的打印方法
        metrics_manager._print_epoch_info(epoch)

        # --- 重置批次指标（为下一个 epoch 做准备）---
        # 注意：这里不清除 epoch 级别的统计，只清除 batch 级别的
        # 如果需要清除，可以调用 metrics_manager.reset()
        metrics_manager.reset()

