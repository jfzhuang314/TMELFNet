"""
质心预测指标管理器
用于记录和管理 centroid loss、distance loss、chamfer loss
"""
import torch


class CentroidMetricsManager:
    def __init__(self, task_type: str = "centroid_regression",
                 device: torch.device = None,
                 num_epochs: int = 30):
        """
        初始化质心预测指标管理器

        Args:
            task_type: 任务类型标识
            device: 计算设备
            num_epochs: 总训练轮数
        """
        self.task_type = task_type
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_epochs = num_epochs

        self._init_metrics()

    def _init_metrics(self):
        """初始化训练和验证的指标存储"""
        self.train_metrics = {
            'losses': [],           # 总损失（可选：dist_loss + cent_loss + chamf_loss）
            'dist_losses': [],      # 距离损失
            'cent_losses': [],      # 质心损失
            'chamf_losses': [],     # Chamfer 距离损失

            # 聚类检测指标
            'TP': [],               # True Positives
            'FP': [],               # False Positives
            'FN': [],               # False Negatives
            'precision': [],        # 精确率
            'recall': [],           # 召回率
            'f1': [],               # F1-Score
            'cd': [],               # Chamfer Distance
            'p2g_dist': [],         # Prediction to Ground Truth Distance
            'g2p_dist': []          # Ground Truth to Prediction Distance
        }
        self.val_metrics = {
            'test_losses': [],
            'dist_losses': [],
            'cent_losses': [],
            'chamf_losses': [],

            # 聚类检测指标
            'TP': [],
            'FP': [],
            'FN': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'cd': [],
            'p2g_dist': [],
            'g2p_dist': []
        }


    def update(self, loss, losses_dict, is_train=True, clu_metrics=None):
        """
        更新指标记录

        Args:
            loss: 总损失值
            losses_dict: 包含三个损失的字典
                - dist_loss: 距离损失
                - cent_loss: 质心损失
                - chamf_loss: Chamfer 损失
            is_train: 是否为训练阶段
            clu_metrics: 聚类检测指标字典（可选）
                - TP, FP, FN, precision, recall, f1, cd, p2g_dist, g2p_dist
        """
        metrics = self.train_metrics if is_train else self.val_metrics

        # 记录各个损失
        metrics['dist_losses'].append(losses_dict['dist_loss'])
        metrics['cent_losses'].append(losses_dict['cent_loss'])
        metrics['chamf_losses'].append(losses_dict['chamf_loss'])

        # 记录总损失
        total_loss = (losses_dict.get('dist_loss', 0) +
                     losses_dict.get('cent_loss', 0) +
                     losses_dict.get('chamf_loss', 0))
        metrics['losses' if is_train else 'test_losses'].append(total_loss)

        # 记录聚类检测指标（如果提供）
        if clu_metrics is not None:
            metrics['TP'].append(clu_metrics.get('TP', 0))
            metrics['FP'].append(clu_metrics.get('FP', 0))
            metrics['FN'].append(clu_metrics.get('FN', 0))
            metrics['precision'].append(clu_metrics.get('precision', 0.0))
            metrics['recall'].append(clu_metrics.get('recall', 0.0))
            metrics['f1'].append(clu_metrics.get('f1', 0.0))
            metrics['cd'].append(clu_metrics.get('cd', 0.0))
            metrics['p2g_dist'].append(clu_metrics.get('p2g_dist', 0.0))
            metrics['g2p_dist'].append(clu_metrics.get('g2p_dist', 0.0))
# ... existing code ...
    def compute(self, is_train=True):
        """
        计算平均损失和指标

        Args:
            is_train: 是否为训练阶段

        Returns:
            dict: 包含所有损失和指标的平均值
        """
        metrics = self.train_metrics if is_train else self.val_metrics

        # 计算损失的平均值
        for metric_key in ['dist_losses', 'cent_losses', 'chamf_losses']:
            values = metrics[metric_key]
            metrics[metric_key] = [sum(values) / len(values) if values else 0.0]

        # 计算总损失的平均值
        total_values = metrics['losses' if is_train else 'test_losses']
        metrics['losses' if is_train else 'test_losses'] = [
            sum(total_values) / len(total_values) if total_values else 0.0
        ]

        # 计算聚类指标的平均值
        cluster_metric_keys = ['TP', 'FP', 'FN', 'precision', 'recall', 'f1',
                               'cd', 'p2g_dist', 'g2p_dist']
        for metric_key in cluster_metric_keys:
            values = metrics[metric_key]
            if metric_key in ['TP', 'FP', 'FN']:
                # TP/FP/FN 保持为整数总和
                metrics[metric_key] = [sum(values)]
            else:
                # 其他指标计算平均值
                metrics[metric_key] = [sum(values) / len(values) if values else 0.0]

        return metrics

    def reset(self, is_train=True):
        """
        重置指标记录

        Args:
            is_train: 是否重置训练阶段的指标
        """
        metrics = self.train_metrics if is_train else self.val_metrics
        for key in metrics.keys():
            metrics[key].clear()

    def get_metrics_dict(self, loss, losses_dict, is_train=True, clu_metrics=None):
        """
        构造并返回标准格式的指标字典（用于日志记录）

        Args:
            loss: 总损失值
            losses_dict: 原始损失字典
            is_train: 是否为训练阶段
            clu_metrics: 聚类检测指标字典（可选）

        Returns:
            dict: 格式化的指标字典
        """
        prefix = "train/" if is_train else "test/"

        metrics = {
            f"{prefix}total_loss": (losses_dict.get('dist_loss', 0) +
                                   losses_dict.get('cent_loss', 0) +
                                   losses_dict.get('chamf_loss', 0)),
            f"{prefix}dist_loss": losses_dict.get('dist_loss', 0),
            f"{prefix}cent_loss": losses_dict.get('cent_loss', 0),
            f"{prefix}chamf_loss": losses_dict.get('chamf_loss', 0),
        }

        # 添加聚类指标（如果提供）
        if clu_metrics is not None:
            metrics.update({
                f"{prefix}TP": float(clu_metrics.get('TP', 0)),
                f"{prefix}FP": float(clu_metrics.get('FP', 0)),
                f"{prefix}FN": float(clu_metrics.get('FN', 0)),
                f"{prefix}precision": clu_metrics.get('precision', 0.0),
                f"{prefix}recall": clu_metrics.get('recall', 0.0),
                f"{prefix}f1": clu_metrics.get('f1', 0.0),
                f"{prefix}cd": clu_metrics.get('cd', 0.0),
                f"{prefix}p2g_dist": clu_metrics.get('p2g_dist', 0.0),
                f"{prefix}g2p_dist": clu_metrics.get('g2p_dist', 0.0),
            })

        return metrics
# ... existing code ...
    def _prepare_epoch_metrics(self, current_epoch):
        """
        准备 epoch 级别的指标数据，供打印和保存 CSV 使用

        Args:
            current_epoch: 当前 epoch 编号（从 0 开始）

        Returns:
            dict: 包含所有指标的字典，或 None（如果数据不足）
        """
        if len(self.train_metrics['losses']) > 0 and len(self.val_metrics['test_losses']) > 0:
            pd_dict = {
                'epoch': current_epoch + 1,

                # 训练集指标
                'train_total_loss': self.train_metrics['losses'][-1],
                'train_dist_loss': self.train_metrics['dist_losses'][-1],
                'train_cent_loss': self.train_metrics['cent_losses'][-1],
                'train_chamf_loss': self.train_metrics['chamf_losses'][-1],

                # 训练集聚类指标
                'train_TP': self.train_metrics['TP'][-1] if self.train_metrics['TP'] else 0,
                'train_FP': self.train_metrics['FP'][-1] if self.train_metrics['FP'] else 0,
                'train_FN': self.train_metrics['FN'][-1] if self.train_metrics['FN'] else 0,
                'train_precision': self.train_metrics['precision'][-1] if self.train_metrics['precision'] else 0.0,
                'train_recall': self.train_metrics['recall'][-1] if self.train_metrics['recall'] else 0.0,
                'train_f1': self.train_metrics['f1'][-1] if self.train_metrics['f1'] else 0.0,
                'train_cd': self.train_metrics['cd'][-1] if self.train_metrics['cd'] else 0.0,
                'train_p2g_dist': self.train_metrics['p2g_dist'][-1] if self.train_metrics['p2g_dist'] else 0.0,
                'train_g2p_dist': self.train_metrics['g2p_dist'][-1] if self.train_metrics['g2p_dist'] else 0.0,

                # 验证集指标
                'test_total_loss': self.val_metrics['test_losses'][-1],
                'test_dist_loss': self.val_metrics['dist_losses'][-1],
                'test_cent_loss': self.val_metrics['cent_losses'][-1],
                'test_chamf_loss': self.val_metrics['chamf_losses'][-1],

                # 验证集聚类指标
                'test_TP': self.val_metrics['TP'][-1] if self.val_metrics['TP'] else 0,
                'test_FP': self.val_metrics['FP'][-1] if self.val_metrics['FP'] else 0,
                'test_FN': self.val_metrics['FN'][-1] if self.val_metrics['FN'] else 0,
                'test_precision': self.val_metrics['precision'][-1] if self.val_metrics['precision'] else 0.0,
                'test_recall': self.val_metrics['recall'][-1] if self.val_metrics['recall'] else 0.0,
                'test_f1': self.val_metrics['f1'][-1] if self.val_metrics['f1'] else 0.0,
                'test_cd': self.val_metrics['cd'][-1] if self.val_metrics['cd'] else 0.0,
                'test_p2g_dist': self.val_metrics['p2g_dist'][-1] if self.val_metrics['p2g_dist'] else 0.0,
                'test_g2p_dist': self.val_metrics['g2p_dist'][-1] if self.val_metrics['g2p_dist'] else 0.0,
            }
            return pd_dict
        return None

    def _print_epoch_info(self, current_epoch):
        """
        打印 epoch 级别的指标信息

        Args:
            current_epoch: 当前 epoch 编号（从 0 开始）
        """
        pd_dict = self._prepare_epoch_metrics(current_epoch)
        if pd_dict:
            print(f"\nEpoch: {pd_dict['epoch']}/{self.num_epochs}")
            print("=" * 120)
            print(f"{'Metric':<20} | {'Train':<15} | {'Test':<15}")
            print("-" * 120)

            # 损失指标
            print(f"{'Total Loss':<20} | {pd_dict['train_total_loss']:<15.4f} | {pd_dict['test_total_loss']:<15.4f}")
            print(f"{'Dist Loss':<20} | {pd_dict['train_dist_loss']:<15.4f} | {pd_dict['test_dist_loss']:<15.4f}")
            print(f"{'Cent Loss':<20} | {pd_dict['train_cent_loss']:<15.4f} | {pd_dict['test_cent_loss']:<15.4f}")
            print(f"{'Chamf Loss':<20} | {pd_dict['train_chamf_loss']:<15.4f} | {pd_dict['test_chamf_loss']:<15.4f}")
            print("-" * 120)

            # 聚类检测指标
            print(f"{'TP':<20} | {pd_dict['train_TP']:<15.0f} | {pd_dict['test_TP']:<15.0f}")
            print(f"{'FP':<20} | {pd_dict['train_FP']:<15.0f} | {pd_dict['test_FP']:<15.0f}")
            print(f"{'FN':<20} | {pd_dict['train_FN']:<15.0f} | {pd_dict['test_FN']:<15.0f}")
            print(f"{'Precision':<20} | {pd_dict['train_precision']:<15.4f} | {pd_dict['test_precision']:<15.4f}")
            print(f"{'Recall':<20} | {pd_dict['train_recall']:<15.4f} | {pd_dict['test_recall']:<15.4f}")
            print(f"{'F1-Score':<20} | {pd_dict['train_f1']:<15.4f} | {pd_dict['test_f1']:<15.4f}")
            print(f"{'CD':<20} | {pd_dict['train_cd']:<15.6f} | {pd_dict['test_cd']:<15.6f}")
            print(f"{'P2G Dist':<20} | {pd_dict['train_p2g_dist']:<15.6f} | {pd_dict['test_p2g_dist']:<15.6f}")
            print(f"{'G2P Dist':<20} | {pd_dict['train_g2p_dist']:<15.6f} | {pd_dict['test_g2p_dist']:<15.6f}")
            print("=" * 120)


if __name__ == "__main__":
    """
    CentroidMetricsManager 调用示例
    演示如何在训练流程中使用指标管理器
    """

    import torch

    print("="*80)
    print("🧪 CentroidMetricsManager 调用示例")
    print("="*80)

    # ========== 1. 初始化配置 ==========
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 10

    print(f"\n📋 配置参数:")
    print(f"   - Device: {device}")
    print(f"   - Num Epochs: {num_epochs}")

    # ========== 2. 创建指标管理器 ==========
    metrics_manager = CentroidMetricsManager(
        task_type="centroid_regression",
        device=device,
        num_epochs=num_epochs
    )

    print(f"\n✅ 指标管理器已初始化")
    print(f"   - Train Metrics Keys: {list(metrics_manager.train_metrics.keys())}")
    print(f"   - Val Metrics Keys: {list(metrics_manager.val_metrics.keys())}")

    # ========== 3. 模拟训练循环 ==========
    print("\n" + "="*80)
    print("🚀 开始模拟训练...")
    print("="*80)

    for epoch in range(num_epochs):
        print(f"\n{'='*80}")
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"{'='*80}")

        # --- 训练阶段 ---
        num_train_batches = 5
        for batch_idx in range(num_train_batches):
            # 模拟损失值
            losses_dict = {
                'dist_loss': 0.3 + torch.rand(1).item() * 0.2,
                'cent_loss': 0.2 + torch.rand(1).item() * 0.15,
                'chamf_loss': 0.1 + torch.rand(1).item() * 0.1,
            }

            # 更新指标
            metrics_manager.update(losses_dict, is_train=True)

        # 计算训练指标
        train_metrics = metrics_manager.compute(is_train=True)

        # --- 验证阶段 ---
        num_val_batches = 2
        with torch.no_grad():
            for batch_idx in range(num_val_batches):
                # 模拟验证损失
                losses_dict = {
                    'dist_loss': 0.35 + torch.rand(1).item() * 0.2,
                    'cent_loss': 0.25 + torch.rand(1).item() * 0.15,
                    'chamf_loss': 0.12 + torch.rand(1).item() * 0.1,
                }

                # 更新指标
                metrics_manager.update(losses_dict, is_train=False)

        # 计算验证指标
        val_metrics = metrics_manager.compute(is_train=False)

        # --- 打印 epoch 指标 ---
        metrics_manager._print_epoch_info(epoch)

        # --- 重置批次指标 ---
        metrics_manager.reset()

    print("\n✅ 训练完成！")
