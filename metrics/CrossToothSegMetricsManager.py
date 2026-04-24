from torchmetrics import JaccardIndex
import numpy as np

from metrics.SegMetricsManager import *
from utils.points_utils import *
import torch_geometric.transforms as T



class DynamicRadiusNNEdgeMaskWithPos(T.BaseTransform):
    def __init__(self,
                 rmin: float = 1.8,
                 rmax: float = 2.2,
                 k: int = 2,
                 disp_avg_dist: bool = False,
                 max_num_neighbors: int = 16):
        """
        初始化DynamicRadiusNNEdgeMask
        :param rmin: 最小半径
        :param rmax: 最大半径
        :param max_num_neighbors: 最大邻居数，用于限制邻域大小
        """
        self.rmin = rmin
        self.rmax = rmax
        self.k = k
        self.max_num_neighbors = max_num_neighbors
        self.disp_avg_dist = disp_avg_dist

    def generate_edge_masks(self, pos, labels):
        knn_avg_dist = compute_knn_average_distance(pos, self.k)
        if self.disp_avg_dist:
            print(f"average distance: {knn_avg_dist}")
        # 计算极角
        _, abs_ang = compute_polar_angles(pos)
        radius = (self.rmin + abs_ang * (self.rmax - self.rmin)) * knn_avg_dist
        tae_gt_bin_labels = compute_tae_gt_bin_labels_ang(pos, labels, r=radius)
        tte_gt_bin_labels = compute_tte_gt_bin_labels(pos, labels, r=radius)
        return tae_gt_bin_labels, tte_gt_bin_labels


class CrossToothSegMetricsManager(SegMetricsManager):
    def __init__(self, task_type: str,
                 device: torch.device = None,
                 num_classes: int = 15,
                 num_epochs: int = 30,
                 compute_edge_binary_iou: bool = False
                 ):

        # 定义牙齿对称合并对
        self.merge_pairs = [(1, 8), (2, 9), (3, 10), (4, 11), (5, 12), (6, 13), (7, 14)]
        self.compute_edge_binary_iou = compute_edge_binary_iou

        super().__init__(task_type, device, num_classes, num_epochs)

    def _init_metrics(self):
        """初始化基础指标和新增的跨牙齿分割特有指标"""
        # 调用父类初始化基础指标
        super()._init_metrics()

        # 添加跨牙齿分割特有的指标
        self.train_metrics.update({
            # 'miou_cross': [],      # 跨牙齿mIOU
            'boundary_iou': [],  # 边界IOU
            'background_iou': [],  # 背景IOU
            'edge_binary_iou': [],  # 边界IOU
            'tte_binary_iou': [],  # 跨牙齿IOU
        })

        self.val_metrics.update({
            # 'miou_cross': [],      # 跨牙齿mIOU
            'boundary_iou': [],  # 边界IOU
            'background_iou': [],  # 背景IOU
            'edge_binary_iou': [],  # 边界IOU
            'tte_binary_iou': [],  # 跨牙齿IOU
        })

        # 为每对对称牙齿添加单独的IOU指标
        for a, b in self.merge_pairs:
            metric_name = f'mIOU_{a}_{b}'
            self.train_metrics[metric_name] = []
            self.val_metrics[metric_name] = []

        # 初始化新增的torchmetrics指标
        # self.miou_cross = JaccardIndex(task='multiclass', num_classes=self.num_classes, average='macro').to(self.device)
        # self.boundary_iou = JaccardIndex(task='binary').to(self.device)
        # self.background_iou = JaccardIndex(task='binary').to(self.device)

    def calc_cross_tooth_metrics(self, pred, labels, gt_edge_mask, num_classes=15):
        """
        计算跨牙齿分割特有的指标
        Args:
            pred: 预测结果 [B, num_classes, N]
            labels: 真实标签 [B, N] 或 [B, num_classes, N]
            class_weights: 类别权重
        Returns:
            miou_cross, boundary_iou, background_iou, t_symmetry_ious_dict
        """
        # 处理标签格式
        if labels.dim() == 3 and labels.shape[1] == num_classes:
            labels = torch.argmax(labels, dim=1)  # 转换为类别索引

        pred_classes = torch.argmax(pred, dim=-1)  # [B, N]

        # 确保数据类型和设备
        pred_classes = pred_classes.reshape(-1, ).long().to(self.device)  # B,N
        labels = labels.long().to(self.device)

        # 1. 计算跨牙齿mIOU (整体mIOU)
        # miou_cross = self.miou_cross(pred_classes, labels)

        # 2. 计算边界IOU - 假设类别0是背景，其他是前景
        boundary_iou = calculate_boundary_miou(gt_labels=labels, pred_labels=pred_classes,
                                               gt_boundary=gt_edge_mask, n_class=num_classes)

        # 3. 计算背景IOU - 类别0
        background_pred = (pred_classes == 0).squeeze().float()
        background_labels = (labels == 0).squeeze().float()
        # background_iou = self.background_iou(background_pred, background_labels)
        _, _, _, _, background_iou, _ = cal_binary_metric_label(gt_labels=background_labels,
                                                          pred_classes=background_pred,
                                                          target_class=1)
        # 4. 计算T对称合并类别IOU (展开为Ta_b形式)
        merge_pairs_ious_dict = self._compute_merge_pairs_ious(pred_classes, labels)

        return boundary_iou, background_iou, merge_pairs_ious_dict

    def _compute_merge_pairs_ious(self, pred_classes, labels):
        """
        计算T对称合并类别IOUs，返回字典形式 {metric_name: iou_value}
        """
        symmetry_ious = {}

        for a, b in self.merge_pairs:
            # 创建合并的标签和预测
            gt_merge = ((labels == a) | (labels == b)).reshape(-1).long()
            pred_merge = ((pred_classes == a) | (pred_classes == b)).reshape(-1).long()

            # 计算二分类IOU
            # merge_iou = self._binary_iou(pred_merge, gt_merge)
            _, _, _, _, merge_iou,_ = cal_binary_metric_label(gt_labels=gt_merge,
                                                           pred_classes=pred_merge,
                                                           target_class=1)
            metric_name = f'mIOU_{a}_{b}'
            symmetry_ious[metric_name] = merge_iou

        return symmetry_ious

    def _binary_iou(self, pred, target):
        """计算二分类IOU"""
        intersection = torch.sum(pred * target)
        union = ((pred + target) > 0).sum().float()
        eps = 1e-6

        iou = 0.0 if union == 0 else intersection / (union + eps)

        return iou

    def compute_step_metrics(self, predictions, ground_truth, **kwargs):
        """
        重写父类方法，同时计算基础指标和跨牙齿特有指标
        """
        pos = kwargs.get('pos')  # BN,3
        # assert pos is not None

        # 计算基础指标
        dsc, sen, ppv, acc, iou = super().compute_step_metrics(predictions, ground_truth, **kwargs)

        pred, labels = predictions['seg_result'], ground_truth['gt_labels']

        # 1. 确保gt_edge_mask存在
        if 'gt_edge_mask' in ground_truth:
            gt_edge_mask = ground_truth['gt_edge_mask'].to(labels.device, dtype=torch.float32)
        else:
            gt_edge_mask,_ = DynamicRadiusNNEdgeMaskWithPos().generate_edge_masks(pos, labels).to(labels.device,
                                                                                                  dtype=torch.float32)

        if 'gt_tte_mask' in ground_truth:
            gt_tte_mask = ground_truth['gt_tte_mask'].to(labels.device, dtype=torch.float32)
        else:
            _, gt_tte_mask = DynamicRadiusNNEdgeMaskWithPos().generate_edge_masks(pos, labels).to(labels.device,
                                                                                                  dtype=torch.float32)

        # 2. 计算跨牙齿特有指标
        # boundary_iou, background_iou, merge_pairs_ious_dict = self.calc_cross_tooth_metrics(
        #     pred, labels, gt_edge_mask, self.num_classes)
        pred_classes = torch.argmax(pred, dim=-1)  # [B, N]

        # 确保数据类型和设备
        pred_classes = pred_classes.reshape(-1, ).long().to(self.device)  # B,N
        labels = labels.long().to(self.device)

        # 2. 计算边界IOU - 假设类别0是背景，其他是前景
        boundary_iou = calculate_boundary_miou(gt_labels=labels, pred_labels=pred_classes,
                                               gt_boundary=gt_edge_mask, n_class=self.num_classes)

        # 3. 计算背景IOU - 类别0
        background_labels = (labels == 0).squeeze().float()
        if 'tg_seg_result' in predictions:
            background_pred = 1-torch.argmax(predictions['tg_seg_result'], dim=1).squeeze().float()
            _, _, _, _, background_iou, _ = cal_binary_metric_label(gt_labels=background_labels,
                                                              pred_classes=background_pred,
                                                              target_class=1)
        else:
            background_pred = (pred_classes == 0).squeeze().float()
            _, _, _, _, background_iou, _ = cal_binary_metric_label(gt_labels=background_labels,
                                                                    pred_classes=background_pred,
                                                                    target_class=1)
        # 4. 计算T对称合并类别IOU (展开为Ta_b形式)
        merge_pairs_ious_dict = self._compute_merge_pairs_ious(pred_classes, labels)

        # 5.计算牙齿的边缘二分类IOU指标
        if self.compute_edge_binary_iou:
            if 'edge_seg_result' in predictions:
                pred_edge_mask = torch.argmax(predictions['edge_seg_result'], dim=1).reshape(-1, )  # 结果形状 (B, N)
            else:
                pred_class = torch.argmax(pred, dim=-1).reshape(-1, )
                pred_edge_mask,_ = DynamicRadiusNNEdgeMaskWithPos().generate_edge_masks(pos, pred_class).to(labels.device,dtype=torch.float32)

            # valid_mask = gt_edge_mask.reshape(-1, ) > 0
            # _, _, _, _, edge_binary_iou, _ = cal_binary_metric_label(gt_labels=gt_edge_mask.reshape(-1, ),
            #                                             pred_classes=pred_edge_mask.reshape(-1, ),
            #                                             target_class=1)
            edge_binary_iou = cal_edge_iou(gt_edge_mask.cpu().numpy(), pred_edge_mask.cpu().numpy(),target_class=1)

            if 'tte_seg_result' in predictions:
                pred_tte_mask = torch.argmax(predictions['tte_seg_result'], dim=1).reshape(-1, )  # 结果形状 (B, N)
            else:
                pred_class = torch.argmax(pred, dim=-1).reshape(-1, )
                _,pred_tte_mask = DynamicRadiusNNEdgeMaskWithPos().generate_edge_masks(pos, pred_class).to(labels.device,dtype=torch.float32)

            # valid_mask = pred_tte_mask.reshape(-1, ) > 0
            # _, _, _, _, tte_binary_iou, _ = cal_binary_metric_label(gt_labels=gt_tte_mask.reshape(-1, ),
            #                                             pred_classes=pred_tte_mask.reshape(-1, ),
            #                                             target_class=1)
            tte_binary_iou = cal_edge_iou(gt_tte_mask.cpu().numpy(), pred_tte_mask.cpu().numpy(),target_class=1)
        else:
            edge_binary_iou = 0.0
            tte_binary_iou = 0.0

        # 返回所有指标的字典
        metrics_dict = {
            'dsc': dsc,
            'sen': sen,
            'ppv': ppv,
            'acc': acc,
            'iou': iou,
            'boundary_iou': boundary_iou,
            'background_iou': background_iou,
            'edge_binary_iou': edge_binary_iou,
            'tte_binary_iou': tte_binary_iou,
            'merge_pairs_ious_dict': merge_pairs_ious_dict
        }

        return metrics_dict

    def update(self, loss, metrics_dict, is_train=True):
        """
        更新所有指标，包括新增的跨牙齿指标
        """

        # 调用父类方法更新基础指标
        super().update(loss, metrics_dict, is_train)

        boundary_iou = metrics_dict['boundary_iou']
        background_iou = metrics_dict['background_iou']
        edge_binary_iou = metrics_dict['edge_binary_iou']
        tte_binary_iou = metrics_dict['tte_binary_iou']
        merge_pairs_ious_dict = metrics_dict['merge_pairs_ious_dict']

        # 更新新增指标
        metrics = self.train_metrics if is_train else self.val_metrics

        metrics['edge_binary_iou'].append(edge_binary_iou) if edge_binary_iou is not None else None
        metrics['boundary_iou'].append(boundary_iou) if boundary_iou is not None else None
        metrics['tte_binary_iou'].append(tte_binary_iou) if tte_binary_iou is not None else None
        metrics['background_iou'].append(background_iou) if background_iou is not None else None

        if merge_pairs_ious_dict is not None:
            for metric_name, iou_value in merge_pairs_ious_dict.items():
                if metric_name in metrics:
                    metrics[metric_name].append(iou_value)

    def compute(self, is_train=True):
        """
        计算平均指标值，包括新增指标
        """
        # 调用父类方法计算基础指标
        metrics = super().compute(is_train)

        # 计算新增指标的平均值
        cross_metrics = self.train_metrics if is_train else self.val_metrics

        # 计算基础新增指标
        for metric in ['boundary_iou', 'background_iou', 'edge_binary_iou', 'tte_binary_iou']:
            values = cross_metrics[metric]
            if values:
                metrics[metric] = [sum(values) / len(values)]
            else:
                metrics[metric] = [0.0]

        # 计算对称牙齿IOU指标
        for a, b in self.merge_pairs:
            metric_name = f'mIOU_{a}_{b}'
            values = cross_metrics[metric_name]
            if values:
                metrics[metric_name] = [sum(values) / len(values)]
            else:
                metrics[metric_name] = [0.0]

        return metrics

    def reset(self, is_train=True):
        """
        重置所有指标，包括新增指标
        """
        # 调用父类方法重置基础指标
        super().reset(is_train)

        # 重置新增指标
        metrics = self.train_metrics if is_train else self.val_metrics
        base_metrics = ['boundary_iou', 'background_iou', 'edge_binary_iou', 'tte_binary_iou']
        for metric in base_metrics:
            metrics[metric].clear()

        # 重置对称牙齿IOU指标
        for a, b in self.merge_pairs:
            metric_name = f'mIOU_{a}_{b}'
            metrics[metric_name].clear()

    def get_metrics_dict(self, loss, metrics_dict, is_train=True):
        """
        更新所有指标，包括新增的跨牙齿指标
        """

        edge_binary_iou = metrics_dict['edge_binary_iou']
        tte_binary_iou = metrics_dict['tte_binary_iou']
        boundary_iou = metrics_dict['boundary_iou']
        background_iou = metrics_dict['background_iou']
        merge_pairs_ious_dict = metrics_dict['merge_pairs_ious_dict']

        # 获取基础指标字典
        base_metrics = super().get_metrics_dict(loss, metrics_dict, is_train)
        # 添加新增指标
        prefix = "train/" if is_train else "test/"

        if edge_binary_iou is not None:
            base_metrics[f"{prefix}edge_binary_iou"] = edge_binary_iou
        if tte_binary_iou is not None:
            base_metrics[f"{prefix}tte_binary_iou"] = tte_binary_iou
        if boundary_iou is not None:
            base_metrics[f"{prefix}boundary_iou"] = boundary_iou
        if background_iou is not None:
            base_metrics[f"{prefix}background_iou"] = background_iou
        if merge_pairs_ious_dict is not None:
            for metric_name, iou_value in merge_pairs_ious_dict.items():
                base_metrics[f"{prefix}{metric_name}"] = iou_value

        return base_metrics

    def _prepare_epoch_metrics(self, current_epoch):
        """
        准备包含所有指标的epoch数据
        """
        base_dict = super()._prepare_epoch_metrics(current_epoch)
        if not base_dict:
            return None

        # 添加新增指标
        if len(self.train_metrics['edge_binary_iou']) > 0 and len(self.val_metrics['edge_binary_iou']) > 0:
            base_dict.update({
                'train_edge_binary_iou': self.train_metrics['edge_binary_iou'][-1],
                'train_tte_binary_iou': self.train_metrics['tte_binary_iou'][-1],
                'train_boundary_iou': self.train_metrics['boundary_iou'][-1],
                'train_background_iou': self.train_metrics['background_iou'][-1],
                'test_edge_binary_iou': self.val_metrics['edge_binary_iou'][-1],
                'test_tte_binary_iou': self.val_metrics['tte_binary_iou'][-1],
                'test_boundary_iou': self.val_metrics['boundary_iou'][-1],
                'test_background_iou': self.val_metrics['background_iou'][-1],
            })

            # 添加对称牙齿IOU指标
            for a, b in self.merge_pairs:
                metric_name = f'mIOU_{a}_{b}'
                base_dict[f'train_{metric_name}'] = self.train_metrics[metric_name][-1]
                base_dict[f'test_{metric_name}'] = self.val_metrics[metric_name][-1]

        return base_dict

    def _print_epoch_info(self, current_epoch):
        """
        打印包含所有指标的epoch信息
        """
        pd_dict = self._prepare_epoch_metrics(current_epoch)
        if pd_dict:
            print(f"\nEpoch: {pd_dict['epoch']}/{self.num_epochs}")
            print("=" * 120)
            print(f"{'Metric':<25} | {'Train':<15} | {'Test':<15}")
            print("-" * 120)

            # 基础指标
            print(f"{'Loss':<25} | {pd_dict['train_loss']:<15.4f} | {pd_dict['test_loss']:<15.4f}")
            print(f"{'DSC':<25} | {pd_dict['train_DSC']:<15.4f} | {pd_dict['test_DSC']:<15.4f}")
            print(f"{'SEN':<25} | {pd_dict['train_SEN']:<15.4f} | {pd_dict['test_SEN']:<15.4f}")
            print(f"{'PPV':<25} | {pd_dict['train_PPV']:<15.4f} | {pd_dict['test_PPV']:<15.4f}")
            print(f"{'ACC':<25} | {pd_dict['train_ACC']:<15.4f} | {pd_dict['test_ACC']:<15.4f}")
            print(f"{'IOU':<25} | {pd_dict['train_IOU']:<15.4f} | {pd_dict['test_IOU']:<15.4f}")

            # 新增指标
            print("-" * 120)
            print(
                f"{'Boundary_IOU':<25} | {pd_dict['train_boundary_iou']:<15.4f} | {pd_dict['test_boundary_iou']:<15.4f}")
            print(
                f"{'Background_IOU':<25} | {pd_dict['train_background_iou']:<15.4f} | {pd_dict['test_background_iou']:<15.4f}")
            print(
                f"{'Edge_Binary_IOU':<25} | {pd_dict['train_edge_binary_iou']:<15.4f} | {pd_dict['test_edge_binary_iou']:<15.4f}")
            print(
                f"{'TTE_Binary_IOU':<25} | {pd_dict['train_tte_binary_iou']:<15.4f} | {pd_dict['test_tte_binary_iou']:<15.4f}")

            # 对称牙齿IOU指标
            print("-" * 120)
            print("T对称合并类别IOU:")
            for a, b in self.merge_pairs:
                metric_name = f'mIOU_{a}_{b}'
                train_key = f'train_{metric_name}'
                test_key = f'test_{metric_name}'
                if train_key in pd_dict and test_key in pd_dict:
                    print(
                        f"{'T' + str(a) + '_T' + str(b) + '_IOU':<25} | {pd_dict[train_key]:<15.4f} | {pd_dict[test_key]:<15.4f}")
            print("=" * 120)


def test_cross_tooth_seg_metrics_manager():
    """测试CrossToothSegMetricsManager的功能"""

    # 1. 初始化指标管理器
    print("1. 初始化CrossToothSegMetricsManager...")
    metrics_mgr = CrossToothSegMetricsManager(
        task_type="segmentation",
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        num_classes=15,
        num_epochs=30
    )
    print("✓ 初始化完成")

    # 2. 创建测试数据
    print("\n2. 创建测试数据...")
    batch_size = 1
    num_points = 10000

    # 模拟预测结果 [B, num_classes, N]
    pred_logits = torch.randn(batch_size, 15, num_points)
    predictions = {
        'seg_result': pred_logits.permute(0, 2, 1)  # BNC
    }

    # 模拟真实标签
    gt_labels = torch.randint(0, 15, (batch_size * num_points,))  # [BN]
    one_hot_labels = (torch.nn.functional.one_hot(gt_labels, num_classes=15).
                      reshape(batch_size, num_points, 15))  # [B,N, 15]

    ground_truth = {
        'gt_labels': gt_labels,  # BN,
        'onehot_gt_labels': one_hot_labels  # BNC
    }

    # 模拟边缘掩码
    gt_edge_mask = torch.randint(0, 2, (batch_size * num_points,)).float()
    ground_truth['gt_edge_mask'] = gt_edge_mask

    # 模拟点云坐标
    pos = torch.randn(batch_size * num_points, 3)  # [BN, 3]

    print("✓ 测试数据创建完成")
    print(f"  - 预测结果形状: {pred_logits.shape}")
    print(f"  - 真实标签形状: {gt_labels.shape}")
    print(f"  - 真实标签形状: {one_hot_labels.shape}")
    print(f"  - 边缘掩码形状: {gt_edge_mask.shape}")
    print(f"  - 点云坐标形状: {pos.shape}")

    # 3. 测试单步指标计算
    print("\n3. 测试单步指标计算...")
    try:
        metrics_dict = metrics_mgr.compute_step_metrics(
            predictions,
            ground_truth,
            pos=pos
        )

        print("✓ 单步指标计算成功")
        print("计算得到的指标:")
        for key, value in metrics_dict.items():
            if isinstance(value, dict):
                print(f"  {key}: {list(value.keys())}")
            else:
                print(f"  {key}: {value:.4f}")

    except Exception as e:
        print(f"✗ 单步指标计算失败: {e}")
        return False

    # 4. 测试指标更新和累积
    print("\n4. 测试指标更新和累积...")
    try:
        # 模拟多个step的loss
        losses = [0.5, 0.4, 0.3, 0.2, 0.1]

        for i, loss in enumerate(losses):
            metrics_mgr.update(loss, metrics_dict, is_train=True)

        print("✓ 指标更新成功")
        print(f"  已更新 {len(losses)} 个step的指标")

    except Exception as e:
        print(f"✗ 指标更新失败: {e}")
        return False

    # 5. 测试指标计算和重置
    print("\n5. 测试指标计算和重置...")
    try:
        # 计算平均指标
        avg_metrics = metrics_mgr.compute(is_train=True)
        print("✓ 平均指标计算成功")
        print("平均指标:")
        for key, value in avg_metrics.items():
            if isinstance(value, list) and len(value) > 0:
                print(f"  {key}: {value[0]:.4f}")

        # 重置指标
        metrics_mgr.reset(is_train=True)
        print("✓ 指标重置成功")

    except Exception as e:
        print(f"✗ 指标计算/重置失败: {e}")
        return False

    # 6. 测试指标字典生成
    print("\n6. 测试指标字典生成...")
    try:
        # 使用最后一个step的数据
        loss = 0.1
        metrics_dict_final = metrics_mgr.get_metrics_dict(loss, metrics_dict, is_train=True)
        print("✓ 指标字典生成成功")
        print(f"  生成了 {len(metrics_dict_final)} 个指标项")
        print("  前5个指标项:")
        for i, (key, value) in enumerate(list(metrics_dict_final.items())[:5]):
            print(f"    {key}: {value}")

    except Exception as e:
        print(f"✗ 指标字典生成失败: {e}")
        return False

    print("\n" + "=" * 50)
    print("🎉 所有测试通过!")
    print("=" * 50)
    return True


def test_with_realistic_data():
    """使用更真实的测试数据"""
    print("\n" + "=" * 50)
    print("使用真实数据测试")
    print("=" * 50)

    metrics_mgr = CrossToothSegMetricsManager(
        task_type="segmentation",
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        num_classes=15,
        num_epochs=30
    )

    # 创建更真实的测试场景
    batch_size = 1
    num_points = 10000

    # 模拟更好的预测结果（接近真实情况）
    pred_logits = torch.randn(batch_size, num_points, 15)

    # 创建有结构的真实标签
    gt_labels = torch.zeros(batch_size * num_points, dtype=torch.long)
    # 模拟几个牙齿区域
    gt_labels[1000:1500] = 1  # 牙齿1
    gt_labels[2000:2500] = 2  # 牙齿2
    gt_labels[3000:3500] = 8  # 对应对称牙齿8
    gt_labels[4000:4500] = 9  # 对应对称牙齿9

    one_hot_labels = torch.nn.functional.one_hot(gt_labels, num_classes=15).reshape(batch_size, -1, 15)

    ground_truth = {
        'gt_labels': gt_labels,
        'onehot_gt_labels': one_hot_labels
    }

    # 创建对应的边缘掩码
    gt_edge_mask = torch.zeros(batch_size * num_points)
    # 在牙齿边界添加边缘点
    gt_edge_mask[1000:1005] = 1  # 牙齿1边界
    gt_edge_mask[1495:1505] = 1  # 牙齿1边界
    gt_edge_mask[2000:2005] = 1  # 牙齿2边界
    gt_edge_mask[2495:2505] = 1  # 牙齿2边界
    ground_truth['gt_edge_mask'] = gt_edge_mask

    # 点云坐标
    pos = torch.randn(batch_size * num_points, 3)

    # 计算指标
    predictions = {'seg_result': pred_logits}
    metrics_dict = metrics_mgr.compute_step_metrics(predictions, ground_truth, pos=pos)

    print("真实数据测试结果:")
    for key, value in metrics_dict.items():
        if not isinstance(value, dict):
            print(f"  {key}: {value:.4f}")


def main():
    # 运行基本测试
    success = test_cross_tooth_seg_metrics_manager()

    if success:
        # 运行真实数据测试
        test_with_realistic_data()
    else:
        print("基本测试失败，跳过真实数据测试")


if __name__ == '__main__':
    main()
