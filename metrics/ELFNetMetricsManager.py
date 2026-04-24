import torch
from torch import nn
from torchmetrics import Dice, Accuracy, Precision, Recall, JaccardIndex


class ELFNetMetricsManager:
    """
    用于管理三分支模型评估指标。目前尚未确定稳定的接口，后续可能会根据需求进行调整。
    """
    def __init__(self, task_type: str, device: torch.device = None, num_classes: int = 2, num_epochs: int = 30):

        # super().__init__(task_type, device)
        self.task_type = task_type
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_epochs = num_epochs
        self.num_classes = num_classes

        self._init_metrics()

    def _init_metrics(self):
        self.train_metrics = {
            'losses': [],
            'tae': {'mdsc': [], 'msen': [], 'mppv': [], 'macc': [], 'miou': []},
            'tte': {'mdsc': [], 'msen': [], 'mppv': [], 'macc': [], 'miou': []},
            'tg': {'mdsc': [], 'msen': [], 'mppv': [], 'macc': [], 'miou': []}
        }
        self.val_metrics = {
            'test_losses': [],
            'tae': {'mdsc': [], 'msen': [], 'mppv': [], 'macc': [], 'miou': []},
            'tte': {'mdsc': [], 'msen': [], 'mppv': [], 'macc': [], 'miou': []},
            'tg': {'mdsc': [], 'msen': [], 'mppv': [], 'macc': [], 'miou': []}
        }

        self.dice = Dice(num_classes=self.num_classes, average='macro').to(self.device)
        self.accuracy = Accuracy(task='multiclass', num_classes=self.num_classes).to(self.device)
        self.precision = Precision(task='multiclass', num_classes=self.num_classes, average='macro').to(self.device)
        self.recall = Recall(task='multiclass', num_classes=self.num_classes, average='macro').to(self.device)
        self.iou = JaccardIndex(task='multiclass', num_classes=self.num_classes, average='macro').to(self.device)

    def calc_branch_metrics(self, pred, labels, class_weights):
        """计算单个分支的评估指标
        Args:
        pred: [1, 2, 10000] - 概率分布 (batch_size=1, num_classes=2, num_points=10000)
        labels: [10000, 1, 2] - 需要转换为标准格式
        """
        # dsc = weighting_DSC(pred.transpose(2, 1), labels, class_weights)
        # sen = weighting_SEN(pred.transpose(2, 1), labels, class_weights)
        # ppv = weighting_PPV(pred.transpose(2, 1), labels, class_weights)
        # acc = weighting_ACC(pred.transpose(2, 1), labels, class_weights)
        # iou = weighting_IOU(pred.transpose(2, 1), labels, class_weights)
        # 处理 labels 形状
        # if labels.shape == torch.Size([10000, 1, 2]):
        labels = torch.argmax(labels, dim=-1).squeeze(1).unsqueeze(0)
        pred_classes = torch.argmax(pred, dim=1)

        # 确保数据类型
        pred_classes = pred_classes.long().to(self.device)
        labels = labels.long().to(self.device)

        # 计算指标
        dsc = self.dice(pred_classes, labels)
        sen = self.recall(pred_classes, labels)
        ppv = self.precision(pred_classes, labels)
        acc = self.accuracy(pred_classes, labels)
        iou = self.iou(pred_classes, labels)


        return dsc, sen, ppv, acc, iou

    # 在_compute_groups_metrics中使用
    def calc_groups_metrics(self, predictions, ground_truth, class_weights):
        """计算所有分支的指标"""
        # 背景分支
        bg_gt_bin_labels = (ground_truth['labels']!=0).long().reshape([-1, 1])          # [B * N, 1]
        bg_pd_1, bg_weight_1, bg_pd_2 = predictions['bg']
        onehot_bg_labels = nn.functional.one_hot(bg_gt_bin_labels, num_classes=2)       # [B , N, 2]
        bg_pd = torch.cat([-bg_pd_2, bg_pd_2], dim=1)                            # [B, 2, N ]
        dsc_tg, sen_tg, ppv_tg, acc_tg, iou_tg = self.calc_branch_metrics(
            bg_pd, onehot_bg_labels, class_weights)

        # tae分支
        tae_gt_bin_labels = ground_truth['tae_bin_labels'].reshape([-1, 1])
        tae_pd_1, tae_weight_1, tae_pd_2 = predictions['tae']
        onehot_tae_labels = nn.functional.one_hot(tae_gt_bin_labels, num_classes=2)
        tae_pd = torch.cat([-tae_pd_2, tae_pd_2], dim=1)
        dsc_tae, sen_tae, ppv_tae, acc_tae, iou_tae = self.calc_branch_metrics(
            tae_pd, onehot_tae_labels, class_weights)

        # tte分支
        tte_gt_bin_labels = ground_truth['tte_bin_labels'].reshape([-1, 1])
        tte_pd_1, tte_weight_1, tte_pd_2 = predictions['tte']
        onehot_tte_labels = nn.functional.one_hot(tte_gt_bin_labels, num_classes=2)
        tte_pd = torch.cat([-tte_pd_2, tte_pd_2], dim=1)
        dsc_tte, sen_tte, ppv_tte, acc_tte, iou_tte = self.calc_branch_metrics(
            tte_pd, onehot_tte_labels, class_weights)

        return {
            'tg': (dsc_tg, sen_tg, ppv_tg, acc_tg, iou_tg),
            'tae': (dsc_tae, sen_tae, ppv_tae, acc_tae, iou_tae),
            'tte': (dsc_tte, sen_tte, ppv_tte, acc_tte, iou_tte)
        }

    def update(self, loss, tg_metrics, tae_metrics, tte_metrics, is_train=True):
        """分组传参方式"""
        metrics = self.train_metrics if is_train else self.val_metrics
        metrics['losses' if is_train else 'test_losses'].append(loss)

        # 按分支处理
        for branch, branch_metrics in [('tg', tg_metrics), ('tae', tae_metrics), ('tte', tte_metrics)]:
            for i, metric_name in enumerate(['mdsc', 'msen', 'mppv', 'macc', 'miou']):
                metrics[branch][metric_name].append(branch_metrics[i])

    def compute(self, is_train=True):
        metrics = self.train_metrics if is_train else self.val_metrics
        for branch in ['tg', 'tae', 'tte']:
            for metric in ['mdsc', 'msen', 'mppv', 'macc', 'miou']:
                values = metrics[branch][metric]
                # result[f'{branch}_{metric}'] = sum(values) / len(values) if values else 0.0
                # 存储为列表，便于后续的打印、重置等操作
                metrics[branch][metric] = [sum(values) / len(values) if values else 0.0]
        return metrics

    def reset(self, is_train=True):
        metrics = self.train_metrics if is_train else self.val_metrics
        for branch in ['tg', 'tae', 'tte']:
            for metric in ['mdsc', 'msen', 'mppv', 'macc', 'miou']:
                metrics[branch][metric].clear()
        metrics['losses' if is_train else 'test_losses'].clear()

    def get_metrics_dict(self, loss, tg_metrics, tae_metrics, tte_metrics, is_train=True):
        """构造并返回标准格式的指标字典 - 分组传参版本"""
        prefix = "train/" if is_train else "test/"

        # 解包指标
        dsc_tg, sen_tg, ppv_tg, acc_tg, iou_tg = tg_metrics
        dsc_tae, sen_tae, ppv_tae, acc_tae, iou_tae = tae_metrics
        dsc_tte, sen_tte, ppv_tte, acc_tte, iou_tte = tte_metrics

        metrics = {
            f"{prefix}loss": loss,
            f"{prefix}tae/dsc": dsc_tae,
            f"{prefix}tae/sen": sen_tae,
            f"{prefix}tae/ppv": ppv_tae,
            f"{prefix}tae/acc": acc_tae,
            f"{prefix}tae/iou": iou_tae,
            f"{prefix}tte/dsc": dsc_tte,
            f"{prefix}tte/sen": sen_tte,
            f"{prefix}tte/ppv": ppv_tte,
            f"{prefix}tte/acc": acc_tte,
            f"{prefix}tte/iou": iou_tte,
            f"{prefix}tg/dsc": dsc_tg,
            f"{prefix}tg/sen": sen_tg,
            f"{prefix}tg/ppv": ppv_tg,
            f"{prefix}tg/acc": acc_tg,
            f"{prefix}tg/iou": iou_tg,
        }
        return metrics

    def _prepare_epoch_metrics(self, current_epoch):
        """准备epoch指标数据，供打印和保存CSV使用"""
        if len(self.train_metrics['losses']) > 0 and len(self.val_metrics['test_losses']) > 0:
            pd_dict = {
                'epoch': current_epoch + 1,
                'train_loss': self.train_metrics['losses'][-1],
                'train_DSC_tae': self.train_metrics['tae']['mdsc'][-1],
                'train_SEN_tae': self.train_metrics['tae']['msen'][-1],
                'train_PPV_tae': self.train_metrics['tae']['mppv'][-1],
                'train_ACC_tae': self.train_metrics['tae']['macc'][-1],
                'train_IOU_tae': self.train_metrics['tae']['miou'][-1],
                'train_DSC_tte': self.train_metrics['tte']['mdsc'][-1],
                'train_SEN_tte': self.train_metrics['tte']['msen'][-1],
                'train_PPV_tte': self.train_metrics['tte']['mppv'][-1],
                'train_ACC_tte': self.train_metrics['tte']['macc'][-1],
                'train_IOU_tte': self.train_metrics['tte']['miou'][-1],
                'train_DSC_tg': self.train_metrics['tg']['mdsc'][-1],
                'train_SEN_tg': self.train_metrics['tg']['msen'][-1],
                'train_PPV_tg': self.train_metrics['tg']['mppv'][-1],
                'train_ACC_tg': self.train_metrics['tg']['macc'][-1],
                'train_IOU_tg': self.train_metrics['tg']['miou'][-1],
                'test_loss': self.val_metrics['test_losses'][-1],
                'test_DSC_tae': self.val_metrics['tae']['mdsc'][-1],
                'test_SEN_tae': self.val_metrics['tae']['msen'][-1],
                'test_PPV_tae': self.val_metrics['tae']['mppv'][-1],
                'test_ACC_tae': self.val_metrics['tae']['macc'][-1],
                'test_IOU_tae': self.val_metrics['tae']['miou'][-1],
                'test_DSC_tte': self.val_metrics['tte']['mdsc'][-1],
                'test_SEN_tte': self.val_metrics['tte']['msen'][-1],
                'test_PPV_tte': self.val_metrics['tte']['mppv'][-1],
                'test_ACC_tte': self.val_metrics['tte']['macc'][-1],
                'test_IOU_tte': self.val_metrics['tte']['miou'][-1],
                'test_DSC_tg': self.val_metrics['tg']['mdsc'][-1],
                'test_SEN_tg': self.val_metrics['tg']['msen'][-1],
                'test_PPV_tg': self.val_metrics['tg']['mppv'][-1],
                'test_ACC_tg': self.val_metrics['tg']['macc'][-1],
                'test_IOU_tg': self.val_metrics['tg']['miou'][-1]
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
            print(f"{'DSC_TAE':<20} | {pd_dict['train_DSC_tae']:<15.4f} | {pd_dict['test_DSC_tae']:<15.4f}")
            print(f"{'SEN_TAE':<20} | {pd_dict['train_SEN_tae']:<15.4f} | {pd_dict['test_SEN_tae']:<15.4f}")
            print(f"{'PPV_TAE':<20} | {pd_dict['train_PPV_tae']:<15.4f} | {pd_dict['test_PPV_tae']:<15.4f}")
            print(f"{'ACC_TAE':<20} | {pd_dict['train_ACC_tae']:<15.4f} | {pd_dict['test_ACC_tae']:<15.4f}")
            print(f"{'IOU_TAE':<20} | {pd_dict['train_IOU_tae']:<15.4f} | {pd_dict['test_IOU_tae']:<15.4f}")
            print(f"{'DSC_TTE':<20} | {pd_dict['train_DSC_tte']:<15.4f} | {pd_dict['test_DSC_tte']:<15.4f}")
            print(f"{'SEN_TTE':<20} | {pd_dict['train_SEN_tte']:<15.4f} | {pd_dict['test_SEN_tte']:<15.4f}")
            print(f"{'PPV_TTE':<20} | {pd_dict['train_PPV_tte']:<15.4f} | {pd_dict['test_PPV_tte']:<15.4f}")
            print(f"{'ACC_TTE':<20} | {pd_dict['train_ACC_tte']:<15.4f} | {pd_dict['test_ACC_tte']:<15.4f}")
            print(f"{'IOU_TTE':<20} | {pd_dict['train_IOU_tte']:<15.4f} | {pd_dict['test_IOU_tte']:<15.4f}")
            print(f"{'DSC_TG':<20} | {pd_dict['train_DSC_tg']:<15.4f} | {pd_dict['test_DSC_tg']:<15.4f}")
            print(f"{'SEN_TG':<20} | {pd_dict['train_SEN_tg']:<15.4f} | {pd_dict['test_SEN_tg']:<15.4f}")
            print(f"{'PPV_TG':<20} | {pd_dict['train_PPV_tg']:<15.4f} | {pd_dict['test_PPV_tg']:<15.4f}")
            print(f"{'ACC_TG':<20} | {pd_dict['train_ACC_tg']:<15.4f} | {pd_dict['test_ACC_tg']:<15.4f}")
            print(f"{'IOU_TG':<20} | {pd_dict['train_IOU_tg']:<15.4f} | {pd_dict['test_IOU_tg']:<15.4f}")
            print("=" * 80)