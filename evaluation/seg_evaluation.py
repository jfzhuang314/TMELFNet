import numpy as np
import torch


def cal_binary_metric(gt_labels, pred_labels, target_class=None):
    """ 二分类评价指标
    缺省情况下，target_class=1
    """

    pred_softmax = torch.nn.functional.softmax(pred_labels, dim=1)
    _, pred_classes = torch.max(pred_softmax, dim=1)

    # 一般用以二分类评价任务
    if target_class is not None:
        gt_labels = (gt_labels == target_class)
        pred_classes = (pred_classes == target_class)

    correct_mask = (pred_classes == gt_labels)
    wrong_mask = (pred_classes != gt_labels)
    positive_mask = (gt_labels != 0)
    negative_mask = (gt_labels == 0)

    TP = torch.sum(positive_mask * correct_mask).item()
    TN = torch.sum(negative_mask * correct_mask).item()
    FP = torch.sum(positive_mask * wrong_mask).item()
    FN = torch.sum(negative_mask * wrong_mask).item()

    eps = 1e-6

    ACC = (TP + TN) / (FP + TP + FN + TN + eps)
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    F1 = 2 * (precision * recall) / (precision + recall + eps)
    IOU = TP / (FP + TP + FN + eps)

    return ACC, precision, recall, F1, IOU, pred_classes

def cal_binary_metric_label(gt_labels, pred_classes, target_class=None):
    """ 二分类评价指标
    缺省情况下，target_class=1
    """

    # 一般用以二分类评价任务
    if target_class is not None:
        gt_labels = (gt_labels == target_class)
        pred_classes = (pred_classes == target_class)

    correct_mask = (pred_classes == gt_labels)
    wrong_mask = (pred_classes != gt_labels)
    positive_mask = (gt_labels != 0)
    negative_mask = (gt_labels == 0)

    TP = torch.sum(positive_mask * correct_mask).item()
    TN = torch.sum(negative_mask * correct_mask).item()
    FN = torch.sum(positive_mask * wrong_mask).item()
    FP = torch.sum(negative_mask * wrong_mask).item()

    eps = 1e-6

    ACC = (TP + TN) / (FP + TP + FN + TN + eps)
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    F1 = 2 * (precision * recall) / (precision + recall + eps)
    IOU = TP / (FP + TP + FN + eps)

    return ACC, precision, recall, F1, IOU, pred_classes



def cal_edge_iou(gt_labels, pred_classes, target_class=None):
    """ 二分类评价指标
    缺省情况下，target_class=1
    """
    # 一般用以二分类评价任务
    if target_class is not None:
        gt_labels = (gt_labels == target_class)
        pred_classes = (pred_classes == target_class)

    intersection = np.logical_and(gt_labels, pred_classes)
    union = np.logical_or(gt_labels, pred_classes)

    esp = 10-6
    IOU = np.sum(intersection) / (np.sum(union) + esp)

    return IOU




def calculate_miou(gt_labels, pred_labels, n_class):
    ious = []
    weights = []

    # 确保输入是torch张量
    if not isinstance(gt_labels, torch.Tensor):
        gt_labels = torch.from_numpy(gt_labels)
    if not isinstance(pred_labels, torch.Tensor):
        pred_labels = torch.from_numpy(pred_labels)

    # 确保数据类型为long
    gt_labels = gt_labels.long()
    pred_labels = pred_labels.long()

    # 如果是批量数据，展平处理
    if gt_labels.dim() > 1:
        gt_labels = gt_labels.flatten()
    if pred_labels.dim() > 1:
        pred_labels = pred_labels.flatten()

    for c in range(1, n_class):  # 排除背景0
        gt_mask = (gt_labels == c).long()
        pred_mask = (pred_labels == c).long()
        pred_mask_onehot = torch.nn.functional.one_hot(pred_mask, num_classes=2).float()
        _, _, _, _, iou, _ = cal_binary_metric(gt_labels=gt_mask, pred_labels=pred_mask_onehot, target_class=1)
        ious.append(iou)
        weights.append(gt_mask.sum())

    valid_ious = [iou for iou in ious if iou >= 0.001]
    if len(valid_ious) > 0:
        mean_miou = np.mean(valid_ious)
    else:
        mean_miou = 0.0
    return mean_miou, ious, weights


def calculate_boundary_miou(gt_labels, pred_labels, gt_boundary, n_class):
    """
    只在GT边界面上计算mIoU
    """
    # 确保边界掩码是正确的数据类型
    if gt_boundary.dtype != torch.bool:
        gt_boundary = gt_boundary.bool()  # 转换为布尔类型

    gt_boundary_labels = gt_labels[gt_boundary]
    pred_boundary_labels = pred_labels[gt_boundary]
    miou, _, _ = calculate_miou(gt_boundary_labels, pred_boundary_labels, n_class=n_class)
    return miou


def cal_multi_cls_metric(gt_labels, pred_sem_labels, pred_ins_labels, is_half=None, vertices=None):
    """"
    1. It 's more resonable to evaluate on ground truth labels other than the predicted instance labels
    2. We report instance indexes over all observed classes, ignoring the background and absent categories.
    """

    gt_label_names = np.unique(gt_labels)
    gt_label_names = gt_label_names[gt_label_names != 0]
    IOU = 0
    F1 = 0
    ACC = 0
    PRECISION = 0
    RECALL = 0
    SEM_ACC = 0
    IOU_arr = []

    fg_len = len(gt_label_names)    # 前景类别数

    for gt_label_name in gt_label_names:
        gt_mask = (gt_labels == gt_label_name)
        ins_mask = (pred_ins_labels == gt_label_name)

        TP = np.count_nonzero(gt_mask * ins_mask)
        FN = np.count_nonzero(gt_mask * np.invert(ins_mask))
        FP = np.count_nonzero(np.invert(gt_mask) * ins_mask)
        TN = np.count_nonzero(np.invert(gt_mask) * np.invert(ins_mask))

        # ACC += (TP + TN) / (FP + TP + FN + TN)
        eps = 1e-6

        precision = TP / (TP+FP+eps)
        recall = TP / (TP+FN+eps)
        PRECISION += precision
        RECALL += recall
        F1 += 2*(precision*recall) / (precision + recall + eps)
        IOU += TP / (FP+TP+FN+eps)
        IOU_arr.append(TP / (FP+TP+FN+eps))
        #segmentation accuracy
        pred_sem_label_uniqs, pred_sem_label_counts = np.unique(pred_sem_labels[gt_mask], return_counts=True)
        sem_label_name = pred_sem_label_uniqs[np.argmax(pred_sem_label_counts)]
        if is_half:
            if sem_label_name == gt_label_name or sem_label_name + 8 == gt_label_name:
                SEM_ACC +=1
        else:
            if sem_label_name == gt_label_name:
                SEM_ACC +=1

    gt_fg_mask = gt_labels>-1
    ACC = np.sum(gt_labels[gt_fg_mask]==pred_ins_labels[gt_fg_mask]) / np.sum(gt_fg_mask)

    return (ACC, IOU/fg_len, F1/fg_len,PRECISION/fg_len, RECALL/fg_len,
            SEM_ACC/fg_len, IOU_arr)


def calc_bnd_bg_merged_iou(pred, labels, gt_edge_mask, num_classes=15):
    """
    计算边界、背景、合并对称IOU
    Args:
        pred: 预测结果 N,
        labels: 真实标签 N,
        gt_edge_mask: 边界掩码 N,
        num_classes: 类别数
    Returns:
        boundary_iou, background_iou, merge_pairs_ious_dict
    """
    # 1. 计算边界IOU - 假设类别0是背景，其他是前景
    boundary_iou = calculate_boundary_miou(gt_labels=labels, pred_labels=pred,
                                           gt_boundary=gt_edge_mask, n_class=num_classes)

    # 2. 计算背景IOU - 类别0
    background_pred = (pred == 0).squeeze().float()
    background_labels = (labels == 0).squeeze().float()
    _, _, _, _, background_iou, _ = cal_binary_metric_label(gt_labels=background_labels,
                                                      pred_classes=background_pred,
                                                      target_class=1)
    # 4. 计算T对称合并类别IOU (展开为Ta_b形式)
    merge_pairs_ious_dict = compute_merge_pairs_ious(pred, labels)

    return boundary_iou, background_iou, merge_pairs_ious_dict

def compute_merge_pairs_ious(pred, labels):
    """
    计算T对称合并类别IOUs，返回字典形式 {metric_name: iou_value}
    """
    symmetry_ious = {}
    merge_pairs = [(1, 8), (2, 9), (3, 10), (4, 11), (5, 12), (6, 13), (7, 14)]

    for a, b in merge_pairs:
        # 创建合并的标签和预测
        gt_merge = ((labels == a) | (labels == b)).reshape(-1).long()
        pred_merge = ((pred == a) | (pred == b)).reshape(-1).long()

        # 计算二分类IOU
        _, _, _, _, merge_iou,_ = cal_binary_metric_label(gt_labels=gt_merge,
                                                       pred_classes=pred_merge,
                                                       target_class=1)
        metric_name = f'mIOU_{a}_{b}'
        symmetry_ious[metric_name] = merge_iou

    return symmetry_ious
