import torch
from external_libs.pointnet2_utils.pointnet2_utils import square_distance

# def distance_loss(pred_distance, sample_xyz, centroid):
#     pred_distance = pred_distance.view(-1, sample_xyz.shape[2])
#     sample_xyz = sample_xyz.permute(0,2,1)
#     centroid = centroid.permute(0,2,1)
#     dists = square_distance(sample_xyz, centroid)
#     sorted_dists, _ = dists.sort(dim=-1)
#
#     # Check if the third dimension of sorted_dists has a non-zero size
#     if sorted_dists.size(2) == 0:
#         print("Warning: sorted_dists has size 0 in dimension 2. Returning 0 loss.")
#         return torch.tensor(0.0, device=pred_distance.device)
#
#
#     min_dists = sorted_dists[:, :, 0]
#     min_dists = torch.sqrt(min_dists)
#     loss = torch.nn.functional.smooth_l1_loss(pred_distance, min_dists)
#     return loss
#
# def centroid_dist_loss(pred_offset, sample_xyz, distance, centroid):
#     distance = distance.view(-1, sample_xyz.shape[2])
#     pred_offset = pred_offset.permute(0,2,1)
#     sample_xyz = sample_xyz.permute(0,2,1)
#     centroid = centroid.permute(0,2,1)
#
#     pred_centroid = torch.add(pred_offset, sample_xyz)
#
#     pred_ct_dists = square_distance(pred_centroid, centroid)
#     sorted_pred_ct_dists, _ = pred_ct_dists.sort(dim=-1)
#     min_pred_ct_dists = sorted_pred_ct_dists[:, :, 0]
#     pred_ct_mask = distance.le(0.2)
#     fin_pred_ct_dists = torch.masked_select(min_pred_ct_dists, pred_ct_mask)
#     loss = torch.div(torch.sum(fin_pred_ct_dists), torch.count_nonzero(pred_ct_mask))
#
#     ct_dists = square_distance(centroid, pred_centroid)
#     sorted_ct_dists, _ = ct_dists.sort(dim=-1)
#     min_ct_dists = sorted_ct_dists[:, :, 0]
#     ct_mask = min_ct_dists.le(0.2)
#     fin_ct_dists = torch.masked_select(min_ct_dists, ct_mask)
#     loss += torch.div(torch.sum(fin_ct_dists), torch.count_nonzero(ct_mask))
#     return loss
#
# def chamfer_distance_loss(pred_offset, sample_xyz, centroid):
#     pred_offset = pred_offset.permute(0,2,1)
#     sample_xyz = sample_xyz.permute(0,2,1)
#     centroid = centroid.permute(0,2,1)
#
#     pred_centroid = torch.add(pred_offset, sample_xyz)
#
#     pred_ct_dists = square_distance(pred_centroid, centroid)
#     sorted_pred_ct_dists, _ = pred_ct_dists.sort(dim=-1)
#     min_pred_ct_dists = sorted_pred_ct_dists[:, :, :2]
#
#     pred_ct_mask = min_pred_ct_dists[:,:,0].le(0.2)
#
#     # ratio = torch.div(min_pred_ct_dists[:,:,0], min_pred_ct_dists[:,:,1])
#     # Check if the third dimension has at least 2 elements
#     if min_pred_ct_dists.size(2) >= 2:
#         ratio = torch.div(min_pred_ct_dists[:,:,0], min_pred_ct_dists[:,:,1])
#     else:
#         # Handle the case where the third dimension size is less than 2
#         # You can choose to return a default value or raise a more informative error
#         print("Warning: min_pred_ct_dists has size less than 2 in dimension 2. Using default ratio of 1.")
#         ratio = torch.ones_like(min_pred_ct_dists[:,:,0])
#
#     ratio = torch.masked_select(ratio, pred_ct_mask)
#
#     loss = torch.div(torch.sum(ratio), torch.count_nonzero(pred_ct_mask))
#     return loss
#
# def centroid_loss(pred_offset, sample_xyz, distance, centroid):
#     dist_loss = distance_loss(distance, sample_xyz, centroid)
#     cent_loss = centroid_dist_loss(pred_offset, sample_xyz, distance, centroid)
#     chamf_loss = chamfer_distance_loss(pred_offset, sample_xyz, centroid)
#     return dist_loss, cent_loss, chamf_loss

def distance_loss(pred_distance, sample_xyz, centroid):
    # 统一转换为 float32
    pred_distance = pred_distance.float()
    sample_xyz = sample_xyz.float()
    centroid = centroid.float()

    pred_distance = pred_distance.view(-1, sample_xyz.shape[2])
    sample_xyz = sample_xyz.permute(0,2,1)
    centroid = centroid.permute(0,2,1)
    dists = square_distance(sample_xyz, centroid)

    # 防止数值不稳定导致的负数
    # dists = torch.clamp(dists, min=0.0)

    sorted_dists, _ = dists.sort(dim=-1)

    # Check if the third dimension of sorted_dists has a non-zero size
    if sorted_dists.size(2) == 0:
        print("Warning: sorted_dists has size 0 in dimension 2. Returning 0 loss.")
        return torch.tensor(0.0, device=pred_distance.device)


    min_dists = sorted_dists[:, :, 0]

    # 防止负数开平方
    # min_dists = torch.clamp(min_dists, min=1e-8)
    min_dists = torch.sqrt(min_dists)

    # 检查并处理 NaN/Inf
    if torch.isnan(min_dists).any() or torch.isinf(min_dists).any():
        print(f"Warning: min_dists contains NaN/Inf. Replacing with 100.0.")
        min_dists = torch.nan_to_num(min_dists, nan=100.0, posinf=100.0, neginf=100.0)

    loss = torch.nn.functional.smooth_l1_loss(pred_distance, min_dists)

    # 最终检查 loss
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"Warning: distance_loss is NaN/Inf. Returning 0.")
        return torch.tensor(100.0, device=pred_distance.device)

    return loss

def centroid_dist_loss(pred_offset, sample_xyz, distance, centroid):
    # 统一转换为 float32
    pred_offset = pred_offset.float()
    sample_xyz = sample_xyz.float()
    distance = distance.float()
    centroid = centroid.float()

    distance = distance.view(-1, sample_xyz.shape[2])
    pred_offset = pred_offset.permute(0,2,1)
    sample_xyz = sample_xyz.permute(0,2,1)
    centroid = centroid.permute(0,2,1)

    pred_centroid = torch.add(pred_offset, sample_xyz)

    pred_ct_dists = square_distance(pred_centroid, centroid)
    sorted_pred_ct_dists, _ = pred_ct_dists.sort(dim=-1)
    min_pred_ct_dists = sorted_pred_ct_dists[:, :, 0]
    pred_ct_mask = distance.le(0.2)
    fin_pred_ct_dists = torch.masked_select(min_pred_ct_dists, pred_ct_mask)

    # 防止除以零
    nonzero_count = torch.count_nonzero(pred_ct_mask)
    if nonzero_count > 0:
        loss = torch.div(torch.sum(fin_pred_ct_dists), nonzero_count)
    else:
        loss = torch.tensor(0.0, device=pred_offset.device)

    ct_dists = square_distance(centroid, pred_centroid)
    sorted_ct_dists, _ = ct_dists.sort(dim=-1)
    min_ct_dists = sorted_ct_dists[:, :, 0]
    ct_mask = min_ct_dists.le(0.2)
    fin_ct_dists = torch.masked_select(min_ct_dists, ct_mask)

    # 防止除以零
    nonzero_count = torch.count_nonzero(ct_mask)
    if nonzero_count > 0:
        loss += torch.div(torch.sum(fin_ct_dists), nonzero_count)

    # 检查 NaN/Inf
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"Warning: centroid_dist_loss is NaN/Inf. Returning 0.")
        return torch.tensor(100.0, device=pred_offset.device)

    return loss

def chamfer_distance_loss(pred_offset, sample_xyz, centroid):
    # 统一转换为 float32
    pred_offset = pred_offset.float()
    sample_xyz = sample_xyz.float()
    centroid = centroid.float()

    pred_offset = pred_offset.permute(0,2,1)
    sample_xyz = sample_xyz.permute(0,2,1)
    centroid = centroid.permute(0,2,1)

    pred_centroid = torch.add(pred_offset, sample_xyz)

    pred_ct_dists = square_distance(pred_centroid, centroid)

    # 防止负数
    pred_ct_dists = torch.clamp(pred_ct_dists, min=0.0)

    sorted_pred_ct_dists, _ = pred_ct_dists.sort(dim=-1)
    min_pred_ct_dists = sorted_pred_ct_dists[:, :, :2]

    pred_ct_mask = min_pred_ct_dists[:,:,0].le(0.2)

    # Check if the third dimension has at least 2 elements
    if min_pred_ct_dists.size(2) >= 2:
        # 防止除以零
        denominator = torch.clamp(min_pred_ct_dists[:,:,1], min=1e-8)
        ratio = torch.div(min_pred_ct_dists[:,:,0], denominator)
    else:
        print("Warning: min_pred_ct_dists has size less than 2 in dimension 2. Using default ratio of 1.")
        ratio = torch.ones_like(min_pred_ct_dists[:,:,0])

    ratio = torch.masked_select(ratio, pred_ct_mask)

    # 防止 NaN/Inf
    ratio = torch.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=0.0)

    nonzero_count = torch.count_nonzero(pred_ct_mask)
    if nonzero_count > 0:
        loss = torch.div(torch.sum(ratio), nonzero_count)
    else:
        loss = torch.tensor(0.0, device=pred_offset.device)

    # 检查 NaN/Inf
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"Warning: chamfer_distance_loss is NaN/Inf. Returning 0.")
        return torch.tensor(100.0, device=pred_offset.device)

    return loss

def centroid_loss(pred_offset, sample_xyz, distance, centroid):
    dist_loss = distance_loss(distance, sample_xyz, centroid)
    cent_loss = centroid_dist_loss(pred_offset, sample_xyz, distance, centroid)
    chamf_loss = chamfer_distance_loss(pred_offset, sample_xyz, centroid)
    return dist_loss, cent_loss, chamf_loss


def first_seg_loss(pred_mask_1, pred_weight_1, gt_bin_label):
    # pred_mask_1: batch_size, 2, cropped
    # pred_weight_1: batch_size, 1, cropped
    # gt_label: batch_size, 1, cropped
    
    gt_bin_label = gt_bin_label.type(torch.long).view(gt_bin_label.shape[0],-1)

    bce_1 = torch.nn.NLLLoss(reduction='none')(pred_mask_1, gt_bin_label)
    pred_weight_1 = torch.sigmoid(pred_weight_1).view(pred_weight_1.shape[0],-1)
    loss = torch.mean((bce_1 * pred_weight_1) ** 2 + (1-pred_weight_1)**2)
    
    
    return loss

def first_seg_mask_loss(pred_mask_1, pred_weight_1, gt_label):
    # pred_mask_1: batch_size, 2, cropped
    # pred_weight_1: batch_size, 1, cropped
    # gt_label: batch_size, 1, cropped
    
    gt_label = gt_label.view(gt_label.shape[0],-1)
    gt_bin_label = torch.ones_like(gt_label)
    gt_bin_label[gt_label == -1] = 0
    gt_bin_label = gt_bin_label.type(torch.long)

    bce_1 = torch.nn.CrossEntropyLoss(reduction='none')(pred_mask_1, gt_bin_label)
    pred_weight_1 = torch.sigmoid(pred_weight_1).view(pred_weight_1.shape[0],-1)    
    loss = torch.mean((bce_1 * pred_weight_1) ** 2 + (1-pred_weight_1)**2)/pred_weight_1.shape[1]
    return loss

def second_seg_loss(pred_mask_2, pred_weight_1, gt_bin_label):
    # pred_mask_2: batch_size, 2, cropped
    # pred_weight_1: batch_size, 1, cropped
    # gt_label: batch_size, 1, cropped

    gt_bin_label = gt_bin_label.type(torch.float32).view(gt_bin_label.shape[0],-1)

    pred_mask_2 = pred_mask_2.view(pred_mask_2.shape[0], -1)
    bce_2 = torch.nn.BCEWithLogitsLoss(reduction='none')(pred_mask_2, gt_bin_label)
    pred_weight_1 = torch.sigmoid(pred_weight_1).view(pred_weight_1.shape[0],-1)
    loss = torch.mean((2.0-pred_weight_1)*bce_2)
    
    return loss

def second_seg_mask_loss(pred_mask_2, pred_weight_1, gt_label):
    # pred_mask_2: batch_size, 2, cropped
    # pred_weight_1: batch_size, 1, cropped
    # gt_label: batch_size, 1, cropped
    
    gt_label = gt_label.view(gt_label.shape[0],-1)
    gt_bin_label = torch.ones_like(gt_label)
    gt_bin_label[gt_label == -1] = 0
    gt_bin_label = gt_bin_label.type(torch.float32)

    pred_mask_2 = pred_mask_2.view(pred_mask_2.shape[0], -1)
    bce_2 = torch.nn.BCEWithLogitsLoss(reduction='none')(pred_mask_2, gt_bin_label)
    pred_weight_1 = torch.sigmoid(pred_weight_1).view(pred_weight_1.shape[0],-1)
    loss = torch.sum((2.0-pred_weight_1)*bce_2)/pred_weight_1.shape[1]
    return loss

def id_loss(gt_label, pred_id):
    # gt_label: batch_size, 1, 1
    # pred_id : batch_size, 8
    
    gt_label = gt_label.view(-1).type(torch.long)
    loss = torch.nn.CrossEntropyLoss()(pred_id, gt_label)
    return loss

def segmentation_loss(pd_1, weight_1, pd_2, id_pred, pred_cluster_gt_ids, pred_cluster_gt_points_bin_labels):
    """
        Calculate the segmentation - related losses, including identity prediction loss,
        first - stage segmentation loss, and second - stage segmentation loss.

        :param pd_1: Output of the first - stage segmentation model, typically a tensor of shape (batch_size, 2, cropped)
        :param weight_1: Weights for the first - stage segmentation, a tensor of shape (batch_size, 1, cropped)
        :param pd_2: Output of the second - stage segmentation model, typically a tensor of shape (batch_size, 2, cropped)
        :param id_pred: Identity prediction output, a tensor of shape (batch_size, 17)
        :param pred_cluster_gt_ids: Ground truth identity labels for the predicted clusters, a tensor of shape (batch_size, 1, 1)
        :param pred_cluster_gt_points_bin_labels: Ground truth binary labels for the points in the predicted clusters, a tensor of shape (batch_size, 1, cropped)
        :return: A tuple containing the first - stage segmentation loss, second - stage segmentation loss, and identity prediction loss
        """
    id_pred_loss = id_loss(pred_cluster_gt_ids, id_pred)
    seg_1_loss = first_seg_loss(pd_1, weight_1, pred_cluster_gt_points_bin_labels)
    seg_2_loss = second_seg_loss(pd_2, weight_1, pred_cluster_gt_points_bin_labels)

    return seg_1_loss, seg_2_loss, id_pred_loss

def segmentation_mask_loss(pd_1, weight_1, pd_2, id_pred, cropped_gt_labels):
    seg_1_loss = first_seg_mask_loss(pd_1, weight_1, cropped_gt_labels)
    seg_2_loss = second_seg_mask_loss(pd_2, weight_1, cropped_gt_labels)
    loss = seg_1_loss + seg_2_loss
    return loss