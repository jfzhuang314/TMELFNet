import functools
import os
from functools import lru_cache
from os.path import exists

import torch
from scipy.spatial import Delaunay, KDTree
from datasets.transform.t3dsPlus_transforms import *
from datasets.transform.PointSamplingIndexer import *
from external_libs.mesh_vis import TMCVisPV
from external_libs.pv_gen_utils import np_to_mesh
from metrics.CrossToothSegMetricsManager import DynamicRadiusNNEdgeMaskWithPos
from collections import defaultdict
from external_libs import gen_utils as gu, mesh_vis
from external_libs import ops_utils as ou
from scipy.spatial import cKDTree
from external_libs.mesh_utils import *
import external_libs.pv_gen_utils as pvgu
import open3d as o3d


def interp_contour(contour_points):
    """
    针对平均边长较长的边拆分为多条较短的边，要求contour_points是闭合的。
    参数:
    contour_points: 输入的轮廓点，形状为(n_points, 2)或(n_points, 3)
    返回:
    new_points: 新的点集，包含原始点和插值点
    is_original: 布尔数组，指示每个点是否为原始点
    avg_length: 原始轮廓的平均边长
    """
    n_points = contour_points.shape[0]
    if n_points < 3:
        raise ValueError('输入的轮廓点必须至少包含3个顶点')

    # 计算每条边的长度
    edge_lengths = np.zeros(n_points)
    for i in range(n_points):
        p1 = contour_points[i]
        p2 = contour_points[(i + 1) % n_points]
        edge_lengths[i] = np.linalg.norm(p2 - p1)

    # 计算平均边长
    avg_length = np.mean(edge_lengths)
    # print(f'平均边长: {avg_length:.4f}')

    # 初始化结果列表
    new_points_list = []
    is_original_list = []

    # 处理每条边
    for i in range(n_points):
        p1 = contour_points[i]
        p2 = contour_points[(i + 1) % n_points]
        edge_length = edge_lengths[i]

        # 添加原始起点
        new_points_list.append(p1)
        is_original_list.append(True)

        # 在长边上插入点
        if edge_length > avg_length:
            # 至少插入1个点
            n_insert = max(1, int(np.ceil(edge_length / avg_length)) - 1)
            for j in range(1, n_insert + 1):
                t = j / (n_insert + 1)
                new_point = p1 + t * (p2 - p1)
                new_points_list.append(new_point)
                is_original_list.append(False)

    # 转换为NumPy数组
    new_points = np.array(new_points_list)
    is_original = np.array(is_original_list, dtype=bool)

    return new_points, is_original, avg_length


def update_interp_points(new_points, is_original, V, flag_wo_bnd):
    """
    根据最近邻点的平均值更新插值点的位置
    参数:
    new_points: 包含原始点和插值点的点集，形状为(N, 2)或(N, 3)
    is_original: 布尔数组，指示每个点是否为原始点，形状为(N,)
    V: 参考点集，形状为(M, 2)或(M, 3)
    flag_wo_bnd: 布尔数组，指示V中哪些点参与计算，形状为(M,)

    返回:
    updated_points: 更新后的点集，形状为(N, 2)或(N, 3)
    """
    # 提取插值点
    interpolation_points = new_points[~is_original, :2]

    # 提取参与计算的参考点
    reference_points = V[flag_wo_bnd, :2]

    # 如果没有参考点，直接返回原结果
    if reference_points.size == 0:
        updated_points = new_points.copy()
        return updated_points

    # 对每个插值点，找到其在参考点中的3个最近邻点
    k = 3
    tree = KDTree(reference_points)
    distances, indices = tree.query(interpolation_points, k=k)

    # 初始化更新后的插值点
    updated_interpolation_points = np.zeros_like(interpolation_points)

    # 计算每个插值点的新位置
    for i in range(interpolation_points.shape[0]):
        # 获取3个最近邻点的坐标
        nearest_neighbors = reference_points[indices[i]]
        # 计算平均值作为新位置
        updated_interpolation_points[i] = np.mean(nearest_neighbors, axis=0)
    # 更新结果点集
    updated_points = new_points.copy()
    updated_points[~is_original, :2] = updated_interpolation_points
    return updated_points


def label_contour_segment(C):
    """
    标记顺时针有向轮廓中的特定线段
    参数:
    C: 顺时针有向的轮廓点集，形状为(N, 2)
    返回:
    segment_labels: 线段标记，形状为(N,)，0表示p1到p2的线段，1表示其余部分
    """
    # 寻找x<0区域的最高点p1
    left_points = C[C[:, 0] < 0]
    if len(left_points) == 0:
        raise ValueError("轮廓中没有x<0的点")
    idx_p1 = np.argmax(left_points[:, 1])
    p1 = left_points[idx_p1]

    # 寻找x>0区域的最高点p2
    right_points = C[C[:, 0] > 0]
    if len(right_points) == 0:
        raise ValueError("轮廓中没有x>0的点")
    idx_p2 = np.argmax(right_points[:, 1])
    p2 = right_points[idx_p2]

    # 在原始点集中找到p1和p2的索引
    # 使用KDTree实现knnsearch功能
    tree = KDTree(C)
    _, idx_p1_original = tree.query(p1.reshape(1, -1), k=1)
    _, idx_p2_original = tree.query(p2.reshape(1, -1), k=1)

    idx_p1_original = idx_p1_original.squeeze()
    idx_p2_original = idx_p2_original.squeeze()

    # 由于是顺时针方向，p1到p2的线段应该是从p1开始，沿着索引增加的方向到p2
    # 处理p1索引大于p2索引的情况（即线段跨越了点集的末尾和开头）
    N = len(C)
    segment_labels = np.ones(N, dtype=int)  # 初始全部标记为1

    if idx_p1_original < idx_p2_original:
        # p1在p2之前，直接标记p1到p2之间的线段为0
        segment_labels[idx_p1_original:idx_p2_original + 1] = 0
    else:
        # p1在p2之后，线段跨越了点集的末尾和开头
        segment_labels[idx_p1_original:] = 0
        segment_labels[:idx_p2_original + 1] = 0

    return segment_labels


# def point_to_segment_distance(V, flag, header, tail):
#     """
#     计算点集到线段集的距离（完全向量化版本）
#
#     参数:
#     V: 点集，形状为(n, 2)或(n, 3)
#     flag: 布尔数组，指示哪些点参与计算，形状为(n,)
#     header: 线段起点集，形状为(m, 2)或(m, 3)
#     tail: 线段终点集，形状为(m, 2)或(m, 3)
#     返回:
#     distances: 距离矩阵，形状为(n, m)，其中distances[i,j]表示点i到线段j的距离
#     """
#     n = V.shape[0]  # 顶点数量
#     m = header.shape[0]  # 线段数量
#
#     # 初始化距离矩阵
#     distances = np.zeros((n, m))
#     # 对flag为false的点，设置距离为一个很大的值
#     distances[~flag, :] = 10e2
#     # 获取flag为true的点的索引
#     idxs = np.where(flag)[0]
#
#     # 对每个顶点计算到所有线段的距离
#     for i in idxs:
#         p = V[i]
#         # 计算线段向量
#         ab = tail - header
#         # 计算线段长度的平方
#         ab_length_sq = np.sum(ab ** 2, axis=1) + 10e-6
#         # 计算顶点到线段起点的向量
#         ap = p - header
#         # 计算投影比例 t
#         t = np.sum(ap * ab, axis=1) / ab_length_sq
#         # 根据投影比例确定距离计算方式
#         # 预计算三种情况下的距离
#         norm_ap = np.sqrt(np.sum(ap ** 2, axis=1))  # t < 0 时的距离
#         bp = p - tail
#         norm_bp = np.sqrt(np.sum(bp ** 2, axis=1))  # t > 1 时的距离
#         proj = header + np.expand_dims(t, axis=1) * ab
#         norm_p_minus_proj = np.sqrt(np.sum((p - proj) ** 2, axis=1))  # 0 <= t <= 1 时的距离
#
#         # 创建标志向量
#         flag_less_than_0 = t < 0
#         flag_greater_than_1 = t > 1
#         flag_between_0_1 = ~flag_less_than_0 & ~flag_greater_than_1
#
#         # Ensure boolean masks are 1D
#         flag_less_than_0 = flag_less_than_0.flatten()
#         flag_greater_than_1 = flag_greater_than_1.flatten()
#         flag_between_0_1 = flag_between_0_1.flatten()
#
#         # Check if the lengths match
#         assert len(flag_less_than_0) == m, f"flag_less_than_0 length {len(flag_less_than_0)} does not match m {m}"
#         assert len(flag_greater_than_1) == m, f"flag_greater_than_1 length {len(flag_greater_than_1)} does not match m {m}"
#         assert len(flag_between_0_1) == m, f"flag_between_0_1 length {len(flag_between_0_1)} does not match m {m}"
#
#         # 根据标志向量选择合适的距离
#         distances[i, flag_less_than_0] = norm_ap[flag_less_than_0]
#         distances[i, flag_greater_than_1] = norm_bp[flag_greater_than_1]
#         distances[i, flag_between_0_1] = norm_p_minus_proj[flag_between_0_1]
#
#     return distances

def point_to_segment_distance(V, flag, header, tail, distance_threshold=0.01):
    """
    计算点集到线段集的距离（KDTree 优化版本 + 距离阈值）

    参数:
    V: 点集，形状为 (n, 2) 或 (n, 3)
    flag: 布尔数组，指示哪些点参与计算，形状为 (n,)
    header: 线段起点集，形状为 (m, 2) 或 (m, 3)
    tail: 线段终点集，形状为 (m, 2) 或 (m, 3)
    distance_threshold: 距离阈值，超过此阈值的点对不参与详细计算 (默认 0.1)
    返回:
    distances: 距离矩阵，形状为 (n, m)，其中 distances[i,j] 表示点 i 到线段 j 的距离
    """
    n = V.shape[0]
    m = header.shape[0]

    # 初始化距离矩阵，默认值设为一个大数
    distances = np.full((n, m), 10e6, dtype=np.float64)

    # 获取有效点的索引
    valid_indices = np.where(flag)[0]

    if len(valid_indices) == 0:
        return distances

    # 提取有效点
    valid_points = V[valid_indices]
    n_valid = len(valid_points)

    # ========== 优化 1: 使用 KDTree 替代广播计算 ==========
    # 计算线段中点
    segment_centers = (header + tail) / 2.0  # [m, 2]

    # 构建线段中心的 KDTree
    tree = cKDTree(segment_centers)

    # 查询每个有效点在阈值范围内的所有线段中心
    # query_ball_point 返回每个点在半径内的所有邻居索引
    neighbors_list = tree.query_ball_point(valid_points, r=distance_threshold * 10)  # 放宽系数

    # ========== 优化 2: 只计算邻近的点对 ==========
    # 收集需要计算的点对索引
    valid_point_idx = []
    valid_line_idx = []

    for pt_idx, line_indices in enumerate(neighbors_list):
        if line_indices:  # 如果有邻近的线段
            valid_point_idx.extend([pt_idx] * len(line_indices))
            valid_line_idx.extend(line_indices)

    # 如果没有需要计算的点对，直接返回
    if len(valid_point_idx) == 0:
        return distances

    valid_point_idx = np.array(valid_point_idx)
    valid_line_idx = np.array(valid_line_idx)
    k = len(valid_point_idx)  # 实际需要计算的点对数量

    # print(f"优化：从 {n_valid * m} 对减少到 {k} 对 ({k/(n_valid*m)*100:.2f}%)")

    # 预计算选中线段的相关量
    ab_selected = tail[valid_line_idx] - header[valid_line_idx]  # [k, 2]
    ab_length_sq_selected = np.sum(ab_selected ** 2, axis=1, keepdims=True) + 1e-6  # [k, 1]

    # 计算选中点到选中线段的向量 AP
    ap_selected = valid_points[valid_point_idx] - header[valid_line_idx]  # [k, 2]

    # 计算投影比例 t
    t_selected = np.sum(ap_selected * ab_selected, axis=1, keepdims=True) / ab_length_sq_selected  # [k, 1]

    # 计算三种情况的距离
    # 1. t < 0: 距离为 |AP|
    dist_ap_selected = np.sqrt(np.sum(ap_selected ** 2, axis=1))  # [k,]

    # 2. t > 1: 距离为 |BP|
    bp_selected = valid_points[valid_point_idx] - tail[valid_line_idx]  # [k, 2]
    dist_bp_selected = np.sqrt(np.sum(bp_selected ** 2, axis=1))  # [k,]

    # 3. 0 <= t <= 1: 距离为点到投影点的距离
    proj_selected = header[valid_line_idx] + t_selected * ab_selected  # [k, 2]
    dist_proj_selected = np.sqrt(np.sum((valid_points[valid_point_idx] - proj_selected) ** 2, axis=1))  # [k,]

    # 根据 t 的值选择合适的距离
    mask_t_less_0 = t_selected.flatten() < 0
    mask_t_greater_1 = t_selected.flatten() > 1
    mask_t_between = ~mask_t_less_0 & ~mask_t_greater_1

    final_distances = np.zeros(k, dtype=np.float64)
    final_distances[mask_t_less_0] = dist_ap_selected[mask_t_less_0]
    final_distances[mask_t_greater_1] = dist_bp_selected[mask_t_greater_1]
    final_distances[mask_t_between] = dist_proj_selected[mask_t_between]

    # 将结果填回完整的距离矩阵
    for idx in range(k):
        vp_idx = valid_point_idx[idx]
        vl_idx = valid_line_idx[idx]
        distances[valid_indices[vp_idx], vl_idx] = final_distances[idx]

    return distances



class TMCUtils:
    @staticmethod
    def square_distance(src, dst):
        """
        Calculate Euclid distance between each two points.

        src^T * dst = xn * xm + yn * ym + zn * zm；
        sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
        sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
        dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
             = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
        Input:
            src: source points, [B, N, C]
            dst: target points, [B, M, C]
        Output:
            dist: per-point square distance, [B, N, M]
        """
        # # Convert tensors to float32 if they are not already
        # if src.dtype not in [torch.float32, torch.float64]:
        #     src = src.float()
        # if dst.dtype not in [torch.float32, torch.float64]:
        #     dst = dst.float()
        src = src.float()
        dst = dst.float()

        B, N, _ = src.shape
        _, M, _ = dst.shape
        dist = -2 * torch.matmul(src, dst.permute(0, 2, 1)).float()
        dist += torch.sum(src ** 2, -1).view(B, N, 1).float()
        dist += torch.sum(dst ** 2, -1).view(B, 1, M).float()
        return dist

    @staticmethod
    def get_ddf(cropped_coord, center_points):
        """
        :param self:
        :param cropped_coord: BNC格式
        :param center_points: B
        :return:
        """
        device = center_points.device
        cropped_coord = cropped_coord.to(device)

        B, N, C = cropped_coord.shape

        # Check if center_points is already a tensor
        if not isinstance(center_points, torch.Tensor):
            center_points = torch.from_numpy(center_points).cuda()
        # else:
        #     center_points = center_points.cuda()  # Move the tensor to GPU
        ddf = TMCUtils.square_distance(cropped_coord, center_points)
        ddf = torch.sqrt(ddf)
        ddf *= (-4)
        ddf = torch.exp(ddf)
        ddf = ddf.permute(0, 2, 1)
        return ddf

    @staticmethod
    def map_crop2ori_label(pred_bin_labels, crop_nn_indexes, num_points):
        """注意这是二值语义标签，不是语义标签"""

        # 1. 获取预测标签
        pred_labels = torch.argmax(pred_bin_labels, dim=1)  # [14, 2048]
        pred_labels = pred_labels.view(-1)  # [14*2048]
        # 2. 获取原始点索引
        crop_indices = crop_nn_indexes.view(-1).to(pred_labels.device)  # [14*2048]

        final_pred = torch.zeros(num_points, dtype=torch.long).to(pred_labels.device)
        mask = (pred_labels>0)
        final_pred[crop_indices[mask]] = 1

        return final_pred

    @staticmethod
    def vote_seg_label(binseg_labels, pred_seg_labels, crop_nn_indexes, num_classes=15):
        """
        按照 prompt 要求实现语义分割投票：
        1. 确定每个 crop 的类别：统计 2048 个前景点中占比最大的非零类标
        2. 将 crop 中所有非背景的点都设为该类标，背景的点设为 0
        3. 使用投票机制映射回原始点云
        Args:
            binseg_labels: 形状是[clu, 2, crop_num]取值是0或1，代表背景或前景。
            pred_seg_labels:,形状是[16000,]，取值是0到14，代表预测的语义标签。
            crop_nn_indexes: 形状是[1, clu, crop_num]，代表了14个crop在原始形状中的序号。
        Returns:
            final_pred: 语义分割标签 [num_points]
        """
        # 1. 先要确定每个crop的类别；其规则是2048中属于前景的类标中占比最大的非零类标。
        device = pred_seg_labels.device
        num_points = pred_seg_labels.shape[0]                                 # 顶点数
        crop_num = crop_nn_indexes.shape[1]                                 # crop数
        crop_binseg_labels = torch.argmax(binseg_labels, dim=1).unsqueeze(0).to(device)      #[crop, 1, 2048]
        crop_seg_labels = pred_seg_labels[crop_nn_indexes].to(device)       # [crop, 1, 2048]
        # valid_mask = (crop_binseg_labels > 0)                              # [crop, 1, 2048]
        valid_crop_seg_labels = torch.zeros_like(crop_seg_labels).to(device)              # [crop, 1, 2048]


        for i in range(crop_num):
            foreground_mask = (crop_binseg_labels[0, i] > 0)
            if foreground_mask.any():
                foreground_labels = crop_seg_labels[0, i][foreground_mask].view(-1)  # 提取非零前景类标
                counts = torch.bincount(foreground_labels, minlength=num_classes)     # 统计每个类标出现的次数
                counts[0] = -1
                max_label = counts.argmax()                                         # 找到出现次数最多的类标
                valid_crop_seg_labels[0,i,foreground_mask] = max_label               # 将所有非背景点设为该类标

            else:
                valid_crop_seg_labels[0,i] = 0            # 如果没有前景点，设为背景类标

        # 3. 创建投票矩阵 (num_classes, num_points)
        vote_mat = torch.zeros((num_points, num_classes), dtype=torch.long).to(device)
        # 4. 使用scatter_add进行投票
        crop_indices = crop_nn_indexes.view(-1).to(device)                  # [14*2048]
        valid_crop_seg_labels = valid_crop_seg_labels.view(-1).to(device)   # [14*2048]
        values = torch.ones_like(valid_crop_seg_labels)                     # [28672]

        for cls in range(1,num_classes,1):
            mask = (valid_crop_seg_labels == cls)
            if mask.any():
                vote_mat[:, cls].scatter_add_(0, crop_indices[mask], values[mask])

        # vote_mat[:,0] = 0
        # 5. 获取最终预测结果 (多数投票)
        final_pred = torch.argmax(vote_mat, dim=1)  # [num_points]

        return final_pred

    @staticmethod
    def build_crop_batch(data, is_train=False, crop_num_16000=2048):
        """
        构建裁剪批次，根据TMC聚类结果，裁剪出每个聚类中心周围的点云样本。
        Args:
            data: 包含TMC聚类结果的数据集对象，包含点云、聚类中心、聚类标签等信息。
            is_train: 是否为训练模式，默认 False。
            crop_num: 裁剪后的点云样本数量，默认 2048。
        Returns:
            torch.Size([14, 6, 2048]) torch.Size([14, 1, 2048]) torch.Size([14, 1, 2048]) torch.Size([14, 3, 2048]) torch.Size([14])
            crop_nn_indexes: 每个聚类中心对应的裁剪点云索引，形状为 (1, clu, crop)，其中 clu 为聚类数量。
            cropped_feats: 裁剪后的点云特征，形状为 (clu, 32, crop)，其中 clu 为聚类数量，crop 为裁剪后的点云样本数量。
            crop_gt_binseg_label: 裁剪后的点云二值语义标签，形状为 (clu, 1, crop)
            crop_gt_seg_label: 裁剪后的点云语义标签，形状为 (clu, 1, crop)
            cropped_pos: 裁剪后的点云坐标，形状为 (clu, 3, crop)
            cluster_gt_seg_labels: 每个聚类中心对应的 gt 语义标签，形状为 (clu)，其中 clu 为聚类数量。
        """

        pts_num = data.pos.shape[0]
        # crop_num = crop_num_16000 * pts_num // 16000
        crop_num = int(crop_num_16000 * (pts_num / 16000)) # 裁剪后的点云样本数量； 16k点云是2048，

        # 1. 提取proposal中的聚类中心和聚类标签
        if is_train or not hasattr(data, 'tmc_proposal'):
            pred_cent = data.instance_centroids   # 训练时直接使用gt的聚类中心
            noise = torch.clamp(torch.randn_like(pred_cent) * 0.02, -0.02, 0.02)
            pred_cent = pred_cent + noise
        else:
            # print("has tmc_proposal")
            tmc_proposal = data.tmc_proposal
            pred_cent = tmc_proposal['cluster_centroids']

        # print(tmc_proposal['cluster_centroids'].shape,tmc_proposal['cluster_labels'],tmc_proposal['per_cluster_count'],
        #       tmc_proposal['cluster_num'])

        outs = TMCUtils.get_proposal(pos = data.pos, normals = data.normals, gt_labels = data.y,
                                     gt_cent = data.instance_centroids, gt_cent_label = data.instance_labels,
                                 pred_cent = pred_cent, crop_num = crop_num)

        return  outs

    @staticmethod
    def get_proposal( pos= None, normals=None, crop_num=2048, gt_labels=None,
                     gt_cent=None, gt_cent_label=None,pred_cent=None):
        """ 注意，输入的gt_cent和pred_cent都包含背景类，0 为背景类，不参与计算；
            输出的crops是不包含背景类的；我们在程序中做了相应处理。
        """

        if gt_labels is None or gt_cent is None or gt_cent_label is None or pred_cent is None:
            raise ValueError("gt_labels, gt_cent, gt_cent_label, pred_cent must be provided if pos, normals, crop_num, gt_labels is None")

        # 2. 提取gt聚类中心和gt聚类标签
        pred_cent = pred_cent.float()
        gt_cent = gt_cent.float()
        # 注意：0 为背景类，不参与计算
        dist_matrix = torch.cdist(pred_cent[1:], gt_cent[1:], p=2)
        sqd_argmin = torch.argmin(dist_matrix, dim=1)  # 最近邻匹配，容易出现多个预测点对应同一个 gt 点的情况
        cluster_gt_seg_labels = gt_cent_label[sqd_argmin + 1]  # 注意，0 为背景类，不参与计算
        pred_cent_wo_bg = pred_cent[1:]
        # 3. 提取点云特征、坐标、法线、gt 语义标签并reshape
        center_points = pred_cent_wo_bg.reshape(1, -1, 3)  # 1, clu， 3
        l0_xyz = pos.reshape(1, -1, 3)
        l0_normals = normals.reshape(1, -1, 3)  # B, N, 3
        gt_labels = gt_labels.reshape(1, -1, 1)


        # 4. 批量提取每个聚类中心周围的点云样本
        crop_nn_indexes = ou.get_nearest_neighbor_idx(gu.torch_to_numpy(l0_xyz), gu.torch_to_numpy(center_points),
                                                      crop_num=crop_num)
        crop_nn_indexes = torch.tensor(np.array(crop_nn_indexes))  # 将list转换为B, clu, 2048
        # 提取裁剪后的点云特征、坐标、法线
        crop_pos = ou.get_indexed_features(l0_xyz.permute(0, 2, 1), crop_nn_indexes)  # clu, 3,  crop
        crop_normals = ou.get_indexed_features(l0_normals.permute(0, 2, 1), crop_nn_indexes)  # clu, 3,  crop
        # # 计算相对于聚类中心的坐标（去中心化）
        ddf = ou.get_ddf(crop_pos[:, :3, :].permute(0, 2, 1), center_points.cpu().numpy())  # B, 1, crop
        crop_feats = torch.cat([crop_pos[:, :3, :], crop_normals[:, :3, :], ddf], axis=1)  # clu, 7, crop
        # 5. 提取裁剪后的gt语义标签
        crop_gt_seg_label = ou.get_indexed_features(gt_labels.permute(0, 2, 1), crop_nn_indexes)  # clu,1,crop
        crop_gt_binseg_label = torch.zeros_like(crop_gt_seg_label)
        for i in range(crop_gt_seg_label.shape[0]):
            mask = (crop_gt_seg_label[i, 0] == cluster_gt_seg_labels[i])
            crop_gt_binseg_label[i, 0, mask] = 1
        outs = {
            'crop_nn_indexes': crop_nn_indexes,
            'crop_feats': crop_feats,
            'crop_pos': crop_pos,
            'crop_gt_binseg_label': crop_gt_binseg_label,
            'crop_gt_seg_label': crop_gt_seg_label,
            'cluster_gt_seg_labels': cluster_gt_seg_labels,
        }
        return outs

    @staticmethod
    def vis_crops(cluster_gt_seg_labels, crop_gt_binseg_label, cropped_pos, data):
        # 网格化排列 crops 点云
        crops = []
        xy_offset = 0.5  # x,y 轴方向平移步长
        cols = int(np.ceil(np.sqrt(cropped_pos.shape[0])))  # 网格列数
        # 动态计算平移距离，避免重叠
        for i in range(cropped_pos.shape[0]):
            label_i = cluster_gt_seg_labels[i].cpu().numpy()
            j = label_i - 1  # 注意，0 为背景类，不参与计算，因而cluster_gt_seg_labels序号从1开始
            j = j + 1 if j > 6 else j

            crop_pcd = gu.np_to_pcd_with_label(
                cropped_pos[i].cpu().T.numpy(),
                crop_gt_binseg_label[i].cpu().T.numpy() * label_i
            )

            # 1. 先减去中心点（归一化到原点）
            points = np.asarray(crop_pcd.points)
            center = np.mean(points, axis=0)
            crop_pcd.translate(-center)

            # 网格状排列：按行列计算 x,y 偏移
            row = j // cols - cols // 2
            col = j % cols - cols // 2
            translate_x = col * xy_offset
            translate_y = row * xy_offset
            crop_pcd.translate([translate_x, translate_y, 0])
            crops.append(crop_pcd)

        # gt sems
        crop_pcd = gu.np_to_pcd_with_label(data.pos.cpu().numpy(),data.y.cpu().numpy())
        crop_pcd.translate([-cols * xy_offset * 1.2, 0, 0])
        crops.append(crop_pcd)

        # y_binseg
        crop_pcd = gu.np_to_pcd_with_label(data.pos.cpu().numpy(),data.y_binseg.cpu().numpy())
        crop_pcd.translate([-cols * xy_offset * 1.2, -cols * xy_offset * 1.2 / 2, 0])
        crops.append(crop_pcd)

        # y_voteseg
        crop_pcd = gu.np_to_pcd_with_label(data.pos.cpu().numpy(),data.y_voteseg.cpu().numpy())
        crop_pcd.translate([-cols * xy_offset * 1.2, cols * xy_offset * 1.2 / 2, 0])
        crops.append(crop_pcd)

        # version_name = generate_exp_name(cfg_obj.logger.exp_args)
        # print(version_name)
        gu.print_3d(*crops)


def count_calls(func):
    """统计函数调用次数的装饰器"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        wrapper.call_count += 1
        return func(*args, **kwargs)

    wrapper.call_count = 0
    wrapper.reset_count = lambda: setattr(wrapper, 'call_count', 0)
    wrapper.get_count = lambda: wrapper.call_count
    return wrapper


class AlphaShape:

    @staticmethod
    @count_calls
    def build_alpha_shape(points, alpha):
        """
        计算 alpha shape 的三角形面片
        参数:
        points: 点集，形状为 (n, 2)
        alpha: alpha 参数
        返回:
        vertices: 顶点坐标
        triangles: 三角形面片索引
        """
        # 进行 Delaunay 三角剖分
        tri = Delaunay(points)
        triangles = tri.simplices

        # 计算每个三角形的外接圆半径
        a = np.sqrt(np.sum((points[triangles[:, 0]] - points[triangles[:, 1]]) ** 2, axis=1))
        b = np.sqrt(np.sum((points[triangles[:, 1]] - points[triangles[:, 2]]) ** 2, axis=1))
        c = np.sqrt(np.sum((points[triangles[:, 2]] - points[triangles[:, 0]]) ** 2, axis=1))
        s = (a + b + c) / 2.0
        areas = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * areas + 10e-6)

        # 过滤掉外接圆半径大于 alpha 的三角形
        filtered_triangles = triangles[circum_r < alpha]

        return points, filtered_triangles

    # 使用 Open3D 从顶点和三角形创建网格
    @staticmethod
    def create_mesh_from_VF(vertices, triangles):
        """
        从顶点和三角形创建网格
        参数:
        vertices: 顶点坐标，形状为 (n, 2) 或 (n, 3)
        triangles: 三角形面片索引，形状为 (m, 3)
        返回:
        mesh: Open3D 网格对象
        """
        mesh = o3d.geometry.TriangleMesh()
        if vertices.shape[1] == 2:
            # 将 2D 坐标转换为 3D 坐标（z=0）
            vertices_3d = np.column_stack((vertices, np.zeros(vertices.shape[0])))
        else:
            vertices_3d = vertices

        # 设置顶点和三角形
        mesh.vertices = o3d.utility.Vector3dVector(vertices_3d)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)

        # 计算顶点法线，使网格在可视化时正确显示光照
        mesh.compute_vertex_normals()

        # 可选：设置网格颜色
        mesh.paint_uniform_color([0.5, 0.5, 0.5])  # 灰色
        # mesh.visualization.show_wireframe = True # 显示线框
        return mesh

    @staticmethod
    def clean_mesh(alpha_mesh):
        # 5. 提取alpha_mesh的边界
        # 预清理：确保 mesh 有效
        alpha_mesh.remove_degenerate_triangles()
        alpha_mesh.remove_duplicated_vertices()
        alpha_mesh.remove_unreferenced_vertices()


        # print(f"✓ alpha_mesh 构建完成：{len(alpha_mesh.vertices)} 顶点，{len(alpha_mesh.triangles)} 面片")

        # print(f"✓ alpha_mesh 构建完成：{len(alpha_mesh.vertices)} 顶点，{len(alpha_mesh.triangles)} 面片")

        return alpha_mesh

    @staticmethod
    def extract_boundary_from_alpha_shape(mesh, visualize=True):
        """
        专门针对 alpha shape 提取边界环的鲁棒方法
        参数:
        vertices: 顶点数组 (n, 2) 或 (n, 3)
        triangles: 三角形面片 (m, 3)
        visualize: 是否可视化
        返回:
        boundary_vertices: 边界顶点索引列表
        boundary_loops: 边界环列表，每个环按顺时针排列
        """
        vertices = check_n3(np.asarray(mesh.vertices))
        triangles = check_n3(np.asarray(mesh.triangles))

        try:
            # 转换为 HalfEdgeTriangleMesh
            halfedge_mesh = o3d.geometry.HalfEdgeTriangleMesh.create_from_triangle_mesh(mesh)
            boundary_loops = halfedge_mesh.get_boundaries()
        except Exception as e:
            print(f"警告：HalfEdge 方法失败 - {str(e)}，尝试使用备用方法")
            boundary_loops = []

        # 如果没有边界环，尝试使用备用方法
        is_valid = True
        if not boundary_loops:
            try:
                boundary_loops, is_valid = AlphaShape.extract_boundary_with_edges_count(triangles)
            except Exception as e:
                print("警告：网格没有边界或无法提取边界")
                return []
        if not is_valid or not boundary_loops:
            print("警告：网格没有边界或无法提取边界")
            return [],[]

        # 步骤 4: 确保每个环按顺时针排列
        clockwise_loops = []
        for loop in boundary_loops:
            clockwise_loop = AlphaShape.ensure_clockwise(loop, vertices)
            clockwise_loops.append(clockwise_loop)

        # 步骤 5: 获取所有边界顶点
        boundary_vertices = set()
        for loop in clockwise_loops:
            boundary_vertices.update(loop)

        # 可视化
        if visualize and clockwise_loops:
            AlphaShape.visualize_boundary_loops(vertices, clockwise_loops)

        return list(boundary_vertices), clockwise_loops

    @staticmethod
    def extract_boundary_with_edges_count(triangles):
        edge_count = {}
        for tri in triangles:
            for i in range(3):
                v1, v2 = tri[i], tri[(i + 1) % 3]
                edge = tuple(sorted([v1, v2]))
                edge_count[edge] = edge_count.get(edge, 0) + 1

        # 步骤 2: 提取边界边（只出现一次的边）
        boundary_edges = [edge for edge, count in edge_count.items() if count == 1]

        if not boundary_edges:
            print("警告：alpha shape 没有边界")
            return [], []

        # 步骤 3: 构建边界环
        boundary_loops, is_valid = AlphaShape.build_boundary_loops(boundary_edges)

        return boundary_loops, is_valid

    @staticmethod
    def build_boundary_loops(boundary_edges):
        """
        从边界边构建完整的边界环
        参数:
        boundary_edges: 边界边列表，每条边为 (v1, v2)
        返回:
        boundary_loops: 边界环列表，每个环为顶点索引列表
        is_valid: 是否验证边界环是否有效。若is_valid为True，则检查每个环是否按顺时针排列。否则boundary_loops为空
        """
        if not boundary_edges:
            return [], False

        is_valid = True

        # 构建邻接表
        adjacency = {}
        for edge in boundary_edges:
            v1, v2 = edge
            if v1 not in adjacency:
                adjacency[v1] = []
            if v2 not in adjacency:
                adjacency[v2] = []
            adjacency[v1].append(v2)
            adjacency[v2].append(v1)

        # 检查是否有顶点的度数不为 2（说明有多个环或开口）
        for vertex, neighbors in adjacency.items():
            if len(neighbors) != 2:
                print(f"警告：顶点{vertex}的度数为{len(neighbors)}，可能存在多个环或开口")
                return [], False

        # 追踪边界环
        boundary_loops = []
        used_edges = set()

        for start_edge in boundary_edges:
            if start_edge in used_edges:
                continue

            # 从这条边开始构建环
            current_loop = [start_edge[0]]
            current_vertex = start_edge[0]
            prev_vertex = None

            while True:
                # 找到下一个顶点
                neighbors = adjacency[current_vertex]
                next_vertex = None

                for neighbor in neighbors:
                    edge = tuple(sorted([current_vertex, neighbor]))
                    if edge not in used_edges and neighbor != prev_vertex:
                        next_vertex = neighbor
                        used_edges.add(edge)
                        break

                if next_vertex is None:
                    break

                current_loop.append(next_vertex)
                prev_vertex = current_vertex
                current_vertex = next_vertex

                # 检查是否形成闭环
                if current_vertex == start_edge[0]:
                    break

            if len(current_loop) >= 3:
                boundary_loops.append(current_loop)

        return boundary_loops, True

    @staticmethod
    def ensure_clockwise(boundary_loop, vertices):
        """
        更鲁棒的方法确保边界环按顺时针排列
        使用多组点计算方向并投票决定
        参数:
        boundary_loop: 边界环的顶点索引列表
        vertices: 网格的所有顶点坐标
        返回:
        clockwise_loop: 顺时针排列的边界环
        """
        n = len(boundary_loop)
        if n < 3:
            return boundary_loop

        loop_vertices = vertices[boundary_loop]

        # 处理 2D 情况
        if loop_vertices.shape[1] == 2:
            # 对于 2D 点，使用多边形面积法判断方向
            area = 0.0
            for i in range(n):
                j = (i + 1) % n
                area += loop_vertices[i, 0] * loop_vertices[j, 1]
                area -= loop_vertices[j, 0] * loop_vertices[i, 1]

            # 面积为正表示逆时针，需要反转
            if area > 0:
                return boundary_loop[::-1]
            else:
                return boundary_loop
        else:
            # 3D 情况：投影到 2D 平面后判断
            # 计算边界环所在平面的法向量
            centroid = np.mean(loop_vertices, axis=0)
            centered_vertices = loop_vertices - centroid
            cov_matrix = np.cov(centered_vertices.T)
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
            normal = eigenvectors[:, np.argmin(eigenvalues)]

            # 确保法向量指向观察者（z 轴正方向）
            if normal[2] < 0:
                normal = -normal

            # 将 3D 点投影到 xy 平面
            projected_vertices = loop_vertices[:, :2]

            # 使用多边形面积法判断投影后的方向
            area = 0.0
            for i in range(n):
                j = (i + 1) % n
                area += projected_vertices[i, 0] * projected_vertices[j, 1]
                area -= projected_vertices[j, 0] * projected_vertices[i, 1]

            # 结合法向量方向判断
            # 如果法向量向上且面积为正（逆时针），则需要反转
            if normal[2] > 0 and area > 0:
                return boundary_loop[::-1]
            elif normal[2] < 0 and area < 0:
                return boundary_loop[::-1]
            else:
                return boundary_loop

    @staticmethod
    def visualize_boundary_loops(vertices, boundary_loops):
        """
        可视化边界环
        参数:
        vertices: 顶点数组
        boundary_loops: 边界环列表
        """
        try:
            import open3d as o3d
        except ImportError:
            print("Open3D 未安装，跳过可视化")
            return

        # 创建边界点的点云
        boundary_vertices_pcd = o3d.geometry.PointCloud()
        all_boundary_points = []
        for loop in boundary_loops:
            all_boundary_points.extend(vertices[loop])
        boundary_vertices_pcd.points = o3d.utility.Vector3dVector(np.array(all_boundary_points))
        boundary_vertices_pcd.paint_uniform_color([1, 0, 0])  # 红色

        # 创建边界环的线集
        line_sets = []
        colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1],
                  [1, 1, 0], [1, 0, 1], [0, 1, 1]]

        for i, loop in enumerate(boundary_loops):
            line_set = o3d.geometry.LineSet()
            loop_points = vertices[loop]

            # 如果是 2D 点，转换为 3D
            if loop_points.shape[1] == 2:
                loop_points = np.column_stack((loop_points, np.zeros(len(loop_points))))

            line_set.points = o3d.utility.Vector3dVector(loop_points)

            # 设置线
            lines = []
            for j in range(len(loop) - 1):
                lines.append([j, j + 1])
            lines.append([len(loop) - 1, 0])  # 闭合环
            line_set.lines = o3d.utility.Vector2iVector(lines)

            # 设置颜色
            color = colors[i % len(colors)]
            line_set.paint_uniform_color(color)
            line_sets.append(line_set)

        # 可视化
        o3d.visualization.draw_geometries(
            [boundary_vertices_pcd] + line_sets + [o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vertices))],
            window_name="Alpha Shape 边界环"
        )



class GTTEComputer(T.BaseTransform):
    """geometric tooth-tooth edge logits field computer"""
    def __init__(self,
                 alpha = 5,
                 k = 3,
                 max_iter = 5,
                 iter_thred_len = 0.01,
                 field_bound = 0.03,
                 field_sigma = 0.1,
                ):

        super().__init__()

        self.alpha = alpha                      # alpha长度
        self.k = k                              # knn-ica邻域
        self.max_iter = max_iter                # 最大迭代次数
        self.iter_thred_len = iter_thred_len    # 迭代阈值长度
        self.field_bound = field_bound          # 场边界
        self.field_sigma = field_sigma          # 场sigma系数

        # 试探法alpha参数
        self.alpha_step = 0.02  # 每次增量 0.02，介于 0.01~0.03 之间
        self.max_alpha = 0.25  # 最大 alpha 值，避免无限增大
        self.max_attempts = 15  # 最大尝试次数

    def __call__(self, V, F, tae_lf, tte_lf, bg_lf):

        timer = Timer(disp_info=True)

        # 1. 提取TG-edges, BT-edges, Tooth points这三类点
        tte_mask = np.squeeze(tte_lf) > 0  # tooth-tooth edge mask
        tae_mask = np.squeeze(tae_lf) > 0  # tooth-all edge mask
        bg_mask = np.squeeze(bg_lf) > 0  # tooth-gin edge mask
        edge_mask_init = np.array(bg_mask * tae_mask, dtype=np.float64) > 0  # filtered tooth-all edge mask

        # V,F满足n3
        V = check_n3(V)
        F = check_n3(F)

        # 2. 选择投影边缘点
        edge_points_2d = V[edge_mask_init, 0:2]
        # tte_points_2d = V[tte_mask, 0:2]

        # 3. 提取无边缘的牙齿点
        sep_teeth_mask = np.zeros_like(edge_mask_init, dtype=bool)  # sep-teeth mask；tae_mask，tg_mask,tte_mask的形状是相同的
        tooth_faces = get_filtered_faces(bg_mask, F)  # tooth faces
        tae_faces = get_filtered_faces(edge_mask_init, F)  # tae faces
        sep_teeth_faces = setdiff(tooth_faces, tae_faces)  # sep-teeth faces
        sep_teeth_mask[np.unique(sep_teeth_faces)] = True  # sep_teeth_mask=tooth_mask-tae_mask；实验表明根据面片进行提取的效果会更好些

        # 4. alphaShape构建2D网格形状
        timer.start()

        alpha_mesh, bnd_loops, bnd_verts, alpha_verts = (
            self.create_alpha_mesh(edge_points_2d, self.alpha, self.alpha_step, self.max_alpha,self.max_attempts))

        timer.end("4 create_alpha_mesh")

        # 如果所有尝试都失败，使用原始 alpha 并警告
        if alpha_mesh is None or not bnd_verts or not bnd_loops:
            outpus = {
                'gtte_lf': np.zeros_like(tte_lf),
                'alpha_mesh': None,
                'bnd_contour_3d': None,
                'countour_3d': None,
                'vector_lines': None,
                'is_success': False
            }
            return outpus

        bnd_contour = alpha_verts[bnd_loops[0], :]  # boundary_contour是顺时针分布的
        bnd_contour_3d = np.column_stack((bnd_contour, np.zeros(bnd_contour.shape[0])))
        bnd_contour_3d = gu.np_to_pcd(bnd_contour_3d, color=[0, 0, 0])

        # 6. 迭代优化轮廓，使得轮廓尽可能接近alpha_mesh的边界
        timer.start()
        countour  = self.KNN_ICA(V, bnd_contour, edge_mask_init)
        countour_3d = np.column_stack((countour, np.zeros(countour.shape[0])))
        countour_3d = gu.np_to_pcd(countour_3d, color=[1, 0, 0])
        timer.end("6 KNN_ICA")


        # 7. 划分为上下两段, 提取轮廓的两部分
        timer.start()
        segment_labels = label_contour_segment(countour)
        idx_0, idx_1 = (segment_labels == 0), (segment_labels == 1)
        part_pts0, part_pts1 = countour[idx_0], countour[idx_1]

        q1, q2 = self.compute_tte_lines(V, tte_mask, part_pts0, part_pts1)

        vector_lines = self.create_vector_lines(q1, q2)
        timer.end("7 compute_tte_lines")

        # gu.print_3d(alpha_mesh, bnd_contour_3d, countour_3d, vector_lines)


        # 9. 计算点到线段的距离并应用场效应
        timer.start()
        V_2d = V[:, :2]
        distances = point_to_segment_distance(V_2d, sep_teeth_mask, q1.squeeze(), q2.squeeze(),
                                              self.field_bound)

        # ========== 优化后的场效应计算 ==========
        # 1. 创建掩码（只保留有效距离）
        mask = (distances <= self.field_bound)
        # 2. 初始化场效应值
        gtte_lf = np.zeros(V.shape[0], dtype=np.float64)
        # 3. 向量化计算（避免 Python 循环）
        if np.any(mask):
            # 获取有效位置的索引
            row_idx = np.where(mask)[0]
            # 一次性计算所有有效距离的指数
            exp_values = np.exp(-distances[mask] / self.field_sigma)
            # 使用 np.add.at 按行累加
            np.add.at(gtte_lf, row_idx, exp_values)
        # 4. 标记无效点（距离都大于 field_bound 的点）
        min_dist = np.min(distances, axis=1)
        gtte_lf[min_dist > self.field_bound] = -10e3


        # 将大于field_bound的距离设置为1000
        # distances[distances > self.field_bound] = 10e3
        # # 计算场效应 (使用指数衰减模型)
        # min_dist = np.min(distances, axis=1)
        # gtte_lf = np.nansum(np.exp(-distances / self.field_sigma), axis=1)
        # gtte_lf[min_dist > self.field_bound] = -10e3

        timer.end("9 compute_distance")

        outpus= {
            'gtte_lf': gtte_lf,
            'alpha_mesh': alpha_mesh,
            'bnd_contour_3d': bnd_contour_3d,
            'countour_3d': countour_3d,
            'vector_lines': vector_lines,
            'is_success': True
        }

        return outpus

    def create_alpha_mesh(self, as_bnd_points, alpha, alpha_step, max_alpha, max_attempts):
        vertices, triangles = None, None
        alpha_mesh = None
        bnd_verts, bnd_loops = [], []
        for attempt in range(max_attempts):
            try:
                # 尝试构建 alpha shape
                vertices, triangles = AlphaShape.build_alpha_shape(as_bnd_points, alpha)
                alpha_mesh = AlphaShape.create_mesh_from_VF(vertices, triangles)
                bnd_verts, bnd_loops = AlphaShape.extract_boundary_from_alpha_shape(alpha_mesh, visualize=False)

                # 检查是否成功提取边界环
                if not bnd_verts or not bnd_loops or len(bnd_loops) == 0:
                    # print(f"⚠️ alpha={alpha:.3f}: 无法提取边界环，增加 alpha")
                    alpha += alpha_step
                    continue

                # 检查边界环是否有效（至少 3 个点）
                if len(bnd_loops[0]) < 3:
                    # print(f"⚠️ alpha={alpha:.3f}: 边界环点数过少 ({len(bnd_loops[0])})，增加 alpha")
                    alpha += alpha_step
                    continue

                if alpha > max_alpha:
                    # print(f"⚠️ alpha={alpha:.3f}: 尝试次数过多，使用初始值 {self.alpha:.3f}")
                    break

                # 成功！打印信息并退出循环
                # print(f"✅ alpha 调整成功：{alpha:.3f} (初始={self.alpha:.3f}, 尝试次数={attempt + 1})")
                # print(f"   顶点数：{len(vertices)}, 三角形数：{len(triangles)}, 边界环数：{len(bnd_loops)}")
                break

            except Exception as e:
                # print(f"⚠️ alpha={alpha:.3f}: 构建失败 - {str(e)}，增加 alpha")
                alpha += alpha_step
                continue
        return alpha_mesh, bnd_loops, bnd_verts, vertices

    def compute_tte_lines(self, V,tte_mask, part0, part1):
        """
        计算 TTE 线段，并根据公共端点进行聚类分组
        参数:
            V: 顶点坐标 [N, 3]
            tte_mask: TTE 点掩码 [N,]
            part0: 轮廓第一部分 [K0, 2]
            part1: 轮廓第二部分 [K1, 2]
        返回:
            q1_extended: 延长后的 q1 点 [L, 2]
            q2_extended: 延长后的 q2 点 [L, 2]
            line_clusters: 线段聚类字典，key 为公共端点索引，value 为该端点相关的所有线段索引
        """

        # 1. 获取 TTE 点的 2D 坐标
        tte_pts_2d = V[tte_mask, 0:2]

        # 2. 为每个 TTE 点找到在两部分轮廓上的最近邻点
        tree0, tree1 = KDTree(part0), KDTree(part1)
        _, q1_idx = tree0.query(tte_pts_2d, k=1)
        _, q2_idx = tree1.query(tte_pts_2d, k=1)

        # 3. 去重，获取唯一的线段对
        q_pair = np.column_stack((q1_idx, q2_idx))
        q_pair_unique, unique_indices = np.unique(q_pair, axis=0, return_index=True)

        # print(f'原始 TTE 点数：{len(tte_pts_2d)}')
        # print(f'去重后线段数：{len(q_pair_unique)}')

        # 4. 根据公共端点进行聚类
        line_clusters = defaultdict(list)

        for i, (q1_i, q2_i) in enumerate(q_pair_unique):
            line_clusters[f'p0_{q1_i}'].append(i) # 使用 q1_idx 作为 key（part0 上的端点）
            line_clusters[f'p1_{q2_i}'].append(i) # 使用 q2_idx 作为 key（part1 上的端点）

        # 5. 合并有公共端点的线段组
        # 使用并查集来合并有公共端点的线段
        parent = list(range(len(q_pair_unique)))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # 遍历所有线段，连接有公共端点的线段
        endpoint_to_lines = defaultdict(list)
        for i, (q1_i, q2_i) in enumerate(q_pair_unique):
            endpoint_to_lines[q1_i].append(i)  # part0 的端点
            endpoint_to_lines[q2_i + len(part0)].append(i)  # part1 的端点（偏移避免冲突）

        # 对有公共端点的线段进行合并
        for endpoint, line_indices in endpoint_to_lines.items():
            for i in range(1, len(line_indices)):
                union(line_indices[0], line_indices[i])

        # 6. 收集最终的聚类
        final_clusters = defaultdict(list)
        for i in range(len(q_pair_unique)):
            root = find(i)
            final_clusters[root].append(i)

        # 7. 过滤掉只有一个线段的聚类（可选）
        significant_clusters = {k: v for k, v in final_clusters.items() if len(v) > 0}

        # print(f'\n线段聚类统计:')
        # print(f'  总线段数：{len(q_pair_unique)}')
        # print(f'  聚类数量（包含公共端点）：{len(final_clusters)}')
        # print(f'  显著聚类数量（线段数>1）：{len(significant_clusters)}')

        if len(significant_clusters) > 0:
            cluster_sizes = [len(v) for v in significant_clusters.values()]
            # print(f'  平均每个聚类的线段数：{np.mean(cluster_sizes):.2f}')
            # print(f'  最大聚类大小：{max(cluster_sizes)}')
            # print(f'  最小聚类大小：{min(cluster_sizes)}')

        # 8. 获取唯一的 q1, q2
        q1_idx_unique = q_pair_unique[:, 0]
        q2_idx_unique = q_pair_unique[:, 1]
        q1 = part0[q1_idx_unique]
        q2 = part1[q2_idx_unique]

        # 9. 对 significant_clusters 中的同类顶点取平均
        # 为每个聚类计算平均的 q1 和 q2 点
        averaged_q1 = []
        averaged_q2 = []
        cluster_line_map = []  # 记录每个平均点对应的原始线段索引

        for cluster_id, line_indices in significant_clusters.items():
            # 获取该聚类中所有线段的 q1 和 q2 索引
            cluster_q1_idx = q1_idx_unique[line_indices]
            cluster_q2_idx = q2_idx_unique[line_indices]

            # 获取对应的坐标
            cluster_q1_coords = part0[cluster_q1_idx]
            cluster_q2_coords = part1[cluster_q2_idx]

            # 计算平均坐标
            avg_q1 = np.mean(cluster_q1_coords, axis=0, keepdims=True)
            avg_q2 = np.mean(cluster_q2_coords, axis=0, keepdims=True)

            # 保存平均后的点
            averaged_q1.append(avg_q1)
            averaged_q2.append(avg_q2)
            cluster_line_map.append(line_indices)

            # print(f"聚类 {cluster_id}: {len(line_indices)} 条线段 -> "
            #       f"平均 q1={avg_q1.flatten()}, 平均 q2={avg_q2.flatten()}")

        # 转换为 numpy 数组
        if len(averaged_q1) > 0:
            averaged_q1 = np.vstack(averaged_q1)  # [num_clusters, 2]
            averaged_q2 = np.vstack(averaged_q2)  # [num_clusters, 2]
        else:
            averaged_q1 = deepcopy(q1)
            averaged_q2 = deepcopy(q2)

        # print(f'\n平均后聚类数量：{len(averaged_q1)}')

        # 11. 获取线段向量
        tree0, tree1 = KDTree(averaged_q1), KDTree(averaged_q2)
        _, q1_idx = tree0.query(tte_pts_2d, k=1)
        _, q2_idx = tree1.query(tte_pts_2d, k=1)
        # q_pair = np.column_stack((q1_idx, q2_idx))
        # q_pair_unique = np.unique(q_pair, axis=0)
        # q1_idx, q2_idx = q_pair_unique[:, 0], q_pair_unique[:, 1]
        _, q12_idx = tree0.query(averaged_q2, k=1)  # 查询averaged_q2的最近邻点在averaged_q1中的索引


        # 12 过滤掉夹角小于90度的线段
        vec1  = tte_pts_2d - averaged_q1[q1_idx]
        vec2  = tte_pts_2d - averaged_q2[q2_idx]

        # 单位化向量
        # vec1 = vec1 / (np.linalg.norm(vec1, axis=1, keepdims=True) + 1e-8)
        # vec2 = vec2 / (np.linalg.norm(vec2, axis=1, keepdims=True) + 1e-8)
        # val_mask = np.sum(vec1 * vec2, axis=1) < np.cos(np.radians(105))  # cos(105°) = -0.5
        eq_mask = (q1_idx == q2_idx)
        val_mask = np.sum(vec1 * vec2, axis=1) < 0
        val_mask = eq_mask * val_mask

        q1_idx = q1_idx[val_mask]
        q2_idx = q2_idx[val_mask]
        tte_pts_2d = tte_pts_2d[val_mask]

        q1 = np.vstack([tte_pts_2d, tte_pts_2d])
        q2 = np.vstack([averaged_q1[q1_idx], averaged_q2[q2_idx]])

        # 13. 延长线段（如果需要）
        segment_vector = q2 - q1
        segment_length = np.linalg.norm(segment_vector, axis=1, keepdims=True)
        unit_direction = segment_vector / (segment_length + 1e-8)
        # extend_distance = min(segment_length * 2, 0.05)
        extend_distance = np.minimum(segment_length * 2, 0.05)

        q1_extended = q1
        q2_extended = q2 + unit_direction * extend_distance

        # print(f'\n原线段平均长度：{np.mean(segment_length):.4f}')
        # print(f'延长后总平均长度：{np.mean(np.linalg.norm(q2_extended - q1_extended, axis=1)):.4f}')

        return q1_extended, q2_extended

    def create_vector_lines(self, q1, q2):
        # 创建向量线段集合（用于可视化）
        vector_lines = []
        for i in range(len(q1)):
            # 每个 TTE 点对应一条从 q1[i] 到 q2[i] 的线段
            line_start = q1[i]  # [2]
            line_end = q2[i]  # [2]

            # 扩展到 3D（z=0）
            line_start_3d = np.array([line_start[0], line_start[1], 0.0])
            line_end_3d = np.array([line_end[0], line_end[1], 0.0])

            vector_lines.append([line_start_3d, line_end_3d])
        # 转换为 numpy 数组 [N_lines, 2, 3]
        vector_lines = np.array(vector_lines)


        # print(f"  向量线段数：{len(vector_lines)}")
        # 可视化方案 1：使用 Open3D 绘制线段
        vector_line_set = o3d.geometry.LineSet()
        # 顶点：所有线段的起点和终点
        line_points = vector_lines.reshape(-1, 3)  # [N_lines*2, 3]
        # 边：每两个点构成一条线段
        line_indices = [[i * 2, i * 2 + 1] for i in range(len(vector_lines))]
        vector_line_set.points = o3d.utility.Vector3dVector(line_points)
        vector_line_set.lines = o3d.utility.Vector2iVector(line_indices)
        # 设置颜色（使用渐变或固定颜色）
        # 方案 A：固定颜色（蓝色）
        vector_line_set.paint_uniform_color([0, 0, 1])

        return vector_line_set

    def KNN_ICA(self, V, bnd_contour, tae_mask):

        contour0 = bnd_contour.copy()
        avg_len = 10e3  # 初始平均边长设置为10000
        for i in range(self.max_iter):
            if avg_len < self.iter_thred_len:
                # print(f"迭代 {i + 1}: 平均边长 {avg_len:.2f} 小于阈值 {self.iter_thred_len}，停止迭代")
                break

            contour = contour0.copy()
            contour2, org_flag, avg_len = interp_contour(contour)  # 插值
            contour2 = update_interp_points(contour2, np.zeros_like(org_flag, dtype=bool), V, tae_mask)  # 位置更新
            contour0 = contour2
        return contour0


class SegPredictor(T.BaseTransform):
    """
    SegPredictor
    神经网络预测器。本质就是一个plmodule。要求有一个predict_step_forward的方法，并返回outputs字典。
    返回：outputs字典，包括预测结果列表；这种返回格式具有灵活性，但对于我们这个项目最好要统一outputs的输出。
    例如，对于DTSegNet，其返回的outputs字典包括：
    'seg_result':  # BNC
    'tg_seg_result':  # BNC
    'edge_seg_result':   # B2N
    'tte_seg_result':   # B2N
    'cbl_inputs'  # PXO [pos, feat, offset]
    """

    def __init__(self,
                 num_classes=15,                                # 类别数
                 predictor_type='DTSegNet',                     # 预测器类型
                 ckpt_path='../../../common/DTSegNet.ckpt',     # 模型检查点路径
                 net_para_name='seg_base',                      # 网络参数名称
                 to_cpu = True      # 是否将数据移动到CPU, 因为通常这个步骤处于预处理阶段，因而最好将数据移到CPU上
                ):
        # 调用父类初始化（完全仿照 FPSSample）
        super().__init__()

        # 类别数
        self.num_classes = num_classes

        # 预测器相关参数。网络结构参数及预测器对象
        self.predictor_type = predictor_type
        self.ckpt_path = ckpt_path
        self.net_para_name = net_para_name

        # 设备相关
        self.to_cpu = to_cpu

        self._predictor = None    # 延迟加载预测器
    @property
    def predictor(self):
        """懒加载预测器"""
        if self._predictor is None:
            try:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self._predictor = PredictorFactory.get_predictor(
                    predictor_type=self.predictor_type,
                    ckpt_path=self.ckpt_path,
                    net_para_name=self.net_para_name,
                    device=device
                )
            except Exception as e:
                print(f"❌ 预测器加载失败：{str(e)}")
                raise RuntimeError(f"BAFPSSample 无法加载预测器：{str(e)}") from e

        return self._predictor

    def set_predictor(self, predictor):
        """外部手动设置预测器"""
        self._predictor = predictor

    def __call__(self, data):
        """
        Args:
            data: PyG Data 对象
        Returns:
            data: 处理后的 Data 对象（增加一个outputs对象）
        """
        #--------------- 1. 预测fps16000的的 elf-----------------------------#
        batch = Batch.from_data_list([data])
        with torch.no_grad():
            pred_data = self.predictor.predict_step_forward(batch)
            pred_data = pred_data.cpu() if self.to_cpu else pred_data    # 转为cpu，因为通常这个步骤处于预处理阶段，因而最好将数据移到CPU上

        return pred_data

class Timer:
    def __init__(self, disp_info=True):
        self.start_time = time.time()
        self.disp_info = disp_info

        self.disp_info = False
    def start(self):
        self.start_time = time.time()

    def end(self, tag):
        time_end = time.time()
        print(f"{tag} 耗时：{time_end - self.start_time:.2f}s") if self.disp_info else None

class TopoMorphCluster(T.BaseTransform):
    """
    Topology morph cluster (TMC)
    拓扑形态学聚类。
    主要流程：
    1. 通过神经网络提取FPSSampling下的ELFs
    2. 将FPSSampling的ELFs映射到QSlimSampling的ELFs
    3. 提取Edge Mask（存在多种算法，需要进行消融实验）
    4. 去除Edge Mask获得连通分量
    5. 连通分量过滤
    6. 拓扑连通可视化及评估
    返回：proposal字典，包括预测质心列表，顶点聚类标签，质心数量等；
    """

    def __init__(self,
                 num_classes=15,  # 类别数
                 use_outputs_data = False,     # 使用batch内置的outputs数据
                 seg_predictor=None,         # 直接给定预测器
                 predictor_type='DTSegNet',  # 预测器类型
                 ckpt_path='../../../common/DTSegNet.ckpt',  # 模型检查点路径
                 net_para_name='seg_base',  # 网络参数名称
                 mounted_segnet_outputs = True,  # 是否挂载SegNet输出，理由包含了丰富的信息
                 mounted_mask_outputs = True,  # 是否挂载Mask相关输出，包含各种掩码
                 mounted_knn_ica_meshes=True,  # 是否挂载可视化KNN-ICA结果
                 save_lf_path = None,
                 filter_type='tae_tte',  # 过滤类型，包含bg、tae、tae_tte、tae_tte_gtte、tae_tte_ptte、tae_tte_ptae
                 aggr_type='boolean',  # 聚合类型。默认为布尔聚合;包含boolean, add两种方式
                 small_cluster_size_thre = 10,  # 过滤小聚类大小的阈值
                 alpha = 0.12,  # alpha边长阈值
                 k = 3,  # KNN-ICA迭代邻域数
                 max_iter = 5,  # KNN-ICA最大迭代次数
                 iter_thred_len_ratio = 0.1,  # KNN-ICA迭代阈值长度比率
                 field_bound_ratio = 3,  # UL线段带宽比率
                 field_sigma_ratio = 10,  # 距离带宽比率
                 ):
        # 调用父类初始化（完全仿照 FPSSample）
        super().__init__()

        # 类别数
        self.num_classes = num_classes
        self.use_outputs_data = use_outputs_data     # 使用batch内置的outputs数据

        # 预测器相关参数。网络结构参数及预测器对象
        self.predictor_type = predictor_type
        self.ckpt_path = ckpt_path
        self.net_para_name = net_para_name
        self.mounted_segnet_outputs = mounted_segnet_outputs
        self.mounted_mask_outputs = mounted_mask_outputs
        self.save_lf_path = save_lf_path

        # 优先使用内置的数据；不行的话再载入分割预测器，给定两种载入方法，优先使用给定的预测器，否则通过ckpt加载预测器对象
        if not use_outputs_data:
            if seg_predictor is not None:
                self._seg_predictor = SegPredictor()
                self._seg_predictor.set_predictor(seg_predictor)
            else:
                self._seg_predictor = SegPredictor(
                     num_classes=num_classes,
                     predictor_type=predictor_type,
                     ckpt_path=ckpt_path,
                     net_para_name=net_para_name,
                     to_cpu = True      # 是否将数据移动到CPU, 因为通常这个步骤处于预处理阶段，因而最好将数据移到CPU上
            )

        # 拓扑聚合参数
        self.filter_type = filter_type
        self.aggr_type = aggr_type
        self.small_cluster_size_thre = small_cluster_size_thre

        # KNN-ICA参数
        self.alpha = alpha
        self.max_iter = max_iter
        self.iter_thred_len_ratio = iter_thred_len_ratio
        self.field_bound_ratio = field_bound_ratio
        self.field_sigma_ratio = field_sigma_ratio
        self.k = k
        self.knn_ica_meshes = None

        self.mounted_knn_ica_meshes = mounted_knn_ica_meshes


    def check_inputs(self, data, timer):
        if not hasattr(data, 'data_qslim'):
            raise ValueError("data 中必须包含 data_qslim 字段")
        # --------------- 1. 获取预测结果（优先使用已有outputs，否则通过神经网络预测）-----------------------------#
        # 检查是否已有预测结果
        has_existing_outputs = (
                hasattr(data, 'outputs') and
                data.outputs is not None and
                'edge_seg_result' in data.outputs and
                'tte_seg_result' in data.outputs
        )
        # print(has_existing_outputs)

        if has_existing_outputs:
            if 'tg_seg_result' not in data.outputs:
                seg_logits = data.outputs['seg_result']  # [B, N, 15]

                bg_logits = seg_logits[:, :, :1]  # [B, N, 1]# 1. 提取背景类的 Logits (索引 0)
                fg_logits = seg_logits[:, :, 1:]  # [B, N, 14] # 2. 提取所有前景类的 Logits (索引 1 到 14)
                max_fg_logits, _ = torch.max(fg_logits, dim=-1, keepdim=True)  # [B, N, 1]
                bg_lf = torch.cat([bg_logits, max_fg_logits], dim=-1)  # [B, N, 2]

                data.outputs['tg_seg_result'] = bg_lf.permute(0, 2, 1)


            pred_data = data
        else:
            # 通过神经网络预测
            timer.start()
            pred_data = self._seg_predictor(data)
            timer.end("1 seg_predictor")
        return pred_data


    def __call__(self,data: Data,) -> Data:
        """
        Args:
            data: PyG Data 对象
        Returns:
            data: 处理后的 Data 对象（增加一个proposal对象）
        """
        #--------------- 0. 检查必要字段 -----------------------------#
        timer = Timer(disp_info=True)

        pred_data = self.check_inputs(data, timer)

        pred_data = pred_data.to(data.pos.device)

        #---------------- 2. 提取data_qslilm采样结构并构建mesh--------------------------#
        # pos = pred_data.pos
        data_qslim = data.data_qslim
        data_qslim = data_qslim[0] if isinstance(data_qslim, list) else data_qslim
        # mesh = create_mesh_from_pygdata(data_qslim)
        # gu.print_3d(gu.get_colored_mesh(mesh, data_qslim.y.squeeze().cpu().numpy()))  # color is random

        #-----------------3. 提取elfs，并构建fps点云与qslim点云之间的映射关系--------------#

        timer.start()
        (bg_mask_ori, tae_mask_ori, tte_mask_ori, edge_mask_ori, ptae_mask_ori,
                ptte_mask_ori, tae_lf_ori, tte_lf_ori, bg_lf_ori) = self.get_elfs(pred_data)
        timer.end("2 get_elfs")

        timer.start()
        f2q_idx, _ = NNPointSamplingIndexer()(data.pos, data_qslim.pos)
        f2q_idx = torch.as_tensor(f2q_idx).long().view(-1)
        timer.end("3 NNPointSamplingIndexer")

        bg_mask, tae_mask, tte_mask, ptae_mask, ptte_mask, tae_lf, tte_lf, bg_lf = (
            bg_mask_ori[f2q_idx], tae_mask_ori[f2q_idx], \
                tte_mask_ori[f2q_idx], ptae_mask_ori[f2q_idx], ptte_mask_ori[f2q_idx], \
                tae_lf_ori[:,:,f2q_idx], tte_lf_ori[:,:,f2q_idx], bg_lf_ori[:,:,f2q_idx])
        # gu.print_3d(gu.get_colored_mesh(mesh, tae_flag_q.long().squeeze().cpu().numpy()))  # color is random
        # gu.print_3d(gu.get_colored_mesh(mesh, tte_flag_q.long().squeeze().cpu().numpy()))  # color is random
        # gu.print_3d(gu.get_colored_mesh(mesh, bg_flag_q.long().squeeze().cpu().numpy()))  # color is random

        # ---------------------------4. 获取边缘点（key step）-----------------------#
        elfs={'bg_mask': bg_mask,'tae_mask': tae_mask,'tte_mask': tte_mask,
              'ptae_mask': ptae_mask,'ptte_mask': ptte_mask,
              'tae_lf': tae_lf,'tte_lf': tte_lf,'bg_lf': bg_lf}

        # timer.start()
        outs = self.compute_edge_mask(filter_type=self.filter_type,
                                           aggr_type=self.aggr_type,
                                           elfs=elfs,
                                           data_qslim = data_qslim)
        edge_mask = outs['edge_mask'].squeeze()
        gtte_lf = outs['gtte_lf'] if 'gtte_lf' in outs else tte_lf[:,1,:]

        elfs.update({'gtte_lf': gtte_lf})
        # timer.end("4 compute_edge_mask"

        # if self.save_lf_path is not None:
        #     if not exists(self.save_lf_path):
        #         os.makedirs(self.save_lf_path)
        #     save_path = os.path.join(self.save_lf_path, f"{data.filename}.png")
        #     self.save_logits_fields(data_qslim, elfs, save_path)

        # ----------------------------5. TMC聚类（key step）-------------------------#
        V = data_qslim.pos.cpu().numpy()
        F = data_qslim.face.cpu().numpy()
        V,F = check_n3(V), check_n3(F)
        # 查看边缘信息
        # gu.print_3d(gu.get_colored_mesh(V, edge_mask.long().squeeze().cpu().numpy()))

        # ----------------------------6. 聚类-----------------------------------------#
        timer.start()

        try:
            verts_clu_labels = self.perform_topo_cluster(V, F, bg_mask.cpu().numpy(), edge_mask.cpu().numpy())  # 每个顶点的聚类标签
        except Exception as e:
            print(f"❌ 聚类失败：{str(e)}")
            # raise RuntimeError(f"BAFPSSample 聚类失败：{str(e)}") from e
            verts_clu_labels = np.zeros_like(bg_mask.cpu().numpy(), dtype=np.int64)
            verts_clu_labels[bg_mask.cpu().numpy()] = 1

        timer.end("5 perform_topo_cluster")
        # ----------------------------6. 可视化-----------------------------------------#
        # self.plot_meshes(V, F, bg_mask, edge_mask, tae_mask, tte_mask, verts_clu_labels)

        # ----------------------------7. 构建proposal-----------------------------------------#
        instances = torch.from_numpy(verts_clu_labels)
        # 获取聚类中心, 每个聚类点的数量, 聚类中心的类标, 聚类数量(包括背景点0)
        cluster_centroids = scatter_mean(src=data_qslim.pos,index=instances,dim=0,)
        per_cluster_count = scatter_sum(src=torch.ones_like(instances),index=instances,dim=0,)
        cluster_labels = scatter_max(src=instances,index=instances,dim=0,)[0]
        cluster_num = cluster_centroids.shape[0]


        # print('cluster_centroids:', cluster_centroids.shape,
        #       'cluster_labels:', cluster_labels,
        #       'per_cluster_count:', per_cluster_count,
        #       'cluster_num:', cluster_num)

        valid_ids, reindexed = self.filter_small_cluster(cluster_labels, per_cluster_count, self.small_cluster_size_thre)

        cluster_labels = reindexed[valid_ids]
        cluster_centroids = cluster_centroids[valid_ids]
        per_cluster_count = per_cluster_count[valid_ids]
        cluster_num = cluster_centroids.shape[0]

        # print("After filter small cluster:")
        # print('cluster_centroids:', cluster_centroids.shape,
        #       'cluster_labels:', cluster_labels,
        #       'per_cluster_count:', per_cluster_count,
        #       'cluster_num:', cluster_num)

        # 创建 proposal 配置对象
        tmc_proposal = {
            'cluster_centroids': cluster_centroids,  # 聚类中心
            'cluster_labels': cluster_labels,        # 聚类标签
            'per_cluster_count': per_cluster_count,  # 每个聚类的点数量
            'cluster_num': cluster_num,              # 总的聚类数量，包括背景点0
        }

        if self.mounted_mask_outputs:
            tmc_proposal['bg_mask'] = bg_mask                      # qslim的背景点mask，保留这些主要用以绘制
            tmc_proposal['tae_mask'] = tae_mask                    # tae点mask
            tmc_proposal['tte_mask'] = tte_mask                    # tte点mask
            tmc_proposal['edge_mask'] = edge_mask                  # 边缘点mask
            tmc_proposal['gtte_lf'] = gtte_lf                      # gtte_lf
            tmc_proposal['tae_lf'] = tae_lf                        # tae_lf
            tmc_proposal['tte_lf'] = tte_lf                        # tte_lf
            tmc_proposal['bg_lf'] = bg_lf                          # bg_lf
            tmc_proposal['verts_clu_labels'] = verts_clu_labels    # qslim点云的聚类标签


        # 是否挂载KNN ICA 输出
        if self.mounted_knn_ica_meshes and self.knn_ica_meshes is not None and len(self.knn_ica_meshes) == 4:
            tmc_proposal['knn_ica_meshes'] = self.knn_ica_meshes
        else:
            tmc_proposal['knn_ica_meshes'] = None

        # 是否挂载SegNet输出
        tmc_proposal['outputs'] = pred_data.outputs if self.mounted_segnet_outputs else None

        # 将 proposal 添加到 data 对象
        data.tmc_proposal = tmc_proposal


        return data


    def filter_small_cluster(self, cluster_labels, per_cluster_count, min_size=10):
        # 过滤掉数量小于min_size的聚类,并重新调整索引，使之从0开始连续编号
        # 步骤 1: 找出保留的聚类
        valid_ids = np.where(per_cluster_count >= min_size)[0]
        # print(f"\n保留的聚类 ID: {valid_ids}")  # [3, 5]
        # 步骤 2: 过滤点
        mask = np.isin(cluster_labels, valid_ids)
        filtered = np.zeros_like(cluster_labels)
        filtered[mask] = cluster_labels[mask]
        # print(f"过滤后标签：{filtered}")  # [0 0 0 0 0 0 3 3 3 0 0 5 5 5 5 5]
        # 步骤 3: 使用 unique 重标号
        unique_labels = np.unique(filtered)  # [0, 3, 5]
        # print(f"唯一标签：{unique_labels}")
        # 步骤 4: 映射到新标签
        reindexed = np.searchsorted(unique_labels, filtered)
        # print(f"重标号后：{reindexed}")  # [0 0 0 0 0 0 1 1 1 0 0 2 2 2 2 2]
        # print(f"新标签分布：{np.bincount(reindexed)}")  # [7 3 5]

        return valid_ids, reindexed

    def compute_gt_clu_counts_mat(self, gt_labels, clu_labels, cluster_num):
        """
        计算gt-cluster的相关数据。 这个本质就是一个匹配问题；可以考虑用匈牙利算法来解决。
        gt_clu_counts_mat的计算规则：
        1. 仅统计非牙齿点的聚类。
        2. 非牙齿点的聚类，其所有点clu均映射为gt为0。这么做就是把所有非牙齿点的聚类，都映射为gt为0，
        对于边缘点而言，这么做可能不合理，但是也是为了后续处理非方便。
        1.
        Args:
            cluster_num: 聚类数量，包括非牙齿点0
            gt_labels: 每个点的真实类别标签
            clu_labels: 每个点的聚类标，包括非牙齿点0
        Returns:
            gt_clu_counts_mat： 每个gt-cluster的数量矩阵，大小为(num_classes, cluster_num)
        """

        # 1. 获取有效的聚类标签和gt标签
        valid_cluster_mask = (clu_labels > 0)   # 仅统计非牙齿点
        valid_cluster_labels = clu_labels[valid_cluster_mask] # 有效的聚类标签
        valid_gt_labels = gt_labels[valid_cluster_mask] # 有效的gt标签
        row_indices = valid_gt_labels.squeeze()
        col_indices = valid_cluster_labels.squeeze()

        # 2. 确保聚类个数的一致性
        # 2. 确保聚类个数的一致性（全 PyTorch 版本）
        gt_clu_counts_mat = torch.zeros((self.num_classes, cluster_num))
        gt_clu_counts_mat.index_put_((row_indices, col_indices),
                                      torch.ones_like(row_indices, dtype=torch.float),
                                      accumulate=True)
        gt_clu_counts_mat[0, 0] = torch.sum(clu_labels == 0)

        return gt_clu_counts_mat

    def perform_topo_cluster(self, V, F, bg_mask, edge_mask):
        """
        对牙齿拓扑结构进行聚类。这是一个numpy实现。要求输入输出都为numpy数组。
        """
        # numpy格式检查
        assert (isinstance(V, np.ndarray) and isinstance(F, np.ndarray)
                and isinstance(bg_mask, np.ndarray) and isinstance(edge_mask, np.ndarray))

        only_bg_flag = True if np.all(bg_mask==edge_mask) else False

        # n3格式检查
        V, F = check_n3(V), check_n3(F)

        # 提取无边缘的牙齿点
        sep_teeth_mask = np.zeros_like(bg_mask, dtype=bool)  # sep-teeth mask；tae_mask，tg_mask,tte_mask的形状是相同的
        tooth_faces = get_filtered_faces(bg_mask, F)  # tooth faces

        # support only background_mask
        if only_bg_flag:
            sep_teeth_faces = tooth_faces
        else:
            edge_faces = get_filtered_faces(edge_mask.squeeze(), F)  # tae faces
            sep_teeth_faces = setdiff(tooth_faces, edge_faces)  # sep-teeth faces

        sep_teeth_mask[np.unique(sep_teeth_faces)] = True  # sep_teeth_mask=tooth_mask-tae_mask；实验表明根据面片进行提取的效果会更好些
        # 构建牙齿拓扑结构
        tooth_mesh = o3d.geometry.TriangleMesh()
        tooth_mesh.vertices = o3d.utility.Vector3dVector(V)
        tooth_mesh.triangles = o3d.utility.Vector3iVector(sep_teeth_faces)

        # 调用三角形聚类连通量函数求取连通分量。其值从0开始计数。
        triangle_clusters, cluster_n_triangles, cluster_area = tooth_mesh.cluster_connected_triangles()

        triangle_clusters = np.asarray(triangle_clusters)
        # print(triangle_clusters.min(), triangle_clusters.max())

        # 投票法获得顶点聚类标签
        row_indices = sep_teeth_faces.T.flatten(order='C')  # M*3, [F,3]->[3,F]->3F, C为行优先
        col_indices = np.tile(triangle_clusters, 3)  # M*3,
        n_vertices = V.shape[0]
        n_clusters = len(cluster_n_triangles)
        vote_mat = np.zeros((n_vertices, n_clusters), dtype=np.int32)
        np.add.at(vote_mat, (row_indices, col_indices), 1)      # 相应位置累加1
        verts_labels = np.asarray(vote_mat.argmax(axis=1)).flatten() + 1 # 最佳聚类标签
        verts_labels[~sep_teeth_mask] = 0                                # 非牙齿的顶点聚类标签为0
        return verts_labels


    def compute_edge_mask(self, filter_type='tae', aggr_type='boolean', elfs = None, data_qslim=None):
        """
        获取待过滤的顶点
        Args:
            filter_type:
            logits_type:
            elfs:
            pos:
        Returns:
        """
        bg_mask = elfs['bg_mask']
        tae_mask = elfs['tae_mask']
        tte_mask = elfs['tte_mask']
        ptae_mask = elfs['ptae_mask']
        ptte_mask = elfs['ptte_mask']
        tae_lf = elfs['tae_lf']
        tte_lf = elfs['tte_lf']

        outs = {}

        if filter_type not in ['bg', 'tae', 'tae_tte', 'tae_tte_gtte', 'tae_tte_ptae', 'tae_tte_gtte_ptte']:
            raise ValueError(f"Invalid filter type: {filter_type}")

        if aggr_type not in ['boolean','add']:
            raise ValueError(f"Invalid aggr type: {aggr_type}")

        if filter_type == 'bg':
            outs['edge_mask'] = (bg_mask) > 0
            return outs
        elif filter_type == 'tae':
            outs['edge_mask'] = (tae_mask * bg_mask) > 0
            return outs
        elif filter_type == 'tae_tte':
            # if aggr_type == 'boolean':
            outs['edge_mask'] = (tae_mask + tte_mask * bg_mask) > 0
            return outs
            # elif aggr_type == 'add':
            #     tmp_lf = tae_lf + tte_lf
            #     tmp_mask = ((torch.argmax(tmp_lf, dim=1) > 0) + (tmp_lf[:, 1, :] > 0)) > 0
            #     return tmp_mask
        elif filter_type == 'tae_tte_ptae':
            if aggr_type == 'boolean':
                outs['edge_mask'] = (tae_mask + tte_mask + ptae_mask) > 0
                return outs
            elif aggr_type == 'add':
                tmp_lf = tae_lf + tte_lf
                tmp_mask = ((torch.argmax(tmp_lf, dim=1) > 0) + (tmp_lf[:, 1, :] > 0) + (ptae_mask > 0)) > 0
                outs['edge_mask'] = tmp_mask
                return outs

        elif filter_type == 'tae_tte_gtte':
            try:
                knn_avg_dist = compute_knn_average_distance(data_qslim.pos, self.k).cpu().numpy()
                # print("knn_avg_dist", knn_avg_dist)

                evaluator = GTTEComputer(
                    alpha = self.alpha,
                    max_iter=self.max_iter,
                    iter_thred_len=knn_avg_dist * self.iter_thred_len_ratio,
                    field_bound=knn_avg_dist * self.field_bound_ratio,
                    field_sigma=knn_avg_dist * self.field_sigma_ratio,
                )

                V = data_qslim.pos.cpu().numpy()
                F = data_qslim.face.cpu().numpy()
                output = evaluator(V, F, tae_mask.cpu().numpy(), tte_mask.cpu().numpy(), bg_mask.cpu().numpy())
                gtte_lf = output['gtte_lf']

                self.knn_ica_meshes = [output['alpha_mesh'],  output['bnd_contour_3d'], \
                    output['countour_3d'], output['vector_lines'] ]

                if aggr_type == 'boolean':
                    gtte_mask = (gtte_lf > 0)
                    outs['edge_mask'] = ((tae_mask + tte_mask + gtte_mask ) * bg_mask) > 0
                    outs['gtte_lf'] = gtte_lf
                    return outs
                elif aggr_type == 'add':
                    tte_lf[:,1,:] = tte_lf[:,1,:] + 10*gtte_lf
                    tte_lf[:, 0, :] = tte_lf[:, 0, :] - 10*gtte_lf
                    tmp_lf = tae_lf + tte_lf
                    tmp_tte_mask = ((torch.argmax(tte_lf, dim=1) > 0) + (tte_lf[:, 1, :] > 0)) > 0
                    tmp_mask = ((tae_mask + tmp_tte_mask) * bg_mask) > 0
                    outs['edge_mask'] = tmp_mask
                    outs['gtte_lf'] = gtte_lf
                    return outs

            except Exception as e:
                print(f"An error occurred: {e}")
                self.knn_ica_meshes =  None
                outs['edge_mask'] = (tae_mask + tte_mask)
                return outs

        else:
            raise ValueError(f'Invalid filter_type: {filter_type}')

    def get_elfs(self, pred_data):
        """
        获取预测的各种elfs
        bg_mask: background mask
        tae_mask: tooth-all edge mask
        tte_mask: tooth-tooth edge mask
        edge_mask: edge mask = tae_mask ∪ tte_mask
        tae_lf: tooth-tooth edge logits field
        tte_lf: tooth-tooth edge logits field
        """

        # 如果有背景分支，则使用背景分支的logits，否则使用牙齿分支的logits
        if 'tg_seg_result' in pred_data.outputs and pred_data.outputs['tg_seg_result'] is not None:
            bg_lf = pred_data.outputs['tg_seg_result']
            bg_mask = torch.argmax(bg_lf, dim=1) > 0  # 0:bf, 1:tooth
            # print('tg_seg_result')
        else:
            bg_mask = torch.argmax(pred_data.outputs['seg_result'], dim=-1,keepdim=True) > 0  # 0:bf, 1:tooth
            bg_lf = torch.cat([pred_data.outputs['seg_result'][:, :, :1], -pred_data.outputs['seg_result'][:, :, :1]],dim=-1)
            bg_lf = bg_lf.permute(0, 2, 1)


        tae_lf = pred_data.outputs['edge_seg_result']  # B2N
        tte_lf = pred_data.outputs['tte_seg_result']  # B2N
        tae_mask = ((torch.argmax(tae_lf, dim=1) > 0) * (tae_lf[:, 1, :] > 0)) > 0  # 需满足两个条件
        tte_mask = ((torch.argmax(tte_lf, dim=1) > 0) * (tte_lf[:, 1, :] > 0)) > 0
        edge_mask = (tae_mask + tte_mask) > 0
        bg_mask, tae_mask, tte_mask, edge_mask = (
            bg_mask.view(-1), tae_mask.view(-1), tte_mask.view(-1), edge_mask.view(-1))  # N,


        # 通过pred_labels获取背景、牙齿、牙齿间、边缘的mask
        pred_labels = torch.argmax(pred_data.outputs['seg_result'], dim=-1).squeeze()

        drnn = DynamicRadiusNNEdgeMaskWithPos(rmin=2.2, rmax=3.0, k=5)
        ptae_mask, ptte_mask = drnn.generate_edge_masks(pred_data.pos, pred_labels)

        return bg_mask, tae_mask, tte_mask, edge_mask, ptae_mask, ptte_mask, tae_lf, tte_lf, bg_lf

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(' \
               f'alpha={self.alpha}, ' \
               f'max_iter={self.max_iter}, ' \
               f'k={self.k}, ' \
               f'iter_thred_len_ratio={self.iter_thred_len_ratio}, ' \
               f'field_bound_ratio={self.field_bound_ratio}, ' \
               f'field_sigma_ratio={self.field_sigma_ratio})'



class LoadTMCProposal(T.BaseTransform):
    """
    Topology morph cluster (TMC)
    拓扑形态学聚类。
    主要流程：
    1. 通过神经网络提取FPSSampling下的ELFs
    2. 将FPSSampling的ELFs映射到QSlimSampling的ELFs
    3. 提取Edge Mask（存在多种算法，需要进行消融实验）
    4. 去除Edge Mask获得连通分量
    5. 连通分量过滤
    6. 拓扑连通可视化及评估
    返回：proposal字典，包括预测质心列表，顶点聚类标签，质心数量等；
    """

    # pt: ${tmc_cluster.predictor_type}
    # npn: ${tmc_cluster.net_para_name}
    # ft: ${tmc_cluster.filter_type}
    # at: ${tmc_cluster.aggr_type}
    # scs: ${tmc_cluster.small_cluster_size_thre}
    # alpha: ${tmc_cluster.alpha}
    # k: ${tmc_cluster.k}
    # mi: ${tmc_cluster.max_iter}
    # itlr: ${tmc_cluster.iter_thred_len_ratio}
    # fbr: ${tmc_cluster.field_bound_ratio}
    # fsr: ${tmc_cluster.field_sigma_ratio}
    # td: ${tmc_cluster.thre_dist}
    # cn: ${tmc_cluster.ckpt_name}

    def __init__(self,
                 tmc_path = None,
                 pt='DTSegNet',  # 预测器类型
                 npn='seg_base',  # 网络参数名称
                 ft='tae_tte',  # 过滤类型，包含bg、tae、tae_tte、tae_tte_gtte、tae_tte_ptte、tae_tte_ptae
                 at='boolean',  # 聚合类型。默认为布尔聚合;包含boolean, add两种方式
                 scs = 10,  # 过滤小聚类大小的阈值
                 alpha = 0.12,  # alpha边长阈值
                 k = 3,  # KNN-ICA迭代邻域数
                 mi = 5,  # KNN-ICA最大迭代次数
                 itlr = 0.1,  # KNN-ICA迭代阈值长度比率
                 fbr = 3,  # UL线段带宽比率
                 fsr = 10,  # 距离带宽比率
                 td = 0.05,  # 聚类阈值
                 cn = 'DTSegNet',  # 模型检查点名称
                 ):
        # 调用父类初始化（完全仿照 FPSSample）
        super().__init__()

        self.tmc_path = tmc_path
        self.values = [pt, npn, ft, at, scs, alpha, k, mi, itlr, fbr, fsr, td, cn]
    def __call__(self,data: Data,) -> Data:
        """
        tmc_proposal:
        cluster_centroids: torch.Size([14, 3])
        cluster_labels :torch.Size([14, 1])
        per_cluster_count: torch.Size([14, 1])
        cluster_num: 14
        outputs:
          seg_result: torch.Size([1, 16000, 15]) #语义分割结果
          cbl_inputs
            0 : torch.Size([16000, 3]): 点云数据
            1 : None
            2 : torch.Size([1])


        """

        # 构建tmc实验版本号
        template_parts = [f"{value}" for value in self.values]
        version_name = "__".join(template_parts)

        file_name = f"{data.filename}.pt"
        save_path = os.path.join(self.tmc_path, version_name, file_name)
        tmc_proposal = torch.load(save_path)
        data.tmc_proposal = tmc_proposal

        return data
