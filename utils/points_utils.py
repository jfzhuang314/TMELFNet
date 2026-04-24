import torch
from torch_geometric.nn import radius,knn

def compute_polar_angles(pos):
    # 计算中心点 (等价于MATLAB中的 mean(V,1) + [0,1,0])
    cen = torch.mean(pos, dim=0) + torch.tensor([0, 1.0, 0], device=pos.device)
    # 计算从中心点到每个点的向量 (等价于MATLAB中的 vectors = V - repmat(cen, size(V,1), 1))
    vectors = pos - cen.expand_as(pos)
    # 计算XY平面上的角度 (等价于MATLAB中的 atan2(vectors(:,2), vectors(:,1)))
    # PyTorch的atan2输入顺序为(y, x)
    angles = torch.atan2(vectors[:, 1], vectors[:, 0])  # 范围是[-π, π]

    # 线性归一化到[-1, 1]区间 (等价于MATLAB中的线性归一化)
    min_angle = torch.min(angles)
    max_angle = torch.max(angles)
    # cen_angle = (max_angle + min_angle)/2.0
    normalized_angles = 2 * (angles - min_angle) / (max_angle - min_angle) - 1

    return normalized_angles, torch.abs(normalized_angles)

# 给定点的角度计算，注意与前面的区别
def compute_polar_angles_query(pos, clu_cen):
    # 计算中心点 (等价于MATLAB中的 mean(V,1) + [0,1,0])
    cen = torch.mean(pos, dim=0) + torch.tensor([0, 1.0, 0], device=pos.device)
    # 计算从中心点到每个点的向量 (等价于MATLAB中的 vectors = V - repmat(cen, size(V,1), 1))
    vectors = clu_cen - cen.expand_as(clu_cen)
    # 计算XY平面上的角度 (等价于MATLAB中的 atan2(vectors(:,2), vectors(:,1)))
    # PyTorch的atan2输入顺序为(y, x)
    angles = torch.atan2(vectors[:, 1], vectors[:, 0])  # 范围是[-π, π]

    # 线性归一化到[-1, 1]区间 (等价于MATLAB中的线性归一化)
    min_angle = torch.min(angles)
    max_angle = torch.max(angles)
    # cen_angle = (max_angle + min_angle)/2.0
    normalized_angles = 2 * (angles - min_angle) / (max_angle - min_angle) - 1

    return normalized_angles, torch.abs(normalized_angles)

def compute_knn_average_distance(pos, k=8):
    """
    计算点云中每个点与其k个最近邻点之间的平均距离
    :param pos: 点云坐标，形状为 [N, 3] 的tensor
    :param k: 最近邻的数量
    :return: 每个点与其k个最近邻点之间的平均距离，形状为 [N,] 的tensor
    """
    N = pos.size(0)

    # 使用knn函数找到每个点的k个最近邻
    edge_index = knn(pos, pos, k=k + 1)  # +1是因为点本身会被包含

    # 分离源节点和目标节点
    src_nodes = edge_index[0]  # 源节点索引
    dst_nodes = edge_index[1]  # 目标节点索引

    # 计算距离
    distances = torch.norm(pos[src_nodes] - pos[dst_nodes], p=2, dim=1)

    # 重塑为[N, k+1]，第一列是点到自身的距离（应该为0）
    distances = distances.view(N, k + 1)

    # 排除点到自身的距离（第一列），只保留到k个最近邻的距离
    knn_distances = distances[:, 1:]  # [N, k]

    # 计算平均距离
    avg_distances = torch.mean(knn_distances, dim=1)  # [N,]
    # 求平均距离的平均值
    avg_distances = torch.mean(avg_distances)

    return avg_distances


def compute_tae_gt_bin_labels_ang(pos, labels, r):
    """
    计算 edge_gt_bin_labels，支持每个点有不同的搜索半径,通过将半径分为4组来提高计算效率
    参数:
    pos (torch.Tensor): 点的位置张量，形状为 (num_points, num_dim)。
    labels (torch.Tensor): 点的标签张量，形状为 (num_points,)。
    r (torch.Tensor): 每个点的搜索半径，形状为 (num_points,)。
    返回:
    torch.Tensor: edge_gt_bin_labels 张量，形状为 (num_points,)。
    """
    num_points = pos.size(0)
    edge_gt_bin_labels = torch.zeros(num_points, dtype=torch.long, device=pos.device)

    # 根据半径分布将其分为4组
    # 使用四分位数进行分组
    q1, q2, q3 = torch.quantile(r, torch.tensor([0.20, 0.45, 0.70], device=r.device, dtype=r.dtype))

    # 定义4个半径区间
    radius_bins = [(r <= q1), (r > q1) & (r <= q2), (r > q2) & (r <= q3), (r > q3) ]

    # 为每个区间计算一个代表性半径
    representative_radii = [ torch.mean(r[mask]) if torch.sum(mask) > 0 else 0 for mask in radius_bins]

    for i, (mask, rii) in enumerate(zip(radius_bins, representative_radii)):
        if torch.sum(mask) == 0:
            continue

        point_indices = torch.nonzero(mask).squeeze()

        # 为当前组的点构建邻域图
        edge_index = radius(
            x=pos,  # 参考点（所有点）
            y=pos[point_indices],  # 查询点（当前组的点）
            r=rii,
            max_num_neighbors=64
        )

        # 处理邻域图，与之前相同
        ref_indices = edge_index[1]
        query_rel_indices = edge_index[0]
        query_abs_indices = point_indices[query_rel_indices]

        query_labels = labels[query_abs_indices]
        neighbor_labels = labels[ref_indices]

        has_diff_neighbor = (query_labels != neighbor_labels)

        if has_diff_neighbor.any():
            # 找出有不同标签邻居的查询点索引（去重）
            boundary_points = torch.unique(query_rel_indices[has_diff_neighbor.squeeze()])
            # 将相对索引转换为全局索引
            global_indices = point_indices[boundary_points]
            # 一次性标记所有边界点
            edge_gt_bin_labels[global_indices] = 1

    return edge_gt_bin_labels


def compute_tte_gt_bin_labels(pos, labels, r):
    """
    计算 compute_edge_gt_bin_labels_wo_bg，支持每个点有不同的搜索半径,通过将半径分为4组来提高计算效率。
    相比于上面的函数，这个函数会忽略标签为0的点，即背景点。这样提取的separable boundary会更有区分度
    参数:
    pos (torch.Tensor): 点的位置张量，形状为 (num_points, num_dim)。
    labels (torch.Tensor): 点的标签张量，形状为 (num_points,)。
    r (torch.Tensor): 每个点的搜索半径，形状为 (num_points,)。
    返回:
    torch.Tensor: edge_gt_bin_labels 张量，形状为 (num_points,)。
    """
    num_points = pos.size(0)
    edge_gt_bin_labels = torch.zeros(num_points, dtype=torch.long, device=pos.device)

    # 根据半径分布将其分为4组
    # 使用四分位数进行分组
    q1, q2, q3 = torch.quantile(r, torch.tensor([0.20, 0.45, 0.70], device=r.device, dtype=r.dtype))

    # 定义4个半径区间
    radius_bins = [(r <= q1), (r > q1) & (r <= q2), (r > q2) & (r <= q3), (r > q3) ]

    # 为每个区间计算一个代表性半径
    representative_radii = [ torch.mean(r[mask]) if torch.sum(mask) > 0 else 0 for mask in radius_bins]

    for i, (mask, rii) in enumerate(zip(radius_bins, representative_radii)):
        if torch.sum(mask) == 0:
            continue

        point_indices = torch.nonzero(mask).squeeze()

        # 为当前组的点构建邻域图
        edge_index = radius(
            x=pos,  # 参考点（所有点）
            y=pos[point_indices],  # 查询点（当前组的点）
            r=rii,
            max_num_neighbors=64
        )

        # 处理邻域图，与之前相同
        ref_indices = edge_index[1]
        query_rel_indices = edge_index[0]
        query_abs_indices = point_indices[query_rel_indices]

        query_labels = labels[query_abs_indices]
        neighbor_labels = labels[ref_indices]

        # 过滤掉 labels <= 0 的点
        valid_query_mask = query_labels > 0
        valid_neighbor_mask = neighbor_labels > 0
        valid_mask = valid_query_mask & valid_neighbor_mask

        assert valid_mask.numel() == query_rel_indices.numel(), f"valid_mask size {valid_mask.numel()} does not match query_rel_indices size {query_rel_indices.numel()}"
        # 仅在有效点中比较标签
        if valid_mask.any():
            valid_query_labels = query_labels[valid_mask]
            valid_neighbor_labels = neighbor_labels[valid_mask]

            query_rel_indices = query_rel_indices.unsqueeze(dim=1)
            valid_query_rel_indices = query_rel_indices[valid_mask]

            has_diff_neighbor = (valid_query_labels != valid_neighbor_labels)

            if has_diff_neighbor.any():
                # 找出有不同标签邻居的查询点索引（去重）
                boundary_points = torch.unique(valid_query_rel_indices[has_diff_neighbor.squeeze()])
                # 将相对索引转换为全局索引
                global_indices = point_indices[boundary_points]
                # 一次性标记所有边界点
                edge_gt_bin_labels[global_indices] = 1

    return edge_gt_bin_labels