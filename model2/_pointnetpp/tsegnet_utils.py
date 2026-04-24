import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np

from external_libs.pointops.functions import pointops


def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()


def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


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
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    # device = xyz.device
    # B, N, C = xyz.shape
    # centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    # distance = torch.ones(B, N).to(device) * 1e10
    # farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    # batch_indices = torch.arange(B, dtype=torch.long).to(device)
    # for i in range(npoint):
    #     centroids[:, i] = farthest
    #     centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
    #     dist = torch.sum((xyz - centroid) ** 2, -1)
    #     mask = dist < distance
    #     distance[mask] = dist[mask]
    #     farthest = torch.max(distance, -1)[1]
    # return centroids


    xyz_batch = xyz.reshape(-1, 3).clone().detach()
    offset = torch.range(1, xyz.shape[0]) * xyz.shape[1]
    offset = torch.tensor(offset).cuda().type(torch.int)
    new_offset = torch.range(1, xyz.shape[0]) * npoint
    new_offset = new_offset.cuda().type(torch.int)
    xyz_batch = xyz_batch.contiguous()
    results = pointops.furthestsampling(xyz_batch, offset, new_offset)
    results = results.reshape(xyz.shape[0], npoint).type(torch.long)
    results = results - (torch.range(0, xyz.shape[0] - 1).cuda().type(torch.long) * xyz.shape[1]).reshape(-1, 1)

    return results


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points1: sampled points data with origin radius, [B, npoint, nsample, 3+D]
        new_points2: sampled points data with doubled radius, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx1 = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx1)  # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx1)
        new_points1 = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]
    else:
        new_points1 = grouped_xyz_norm
    """an SA module is divided into two block"""
    idx2 = query_ball_point(radius * 2, nsample * 2, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx2)  # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx2)
        new_points2 = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]
    else:
        new_points2 = grouped_xyz_norm

    if returnfps:
        return new_xyz, new_points1, new_points2, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points1, new_points2


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


def evalue_distance(centroids_pred, origin_points, target):
    """
    chamfer distance loss
    """
    centroids_gt = get_centroids(origin_points.transpose(1, 2), target)
    batch_num = centroids_pred.size(0)
    num_centroids_pred = centroids_pred.size(2)
    total_distance = 0

    for i in range(batch_num):
        try:
            single_centroids = (torch.stack(centroids_gt[i])).float().cuda().transpose(0, 1)  # 3, num_centroids
        except:
            continue
        num_centroids = single_centroids.size(1)
        pred_pos_ex = centroids_pred[i, :, :].unsqueeze(2).repeat(1, 1, num_centroids)
        target_pos_ex = single_centroids.unsqueeze(1).repeat(1, num_centroids_pred, 1)
        distance = (pred_pos_ex - target_pos_ex).pow(2).sum(0).min(1)[0].sum()
        distance2 = (pred_pos_ex - target_pos_ex).pow(2).sum(0).min(0)[0].sum()
        total_distance += distance / num_centroids_pred + distance2 / num_centroids
    return total_distance / batch_num


def get_centroids(origin_points, target):
    """
    :param origin_points: [B, npoint=16384, 9]
    :param target: [B, npoint=16384]
    :return: ground truth centroids coords list
    """
    batch_num = target.size(0)
    npoint = target.size(1)
    points = origin_points.data.numpy()
    target_np = target.cpu().data.numpy()
    all_centroids = list()
    for j in range(batch_num):
        single_tooth_centroids = list()
        a = np.unique(target_np[j, :])
        kinds = np.size(a)
        for k in a.tolist():  # except gingiva
            if k > 0:
                label = k
                index = np.argwhere(target_np[j, :] == label)
                index = torch.from_numpy(index)
                index = index.squeeze()
                try:
                    index = index.unsqueeze(1)
                except:
                    index = index.unsqueeze(0).unsqueeze(1)
                index = index.repeat(1, 3)
                single_tooth_tensor = origin_points[j, :, :3].gather(dim=0, index=index.long())
                centroid = single_tooth_tensor.mean(0)
                single_tooth_centroids.append(centroid)
            else:
                continue
        all_centroids.append(single_tooth_centroids)
    return all_centroids


def crop_proposal(xyz, centroids, npoint):
    """
    :param xyz: pointcloud data, [B, N, 3]
    :param centroids, [B, num_centroids, 3]
    :param npoint: number of samples
    :return: crop_points: sampled pointcloud index, [B, num_centroids, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    num_proposals = centroids.shape[1]
    crop_points = torch.zeros(B, num_proposals, int(npoint), dtype=torch.long).to(device)
    distance = torch.zeros(B, N).to(device)
    for i in range(num_proposals):
        distance = torch.sum((xyz - centroids[:, i, :].unsqueeze(1)) ** 2, -1)
        _, single_index = torch.topk(distance, int(npoint), dim=-1, largest=False)
        crop_points[:, i, :] = single_index
    return crop_points


def gather_points(xyz, index):
    """
    :param xyz: pointcloud data, [B, x, N]
    :param index: cropping proposals index, [B, num_centroids, number of samples]
    :return: proposals_xyz: cropping points [B, num_centroids, x, number of samples]
    """
    batch_size, num_centroids, npoint = index.shape
    dimensions = xyz.shape[1]
    proposals_xyz = torch.zeros(batch_size, num_centroids, dimensions, npoint)
    for i in range(num_centroids):
        idx = index[:, i, :].unsqueeze(1).repeat(1, dimensions, 1)
        proposals_xyz[:, i, :, :] = torch.gather(xyz, 2, idx)
    return proposals_xyz


def get_distance_field(xyz, centroids):
    """
    calculate proposal dense distance field
    :param xyz: proposal coordinates, [B, num_centroids, 3, number of samples]
    :param centroids: centroids coordinates, [B, num_centroids, 3]
    :return: distance_field, [B, num_centroids, number of samples]
    """
    distance_field = torch.sum((xyz - centroids.unsqueeze(-1)) ** 2, -2)
    distance_field = torch.exp(-4 * distance_field)
    return distance_field


def get_proposal(xyz, points, centroids):
    """
    get proposal information for segmentation
    :param xyz: pointcloud data, [B, 3, N]
    :param points: pointcloud data, [B, x, N]
    :param centroids: [B, num_centroids, 3]
    :return: proposals: [B, 3+x+1, n]
             proposals_index: [B, num_centroids, n]
    """
    npoint = xyz.shape[2] * 0.25
    proposals_index = crop_proposal(xyz.permute(0, 2, 1), centroids.permute(0, 2, 1), npoint)   # 16, 14, 1024
    proposals_xyz = gather_points(xyz, proposals_index).float().cuda()  # B, 14, 3, n
    proposals_points = gather_points(points, proposals_index).float().cuda()  # B, 14, 32, n
    distance_field = get_distance_field(proposals_xyz, centroids.permute(0, 2, 1)).unsqueeze(2)   # B, 14, 1, n
    proposals = torch.concat((proposals_xyz, proposals_points, distance_field), 2)
    return proposals, proposals_index


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points1 = sample_and_group_all(xyz, points)
            new_points2 = new_points1
        else:
            new_xyz, new_points1, new_points2 = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points1 = new_points1.permute(0, 3, 2, 1)  # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points1 = F.relu(bn(conv(new_points1)))

        new_points1 = torch.max(new_points1, 2)[0]

        new_points2 = new_points2.permute(0, 3, 2, 1)  # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points2 = F.relu(bn(conv(new_points2)))

        new_points2 = torch.max(new_points2, 2)[0]

        new_points = torch.cat((new_points1, new_points2), 1)  # Sampling at two scales
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


# class PointNetFeaturePropagation(nn.Module):
#     def __init__(self, in_channel, mlp):
#         super(PointNetFeaturePropagation, self).__init__()
#         self.mlp_convs = nn.ModuleList()
#         self.mlp_bns = nn.ModuleList()
#         last_channel = in_channel
#         for out_channel in mlp:
#             self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
#             self.mlp_bns.append(nn.BatchNorm1d(out_channel))
#             last_channel = out_channel
#
#     def forward(self, xyz1, xyz2, points1, points2):
#         """
#         Input:
#             xyz1: input points position data, [B, C, N]
#             xyz2: sampled input points position data, [B, C, S]
#             points1: input points data, [B, D, N]
#             points2: input points data, [B, D, S]
#         Return:
#             new_points: upsampled points data, [B, D', N]
#         """
#         xyz1 = xyz1.permute(0, 2, 1)
#         xyz2 = xyz2.permute(0, 2, 1)
#
#         points2 = points2.permute(0, 2, 1)
#         B, N, C = xyz1.shape
#         _, S, _ = xyz2.shape
#
#         if S == 1:
#             interpolated_points = points2.repeat(1, N, 1)
#         else:
#             dists = square_distance(xyz1, xyz2)
#             dists, idx = dists.sort(dim=-1)
#             dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]
#
#             dist_recip = 1.0 / (dists + 1e-8)
#             norm = torch.sum(dist_recip, dim=2, keepdim=True)
#             weight = dist_recip / norm
#             interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)
#
#         if points1 is not None:
#             points1 = points1.permute(0, 2, 1)
#             new_points = torch.cat([points1, interpolated_points], dim=-1)
#         else:
#             new_points = interpolated_points
#
#         new_points = new_points.permute(0, 2, 1)
#         for i, conv in enumerate(self.mlp_convs):
#             bn = self.mlp_bns[i]
#             new_points = F.relu(bn(conv(new_points)))
#         return new_points

class PointNetFeaturePropagation(nn.Module):
    """
    彻底重构的 Feature Propagation
    核心改进：
    1. 更激进的距离截断（防止重合点）
    2. 使用立方反比权重（比平方反比更稳定）
    3. 强制 BN 进入 eval 模式
    """
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: [B, C, N] - 目标点（密集）
            xyz2: [B, C, S] - 源点（稀疏）
            points1: [B, D1, N]
            points2: [B, D2, S]
        Return:
            new_points: [B, D', N]
        """
        # 转 BNC
        xyz1 = xyz1.permute(0, 2, 1)  # [B, N, 3]
        xyz2 = xyz2.permute(0, 2, 1)  # [B, S, 3]
        points2 = points2.permute(0, 2, 1)  # [B, S, D]

        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            # ========== 关键改进 1：距离计算 + 立即清理 ==========
            dists = square_distance(xyz1, xyz2)  # [B, N, S]

            # 立即替换 NaN（防止后续传播）
            if torch.isnan(dists).any():
                max_finite = torch.finfo(dists.dtype).max / 1e6
                dists = torch.nan_to_num(dists, nan=max_finite)

            # 排序取 k 近邻
            k = min(3, S)
            dists, idx = dists.sort(dim=-1)
            dists = dists[:, :, :k]
            idx = idx[:, :, :k]

            # ========== 关键改进 2：超激进距离截断 ==========
            # 使用比之前大 10 倍的阈值！
            min_dist = 10e-4  # ← 关键：从 1e-4 提升到 0.01
            dists = torch.clamp(dists, min=min_dist)

            # ========== 关键改进 3：立方反比权重（更平滑）==========
            # w = 1 / d^3 而不是 1/d² 或 1/d
            dists_cube = dists
            dist_recip = 1.0 / (dists_cube + 1e-6)  # 加大 epsilon

            # 归一化
            norm = torch.sum(dist_recip, dim=2, keepdim=True) + 1e-6
            weight = dist_recip / norm

            # 检查权重集中程度
            max_w = weight.max().item()
            if max_w > 0.9:  # 如果某个点权重过大
                # print(f"⚠️ FP: 权重集中 (max={max_w:.4f})，使用均匀权重")
                # weight = torch.ones_like(weight) / k

                # 找到权重最大的点的索引
                max_idx = torch.argmax(weight, dim=2)  # [B, N]
                B, N, K = weight.shape
                # 创建掩码标记最大权重的点
                mask_max = torch.zeros_like(weight).scatter_(2, max_idx.unsqueeze(2), 1.0)
                # 将最大权重限制为 0.9
                # weight = weight * mask_max * 0.9 / (max_w + 1e-8) + \
                #          weight * (1 - mask_max)  # 其他点保持原值

                # 计算剩余权重（0.1）平均分配给其他 k-1 个点
                remaining_weight = 0.1
                other_points_count = K - 1
                avg_other_weight = remaining_weight / other_points_count

                # 重新归一化：最大点 0.9，其他点平均分配 0.1
                weight = mask_max * 0.9 + (1 - mask_max) * avg_other_weight

            # 插值
            neighbor_features = index_points(points2, idx)
            interpolated_points = torch.sum(
                neighbor_features * weight.view(B, N, k, 1),
                dim=2
            )

        # 拼接
        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        # 转 BCN
        new_points = new_points.permute(0, 2, 1)

        # ========== 关键改进 4：MLP 层强制 BN eval ==========
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]

            # 清理输入
            if torch.isnan(new_points).any() or torch.isinf(new_points).any():
                print(f"⚠️ FP MLP{i}: 清理异常输入")
                new_points = torch.nan_to_num(new_points, nan=0.0, posinf=1e6, neginf=-1e6)

            new_points = conv(new_points)

            # 强制 BN 使用全局统计量（解决 batch_size=1 问题）
            bn.eval()

            if torch.isnan(new_points).any() or torch.isinf(new_points).any():
                print(f"⚠️ FP MLP{i}: conv 后异常，跳过 BN")
                new_points = torch.nan_to_num(new_points, nan=0.0)
                new_points = F.relu(new_points)
            else:
                try:
                    new_points = F.relu(bn(new_points))
                except:
                    print(f"⚠️ FP MLP{i}: BN 失败，简化处理")
                    new_points = F.relu(new_points)

        return new_points
