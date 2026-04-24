import torch
from torch import nn
import torch.nn.functional as F

from model._cross_tooth.PointTransformer.libs.pointops.functions import pointops

# ========== 使用 PyG 替代 pointops.knnquery ==========
from torch_cluster import knn_graph, radius_graph

_inf = 1e9
_eps = 1e-12


class CBLLoss(nn.Module):

    def __init__(self, nsample=8, num_classes=15, dist_func='l2', contrast_func='nce'):
        super().__init__()
        # KNN邻居的个数k=16
        self.nsample = nsample
        self.num_classes = num_classes
        assert dist_func in ['l2', 'kl']
        self.dist_func = self.dist_l2 if dist_func == 'l2' else self.dist_kl
        assert contrast_func in ['nce', 'softnn']
        self.contrast_func = self.contrast_nce if contrast_func == 'nce' else self.contrast_softnn
        self.main_contrast = self.point_contrast

        self.dist = 'norm'
        self.temperature = None

    def dist_l2(self, features, neighbor_feature):
        dist = torch.unsqueeze(features, -2) - neighbor_feature
        dist = torch.sqrt(torch.sum(dist ** 2, axis=-1) + _eps)  # [m, nsample]
        return dist

    def dist_kl(self, features, neighbor_feature, normalized, normalized_neighbor):
        if normalized in [False, 'softmax']:
            features = F.log_softmax(features, dim=-1)
            log_target = True
        elif normalized == True:
            log_target = False
        else:
            raise ValueError(f'kl dist not support normalized = {normalized}')
        features = features.unsqueeze(-2)

        if normalized_neighbor in [False, 'softmax']:
            neighbor_feature = F.log_softmax(neighbor_feature, dim=-1)
        elif normalized_neighbor == True:
            neighbor_feature = torch.maximum(neighbor_feature, neighbor_feature.new_full([], _eps)).log()
        else:
            raise ValueError(f'kl dist not support normalized_neighbor = {normalized}')

        # (input, target) - i.e. (pred, gt), where input/pred should be in log space
        dist = F.kl_div(neighbor_feature, features, reduction='none', log_target=log_target)
        dist = dist.sum(-1)  # [m, nsample]
        return dist

    def contrast_softnn(self, dist, posmask, invalid_mask=None):
        dist = -dist
        dist = dist - torch.max(dist, -1, keepdim=True)[0]  # NOTE: max return both (max value, index)
        if self.temperature is not None:
            dist = dist / self.temperature
        exp = torch.exp(dist)

        if invalid_mask is not None:
            valid_mask = 1 - invalid_mask
            exp = exp * valid_mask

        pos = torch.sum(exp * posmask, axis=-1)  # (m)
        neg = torch.sum(exp, axis=-1)  # (m)
        loss = -torch.log(pos / neg + _eps)
        return loss

    def contrast_nce(self, dist, posmask, invalid_mask=None):
        dist = -dist
        dist = dist - torch.max(dist, -1, keepdim=True)[0]  # NOTE: max return both (max value, index)
        if self.temperature is not None:
            dist = dist / self.temperature
        exp = torch.exp(dist)

        if invalid_mask is not None:
            valid_mask = 1 - invalid_mask
            exp = exp * valid_mask

        # each Log term an example; per-pos vs. all negs
        neg = torch.sum(exp * (1 - posmask.int()), axis=-1)  # (m)
        under = exp + neg.unsqueeze(-1)
        loss = (exp / under)[posmask]  # each Log term an example
        loss = -torch.log(loss)
        return loss

    def point_contrast(self, p, x, o, nsample, target, pos_mask=None):
        # 适合 point transformer
        p, features, o = p, x, o

        # 目标分割标签
        labels = target
        neighbor_idx = self.get_neighbors(nsample, o, p)

        # exclude self-loop
        nsample = nsample - 1
        neighbor_idx = neighbor_idx[..., 1:].contiguous()
        m = neighbor_idx.shape[0]

        neighbor_label = labels[neighbor_idx.view(-1).long()].view(m, nsample)  # (m, nsample)

        if 'norm' in self.dist or self.dist == 'cos':
            features = F.normalize(features, dim=-1)  # p2-norm

        neighbor_feature = features[neighbor_idx.view(-1).long(), :].view(m, nsample, features.shape[1])


        # select only pos-neg co-exists
        if pos_mask is not None:
            point_mask = pos_mask
        else:
            posmask = labels.unsqueeze(-1) == neighbor_label
            point_mask = torch.sum(posmask.int(), -1)  # (m)
            point_mask = torch.logical_and(0 < point_mask, point_mask < nsample)

        # 特殊情况
        if not torch.any(point_mask):
            return torch.tensor(.0)

        posmask = posmask[point_mask]
        features = features[point_mask]
        neighbor_feature = neighbor_feature[point_mask]

        dist = self.dist_func(features, neighbor_feature)
        loss = self.contrast_func(dist, posmask)  # (m)

        loss = torch.mean(loss)
        return loss

    def get_neighbors(self, nsample, o, p):
        neighbor_idx, _ = pointops.knnquery(nsample, p, p, o, o)  # (m, nsample)
        return neighbor_idx

    def forward(self, output, target, pos_mask=None):
        p, x, o = output[0], output[1], output[2]
        loss = self.main_contrast(p, x, o, self.nsample, target, pos_mask)
        return loss

class PointopsKNN:
    def __init__(self, nsample):
        self.nsample = nsample
    def __call__(self, inputs):
        p, x, o = inputs
        neighbor_idx, _ = pointops.knnquery(self.nsample, p, p, o, o)  # (m, nsample)
        return neighbor_idx

class PyGKNN:
    def __init__(self, nsample):
        self.nsample = nsample
    def __call__(self, inputs):
        p, x, o = inputs  # (n, 3), (n, c), (b)

        # 构建 KNN 图（返回的是边索引）
        edge_index = knn_graph(
            x=p,  # 点坐标 [N, 3]
            k=self.nsample,  # 邻域数
            batch=o,  # batch 偏移量
            loop=True,  # 包含自环
            flow='source_to_target'  # 边方向
        )

        # 将 edge_index 转换为 neighbor_idx 格式 [N, K]
        # edge_index: [2, E] where E = N * K
        row, col = edge_index  # row: 源节点，col: 目标节点

        # 重新整理为每个点的 K 个邻居索引
        neighbor_idx = col.view(-1, self.nsample)  # [N, K]

class PyGRadiusNN:
    def __init__(self, radius, nsample):
        self.radius = radius
        self.nsample = nsample
    def __call__(self, inputs):
        p, x, o = inputs  # (n, 3), (n, c), (b)

        # 如果需要根据半径而不是固定 K 值查找邻居
        edge_index = radius_graph(
            x=p,  # 点坐标 [N, 3]
            r=self.radius,  # 搜索半径
            batch=o,  # batch 偏移量
            loop=True,
            max_num_neighbors=self.nsample  # 最大邻居数限制
        )

        # 转换为 neighbor_idx
        row, col = edge_index
        if col.numel() > 0:
            neighbor_idx = col.view(-1, self.nsample)
        else:
            neighbor_idx = torch.empty((p.size(0), 0), dtype=torch.long,device=p.device)


if __name__ == '__main__':
    criterion = CBLLoss().cuda()
    p = torch.rand((32000, 3)).cuda()
    x = torch.rand((32000, 17)).cuda()
    o = torch.IntTensor([16000, 32000]).cuda()
    target = torch.randint(0, 17, (32000,)).cuda()
    cbl_loss = criterion([p, x, o], target)
    print(cbl_loss.item())
