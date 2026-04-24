import torch
import torch.nn as nn
import torch.nn.functional as F
from models.PointTransformer.libs.pointops.functions import pointops

_inf = 1e9
_eps = 1e-12


def get_subscene_features(stage_n, stage_i, stage_list, x, nstride, kr=None, extend=False, return_neighbor=False):
    if stage_i == 0 and not extend:
        return x.float()

    if kr is None:  # infer from sub-sampling (nstride) as default
        i = 1 if stage_i == 0 and extend else stage_i
        kr = torch.prod(nstride[:i])

    stage_from = stage_list['up'][0]  # support
    p_from, o_from = stage_from['p_out'], stage_from['offset']

    stage_to = stage_list[stage_n][stage_i]  # query
    p_to, o_to = stage_to['p_out'], stage_to['offset']

    neighbor_idx, _ = pointops.knnquery(kr, p_from, p_to, o_from, o_to)  # (m, kr) - may have invalid neighbor

    # print('kr - x', kr, x.shape)
    # print(p_from.shape, p_to.shape)
    # print(o_from, o_to)
    # print('neighbor_idx.shape = ', neighbor_idx.shape)
    # print(neighbor_idx.min(), neighbor_idx.max(), (neighbor_idx == neighbor_idx.max()).int().sum())

    # neighbor_idx[neighbor_idx > p.shape[0]] = p.shape[0]  # collect all 0s if invalid neighbor
    # x = torch.cat([x, torch.zeros([1, self.config.num_classes])])

    neighbor_idx = neighbor_idx.view(-1).long()
    x = x[neighbor_idx, :].view(p_to.shape[0], kr, x.shape[1]) # (m, kr, ncls)
    x = x.float().mean(-2)  # (m, ncls)

    # x = x.float().sum(-2)  # (m, ncls)
    # cnt = (neighbor_idx < p.shape[0]).float().sum(-1, keepdim=True)  # (m, 1)
    # x /= cnt
    if return_neighbor:
        return x, neighbor_idx, kr
    return x
