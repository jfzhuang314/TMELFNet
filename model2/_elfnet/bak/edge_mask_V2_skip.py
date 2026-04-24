"""
@File    :   skip_mask.py
这一版本主要简化了输入特征的提取，仅仅使用两个edgeconv层提取特征；
"""

import argparse
import json
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN


import os
import random

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import time

from torch_geometric.nn import EdgeConv, global_max_pool, radius_graph, GATConv
from torch_cluster import knn, radius
from torch_geometric.data import DataLoader as GeoDataLoader, Data
from torch_scatter import scatter_min, scatter_max

from datasets.misc.DBGAMeshDataset import load_DBGA_dataset,load_DBGA_dataset_with_tbase
from metrics.DBGA_metrics import *
from model.DBGA_base_model import BaseNet
from model.BaseNet import BaseNet # 新的的BaseNet
from metrics.dice_loss import DiceLoss
# from model._edge_mask.modules.mask_dgcnn_atten import SemSegBlock
from model._dynamic_duo.utils import gen_utils as gu, tsg_loss, ops_utils as ou
import torch
from torch_cluster import radius_graph

from utils.points_utils import compute_polar_angles, compute_tae_gt_bin_labels_ang


class EdgeCNNBlockA(nn.Module):
    def __init__(self, input_dim):
        super(EdgeCNNBlockA, self).__init__()
        self.input_dim = input_dim
        # 定义 EdgeConv 层
        self.conv1 = EdgeConv(Seq(Lin(self.input_dim * 2, 32), ReLU()))
        self.conv2 = EdgeConv(Seq(Lin(32 * 2, 64), ReLU()))
        self.conv3 = EdgeConv(Seq(Lin(64 * 2, 64), ReLU()))

        self.attn1 = GATConv(32, 8, heads=4, dropout=0.1)
        self.attn2 = GATConv(64, 16, heads=4, dropout=0.1)
        self.attn3 = GATConv(64, 16, heads=4, dropout=0.1)

        self.fc = nn.Linear(32 + 64 + 64, 32)


    def forward(self, batch):
        data = batch

        x0 = data.x.float()
        edge_index1 = radius_graph(data.pos, r=0.02).to(data.pos.device, dtype=torch.long)
        edge_index2 = radius_graph(data.pos, r=0.04).to(data.pos.device, dtype=torch.long)
        edge_index3 = radius_graph(data.pos, r=0.08).to(data.pos.device, dtype=torch.long)


        x1 = F.relu(self.conv1(x0, edge_index1))
        x1 = self.attn1(x1, edge_index1) + x1

        x2 = F.relu(self.conv2(x1, edge_index2))
        x2 = self.attn2(x2, edge_index1) + x2

        x3 = F.relu(self.conv3(x2, edge_index3))
        x3 = self.attn3(x3, edge_index1) + x3

        x_concat = torch.cat([x1, x2, x3], dim=-1)

        x_low =self.fc(x_concat)   # 256

        return x_low

def first_seg_loss(pred_mask_1, pred_weight_1, gt_bin_label):
    # pred_mask_1: batch_size, 2, cropped
    # pred_weight_1: batch_size, 1, cropped
    # gt_label: batch_size, 1, cropped

    gt_bin_label = gt_bin_label.type(torch.long).view(gt_bin_label.shape[0], -1)

    bce_1 = torch.nn.NLLLoss(reduction='none')(pred_mask_1, gt_bin_label)
    pred_weight_1 = torch.sigmoid(pred_weight_1).view(pred_weight_1.shape[0], -1)
    loss = torch.mean((bce_1 * pred_weight_1) ** 2 + (1 - pred_weight_1) ** 2)

    return loss

def second_seg_loss(pred_mask_2, pred_weight_1, gt_bin_label):
    # pred_mask_2: batch_size, 2, cropped
    # pred_weight_1: batch_size, 1, cropped
    # gt_label: batch_size, 1, cropped

    gt_bin_label = gt_bin_label.type(torch.float32).view(gt_bin_label.shape[0], -1)

    pred_mask_2 = pred_mask_2.view(pred_mask_2.shape[0], -1)
    bce_2 = torch.nn.BCEWithLogitsLoss(reduction='none')(pred_mask_2, gt_bin_label)
    pred_weight_1 = torch.sigmoid(pred_weight_1).view(pred_weight_1.shape[0], -1)
    loss = torch.mean((2.0 - pred_weight_1) * bce_2)

    return loss

def compute_edge_gt_bin_labels(pos, labels,r=0.015):
    """
    计算 edge_gt_bin_labels。对于每个点，若其邻域内的 labels 不全部相同，则对应位置置为 1，否则置为 0。
    参数:
    pos (torch.Tensor): 点的位置张量，形状为 (num_points, num_dim)。
    labels (torch.Tensor): 点的标签张量，形状为 (num_points,)。
    r (float, 可选): 用于 radius_graph 的半径阈值，默认为 0.02。
    返回:
    torch.Tensor: edge_gt_bin_labels 张量，形状为 (num_points,)，数据类型为 torch.long。
    """
    num_points = pos.size(0)
    edge_gt_bin_labels = torch.zeros(num_points, dtype=torch.long, device=pos.device)

    # 提前计算一次 edge_index
    edge_index = radius_graph(pos, r=r).to(pos.device, dtype=torch.long)
    # edge_index = knn(pos, pos, k=k, batch_x=None, batch_y=None)

    # 获取每个中心节点对应的邻居标签
    center_node_indices = edge_index[0]
    neighbor_labels = labels[edge_index[1]]

    # 使用 scatter_min 和 scatter_max 分别找出每个中心节点邻居标签的最小值和最大值
    min_labels, _ = scatter_min(neighbor_labels, center_node_indices, dim_size=num_points)
    max_labels, _ = scatter_max(neighbor_labels, center_node_indices, dim_size=num_points)

    # 若最小值和最大值不相等，说明邻居标签不全部相同
    inconsistent_mask = min_labels != max_labels

    # 将标签不一致的节点对应位置置为 1
    edge_gt_bin_labels[inconsistent_mask] = 1

    return edge_gt_bin_labels

class EdgeCNNBlockAtten(nn.Module):
    def __init__(self, input_dim):
        super(EdgeCNNBlockAtten, self).__init__()
        self.input_dim = input_dim
        # 定义 EdgeConv 层
        self.conv1 = EdgeConv(Seq(Lin(self.input_dim * 2, 32), ReLU()))
        self.conv2 = EdgeConv(Seq(Lin(32 * 2, 64), ReLU()))
        self.conv3 = EdgeConv(Seq(Lin(64 * 2, 128), ReLU()))
        self.conv4 = EdgeConv(Seq(Lin(128 * 2, 128), ReLU()))

        # 注意力机制
        self.attn1 = GATConv(32, 8, heads=4, dropout=0.1)
        self.attn2 = GATConv(64, 16, heads=4, dropout=0.1)
        self.attn3 = GATConv(128, 32, heads=4, dropout=0.1)
        self.attn4 = GATConv(128, 32, heads=4, dropout=0.1)

        # 定义 MLP 层
        self.mlp1 = nn.Sequential(
            nn.Linear(128 + 128 + 64 + 32 + self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256)
        )

        self.fc1 = nn.Linear(128 + 128 + 64 + 32 + self.input_dim + 256, 128)
        self.fc2 = nn.Linear(128, 32)

        self.dp1 = nn.Dropout(p=0.5)

    def forward(self, x, edge_index_list):
        x1 = F.relu(self.conv1(x, edge_index_list[0]))
        x1 = self.attn1(x1, edge_index_list[0]) + x1

        x2 = F.relu(self.conv2(x1, edge_index_list[1]))
        x2 = self.attn2(x2, edge_index_list[1]) + x2

        x3 = F.relu(self.conv3(x2, edge_index_list[2]))
        x3 = self.attn3(x3, edge_index_list[2]) + x3

        x4 = F.relu(self.conv4(x3, edge_index_list[3]))
        x4 = self.attn4(x4, edge_index_list[3]) + x4

        x_concat = torch.cat([x, x1, x2, x3, x4], dim=-1)
        x_mlp1 = self.mlp1(x_concat)

        x_mlp1_pooled = torch.mean(x_mlp1, dim=0, keepdim=True).expand(x.size(0), -1)

        x_fc_input = torch.cat([x_concat, x_mlp1_pooled], dim=-1)
        x_fc_input = self.dp1(x_fc_input)
        x_F1 = F.relu(self.fc1(x_fc_input))   # 256
        x_F2 = self.fc2(x_F1)                 # 32

        return x1, x_F1, x_F2, x_mlp1


class PDMaskBlock(torch.nn.Module):
    """
    predict mask and weight
    """
    def __init__(self, num_classes=15, num_feat=35, k=16):
        super().__init__()
        # num_feat = 6
        self.input_dim = num_feat
        self.output_dim = num_classes

        self.num_feat = num_feat
        self.num_classes = num_classes

        self.k = k

        # 定义模型结构
        # 初始化 EdgeCNNModule 子类
        self.block1 = EdgeCNNBlockAtten(self.input_dim)
        self.block2 = EdgeCNNBlockAtten(self.input_dim + 2)

        self.pd_mask_1 = torch.nn.Linear(32, 2)
        self.pd_mask_1_softmax = torch.nn.Softmax(dim=1)
        self.wt_mask_1 = torch.nn.Linear(32, 1)

        self.pd_mask_2 = torch.nn.Linear(32, 1)

    def forward(self, batch, edge_index_list):
        data = batch

        points1 = data.float()
        # xyz = points1[:, :3, :]  # [B, C, N]
        #
        B, C, N = points1.shape

        # 展平 points1 以匹配边索引
        points1_flat = points1.transpose(1, 2).reshape(-1, points1.size(1))  # [B * N, C_total]

        # 步骤 3: 执行卷积操作
        x1, x1_F1, x1_F2, _ = self.block1(points1_flat, edge_index_list)

        pd_1 = self.pd_mask_1_softmax(self.pd_mask_1(x1_F2))
        weight_1 = self.wt_mask_1(x1_F2)

        points2_flat = torch.cat([points1_flat, pd_1], dim=1)
        x2, x2_F1, x2_F2, x_mlp1 = self.block2(points2_flat, edge_index_list)


        pd_2 = self.pd_mask_2(x2_F2)

        # 步骤 4: 恢复批量维度
        def reshape_to_batch(x):
            return x.reshape(B, N, -1).transpose(1, 2)

        pd_1 = reshape_to_batch(pd_1)
        weight_1 = reshape_to_batch(weight_1)
        pd_2 = reshape_to_batch(pd_2)
        # id_pred = reshape_to_batch(id_pred)

        return pd_1, weight_1, pd_2

class SemSegBlock(torch.nn.Module):
    def __init__(self, num_classes = 15, num_feat = 6, k=16):
        super().__init__()
        # num_feat = 6
        self.input_dim = num_feat
        self.output_dim = num_classes

        # self.num_feat = num_feat
        # self.num_classes = num_classes

        self.k = k

        # 定义模型结构
        self.conv1 = EdgeConv(Seq(Lin(self.input_dim*2, 32), ReLU()))
        self.conv2 = EdgeConv(Seq(Lin(32*2, 64), ReLU()))
        self.conv3 = EdgeConv(Seq(Lin(64*2, 128), ReLU()))
        self.conv4 = EdgeConv(Seq(Lin(128*2, 128), ReLU()))
        self.conv5 = EdgeConv(Seq(Lin(128*2, 128), ReLU()))
        self.conv6 = EdgeConv(Seq(Lin(128*2, 128), ReLU()))

        # 注意力机制
        self.attn1 = GATConv(32, 8, heads=4, dropout=0.1)
        self.attn2 = GATConv(64, 16, heads=4, dropout=0.1)
        self.attn3 = GATConv(128, 32, heads=4, dropout=0.1)
        self.attn4 = GATConv(128, 32, heads=4, dropout=0.1)
        self.attn5 = GATConv(128, 32, heads=4, dropout=0.1)
        self.attn6 = GATConv(128, 32, heads=4, dropout=0.1)

        self.mlp1 = nn.Sequential(nn.Linear(128+128+128+128+64+32 + self.input_dim, 128),nn.ReLU(),
                                  nn.Linear(128, 1024))


        self.fc1 = torch.nn.Linear(128+128+128+128+64+32+ self.input_dim + 1024, 256)
        self.fc2 = torch.nn.Linear(256, self.output_dim)

    def forward(self, batch):
        data = batch

        x0 = data.x.float()

        # 构造图
        edge_index = radius_graph(data.pos, r=0.12).to(data.pos.device, dtype=torch.long)
        edge_index2 = radius_graph(data.pos, r=0.24).to(data.pos.device, dtype=torch.long)
        edge_index3 = radius_graph(data.pos, r=0.36).to(data.pos.device, dtype=torch.long)
        edge_index4 = radius_graph(data.pos, r=0.50).to(data.pos.device, dtype=torch.long)
        edge_index5 = radius_graph(data.pos, r=0.70).to(data.pos.device, dtype=torch.long)
        edge_index6 = radius_graph(data.pos, r=0.90).to(data.pos.device, dtype=torch.long)

        # 第一层CNN模型的卷积 + 池化处理
        x1 = F.relu(self.conv1(x=x0, edge_index=edge_index))
        x1 = self.attn1(x1, edge_index) + x1
        # x = global_max_pool(x, batch_1)
        # 第二层CNN模型的卷积 + 池化处理
        x2 = F.relu(self.conv2(x=x1, edge_index=edge_index2))
        x2 = self.attn2(x2, edge_index2) + x2
        # x = global_max_pool(x, batch_1)
        # 第三层CNN模型的卷积 + 池化处理
        x3 = F.relu(self.conv3(x=x2, edge_index=edge_index3))
        x3 = self.attn3(x3, edge_index3)
        # 第四层CNN模型的卷积 + 池化处理
        x4 = F.relu(self.conv4(x=x3, edge_index=edge_index4))
        x4 = self.attn4(x4, edge_index4) + x3
        # # 第五层CNN模型的卷积 + 池化处
        x5 = F.relu(self.conv5(x=x4, edge_index=edge_index5))
        x5 = self.attn5(x5, edge_index5)  + x4
        # # 第六层CNN模型的卷积 + 池化处
        x6 = F.relu(self.conv6(x=x5, edge_index=edge_index6))
        x6 = self.attn5(x6, edge_index6)  + x5

        x = torch.cat([x0, x1, x2, x3, x4, x5, x6], dim=-1)

        x_mlp1 = F.relu(self.mlp1(x))
        # 进行平均池化，得到 1024 维向量
        x_mlp1_pooled = torch.mean(x_mlp1, dim=0, keepdim=True).expand(x.size(0), -1)
        # 将池化结果拼接到原始向量后方，形成 N*1216 向量
        x_F = torch.cat([x, x_mlp1_pooled], dim=1)

        x_sem = F.relu(self.fc1(x_F))
        x_sem = self.fc2(x_sem)
        x_sem = F.log_softmax(x_sem, dim=1)

        return x_sem, x1, x3


class get_seg_model(torch.nn.Module):
    def __init__(self, num_classes=15, num_feat=6, k=16):
        super().__init__()
        self.input_dim = num_feat
        self.output_dim = num_classes
        self.k = k

        # 定义模型结构
        # 初始化 EdgeCNNModule 子类
        self.input_trans_block = EdgeCNNBlockA(input_dim=num_feat)
        self.bg_pdmask_block = PDMaskBlock(num_feat=3+32, num_classes=num_classes, k=k)
        self.skip_pdmask_block = PDMaskBlock(num_feat=3+32, num_classes=num_classes, k=k)
        self.edge_pdmask_block = PDMaskBlock(num_feat=3+32, num_classes=num_classes, k=k)

        self.semseg_block = SemSegBlock(num_feat=6 +32 + 2 + 3 + 3 + 3, num_classes=num_classes, k=k)

        # self.fc1 = torch.nn.Linear(256 + 128 + 128 + 64 + 32 + 1024, 256)

        self.mlp1 = nn.Sequential(nn.Linear(128, 32))

    def forward(self, batch):
        pos = batch.pos.float()
        x = batch.x.float()
        x_1 = F.relu(self.input_trans_block(batch))

        ang, abs_ang = compute_polar_angles(pos)
        ang = ang.view([-1,1])
        abs_ang = abs_ang.view([-1,1])

        ####################
        # edge_index_list
        pos = batch.pos.float()
        xyz = pos.unsqueeze(dim=0).transpose(1, 2).float()  # [B, 3, N]
        B, _, N = xyz.shape # [B, C, N]
        # 步骤 1: 数据展平
        xyz_flat = xyz.transpose(1, 2).reshape(-1, 3)  # [B * N, C]
        batch_idx = torch.arange(B, device=xyz.device).repeat_interleave(N)  # 记录每个节点所属的批次
        # 步骤 2: 构建边索引
        edge_index_list = []
        radii = [0.03, 0.07, 0.15, 0.4]
        for r in radii:
            edge_index = radius_graph(xyz_flat, r=r, batch=batch_idx, max_num_neighbors=32)
            edge_index_list.append(edge_index)
        ####################


        xyz_bg = torch.cat((pos, x_1), dim=1).unsqueeze(dim=0).transpose(1, 2)
        xyz_edge = xyz_bg

        bg_pd_1, bg_weight_1, bg_pd_2 = self.bg_pdmask_block(xyz_bg,edge_index_list)
        skip_pd_1, skip_weight_1, skip_pd_2 = self.skip_pdmask_block(xyz_bg,edge_index_list)
        edge_pd_1, edge_weight_1, edge_pd_2 = self.edge_pdmask_block(xyz_edge,edge_index_list)


        xyz_3 = torch.cat((x, x_1, ang.view([-1,1]), abs_ang.view([-1,1]),
                           bg_pd_1.squeeze(dim=0).transpose(0, 1), bg_pd_2.squeeze(dim=0).transpose(0, 1),
                           skip_pd_1.squeeze(dim=0).transpose(0, 1), skip_pd_2.squeeze(dim=0).transpose(0, 1),
                           edge_pd_1.squeeze(dim=0).transpose(0, 1), edge_pd_2.squeeze(dim=0).transpose(0, 1)), dim=1)

        batch3 = batch.clone()
        batch3.x = xyz_3
        outputs, _, _ = self.semseg_block(batch3)

        return outputs, bg_pd_1, bg_weight_1, bg_pd_2, edge_pd_1,  edge_weight_1, edge_pd_2, skip_pd_1, skip_weight_1, skip_pd_2

class EdgeCNN(BaseNet):
    def __init__(self, config):
        super().__init__(config)

        num_classes = config.get('num_classes')

        # 模型
        self.model = get_seg_model(num_classes=num_classes,num_feat=6,k=16)
        self.model = nn.DataParallel(self.model)

        # dice loss
        self.dice_loss = DiceLoss()


    def on_train_batch_start(self, batch, batch_idx):
        # 在每个训练批次开始前清理内存（可选，但可能过于频繁）
        if batch_idx % 60 == 0:  # 每30个批次清理一次
            torch.cuda.empty_cache()
            print(f"Epoch end, GPU内存清理完成: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
    def on_validation_batch_start(self, batch, batch_idx):
        # 在每个训练批次开始前清理内存（可选，但可能过于频繁）
        if batch_idx % 60 == 0:  # 每30个批次清理一次
            torch.cuda.empty_cache()
            print(f"Epoch end, GPU内存清理完成: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")

    def _update_step_metrics(self, batch):
        # batch = self.normalize(batch)

        labels = batch.y.to(self.device, dtype=torch.long)
        one_hot_labels = nn.functional.one_hot(labels, num_classes=self.class_weights.size(0))

        # 调用网络模型获取结果
        outputs, bg_pd_1, bg_weight_1, bg_pd_2, edge_pd_1,  edge_weight_1, edge_pd_2, skip_pd_1, skip_weight_1, skip_pd_2  = self.model(batch)

        # semantic segmentation loss
        labels = labels.view(-1, 1)[:, 0]
        outputs1 = outputs.contiguous().view(-1, self.class_weights.size(0))
        seg_loss = F.nll_loss(outputs1, labels)
        # dice_loss
        dice_loss = self.dice_loss(outputs1, labels)

        def get_alpha_beta(epoch):
            alpha = 0.3 if epoch >= 20 else 0.7 - (0.7 - 0.3) * (epoch / 20)
            beta = 0.7 if epoch >= 20 else 0.3 + (0.7 - 0.3) * (epoch / 20)
            return alpha, beta

        # loss = seg_loss + mat_diff_loss * 0.01
        alpha, beta = get_alpha_beta(self.current_epoch)
        # loss = alpha * seg_loss + beta * dice_loss


        # predict bg mask loss
        bg_gt_bin_labels = (labels!=0)
        bg_gt_bin_labels = bg_gt_bin_labels.reshape([1, 1, -1])
        loss_s1 = first_seg_loss(bg_pd_1, bg_weight_1, bg_gt_bin_labels)
        loss_s2 = second_seg_loss(bg_pd_2, bg_weight_1, bg_gt_bin_labels)

        # predict edge mask loss
        # _, abs_ang = compute_polar_angles(batch.pos)
        # gt_labels = batch.y.to(self.device, dtype=torch.long)
        # radius = 0.02 + abs_ang * 0.06
        # edge_gt_bin_labels = compute_edge_gt_bin_labels_ang(batch.pos, gt_labels, r=radius)
        # gu.print_3d(gu.np_to_pcd_with_label(gu.torch_to_numpy(batch.pos).reshape(-1, 3), gu.torch_to_numpy(edge_gt_bin_labels)))
        edge_gt_bin_labels = batch.edge_gt_bin_labels.to(self.device, dtype=torch.long)
        edge_gt_bin_labels = edge_gt_bin_labels.reshape([1, 1, -1])
        loss_s3 = first_seg_loss(edge_pd_1, edge_weight_1, edge_gt_bin_labels)
        loss_s4 = second_seg_loss(edge_pd_2, edge_weight_1, edge_gt_bin_labels)

        # predict skip mask loss
        skip_gt_bin_labels = torch.isin(labels, torch.tensor([1, 3, 5, 7, 9, 11, 13], device=labels.device))
        skip_gt_bin_labels = skip_gt_bin_labels.reshape([1, 1, -1])
        loss_s5 = first_seg_loss(skip_pd_1, skip_weight_1, skip_gt_bin_labels)
        loss_s6 = second_seg_loss(skip_pd_2, skip_weight_1, skip_gt_bin_labels)

        #
        loss_l2 = self.regularization_losses()

        # loss = seg_loss + mat_diff_loss * 0.01
        lambda0, lambda1, lambda2, lambda3 = 2, 1, 1, 1
        loss = (lambda0*(alpha * seg_loss + beta * dice_loss) +
                lambda1*(loss_s1 + loss_s2) +
                lambda2*(loss_s3 + loss_s4) +
                lambda3 * (loss_s5 + loss_s6) +
                loss_l2)

        dsc = weighting_DSC(outputs.unsqueeze(dim=1), one_hot_labels, self.class_weights)
        sen = weighting_SEN(outputs.unsqueeze(dim=1), one_hot_labels, self.class_weights)
        ppv = weighting_PPV(outputs.unsqueeze(dim=1), one_hot_labels, self.class_weights)
        acc = weighting_ACC(outputs.unsqueeze(dim=1), one_hot_labels, self.class_weights)
        iou = weighting_IOU(outputs.unsqueeze(dim=1), one_hot_labels, self.class_weights)

        return dsc, loss, ppv, sen, acc, iou, seg_loss, dice_loss,loss_s1, loss_s2,loss_s3, loss_s4,alpha, beta,loss_s5, loss_s6

    def validation_step(self, batch, batch_idx):
        outputs = self._update_step_metrics(batch)
        dsc, loss, ppv, sen, acc, iou = outputs[0], outputs[1], outputs[2], outputs[3], outputs[4], outputs[5]
        seg_loss, dice_loss,loss_s1, loss_s2, loss_s3, loss_s4 = outputs[6], outputs[7], outputs[8], outputs[9], outputs[10], outputs[11]
        alpha, beta = outputs[12], outputs[13]
        loss_s5, loss_s6 = outputs[14], outputs[15]

        def get_mesh_name(file_path):
            file_name = os.path.basename(file_path)
            folder_name = os.path.basename(os.path.dirname(file_path))
            return f"{file_name}_{folder_name}"

        mesh_name = get_mesh_name(batch.filename[0])

        print(f"epoch:{self.current_epoch}-{mesh_name}, "
              f"loss:{loss:.4f}=Lseg:{alpha:.2f}*{seg_loss:.4f} + Ldice:{beta:.2f}*{dice_loss:.4f} "
              f"+ L1:{loss_s1:.4f} + L2:{loss_s2:.4f}+ L3:{loss_s3:.4f} + L4:{loss_s4:.4f}+ L5:{loss_s5:.4f} + L6:{loss_s6:.4f}", end=' ')

        metrics = {
            'test_loss': loss,
            'test_dsc': dsc,
            'test_sen': sen,
            'test_ppv': ppv,
            'test_acc': acc,
            'test_iou': iou
        }

        # print(metrics)

        self.log_dict(metrics)
        self.validation_step_outputs.append(metrics)
        return metrics


    def predict_step(self, batch, batch_idx):
        outputs = self._update_step_metrics(batch)
        dsc, loss, ppv, sen, acc = outputs[0], outputs[1], outputs[2], outputs[3], outputs[4]

        filename = batch.filename
        pts_normals = batch.x
        faces = batch.faces
        gt_labels = batch.y.to(self.device, dtype=torch.long)
        outs = self.model(batch)
        outputs, bg_pd_1, bg_weight_1, bg_pd_2, edge_pd_1, edge_weight_1, edge_pd_2 = outs[0:7]
        outputs1 = outputs.contiguous().view(-1, self.class_weights.size(0))
        _, pred_labels = torch.max(outputs1, 1)

        labels = gt_labels.view(-1, 1)[:, 0]
        edge_gt_bin_labels = batch.edge_gt_bin_labels

        metrics = {
            'predict_loss': loss,
            'predict_dsc': dsc,
            'predict_sen': sen,
            'predict_ppv': ppv,
            'predict_acc': acc,
            'pts_normals': pts_normals,
            'faces': faces,
            'gt_labels': gt_labels,
            'pred_labels': pred_labels,
            'mesh_file': filename,
            'bg_pd_1': bg_pd_1,
            'bg_weight_1': bg_weight_1,
            'bg_pd_2': bg_pd_2,
            'edge_pd_1': edge_pd_1,
            'edge_weight_1': edge_weight_1,
            'edge_pd_2': edge_pd_2,
            'edge_gt_bin_labels': edge_gt_bin_labels,
            'predictions': outputs1
        }

        return metrics

    def regularization_losses(self):
        #####################
        # Regularization loss
        #####################

        # 计算 L2 正则化损失
        l2_lambda = 0.0001  # 正则化系数，可根据实际情况调整
        l2_reg = torch.tensor(0., requires_grad=True).to(self.device)
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                l2_reg = l2_reg + torch.norm(param, 2)

        l2_loss = l2_lambda * l2_reg
        return l2_loss



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_args():
    """
    此函数用于创建并返回一个包含训练配置参数的 Namespace 对象。
    返回:
        argparse.Namespace: 包含训练配置参数的命名空间对象。
    """
    current_file_name = os.path.splitext(os.path.basename(__file__))[0]

    return argparse.Namespace(
        kfold_nums=5,
        use_fold=1,
        ncells=10000,
        logger_path=r'../tb_logs/',
        exp_name=r'exp_' + current_file_name,
        ckpt_path=r'../checkpoints/',
        ckpt_name=r'latest_checkpoint.tar',
        results_path=r'../results/',
        with_base_lower_path=r"E:\mesh_datasets\3DTeeth\Teeth3DS_train_test_split\with_base_lower.txt",
        wo_base_lower_path=r"E:\mesh_datasets\3DTeeth\Teeth3DS_train_test_split\wo_base_lower.txt",
        split_600_path=r"E:\mesh_datasets\3DTeeth\Teeth3DS_train_test_split\split_600_lower.txt",
        num_classes=15,
        num_neighbor=32,
        num_epochs=45,
        num_workers=16,
        sample_ratio = 1.0,
        train_val_ratio = 1.0,
        accumulate_grad_batches = 1,
        comb_name = 'comb_PointnetTrans',
        train_batch_size=1,
        val_batch_size=1,
        lr=0.001
    )
def get_cfg_from_args(args):
    config = {
        'num_classes': args.num_classes,
        'num_neighbor': args.num_neighbor,
        'lr': args.lr,
        'num_epochs': args.num_epochs,
        'num_batches_to_print': 150,
        'model_path': args.ckpt_path + '/' + args.version,
        'checkpoint_name': args.ckpt_name,
        'train_loss_csv_file': 'losses.csv',
        'save_model_threshold': 0.9
    }
    return config

def init_ckpts(args):
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.ckpt_path + '/' + args.version,
        filename='model-{epoch:02d}-{test_loss:.2f}',
        save_top_k=1,
        save_last=True,
        monitor='test_loss',
        mode='min'
    )
    # 检查是否存在检查点文件
    checkpoint_dir = args.ckpt_path + '/' + args.version

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)

    return checkpoint_callback, checkpoint_dir

def load_latest_ckpts(checkpoint_path, model):
    if os.path.exists(checkpoint_path) and len(os.listdir(checkpoint_path)) > 0:
        # 获取最新的检查点文件
        checkpoint_files = [os.path.join(checkpoint_path, f) for f in os.listdir(checkpoint_path) if f.endswith('.ckpt')]

        if len(checkpoint_files) < 1:
            return model, None

        checkpoint_files.sort(key=os.path.getmtime)
        latest_checkpoint_path = checkpoint_files[-1]
        checkpoint = torch.load(latest_checkpoint_path)

        # Remove the extra 'model.' prefix from the state_dict
        new_state_dict = {}
        for key, value in checkpoint['state_dict'].items():
            if key.startswith('model.'):
                new_key = key[6:]  # Remove 'model.'
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value

        model.model.load_state_dict(new_state_dict)
        return model,latest_checkpoint_path

    return model, None
def init_logger(args):
    model_name_path = '{}/{}classes_{}k_{}epochs_{}ncells_{}batchsize_{}kfold{}_{}lr'
    model_name_path = model_name_path.format(args.version, args.num_classes, args.num_neighbor, args.num_epochs,
                                             args.ncells,
                                             args.train_batch_size, args.kfold_nums, args.use_fold, args.lr)
    logger_path = os.path.join(args.logger_path, model_name_path)
    if not os.path.exists(logger_path):
        os.makedirs(logger_path, exist_ok=True)
    tb_logger = TensorBoardLogger(save_dir=args.logger_path, name=model_name_path)
    return tb_logger

def get_results_path(args):
    model_name_path = '{}/{}classes_{}epochs_{}ncells_{}batchsize_{}kfold{}_{}lr'
    model_name_path = model_name_path.format(args.version, args.num_classes, args.num_epochs,
                                             args.ncells,
                                             args.train_batch_size, args.kfold_nums, args.use_fold, args.lr)
    results_path = os.path.join(args.results_path, model_name_path)
    return results_path

def test2_DGCNN():
    # global num_classes, val_batch_size, lr, start, model
    setup_seed(1111)
    rseed = 42
    # rseed = 27
    # 全局参数
    args = get_args()
    #logger
    tb_logger = init_logger(args)
    #checkpoints
    checkpoint_callback, checkpoint_path = init_ckpts(args)
    # datasets
    train_ds, val_ds, test_ds = load_DBGA_dataset_with_tbase(sample_ratio=args.sample_ratio, train_val_ratio=1.0,
                                                             comb_name='comb_PointnetTrans', is_full = True,
                                                             retain_file_path=None, rseed=rseed)
    train_loader = GeoDataLoader(dataset=train_ds, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers,pin_memory=True,persistent_workers=True)
    test_loader = GeoDataLoader(dataset=test_ds, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers,pin_memory=True,persistent_workers=True)

    # model
    config = get_cfg_from_args(args)
    model = EdgeCNN(config)
    model,latest_checkpoint_path = load_latest_ckpts(checkpoint_path, model)    # trainer
    # trainer = pl.Trainer(max_epochs=num_epochs, accelerator='auto', num_sanity_val_steps=0)
    trainer = pl.Trainer(max_epochs=args.num_epochs,
                         logger=tb_logger, log_every_n_steps=20,
                         check_val_every_n_epoch=1,
                         accumulate_grad_batches=args.accumulate_grad_batches,
                         precision='16-mixed',  # 自动混合精度
                         callbacks=[checkpoint_callback]
                         )
    start = time.time()
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader, ckpt_path=latest_checkpoint_path)
    end = time.time()
    running_time = end - start
    print(f'Time cost: {running_time:.5f}s, {running_time / 3600} h\n')



def save_to_json(args, predictions):
    # 保存结果
    results_path = get_results_path(args)
    if not os.path.exists(results_path):
        os.makedirs(results_path, exist_ok=True)
    # 遍历predictions，并根据filename保存每个预测结果为json文件
    for prediction in predictions:
        filename = prediction['mesh_file']
        if isinstance(filename, list):
            if len(filename) > 0:
                filename = filename[0]
            else:
                print("Warning: The filename list is empty. Skipping this prediction.")
                continue
        base_name = os.path.basename(filename)
        dir_name = os.path.basename(os.path.dirname(filename))
        new_filename = f"{base_name}_{dir_name}.json"
        # 构造保存路径
        save_path = os.path.join(results_path, new_filename)
        # 转换张量为可序列化的类型
        serializable_prediction = {}
        for key, value in prediction.items():
            if isinstance(value, torch.Tensor):
                serializable_prediction[key] = value.cpu().tolist()
            else:
                serializable_prediction[key] = value
        # 保存预测结果为json文件
        with open(save_path, 'w') as f:
            json.dump(serializable_prediction, f)
            print("Saved prediction to", save_path)

def test3_pred():
    setup_seed(1111)
    # 全局参数
    args = get_args()
    #logger
    tb_logger = init_logger(args)
    #checkpoints
    checkpoint_callback, checkpoint_path = init_ckpts(args)

    args.sample_ratio = 1.0
    _, _, predict_ds = load_DBGA_dataset_with_tbase(sample_ratio=args.sample_ratio,
                                                    train_val_ratio=1.0, comb_name='comb_PointnetTrans',is_full=True,
                                                    retain_file_path=None, rseed=42)
    predict_loader = GeoDataLoader(dataset=predict_ds, batch_size=1, shuffle=False, num_workers=args.num_workers,
                                   pin_memory=True)

    config = get_cfg_from_args(args)
    model = EdgeCNN(config)

    # 检查是否存在检查点文件
    model,latest_checkpoint_path = load_latest_ckpts(checkpoint_path, model)

    trainer = pl.Trainer()
    # 进行预测
    if 'latest_checkpoint_path' in locals():
        print(latest_checkpoint_path, locals())
        predictions = trainer.predict(model, dataloaders=predict_loader, ckpt_path=latest_checkpoint_path)
    else:
        predictions = trainer.predict(model, dataloaders=predict_loader)

    save_to_json(args, predictions)





def test4_cent():
    setup_seed(1111)
    # 全局参数
    args = get_args()
    #logger
    tb_logger = init_logger(args)
    #checkpoints
    checkpoint_callback, checkpoint_path = init_ckpts(args)

    _, _, predict_ds = load_DBGA_dataset(sample_ratio=0.01, train_val_ratio=1.0, comb_name='comb_PointnetTrans')
    predict_loader = GeoDataLoader(dataset=predict_ds, batch_size=1, shuffle=False, num_workers=args.num_workers,
                                   pin_memory=True)

    config = get_cfg_from_args(args)
    model = EdgeCNN(config)

    # 检查是否存在检查点文件
    model,latest_checkpoint_path = load_latest_ckpts(checkpoint_path, model)

    trainer = pl.Trainer()
    # 进行预测
    if 'latest_checkpoint_path' in locals():
        print(latest_checkpoint_path, locals())
        predictions = trainer.predict(model, dataloaders=predict_loader, ckpt_path=latest_checkpoint_path)
    else:
        predictions = trainer.predict(model, dataloaders=predict_loader)

    for prediction in predictions:
        pts_normals = prediction['pts_normals']
        faces = prediction['faces']
        gt_labels = prediction['gt_labels']
        pred_labels = prediction['pred_labels']
        mesh_file = prediction['mesh_file']
        preds = prediction['predictions']
        print(mesh_file)

def test_get_model():
    model = get_seg_model(num_classes=15, num_feat=6, k=16)
    model.cuda()
    xyz = torch.rand(1250,6).cuda()
    data = Data(x=xyz, pos=xyz[:,:3])

    # output is B, C, N order
    for item in model(data):
        print(item.shape)

if __name__ == '__main__':
    # 示例使用
    # test_get_model()

    # test2_DGCNN()
    test3_pred()
    # test4_cent()
