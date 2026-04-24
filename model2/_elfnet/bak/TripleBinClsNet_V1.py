"""
@File    :   skip_mask.py
这一版本主要简化了输入特征的提取，仅仅使用两个edgeconv层提取特征；
"""
import argparse
import json
from abc import abstractmethod
import torch
from functools import lru_cache

import pandas as pd
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
from torch_geometric.data import DataLoader as GeoDataLoader, Data

from datasets.misc.DBGAMeshDataset import load_DBGA_dataset,load_DBGA_dataset_with_tbase
from datasets.TGNDataset import load_datasets
from metrics.DBGA_metrics import *
from model.BaseNet import BaseNet # 新的的BaseNet
from metrics.dice_loss import DiceLoss
# from model._edge_mask.modules.mask_dgcnn_atten import SemSegBlock, PDMaskBlock
from torch_cluster import radius_graph
from utils.points_utils import *
from model._elfnet.topo_clustering import get_json_obj, topo_cluster

class MyLoss:
    @staticmethod
    def first_seg_loss(pred_mask_1, pred_weight_1, gt_bin_label):
        # pred_mask_1: batch_size, 2, cropped
        # pred_weight_1: batch_size, 1, cropped
        # gt_label: batch_size, 1, cropped

        gt_bin_label = gt_bin_label.type(torch.long).view(gt_bin_label.shape[0], -1)

        bce_1 = torch.nn.NLLLoss(reduction='none')(pred_mask_1, gt_bin_label)
        pred_weight_1 = torch.sigmoid(pred_weight_1).view(pred_weight_1.shape[0], -1)
        loss = torch.mean((bce_1 * pred_weight_1) ** 2 + (1 - pred_weight_1) ** 2)

        return loss

    @staticmethod
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

class ShallowFA(nn.Module):
    """
    浅层特征提取网络。Shallow FA 模块，用于处理输入数据并生成输出特征。
    """
    def __init__(self, input_dim):
        super(ShallowFA, self).__init__()
        self.input_dim = input_dim
        # 定义 EdgeConv 层
        self.conv1 = EdgeConv(Seq(Lin(self.input_dim * 2, 32), ReLU()))
        self.conv2 = EdgeConv(Seq(Lin(32 * 2, 32), ReLU()))

    def forward(self, batch):
        data = batch

        x0 = data.x.float()
        edge_index1 = radius_graph(data.pos, r=0.03).to(data.pos.device, dtype=torch.long)
        edge_index2 = radius_graph(data.pos, r=0.10).to(data.pos.device, dtype=torch.long)

        x1 = F.relu(self.conv1(x0, edge_index1))
        x2 = F.relu(self.conv2(x1, edge_index2))

        return x2

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
        self.attn1 = GATConv(32, 8, heads=4, dropout=0.6)
        self.attn2 = GATConv(64, 16, heads=4, dropout=0.6)
        self.attn3 = GATConv(128, 32, heads=4, dropout=0.6)
        self.attn4 = GATConv(128, 32, heads=4, dropout=0.6)

        # 定义 MLP 层
        self.mlp1 = nn.Sequential(
            nn.Linear(128 + 128 + 64 + 32, 128),
            nn.ReLU(),
            nn.Linear(128, 256)
        )

        self.fc1 = nn.Linear(128 + 128 + 64 + 32 + 256, 128)
        self.fc2 = nn.Linear(128, 32)

        # self.dp1 = nn.Dropout(p=0.5)

    def forward(self, x, edge_index_list):
        x1 = F.relu(self.conv1(x, edge_index_list[0]))
        x1 = self.attn1(x1, edge_index_list[0]) + x1

        x2 = F.relu(self.conv2(x1, edge_index_list[1]))
        x2 = self.attn2(x2, edge_index_list[1]) + x2

        x3 = F.relu(self.conv3(x2, edge_index_list[2]))
        x3 = self.attn3(x3, edge_index_list[2]) + x3

        x4 = F.relu(self.conv4(x3, edge_index_list[3]))
        x4 = self.attn4(x4, edge_index_list[3]) + x4

        x_concat = torch.cat([x1, x2, x3, x4], dim=-1)
        x_mlp1 = self.mlp1(x_concat)

        x_mlp1_pooled = torch.mean(x_mlp1, dim=0, keepdim=True).expand(x.size(0), -1)

        x_fc_input = torch.cat([x_concat, x_mlp1_pooled], dim=-1)
        # x_fc_input = self.dp1(x_fc_input)
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

        return pd_1, weight_1, pd_2

class EdgeMaskBlock(torch.nn.Module):
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
        self.block3 = EdgeCNNBlockAtten(self.input_dim + 2)

        # 第一个分支，实现了带背景的预测
        self.pd_mask_1 = torch.nn.Linear(32, 2)
        self.pd_mask_1_softmax = torch.nn.Softmax(dim=1)
        self.wt_mask_1 = torch.nn.Linear(32, 1)
        self.pd_mask_2 = torch.nn.Linear(32, 1)

        # 第二个分支，实现无背景的预测
        self.pd_wobg_mask_1 = torch.nn.Linear(32, 2)
        self.pd_wobg_mask_1_softmax = torch.nn.Softmax(dim=1)
        self.wt_wobg_mask_1 = torch.nn.Linear(32, 1)
        self.pd_wobg_mask_2 = torch.nn.Linear(32, 1)

    def forward(self, batch, edge_index_list):
        data = batch

        points1 = data.float()
        B, C, N = points1.shape
        # 展平 points1 以匹配边索引
        points1_flat = points1.transpose(1, 2).reshape(-1, points1.size(1))  # [B * N, C_total]

        # 步骤 3: 执行卷积操作
        x1, x1_F1, x1_F2, _ = self.block1(points1_flat, edge_index_list)

        # 3a 模块1
        pd_1 = self.pd_mask_1_softmax(self.pd_mask_1(x1_F2))
        weight_1 = self.wt_mask_1(x1_F2)
        points2_flat = torch.cat([points1_flat, pd_1], dim=1)
        x2, x2_F1, x2_F2, x_mlp1 = self.block2(points2_flat, edge_index_list)
        pd_2 = self.pd_mask_2(x2_F2)

        # 3b 模块2
        pd_wobg_1 = self.pd_wobg_mask_1_softmax(self.pd_wobg_mask_1(x1_F2))
        weight_wobg_1 = self.wt_wobg_mask_1(x1_F2)
        points2_wobg_flat = torch.cat([points1_flat, pd_wobg_1], dim=1)
        _, _, x2_wobg_F2, _ = self.block3(points2_wobg_flat, edge_index_list)
        pd_wobg_2 = self.pd_wobg_mask_2(x2_wobg_F2)

        # 步骤 4: 恢复批量维度
        def reshape_to_batch(x):
            return x.reshape(B, N, -1).transpose(1, 2)

        pd_1 = reshape_to_batch(pd_1)
        weight_1 = reshape_to_batch(weight_1)
        pd_2 = reshape_to_batch(pd_2)

        pd_wobg_1 = reshape_to_batch(pd_wobg_1)
        weight_wobg_1 = reshape_to_batch(weight_wobg_1)
        pd_wobg_2 = reshape_to_batch(pd_wobg_2)

        return pd_1, weight_1, pd_2,pd_wobg_1, weight_wobg_1, pd_wobg_2

class TripleBinClsNet(torch.nn.Module):
    """ 双二值分类网络： TGMaskNet及EdgeMaskNet"""
    def __init__(self, num_classes=15, num_feat=6, k=16):
        super().__init__()
        self.input_dim = num_feat
        self.output_dim = num_classes
        self.k = k

        # 定义模型结构
        # 初始化 EdgeCNNModule 子类
        self.shallow_FA = ShallowFA(input_dim=num_feat)
        self.bg_pdmask_block = PDMaskBlock(num_feat=3+32, num_classes=num_classes, k=k)
        self.edge_pdmask_block = EdgeMaskBlock(num_feat=3+32, num_classes=num_classes, k=k)
        # self.edge_wogt_pdmask_block = PDMaskBlock(num_feat=3 + 32, num_classes=num_classes, k=k)

    def forward(self, batch):
        pos = batch.pos.float()
        x = batch.x.float()
        x_1 = F.relu(self.shallow_FA(batch))

        ################3#####################
        x2 = torch.cat((pos, x_1), dim=1).unsqueeze(dim=0).transpose(1, 2)
        points1 = x2.float()
        xyz = points1[:, :3, :]  # [B, C, N]
        B, C, N = points1.shape
        # 步骤 1: 数据展平
        xyz_flat = xyz.transpose(1, 2).reshape(-1, 3)  # [B * N, C]
        batch_idx = torch.arange(B, device=xyz.device).repeat_interleave(N)  # 记录每个节点所属的批次
        # 步骤 2: 构建边索引
        edge_index_list = []
        radii = [0.03, 0.07, 0.15, 0.4]
        for r in radii:
            edge_index = radius_graph(xyz_flat, r=r, batch=batch_idx, max_num_neighbors=32)
            edge_index_list.append(edge_index)

        ######################################
        xyz_bg = x2
        bg_pd_1, bg_weight_1, bg_pd_2 = self.bg_pdmask_block(xyz_bg,edge_index_list)

        xyz_edge = xyz_bg
        edge_pd_1, edge_weight_1, edge_pd_2,edge_wobg_pd_1, edge_wobg_weight_1, edge_wobg_pd_2 = self.edge_pdmask_block(xyz_edge,edge_index_list)

        # xyz_edge = xyz_bg
        # edge_wobg_pd_1, edge_wobg_weight_1, edge_wobg_pd_2 = self.edge_wogt_pdmask_block(xyz_edge,edge_index_list)

        return bg_pd_1, bg_weight_1, bg_pd_2, edge_pd_1,  edge_weight_1, edge_pd_2,edge_wobg_pd_1, edge_wobg_weight_1, edge_wobg_pd_2,x_1

class TripleBinClsModule(BaseNet):
    """ 双二值分类网络训练模块"""
    def __init__(self, config):
        super().__init__(config)

        num_classes = config.get('num_classes')
        # num_classes = 2
        # 模型
        self.model = TripleBinClsNet(num_classes=num_classes, num_feat=6, k=16)
        self.model = nn.DataParallel(self.model)

        # dice loss
        self.dice_loss = DiceLoss()

        # edge_mask_radius
        # self.edge_mask_radius = 0.03
        self.rmin = config.get('rmin')
        self.rmax = config.get('rmax')

    def _init_metrics(self, num_classes):
        self.losses, self.mdsc_tae, self.msen_tae, self.mppv_tae, self.macc_tae, self.miou_tae = [], [], [], [],[],[]
        self.mdsc_tte, self.msen_tte, self.mppv_tte, self.macc_tte, self.miou_tte = [], [], [], [], []
        self.mdsc_tg, self.msen_tg, self.mppv_tg, self.macc_tg, self.miou_tg = [], [], [], [], []

        self.test_losses, self.test_mdsc_tae, self.test_msen_tae, self.test_mppv_tae, self.test_macc_tae, self.test_miou_tae = [], [], [], [], [], []
        self.test_mdsc_tte, self.test_msen_tte, self.test_mppv_tte, self.test_macc_tte, self.test_miou_tte = [], [], [], [], []
        self.test_mdsc_tg, self.test_msen_tg, self.test_mppv_tg, self.test_macc_tg, self.test_miou_tg = [], [], [], [], []

        self.best_test_dsc = 0.0
        self.class_weights = torch.ones(num_classes, dtype=torch.float)

        # 输出信息
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def _update_step_metrics(self, batch):
        # batch = self.normalize(batch)

        labels = batch.y.to(self.device, dtype=torch.long)
        labels = labels.reshape([-1,1])
        # 调用网络模型获取结果
        outputs = self.model(batch)
        bg_pd_1, bg_weight_1, bg_pd_2, edge_pd_1, edge_weight_1, edge_pd_2 = outputs[0], outputs[1], outputs[2], outputs[3], outputs[4], outputs[5]
        edge_wobg_pd_1, edge_wobg_weight_1, edge_wobg_pd_2 = outputs[6],outputs[7],outputs[8]

        def get_alpha_beta(epoch):
            alpha = 0.3 if epoch >= 20 else 0.7 - (0.7 - 0.3) * (epoch / 20)
            beta = 0.7 if epoch >= 20 else 0.3 + (0.7 - 0.3) * (epoch / 20)
            return alpha, beta

        # loss = seg_loss + mat_diff_loss * 0.01
        alpha, beta = get_alpha_beta(self.current_epoch)
        # loss = alpha * seg_loss + beta * dice_loss

        # predict bg mask loss
        bg_gt_bin_labels = (labels!=0).long()
        onehot_bg_labels = nn.functional.one_hot(bg_gt_bin_labels, num_classes=2)

        bg_gt_bin_labels = bg_gt_bin_labels.reshape([1, 1, -1])
        loss_s1 = MyLoss.first_seg_loss(bg_pd_1, bg_weight_1, bg_gt_bin_labels)
        loss_s2 = MyLoss.second_seg_loss(bg_pd_2, bg_weight_1, bg_gt_bin_labels)

        # predict edge mask loss
        _, abs_ang = compute_polar_angles(batch.pos)
        gt_labels = batch.y.to(self.device, dtype=torch.long)
        radius = self.rmin + abs_ang * (self.rmax - self.rmin)
        edge_gt_bin_labels = compute_tae_gt_bin_labels_ang(batch.pos, gt_labels, r=radius)
        # gu.print_3d(gu.np_to_pcd_with_label(gu.torch_to_numpy(batch.pos).reshape(-1, 3), gu.torch_to_numpy(edge_gt_bin_labels)))

        onehot_tae_labels = nn.functional.one_hot(edge_gt_bin_labels, num_classes=2)
        edge_gt_bin_labels = edge_gt_bin_labels.reshape([1, 1, -1])
        loss_s3 = MyLoss.first_seg_loss(edge_pd_1, edge_weight_1, edge_gt_bin_labels)
        loss_s4 = MyLoss.second_seg_loss(edge_pd_2, edge_weight_1, edge_gt_bin_labels)

        edge_gt_bin_labels_wobg = compute_tte_gt_bin_labels(batch.pos, gt_labels, r=radius)
        onehot_tte_labels = nn.functional.one_hot(edge_gt_bin_labels_wobg, num_classes=2)

        edge_gt_bin_labels_wobg = edge_gt_bin_labels_wobg.reshape([1, 1, -1])
        loss_s5 = MyLoss.first_seg_loss(edge_wobg_pd_1, edge_wobg_weight_1, edge_gt_bin_labels_wobg)
        loss_s6 = MyLoss.second_seg_loss(edge_wobg_pd_2, edge_wobg_weight_1, edge_gt_bin_labels_wobg)

        #
        loss_l2 = self.regularization_losses()

        # loss = seg_loss + mat_diff_loss * 0.01
        lambda0, lambda1, lambda2 , lambda3= 2, 1, 1, 1
        loss = lambda1*(loss_s1 + loss_s2) + lambda2*(loss_s3 + loss_s4) + lambda3*(loss_s5 + loss_s6) + loss_l2

        # tooth-ginvigal
        bg_pd = torch.cat([-bg_pd_2, bg_pd_2], dim=1)
        dsc_tg = weighting_DSC(bg_pd.transpose(2,1), onehot_bg_labels, self.class_weights)
        sen_tg = weighting_SEN(bg_pd.transpose(2,1), onehot_bg_labels, self.class_weights)
        ppv_tg = weighting_PPV(bg_pd.transpose(2,1), onehot_bg_labels, self.class_weights)
        acc_tg = weighting_ACC(bg_pd.transpose(2,1), onehot_bg_labels, self.class_weights)
        iou_tg = weighting_IOU(bg_pd.transpose(2,1), onehot_bg_labels, self.class_weights)

        # tooth-all edge
        edge_pd = torch.cat([-edge_pd_2, edge_pd_2], dim=1)
        dsc_tae = weighting_DSC(edge_pd.transpose(2,1), onehot_tae_labels.unsqueeze(dim=0), self.class_weights)
        sen_tae = weighting_SEN(edge_pd.transpose(2,1), onehot_tae_labels.unsqueeze(dim=0), self.class_weights)
        ppv_tae = weighting_PPV(edge_pd.transpose(2,1), onehot_tae_labels.unsqueeze(dim=0), self.class_weights)
        acc_tae = weighting_ACC(edge_pd.transpose(2,1), onehot_tae_labels.unsqueeze(dim=0), self.class_weights)
        iou_tae = weighting_IOU(edge_pd.transpose(2,1), onehot_tae_labels.unsqueeze(dim=0), self.class_weights)

        # tooth-tooth edge
        tte_pd = torch.cat([-edge_wobg_pd_2,edge_wobg_pd_2],dim=1)
        dsc_tte = weighting_DSC(tte_pd.transpose(2,1), onehot_tte_labels.unsqueeze(dim=0), self.class_weights)
        sen_tte = weighting_SEN(tte_pd.transpose(2,1), onehot_tte_labels.unsqueeze(dim=0), self.class_weights)
        ppv_tte = weighting_PPV(tte_pd.transpose(2,1), onehot_tte_labels.unsqueeze(dim=0), self.class_weights)
        acc_tte = weighting_ACC(tte_pd.transpose(2,1), onehot_tte_labels.unsqueeze(dim=0), self.class_weights)
        iou_tte = weighting_IOU(tte_pd.transpose(2,1), onehot_tte_labels.unsqueeze(dim=0), self.class_weights)

        return (dsc_tae, loss, ppv_tae, sen_tae, acc_tae, iou_tae,
                loss_s1, loss_s2,loss_s3, loss_s4,loss_s5, loss_s6,
                alpha, beta,
                dsc_tte,  ppv_tte, sen_tte,acc_tte, iou_tte,
                dsc_tg, ppv_tg, sen_tg,  acc_tg, iou_tg)

    def training_step(self, batch, batch_idx):
        outputs = self._update_step_metrics(batch)
        dsc_tae, loss, ppv_tae, sen_tae, acc_tae, iou_tae = outputs[0], outputs[1], outputs[2], outputs[3], outputs[4], outputs[5]
        dsc_tte, ppv_tte, sen_tte, acc_tte, iou_tte = outputs[14], outputs[15], outputs[16], outputs[17], outputs[18]
        dsc_tg, ppv_tg, sen_tg, acc_tg, iou_tg = outputs[19], outputs[20], outputs[21], outputs[22], outputs[23]

        metrics = {
            'train_loss': loss,
            'train_dsc_tae': dsc_tae,
            'train_sen_tae': sen_tae,
            'train_ppv_tae': ppv_tae,
            'train_acc_tae': acc_tae,
            'train_iou_tae': iou_tae,
            'train_dsc_tte': dsc_tte,
            'train_sen_tte': sen_tte,
            'train_ppv_tte': ppv_tte,
            'train_acc_tte': acc_tte,
            'train_iou_tte': iou_tte,
            'train_dsc_tg': dsc_tg,
            'train_sen_tg': sen_tg,
            'train_ppv_tg': ppv_tg,
            'train_acc_tg': acc_tg,
            'train_iou_tg': iou_tg
        }

        self.log_dict(metrics)
        self.training_step_outputs.append(metrics)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self._update_step_metrics(batch)
        dsc_tae, loss, ppv_tae, sen_tae, acc_tae, iou_tae = outputs[0], outputs[1], outputs[2], outputs[3], outputs[4], outputs[5]
        dsc_tte, ppv_tte, sen_tte, acc_tte, iou_tte = outputs[14], outputs[15], outputs[16], outputs[17], outputs[18]
        dsc_tg, ppv_tg, sen_tg, acc_tg, iou_tg = outputs[19], outputs[20], outputs[21], outputs[22], outputs[23]
        loss_s1, loss_s2,loss_s3, loss_s4,loss_s5, loss_s6 = outputs[6], outputs[7], outputs[8], outputs[9], outputs[10], outputs[11]
        alpha, beta = outputs[12], outputs[13]
        def get_mesh_name(file_path):
            file_name = os.path.basename(file_path)
            folder_name = os.path.basename(os.path.dirname(file_path))
            return f"{file_name}_{folder_name}"

        mesh_name = get_mesh_name(batch.filename[0])

        print(f"epoch:{self.current_epoch}-{mesh_name}, "
              f"loss:{loss:.4f}=Lseg:L1:{loss_s1:.4f} + L2:{loss_s2:.4f}+ L3:{loss_s3:.4f} + L4:{loss_s4:.4f}+"
              f"L5:{loss_s5:.4f} + L6:{loss_s6:.4f}", end=' ')

        metrics = {
            'test_loss': loss,
            'test_dsc_tae': dsc_tae,
            'test_sen_tae': sen_tae,
            'test_ppv_tae': ppv_tae,
            'test_acc_tae': acc_tae,
            'test_iou_tae': iou_tae,
            'test_dsc_tte': dsc_tte,
            'test_sen_tte': sen_tte,
            'test_ppv_tte': ppv_tte,
            'test_acc_tte': acc_tte,
            'test_iou_tte': iou_tte,
            'test_dsc_tg': dsc_tg,
            'test_sen_tg': sen_tg,
            'test_ppv_tg': ppv_tg,
            'test_acc_tg': acc_tg,
            'test_iou_tg': iou_tg
        }

        # print(metrics)

        self.log_dict(metrics)
        self.validation_step_outputs.append(metrics)
        return metrics

    @abstractmethod
    def _print_epoch_info(self):
        if len(self.losses) > 0 and len(self.test_losses) > 0:
            print(f'*****\nEpoch: {self.current_epoch + 1}/{self.num_epochs}, '
                  f'loss: {self.losses[-1]}, \n'
                  f'dsc_tae: {self.mdsc_tae[-1]}, sen_tae: {self.msen_tae[-1]}, ppv_tae: {self.mppv_tae[-1]},acc_tae:{self.macc_tae[-1]},iou_tae:{self.miou_tae[-1]},\n'
                  f' dsc_tte:{self.mdsc_tte[-1]},sen_tte:{self.msen_tte[-1]}, ppv_tte: {self.mppv_tte[-1]},acc_tte:{self.macc_tte[-1]},iou_tte:{self.miou_tte[-1]},\n'
                  f' dsc_tg:{self.mdsc_tg[-1]},sen_tg:{self.msen_tg[-1]}, ppv_tg: {self.mppv_tg[-1]},acc_tg:{self.macc_tg[-1]},iou_tg:{self.miou_tg[-1]}'
                  f'\n'
                  f'test_loss: {self.test_losses[-1]}, \n'
                  f'test_dsc_tae: {self.test_mdsc_tae[-1]}, test_sen_tae: {self.test_msen_tae[-1]},test_ppv_tae: {self.test_mppv_tae[-1]},test_acc_tae:{self.test_macc_tae[-1]},test_iou_tae:{self.test_miou_tae[-1]},\n'
                  f' test_dsc_tte:{self.test_mdsc_tte[-1]},test_sen_tte:{self.test_msen_tte[-1]}, test_ppv_tte: {self.test_mppv_tte[-1]},test_acc_tte:{self.test_macc_tte[-1]},test_iou_tte:{self.test_miou_tte[-1]},\n'
                  f' test_dsc_tg:{self.test_mdsc_tg[-1]},test_sen_tg:{self.test_msen_tg[-1]}, test_ppv_tg: {self.test_mppv_tg[-1]},test_acc_tg:{self.test_macc_tg[-1]},test_iou_tg:{self.test_miou_tg[-1]}'
                  f'\n*****')

    def _save_epoch_csv(self):
        if len(self.losses) > 0 and len(self.test_losses) > 0:
            pd_dict = {'epoch':self.current_epoch + 1,'loss': self.losses[-1], 'DSC_tae': self.mdsc_tae[-1],
                       'SEN_tae': self.msen_tae[-1], 'PPV_tae': self.mppv_tae[-1],'ACC_tae': self.macc_tae[-1],'IOU_tae': self.miou_tae[-1],
                       'DSC_tte': self.mdsc_tte[-1],'SEN_tte': self.msen_tte[-1], 'PPV_tte': self.mppv_tte[-1],'ACC_tte': self.macc_tte[-1],'IOU_tte': self.miou_tte[-1],
                       'DSC_tg': self.mdsc_tg[-1],'SEN_tg': self.msen_tg[-1], 'PPV_tg': self.mppv_tg[-1],'ACC_tg': self.macc_tg[-1],'IOU_tg': self.miou_tg[-1],
                       'test_loss': self.test_losses[-1], 'test_DSC_tae': self.test_mdsc_tae[-1],'test_SEN_tae': self.test_msen_tae[-1],
                       'test_PPV_tae': self.test_mppv_tae[-1], 'test_ACC_tae': self.test_macc_tae[-1], 'test_IOU_tae': self.test_miou_tae[-1],
                       'test_DSC_tte': self.test_mdsc_tte[-1],'test_SEN_tte': self.test_msen_tte[-1], 'test_PPV_tte': self.test_mppv_tte[-1],'test_ACC_tte': self.test_macc_tte[-1], 'test_IOU_tte': self.test_miou_tte[-1],
                       'test_DSC_tg': self.test_mdsc_tg[-1],'test_SEN_tg': self.test_msen_tg[-1], 'test_PPV_tg': self.test_mppv_tg[-1],'test_ACC_tg': self.test_macc_tg[-1], 'test_IOU_tg': self.test_miou_tg[-1]}
            try:
                csv_file_path = os.path.join(self.model_path, self.train_loss_csv_file)
                file_exists = os.path.isfile(csv_file_path)
                # 传递 index 创建 DataFrame
                stat = pd.DataFrame(pd_dict, index=[0])
                # 追加模式写入 CSV 文件
                stat.to_csv(csv_file_path, mode='a', index=False, header=not file_exists)

            except Exception as e:
                print(f"Error saving CSV file: {e}")

    def on_train_epoch_end(self):

        print("Training epoch ended.")

        # 训练指标处理
        train_metric_keys = ['train_loss', 'train_dsc_tae', 'train_sen_tae', 'train_ppv_tae', 'train_acc_tae', 'train_iou_tae',
                            'train_dsc_tte', 'train_sen_tte', 'train_ppv_tte', 'train_acc_tte', 'train_iou_tte',
                            'train_dsc_tg', 'train_sen_tg', 'train_ppv_tg', 'train_acc_tg', 'train_iou_tg']
        train_metric_lists = [self.losses, self.mdsc_tae, self.msen_tae, self.mppv_tae, self.macc_tae,self.miou_tae,
                            self.mdsc_tte, self.msen_tte, self.mppv_tte, self.macc_tte,self.miou_tte,
                            self.mdsc_tg, self.msen_tg, self.mppv_tg, self.macc_tg,self.miou_tg]
        self._update_epoch_metrics(self.training_step_outputs, train_metric_keys, train_metric_lists)

        torch.cuda.empty_cache()
        print(f"Epoch end, GPU内存清理完成: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")


    def on_validation_epoch_end(self):
        # 验证指标处理
        val_metric_keys = ['test_loss', 'test_dsc_tae', 'test_sen_tae', 'test_ppv_tae', 'test_acc_tae', 'test_iou_tae',
                           'test_dsc_tte', 'test_sen_tte', 'test_ppv_tte', 'test_acc_tte', 'test_iou_tte',
                           'test_dsc_tg', 'test_sen_tg', 'test_ppv_tg', 'test_acc_tg', 'test_iou_tg']
        val_metric_lists = [self.test_losses, self.test_mdsc_tae, self.test_msen_tae, self.test_mppv_tae, self.test_macc_tae, self.test_miou_tae,
                            self.test_mdsc_tte, self.test_msen_tte, self.test_mppv_tte, self.test_macc_tte, self.test_miou_tte,
                            self.test_mdsc_tg, self.test_msen_tg, self.test_mppv_tg, self.test_macc_tg, self.test_miou_tg]

        self._update_epoch_metrics(self.validation_step_outputs, val_metric_keys, val_metric_lists)

        # 打印信息
        self._print_epoch_info()

        # save the checkpoint
        checkpoint_path = os.path.join(self.model_path, self.checkpoint_name)
        metrics_dict = {
            'losses': self.losses,
            'mdsc_tae': self.mdsc_tae,
            'msen_tae': self.msen_tae,
            'mppv_tae': self.mppv_tae,
            'macc_tae': self.macc_tae,
            'miou_tae': self.miou_tae,
            'mdsc_tte': self.mdsc_tte,
            'msen_tte': self.msen_tte,
            'mppv_tte': self.mppv_tte,
            'macc_tte': self.macc_tte,
            'miou_tte': self.miou_tte,
            'mdsc_tg': self.mdsc_tg,
            'msen_tg': self.msen_tg,
            'mppv_tg': self.mppv_tg,
            'macc_tg': self.macc_tg,
            'miou_tg': self.miou_tg,
            'test_losses': self.test_losses,
            'test_mdsc_tae': self.test_mdsc_tae,
            'test_msen_tae': self.test_msen_tae,
            'test_mppv_tae': self.test_mppv_tae,
            'test_macc_tae': self.test_macc_tae,
            'test_iou_tae': self.test_miou_tae,
            'test_msen_tte': self.test_msen_tte,
            'test_mppv_tte': self.test_mppv_tte,
            'test_macc_tte': self.test_macc_tte,
            'test_iou_tte': self.test_miou_tte,
            'test_msen_tg': self.test_msen_tg,
            'test_mppv_tg': self.test_mppv_tg,
            'test_macc_tg': self.test_macc_tg,
            'test_iou_tg': self.test_miou_tg
        }
        self._save_checkpoint(checkpoint_path,metrics_dict)

        # save the best model, modelcheckpoitn callbacks已经自动保存最优的几个模型了
        # self._save_best_model(metrics_dict)

        # save all losses and metrics data to csv file
        self._save_epoch_csv()


    def predict_step(self, batch, batch_idx):
        outputs = self._update_step_metrics(batch)
        dsc, loss, ppv, sen, acc, iou = outputs[0], outputs[1], outputs[2], outputs[3], outputs[4], outputs[5]

        filename = batch.filename
        pts_normals = batch.x
        faces = batch.faces
        gt_labels = batch.y.to(self.device, dtype=torch.long)

        outputs = self.model(batch)
        bg_pd_1, bg_weight_1, bg_pd_2, edge_pd_1, edge_weight_1, edge_pd_2 = outputs[0], outputs[1], outputs[2], outputs[3], outputs[4], outputs[5]
        edge_wobg_pd_1, edge_wobg_weight_1, edge_wobg_pd_2 = outputs[6],outputs[7],outputs[8]
        l0_points = outputs[9]
        pred_labels = []

        #####################################################
        # gts
        bg_gt_bin_labels = (gt_labels!=0)
        bg_gt_bin_labels = bg_gt_bin_labels.reshape([-1])

        # predict edge mask loss
        _, abs_ang = compute_polar_angles(batch.pos)
        radius = self.rmin + abs_ang * (self.rmax - self.rmin) # 可变半径

        edge_gt_bin_labels = compute_tae_gt_bin_labels_ang(batch.pos, gt_labels, r=radius)
        edge_gt_bin_labels = edge_gt_bin_labels.reshape([-1])

        tt_edge_gt_bin_labels = compute_tte_gt_bin_labels(batch.pos, gt_labels, r=radius)
        tt_edge_gt_bin_labels = tt_edge_gt_bin_labels.reshape([-1])

        ######################################################
        # base_name
        # 处理 mesh_file 生成 base_name
        import os
        base_name = os.path.basename(filename[0])
        dir_name = os.path.basename(os.path.dirname(filename[0]))
        base_name = f"{base_name}_{dir_name}"


        #####################################################

        metrics = {
            'predict_loss': loss,
            'predict_dsc': dsc,
            'predict_sen': sen,
            'predict_ppv': ppv,
            'predict_acc': acc,
            'predict_iou': iou,
            'pts_normals': pts_normals,
            'faces': faces,
            'gt_labels': gt_labels,
            'pred_labels': pred_labels,
            'l0_points':l0_points,
            'mesh_file': filename,
            'json_name': base_name,
            'bg_pd_1': bg_pd_1,
            'bg_weight_1': bg_weight_1,
            'bg_pd_2': bg_pd_2,
            'edge_pd_1': edge_pd_1,
            'edge_weight_1': edge_weight_1,
            'edge_pd_2': edge_pd_2,
            'edge_wobg_pd_1': edge_wobg_pd_1,
            'edge_wobg_weight_1': edge_wobg_weight_1,
            'edge_wobg_pd_2': edge_wobg_pd_2,
            'bg_gt_bin_labels':bg_gt_bin_labels,
            'edge_gt_bin_labels': edge_gt_bin_labels,
            'tt_edge_gt_bin_labels': tt_edge_gt_bin_labels,
            'predictions': []
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
        logger_path=r'../../tb_logs/',
        exp_name=r'exp_' + current_file_name,
        ckpt_path=r'../../checkpoints/',
        ckpt_name=r'latest_checkpoint.tar',
        results_path=r'../../results/',
        with_base_lower_path=r"E:\mesh_datasets\3DTeeth\Teeth3DS_train_test_split\with_base_lower.txt",
        wo_base_lower_path=r"E:\mesh_datasets\3DTeeth\Teeth3DS_train_test_split\wo_base_lower.txt",
        split_600_path=r"E:\mesh_datasets\3DTeeth\Teeth3DS_train_test_split\split_600_lower.txt",
        rmin=0.03,
        rmax=0.03,
        num_classes=2,
        num_neighbor=32,
        num_epochs=45,
        num_workers=16,
        sample_ratio = 0.1,
        train_val_ratio = 1.0,
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
        'rmin':args.rmin,
        'rmax':args.rmax,
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
    model_name_path = '{}/{}classes_{}epochs_{}ncells_{}ratio_{}_rmin{}_rmax_{}lr'
    model_name_path = model_name_path.format(args.version, args.num_classes, args.num_epochs,
                                             args.ncells,
                                             args.sample_ratio,
                                             args.rmin, args.rmax, args.lr)
    results_path = os.path.join(args.results_path, model_name_path)
    return results_path

def do_train():
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
    train_ds, val_ds, test_ds = load_DBGA_dataset_with_tbase(sample_ratio=args.sample_ratio,
                                                             train_val_ratio=1.0,
                                                             comb_name='comb_PointnetTrans',
                                                             is_full = True,
                                                             retain_file_path=None,
                                                             rseed=rseed)
    # train_ds, _, test_ds = load_datasets(sample_ratio=args.sample_ratio,comb_name='comb_qslim_5000',rseed=rseed)

    train_loader = GeoDataLoader(dataset=train_ds, batch_size=args.train_batch_size, shuffle=True,
                                 num_workers=args.num_workers,pin_memory=True,persistent_workers=True)
    test_loader = GeoDataLoader(dataset=test_ds, batch_size=args.val_batch_size, shuffle=False,
                                num_workers=args.num_workers,pin_memory=True,persistent_workers=True)

    # model
    config = get_cfg_from_args(args)
    model = TripleBinClsModule(config)


    model,latest_checkpoint_path = load_latest_ckpts(checkpoint_path, model)
    # trainer
    # trainer = pl.Trainer(max_epochs=num_epochs, accelerator='auto', num_sanity_val_steps=0)
    trainer = pl.Trainer(max_epochs=args.num_epochs,
                         logger=tb_logger,
                         log_every_n_steps=20,
                         check_val_every_n_epoch=1,
                         precision='16-mixed',
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


def save_to_topo_clu_json(args, predictions):
    # 保存结果
    results_path = get_results_path(args)
    results_path = results_path + '_topo_clu'
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

def print_pred_metrics(predictions):
    # 检查预测结果列表是否为空
    if not predictions:
        print("No predictions to display.")
        return

    # 初始化统计变量
    total_instances = len(predictions)
    sum_dsc = sum_ppv = sum_sen = sum_acc = sum_iou = 0.0

    # 初始化最佳和最差指标记录
    best_metrics = {
        'dsc': {'value': -float('inf'), 'filename': None},
        'ppv': {'value': -float('inf'), 'filename': None},
        'sen': {'value': -float('inf'), 'filename': None},
        'acc': {'value': -float('inf'), 'filename': None},
        'iou': {'value': -float('inf'), 'filename': None}
    }

    worst_metrics = {
        'dsc': {'value': float('inf'), 'filename': None},
        'ppv': {'value': float('inf'), 'filename': None},
        'sen': {'value': float('inf'), 'filename': None},
        'acc': {'value': float('inf'), 'filename': None},
        'iou': {'value': float('inf'), 'filename': None}
    }

    # 打印表头
    print("\nInstance Metrics:")
    print(f"{'Filename':<50}|{'DSC':<8}|{'Loss':<8}|{'PPV':<8}|{'Sen':<8}|{'Acc':<8}|{'IoU':<8}")
    print("-" * 100)

    # 遍历每个预测结果并打印指标
    for prediction in predictions:
        # 提取文件名
        filename = prediction['mesh_file']
        if isinstance(filename, list):
            if len(filename) > 0:
                filename = filename[0]
            else:
                filename = "Unknown"

        # 提取指标值
        dsc = prediction.get('predict_dsc', 0.0)
        loss = prediction.get('predict_loss', 0.0)
        ppv = prediction.get('predict_ppv', 0.0)
        sen = prediction.get('predict_sen', 0.0)
        acc = prediction.get('predict_acc', 0.0)
        iou = prediction.get('predict_iou', 0.0)  # 注意这里使用了正确的键名

        # 打印当前实例的指标
        print(f"{filename:<50}|{dsc:.4f}|{loss:.4f}|{ppv:.4f}|{sen:.4f}|{acc:.4f}|{iou:.4f}")

        # 累加指标值用于计算平均值
        sum_dsc += dsc
        sum_ppv += ppv
        sum_sen += sen
        sum_acc += acc
        sum_iou += iou

        # 更新最佳和最差指标记录
        for metric in ['dsc', 'ppv', 'sen', 'acc', 'iou']:
            value_key = f'predict_{metric}'
            value = prediction.get(value_key, 0.0)

            # 更新最佳指标
            if value > best_metrics[metric]['value']:
                best_metrics[metric]['value'] = value
                best_metrics[metric]['filename'] = filename

            # 更新最差指标
            if value < worst_metrics[metric]['value']:
                worst_metrics[metric]['value'] = value
                worst_metrics[metric]['filename'] = filename

    # 计算平均指标
    avg_dsc = sum_dsc / total_instances
    avg_ppv = sum_ppv / total_instances
    avg_sen = sum_sen / total_instances
    avg_acc = sum_acc / total_instances
    avg_iou = sum_iou / total_instances

    # 打印平均指标
    print("\nAverage Metrics:")
    print(f"mDSC: {avg_dsc:.4f}, mPPV: {avg_ppv:.4f}, mSen: {avg_sen:.4f}, mAcc: {avg_acc:.4f}, mIoU: {avg_iou:.4f}")

    # 打印最佳和最差指标
    print("\nBest Metrics:")
    for metric, data in best_metrics.items():
        print(f"Best {metric.upper()}: {data['value']:.4f} from {data['filename']}")

    print("\nWorst Metrics:")
    for metric, data in worst_metrics.items():
        print(f"Worst {metric.upper()}: {data['value']:.4f} from {data['filename']}")


import os
import json

# @lru_cache(maxsize=256)
def pred_from_json(json_path, json_name):
    """
    读取指定路径下特定名称的 JSON 文件。
    参数:
    json_path (str): JSON 文件所在的目录路径。
    json_name (str): JSON 文件的基础名称（不包含扩展名）。
    返回:
    tuple: 包含处理后的 JSON 数据字典和预测准确率的元组。
           如果文件读取失败或无预测准确率，预测准确率为 None。
    """
    # 构建完整的 JSON 文件路径
    file_path = os.path.join(json_path, f"{json_name}.json")
    json_data = {}
    pred_acc = None

    try:
        # 读取 JSON 文件内容
        with open(file_path, 'r') as f:
            json_data = json.load(f)

        # 处理 JSON 数据，与 MATLAB 代码保持一致
        # 假设 mesh_file 是嵌套列表，提取第一个元素
        if isinstance(json_data.get('mesh_file'), list) and len(json_data['mesh_file']) > 0:
            json_data['mesh_file'] = json_data['mesh_file'][0][0]

        # 添加 JSON 文件名
        json_data['json_name'] = os.path.basename(file_path)

        # 提取预测准确率
        if 'predict_acc' in json_data:
            pred_acc = json_data['predict_acc']

        # print(f'成功读取文件: {file_path}')

    except Exception as e:
        print(f'读取文件 {file_path} 时出错: {str(e)}')

    return json_data, pred_acc



def do_pred():
    setup_seed(1111)
    # 全局参数
    args = get_args()
    #logger
    tb_logger = init_logger(args)
    #checkpoints
    checkpoint_callback, checkpoint_path = init_ckpts(args)

    # args.sample_ratio = 1.0
    sample_ratio = 1.0
    _, _, predict_ds = load_DBGA_dataset_with_tbase(sample_ratio=sample_ratio, train_val_ratio=1.0, comb_name='comb_PointnetTrans',
                                                             is_full=True, retain_file_path=None, rseed=42)
    predict_loader = GeoDataLoader(dataset=predict_ds, batch_size=1, shuffle=False, num_workers=args.num_workers,
                                   pin_memory=True)

    config = get_cfg_from_args(args)
    model = TripleBinClsModule(config)

    # 检查是否存在检查点文件
    model,latest_checkpoint_path = load_latest_ckpts(checkpoint_path, model)

    trainer = pl.Trainer()
    # 进行预测
    if 'latest_checkpoint_path' in locals():
        print(latest_checkpoint_path, locals())
        predictions = trainer.predict(model, dataloaders=predict_loader, ckpt_path=latest_checkpoint_path)
    else:
        predictions = trainer.predict(model, dataloaders=predict_loader)

    # 保存为json
    save_to_json(args, predictions)
    # 打印信息
    print_pred_metrics(predictions)

def test_get_model():
    import torch  # Add this import statement

    model = TripleBinClsNet(num_classes=15, num_feat=6, k=16)
    model.cuda()
    xyz = torch.rand(10000,6).cuda()
    data = Data(x=xyz, pos=xyz[:,:3])

    # output is B, C, N order
    for item in model(data):
        print(item.shape)

    # -- coding: utf-8 --
    from thop import profile

    flops, params = profile(model, inputs=((data),))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f G, params: %.2f M' % (flops / 1.0e9, params / 1.0e6))


def save_topo_cluster_results():
    """
    保存拓扑聚类结果
    """
    setup_seed(1111)
    args = get_args()
    tb_logger = init_logger(args)
    checkpoint_callback, checkpoint_path = init_ckpts(args)
    # args.sample_ratio = 1.0
    sample_ratio = 1.0
    train_ds, _, predict_ds = load_DBGA_dataset_with_tbase(sample_ratio=sample_ratio, train_val_ratio=1.0,
                                                    comb_name='comb_PointnetTrans',
                                                    is_full=True, retain_file_path=None, rseed=42)
    predict_loader = GeoDataLoader(dataset=train_ds, batch_size=1, shuffle=False, num_workers=args.num_workers,
                                   pin_memory=True)
    config = get_cfg_from_args(args)
    model = TripleBinClsModule(config)
    # 检查是否存在检查点文件
    model, latest_checkpoint_path = load_latest_ckpts(checkpoint_path, model)
    trainer = pl.Trainer()
    # 进行预测
    if 'latest_checkpoint_path' in locals():
        print(latest_checkpoint_path, locals())
        predictions = trainer.predict(model, dataloaders=predict_loader, ckpt_path=latest_checkpoint_path)
    else:
        predictions = trainer.predict(model, dataloaders=predict_loader)
    for id in range(0, len(predictions)):
        json_obj = get_json_obj(predictions, id)

        opts = {
            'alpha': 0.2,
            'max_iter': 5,
            'iter_thred_len': 0.01,
            'field_bound': 0.03,
            'field_sigma': 0.1,
            'num_classes': 15,
            'crop_num': 1280
        }

        ###############################################################
        print(f'id:{id},basename:{json_obj.json_name}') # 拓扑聚类耗时: 1.4425580501556396 秒
        start_time = time.time()
        outs = topo_cluster(json_obj, opts)
        end_time = time.time()
        print(f'cluster num:{outs[0]},拓扑聚类耗时: {end_time - start_time} 秒') # 拓扑聚类耗时: 1.4425580501556396 秒

        ##############################################################

        clu_num, clu_cen, clu_counts, clu_gt_map, counts_mat, gt_clu_num_prop, sep_teeth_mask, clu_labels_all = outs[0:8]

        pred_tmp = {
            'clu_labels_all': torch.from_numpy(clu_labels_all),
            'clu_num': torch.tensor(clu_num),
            'clu_cen': torch.from_numpy(clu_cen),
            'clu_counts': torch.from_numpy(clu_counts),
            'clu_gt_map': torch.from_numpy(clu_gt_map),
            'counts_mat': torch.from_numpy(counts_mat),
            'gt_clu_num_prop': torch.from_numpy(gt_clu_num_prop),
            'sep_teeth_mask': torch.from_numpy(sep_teeth_mask),
            'mesh_file': predictions[id]['mesh_file'],
            # 'l0_points': predictions[id]['l0_points']
        }
        predictions[id] = pred_tmp

    # 要求数据为torch tensor
    save_to_topo_clu_json(args, predictions)
    # 打印信息
    print_pred_metrics(predictions)


def test_read_json_file():
    json_path = r'C:\Users\win10\DualMesh\pynb\teeth3ds\model\results\exp_TripleBinClsNet\2classes_30epochs_10000ncells_0.1ratio_0.04_rmin0.04_rmax_0.001lr'
    json_name = r'013NUWYR_lower'
    #####################################################
    # 测试读取文件耗时
    start_time = time.time()
    json_data, pred_acc = pred_from_json(json_path, json_name)
    end_time = time.time()
    print(f'读取文件耗时: {end_time - start_time} 秒')
    #####################################################
    print(json_data.keys())


if __name__ == '__main__':
    # 示例使用
    # test_get_model()
    do_train()
    # do_pred()
    # test4_cent()
    # save_topo_cluster_results()
    # test_read_json_file()
