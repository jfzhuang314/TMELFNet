"""
@File    :   skip_mask.py
这一版本主要简化了输入特征的提取，仅仅使用两个edgeconv层提取特征；
"""
import argparse
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
from model._elf.topo_clustering import get_json_obj, topo_cluster
import os
import json
import yaml
import hydra
from omegaconf import DictConfig, OmegaConf


class BinSegLoss:
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
    def __init__(self, input_dim, radii=[0.03, 0.10]):
        super(ShallowFA, self).__init__()
        self.input_dim = input_dim
        self.radii = radii
        # 定义 EdgeConv 层
        self.conv1 = EdgeConv(Seq(Lin(self.input_dim * 2, 32), ReLU()))
        self.conv2 = EdgeConv(Seq(Lin(32 * 2, 32), ReLU()))

    def forward(self, batch):
        data = batch

        x0 = data.x.float()
        edge_index1 = radius_graph(data.pos, r=self.radii[0],max_num_neighbors=32).to(data.pos.device, dtype=torch.long)
        edge_index2 = radius_graph(data.pos, r=self.radii[1],max_num_neighbors=32).to(data.pos.device, dtype=torch.long)

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

        # 使用ModuleDict封装分支网络组件
        self.tae_branch = torch.nn.ModuleDict({
            'pd_mask_1': torch.nn.Linear(32, 2),
            'pd_mask_1_softmax': torch.nn.Softmax(dim=1),
            'wt_mask_1': torch.nn.Linear(32, 1),
            'pd_mask_2': torch.nn.Linear(32, 1)
        })

        self.tte_branch = torch.nn.ModuleDict({
            'pd_mask_1': torch.nn.Linear(32, 2),
            'pd_mask_1_softmax': torch.nn.Softmax(dim=1),
            'wt_mask_1': torch.nn.Linear(32, 1),
            'pd_mask_2': torch.nn.Linear(32, 1)
        })

    def forward(self, batch, edge_index_list):
        data = batch

        points1 = data.float()
        B, C, N = points1.shape
        # 展平 points1 以匹配边索引
        points1_flat = points1.transpose(1, 2).reshape(-1, points1.size(1))  # [B * N, C_total]

        # 步骤 3: 执行卷积操作
        x1, x1_F1, x1_F2, _ = self.block1(points1_flat, edge_index_list)

        # 恢复批量维度
        reshape_to_batch = lambda x: x.reshape(B, N, -1).transpose(1, 2)

        # 定义处理分支的辅助函数
        def forward_branch(branch_module, block_module, points_flat, x_F2):
            # forward
            pd_1 = branch_module['pd_mask_1_softmax'](branch_module['pd_mask_1'](x_F2))
            weight_1 = branch_module['wt_mask_1'](x_F2)
            points_next_flat = torch.cat([points_flat, pd_1], dim=1)
            _, _, x_next_F2, _ = block_module(points_next_flat, edge_index_list)
            pd_2 = branch_module['pd_mask_2'](x_next_F2)

            # reshape to batch
            pd_1 = reshape_to_batch(pd_1)
            weight_1 = reshape_to_batch(weight_1)
            pd_2 = reshape_to_batch(pd_2)

            return pd_1, weight_1, pd_2


        # 3a 模块1 - 使用tae分支
        tae_output = forward_branch(self.tae_branch, self.block2, points1_flat, x1_F2)
        # 3b 模块2 - 使用tte分支
        tte_output = forward_branch(self.tte_branch, self.block3, points1_flat, x1_F2)

        return tae_output, tte_output


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

        xyz_bg,xyz_edge = x2,x2
        tg_outputs= self.bg_pdmask_block(xyz_bg,edge_index_list)
        tae_outputs, tte_outputs = self.edge_pdmask_block(xyz_edge,edge_index_list)

        return tg_outputs, tae_outputs, tte_outputs, x_1

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
        self.rmin = config.get('rmin')
        self.rmax = config.get('rmax')

    def _init_metrics(self, num_classes):
        # 使用字典管理训练指标
        self.train_metrics = {
            'losses': [],
            'tae': {'mdsc': [], 'msen': [], 'mppv': [], 'macc': [], 'miou': [] },
            'tte': {'mdsc': [], 'msen': [], 'mppv': [], 'macc': [], 'miou': [] },
            'tg':  {'mdsc': [], 'msen': [], 'mppv': [], 'macc': [], 'miou': [] } }

        # 使用字典管理验证指标
        self.val_metrics = {
            'test_losses': [],
            'tae': {'mdsc': [], 'msen': [], 'mppv': [], 'macc': [], 'miou': []},
            'tte': {'mdsc': [], 'msen': [], 'mppv': [], 'macc': [], 'miou': []},
            'tg':  {'mdsc': [], 'msen': [], 'mppv': [], 'macc': [], 'miou': []} }

        self.best_test_dsc = 0.0
        self.class_weights = torch.ones(num_classes, dtype=torch.float)

        # 输出信息
        self.training_step_outputs = []
        self.validation_step_outputs = []


    def _compute_metrics(self, pred, labels, class_weights):
        """
        计算各种评估指标
        Args:
            pred: 预测值,BNC格式
            labels: 真实标签,BNC格式
            class_weights: 类别权重
        Returns:
            (dsc, sen, ppv, acc, iou): 各种评估指标
        """
        dsc = weighting_DSC(pred.transpose(2, 1), labels, class_weights)
        sen = weighting_SEN(pred.transpose(2, 1), labels, class_weights)
        ppv = weighting_PPV(pred.transpose(2, 1), labels, class_weights)
        acc = weighting_ACC(pred.transpose(2, 1), labels, class_weights)
        iou = weighting_IOU(pred.transpose(2, 1), labels, class_weights)
        return dsc, sen, ppv, acc, iou

    def _update_step_metrics(self, batch):
        # batch = self.normalize(batch)

        labels = batch.y.to(self.device, dtype=torch.long)
        labels = labels.reshape([-1,1])
        # 调用网络模型获取结果
        outputs = self.model(batch)
        # 解包新的返回值格式
        bg_outputs, edge_outputs, tte_outputs, x_1 = outputs
        bg_pd_1, bg_weight_1, bg_pd_2 = bg_outputs
        edge_pd_1, edge_weight_1, edge_pd_2 = edge_outputs
        tte_pd_1, tte_weight_1, tte_pd_2 = tte_outputs

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
        loss_s1 = BinSegLoss.first_seg_loss(bg_pd_1, bg_weight_1, bg_gt_bin_labels)
        loss_s2 = BinSegLoss.second_seg_loss(bg_pd_2, bg_weight_1, bg_gt_bin_labels)

        # predict edge mask loss
        _, abs_ang = compute_polar_angles(batch.pos)
        gt_labels = batch.y.to(self.device, dtype=torch.long)
        radius = self.rmin + abs_ang * (self.rmax - self.rmin)
        edge_gt_bin_labels = compute_tae_gt_bin_labels_ang(batch.pos, gt_labels, r=radius)
        # gu.print_3d(gu.np_to_pcd_with_label(gu.torch_to_numpy(batch.pos).reshape(-1, 3), gu.torch_to_numpy(edge_gt_bin_labels)))

        onehot_tae_labels = nn.functional.one_hot(edge_gt_bin_labels, num_classes=2)
        edge_gt_bin_labels = edge_gt_bin_labels.reshape([1, 1, -1])
        loss_s3 = BinSegLoss.first_seg_loss(edge_pd_1, edge_weight_1, edge_gt_bin_labels)
        loss_s4 = BinSegLoss.second_seg_loss(edge_pd_2, edge_weight_1, edge_gt_bin_labels)

        edge_gt_bin_labels_wobg = compute_tte_gt_bin_labels(batch.pos, gt_labels, r=radius)
        onehot_tte_labels = nn.functional.one_hot(edge_gt_bin_labels_wobg, num_classes=2)

        edge_gt_bin_labels_wobg = edge_gt_bin_labels_wobg.reshape([1, 1, -1])
        loss_s5 = BinSegLoss.first_seg_loss(tte_pd_1, tte_weight_1, edge_gt_bin_labels_wobg)
        loss_s6 = BinSegLoss.second_seg_loss(tte_pd_2, tte_weight_1, edge_gt_bin_labels_wobg)

        #
        loss_l2 = self.regularization_losses()

        # loss = seg_loss + mat_diff_loss * 0.01
        lambda0, lambda1, lambda2 , lambda3= 2, 1, 1, 1
        loss = lambda1*(loss_s1 + loss_s2) + lambda2*(loss_s3 + loss_s4) + lambda3*(loss_s5 + loss_s6) + loss_l2

        # 使用封装的函数计算各项指标
        # tg
        bg_pd = torch.cat([-bg_pd_2, bg_pd_2], dim=1)
        dsc_tg, sen_tg, ppv_tg, acc_tg, iou_tg = self._compute_metrics(bg_pd, onehot_bg_labels, self.class_weights )
        # tae
        edge_pd = torch.cat([-edge_pd_2, edge_pd_2], dim=1)
        dsc_tae, sen_tae, ppv_tae, acc_tae, iou_tae = self._compute_metrics(edge_pd,onehot_tae_labels.unsqueeze(dim=0), self.class_weights)
        # tte
        tte_pd = torch.cat([-tte_pd_2, tte_pd_2], dim=1)
        dsc_tte, sen_tte, ppv_tte, acc_tte, iou_tte = self._compute_metrics(tte_pd,onehot_tte_labels.unsqueeze(dim=0), self.class_weights)

        return (loss,
                (dsc_tg, ppv_tg, sen_tg,  acc_tg, iou_tg),
                (dsc_tae, ppv_tae, sen_tae, acc_tae, iou_tae),
                (dsc_tte,  ppv_tte, sen_tte,acc_tte, iou_tte),
                (loss_s1, loss_s2,loss_s3, loss_s4,loss_s5, loss_s6),
                (alpha, beta))



    def training_step(self, batch, batch_idx):
        outputs = self._update_step_metrics(batch)

        loss = outputs[0]
        dsc_tg, ppv_tg, sen_tg, acc_tg, iou_tg = outputs[1]
        dsc_tae, ppv_tae, sen_tae, acc_tae, iou_tae = outputs[2]
        dsc_tte, ppv_tte, sen_tte, acc_tte, iou_tte = outputs[3]
        loss_s1, loss_s2, loss_s3, loss_s4, loss_s5, loss_s6 = outputs[4]
        alpha, beta = outputs[5]

        metrics = {
            'train/loss': loss,
            'train/tae/dsc': dsc_tae,
            'train/tae/sen': sen_tae,
            'train/tae/ppv': ppv_tae,
            'train/tae/acc': acc_tae,
            'train/tae/iou': iou_tae,
            'train/tte/dsc': dsc_tte,
            'train/tte/sen': sen_tte,
            'train/tte/ppv': ppv_tte,
            'train/tte/acc': acc_tte,
            'train/tte/iou': iou_tte,
            'train/tg/dsc': dsc_tg,
            'train/tg/sen': sen_tg,
            'train/tg/ppv': ppv_tg,
            'train/tg/acc': acc_tg,
            'train/tg/iou': iou_tg
        }

        self.log_dict(metrics)
        self.training_step_outputs.append(metrics)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self._update_step_metrics(batch)

        loss = outputs[0]
        dsc_tg, ppv_tg, sen_tg, acc_tg, iou_tg = outputs[1]
        dsc_tae, ppv_tae, sen_tae, acc_tae, iou_tae = outputs[2]
        dsc_tte, ppv_tte, sen_tte, acc_tte, iou_tte = outputs[3]
        loss_s1, loss_s2, loss_s3, loss_s4, loss_s5, loss_s6 = outputs[4]
        alpha, beta = outputs[5]

        def get_mesh_name(file_path):
            file_name = os.path.basename(file_path)
            folder_name = os.path.basename(os.path.dirname(file_path))
            return f"{file_name}_{folder_name}"

        mesh_name = get_mesh_name(batch.filename[0])

        print(f"epoch:{self.current_epoch}-{mesh_name}, "
              f"loss:{loss:.4f}=Lseg:L1:{loss_s1:.4f} + L2:{loss_s2:.4f}+ L3:{loss_s3:.4f} + L4:{loss_s4:.4f}+"
              f"L5:{loss_s5:.4f} + L6:{loss_s6:.4f}", end=' ')

        metrics = {
            'test/loss': loss,
            'test/tae/dsc': dsc_tae,
            'test/tae/sen': sen_tae,
            'test/tae/ppv': ppv_tae,
            'test/tae/acc': acc_tae,
            'test/tae/iou': iou_tae,
            'test/tte/dsc': dsc_tte,
            'test/tte/sen': sen_tte,
            'test/tte/ppv': ppv_tte,
            'test/tte/acc': acc_tte,
            'test/tte/iou': iou_tte,
            'test/tg/dsc': dsc_tg,
            'test/tg/sen': sen_tg,
            'test/tg/ppv': ppv_tg,
            'test/tg/acc': acc_tg,
            'test/tg/iou': iou_tg
        }

        # print(metrics)

        self.log_dict(metrics)
        self.validation_step_outputs.append(metrics)
        return metrics

    @abstractmethod
    def _print_epoch_info(self):
        # 从train_metrics中获取最新指标
        if len(self.train_metrics['losses']) > 0 and len(self.val_metrics['test_losses']) > 0:
            print(f'*****\nEpoch: {self.current_epoch + 1}/{self.num_epochs}, '
                  f'loss: {self.train_metrics["losses"][-1]}, \n'
                  f'dsc_tae: {self.train_metrics["tae"]["mdsc"][-1]}, sen_tae: {self.train_metrics["tae"]["msen"][-1]}, '
                  f'ppv_tae: {self.train_metrics["tae"]["mppv"][-1]},acc_tae:{self.train_metrics["tae"]["macc"][-1]},'
                  f'iou_tae:{self.train_metrics["tae"]["miou"][-1]},\n'
                  f' dsc_tte:{self.train_metrics["tte"]["mdsc"][-1]},sen_tte:{self.train_metrics["tte"]["msen"][-1]}, '
                  f'ppv_tte: {self.train_metrics["tte"]["mppv"][-1]},acc_tte:{self.train_metrics["tte"]["macc"][-1]},'
                  f'iou_tte:{self.train_metrics["tte"]["miou"][-1]},\n'
                  f' dsc_tg:{self.train_metrics["tg"]["mdsc"][-1]},sen_tg:{self.train_metrics["tg"]["msen"][-1]}, '
                  f'ppv_tg: {self.train_metrics["tg"]["mppv"][-1]},acc_tg:{self.train_metrics["tg"]["macc"][-1]},'
                  f'iou_tg:{self.train_metrics["tg"]["miou"][-1]}'
                  f'\n'
                  f'test_loss: {self.val_metrics["test_losses"][-1]}, \n'
                  f'test_dsc_tae: {self.val_metrics["tae"]["mdsc"][-1]}, test_sen_tae: {self.val_metrics["tae"]["msen"][-1]},'
                  f'test_ppv_tae: {self.val_metrics["tae"]["mppv"][-1]},test_acc_tae:{self.val_metrics["tae"]["macc"][-1]},'
                  f'test_iou_tae:{self.val_metrics["tae"]["miou"][-1]},\n'
                  f' test_dsc_tte:{self.val_metrics["tte"]["mdsc"][-1]},test_sen_tte:{self.val_metrics["tte"]["msen"][-1]}, '
                  f'test_ppv_tte: {self.val_metrics["tte"]["mppv"][-1]},test_acc_tte:{self.val_metrics["tte"]["macc"][-1]},'
                  f'test_iou_tte:{self.val_metrics["tte"]["miou"][-1]},\n'
                  f' test_dsc_tg:{self.val_metrics["tg"]["mdsc"][-1]},test_sen_tg:{self.val_metrics["tg"]["msen"][-1]}, '
                  f'test_ppv_tg: {self.val_metrics["tg"]["mppv"][-1]},test_acc_tg:{self.val_metrics["tg"]["macc"][-1]},'
                  f'test_iou_tg:{self.val_metrics["tg"]["miou"][-1]}'
                  f'\n*****')

    def _prepare_epoch_metrics(self):
        """准备epoch指标数据，供打印和保存CSV使用"""
        if len(self.train_metrics['losses']) > 0 and len(self.val_metrics['test_losses']) > 0:
            pd_dict = {
                'epoch': self.current_epoch + 1,
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

    @abstractmethod
    def _print_epoch_info(self):
        # 从train_metrics中获取最新指标
        pd_dict = self._prepare_epoch_metrics()
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

    def _save_epoch_csv(self):
        pd_dict = self._prepare_epoch_metrics()
        if pd_dict:
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
        # 更新训练指标字典
        if self.training_step_outputs:
            # 计算平均损失
            avg_loss = torch.stack([out['train/loss'] for out in self.training_step_outputs]).mean().item()
            self.train_metrics['losses'].append(avg_loss)

            # 计算各分支的平均指标
            for branch in ['tae', 'tte', 'tg']:
                for metric in ['dsc', 'sen', 'ppv', 'acc', 'iou']:
                    key = f'train/{branch}/{metric}'
                    values = [out[key] for out in self.training_step_outputs if key in out]
                    if values:
                        avg_value = torch.stack(values).mean().item()
                        self.train_metrics[branch][f'm{metric}'].append(avg_value)
                    else:
                        self.train_metrics[branch][f'm{metric}'].append(0.0)

        # 清理训练输出
        self.training_step_outputs.clear()
        torch.cuda.empty_cache()
        print(f"Epoch end, GPU内存清理完成: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")

    def on_validation_epoch_end(self):
        # 验证指标处理
        # 更新验证指标字典
        if self.validation_step_outputs:
            # 计算平均损失
            avg_loss = torch.stack([out['test/loss'] for out in self.validation_step_outputs]).mean().item()
            self.val_metrics['test_losses'].append(avg_loss)

            # 计算各分支的平均指标
            for branch in ['tae', 'tte', 'tg']:
                for metric in ['dsc', 'sen', 'ppv', 'acc', 'iou']:
                    key = f'test/{branch}/{metric}'
                    values = [out[key] for out in self.validation_step_outputs if key in out]
                    if values:
                        avg_value = torch.stack(values).mean().item()
                        self.val_metrics[branch][f'm{metric}'].append(avg_value)
                    else:
                        self.val_metrics[branch][f'm{metric}'].append(0.0)

        # 清理验证输出
        self.validation_step_outputs.clear()

        # 打印信息
        self._print_epoch_info()
        # save all losses and metrics data to csv file
        self._save_epoch_csv()
    def predict_step(self, batch, batch_idx):

        metric_outputs = self._update_step_metrics(batch)

        loss = metric_outputs[0]
        # dsc_tg, ppv_tg, sen_tg, acc_tg, iou_tg = metric_outputs[1]
        # dsc_tae, ppv_tae, sen_tae, acc_tae, iou_tae = metric_outputs[2]
        # dsc_tte, ppv_tte, sen_tte, acc_tte, iou_tte = metric_outputs[3]
        # loss_s1, loss_s2, loss_s3, loss_s4, loss_s5, loss_s6 = metric_outputs[4]
        # alpha, beta = metric_outputs[5]

        dsc, ppv, sen, acc, iou = metric_outputs[1]

        filename = batch.filename
        pts_normals = batch.x
        faces = batch.faces
        gt_labels = batch.y.to(self.device, dtype=torch.long)

        # 解包新的返回值格式
        outputs = self.model(batch)
        bg_outputs, edge_outputs, tte_outputs, x_1 = outputs
        bg_pd_1, bg_weight_1, bg_pd_2 = bg_outputs
        edge_pd_1, edge_weight_1, edge_pd_2 = edge_outputs
        tte_pd_1, tte_weight_1, tte_pd_2 = tte_outputs
        l0_points = x_1
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
            'tte_pd_1': tte_pd_1,
            'tte_weight_1': tte_weight_1,
            'tte_pd_2': tte_pd_2,
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

############################################################################
## Training Teitsng Prediction
############################################################################


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_training_args(config_path: str = "configs/exp", config_name: str = "ELFNet"):
    """可在普通函数中调用的配置加载函数"""
    # with hydra.initialize_config_dir(config_dir=config_path, version_base=None):
    # 将相对路径转换为绝对路径
    config_path = os.path.abspath(config_path)
    with hydra.initialize_config_dir(config_dir=config_path, version_base=None):
        cfg = hydra.compose(config_name=config_name)
        # config_dict = OmegaConf.to_container(cfg, resolve=True)

        config_dict = {
            **OmegaConf.to_container(cfg.dataset),
            **OmegaConf.to_container(cfg.model),
            **OmegaConf.to_container(cfg.training),
            **OmegaConf.to_container(cfg.paths)
        }

        current_file_name = os.path.splitext(os.path.basename(__file__))[0]
        config_dict['exp_name'] = \
            f'exp_{current_file_name}/ratio_{config_dict["sample_ratio"]}_rmin_{config_dict["rmin"]}_rmax_{config_dict["rmax"]}'

        args = argparse.Namespace(**config_dict)

        # 打印返回的内容
        print("create_training_args 返回的内容:")
        print("-" * 50)
        for attr_name in sorted(vars(args)):
            attr_value = getattr(args, attr_name)
            print(f"{attr_name}: {attr_value}")
        print("-" * 50)

        return args

def create_model_config(args):
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


class Utils:
    @staticmethod
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True


class CheckPtMgr:
    @staticmethod
    def init_ckpts(ckpt_path, exp_name):
        """初始化检查点管理器
        Args:
            ckpt_path (str): 检查点路径，通常为 './checkpoints/'
            exp_name (str): 实验名称，exp_name=r'exp_' + current_file_name，这是为了避免不同实验之间的检查点文件冲突
        Returns:
            checkpoint_callback: 模型检查点回调，检测指标为test_loss，保存最优模型
            checkpoint_dir: 检查点目录，用于存储模型检查点文件
        """
        # 检查是否存在检查点文件
        checkpoint_dir = ckpt_path + '/' + exp_name
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_path + '/' + exp_name,
            filename='model-{epoch:02d}-{test_loss:.2f}',
            save_top_k=1,
            save_last=True,
            monitor='test/loss',
            mode='min'
        )

        return checkpoint_callback, checkpoint_dir

    @staticmethod
    def load_latest_ckpts(ckpt_path, model):
        """加载最新的检查点文件
        Args:
            ckpt_path (str): 检查点路径，通常为 './checkpoints/' + exp_name
            model: 模型实例，用于加载检查点文件
        Returns:
            model: 加载了最新检查点文件的模型实例
            latest_checkpoint_path: 最新检查点文件路径
        """
        if os.path.exists(ckpt_path) and len(os.listdir(ckpt_path)) > 0:
            # 获取最新的检查点文件
            checkpoint_files = [os.path.join(ckpt_path, f) for f in os.listdir(ckpt_path) if f.endswith('.ckpt')]

            if not checkpoint_files:
                return model, None

            # 排序并获取最新的检查点文件
            checkpoint_files.sort(key=os.path.getmtime)
            latest_ckpt_path = checkpoint_files[-1]
            checkpoint = torch.load(latest_ckpt_path)

            # 处理state_dict的键名以匹配当前模型结构
            new_state_dict = {}
            for key, value in checkpoint['state_dict'].items():
                # 如果键名以'module.'开头但当前模型是DataParallel包装的，则添加'model.'前缀
                if key.startswith('module.'):
                    new_key = 'model.' + key
                elif key.startswith('model.module.'):
                    # 如果已经是正确的格式，则保持不变
                    new_key = key
                else:
                    # 其他情况直接添加model前缀
                    new_key = 'model.' + key

                new_state_dict[new_key] = value

            # 加载state_dict
            model.load_state_dict(new_state_dict, strict=False)

            return model, latest_ckpt_path

        return model, None


class LoggerMgr:
    @staticmethod
    def init_logger(args):
        model_name_path = '{}/{}classes_{}k_{}epochs_{}ncells_{}batchsize_{}kfold{}_{}lr'
        model_name_path = model_name_path.format(args.version, args.num_classes, args.num_neighbor, args.num_epochs,
                                                 args.ncells,
                                                 args.train_batch_size, args.kfold_nums, args.use_fold, args.lr)
        # 确保logger路径存在
        logger_path = os.path.join(args.logger_path, model_name_path)
        if not os.path.exists(logger_path):
            os.makedirs(logger_path, exist_ok=True)

        tb_logger = TensorBoardLogger(save_dir=args.logger_path, name=model_name_path)
        return tb_logger


class PredMgr:
    @staticmethod
    def get_results_path(args):
        model_name_path = '{}/{}classes_{}epochs_{}ncells_{}ratio_{}_rmin{}_rmax_{}lr'
        model_name_path = model_name_path.format(args.version, args.num_classes, args.num_epochs,
                                                 args.ncells,
                                                 args.sample_ratio,
                                                 args.rmin, args.rmax, args.lr)
        results_path = os.path.join(args.results_path, model_name_path)
        return results_path
    @staticmethod
    def save_to_json(args, predictions):
        # 保存结果
        results_path = PredMgr.get_results_path(args)
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

    @staticmethod
    def save_to_topo_clu_json(args, predictions):
        # 保存结果
        results_path = PredMgr.get_results_path(args)
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
    @staticmethod
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

    # @lru_cache(maxsize=256)
    @staticmethod
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


def do_train():
    # global num_classes, val_batch_size, lr, start, model
    Utils.setup_seed(1111)
    rseed = 42
    # rseed = 27
    # 全局参数
    args = create_training_args()
    #logger
    tb_logger = LoggerMgr.init_logger(args)
    #checkpoints
    checkpoint_callback, checkpoint_path = CheckPtMgr.init_ckpts(args.ckpt_path, args.exp_name)
    # datasets
    train_ds, val_ds, test_ds = load_DBGA_dataset_with_tbase(sample_ratio=args.sample_ratio, train_val_ratio=1.0, comb_name='comb_PointnetTrans', is_full = True,
                                                             retain_file_path=None, rseed=rseed)
    # train_ds, _, test_ds = load_datasets(sample_ratio=args.sample_ratio,comb_name='comb_qslim_5000',rseed=rseed)

    train_loader = GeoDataLoader(dataset=train_ds, batch_size=args.train_batch_size, shuffle=True,
                                 num_workers=args.num_workers,pin_memory=True,persistent_workers=True)
    test_loader = GeoDataLoader(dataset=test_ds, batch_size=args.val_batch_size, shuffle=False,
                                num_workers=args.num_workers,pin_memory=True,persistent_workers=True)

    # model
    config = create_model_config(args)
    model = TripleBinClsModule(config)


    model,latest_checkpoint_path = CheckPtMgr.load_latest_ckpts(checkpoint_path, model)
    # trainer
    # trainer = pl.Trainer(max_epochs=num_epochs, accelerator='auto', num_sanity_val_steps=0)
    trainer = pl.Trainer(max_epochs=args.num_epochs,
                         logger=tb_logger, log_every_n_steps=20,
                         check_val_every_n_epoch=1,
                         precision='16-mixed',
                         callbacks=[checkpoint_callback]
                         )
    start = time.time()
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader, ckpt_path=latest_checkpoint_path)
    end = time.time()
    running_time = end - start
    print(f'Time cost: {running_time:.5f}s, {running_time / 3600} h\n')


def do_pred():
    Utils.setup_seed(1111)
    # 全局参数
    args = create_training_args()
    #logger
    tb_logger = LoggerMgr.init_logger(args)
    #checkpoints
    checkpoint_callback, checkpoint_path = CheckPtMgr.init_ckpts(args.ckpt_path, args.exp_name)

    # args.sample_ratio = 1.0
    sample_ratio = 1.0
    _, _, predict_ds = load_DBGA_dataset_with_tbase(sample_ratio=sample_ratio, train_val_ratio=1.0, comb_name='comb_PointnetTrans',
                                                             is_full=True, retain_file_path=None, rseed=42)
    predict_loader = GeoDataLoader(dataset=predict_ds, batch_size=1, shuffle=False, num_workers=args.num_workers,
                                   pin_memory=True)

    config = create_model_config(args)
    model = TripleBinClsModule(config)

    # 检查是否存在检查点文件
    model,latest_checkpoint_path = CheckPtMgr.load_latest_ckpts(checkpoint_path, model)

    trainer = pl.Trainer()
    # 进行预测
    if 'latest_checkpoint_path' in locals():
        print(latest_checkpoint_path, locals())
        predictions = trainer.predict(model, dataloaders=predict_loader, ckpt_path=latest_checkpoint_path)
    else:
        predictions = trainer.predict(model, dataloaders=predict_loader)

    # 保存为json
    PredMgr.save_to_json(args, predictions)
    # 打印信息
    PredMgr.print_pred_metrics(predictions)

def test_get_model():
    import torch  # Add this import statement

    model = TripleBinClsNet(num_classes=15, num_feat=6, k=16)
    model.cuda()
    xyz = torch.rand(10000,6).cuda()
    data = Data(x=xyz, pos=xyz[:,:3])

    tg_outputs, tae_outputs, tte_outputs, x_1 = model(data)

    # output is B, C, N order
    print(tg_outputs[0].shape, tg_outputs[1].shape, tg_outputs[2].shape)
    print(tae_outputs[0].shape, tae_outputs[1].shape, tae_outputs[2].shape)
    print(tte_outputs[0].shape, tte_outputs[1].shape, tte_outputs[2].shape)
    print(x_1.shape)

    # -- coding: utf-8 --
    from thop import profile

    flops, params = profile(model, inputs=((data),))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f G, params: %.2f M' % (flops / 1.0e9, params / 1.0e6))



def save_topo_cluster_results():
    """
    保存拓扑聚类结果
    """
    Utils.setup_seed(1111)
    args = create_training_args()
    tb_logger = LoggerMgr.init_logger(args)
    checkpoint_callback, checkpoint_path = CheckPtMgr.init_ckpts(args.ckpt_path, args.exp_name)
    # args.sample_ratio = 1.0
    sample_ratio = 1.0
    train_ds, _, predict_ds = load_DBGA_dataset_with_tbase(sample_ratio=sample_ratio, train_val_ratio=1.0,
                                                    comb_name='comb_PointnetTrans',
                                                    is_full=True, retain_file_path=None, rseed=42)
    predict_loader = GeoDataLoader(dataset=train_ds, batch_size=1, shuffle=False, num_workers=args.num_workers,
                                   pin_memory=True)
    config = create_model_config(args)
    model = TripleBinClsModule(config)
    # 检查是否存在检查点文件
    model, latest_checkpoint_path = CheckPtMgr.load_latest_ckpts(checkpoint_path, model)
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
    PredMgr.save_to_topo_clu_json(args, predictions)
    # 打印信息
    PredMgr.print_pred_metrics(predictions)


def test_read_json_file():
    json_path = r'C:\Users\win10\DualMesh\pynb\teeth3ds\model\results\exp_TripleBinClsNet\2classes_30epochs_10000ncells_0.1ratio_0.04_rmin0.04_rmax_0.001lr'
    json_name = r'013NUWYR_lower'
    #####################################################
    # 测试读取文件耗时
    start_time = time.time()
    json_data, pred_acc = PredMgr.from_json(json_path, json_name)
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
