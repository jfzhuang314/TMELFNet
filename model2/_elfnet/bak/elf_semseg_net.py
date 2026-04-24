"""
@File    :   skip_mask.py
这一版本主要简化了输入特征的提取，仅仅使用两个edgeconv层提取特征；
"""
import argparse
import json
import pickle
from abc import abstractmethod
from collections import namedtuple
from functools import lru_cache

from torch import optim
from torch.nn import Sequential as Seq, Linear as Lin, ReLU

import os
import random

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import time

from torch_geometric.nn import EdgeConv, radius_graph, GATConv
from torch_geometric.data import DataLoader as GeoDataLoader, Data

from datasets.misc.DBGAMeshDataset import load_DBGA_dataset_with_tbase
from metrics.DBGA_metrics import *
from model.BaseNet import BaseNet  # 新的的BaseNet
from model._dynamic_duo.utils import gen_utils as gu, tsg_loss, ops_utils as ou
from torch_cluster import radius_graph

# from model._edge_mask.edge_mask_atten_rang2 import EdgeCNN
from model._edge_mask.edge_mask_V2_skip import EdgeCNN

from utils.points_utils import *
from model._edge_mask.edge_mask_V3_triple_bin_cls import TripleBinClsModule
from model._dynamic_duo.utils.loss_meter import LossMap
from model._elf.crop_patch import crop_patches_eef, get_ddf, get_max_frequency_non_zero_class


def get_json_obj(json_data):
    """
    从json_data获取特定对象，并转换为结构化格式
    参数:
        json_data: JSON数据
        i: 要获取的对象索引
    返回:
        json_obj: 结构化的JSON对象
    """
    # 创建一个命名元组作为结构体
    NumpyObj = namedtuple('JsonObj', [
        'json_name',
        'points',  # 点云输入特征
        'l0_points',  # 32维的特征
        'V',  # 点云坐标
        'F',  # 面信息
        'gt_labels',  # 真实标签
        'pred_labels',  # 预测标签
        'bg_pd_1',  # 背景属性1
        'bg_weight_1',  # 背景权重1
        'bg_pd_2',  # 背景属性2
        'edge_pd_1',  # 边缘属性1
        'edge_weight_1',  # 边缘权重1
        'edge_pd_2',  # 边缘属性2
        'edge_pd_3',  # 边缘属性3
        'edge_gt_bin_labels'  # 边缘二值标签
    ])

    # Helper function to convert tensor to numpy array
    def tensor_to_numpy(tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.cpu().numpy()
        return tensor

    def extract_name_from_path(file_path):
        normalized_path = file_path.replace('\\\\', '\\')
        # 分割路径，获取各个部分
        parts = normalized_path.split('\\')
        # 提取倒数第二部分（lower）和最后一部分（01CCTTK4）
        folder_name = parts[-2]
        file_name = parts[-1]
        # 组合并返回结果
        return f"{file_name}_{folder_name}"

    # 提取并转换数据
    numpy_obj = NumpyObj(
        json_name=extract_name_from_path(json_data['mesh_file'][0]),
        points=tensor_to_numpy(json_data['pts_normals'])[:, 0:6],
        l0_points=tensor_to_numpy(json_data['l0_points'])[:, 0:32],
        V=tensor_to_numpy(json_data['pts_normals'])[:, 0:3],
        F=tensor_to_numpy(json_data['faces']),
        gt_labels=tensor_to_numpy(json_data['gt_labels']),
        pred_labels=tensor_to_numpy(json_data['pred_labels']),
        bg_pd_1=tensor_to_numpy(json_data['bg_pd_1']),
        bg_weight_1=tensor_to_numpy(json_data['bg_weight_1']),
        bg_pd_2=tensor_to_numpy(json_data['bg_pd_2']),
        edge_pd_1=tensor_to_numpy(json_data['edge_pd_1']),
        edge_weight_1=tensor_to_numpy(json_data['edge_weight_1']),
        edge_pd_2=tensor_to_numpy(json_data['edge_pd_2']),
        edge_pd_3=tensor_to_numpy(json_data['edge_wobg_pd_2']),
        edge_gt_bin_labels=tensor_to_numpy(json_data['edge_gt_bin_labels'])[:, np.newaxis]
    )
    return numpy_obj


# 全局内存缓存
MEMORY_CACHE = {}


def process_with_cache(prediction, cache_dir='./cache', opts=None):
    """
    处理预测数据并使用缓存来避免重复计算

    参数:
        prediction: 预测数据
        cache_dir: 缓存目录
        opts: 传递给crop_patches_eef的选项

    返回:
        处理结果
    """
    # 确保缓存目录存在
    """使用内存缓存来加速数据读取"""
    numpy_obj = get_json_obj(prediction)
    json_name = numpy_obj.json_name
    if isinstance(json_name, np.ndarray):
        json_name = json_name.item()

    # 首先检查内存缓存
    if json_name in MEMORY_CACHE:
        # print(f"从内存缓存加载: {json_name}")
        return MEMORY_CACHE[json_name]

    # 然后检查文件缓存
    cache_file = os.path.join(cache_dir, f"{json_name}.pkl")
    if os.path.exists(cache_file):
        # print(f"从文件缓存加载: {json_name}")
        with open(cache_file, 'rb') as f:
            result = pickle.load(f)
            # 同时存入内存缓存
            MEMORY_CACHE[json_name] = result
            return result

    # 计算新结果
    print(f"计算新结果: {json_name}")
    if opts is None:
        opts = {'alpha': 0.2, 'max_iter': 5, 'iter_thred_len': 0.01,
                'field_bound': 0.03, 'field_sigma': 0.1, 'crop_num': 1500}

    result = crop_patches_eef(numpy_obj, opts)

    # 保存到文件和内存缓存
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump(result, f)
    MEMORY_CACHE[json_name] = result

    return result


def id_loss(gt_label, pred_id):
    """
    计算ID预测的交叉熵损失，处理3D张量输入。该loss主要用以ELFSemSegModule中的id_loss，相比于之前semsegnet，准确率会低。
    参数:
        gt_label: 形状为[batch_size, 1]的张量，表示每个点的真实类别
        pred_id: 形状为[batch_size, num_classes]的张量，表示每个点的类别预测概率
    返回:
        loss: 计算得到的平均交叉熵损失
    """
    # 获取批次大小和点数
    batch_size, num_classes = pred_id.shape
    # 重塑预测张量为[batch_size * num_points, num_classes]
    pred_id_reshaped = pred_id.contiguous().view(-1, num_classes)
    # 重塑真实标签张量为[batch_size * num_points]
    gt_label_reshaped = gt_label.view(-1).type(torch.long)
    # 计算交叉熵损失
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(pred_id_reshaped, gt_label_reshaped)
    return loss


@lru_cache(maxsize=256)
def get_edge_indices_lru(xyz_flat_tuple, batch_idx_tuple, radii, max_neighbors=32, device='cpu'):
    """Use functools.lru_cache to cache results；
    params:
        xyz_flat: (N, 3) tensor of flattened point coordinates
        batch_idx_tuple: (N,) tuple of batch indices
        radii: (N,) array of radii for each point
        max_neighbors: maximum number of neighbors to consider
        device: 'cpu' or 'cuda'
    Returns:
        edge_index_list: list of edge indices for each radius
    """
    # Convert tuples to tensors
    xyz_flat = torch.tensor(xyz_flat_tuple, dtype=torch.float32).view(-1, 3).to(device)
    batch_idx = torch.tensor(batch_idx_tuple, dtype=torch.long).to(device)

    # Calculate the edge_index list
    edge_index_list = []
    for r in radii:
        edge_index = radius_graph(xyz_flat, r=r, batch=batch_idx, max_num_neighbors=max_neighbors)
        edge_index_list.append(edge_index)

    return edge_index_list


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
        self.attn1 = GATConv(32, 8, heads=4, dropout=0.5)
        self.attn2 = GATConv(64, 16, heads=4, dropout=0.5)
        self.attn3 = GATConv(128, 32, heads=4, dropout=0.5)
        self.attn4 = GATConv(128, 32, heads=4, dropout=0.5)

        # 定义 MLP 层
        self.mlp1 = nn.Sequential(
            nn.Linear(128 + 128 + 64 + 32 + self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256)
        )

        self.fc1 = nn.Linear(128 + 128 + 64 + 32 + 256 + self.input_dim, 128)
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

        x_concat = torch.cat([x, x1, x2, x3, x4], dim=-1)
        x_mlp1 = self.mlp1(x_concat)

        x_mlp1_pooled = torch.mean(x_mlp1, dim=0, keepdim=True).expand(x.size(0), -1)

        x_fc_input = torch.cat([x_concat, x_mlp1_pooled], dim=-1)
        # x_fc_input = self.dp1(x_fc_input)
        x_F1 = F.relu(self.fc1(x_fc_input))  # 256
        x_F2 = self.fc2(x_F1)  # 32

        return x1, x_F1, x_F2, x_mlp1, x_fc_input, x1


class EdgeCNNBlockAtten2(nn.Module):
    def __init__(self, input_dim):
        super(EdgeCNNBlockAtten2, self).__init__()
        self.input_dim = input_dim
        # 定义 EdgeConv 层
        self.conv1 = EdgeConv(Seq(Lin(self.input_dim * 2, 32), ReLU()))
        self.conv2 = EdgeConv(Seq(Lin(32 * 2, 64), ReLU()))
        self.conv3 = EdgeConv(Seq(Lin(64 * 2, 128), ReLU()))
        self.conv4 = EdgeConv(Seq(Lin(128 * 2, 128), ReLU()))

        # 注意力机制
        self.attn1 = GATConv(32, 8, heads=4, dropout=0.5)
        self.attn2 = GATConv(64, 16, heads=4, dropout=0.5)
        self.attn3 = GATConv(128, 32, heads=4, dropout=0.5)
        self.attn4 = GATConv(128, 32, heads=4, dropout=0.5)

        # 定义 MLP 层
        self.mlp1 = nn.Sequential(
            nn.Linear(128 + 128 + 64 + 32 + self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512)
        )

        self.fc1 = nn.Linear(128 + 128 + 64 + 32 + 512 + self.input_dim, 256)
        self.fc2 = nn.Linear(256, 32)

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

        x_concat = torch.cat([x, x1, x2, x3, x4], dim=-1)
        x_mlp1 = self.mlp1(x_concat)

        x_mlp1_pooled = torch.mean(x_mlp1, dim=0, keepdim=True).expand(x.size(0), -1)

        x_fc_input = torch.cat([x_concat, x_mlp1_pooled], dim=-1)
        # x_fc_input = self.dp1(x_fc_input)
        x_F1 = F.relu(self.fc1(x_fc_input))  # 256
        x_F2 = self.fc2(x_F1)  # 32

        return x1, x_F1, x_F2, x_mlp1, x_fc_input


class SemSegNet(torch.nn.Module):
    def __init__(self, num_classes=15, num_feat=36, k=16):
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
        self.block2 = EdgeCNNBlockAtten2(self.input_dim + 2)

        self.pd_mask_1 = torch.nn.Linear(32, 2)
        self.pd_mask_1_softmax = torch.nn.Softmax(dim=1)
        self.wt_mask_1 = torch.nn.Linear(32, 1)

        self.pd_mask_2 = torch.nn.Linear(32, 1)
        # self.fc2 = nn.Linear(256, num_classes)
        self.fc2 = nn.Sequential(
            nn.Linear(512 + 3, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        # # self.fc2.weight.data.fill_(0.00)
        # self.fc2.bias.data.fill_(0.00)

    def forward(self, batch, clu_cent):
        data = batch

        points1 = data.float()
        xyz = points1[:, :3, :]  # [B, C, N]

        B, C, N = points1.shape

        # 步骤 1: 数据展平
        xyz_flat = xyz.transpose(1, 2).reshape(-1, 3)  # [B * N, C]
        batch_idx = torch.arange(B, device=xyz.device).repeat_interleave(N)  # 记录每个节点所属的批次

        # 步骤 2: 构建边索引
        edge_index_list = []
        radii = [0.03, 0.08, 0.2, 0.4]
        # radii = [0.03, 0.06, 0.12, 0.24]
        # Change view to reshape
        xyz_flat_tuple = tuple(xyz_flat.reshape(-1).tolist())
        batch_idx_tuple = tuple(batch_idx.tolist())
        radii_tuple = tuple(radii)
        device = xyz.device
        edge_index_list = get_edge_indices_lru(xyz_flat_tuple, batch_idx_tuple, radii_tuple, max_neighbors=32,
                                               device=device)

        # 展平 points1 以匹配边索引
        points1_flat = points1.transpose(1, 2).reshape(-1, points1.size(1))  # [B * N, C_total]
        points1_flat = points1_flat.to(device)

        # 步骤 3: 执行卷积操作
        outs1 = self.block1(points1_flat, edge_index_list)
        x1, x1_F1, x1_F2, x1_mlp, x1_fc_input, _ = outs1
        pd_1 = self.pd_mask_1_softmax(self.pd_mask_1(x1_F2))
        weight_1 = self.wt_mask_1(x1_F2)

        points2_flat = torch.cat([points1_flat, pd_1], dim=1)
        outs2 = self.block2(points2_flat, edge_index_list)
        x2, x2_F1, x2_F2, x2_mlp, x2_fc_input = outs2

        # pd
        pd_2 = self.pd_mask_2(x2_F2)

        # id_pred
        x_mlp1_pooled = torch.mean(x2_mlp.reshape(B, N, -1), dim=1, keepdim=True).squeeze(dim=1)
        # id_pred = F.softmax(self.fc2(x_mlp1_pooled), dim=1) # B, 15
        x3 = torch.cat([x_mlp1_pooled, clu_cent], dim=1)
        x3 = x3.to(self.fc2[0].weight.dtype)
        id_pred = F.softmax(self.fc2(x3), dim=1)  # B, 15

        # 步骤 4: 恢复批量维度
        def reshape_to_batch(x):
            return x.reshape(B, N, -1).transpose(1, 2)

        pd_1 = reshape_to_batch(pd_1)
        weight_1 = reshape_to_batch(weight_1)
        pd_2 = reshape_to_batch(pd_2)
        # id_pred = reshape_to_batch(id_pred)

        return pd_1, weight_1, pd_2, id_pred


class ELFSemSegModule(BaseNet):
    """ edge logits field semantic segmentation module"""

    def __init__(self, config):
        super().__init__(config)

        self.num_classes = config.get('num_classes')
        self.accumulate_grad_batches = config.get('accumulate_grad_batches')  # 梯度累积步数
        self.clu_batch_size = config.get('clu_batch_size')  # 每个小批次的大小

        # 第一个分割模型(仅为推理模型，trainable=false)
        self.first_segnet_model = TripleBinClsModule(config)
        # 第二个分割模型（实例分割模型，trainable=true）
        self.ins_segnet_model = SemSegNet(num_classes=self.num_classes, num_feat=6 + 2 + 1 + 32).to(device)
        self.ins_segnet_model = nn.DataParallel(self.ins_segnet_model)

        first_segnet_model_path = config.get('first_segnet_name')
        self.first_segnet_model, latest_ckpt_path = ELFSemSegModule.load_latest_ckpts_to_model(first_segnet_model_path,
                                                                                               self.first_segnet_model)
        if latest_ckpt_path is not None:
            print(f"Load first_segnet_model from {latest_ckpt_path}")

        # 设定 self.ins_segnet_model 不参与梯度更新
        for param in self.first_segnet_model.parameters():
            param.requires_grad = False

    @staticmethod
    def load_latest_ckpts_to_model(checkpoint_path, model):
        def regulate_state_dict(checkpoint):
            new_state_dict = {}
            for key, value in checkpoint['state_dict'].items():
                if key.startswith('model.'):
                    new_key = key[6:]  # Remove 'model.'
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            return new_state_dict

        if os.path.exists(checkpoint_path) and len(os.listdir(checkpoint_path)) > 0:
            # 获取最新的检查点文件
            checkpoint_files = [os.path.join(checkpoint_path, f) for f in os.listdir(checkpoint_path) if
                                f.endswith('.ckpt')]
            checkpoint_files.sort(key=os.path.getmtime)
            latest_checkpoint_path = checkpoint_files[-1]
            checkpoint = torch.load(latest_checkpoint_path)

            # Remove the extra 'model.' prefix from the state_dict
            new_state_dict = regulate_state_dict(checkpoint)

            model.model.load_state_dict(new_state_dict)
            return model, latest_checkpoint_path

        return model, None

    @abstractmethod
    def _update_step_metrics(self, batch):
        """
        该函数侧重于dsc、ppv、sen等指标的计算。
        :returns
            dsc, loss, ppv, sen, acc, iou： 六大指标
            id_pred_loss, seg_1_loss, seg_2_loss：三个损失（语义分割损失与TSegNet的两个损失）
            ins_pred_labels：预测的语义分割标签， [N,]
            nn_crop_indexes：来源于common step，记录每个patch的索引，[1, clu, 1500]；索引的范围为[0, N]
            elf_pd_2：来源于common step，记录每个顶点的边缘逻辑场，[N,1]； elf_pd_2>0为patch前景，elf_pd_2<0为背景
        """
        # batch = self.normalize(batch)
        # 真实类标
        labels = batch.y.to(self.device, dtype=torch.long) # [N,]
        one_hot_labels = nn.functional.one_hot(labels, num_classes=self.class_weights.size(0)) # [N, 15]

        outs = self.common_step(batch)
        cropped_feature_ls, id_pred, points, loss, nn_crop_indexes, elf_pd_2 = (outs[0], outs[1], outs[2], outs[3],
                                                                                outs[4], outs[5])
        id_pred_loss, seg_1_loss, seg_2_loss = outs[6], outs[7], outs[8]
        numpy_obj, clu_gt_bin_ls, clu_gt_map = outs[9], outs[10], outs[11]

        # 最大投票法决出ins_pred_labels
        crop_mat = torch.zeros((numpy_obj.V.shape[0], cropped_feature_ls.shape[0]), dtype=torch.long,
                               device=points.device)
        for i in range(0, cropped_feature_ls.shape[0]):
            pred_bin_labels = torch.zeros(cropped_feature_ls[i].shape[1], device=points.device)
            pred_bin_labels[elf_pd_2[i].reshape(-1) > 0] = 1
            # gu.print_3d(gu.np_to_pcd_with_label(gu.torch_to_numpy(cropped_feature_ls)[i,:3,:].T, pred_bin_labels))

            positive_indices = nn_crop_indexes[0][i][pred_bin_labels.cpu().numpy() == 1]
            if positive_indices.size > 0:
                # 获取对应位置的 id_pred 最大值索引
                filtered_id_pred = (id_pred.argmax(axis=1))[i]
                crop_mat[positive_indices, i] = filtered_id_pred

        # vis_crop_labels(numpy_obj, nn_crop_indexes, crop_mat.T.unsqueeze(dim=1).cpu().numpy())
        ins_pred_labels = get_max_frequency_non_zero_class(crop_mat.T.cpu().numpy())
        # gu.print_3d(gu.np_to_pcd_with_label(numpy_obj.V, ins_pred_labels))

        ins_pred_labels = torch.tensor(ins_pred_labels).long().to(self.device)
        ins_pred_labels_onehot = nn.functional.one_hot(ins_pred_labels, num_classes=self.class_weights.size(0))

        dsc = weighting_DSC(ins_pred_labels_onehot.unsqueeze(dim=1), one_hot_labels, self.class_weights)
        sen = weighting_SEN(ins_pred_labels_onehot.unsqueeze(dim=1), one_hot_labels, self.class_weights)
        ppv = weighting_PPV(ins_pred_labels_onehot.unsqueeze(dim=1), one_hot_labels, self.class_weights)
        acc = weighting_ACC(ins_pred_labels_onehot.unsqueeze(dim=1), one_hot_labels, self.class_weights)
        iou = weighting_IOU(ins_pred_labels_onehot.unsqueeze(dim=1), one_hot_labels, self.class_weights)

        # 集中删除不再需要的大张量
        if pred_bin_labels is not None:
            del pred_bin_labels
        del crop_mat, one_hot_labels, ins_pred_labels_onehot

        return (dsc, loss, ppv, sen, acc, iou, id_pred_loss, seg_1_loss, seg_2_loss,
                ins_pred_labels, nn_crop_indexes, elf_pd_2)

    def common_step(self, batch):
        """
        该函数侧重于数据转换及处理、计算损失，不涉及dsc、ppv、sen等指标的计算。
        """

        # 第一个分割模型推理得到结果
        prediction = self.first_segnet_model.predict_step(batch, 0)
        numpy_obj = get_json_obj(prediction)

        # eef算法得到的聚类结果
        opts = {'alpha': 0.2, 'max_iter': 5, 'iter_thred_len': 0.01, 'field_bound': 0.03, 'field_sigma': 0.1,
                'crop_num': 2500}
        outs = process_with_cache(prediction, cache_dir='../_elf/cache', opts=opts)
        # clu_num: [1,], clu_cen : [clu, 3], clu_counts: [clu, 1], clu_gt_map: [clu, 1],
        # nn_crop_indexes: [1, clu, 1500], clu_gt_label: [clu,1, 1500], clu_gt_bin_ls: [clu,1, 1500], tooth_flag: [10000,]
        clu_num, clu_cen, clu_counts, clu_gt_map, nn_crop_indexes, clu_gt_label, clu_gt_bin_ls, tooth_flag = outs[0:8]
        pca_max_comps, shape_feats, obb_extents = outs[8:]  # [clu, 3], [clu, 2], [clu, 3]

        # 数据格式转换
        points = prediction['pts_normals']  # 10000,6
        xyz = points[:, :3]  # 10000,3
        gt_labels = prediction['gt_labels']  # 10000,1
        l0_points = prediction['l0_points']  # 10000,32
        bg_pd_1 = prediction['bg_pd_1']  # 1,2,10000
        edge_pd_1 = prediction['edge_pd_1']  # 1,2,10000
        edge_wobg_pd_1 = prediction['edge_wobg_pd_1']  # 1,2,10000
        clu_feat_tensor = torch.from_numpy(clu_cen).to(points.device)  # B,3

        # 角度函数
        ang, abs_ang = compute_polar_angles(xyz.float())

        if len(xyz.shape) == 2: xyz = xyz.unsqueeze(dim=0)  # 1,10000,3
        if len(points.shape) == 2: points = points.unsqueeze(dim=0)  # 1,10000,6
        if len(ang.shape) == 1: ang = ang.unsqueeze(dim=0).unsqueeze(dim=2)  # 1,10000,1
        if len(abs_ang.shape) == 1: abs_ang = abs_ang.unsqueeze(dim=0).unsqueeze(dim=2)  # 1,10000,1
        if len(l0_points.shape) == 2: l0_points = l0_points.unsqueeze(dim=0)  # 1,10000,32
        if len(gt_labels.shape) == 2: gt_labels = gt_labels.unsqueeze(dim=0)  # 1,10000,1

        cropped_input_ls = ou.get_indexed_features(points.permute(0, 2, 1), nn_crop_indexes)  # clu, 6, 1500
        cropped_ang_ls = ou.get_indexed_features(ang.permute(0, 2, 1), nn_crop_indexes)  # clu, 6, 1500
        cropped_abs_ang_ls = ou.get_indexed_features(abs_ang.permute(0, 2, 1), nn_crop_indexes)  # clu, 6, 1500
        cropped_feature_ls = ou.get_indexed_features(l0_points.permute(0, 2, 1), nn_crop_indexes)  # clu, 32, 1500
        cropped_bg_pd1_ls = ou.get_indexed_features(bg_pd_1, nn_crop_indexes)  # clu, 2, 1500
        cropped_edge_pd1_ls = ou.get_indexed_features(edge_pd_1, nn_crop_indexes)  # clu, 2, 1500
        cropped_edge_wobg_pd1_ls = ou.get_indexed_features(edge_wobg_pd_1, nn_crop_indexes)  # clu, 2, 1500

        ddf = get_ddf(cropped_input_ls[:, :3, :].permute(0, 2, 1),
                      torch.tensor(np.expand_dims(clu_cen, axis=0)).permute(1, 0, 2).to(cropped_input_ls.device))
        # cropped_feature_ls = torch.cat([cropped_input_ls, cropped_feature_ls,
        #                                 torch.sigmoid(cropped_bg_pd1_ls), torch.sigmoid(cropped_edge_pd1_ls),
        #                                 torch.sigmoid(cropped_edge_wobg_pd1_ls)], axis=1).float()
        cropped_feature_ls = torch.cat([cropped_input_ls, cropped_feature_ls,
                                        cropped_ang_ls, cropped_abs_ang_ls, ddf],
                                       axis=1).float()
        cropped_feature_ls = cropped_feature_ls.to(device)  # clu, 12, 1500
        # gt
        gt = {"seg_label": gt_labels}

        # 语义分割网络
        # pd_1, weight_1, pd_2, id_pred = self.ins_segnet_model(cropped_feature_ls)

        # 分批处理，因为后面有一个BN*3的操作，一下喂太多，GPU吃不消
        batch_size = self.clu_batch_size  # 根据GPU内存调整
        num_clusters = cropped_feature_ls.shape[0]
        # 初始化损失累积器
        pd_1_list, weight_1_list, pd_2_list, id_pred_list = [], [], [], []
        for i in range(0, num_clusters, batch_size):
            start_idx = i
            end_idx = min(i + batch_size, num_clusters)

            batch_features = cropped_feature_ls[start_idx:end_idx]
            batch_clu_feat_tensor = clu_feat_tensor[start_idx:end_idx, :]
            # 对每个小批次进行推理，
            pd_1_batch, weight_1_batch, pd_2_batch, id_pred_batch = self.ins_segnet_model(batch_features,
                                                                                          batch_clu_feat_tensor)

            pd_1_list.append(pd_1_batch)
            weight_1_list.append(weight_1_batch)
            pd_2_list.append(pd_2_batch)
            id_pred_list.append(id_pred_batch)

            del pd_1_batch, weight_1_batch, pd_2_batch, id_pred_batch

        # 合并结果
        pd_1 = torch.cat(pd_1_list, dim=0)
        weight_1 = torch.cat(weight_1_list, dim=0)
        pd_2 = torch.cat(pd_2_list, dim=0)
        id_pred = torch.cat(id_pred_list, dim=0)

        outputs = dict()
        outputs.update({
            "pd_1": pd_1, "weight_1": weight_1, "pd_2": pd_2, "id_pred": id_pred,
            "center_points": torch.tensor(clu_cen), "clu_gt_map": clu_gt_map,
            "cluster_gt_seg_label": torch.tensor(clu_gt_label),
            "cluster_gt_bin_ls": torch.tensor(clu_gt_bin_ls),
            "cropped_feature_ls": cropped_feature_ls
        })

        loss_meter = LossMap()

        losses, id_pred_loss, seg_1_loss, seg_2_loss = self.get_loss(outputs, gt)
        loss_meter.add_loss_by_dict(losses)
        loss = loss_meter.get_sum()

        # 删除不再需要的prediction字典
        del prediction

        return (cropped_feature_ls, id_pred, points, loss, nn_crop_indexes, pd_2,
                id_pred_loss, seg_1_loss, seg_2_loss,
                numpy_obj, clu_gt_bin_ls, clu_gt_map)

    def forward(self, x):
        outs = self.common_step(x)
        cropped_feature_ls, id_pred, points, loss, nn_crop_indexes, pd_2, id_pred_loss, seg_1_loss, seg_2_loss = outs[
                                                                                                                 0:9]
        return loss

    def validation_step(self, batch, batch_idx):
        outs = self._update_step_metrics(batch)
        dsc, loss, ppv, sen, acc, iou, = outs[0], outs[1], outs[2], outs[3], outs[4], outs[5]
        id_pred_loss, seg_1_loss, seg_2_loss = outs[6], outs[7], outs[8]

        def get_mesh_name(file_path):
            file_name = os.path.basename(file_path)
            folder_name = os.path.basename(os.path.dirname(file_path))
            return f"{file_name}_{folder_name}"

        mesh_name = get_mesh_name(batch.filename[0])

        print(f"epoch:{self.current_epoch}-{mesh_name}, "
              f"loss:{loss:.4f}=id_pred_loss:{id_pred_loss:.4f} + seg_1_loss:{seg_1_loss:.4f}+ seg_2_loss:{seg_2_loss:.4f}",
              end=' ')

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
        ins_pred_labels = outputs[9]  # (10000,1)
        # _, pred_labels = torch.max(ins_pred_labels, 1)
        nn_crop_indexes, elf_pd_2 = outputs[10], outputs[11]

        ######################
        # triple_bin_cls产生的结果
        triple_bin_cls_pred = self.first_segnet_model.predict_step(batch, 0)
        bg_pd_2 = triple_bin_cls_pred['bg_pd_2']  # 1,1,10000
        edge_pd_2 = triple_bin_cls_pred['edge_pd_2']  # 1,1,10000
        edge_wobg_pd_2 = triple_bin_cls_pred['edge_wobg_pd_2']  # 1,1,10000
        ######################



        mesh_file = batch.filename
        pts_normals = batch.x
        faces = batch.faces
        labels = batch.y.to(self.device, dtype=torch.long)

        outs = {
            'predict_loss': loss,
            'predict_dsc': dsc,
            'predict_sen': sen,
            'predict_ppv': ppv,
            'predict_acc': acc,
            'pts_normals': pts_normals,
            'faces': faces,
            'labels': labels,
            'mesh_file': mesh_file,
            'bg_pd_2':bg_pd_2, # 1,1,10000; 背景逻辑场
            'edge_pd_2':edge_pd_2, # 1,1,10000; 边缘逻辑场
            'edge_wobg_pd_2':edge_wobg_pd_2, # 1,1,10000; 无背景边缘逻辑场
            'ins_pred_labels': ins_pred_labels,  # (10000,1)， 0-14， 预测的实例标签
            'nn_crop_indexes': nn_crop_indexes,  # (1, clu, 1500)， 每个clu的crop点的索引
            'elf_pd_2': elf_pd_2  # (10000,1) 每个顶点的elf 边缘逻辑场
        }

        return outs

    def regularization_losses(self):
        #####################
        # Regularization loss
        #####################

        # 计算 L2 正则化损失
        l2_lambda = 0.0001  # 正则化系数，可根据实际情况调整
        l2_reg = torch.tensor(0., requires_grad=True).to(self.device)
        for name, param in self.ins_segnet_model.named_parameters():
            if 'weight' in name:
                l2_reg = l2_reg + torch.norm(param, 2)

        l2_loss = l2_lambda * l2_reg
        return l2_loss

    def get_loss(self, outputs, gt):
        """
        计算模型的损失。
        :param outputs: 模型的输出字典，包含各种预测结果
        :param gt: 真实标签字典，包含分割标签、质心坐标和质心标签
        :return: 包含各种损失及其权重的字典
        """
        losses = {}

        id_pred_loss = id_loss(torch.tensor(outputs["clu_gt_map"]).to(device), outputs["id_pred"].to(device))
        seg_1_loss = tsg_loss.first_seg_loss(outputs["pd_1"].to(device), outputs["weight_1"].to(device),
                                             outputs["cluster_gt_bin_ls"].to(device))
        seg_2_loss = tsg_loss.second_seg_loss(outputs["pd_2"].to(device), outputs["weight_1"].to(device),
                                              outputs["cluster_gt_bin_ls"].to(device))
        l2_loss = self.regularization_losses()

        losses.update({
            "seg_1_loss": (seg_1_loss, 1),
            "seg_2_loss": (seg_2_loss, 1),
            "id_pred_loss": (id_pred_loss, 1)
        })

        return losses, id_pred_loss, seg_1_loss, seg_2_loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.ins_segnet_model.parameters(), amsgrad=True, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        return [optimizer], [scheduler]

    def _save_checkpoint(self, path, metrics_dict):

        save_dict = {
            'epoch': self.current_epoch + 1,
            'model_state_dict': self.ins_segnet_model.state_dict(),
            'optimizer_state_dict': self.trainer.optimizers[0].state_dict(),
            **metrics_dict
        }

        torch.save(save_dict, path)


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
        first_segnet_name=r'exp_TripleBinClsNet', # 三个elf的模型
        second_segnet_name=r'exp_edge_mask_V2_skip',  # 记得更改import，exp_edge_mask_atten_rang2，语义分割结果
        with_base_lower_path=r"E:\mesh_datasets\3DTeeth\Teeth3DS_train_test_split\with_base_lower.txt",
        wo_base_lower_path=r"E:\mesh_datasets\3DTeeth\Teeth3DS_train_test_split\wo_base_lower.txt",
        split_600_path=r"E:\mesh_datasets\3DTeeth\Teeth3DS_train_test_split\split_600_lower.txt",
        rmin=0.02,
        rmax=0.05,
        num_classes=15,
        num_neighbor=32,
        num_epochs=30,
        num_workers=16,
        accumulate_grad_batches=1,  # 梯度累积步数
        clu_batch_size=1,  # 每个小批次的大小
        sample_ratio=1.0,
        train_val_ratio=1.0,
        comb_name='comb_PointnetTrans',
        train_batch_size=1,  # 训练批次
        val_batch_size=1,  # 测试批次
        lr=0.001
    )


def get_cfg_from_args(args):
    config = {
        'num_classes': args.num_classes,
        'num_neighbor': args.num_neighbor,
        'lr': args.lr,
        'accumulate_grad_batches': args.accumulate_grad_batches,
        'clu_batch_size': args.clu_batch_size,  # 每个小批次的大小
        'num_epochs': args.num_epochs,
        'rmin': args.rmin,
        'rmax': args.rmax,
        'num_batches_to_print': 150,
        'model_path': args.ckpt_path + '/' + args.version,
        'first_segnet_name': args.ckpt_path + '/' + args.first_segnet_name,
        'second_segnet_name': args.ckpt_path + '/' + args.second_segnet_name,
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
        checkpoint_files = [os.path.join(checkpoint_path, f) for f in os.listdir(checkpoint_path) if
                            f.endswith('.ckpt')]

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

        # model.model.load_state_dict(new_state_dict)
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(new_state_dict)
        return model, latest_checkpoint_path

        return model, latest_checkpoint_path

    return model, None


def load_latest_edge_masknet_ckpts(checkpoint_path, model):
    if os.path.exists(checkpoint_path) and len(os.listdir(checkpoint_path)) > 0:
        # 获取最新的检查点文件
        checkpoint_files = [os.path.join(checkpoint_path, f) for f in os.listdir(checkpoint_path) if
                            f.endswith('.ckpt')]

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
        return model, latest_checkpoint_path

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


def test2_DGCNN():
    # global num_classes, val_batch_size, lr, start, model
    setup_seed(1111)
    rseed = 42
    # rseed = 27
    # 全局参数
    args = get_args()
    # logger
    tb_logger = init_logger(args)
    # checkpoints
    checkpoint_callback, checkpoint_path = init_ckpts(args)
    # datasets
    train_ds, val_ds, test_ds = load_DBGA_dataset_with_tbase(sample_ratio=args.sample_ratio, train_val_ratio=1.0,
                                                             comb_name='comb_PointnetTrans',
                                                             is_full=False,
                                                             retain_file_path=None, rseed=rseed)
    train_loader = GeoDataLoader(dataset=train_ds, batch_size=args.train_batch_size, shuffle=True,
                                 num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
    test_loader = GeoDataLoader(dataset=test_ds, batch_size=args.val_batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True, persistent_workers=True)

    # model
    config = get_cfg_from_args(args)
    model = ELFSemSegModule(config)

    model, latest_checkpoint_path = load_latest_ckpts(checkpoint_path, model)
    # trainer
    # trainer = pl.Trainer(max_epochs=num_epochs, accelerator='auto', num_sanity_val_steps=0)
    trainer = pl.Trainer(max_epochs=args.num_epochs,
                         logger=tb_logger, log_every_n_steps=20,
                         check_val_every_n_epoch=1,
                         accumulate_grad_batches=args.accumulate_grad_batches,
                         precision='16-mixed',  # 自动混合精度
                         callbacks=[checkpoint_callback]
                         )
    # trainer = pl.Trainer(max_epochs=args.num_epochs,
    #                      logger=tb_logger, log_every_n_steps=20,
    #                      check_val_every_n_epoch=1,
    #                      callbacks=[checkpoint_callback]
    #                      )

    start = time.time()
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader, ckpt_path=latest_checkpoint_path)
    end = time.time()
    running_time = end - start
    print(f'Time cost: {running_time:.5f}s, {running_time / 3600} h\n')


# def save_to_json(args, predictions):
#     # 保存结果
#     results_path = get_results_path(args)
#     if not os.path.exists(results_path):
#         os.makedirs(results_path, exist_ok=True)
#     # 遍历predictions，并根据filename保存每个预测结果为json文件
#     for prediction in predictions:
#         filename = prediction['mesh_file']
#         if isinstance(filename, list):
#             if len(filename) > 0:
#                 filename = filename[0]
#             else:
#                 print("Warning: The filename list is empty. Skipping this prediction.")
#                 continue
#         base_name = os.path.basename(filename)
#         dir_name = os.path.basename(os.path.dirname(filename))
#         new_filename = f"{base_name}_{dir_name}.json"
#         # 构造保存路径
#         save_path = os.path.join(results_path, new_filename)
#         # 转换张量为可序列化的类型
#         serializable_prediction = {}
#         for key, value in prediction.items():
#             if isinstance(value, torch.Tensor):
#                 serializable_prediction[key] = value.cpu().tolist()
#             else:
#                 serializable_prediction[key] = value
#         # 保存预测结果为json文件
#         with open(save_path, 'w') as f:
#             json.dump(serializable_prediction, f)
#             print("Saved prediction to", save_path)

def save_to_json(args, predictions):
    # Convert numpy arrays to lists
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj

    results_path = get_results_path(args)
    if not os.path.exists(results_path):
        os.makedirs(results_path, exist_ok=True)

    for pred in predictions:
        serializable_pred = {}
        for key, value in pred.items():
            serializable_pred[key] = convert_to_serializable(value)

        filename = pred['mesh_file']
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

        # 保存结果
        with open(save_path, 'w') as f:
            json.dump(serializable_pred, f)
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


def test3_pred():
    setup_seed(1111)
    # 全局参数
    args = get_args()
    # logger
    tb_logger = init_logger(args)
    # checkpoints
    checkpoint_callback, checkpoint_path = init_ckpts(args)

    args.sample_ratio = 1.0
    _, _, predict_ds = load_DBGA_dataset_with_tbase(sample_ratio=args.sample_ratio, train_val_ratio=1.0,
                                                    comb_name='comb_PointnetTrans',
                                                    retain_file_path=None, rseed=42)
    predict_loader = GeoDataLoader(dataset=predict_ds, batch_size=1, shuffle=False, num_workers=args.num_workers,
                                   pin_memory=True)

    config = get_cfg_from_args(args)
    model = ELFSemSegModule(config)

    # 检查是否存在检查点文件
    model, latest_checkpoint_path = load_latest_ckpts(checkpoint_path, model)

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


def test3_pred_combined():
    class_weights, predictions_edgemask, predictions_elf = get_elfsemnet_edgecnn_predictions()

    ins_pred_labels = predictions_elf[0]['ins_pred_labels'] # elfSemNet预测的实例标签
    # xyz = predictions_elf[0]['pts_normals'][:, :3]
    nn_crop_indexes = predictions_elf[0]['nn_crop_indexes'] # 每个clu的crop点的索引
    elf_pd_2 = predictions_elf[0]['elf_pd_2'] # elf的边缘逻辑场
    gt_labels = predictions_elf[0]['labels'] # 真实标签
    # gu.print_3d(gu.np_to_pcd_with_label(xyz.cpu().numpy(), ins_pred_labels.cpu().numpy()))

    sem_pred_labels = predictions_edgemask[0]['pred_labels']
    xyz = predictions_elf[0]['pts_normals'][:, :3]
    # gu.print_3d(gu.np_to_pcd_with_label(xyz.cpu().numpy(), sem_pred_labels.cpu().numpy()))

    # 合并预测结果
    # 最大投票法决出ins_pred_labels
    crop_mat = torch.zeros((xyz.shape[0], nn_crop_indexes[0].shape[0]), dtype=torch.long, device=xyz.device)
    for i in range(0, nn_crop_indexes[0].shape[0]):
        pred_bin_labels = torch.zeros(nn_crop_indexes[0].shape[-1], device=xyz.device)  # (1500,)
        pred_bin_labels[elf_pd_2[i].reshape(-1) > 0] = 1

        positive_indices = nn_crop_indexes[0][i][pred_bin_labels.cpu().numpy() == 1]  #
        if positive_indices.size > 0:
            # 找出出现频率最高的那个标签
            vector = sem_pred_labels[positive_indices]
            counts = torch.bincount(vector)
            most_frequent_label = torch.argmax(counts)

            crop_mat[positive_indices, i] = most_frequent_label

    combined_pred_labels = get_max_frequency_non_zero_class(crop_mat.T.cpu().numpy())
    gu.print_3d(gu.np_to_pcd_with_label(xyz.cpu().numpy(), combined_pred_labels))

    combined_pred_labels = torch.tensor(combined_pred_labels).long().to(xyz.device)
    pred_labels_onehot = nn.functional.one_hot(combined_pred_labels, num_classes=class_weights.size(0))
    one_hot_labels = nn.functional.one_hot(gt_labels, num_classes=class_weights.size(0))

    dsc = weighting_DSC(pred_labels_onehot.unsqueeze(dim=1), one_hot_labels, class_weights)
    sen = weighting_SEN(pred_labels_onehot.unsqueeze(dim=1), one_hot_labels, class_weights)
    ppv = weighting_PPV(pred_labels_onehot.unsqueeze(dim=1), one_hot_labels, class_weights)
    acc = weighting_ACC(pred_labels_onehot.unsqueeze(dim=1), one_hot_labels, class_weights)
    iou = weighting_IOU(pred_labels_onehot.unsqueeze(dim=1), one_hot_labels, class_weights)

    print(
        f"elf_net指标  DSC: {predictions_elf[0]['predict_dsc']:.4f}, SEN: {predictions_elf[0]['predict_sen']:.4f}, PPV: {predictions_elf[0]['predict_ppv']:.4f}, ACC: {predictions_elf[0]['predict_acc']:.4f}")
    print(
        f"edge_mask指标  DSC: {predictions_edgemask[0]['predict_dsc']:.4f}, SEN: {predictions_edgemask[0]['predict_sen']:.4f}, PPV: {predictions_edgemask[0]['predict_ppv']:.4f}, ACC: {predictions_edgemask[0]['predict_acc']:.4f}")
    print(
        f"优化后指标  DSC: {dsc.item():.4f}, SEN: {sen.item():.4f}, PPV: {ppv.item():.4f}, ACC: {acc.item():.4f}, IOU: {iou.item():.4f}")

    # # 保存为json
    # save_to_json(args, predictions)
    # # 打印信息
    # print_pred_metrics(predictions)


def majoy_voting(xyz, nn_crop_indexes, elf_pd_2, bg_pd_2, sem_pred_labels):
    # 合并预测结果
    # 最大投票法决出ins_pred_labels
    crop_mat = torch.zeros((xyz.shape[0], nn_crop_indexes[0].shape[0]), dtype=torch.long, device=xyz.device)
    conf = torch.ones((xyz.shape[0]), dtype=torch.float, device=xyz.device)
    crop_conf = torch.ones((nn_crop_indexes[0].shape[0]), dtype=torch.float, device=xyz.device)
    for i in range(0, nn_crop_indexes[0].shape[0]):
        pred_bin_labels = torch.zeros(nn_crop_indexes[0].shape[-1], device=xyz.device)  # (1500,)
        # pred_bin_labels[elf_pd_2[i].reshape(-1) > 0 & bg_pd_2.reshape(-1)] = 1
        mask = (elf_pd_2[i].reshape(-1) > 0) & (bg_pd_2[nn_crop_indexes[0][i]].reshape(-1) > 0)
        pred_bin_labels[mask] = 1

        positive_indices = nn_crop_indexes[0][i][pred_bin_labels.cpu().numpy() == 1]  #
        if positive_indices.size > 0:
            # 找出出现频率最高的那个标签
            vector = sem_pred_labels[positive_indices]
            counts = torch.bincount(vector)
            most_frequent_label = torch.argmax(counts)

            crop_mat[positive_indices, i] = most_frequent_label
            conf[positive_indices] = counts[most_frequent_label] / positive_indices.size
            crop_conf[i] = counts[most_frequent_label] / positive_indices.size
    # print(conf)
    crop_pred_labels = get_max_frequency_non_zero_class(crop_mat.T.cpu().numpy())
    pred_labels = crop_pred_labels
    # Check if pred_labels and crop_pred_labels are 1D arrays
    if pred_labels.ndim == 1 and crop_pred_labels.ndim == 1:
        # Adjust indexing for 1D arrays
        mask = conf > 0.90
        pred_labels[mask] = sem_pred_labels[mask]
    else:
        print(
            f"Unexpected array dimensions: pred_labels {pred_labels.shape}, crop_pred_labels {crop_pred_labels.shape}")

    return pred_labels

def test3_pred_combined_to_json():

    setup_seed(1111)
    # 全局参数
    args = get_args()

    class_weights, predictions_edgemask, predictions_elf = get_elfsemnet_edgecnn_predictions(sample_ratio=1.0)

    predictions = predictions_elf

    for i in range(len(predictions)):

        nn_crop_indexes = predictions_elf[i]['nn_crop_indexes']  # 每个clu的crop点的索引
        elf_pd_2 = predictions_elf[i]['elf_pd_2']  # elf的边缘逻辑场
        gt_labels = predictions_elf[i]['labels']  # 真实标签
        xyz = predictions_elf[i]['pts_normals'][:, :3]

        bg_pd_2 = predictions_edgemask[i]['bg_pd_2'].squeeze()
        sem_pred_labels = predictions_edgemask[i]['pred_labels']

        combined_pred_labels = majoy_voting(xyz, nn_crop_indexes, elf_pd_2, bg_pd_2, sem_pred_labels)
        # combined_pred_labels = pred_labels

        # # 最大投票法决出ins_pred_labels
        # crop_mat = torch.zeros((xyz.shape[0], nn_crop_indexes[0].shape[0]), dtype=torch.long, device=xyz.device)
        # for j in range(0, nn_crop_indexes[0].shape[0]):
        #     pred_bin_labels = torch.zeros(nn_crop_indexes[0].shape[-1], device=xyz.device)  # (1500,)
        #     mask = (elf_pd_2[j].reshape(-1) > 0) & (bg_pd_2[nn_crop_indexes[0][j]].reshape(-1) > 0)
        #     pred_bin_labels[mask] = 1
        #
        #     positive_indices = nn_crop_indexes[0][j][pred_bin_labels.cpu().numpy() == 1]  #
        #     if positive_indices.size > 0:
        #         # 找出出现频率最高的那个标签
        #         vector = sem_pred_labels[positive_indices]
        #         counts = torch.bincount(vector)
        #         most_frequent_label = torch.argmax(counts)
        #
        #         crop_mat[positive_indices, j] = most_frequent_label
        #
        # combined_pred_labels = get_max_frequency_non_zero_class(crop_mat.T.cpu().numpy())
        # gu.print_3d(gu.np_to_pcd_with_label(xyz.cpu().numpy(), combined_pred_labels))

        # combined_pred_labels = torch.tensor(combined_pred_labels).long().to(xyz.device)
        # pred_labels_onehot = nn.functional.one_hot(combined_pred_labels, num_classes=class_weights.size(0))
        # one_hot_labels = nn.functional.one_hot(gt_labels, num_classes=class_weights.size(0))
        #
        # dsc = weighting_DSC(pred_labels_onehot.unsqueeze(dim=1), one_hot_labels, class_weights)
        # sen = weighting_SEN(pred_labels_onehot.unsqueeze(dim=1), one_hot_labels, class_weights)
        # ppv = weighting_PPV(pred_labels_onehot.unsqueeze(dim=1), one_hot_labels, class_weights)
        # acc = weighting_ACC(pred_labels_onehot.unsqueeze(dim=1), one_hot_labels, class_weights)
        # iou = weighting_IOU(pred_labels_onehot.unsqueeze(dim=1), one_hot_labels, class_weights)
        #
        #
        # # 语义分割的结果
        # predictions[i]['sem_pred_labels'] = sem_pred_labels
        # predictions[i]['sem_pred_labels15'] = predictions_edgemask[i]['predictions']
        # predictions[i]['sem_pred_dsc'] = predictions_edgemask[i]['predict_dsc']
        # predictions[i]['sem_pred_sen'] = predictions_edgemask[i]['predict_sen']
        # predictions[i]['sem_pred_ppv'] = predictions_edgemask[i]['predict_ppv']
        # predictions[i]['sem_pred_acc'] = predictions_edgemask[i]['predict_acc']
        #
        # # 综合分割的结果
        # predictions[i]['combined_pred_labels'] = combined_pred_labels
        # predictions[i]['combined_pred_dsc'] = dsc.item()
        # predictions[i]['combined_pred_sen'] = sen.item()
        # predictions[i]['combined_pred_ppv'] = ppv.item()
        # predictions[i]['combined_pred_acc'] = acc.item()
        # predictions[i]['combined_pred_iou'] = iou.item()

        combined_pred_labels = torch.tensor(combined_pred_labels).long().to(xyz.device)
        pred_labels_onehot = nn.functional.one_hot(combined_pred_labels, num_classes=class_weights.size(0))
        one_hot_labels = nn.functional.one_hot(gt_labels, num_classes=class_weights.size(0))

        dsc = weighting_DSC(pred_labels_onehot.unsqueeze(dim=1), one_hot_labels, class_weights)
        sen = weighting_SEN(pred_labels_onehot.unsqueeze(dim=1), one_hot_labels, class_weights)
        ppv = weighting_PPV(pred_labels_onehot.unsqueeze(dim=1), one_hot_labels, class_weights)
        acc = weighting_ACC(pred_labels_onehot.unsqueeze(dim=1), one_hot_labels, class_weights)
        iou = weighting_IOU(pred_labels_onehot.unsqueeze(dim=1), one_hot_labels, class_weights)

        # 语义分割的结果
        predictions[i]['sem_pred_labels'] = sem_pred_labels.cpu().tolist()
        predictions[i]['sem_pred_labels15'] = predictions_edgemask[i]['predictions'].cpu().tolist()
        predictions[i]['sem_pred_dsc'] = predictions_edgemask[i]['predict_dsc'].item()
        predictions[i]['sem_pred_sen'] = predictions_edgemask[i]['predict_sen'].item()
        predictions[i]['sem_pred_ppv'] = predictions_edgemask[i]['predict_ppv'].item()
        predictions[i]['sem_pred_acc'] = predictions_edgemask[i]['predict_acc'].item()

        # 综合分割的结果
        predictions[i]['combined_pred_labels'] = combined_pred_labels.cpu().tolist()
        predictions[i]['combined_pred_dsc'] = dsc.item()
        predictions[i]['combined_pred_sen'] = sen.item()
        predictions[i]['combined_pred_ppv'] = ppv.item()
        predictions[i]['combined_pred_acc'] = acc.item()
        predictions[i]['combined_pred_iou'] = iou.item()

    # 保存为json
    save_to_json(args, predictions)
    # 打印信息
    print_pred_metrics(predictions)


def get_elfsemnet_edgecnn_predictions(sample_ratio=0.01):
    """
    :param sample_ratio:
    :return:
        class_weights: (15,)
        predictions_edgemask: list
        predictions_elf: list
    """

    setup_seed(1111)
    # 全局参数
    args = get_args()
    # logger
    tb_logger = init_logger(args)
    # checkpoints
    checkpoint_callback, checkpoint_path = init_ckpts(args)
    args.sample_ratio = sample_ratio
    _, _, predict_ds = load_DBGA_dataset_with_tbase(sample_ratio=args.sample_ratio, train_val_ratio=1.0,
                                                    comb_name='comb_PointnetTrans',
                                                    is_full=True,
                                                    retain_file_path=None, rseed=42)
    predict_loader = GeoDataLoader(dataset=predict_ds, batch_size=1, shuffle=False, num_workers=args.num_workers,
                                   pin_memory=True)

    # 1. 加载ELFSemSegModule的预测结果，这是patch-based的预测结果
    config = get_cfg_from_args(args)
    model_elf = ELFSemSegModule(config)
    # 检查是否存在检查点文件
    model_elf, latest_checkpoint_path = load_latest_ckpts(checkpoint_path, model_elf)
    trainer = pl.Trainer()
    # 进行预测
    if 'latest_checkpoint_path' in locals():
        print(latest_checkpoint_path, locals())
        predictions_elf = trainer.predict(model_elf, dataloaders=predict_loader, ckpt_path=latest_checkpoint_path)
    else:
        predictions_elf = trainer.predict(model_elf, dataloaders=predict_loader)


    # 2. 加载EdgeCNN的预测结果，这是proposal-free的预测结果
    config_edgemask = get_cfg_from_args(args)
    config_edgemask['model_path'] = config_edgemask['second_segnet_name']
    checkpoint_path = config_edgemask['model_path']
    model_edgemask = EdgeCNN(config_edgemask)
    # 检查是否存在检查点文件
    model_edgemask, latest_checkpoint_path = load_latest_edge_masknet_ckpts(checkpoint_path, model_edgemask)
    # 进行预测
    if 'latest_checkpoint_path' in locals():
        print(latest_checkpoint_path, locals())
        predictions_edgemask = trainer.predict(model_edgemask, dataloaders=predict_loader,
                                               ckpt_path=latest_checkpoint_path)
    else:
        predictions_edgemask = trainer.predict(model_edgemask, dataloaders=predict_loader)

    return model_elf.class_weights, predictions_edgemask, predictions_elf


def test_get_model():
    # model
    args = get_args()
    config = get_cfg_from_args(args)
    model = ELFSemSegModule(config)
    # model = ELFSemSegModule(num_classes=15, num_feat=6, k=16)
    model.cuda()
    xyz = torch.rand(1250, 6).cuda()
    num_points = xyz.shape[0]  # 获取点的数量
    y = torch.randint(low=0, high=15, size=(num_points,), dtype=torch.long)
    data = Data(x=xyz, pos=xyz[:, :3], y=y, filename="aaa", faces=torch.tensor([[0, 1, 3]], dtype=torch.long))

    # output is B, C, N order
    outs = model(data)
    for item in outs:
        print(item.shape)


if __name__ == '__main__':
    # 示例使用
    # test_get_model()

    # test2_DGCNN()
    # test3_pred()
    # test3_pred_combined()
    test3_pred_combined_to_json()