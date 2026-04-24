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

import torch
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
from datasets.TGNDataset import load_datasets
from metrics.DBGA_metrics import *
from model.BaseNet import BaseNet  # 新的的BaseNet
from model._elfnet.bak.TripleBinClsNet_V1 import pred_from_json

from model._elfnet.cbl_point_transformer.cbl_point_transformer_module import get_model
from model._elfnet.point_transformer import PTInsSegModule
from model._elfnet.topo_clustering import get_patches

from utils.points_utils import *
# from model._edge_mask.edge_mask_V3_triple_bin_cls import TripleBinClsModule
# from model._dynamic_duo.utils.loss_meter import LossMap
from model._elfnet.crop_patch import crop_patches_eef, get_ddf, get_max_frequency_non_zero_class
# from model._elfnet.ELFNetSeg import pred_from_json
from external_libs import gen_utils as gu, ops_utils as ou

class PointTransformerNet(torch.nn.Module):
    def __init__(self, num_classes = 2, num_feat = 6):
        self.input_dim = num_feat
        self.output_dim = num_classes

        model_parameter = {
            "input_feat": num_feat,
            "stride": [1, 4, 2, 2, 4],
            "nstride": [2, 2, 2, 2],
            "nsample": [36, 24, 24, 24, 24],
            "blocks": [2, 3, 4, 6, 3],
            "block_num": 5,
            "planes": [32, 64, 128, 256, 256],
            "contain_weight": False,
            "crop_sample_size": 3072,
        }

        super().__init__()
        self.PT_model = get_model(c=self.input_dim, k=self.output_dim, **model_parameter).to(device)

    def forward(self, inputs):
        """
        inputs
            inputs[0] => B, 6, 24000 : point features
            inputs[1]=> B, 1, 24000 : ground truth segmentation
        """
        # B, C, N = inputs[0].shape
        outputs = {}

        if len(inputs)>= 2:
            cbl_loss, sem_1, offset_1, mask_1, first_features = self.PT_model([inputs[0],inputs[1]])
            outputs.update({
                "cbl_loss": cbl_loss,
                "sem_1": sem_1,    # [1,2,24000]
                "offset_1": offset_1,
                "mask_1": mask_1,
                "first_features": first_features,
                "cls_pred": sem_1
            })
        else:
            sem_1, offset_1, mask_1, first_features = self.PT_model([inputs[0]])
            outputs.update({
                "sem_1": sem_1,  # [1,2,24000]
                "offset_1": offset_1,
                "mask_1": mask_1,
                "first_features": first_features,
                "cls_pred": sem_1
            })

        return outputs

class ELFSemSegModule(BaseNet):
    """ edge logits field semantic segmentation module"""

    def __init__(self, config):
        super().__init__(config)

        self.num_classes = config.get('num_classes')

        # 第一个分割模型(仅为推理模型，trainable=false)
        self.first_segnet_model = PTInsSegModule(config, num_classes=15)
        first_segnet_model_path = config.get('first_segnet_name')
        # self.first_segnet_model, latest_ckpt_path = ELFSemSegModule.load_latest_ckpts_to_model(first_segnet_model_path,
        #                                                                                        self.first_segnet_model)
        self.first_segnet_model.model, latest_ckpt_path = CheckPtMgr.load_latest_ckpts(first_segnet_model_path,
                                                                                       self.first_segnet_model.model)

        # 第二个分割模型（实例分割模型，trainable=true）
        self.ins_segnet_model = PointTransformerNet(num_classes=2,num_feat=6).to(device)
        self.ins_segnet_model = nn.DataParallel(self.ins_segnet_model)


        if latest_ckpt_path is not None:
            print(f"Load first_segnet_model from {latest_ckpt_path}")
        # 设定 self.ins_segnet_model 不参与梯度更新
        for param in self.first_segnet_model.parameters():
            param.requires_grad = False

        self.pred_results_json_path = config.get('pred_results_json_path')
        assert(self.pred_results_json_path is not None)

        # cross entropy loss
        self.xe_loss = nn.CrossEntropyLoss()


    def common_step(self, batch):
        """
        该函数侧重于数据转换及处理、计算损失，不涉及dsc、ppv、sen等指标的计算。
        """

        # 数据格式转换
        points = batch.x.to(self.device,dtype=torch.float)  # 24000,6
        xyz = points[:, :3]  # 24000,3
        # l0_points = pred_PT['l0_points'] # 24000,6
        gt_labels = batch.y.to(self.device, dtype=torch.long).reshape([-1,1]) # 24000,1

        # 第一个分割模型推理得到结果
        pred_PT = self.first_segnet_model.predict_step(batch, 0)

        # 第一个分割模型推理得到结果
        base_name = batch.filename[0] # batch.filename是个列表
        pred_clu,_ = pred_from_json(self.pred_results_json_path, base_name)
        assert(pred_clu is not None)

        pred_clu['clu_labels_all'] = torch.tensor(pred_clu['clu_labels_all'])
        pred_clu['clu_num'] = torch.tensor(pred_clu['clu_num'])
        pred_clu['clu_cen'] = torch.tensor(pred_clu['clu_cen'])
        pred_clu['clu_counts'] = torch.tensor(pred_clu['clu_counts'])
        pred_clu['clu_gt_map'] = torch.tensor(pred_clu['clu_gt_map'])
        pred_clu['counts_mat'] = torch.tensor(pred_clu['counts_mat'])
        pred_clu['gt_clu_num_prop'] = torch.tensor(pred_clu['gt_clu_num_prop'])
        # pred_clu['sep_teeth_mask'] = torch.tensor(pred_clu['sep_teeth_mask'])


        opts = {'crop_num': 3072}
        pred_clu_new = get_patches(pred_clu, xyz.cpu().numpy(), gt_labels.cpu().numpy(), opts)
        clu_gt_bin_ls = pred_clu_new['clu_gt_bin_ls'] # clu, 1, 3072
        # clu_gt_label = pred_clu_new['clu_gt_label'] # clu, 1
        nn_crop_indexes = pred_clu_new['nn_crop_indexes'] # 1, clu, 3072
        # clu_cen = pred_clu_new['clu_cen']

        # clu_feat_tensor = torch.from_numpy(clu_cen).to(points.device)  # Clu,3
        ins_seg_pred = pred_PT['predictions']

        if len(xyz.shape) == 2: xyz = xyz.unsqueeze(dim=0)  # 1,10000,3
        if len(points.shape) == 2: points = points.unsqueeze(dim=0)  # 1,10000,6
        # if len(l0_points.shape) == 2: l0_points = l0_points.unsqueeze(dim=0)  # 1,10000,32
        if len(gt_labels.shape) == 2: gt_labels = gt_labels.unsqueeze(dim=0)  # 1,10000,1

        cropped_input_ls = ou.get_indexed_features(points.permute(0, 2, 1), nn_crop_indexes)  # clu, 6, 1500
        # cropped_feature_ls = ou.get_indexed_features(l0_points.permute(0, 2, 1), nn_crop_indexes)  # clu, 32, 1500


        # ddf = get_ddf(cropped_input_ls[:, :3, :].permute(0, 2, 1),
        #               torch.tensor(np.expand_dims(clu_cen, axis=0)).permute(1, 0, 2).to(cropped_input_ls.device))
        # cropped_feature_ls = torch.cat([cropped_input_ls, ddf],axis=1).float()
        cropped_feature_ls = cropped_input_ls
        cropped_feature_ls = cropped_feature_ls.to(device)  # clu, 12, 3072
        clu_gt_bin_ls = torch.from_numpy(clu_gt_bin_ls.astype(np.int64)).to(device) # clu,1,3072

        inputs = [cropped_feature_ls, clu_gt_bin_ls]
        # 语义分割网络
        outputs = self.ins_segnet_model(inputs)

        pred_clu_sem = outputs['sem_1'].permute([0,2,1]).reshape([-1, 2]) # (Clu*3072, 2)
        clu_gt_bin_ls = clu_gt_bin_ls.reshape([-1, 1]).squeeze()
        seg_loss = self.xe_loss(pred_clu_sem, clu_gt_bin_ls)
        cbl_loss = outputs['cbl_loss'].sum() # (Clu,5)
        loss = seg_loss + cbl_loss

        pred_clu_tar = torch.argmax(pred_clu_sem, dim=1)  # (Clu*3072,)
        pred_clu_tar_mask = (pred_clu_tar==1).reshape([cropped_feature_ls.shape[0],-1,1]) # (Clu，3072,1)

        # pred_clu_sem = pred_clu_sem.reshape([cropped_feature_ls.shape[0],-1,self.num_classes])

        return (cropped_feature_ls, ins_seg_pred, points, loss, seg_loss, cbl_loss,
                pred_clu_tar_mask,pred_clu_new)

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
            cropped_feature_ls: clu, 12, 1500
            pred_clu_tar_mask: clu,1500,1
        """
        # batch = self.normalize(batch)
        # 真实类标
        gt_labels = batch.y.to(self.device, dtype=torch.long)  # [N,]
        one_hot_labels = nn.functional.one_hot(gt_labels, num_classes=self.class_weights.size(0))  # [N, 15]
        one_hot_labels = one_hot_labels.unsqueeze(dim=0)

        outs = self.common_step(batch)
        (cropped_feature_ls, ins_seg_pred, points, loss, seg_loss, cbl_loss,
         pred_clu_tar_mask, pred_clu) = outs

        clu_gt_bin_ls = pred_clu['clu_gt_bin_ls'] # clu, 1, 3072
        clu_gt_label = pred_clu['clu_gt_label'] # clu, 1
        nn_crop_indexes = pred_clu['nn_crop_indexes'] # 1, clu, 3072

        # gu.print_3d(gu.np_to_pcd_with_label(gu.torch_to_numpy(points)[0,:,:3],
        #                                     gu.torch_to_numpy(ins_seg_pred.argmax(axis=1)).reshape(-1) ))
        # 最大投票法决出ins_pred_labels
        crop_mat = torch.zeros((points.shape[1], cropped_feature_ls.shape[0]), dtype=torch.long,
                               device=points.device)  # 24000 * 15
        ratios = torch.ones(cropped_feature_ls.shape[0], device=points.device)

        for i in range(0, cropped_feature_ls.shape[0]):
            pred_bin_labels = torch.zeros(cropped_feature_ls[i].shape[1], device=points.device)
            pred_bin_labels[pred_clu_tar_mask[i].reshape(-1) > 0] = 1
            # gu.print_3d(gu.np_to_pcd_with_label(gu.torch_to_numpy(cropped_feature_ls)[i,:3,:].T, pred_bin_labels))

            positive_indices = nn_crop_indexes[0][i][pred_bin_labels.cpu().numpy() == 1]
            if positive_indices.size > 0:
                selected_indices = (ins_seg_pred.argmax(axis=1))[positive_indices]
                unique, counts = torch.unique(selected_indices, return_counts=True)
                filtered_id_pred = unique[torch.argmax(counts)]
                crop_mat[positive_indices, i] = filtered_id_pred

                ratio = counts.max().item() / counts.sum().item()  # 计算最大类占比
                ratios[i] = ratio

        low_ratio_count = (ratios < 0.7).sum().item()

        if low_ratio_count > 0:
            ins_pred_labels = get_max_frequency_non_zero_class(crop_mat.T.cpu().numpy())
            ins_pred_labels = torch.from_numpy(ins_pred_labels).long().to(self.device)  # Fixed warning
            print(low_ratio_count)
        else:
            # 提取每行的最大值（预测类别）
            _, ins_pred_labels = torch.max(ins_seg_pred, dim=1)  # [N,]
            ins_pred_labels = ins_pred_labels.squeeze().long().to(self.device)  # 去除多余的维度

        # ins_pred_labels = torch.tensor(ins_pred_labels).long().to(self.device)
        ins_pred_labels_onehot = nn.functional.one_hot(ins_pred_labels, num_classes=self.class_weights.size(0))

        dsc = weighting_DSC(ins_pred_labels_onehot.unsqueeze(dim=1), one_hot_labels, self.class_weights)
        sen = weighting_SEN(ins_pred_labels_onehot.unsqueeze(dim=1), one_hot_labels, self.class_weights)
        ppv = weighting_PPV(ins_pred_labels_onehot.unsqueeze(dim=1), one_hot_labels, self.class_weights)
        acc = weighting_ACC(ins_pred_labels_onehot.unsqueeze(dim=1), one_hot_labels, self.class_weights)
        iou = weighting_IOU(ins_pred_labels_onehot.unsqueeze(dim=1), one_hot_labels, self.class_weights)

        return (dsc, loss, ppv, sen, acc, iou,
                ins_pred_labels, pred_clu_tar_mask,pred_clu,
                seg_loss, cbl_loss)

    def forward(self, x):
        outs = self.common_step(x)
        loss= outs[3]
        return loss

    def validation_step(self, batch, batch_idx):
        outs = self._update_step_metrics(batch)
        dsc, loss, ppv, sen, acc, iou, = outs[0], outs[1], outs[2], outs[3], outs[4], outs[5]
        seg_loss, cbl_loss = outs[-2],outs[-1]

        def get_mesh_name(file_path):
            file_name = os.path.basename(file_path)
            folder_name = os.path.basename(os.path.dirname(file_path))
            return f"{file_name}_{folder_name}"

        mesh_name = get_mesh_name(batch.filename[0])

        print(f"epoch:{self.current_epoch}-{mesh_name}, "
              f"loss:{loss:.4f}=seg_loss:{seg_loss:.4f} + cbl_loss:{cbl_loss:.4f}",
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
        ins_pred_labels = outputs[6]  # (10000,1)
        pred_clu_tar_mask = outputs[7]
        pred_clu = outputs[8]

        mesh_file = batch.filename
        pts_normals = batch.x
        # faces = batch.faces
        gt_labels = batch.y.to(self.device, dtype=torch.long)

        outs = {
            'predict_loss': loss,
            'predict_dsc': dsc,
            'predict_sen': sen,
            'predict_ppv': ppv,
            'predict_acc': acc,
            'pts_normals': pts_normals,
            'gt_labels': gt_labels,
            'mesh_file': mesh_file,
            'ins_pred_labels': ins_pred_labels,  # (24000,1)， 0-14， 预测的实例标签
            'pred_clu': pred_clu,  # 这是一个字典，包含clu_gt_bin_ls, clu_gt_label, nn_crop_indexes等
            'pred_clu_tar_mask': pred_clu_tar_mask,  # (1, clu, 3072)， 每个clu的crop点的预测标签
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

def create_training_args():
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
        first_segnet_name=r'exp_point_transformer', # 三个elf的模型
        second_segnet_name=r'exp_edge_mask_V2_skip',  # 记得更改import，exp_edge_mask_atten_rang2，语义分割结果
        pred_results_json_path=r"C:\Users\win10\DualMesh\pynb\teeth3ds\model\results\exp_TripleBinClsNet\2classes_30epochs_10000ncells_0.1ratio_0.03_rmin0.03_rmax_0.001lr_topo_clu",
        rmin=0.03,
        rmax=0.04,
        num_classes=15,
        num_neighbor=32,
        num_epochs=30,
        num_workers=16,
        accumulate_grad_batches=1,  # 梯度累积步数
        clu_batch_size=1,  # 每个小批次的大小
        sample_ratio=1.0,
        train_val_ratio=1.0,
        comb_name = 'comb_fps_24000',
        train_batch_size=1,  # 训练批次
        val_batch_size=1,  # 测试批次
        lr=0.001
    )

def create_model_config(args):
    """代码兼容性的需要，需将args转换为config"""
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
        'pred_results_json_path': args.pred_results_json_path,
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
        checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_path + '/' + exp_name,
            filename='model-{epoch:02d}-{test_loss:.2f}',
            save_top_k=1,
            save_last=True,
            monitor='test_loss',
            mode='min'
        )
        # 检查是否存在检查点文件
        checkpoint_dir = ckpt_path + '/' + exp_name

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)

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

            return (model, None) if not checkpoint_files else None  # 然后继续后续逻辑

            # 排序并获取最新的检查点文件
            checkpoint_files.sort(key=os.path.getmtime)
            latest_ckpt_path = checkpoint_files[-1]
            checkpoint = torch.load(latest_ckpt_path)

            # 因为模型有可能是在DataParallel包装下训练的，所以检查点文件中的state_dict键名有可能有前缀'model.'
            new_state_dict = {}
            for key, value in checkpoint['state_dict'].items():
                if key.startswith('model.'):
                    new_key = key[6:]  # Remove 'model.'
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value

            if isinstance(model, nn.DataParallel):
                model.module.load_state_dict(new_state_dict)
            else:
                model.load_state_dict(new_state_dict)

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

        results_path = PredMgr.get_results_path(args)
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
            # dir_name = os.path.basename(os.path.dirname(filename))
            new_filename = f"{base_name}.json"
            # 构造保存路径
            save_path = os.path.join(results_path, new_filename)

            # 保存结果
            with open(save_path, 'w') as f:
                json.dump(serializable_pred, f)
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

    @staticmethod
    def get_elfnet_predictions(sample_ratio=0.01):
        """
        :param sample_ratio:
        :return:
            class_weights: (15,)
            predictions_edgemask: list
            predictions_elf: list
        """

        Utils.setup_seed(1111)
        rseed = 42
        # 全局参数
        args = create_training_args()
        # logger
        tb_logger = LoggerMgr.init_logger(args)
        # checkpoints
        checkpoint_callback, checkpoint_path = CheckPtMgr.init_ckpts(args.ckpt_path, args.exp_name)
        args.sample_ratio = sample_ratio
        # _, _, predict_ds = load_DBGA_dataset_with_tbase(sample_ratio=args.sample_ratio, train_val_ratio=1.0,
        #                                                 comb_name=args.comb_name,
        #                                                 is_full=True,
        #                                                 retain_file_path=None, rseed=42)
        _, _, predict_ds = load_datasets(sample_ratio=args.sample_ratio, comb_name=args.comb_name, rseed=rseed)
        predict_loader = GeoDataLoader(dataset=predict_ds, batch_size=1, shuffle=False, num_workers=args.num_workers,
                                       pin_memory=True)

        # 1. 加载ELFSemSegModule的预测结果，这是patch-based的预测结果
        config = create_model_config(args)
        model_elf = ELFSemSegModule(config)
        # 检查是否存在检查点文件
        model_elf, latest_checkpoint_path = CheckPtMgr.load_latest_ckpts(checkpoint_path, model_elf)
        trainer = pl.Trainer()
        # 进行预测
        if 'latest_checkpoint_path' in locals():
            print(latest_checkpoint_path, locals())
            predictions_elf = trainer.predict(model_elf, dataloaders=predict_loader, ckpt_path=latest_checkpoint_path)
        else:
            predictions_elf = trainer.predict(model_elf, dataloaders=predict_loader)

        return model_elf.class_weights, predictions_elf



def do_train():
    # global num_classes, val_batch_size, lr, start, model
    Utils.setup_seed(1111)
    rseed = 42
    # rseed = 27

    args = create_training_args() # 全局参数
    tb_logger = LoggerMgr.init_logger(args)  # logger
    checkpoint_callback, checkpoint_path = CheckPtMgr.init_ckpts(args.ckpt_path, args.exp_name) # checkpoints
    # datasets
    # train_ds, val_ds, test_ds = load_DBGA_dataset_with_tbase(sample_ratio=args.sample_ratio, train_val_ratio=1.0,
    #                                                          comb_name='comb_PointnetTrans',
    #                                                          is_full=False,
    #                                                          retain_file_path=None, rseed=rseed)
    train_ds, _, test_ds = load_datasets(sample_ratio=args.sample_ratio,comb_name=args.comb_name,rseed=rseed)
    train_loader = GeoDataLoader(dataset=train_ds, batch_size=args.train_batch_size, shuffle=True,
                                 num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
    test_loader = GeoDataLoader(dataset=test_ds, batch_size=args.val_batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True, persistent_workers=True)

    # model
    config = create_model_config(args)
    model = ELFSemSegModule(config)

    model, latest_checkpoint_path = CheckPtMgr.load_latest_ckpts(checkpoint_path, model)
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


def do_pred():
    Utils.setup_seed(1111)
    # 全局参数
    args = create_training_args()
    # logger
    tb_logger = LoggerMgr.init_logger(args)
    # checkpoints
    checkpoint_callback, checkpoint_path = CheckPtMgr.init_ckpts(args.ckpt_path, args.exp_name)

    args.sample_ratio = 1.0
    _, _, predict_ds = load_DBGA_dataset_with_tbase(sample_ratio=args.sample_ratio, train_val_ratio=1.0,
                                                    comb_name='comb_PointnetTrans',
                                                    retain_file_path=None, rseed=42)
    predict_loader = GeoDataLoader(dataset=predict_ds, batch_size=1, shuffle=False, num_workers=args.num_workers,
                                   pin_memory=True)

    config = create_model_config(args)
    model = ELFSemSegModule(config)

    # 检查是否存在检查点文件
    model, latest_checkpoint_path = CheckPtMgr.load_latest_ckpts(checkpoint_path, model)

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


def do_pred_combined():
    class_weights, predictions_edgemask, predictions_elf = PredMgr.get_elfnet_predictions()

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

def test3_pred_combined_to_json():

    Utils.setup_seed(1111)
    # 全局参数
    args = create_training_args()

    class_weights, predictions_elf = PredMgr.get_elfnet_predictions(sample_ratio=1.0)
    predictions = predictions_elf

    for i in range(len(predictions)):

        gt_labels = predictions[i]['gt_labels']  # 真实标签
        xyz = predictions_elf[i]['pts_normals'][:, :3]
        ins_pred_labels = predictions[i]['ins_pred_labels']  # 实例预测标签

        combined_pred_labels = torch.tensor(ins_pred_labels).long().to(xyz.device)
        pred_labels_onehot = nn.functional.one_hot(combined_pred_labels, num_classes=args.num_classes)
        one_hot_labels = nn.functional.one_hot(gt_labels, num_classes=args.num_classes)

        dsc = weighting_DSC(pred_labels_onehot.unsqueeze(dim=0), one_hot_labels.unsqueeze(dim=0), class_weights)
        sen = weighting_SEN(pred_labels_onehot.unsqueeze(dim=0), one_hot_labels.unsqueeze(dim=0), class_weights)
        ppv = weighting_PPV(pred_labels_onehot.unsqueeze(dim=0), one_hot_labels.unsqueeze(dim=0), class_weights)
        acc = weighting_ACC(pred_labels_onehot.unsqueeze(dim=0), one_hot_labels.unsqueeze(dim=0), class_weights)
        iou = weighting_IOU(pred_labels_onehot.unsqueeze(dim=0), one_hot_labels.unsqueeze(dim=0), class_weights)


        # 综合分割的结果
        predictions[i]['combined_pred_labels'] = combined_pred_labels.cpu().tolist()
        predictions[i]['combined_pred_dsc'] = dsc.item()
        predictions[i]['combined_pred_sen'] = sen.item()
        predictions[i]['combined_pred_ppv'] = ppv.item()
        predictions[i]['combined_pred_acc'] = acc.item()
        predictions[i]['combined_pred_iou'] = iou.item()

    # 保存为json
    PredMgr.save_to_json(args, predictions)
    # 打印信息
    PredMgr.print_pred_metrics(predictions)





def test_get_model():
    # model
    args = create_training_args()
    config = create_model_config(args)
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

def test_PointTransformerNet():
    PTNet = PointTransformerNet(num_classes=2, num_feat=7)
    inputs = 0.1 * torch.rand(14, 7, 3072).cuda()  # BCN格式
    targets = torch.randint(0, 2, (14, 1, 3072)).cuda()  # BCN格式
    outputs = PTNet([inputs, targets])
    xe_loss = nn.CrossEntropyLoss()

    pred_clu_sem = outputs['sem_1'].permute([0, 2, 1]).reshape([-1, 2])  # (Clu*3072, 2)
    clu_gt_bin_ls = targets.reshape([-1, 1]).squeeze()
    seg_loss = xe_loss(pred_clu_sem, clu_gt_bin_ls)
    cbl_loss = outputs['cbl_loss'].sum()  # (Clu,3)
    loss = seg_loss + cbl_loss

    print(outputs)
    print(loss)

if __name__ == '__main__':
    # 示例使用
    # test_get_model()

    do_train()
    # test3_pred()
    # test3_pred_combined()
    # test3_pred_combined_to_json()

    # test_PointTransformerNet()