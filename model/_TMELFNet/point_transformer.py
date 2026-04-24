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

# from datasets.misc.DBGAMeshDataset import load_DBGA_dataset,load_DBGA_dataset_with_tbase
# from datasets.TGNDataset import load_datasets
from metrics.DBGA_metrics import *
# from model.DBGA_base_model import BaseNet
# from model.BaseNet import BaseNet # 新的的BaseNet
from metrics.dice_loss import DiceLoss

import torch
from torch_cluster import radius_graph

from utils.points_utils import compute_polar_angles, compute_tae_gt_bin_labels_ang
from model._TMELFNet.cbl_point_transformer.cbl_point_transformer_module import get_model

class PointTransformerNet(torch.nn.Module):
    def __init__(self, num_classes = 15, num_feat = 7):
        self.input_dim = num_feat
        self.output_dim = num_classes

        model_parameter = {
            "input_feat": 7,
            "stride": [1, 4, 4, 4, 4],
            "nstride": [2, 2, 2, 2],
            "nsample": [36, 24, 24, 24, 24],
            "blocks": [2, 3, 4, 6, 3],
            "block_num": 5,
            "planes": [32, 64, 128, 256, 512],
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
                "sem_1": sem_1,    # [1,15,24000]
                "offset_1": offset_1,
                "mask_1": mask_1,
                "first_features": first_features,
                "cls_pred": sem_1
            })
        else:
            sem_1, offset_1, mask_1, first_features = self.PT_model([inputs[0]])
            outputs.update({
                "sem_1": sem_1,
                "offset_1": offset_1,
                "mask_1": mask_1,
                "first_features": first_features,
                "cls_pred": sem_1
            })

        return outputs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#
# class PTInsSegModule(BaseNet):
#     def __init__(self, config, num_classes = 15):
#         super().__init__(config)
#
#         self.num_classes = num_classes
#         self.input_dim = 7
#
#         # 模型
#         # self.model = get_seg_model(num_classes=num_classes,num_feat=6,k=16)
#         self.model = PointTransformerNet(num_classes=self.num_classes,num_feat=self.input_dim)
#         self.model = nn.DataParallel(self.model)
#
#         # cross entropy loss
#         self.xe_loss = nn.CrossEntropyLoss()
#         # dice loss
#         self.dice_loss = DiceLoss()
#
#
#     def on_train_batch_start(self, batch, batch_idx):
#         # 在每个训练批次开始前清理内存（可选，但可能过于频繁）
#         if batch_idx % 60 == 0:  # 每30个批次清理一次
#             torch.cuda.empty_cache()
#             print(f"Epoch end, GPU内存清理完成: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
#     def on_validation_batch_start(self, batch, batch_idx):
#         # 在每个训练批次开始前清理内存（可选，但可能过于频繁）
#         if batch_idx % 60 == 0:  # 每30个批次清理一次
#             torch.cuda.empty_cache()
#             print(f"Epoch end, GPU内存清理完成: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
#
#     def _update_step_metrics(self, batch, phase):
#         # batch = self.normalize(batch)
#
#         x0 = batch.x.to(device, dtype=torch.float32) # NC
#         gt_labels = batch.y.to(device, dtype=torch.long) # N
#         one_hot_labels = nn.functional.one_hot(gt_labels, num_classes=self.num_classes).unsqueeze(dim=0) # BNC
#
#         B= 1
#         # Ensure all input tensors are on the same device
#         inputs = [x0.permute([1,0]).reshape(B, self.input_dim, -1), gt_labels.reshape([B, 1, -1])]
#         # inputs = [tensor.to(device) for tensor in inputs]
#
#         # 调用网络模型获取结果
#         if phase == "train":
#             outputs = self.model(inputs)  # BCN格式
#         else:
#             with torch.no_grad():
#                 outputs = self.model(inputs)  # BCN格式
#
#         # outputs = self.model([x0.permute(1,0).unsqueeze(dim=0),gt_labels.reshape([B,1,-1])]) # BCN格式
#         # outputs = self.model([x0.permute(1,0).unsqueeze(dim=0)]) # BCN格式
#
#
#         # semantic segmentation loss
#         gt_labels = gt_labels.view(-1, 1)[:, 0]
#         # outputs1 = outputs['sem_1'].contiguous().view(-1, self.num_classes)
#         outputs1 = outputs['sem_1'].permute([0,2,1]).reshape([-1, self.num_classes])
#         seg_loss = self.xe_loss(outputs1, gt_labels)
#         # dice_loss
#         dice_loss = self.dice_loss(outputs1, gt_labels)
#         # dice_loss = 0
#
#         def get_alpha_beta(epoch):
#             alpha = 0.3 if epoch >= 20 else 0.7 - (0.7 - 0.3) * (epoch / 20)
#             beta = 0.7 if epoch >= 20 else 0.3 + (0.7 - 0.3) * (epoch / 20)
#             return alpha, beta
#
#         # loss = seg_loss + mat_diff_loss * 0.01
#         # alpha, beta = get_alpha_beta(self.current_epoch)
#         # loss = alpha * seg_loss + beta * dice_loss
#         alpha,beta = 1,0
#
#         cbl_loss = outputs['cbl_loss'].sum()
#         l2_loss = self.regularization_losses()
#
#         # loss = seg_loss + mat_diff_loss * 0.01
#         lambda0, lambda1, lambda2, lambda3 = 2, 1, 1, 1
#         # loss = lambda0*(alpha * seg_loss + beta * dice_loss) + l2_loss + cbl_loss
#         loss = seg_loss  + cbl_loss
#         # loss = seg_loss
#
#
#         dsc = weighting_DSC(outputs1.unsqueeze(dim=0), one_hot_labels, self.class_weights)
#         sen = weighting_SEN(outputs1.unsqueeze(dim=0), one_hot_labels, self.class_weights)
#         ppv = weighting_PPV(outputs1.unsqueeze(dim=0), one_hot_labels, self.class_weights)
#         acc = weighting_ACC(outputs1.unsqueeze(dim=0), one_hot_labels, self.class_weights)
#         iou = weighting_IOU(outputs1.unsqueeze(dim=0), one_hot_labels, self.class_weights)
#
#         return (dsc, loss, ppv, sen, acc, iou, seg_loss, dice_loss, cbl_loss, alpha, beta)
#
#     def training_step(self, batch, batch_idx):
#         outputs = self._update_step_metrics(batch,phase="train")
#         dsc, loss, ppv, sen, acc, iou = outputs[0], outputs[1], outputs[2], outputs[3], outputs[4], outputs[5]
#
#         metrics = {
#             'train_loss': loss,
#             'train_dsc': dsc,
#             'train_sen': sen,
#             'train_ppv': ppv,
#             'train_acc': acc,
#             'train_iou': iou
#         }
#
#         self.log_dict(metrics)
#         self.training_step_outputs.append(metrics)
#         return loss
#
#     def validation_step(self, batch, batch_idx):
#         outputs = self._update_step_metrics(batch,"val")
#         dsc, loss, ppv, sen, acc, iou = outputs[0], outputs[1], outputs[2], outputs[3], outputs[4], outputs[5]
#         seg_loss, dice_loss, cbl_loss = outputs[6], outputs[7], outputs[8]
#         alpha, beta = outputs[9], outputs[10]
#
#         print(f"epoch:{self.current_epoch}-{batch.filename[0]}, "
#               f"loss:{loss:.4f}=Lseg:{alpha:.2f}*{seg_loss:.4f} + Ldice:{beta:.2f}*{dice_loss:.4f} "
#               f"+Lcbl:{cbl_loss}", end=' ')
#
#         metrics = {
#             'test_loss': loss,
#             'test_dsc': dsc,
#             'test_sen': sen,
#             'test_ppv': ppv,
#             'test_acc': acc,
#             'test_iou': iou
#         }
#
#         # print(metrics)
#
#         self.log_dict(metrics)
#         self.validation_step_outputs.append(metrics)
#         return metrics
#
#
#     def predict_step(self, batch, batch_idx):
#         outputs = self._update_step_metrics(batch,"val")
#         dsc, loss, ppv, sen, acc = outputs[0], outputs[1], outputs[2], outputs[3], outputs[4]
#
#         filename = batch.filename
#         pts_normals = batch.x
#         # faces = batch.faces
#         x0 = batch.x.to(device, dtype=torch.float32)  # NC
#         gt_labels = batch.y.to(device, dtype=torch.long)  # N
#
#         B = 1
#         # Ensure all input tensors are on the same device
#         inputs = [x0.permute([1, 0]).reshape(B, self.input_dim, -1), gt_labels.reshape([B, 1, -1])]
#
#
#         outs = self.model(inputs)
#         outputs1 = outs['sem_1'].permute([0,2,1]).reshape([-1, self.num_classes])
#         l0_points = outs['first_features']
#         _, pred_labels = torch.max(outputs1, 1)
#
#         metrics = {
#             'predict_loss': loss,
#             'predict_dsc': dsc,
#             'predict_sen': sen,
#             'predict_ppv': ppv,
#             'predict_acc': acc,
#             'pts_normals': pts_normals,
#             # 'faces': faces,
#             'gt_labels': gt_labels,
#             'pred_labels': pred_labels,
#             'l0_points':l0_points,
#             'mesh_file': filename,
#             'predictions': outputs1
#         }
#
#         return metrics
#
#     def regularization_losses(self):
#         #####################
#         # Regularization loss
#         #####################
#
#         # 计算 L2 正则化损失
#         l2_lambda = 0.0001  # 正则化系数，可根据实际情况调整
#         l2_reg = torch.tensor(0., requires_grad=True).to(self.device)
#         for name, param in self.model.named_parameters():
#             if 'weight' in name:
#                 l2_reg = l2_reg + torch.norm(param, 2)
#
#         l2_loss = l2_lambda * l2_reg
#         return l2_loss
#
#
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# def setup_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#
# def get_args():
#     """
#     此函数用于创建并返回一个包含训练配置参数的 Namespace 对象。
#     返回:
#         argparse.Namespace: 包含训练配置参数的命名空间对象。
#     """
#     current_file_name = os.path.splitext(os.path.basename(__file__))[0]
#
#     return argparse.Namespace(
#         kfold_nums=5,
#         use_fold=1,
#         ncells=10000,
#         logger_path=r'../tb_logs/',
#         exp_name=r'exp_' + current_file_name,
#         ckpt_path=r'../checkpoints/',
#         ckpt_name=r'latest_checkpoint.tar',
#         results_path=r'../results/',
#         with_base_lower_path=r"E:\mesh_datasets\3DTeeth\Teeth3DS_train_test_split\with_base_lower.txt",
#         wo_base_lower_path=r"E:\mesh_datasets\3DTeeth\Teeth3DS_train_test_split\wo_base_lower.txt",
#         split_600_path=r"E:\mesh_datasets\3DTeeth\Teeth3DS_train_test_split\split_600_lower.txt",
#         num_classes=15,
#         num_neighbor=32,
#         num_epochs=30,
#         num_workers=16,
#         sample_ratio = 1.0,
#         train_val_ratio = 1.0,
#         accumulate_grad_batches = 1,
#         comb_name = 'comb_fps_24000',
#         train_batch_size=1,
#         val_batch_size=1,
#         lr=0.001
#     )
# def get_cfg_from_args(args):
#     config = {
#         'num_classes': args.num_classes,
#         'num_neighbor': args.num_neighbor,
#         'lr': args.lr,
#         'num_epochs': args.num_epochs,
#         'num_batches_to_print': 150,
#         'model_path': args.ckpt_path + '/' + args.version,
#         'checkpoint_name': args.ckpt_name,
#         'train_loss_csv_file': 'losses.csv',
#         'save_model_threshold': 0.9
#     }
#     return config
#
# def init_ckpts(args):
#     checkpoint_callback = ModelCheckpoint(
#         dirpath=args.ckpt_path + '/' + args.version,
#         filename='model-{epoch:02d}-{test_loss:.2f}',
#         save_top_k=1,
#         save_last=True,
#         monitor='test_loss',
#         mode='min'
#     )
#     # 检查是否存在检查点文件
#     checkpoint_dir = args.ckpt_path + '/' + args.version
#
#     if not os.path.exists(checkpoint_dir):
#         os.makedirs(checkpoint_dir, exist_ok=True)
#
#     return checkpoint_callback, checkpoint_dir
#
# def load_latest_ckpts(checkpoint_path, model):
#     if os.path.exists(checkpoint_path) and len(os.listdir(checkpoint_path)) > 0:
#         # 获取最新的检查点文件
#         checkpoint_files = [os.path.join(checkpoint_path, f) for f in os.listdir(checkpoint_path) if f.endswith('.ckpt')]
#
#         if len(checkpoint_files) < 1:
#             return model, None
#
#         checkpoint_files.sort(key=os.path.getmtime)
#         latest_checkpoint_path = checkpoint_files[-1]
#         checkpoint = torch.load(latest_checkpoint_path)
#
#         # Remove the extra 'model.' prefix from the state_dict
#         new_state_dict = {}
#         for key, value in checkpoint['state_dict'].items():
#             if key.startswith('model.'):
#                 new_key = key[6:]  # Remove 'model.'
#                 new_state_dict[new_key] = value
#             else:
#                 new_state_dict[key] = value
#
#         model.model.load_state_dict(new_state_dict)
#         return model,latest_checkpoint_path
#
#     return model, None
# def init_logger(args):
#     model_name_path = '{}/{}classes_{}k_{}epochs_{}ncells_{}batchsize_{}kfold{}_{}lr'
#     model_name_path = model_name_path.format(args.version, args.num_classes, args.num_neighbor, args.num_epochs,
#                                              args.ncells,
#                                              args.train_batch_size, args.kfold_nums, args.use_fold, args.lr)
#     logger_path = os.path.join(args.logger_path, model_name_path)
#     if not os.path.exists(logger_path):
#         os.makedirs(logger_path, exist_ok=True)
#     tb_logger = TensorBoardLogger(save_dir=args.logger_path, name=model_name_path)
#     return tb_logger
#
# def get_results_path(args):
#     model_name_path = '{}/{}classes_{}epochs_{}ncells_{}batchsize_{}kfold{}_{}lr'
#     model_name_path = model_name_path.format(args.version, args.num_classes, args.num_epochs,
#                                              args.ncells,
#                                              args.train_batch_size, args.kfold_nums, args.use_fold, args.lr)
#     results_path = os.path.join(args.results_path, model_name_path)
#     return results_path
#
# def do_train():
#     # global num_classes, val_batch_size, lr, start, model
#     setup_seed(1111)
#     rseed = 42
#     # rseed = 27
#     # 全局参数
#     args = get_args()
#     #logger
#     tb_logger = init_logger(args)
#     #checkpoints
#     checkpoint_callback, checkpoint_path = init_ckpts(args)
#     # datasets
#     # train_ds, val_ds, test_ds = load_DBGA_dataset_with_tbase(sample_ratio=args.sample_ratio, train_val_ratio=1.0,
#     #                                                          comb_name='comb_PointnetTrans', is_full = True,
#     #                                                          retain_file_path=None, rseed=rseed)
#     train_ds, _, test_ds = load_datasets(sample_ratio=args.sample_ratio,comb_name=args.comb_name,rseed=rseed)
#
#     train_loader = GeoDataLoader(dataset=train_ds, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers,pin_memory=True,persistent_workers=True)
#     test_loader = GeoDataLoader(dataset=test_ds, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers,pin_memory=True,persistent_workers=True)
#
#     # model
#     config = get_cfg_from_args(args)
#     model = PTInsSegModule(config)
#     model,latest_checkpoint_path = load_latest_ckpts(checkpoint_path, model)    # trainer
#     # trainer = pl.Trainer(max_epochs=num_epochs, accelerator='auto', num_sanity_val_steps=0)
#     trainer = pl.Trainer(max_epochs=args.num_epochs,
#                          logger=tb_logger, log_every_n_steps=20,
#                          check_val_every_n_epoch=1,
#                          precision='16-mixed',  # 自动混合精度
#                          callbacks=[checkpoint_callback],
#                          gradient_clip_val=1.0
#                          )
#     start = time.time()
#     trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader, ckpt_path=latest_checkpoint_path)
#     end = time.time()
#     running_time = end - start
#     print(f'Time cost: {running_time:.5f}s, {running_time / 3600} h\n')
#
#
#
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
#
# def do_pred():
#     setup_seed(1111)
#     rseed = 42
#     # 全局参数
#     args = get_args()
#     #logger
#     tb_logger = init_logger(args)
#     #checkpoints
#     checkpoint_callback, checkpoint_path = init_ckpts(args)
#
#     args.sample_ratio = 1.0
#     _, _, predict_ds = load_datasets(sample_ratio=args.sample_ratio,comb_name=args.comb_name,rseed=rseed)
#     predict_loader = GeoDataLoader(dataset=predict_ds, batch_size=1, shuffle=False, num_workers=args.num_workers,
#                                    pin_memory=True)
#
#     config = get_cfg_from_args(args)
#     model = PTInsSegModule(config)
#
#     # 检查是否存在检查点文件
#     model,latest_checkpoint_path = load_latest_ckpts(checkpoint_path, model)
#
#     trainer = pl.Trainer()
#     # 进行预测
#     if 'latest_checkpoint_path' in locals():
#         print(latest_checkpoint_path, locals())
#         predictions = trainer.predict(model, dataloaders=predict_loader, ckpt_path=latest_checkpoint_path)
#     else:
#         predictions = trainer.predict(model, dataloaders=predict_loader)
#
#     save_to_json(args, predictions)
#
#
#
#
#
# def test4_cent():
#     setup_seed(1111)
#     # 全局参数
#     args = get_args()
#     #logger
#     tb_logger = init_logger(args)
#     #checkpoints
#     checkpoint_callback, checkpoint_path = init_ckpts(args)
#
#     _, _, predict_ds = load_DBGA_dataset(sample_ratio=0.01, train_val_ratio=1.0, comb_name='comb_PointnetTrans')
#     predict_loader = GeoDataLoader(dataset=predict_ds, batch_size=1, shuffle=False, num_workers=args.num_workers,
#                                    pin_memory=True)
#
#     config = get_cfg_from_args(args)
#     model = PTInsSegModule(config)
#
#     # 检查是否存在检查点文件
#     model,latest_checkpoint_path = load_latest_ckpts(checkpoint_path, model)
#
#     trainer = pl.Trainer()
#     # 进行预测
#     if 'latest_checkpoint_path' in locals():
#         print(latest_checkpoint_path, locals())
#         predictions = trainer.predict(model, dataloaders=predict_loader, ckpt_path=latest_checkpoint_path)
#     else:
#         predictions = trainer.predict(model, dataloaders=predict_loader)
#
#     for prediction in predictions:
#         pts_normals = prediction['pts_normals']
#         faces = prediction['faces']
#         gt_labels = prediction['gt_labels']
#         pred_labels = prediction['pred_labels']
#         mesh_file = prediction['mesh_file']
#         preds = prediction['predictions']
#         print(mesh_file)
#
#
# def test_PointTransformerNet():
#     PTNet = PointTransformerNet(num_classes=15, num_feat=6)
#     inputs = 0.1 * torch.rand(1, 6, 24000).cuda()  # BCN格式
#     targets = torch.randint(0, 15, (1, 1, 24000)).cuda()  # BCN格式
#     output = PTNet([inputs, targets])
#     print(output)
#
#
# if __name__ == '__main__':
#
#     # test_PointTransformerNet()
#     do_train()
#     # do_pred()