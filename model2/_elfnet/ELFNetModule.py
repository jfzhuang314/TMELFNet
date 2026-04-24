"""
@File    :   skip_mask.py
这一版本主要简化了输入特征的提取，仅仅使用两个edgeconv层提取特征；
"""
from torch import optim
import torch.nn as nn
import pytorch_lightning as pl
import time

from torch_geometric.data import Data

from datasets.T3DSPlusSegDataset import load_datasets_from_cfg_obj
from metrics.ConfAwareBinSegLoss import ConfAwareBinSegLoss as BinSegLoss
from metrics.dice_loss import DiceLoss

from common.interfaces import BaseLoggerManager
from common.BaseTrainer import BaseTrainer
from metrics.ELFNetMetricsManager import ELFNetMetricsManager
from model._elfnet.nets.ELFNet import ELFNet

from utils.points_utils import *
import json

from configs.hydra_yaml_config import *
from utils.seed_utils import SeedUtils




class ELFNetModule(pl.LightningModule):
    """ 双二值分类网络训练模块"""
    def __init__(self, config):

        # 从配置字典中提取参数
        lr = config.get('lr')
        num_epochs = config.get('num_epochs')
        num_batches_to_print = config.get('num_batches_to_print')
        self.num_classes = config.get('num_classes')

        super().__init__()

        # 训练参数
        self.lr = lr
        self.num_epochs = num_epochs
        # self.num_batches_to_print = num_batches_to_print
        self.class_weights = torch.ones(self.num_classes, dtype=torch.float)

        # num_classes = config.get('num_classes')
        # num_classes = 2
        # 模型
        self.model = ELFNet(num_classes=self.num_classes, num_feat=6, k=16)
        self.model = nn.DataParallel(self.model)

        # dice loss
        self.dice_loss = DiceLoss()

        # edge_mask_radius
        self.rmin = config.get('rmin')
        self.rmax = config.get('rmax')

        # metrics
        self.metrics_mgr = ELFNetMetricsManager(task_type="classification",
                                                device=device,
                                                num_classes = self.num_classes,
                                                num_epochs=num_epochs)

    def _forward_pass(self, batch):
        """专门负责模型前向传播，并生成prediction字典"""
        outputs = self.model(batch)
        bg_outputs, tae_outputs, tte_outputs, x_1 = outputs
        return {
            'bg': bg_outputs,     # (bg_pd_1, bg_weight_1, bg_pd_2)
            'tae': tae_outputs,   # (tae_pd_1, tae_weight_1, tae_pd_2)
            'tte': tte_outputs,   # (tte_pd_1, tte_weight_1, tte_pd_2)
            'features': x_1
        }
    def _generate_gt_labels(self, batch):
        """专门负责生成各种Ground Truth标签。
            目前主要包括推理时计算，也可以在生成数据时就计算好。
        """
        labels = batch.y.to(self.device, dtype=torch.long)  #真实类标
         # 1. 背景标签
        bg_gt_bin_labels = (labels!=0).long()  #(BN,1)

        # # 2. tae及tte边缘标签
        if hasattr(batch, 'gt_edge_mask') and hasattr(batch, 'gt_tte_mask'):
            tae_gt_bin_labels = batch.gt_edge_mask.to(self.device, dtype=torch.long)  #(BN,1)
            tte_gt_bin_labels = batch.gt_tte_mask.to(self.device, dtype=torch.long)
        else:
            _, abs_ang = compute_polar_angles(batch.pos)
            radius = self.rmin + abs_ang * (self.rmax - self.rmin)
            tae_gt_bin_labels = compute_tae_gt_bin_labels_ang(batch.pos, labels, r=radius)
            tte_gt_bin_labels = compute_tte_gt_bin_labels(batch.pos, labels, r=radius)

        return {
            'labels': labels,                     # (BN,1)，15类别
            'bg_bin_labels': bg_gt_bin_labels,    # (BN,1)，二类别
            'tae_bin_labels': tae_gt_bin_labels,  # (BN,1)，二类别
            'tte_bin_labels': tte_gt_bin_labels   # (BN,1)，二类别
        }

    def _compute_losses(self, predictions, ground_truth):
        """专门负责损失计算"""
        # 计算背景损失
        bg_gt_bin_labels = ground_truth['bg_bin_labels'].reshape([1, 1, -1])
        bg_pd_1, bg_weight_1, bg_pd_2 = predictions['bg']
        loss_s1 = BinSegLoss.first_seg_loss(bg_pd_1, bg_weight_1, bg_gt_bin_labels)
        loss_s2 = BinSegLoss.second_seg_loss(bg_pd_2, bg_weight_1, bg_gt_bin_labels)

        # 计算tae边缘损失
        tae_gt_bin_labels = ground_truth['tae_bin_labels'].reshape([1, 1, -1])
        tae_pd_1, tae_weight_1, tae_pd_2 = predictions['tae']
        loss_s3 = BinSegLoss.first_seg_loss(tae_pd_1, tae_weight_1, tae_gt_bin_labels)
        loss_s4 = BinSegLoss.second_seg_loss(tae_pd_2, tae_weight_1, tae_gt_bin_labels)

        # 计算tte边缘损失
        tte_gt_bin_labels = ground_truth['tte_bin_labels'].reshape([1, 1, -1])
        tte_pd_1, tte_weight_1, tte_pd_2 = predictions['tte']
        loss_s5 = BinSegLoss.first_seg_loss(tte_pd_1, tte_weight_1, tte_gt_bin_labels)
        loss_s6 = BinSegLoss.second_seg_loss(tte_pd_2, tte_weight_1, tte_gt_bin_labels)

        # regularization loss
        loss_l2 = self.regularization_losses()

        return {
            'loss_s1': loss_s1,
            'loss_s2': loss_s2,
            'loss_s3': loss_s3,
            'loss_s4': loss_s4,
            'loss_s5': loss_s5,
            'loss_s6': loss_s6,
            'loss_l2': loss_l2
        }
    def _aggregate_losses(self, losses):

        loss_s1, loss_s2, loss_s3, loss_s4, loss_s5, loss_s6, loss_l2 = losses.values()
        lambda0, lambda1, lambda2 , lambda3= 2, 1, 1, 1
        loss = (lambda0*(loss_s1 + loss_s2) +
                lambda1*(loss_s3 + loss_s4) +
                lambda2*(loss_s5 + loss_s6) +
                lambda3*loss_l2)

        return loss

    def _compute_metrics(self, predictions, ground_truth):
        """专门负责指标计算"""
        return self.metrics_mgr.calc_groups_metrics(predictions,
                                                    ground_truth,
                                                    self.class_weights)
    def _perform_step_forward(self, batch):
        """简化版：只负责获取预测结果和真实标签"""
        # 1. 模型推理
        predictions = self._forward_pass(batch)
        # 2. 获取真实标签
        ground_truth = self._generate_gt_labels(batch)
        # 3. 计算损失
        loss_components = self._compute_losses(predictions, ground_truth)
        total_loss = self._aggregate_losses(loss_components)
        # 4. 由指标管理器计算指标
        metrics = self._compute_metrics(predictions, ground_truth)

        return {
            'loss': total_loss,
            'predictions': predictions,
            'ground_truth': ground_truth,
            'loss_components': loss_components,
            'metrics': metrics
        }

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        outputs = self._perform_step_forward(batch)

        loss = outputs['loss']
        tg_metrics = outputs['metrics']['tg']
        tae_metrics = outputs['metrics']['tae']
        tte_metrics = outputs['metrics']['tte']

        self.metrics_mgr.update(
            loss.item(), tg_metrics, tae_metrics, tte_metrics, is_train=True
        )

        metrics = self.metrics_mgr.get_metrics_dict(
            loss.item(), tg_metrics, tae_metrics, tte_metrics, is_train=True)

        self.log_dict(metrics)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self._perform_step_forward(batch)

        loss = outputs['loss']
        loss_components = outputs['loss_components']
        loss_s1, loss_s2, loss_s3, loss_s4, loss_s5, loss_s6, loss_l2 = loss_components.values()

        def get_mesh_name(file_path):
            file_name = os.path.basename(file_path)
            folder_name = os.path.basename(os.path.dirname(file_path))
            return f"{file_name}_{folder_name}"

        mesh_name = get_mesh_name(batch.filename[0])

        print(f"epoch:{self.current_epoch}-{mesh_name}, "
              f"loss:{loss:.4f}=Lseg:L1:{loss_s1:.4f} + L2:{loss_s2:.4f}+ L3:{loss_s3:.4f} + L4:{loss_s4:.4f}+"
              f"L5:{loss_s5:.4f} + L6:{loss_s6:.4f}", end=' ')

        tg_metrics = outputs['metrics']['tg']
        tae_metrics = outputs['metrics']['tae']
        tte_metrics = outputs['metrics']['tte']

        self.metrics_mgr.update(loss.item(), tg_metrics, tae_metrics, tte_metrics, is_train=False)

        metrics = self.metrics_mgr.get_metrics_dict(
            loss.item(), tg_metrics, tae_metrics, tte_metrics, is_train=False)

        self.log_dict(metrics)
        return metrics


    def on_train_epoch_end(self):

        print("Training epoch ended.")

        self.metrics_mgr.reset(is_train=True)

        torch.cuda.empty_cache()
        print(f"Epoch end, GPU内存清理完成: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")

    def on_validation_epoch_end(self):

        self.metrics_mgr.compute(is_train=False)
        self.metrics_mgr._print_epoch_info(self.current_epoch)

        self.metrics_mgr.reset(is_train=False)


    def predict_step(self, batch, batch_idx):

        metric_outputs = self._perform_step_forward(batch)

        loss = metric_outputs[0]
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

    def predict_step_forward(self, batch):

        with torch.no_grad():
            outputs = self._forward_pass(batch)
            batch.outputs = outputs

        return  batch



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

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), amsgrad=True, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        return [optimizer], [scheduler]

############################################################################
## Training Teitsng Prediction
############################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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



class ELFNetTrainer(BaseTrainer):
    """ELFNet训练器，继承自BaseTrainer"""
    def __init__(self, config_path="../../configs", config_name="Module_ELFNet"):
        super().__init__(config_path, config_name)

    # 获得模型实例
    def build_module(self, model_obj: DictConfig, ckpt_path=None):
        """获取TripleBinClsModule模型实例"""
        return ELFNetModule(model_obj)

    # 准备数据
    def prepare_data(self, cfg_obj: DictConfig):
        _, self.train_loader, self.test_loader = load_datasets_from_cfg_obj(cfg_obj)

def do_train():

    SeedUtils.setup_seed(1111)

    trainer = ELFNetTrainer(config_path="../../configs", config_name="Module_ELFNet")
    trainer.fit()


# def do_train():
#     # global num_classes, val_batch_size, lr, start, model
#     Utils.setup_seed(1111)
#     rseed = Utils.get_random_seed()
#
#     # 读取全局配置
#     cfg_obj = instantiate_from_yaml_config(config_path="configs/exp", config_name="ELFNet")
#     print_yaml_cfg_dict(cfg_obj)
#
#     # datasets
#     ds_cfg = cfg_obj.dataset
#     dm, train_loader, test_loader = load_datasets_from_cfg(ds_cfg.config_path, ds_cfg.config_name)
#
#     # logger
#     loggerMgr = BaseLoggerManager.from_config(cfg_obj.logger)
#     # model
#     model = TripleBinClsModule(cfg_obj.model)
#     # checkpoints
#     ckpt_mgr = PLCheckpointManager(
#         ckpt_root=cfg_obj.ckpt.ckpt_path,
#         exp_name=cfg_obj.ckpt.exp_name,
#         monitor_metric=cfg_obj.ckpt.monitor_metric,
#         monitor_mode=cfg_obj.ckpt.monitor_mode,
#         save_top_k=cfg_obj.ckpt.save_top_k,  # 保留最优1个检查点
#         save_last=cfg_obj.ckpt.save_last
#     )
#
#     # 加载最新检查点（断点续训）
#     model, loaded_ckpt = ckpt_mgr.load_checkpoint(model)
#     if loaded_ckpt:
#         print(f"✅ 加载最新检查点：{loaded_ckpt}")
#
#     # trainer
#     trainer = pl.Trainer(max_epochs=cfg_obj.training.num_epochs,
#                          logger=loggerMgr.loggers,
#                          log_every_n_steps=20,
#                          check_val_every_n_epoch=1,
#                          precision='16-mixed',
#                          callbacks=[ckpt_mgr.get_ckpt_callback()]
#                          )
#     start = time.time()
#     trainer.fit(model,
#                 train_dataloaders=train_loader,
#                 val_dataloaders=test_loader,
#                 ckpt_path=loaded_ckpt)
#     end = time.time()
#     running_time = end - start
#     print(f'Time cost: {running_time:.5f}s, {running_time / 3600} h\n')


def test_get_model():
    import torch  # Add this import statement

    model = ELFNet(num_classes=15, num_feat=6, k=16)
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


# def do_pred():
#     Utils.setup_seed(1111)
#     # 全局参数
#     args = create_training_args()
#     #logger
#     tb_logger = LoggerMgr.init_logger(args)
#     #checkpoints
#     checkpoint_callback, checkpoint_path = CheckPtMgr.init_ckpts(args.ckpt_path, args.exp_name)
#
#     # args.sample_ratio = 1.0
#     sample_ratio = 1.0
#     _, _, predict_ds = load_DBGA_dataset_with_tbase(sample_ratio=sample_ratio, train_val_ratio=1.0, comb_name='comb_PointnetTrans',
#                                                              is_full=True, retain_file_path=None, rseed=42)
#     predict_loader = GeoDataLoader(dataset=predict_ds, batch_size=1, shuffle=False, num_workers=args.num_workers,
#                                    pin_memory=True)
#
#     config = create_model_config(args)
#     model = TripleBinClsModule(config)
#
#     # 检查是否存在检查点文件
#     model,latest_checkpoint_path = CheckPtMgr.load_latest_ckpts(checkpoint_path, model)
#
#     trainer = pl.Trainer()
#     # 进行预测
#     if 'latest_checkpoint_path' in locals():
#         print(latest_checkpoint_path, locals())
#         predictions = trainer.predict(model, dataloaders=predict_loader, ckpt_path=latest_checkpoint_path)
#     else:
#         predictions = trainer.predict(model, dataloaders=predict_loader)
#
#     # 保存为json
#     PredMgr.save_to_json(args, predictions)
#     # 打印信息
#     PredMgr.print_pred_metrics(predictions)


# def save_topo_cluster_results():
#     """
#     保存拓扑聚类结果
#     """
#     Utils.setup_seed(1111)
#     args = create_training_args()
#     tb_logger = LoggerMgr.init_logger(args)
#     checkpoint_callback, checkpoint_path = CheckPtMgr.init_ckpts(args.ckpt_path, args.exp_name)
#     # args.sample_ratio = 1.0
#     sample_ratio = 1.0
#     train_ds, _, predict_ds = load_DBGA_dataset_with_tbase(sample_ratio=sample_ratio, train_val_ratio=1.0,
#                                                     comb_name='comb_PointnetTrans',
#                                                     is_full=True, retain_file_path=None, rseed=42)
#     predict_loader = GeoDataLoader(dataset=train_ds, batch_size=1, shuffle=False, num_workers=args.num_workers,
#                                    pin_memory=True)
#     config = create_model_config(args)
#     model = TripleBinClsModule(config)
#     # 检查是否存在检查点文件
#     model, latest_checkpoint_path = CheckPtMgr.load_latest_ckpts(checkpoint_path, model)
#     trainer = pl.Trainer()
#     # 进行预测
#     if 'latest_checkpoint_path' in locals():
#         print(latest_checkpoint_path, locals())
#         predictions = trainer.predict(model, dataloaders=predict_loader, ckpt_path=latest_checkpoint_path)
#     else:
#         predictions = trainer.predict(model, dataloaders=predict_loader)
#     for id in range(0, len(predictions)):
#         json_obj = get_json_obj(predictions, id)
#
#         opts = {
#             'alpha': 0.2,
#             'max_iter': 5,
#             'iter_thred_len': 0.01,
#             'field_bound': 0.03,
#             'field_sigma': 0.1,
#             'num_classes': 15,
#             'crop_num': 1280
#         }
#
#         ###############################################################
#         print(f'id:{id},basename:{json_obj.json_name}') # 拓扑聚类耗时: 1.4425580501556396 秒
#         start_time = time.time()
#         outs = topo_cluster(json_obj, opts)
#         end_time = time.time()
#         print(f'cluster num:{outs[0]},拓扑聚类耗时: {end_time - start_time} 秒') # 拓扑聚类耗时: 1.4425580501556396 秒
#
#         ##############################################################
#
#         clu_num, clu_cen, clu_counts, clu_gt_map, counts_mat, gt_clu_num_prop, sep_teeth_mask, clu_labels_all = outs[0:8]
#
#         pred_tmp = {
#             'clu_labels_all': torch.from_numpy(clu_labels_all),
#             'clu_num': torch.tensor(clu_num),
#             'clu_cen': torch.from_numpy(clu_cen),
#             'clu_counts': torch.from_numpy(clu_counts),
#             'clu_gt_map': torch.from_numpy(clu_gt_map),
#             'counts_mat': torch.from_numpy(counts_mat),
#             'gt_clu_num_prop': torch.from_numpy(gt_clu_num_prop),
#             'sep_teeth_mask': torch.from_numpy(sep_teeth_mask),
#             'mesh_file': predictions[id]['mesh_file'],
#             # 'l0_points': predictions[id]['l0_points']
#         }
#         predictions[id] = pred_tmp
#
#     # 要求数据为torch tensor
#     PredMgr.save_to_topo_clu_json(args, predictions)
#     # 打印信息
#     PredMgr.print_pred_metrics(predictions)


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

def test_loggerMgr(logger_cfg_obj):
    loggerMgr = BaseLoggerManager.from_config(logger_cfg_obj)
    print(loggerMgr)
    # 生成 20 条记录并存储到 logger 中
    for i in range(20):
        # 模拟生成记录数据
        record = {
            "train_loss": 0.5 + i * 0.01,  # 模拟训练损失逐渐下降
            "val_acc": 0.85 + i * 0.005,  # 模拟验证准确率逐渐上升
            "epoch": i + 1  # 当前 epoch
        }
        loggerMgr.log_metrics(record, step=i + 1)

def main():
    print_yaml_config(config_path="../../configs", config_name="Module_ELFNet")

    cfg_obj = instantiate_from_yaml_config(config_path="../../configs", config_name="Module_ELFNet")
    print(cfg_obj)
    # 日志管理器初始化
    # test_loggerMgr(cfg_obj.logger)

    # test_get_model()
    do_train()
    # do_pred()
    # test4_cent()
    # save_topo_cluster_results()
    # test_read_json_file()




if __name__ == '__main__':
    main()
