import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from hydra import initialize_config_dir, compose
from omegaconf import DictConfig
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.data import Data, DataLoader
from common.BaseTrainer import BaseTrainer
from configs.hydra_yaml_config import instantiate_from_yaml_config, print_yaml_config, instantiate_from_config, \
    get_yaml_config, print_yaml_cfg_dict

from datasets import T3DSPlusSegDataset
from datasets import TADPMSegDataset
from datasets.T3DSPlusSegDataset import T3DSPlusSegReader
from datasets.transform.PointSamplingIndexer import NNPointSamplingIndexer
from evaluation import eval_segnets
from evaluation.eval_segnets import batch_insseg_pred
from evaluation.seg_evaluation import cal_binary_metric, cal_binary_metric_label

from metrics.dice_loss import DiceLoss
from model._DilatedToothSegNet.DTSegNet import DTSegNet, BADTSegNet
from thop import profile
from metrics.CrossToothSegMetricsManager import CrossToothSegMetricsManager as SegMetricsManager, \
    DynamicRadiusNNEdgeMaskWithPos
from model._DilatedToothSegNet.cbl import CBLLoss
from external_libs import gen_utils as gu
from model._TMELFNet.cbl_point_transformer.cbl_point_transformer_module import get_model
from datasets.transform.TopoMorphCluster import TMCUtils


class CBLPTSegNet2048(torch.nn.Module):
    """
    主要是对CBLPTSegNet的输出做了重新处理
    """

    def __init__(self, num_classes=2, num_feat=7):
        self.input_dim = num_feat
        self.output_dim = num_classes

        super().__init__()

        # 对于2048输入，nstride = [2,2,2,2,2]
        model_parameter = {
            "input_feat": num_feat,
            "stride": [1, 2, 2, 2, 4],
            "nstride": [2, 2, 2, 2],
            "nsample": [36, 24, 24, 24, 24],
            "blocks": [2, 3, 4, 6, 3],
            "block_num": 5,
            "planes": [32, 64, 128, 256, 256],
            "contain_weight": False,
            "crop_sample_size": 2048,
            # "k": self.output_dim,
        }

        self.PT_model = get_model(c=self.input_dim, k=self.output_dim, **model_parameter).to(device)

    def forward(self, inputs):
        """
        inputs
            inputs[0] => B, 6, 24000 : point features
            inputs[1]=> B, 1, 24000 : ground truth segmentation
        """
        # B, C, N = inputs[0].shape
        outputs = {}

        if len(inputs) >= 2:
            output = self.PT_model([inputs[0], inputs[1]])
            outputs.update({
                "cbl_loss": output["cbl_loss"],
                "sem": output["sem"],  # [1,2,24000]
                "offset": output["offset"],
                "mask": output["mask"],
                "first_features": output["first_features"],
                "cls_pred": output["sem"]
            })
        else:
            output = self.PT_model([inputs[0]])
            outputs.update({
                "sem": output["sem"],  # [1,2,24000]
                "offset": output["offset"],
                "mask": output["mask"],
                "first_features": output["first_features"],
                "cls_pred": output["sem"]
            })

        return outputs


class CBLPTSegNet2048_LW(CBLPTSegNet2048):
    """
    主要是对CBLPTSegNet的输出做了重新处理
    """

    def __init__(self, num_classes=2, num_feat=7):
        self.input_dim = num_feat
        self.output_dim = num_classes

        super().__init__()

        # 对于2048输入，nstride = [2,2,2,2,2]
        model_parameter = {
            "input_feat": num_feat,
            "stride": [1, 2, 2, 4, 2],
            "nstride": [2, 2, 2, 2],
            "nsample": [36, 24, 24, 24, 24],
            "blocks": [2, 3, 3, 2, 2],
            "block_num": 5,
            "planes": [32, 64, 128, 128, 128],
            "contain_weight": False,
            "crop_sample_size": 2048,
            # "k": self.output_dim,
        }

        self.PT_model = get_model(c=self.input_dim, k=self.output_dim, **model_parameter).to(device)


class CBLPTSegNet2048_22211(CBLPTSegNet2048):
    """
    主要是对CBLPTSegNet的输出做了重新处理
    """

    def __init__(self, num_classes=2, num_feat=7):
        self.input_dim = num_feat
        self.output_dim = num_classes

        super().__init__()

        # 对于2048输入，nstride = [2,2,2,2,2]
        model_parameter = {
            "input_feat": num_feat,
            "stride": [1, 2, 2, 4, 2],
            "nstride": [2, 2, 2, 2],
            "nsample": [24, 24, 24, 24, 24],
            "blocks": [2, 2, 2, 2, 2],
            "block_num": 5,
            "planes": [32, 64, 64, 128, 128],
            "contain_weight": False,
            "crop_sample_size": 2048,
            # "k": self.output_dim,
        }

        self.PT_model = get_model(c=self.input_dim, k=self.output_dim, **model_parameter).to(device)


class CBLPTSegNet2048_12242(CBLPTSegNet2048):
    """
    主要是对CBLPTSegNet的输出做了重新处理
    """

    def __init__(self, num_classes=2, num_feat=7):
        self.input_dim = num_feat
        self.output_dim = num_classes

        super().__init__()

        # 对于2048输入，nstride = [2,2,2,2,2]
        model_parameter = {
            "input_feat": num_feat,
            "stride": [1, 2, 2, 4, 2],
            "nstride": [2, 2, 2, 2],
            "nsample": [8, 16, 24, 32, 32],
            "blocks": [2, 2, 2, 3, 3],
            "block_num": 5,
            "planes": [32, 64, 64, 160, 128],
            "contain_weight": False,
            "crop_sample_size": 2048,
            # "k": self.output_dim,
        }

        self.PT_model = get_model(c=self.input_dim, k=self.output_dim, **model_parameter).to(device)


class CBLPTSegNet4096_22211(CBLPTSegNet2048):
    """
    主要是对CBLPTSegNet的输出做了重新处理
    """

    def __init__(self, num_classes=2, num_feat=7):
        self.input_dim = num_feat
        self.output_dim = num_classes

        super().__init__()

        # 对于2048输入，nstride = [2,2,2,2,2]
        model_parameter = {
            "input_feat": num_feat,
            "stride": [1, 2, 2, 4, 4],
            "nstride": [2, 2, 2, 2],
            "nsample": [24, 24, 24, 24, 24],
            "blocks": [2, 2, 2, 2, 2],
            "block_num": 5,
            "planes": [32, 64, 64, 128, 128],
            "contain_weight": False,
            "crop_sample_size": 4096,
            # "k": self.output_dim,
        }

        self.PT_model = get_model(c=self.input_dim, k=self.output_dim, **model_parameter).to(device)

class CBLPTSegNet6144_22211(CBLPTSegNet2048):
    """
    主要是对CBLPTSegNet的输出做了重新处理
    """

    def __init__(self, num_classes=2, num_feat=7):
        self.input_dim = num_feat
        self.output_dim = num_classes

        super().__init__()

        # 对于2048输入，nstride = [2,2,2,2,2]
        model_parameter = {
            "input_feat": num_feat,
            "stride": [1, 4, 4, 2, 4],
            "nstride": [2, 2, 2, 2],
            "nsample": [36, 24, 24, 24, 24],
            "blocks": [2, 2, 2, 2, 2],
            "block_num": 5,
            "planes": [32, 64, 64, 128, 128],
            "contain_weight": False,
            "crop_sample_size": 4096,
            # "k": self.output_dim,
        }

        self.PT_model = get_model(c=self.input_dim, k=self.output_dim, **model_parameter).to(device)


class CBLDTSegNet2048(torch.nn.Module):
    def __init__(self, model_cfg: DictConfig):

        super().__init__()
        self.num_classes = 2  # 更改类别数
        model_cfg.num_classes = 2

        self.model = DTSegNet(model_cfg)

        self.criterion_cbl = CBLLoss().to(device)

    def forward(self, inputs):
        """
        inputs
            inputs[0] => B, 6, 24000 : point features
            inputs[1]=> B, 1, 24000 : ground truth segmentation
        """
        # B, C, N = inputs[0].shape
        outputs = {}

        if len(inputs) >= 2:
            outs = self.model(inputs[0])

            gt_labels = inputs[1].view(-1)  # BN,
            outputs.update({
                "sem": outs["seg_result"],  # [1,2,24000]
            })

            if outs.get('cbl_inputs') is not None:
                p1, x1, o1 = outs['cbl_inputs']
                p1 = p1.contiguous()
                x1 = x1.contiguous()
                o1 = o1.contiguous()

                labels_flat = gt_labels.view(-1)
                cbl_loss = self.criterion_cbl([p1, x1, o1], labels_flat)
                outputs.update({
                    "cbl_loss": cbl_loss
                })

        else:
            outs = self.model(inputs[0])

            gt_labels = inputs[1].view(-1)  # BN,
            outputs.update({
                "sem": outs["seg_result"],  # [1,2,24000]
            })

        return outputs


MODEL_REGISTRY = {
    "CBLPTSegNet2048": CBLPTSegNet2048,  # CBL + Point Transformer for 2048 crop
    "CBLPTSegNet2048_LW": CBLPTSegNet2048_LW,
    "CBLPTSegNet2048_22211": CBLPTSegNet2048_22211,
    "CBLPTSegNet2048_12242": CBLPTSegNet2048_12242,
    "CBLPTSegNet4096": CBLPTSegNet4096_22211,
    "CBLDTSegNet2048": CBLDTSegNet2048,
    "CBLPTSegNet6144": CBLPTSegNet6144_22211,
}


class CropEval:
    """Crop 级别的分割评价指标管理器"""

    @staticmethod
    def compute_crop_metrics(crop_gt_binseg_label, crop_pred_seg_labels, cluster_gt_seg_labels):
        """
        更新指标

        Args:
            crops: dict, 包含以下键:
                - 'crop_gt_binseg_label': (clu, 1, crop) 或 (clu, crop)
                - 'crop_pred_seg_labels': (clu, 2, crop)
                - 'cluster_gt_seg_labels': (clu,)
        """
        gt_labels = crop_gt_binseg_label  # (clu, 1, crop) 或 (clu, crop)
        pred_logits = crop_pred_seg_labels  # (clu, 2, crop)
        cluster_gt_seg_labels = cluster_gt_seg_labels  # (clu,)

        # 确保形状正确
        if gt_labels.dim() == 3 and gt_labels.shape[1] == 1:
            gt_labels = gt_labels.squeeze(1)  # (clu, crop)

        # 获取预测类别
        pred_classes = torch.argmax(pred_logits, dim=1)  # (clu, crop)

        mACC, mprecision, mrecall, mF1, mIOU = 0.0, 0.0, 0.0, 0.0, 0.0
        all_metrics = []

        # 计算每个 crop 的指标
        for i, clu in enumerate(cluster_gt_seg_labels):
            gt_flat = gt_labels[i].reshape(-1)  # (crop,)
            pred_flat = pred_classes[i].reshape(-1)  # (crop,)

            ACC, precision, recall, F1, IOU, _ = cal_binary_metric_label(gt_labels=gt_flat, pred_classes=pred_flat,
                                                                         target_class=1)

            mACC += ACC
            mprecision += precision
            mrecall += recall
            mF1 += F1
            mIOU += IOU

            clu_metrics = {
                "label_id": str(int(clu)),  # 第clu个牙齿的标签
                "ACC": ACC,
                "precision": precision,
                "recall": recall,
                "F1": F1,
                "IOU": IOU,
            }

            all_metrics.append(clu_metrics)

        metrics = {
            "label_id": "all",
            "ACC": mACC / len(cluster_gt_seg_labels),
            "precision": mprecision / len(cluster_gt_seg_labels),
            "recall": mrecall / len(cluster_gt_seg_labels),
            "F1": mF1 / len(cluster_gt_seg_labels),
            "IOU": mIOU / len(cluster_gt_seg_labels),
        }
        all_metrics.append(metrics)

        return all_metrics

    def print_crop_metrics(all_metrics, prefix="Crop Evaluation Metrics"):
        """
        打印 all_metrics 中的所有指标

        Args:
            all_metrics: list, 包含每个 crop 的指标字典和平均指标
                [
                    {"label_id": "0", "ACC": 0.95, "precision": 0.92, ...},
                    {"label_id": "1", "ACC": 0.93, "precision": 0.90, ...},
                    ...
                    {"label_id": "all", "ACC": 0.94, "precision": 0.91, ...}
                ]
            prefix: 标题前缀
        """
        if not all_metrics or len(all_metrics) == 0:
            print("⚠️  没有可用的指标数据")
            return

        print(f"\n{'=' * 90}")
        print(f"📊 {prefix}")
        print(f"{'=' * 90}")

        # 分离单个 crop 指标和平均指标
        single_crops = [m for m in all_metrics if m.get('label_id') != 'all']
        avg_metrics = next((m for m in all_metrics if m.get('label_id') == 'all'), None)

        # ========== 1. 每个 Crop 的详细指标 ==========
        if len(single_crops) > 0:
            print(f"\n【每个 Crop 的详细指标】 (共{len(single_crops)}个)")
            print(f"{'Label ID':<12} {'ACC':<10} {'Precision':<12} {'Recall':<10} {'F1':<10} {'IoU':<10}")
            print(f"{'-' * 90}")

            for m in single_crops:
                label_id = m.get('label_id', 'N/A')
                acc = m.get('ACC', 0)
                precision = m.get('precision', 0)
                recall = m.get('recall', 0)
                f1 = m.get('F1', 0)
                iou = m.get('IOU', 0)

                print(f"{label_id:<12} "
                      f"{acc:<10.4f} "
                      f"{precision:<12.4f} "
                      f"{recall:<10.4f} "
                      f"{f1:<10.4f} "
                      f"{iou:<10.4f}")

        # ========== 2. 平均指标 ==========
        if avg_metrics:
            print(f"\n【平均指标】")
            print(f"{'-' * 90}")
            print(f"{'Label ID':<12} {'ACC':<10} {'Precision':<12} {'Recall':<10} {'F1':<10} {'IoU':<10}")
            print(f"{'-' * 90}")

            acc = avg_metrics.get('ACC', 0)
            precision = avg_metrics.get('precision', 0)
            recall = avg_metrics.get('recall', 0)
            f1 = avg_metrics.get('F1', 0)
            iou = avg_metrics.get('IOU', 0)

            print(f"{avg_metrics.get('label_id', 'all'):<12} "
                  f"{acc:<10.4f} "
                  f"{precision:<12.4f} "
                  f"{recall:<10.4f} "
                  f"{f1:<10.4f} "
                  f"{iou:<10.4f}")

        # ========== 3. 统计信息 ==========
        if len(single_crops) > 1:
            import numpy as np

            acc_values = [m['ACC'] for m in single_crops]
            prec_values = [m['precision'] for m in single_crops]
            rec_values = [m['recall'] for m in single_crops]
            f1_values = [m['F1'] for m in single_crops]
            iou_values = [m['IOU'] for m in single_crops]

            print(f"\n【统计信息】")
            print(f"{'-' * 90}")
            print(f"  Crop 数量：{len(single_crops)}")
            print(f"  ACC:  {np.mean(acc_values):.4f} ± {np.std(acc_values):.4f} "
                  f"[{np.min(acc_values):.4f}, {np.max(acc_values):.4f}]")
            print(f"  Prec: {np.mean(prec_values):.4f} ± {np.std(prec_values):.4f} "
                  f"[{np.min(prec_values):.4f}, {np.max(prec_values):.4f}]")
            print(f"  Rec:  {np.mean(rec_values):.4f} ± {np.std(rec_values):.4f} "
                  f"[{np.min(rec_values):.4f}, {np.max(rec_values):.4f}]")
            print(f"  F1:   {np.mean(f1_values):.4f} ± {np.std(f1_values):.4f} "
                  f"[{np.min(f1_values):.4f}, {np.max(f1_values):.4f}]")
            print(f"  IoU:  {np.mean(iou_values):.4f} ± {np.std(iou_values):.4f} "
                  f"[{np.min(iou_values):.4f}, {np.max(iou_values):.4f}]")

        print(f"{'=' * 90}\n")


# def batch_pred(ckpt_path, module, test_loader, save_json=True):
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#     base_save_path = Path(ckpt_path).parent
#     last_folder_name = Path(base_save_path).name
#     # # excel 保存目录
#     # excel_save_path = base_save_path / f"{last_folder_name}_eval_{timestamp}.xlsx"
#     # # JSON 保存目录
#     # json_save_dir = base_save_path / f"batch_json_{timestamp}"
#     # print(f"\n📂 JSON 保存目录：{json_save_dir}")
#     # 存储所有样本的结果
#     # all_results = []
#     # 遍历测试集
#     for batch_idx, batch in enumerate(test_loader):
#         print(f"评估样本 {batch_idx + 1}/{len(test_loader)}: {batch.filename}")
#
#         with torch.no_grad():
#             pred_data = module.predict_step_forward(batch)
#         crops = pred_data.outputs['crops']
#
#         # 计算指标
#         all_metrics = CropEval.compute_crop_metrics(
#             crops['crop_gt_binseg_label'],
#             crops['crop_pred_seg_labels'],
#             crops['cluster_gt_seg_labels']
#         )
#
#         # 打印指标
#         CropEval.print_crop_metrics(all_metrics, prefix="pred")


def save_crop_evaluation_results_to_excel(
        all_results,
        save_path=None
):
    """
    将 Crop 评估结果保存到 Excel 文件（支持多条记录）

    Args:
        all_results: 列表，每个元素是一个字典，包含：
            {
                'filename': '样本文件名',
                'ACC': 准确率,
                'precision': 精确率,
                'recall': 召回率,
                'F1': F1 分数,
                'IOU': 交并比,
                'crop_count': crop 数量
            }
        save_path: Excel 保存路径
    """
    if save_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = f"tmelf_crop_eval_{timestamp}.xlsx"

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

    # ========== Sheet 1: 所有样本的详细指标 ==========
    all_records = []

    for result in all_results:
        filename = result.get('filename', 'Unknown')
        avg_metrics = result.get('avg_metrics', {})
        per_crop = result.get('per_crop', [])

        # 提取核心指标
        record = {
            'Filename': filename,
            'Crop Count': len(per_crop),
            'ACC (Mean)': f"{avg_metrics.get('ACC', 0):.4f}",
            'Precision (Mean)': f"{avg_metrics.get('precision', 0):.4f}",
            'Recall (Mean)': f"{avg_metrics.get('recall', 0):.4f}",
            'F1 Score (Mean)': f"{avg_metrics.get('F1', 0):.4f}",
            'IoU (Mean)': f"{avg_metrics.get('IOU', 0):.4f}",
        }

        # 添加统计信息
        if len(per_crop) > 1:
            acc_values = [m['ACC'] for m in per_crop]
            prec_values = [m['precision'] for m in per_crop]
            rec_values = [m['recall'] for m in per_crop]
            f1_values = [m['F1'] for m in per_crop]
            iou_values = [m['IOU'] for m in per_crop]

            record['ACC (Std)'] = f"{np.std(acc_values):.4f}"
            record['Precision (Std)'] = f"{np.std(prec_values):.4f}"
            record['Recall (Std)'] = f"{np.std(rec_values):.4f}"
            record['F1 (Std)'] = f"{np.std(f1_values):.4f}"
            record['IoU (Std)'] = f"{np.std(iou_values):.4f}"
            record['ACC (Min)'] = f"{np.min(acc_values):.4f}"
            record['ACC (Max)'] = f"{np.max(acc_values):.4f}"

        all_records.append(record)

    df_details = pd.DataFrame(all_records)

    # ========== Sheet 2: 统计摘要 ==========
    numeric_cols = ['ACC (Mean)', 'Precision (Mean)', 'Recall (Mean)',
                    'F1 Score (Mean)', 'IoU (Mean)']

    summary_data = {
        'Metric': [],
        'Mean': [],
        'Std': [],
        'Max': [],
        'Min': []
    }

    for col in numeric_cols:
        if col in df_details.columns:
            values = pd.to_numeric(df_details[col], errors='coerce')
            summary_data['Metric'].append(col)
            summary_data['Mean'].append(f"{values.mean():.4f}")
            summary_data['Std'].append(f"{values.std():.4f}")
            summary_data['Max'].append(f"{values.max():.4f}")
            summary_data['Min'].append(f"{values.min():.4f}")

    summary_data['Metric'].append('Total Samples')
    summary_data['Mean'].append(str(len(all_results)))
    summary_data['Std'].append('-')
    summary_data['Max'].append('-')
    summary_data['Min'].append('-')

    summary_data['Metric'].append('Evaluation Time')
    summary_data['Mean'].append(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    summary_data['Std'].append('-')
    summary_data['Max'].append('-')
    summary_data['Min'].append('-')

    df_summary = pd.DataFrame(summary_data)

    # ========== Sheet 3: 每个样本的 per-crop 详情 ==========
    per_crop_records = []
    for result in all_results:
        filename = result.get('filename', 'Unknown')
        per_crop = result.get('per_crop', [])

        for crop_metric in per_crop:
            per_crop_records.append({
                'Filename': filename,
                'Label ID': crop_metric.get('label_id', 'N/A'),
                'ACC': f"{crop_metric.get('ACC', 0):.4f}",
                'Precision': f"{crop_metric.get('precision', 0):.4f}",
                'Recall': f"{crop_metric.get('recall', 0):.4f}",
                'F1': f"{crop_metric.get('F1', 0):.4f}",
                'IoU': f"{crop_metric.get('IOU', 0):.4f}",
            })

    df_per_crop = pd.DataFrame(per_crop_records)

    # ========== 保存到 Excel ==========
    try:
        with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
            df_details.to_excel(writer, sheet_name='All Samples', index=False)
            df_summary.to_excel(writer, sheet_name='Statistics', index=False)
            df_per_crop.to_excel(writer, sheet_name='Per-Crop Details', index=False)

        print("\n" + "=" * 70)
        print("📊 Crop 评估结果已保存到 Excel")
        print("=" * 70)
        print(f"保存路径：{os.path.abspath(save_path)}")
        print(f"\n包含以下工作表:")
        print(f"  1. All Samples - 所有样本的详细指标 ({len(all_results)} 条记录)")
        print(f"  2. Statistics - 统计摘要 (均值/标准差/最大/最小)")
        print(f"  3. Per-Crop Details - 每个 crop 的详细指标 ({len(df_per_crop)} 条)")
        print("=" * 70)

        return save_path

    except Exception as e:
        print(f"❌ 保存失败：{e}")
        raise

def save_crop_batch_info_to_json(batch, crops,json_save_dir):
    """
    保存 crop 预测的 batch 信息到 JSON 文件（参照 save_batch_info_to_json）

    Args:
        batch: 输入的 batch 数据
        crops: dict, 包含所有 crops 相关信息
        single_crops: list, 每个 crop 的评估指标
        avg_metrics: dict, 平均评估指标
        json_save_dir: JSON 保存目录

    Returns:
        str: JSON 文件路径
    """
    try:
        # 获取文件名
        filename = batch.filename[0] if isinstance(batch.filename, list) else batch.filename

        # 将 crops 中的所有 tensor 转换为可序列化的列表
        crops_serializable = {}
        for key, value in crops.items():
            if isinstance(value, torch.Tensor):
                crops_serializable[key] = value.cpu().numpy().tolist()
            elif isinstance(value, dict):
                # 如果值是嵌套字典，递归处理
                crops_serializable[key] = {
                    k: v.cpu().numpy().tolist() if isinstance(v, torch.Tensor) else v
                    for k, v in value.items()
                }
            else:
                crops_serializable[key] = value

        # 构建完整的 batch 数据
        batch_data = {
            'filename': filename,

            # Ground Truth 数据
            'y': batch.y.cpu().numpy().tolist() if hasattr(batch, 'y') and batch.y is not None else None,
            'pos': batch.pos.cpu().numpy().tolist() if hasattr(batch, 'pos') and batch.pos is not None else None,
            'face': batch.face.cpu().numpy().tolist() if hasattr(batch, 'face') and batch.face is not None else None,
            'y_ori': batch.y_ori.cpu().numpy().tolist() if hasattr(batch, 'y_ori') and batch.y_ori is not None else None,
            'c2f_idx': batch.c2f_idx.cpu().numpy().tolist() if hasattr(batch, 'c2f_idx') and batch.c2f_idx is not None else None,
            'f2c_idx': batch.f2c_idx.cpu().numpy().tolist() if hasattr(batch, 'f2c_idx') and batch.f2c_idx is not None else None,
            'gt_edge_mask': batch.gt_edge_mask.cpu().numpy().tolist() if hasattr(batch, 'gt_edge_mask') and batch.gt_edge_mask is not None else None,
            'gt_tte_mask': batch.gt_tte_mask.cpu().numpy().tolist() if hasattr(batch, 'gt_tte_mask') and batch.gt_tte_mask is not None else None,
            'instance_centroids': batch.instance_centroids.cpu().numpy().tolist() if hasattr(batch, 'instance_centroids') and batch.instance_centroids is not None else None,
            'instance_labels': batch.instance_labels.cpu().numpy().tolist() if hasattr(batch, 'instance_labels') and batch.instance_labels is not None else None,

            # Crops 数据（全部保留）
            'crops': crops_serializable,

            # seg
            'seg_result': batch.seg_result.cpu().numpy().tolist() if hasattr(batch, 'seg_result') and batch.seg_result is not None else None,
        }

        # 确保保存目录存在
        os.makedirs(json_save_dir, exist_ok=True)

        # 保存为 JSON 文件
        json_path = os.path.join(json_save_dir, f"{filename}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(batch_data, f, ensure_ascii=False, indent=2)

        return json_path

    except Exception as e:
        print(f"❌ JSON 保存失败：{e}")
        import traceback
        traceback.print_exc()
        raise


def batch_crop_pred(ckpt_path, module, test_loader, save_json=True, save_excel=True):
    """
    批量预测并评估 TMELFNet 模型性能（Crop 级别二分类指标）

    Args:
        ckpt_path: 检查点路径
        module: 模型模块
        test_loader: 测试数据加载器
        save_json: 是否保存 JSON（暂未实现）
        save_excel: 是否保存 Excel
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_save_path = Path(ckpt_path).parent
    last_folder_name = Path(base_save_path).name

    # Excel 保存目录
    excel_save_path = base_save_path / f"tmelf_crop_eval_{timestamp}.xlsx"

    # JSON 保存目录
    json_save_dir = base_save_path / f"batch_json_{timestamp}"
    print(f"\n📂 Excel 保存路径：{excel_save_path}")
    if save_json:
        print(f"📂 JSON 保存目录：{json_save_dir}")

    # 存储所有样本的结果
    all_results = []

    # 遍历测试集
    for batch_idx, batch in enumerate(test_loader):
        print(f"评估样本 {batch_idx + 1}/{len(test_loader)}: {batch.filename}")

        with torch.no_grad():
            pred_data = module.predict_step_forward(batch)

        # 获取 crops 数据
        crops = pred_data.outputs['crops']

        # 计算 crop 级别指标
        all_metrics = CropEval.compute_crop_metrics(
            crops['crop_gt_binseg_label'],
            crops['crop_pred_seg_labels'],
            crops['cluster_gt_seg_labels']
        )

        # 分离单个 crop 指标和平均指标
        single_crops = [m for m in all_metrics if m.get('label_id') != 'all']
        avg_metrics = next((m for m in all_metrics if m.get('label_id') == 'all'), None)

        # 打印当前样本结果
        print(f"\n{'=' * 70}")
        print(f"【{batch.filename}】 Crop 级别指标 (共{len(single_crops)}个 crops)")
        print(f"{'-' * 70}")
        print(f"{'Label ID':<12} {'ACC':<10} {'Prec':<10} {'Rec':<10} {'F1':<10} {'IoU':<10}")
        print(f"{'-' * 70}")

        # 显示前 10 个 crop
        display_count = min(16, len(single_crops))
        for i in range(display_count):
            m = single_crops[i]
            print(f"{m.get('label_id', 'N/A'):<12} "
                  f"{m.get('ACC', 0):<10.4f} "
                  f"{m.get('precision', 0):<10.4f} "
                  f"{m.get('recall', 0):<10.4f} "
                  f"{m.get('F1', 0):<10.4f} "
                  f"{m.get('IOU', 0):<10.4f}")

        if len(single_crops) > display_count:
            print(f"... 还有 {len(single_crops) - display_count} 个 crops")

        # 打印平均指标
        if avg_metrics:
            print(f"\n【平均指标】")
            print(f"{'-' * 70}")
            print(f"{'Label ID':<12} {'ACC':<10} {'Prec':<10} {'Rec':<10} {'F1':<10} {'IoU':<10}")
            print(f"{'-' * 70}")
            print(f"{avg_metrics.get('label_id', 'all'):<12} "
                  f"{avg_metrics.get('ACC', 0):<10.4f} "
                  f"{avg_metrics.get('precision', 0):<10.4f} "
                  f"{avg_metrics.get('recall', 0):<10.4f} "
                  f"{avg_metrics.get('F1', 0):<10.4f} "
                  f"{avg_metrics.get('IOU', 0):<10.4f}")

        # 打印统计信息
        if len(single_crops) > 1:
            acc_values = [m['ACC'] for m in single_crops]
            prec_values = [m['precision'] for m in single_crops]
            rec_values = [m['recall'] for m in single_crops]
            f1_values = [m['F1'] for m in single_crops]
            iou_values = [m['IOU'] for m in single_crops]

            print(f"\n【统计信息】")
            print(f"  ACC:  {np.mean(acc_values):.4f} ± {np.std(acc_values):.4f} "
                  f"[{np.min(acc_values):.4f}, {np.max(acc_values):.4f}]")
            print(f"  Prec: {np.mean(prec_values):.4f} ± {np.std(prec_values):.4f}")
            print(f"  Rec:  {np.mean(rec_values):.4f} ± {np.std(rec_values):.4f}")
            print(f"  F1:   {np.mean(f1_values):.4f} ± {np.std(f1_values):.4f}")
            print(f"  IoU:  {np.mean(iou_values):.4f} ± {np.std(iou_values):.4f}")

        print(f"{'=' * 70}\n")

        # 添加到结果列表
        result = {
            'filename': batch.filename[0] if isinstance(batch.filename, list) else batch.filename,
            'per_crop': single_crops,
            'avg_metrics': {
                'ACC': avg_metrics.get('ACC', 0) if avg_metrics else 0,
                'precision': avg_metrics.get('precision', 0) if avg_metrics else 0,
                'recall': avg_metrics.get('recall', 0) if avg_metrics else 0,
                'F1': avg_metrics.get('F1', 0) if avg_metrics else 0,
                'IOU': avg_metrics.get('IOU', 0) if avg_metrics else 0,
            },
            'crop_count': len(single_crops)
        }
        all_results.append(result)

        if save_json:
            try:
                json_path = save_crop_batch_info_to_json(
                    batch=batch,
                    crops=crops,
                    json_save_dir=json_save_dir
                )
                print(f"✅ JSON 已保存：{os.path.basename(json_path)}")
            except Exception as e:
                print(f"❌ JSON 保存过程中出错：{e}")


        if batch_idx % 60 == 0:
            torch.cuda.empty_cache()
            print(f"Validation batch start, GPU内存清理完成: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")


    # 保存所有结果到 Excel
    if save_excel and len(all_results) > 0:
        save_crop_evaluation_results_to_excel(
            all_results=all_results,
            save_path=excel_save_path
        )

    return all_results


class PatchSegModule(pl.LightningModule):
    """ 牙齿分割网络训练模块 - 采用TMELFNet的逻辑结构重构"""

    def __init__(self, model_cfg: DictConfig, num_classes=15):
        super().__init__()

        self.config = model_cfg
        self.num_classes = num_classes
        self.input_dim = 6
        self.num_epochs = model_cfg.get("num_epochs", 30)
        self.lr = model_cfg.get("lr", 0.001)

        # CBL相关参数
        model_cfg.add_cbl = True if "_CBL" in model_cfg.loss_type else False

        self.add_cbl = model_cfg.add_cbl
        self.k = model_cfg.get("k", 16)  # 用于边界mask计算
        self.lam = model_cfg.get("lam", 0.1)  # CBL损失权重
        self.compute_edge_binary_iou = model_cfg.get("compute_edge_binary_iou", False)

        self.class_weights = torch.ones(self.num_classes, dtype=torch.float)

        # 模型
        self.net_params = model_cfg.net_params
        self.model = self.build_model(model_cfg)

        # 损失函数
        self.xe_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(ignore_index=0)
        self.criterion_cbl = CBLLoss().to(device)
        self.loss_type = model_cfg.loss_type

        # 指标管理器
        self.metrics_mgr = SegMetricsManager(
            task_type="segmentation",
            device=device,
            num_classes=self.num_classes,
            num_epochs=self.num_epochs,
            compute_edge_binary_iou=self.compute_edge_binary_iou,
        )

    def build_model(self, model_cfg):

        # 提取网络参数
        net_params = model_cfg.net_params
        # kwargs = {
        #     **{k: net_params[k] for k in net_params.keys()}
        # }

        model_class = MODEL_REGISTRY.get(model_cfg.net_name)
        if model_class is None:
            raise ValueError(f"Invalid model type: {model_cfg.net_name}. "
                             f"Available models: {list(MODEL_REGISTRY.keys())}")

        if model_cfg.net_name == "CBLDTSegNet2048":
            # net_params.num_classes = 2
            self.model = model_class(net_params)
        else:
            self.model = model_class()
        self.model = nn.DataParallel(self.model).to(device)

        return self.model

    def _forward_pass(self, batch, is_train=False):
        """专门负责模型前向传播"""
        # outputs = self.model(inputs)
        # seg_result = batch.tmc_proposal['outputs']['seg_result'].to(device)  # 之前模型输出的seg_result，BNC
        # tmc_pred_seg_labels = torch.argmax(seg_result, dim=-1).view(-1).to(device)  # [16000]
        # tmc_pos = batch.tmc_proposal['outputs']['cbl_inputs'][0].to(device)  # 之前模型输出的pos，[BN,3]

        pos = batch.pos.to(device)

        # # 从tmc_pos中类别转移到pos中，要求这二者的数据要一致
        # # 之所以这么做，是因为patchnet不一定是16000的，也可以是24000或者32000，因而要做下类标转换
        # _, q2f_idx = NNPointSamplingIndexer()(pos.cpu().numpy(), tmc_pos.cpu().numpy())
        # q2f_idx = torch.tensor(q2f_idx).to(device)
        # pred_seg_labels = tmc_pred_seg_labels[q2f_idx].to(device)

        if hasattr(batch, 'num_graphs') and batch.num_graphs > 1:
            data = batch[0] if hasattr(batch, '__getitem__') else batch
        else:
            data = batch

        outs = TMCUtils.build_crop_batch(data, is_train=is_train, crop_num_16000=2048)
        crop_nn_indexes = outs['crop_nn_indexes'].to(device)  # (1, clu, crop)
        cropped_feats = outs['crop_feats'].to(device)
        cropped_pos = outs['crop_pos'].to(device)
        crop_gt_binseg_label = outs['crop_gt_binseg_label'].to(device)
        crop_gt_seg_label = outs['crop_gt_seg_label'].to(device)
        cluster_gt_seg_labels = outs['cluster_gt_seg_labels'].to(device)

        inputs = [cropped_feats.float().to(device), crop_gt_binseg_label.long().to(device)]
        bin_outs = self.model(inputs)

        # vote_seg_result = TMCUtils.vote_seg_label(bin_outs['sem'], pred_seg_labels, crop_nn_indexes,
        #                                           num_classes=self.num_classes)
        # vote_seg_result = F.one_hot(vote_seg_result, num_classes=self.num_classes).unsqueeze(0)

        crops = {
            'cropped_nn_indexes': crop_nn_indexes,  # (1, clu, crop)
            'cropped_pos': cropped_pos,  # (clu, 3, crop)
            'crop_gt_seg_label': crop_gt_seg_label,  # (clu, 1,crop)
            'crop_gt_binseg_label': crop_gt_binseg_label,  # (clu, 1,crop)
            'crop_pred_seg_labels': bin_outs['sem'],  # (clu, 2,crop)
            'cluster_gt_seg_labels': cluster_gt_seg_labels,  # (clu,)
        }

        # losses
        pred = bin_outs['sem'].permute(0, 2, 1).reshape(-1, 2).to(device)
        gt = crop_gt_binseg_label.squeeze().view(-1).to(device)

        seg_loss = self.xe_loss(pred, gt).to(device)
        dice_loss = self.dice_loss(pred, gt).to(device)

        if 'cbl_loss' in bin_outs and bin_outs['cbl_loss'] is not None:
            cbl_loss = torch.sum(bin_outs['cbl_loss']).to(device)
        else:
            cbl_loss = torch.tensor(0).to(device)

        vote_seg_result = torch.zeros([1, pos.shape[0], self.num_classes])
        return {
            'seg_result': vote_seg_result.to(device),  # BNC
            'seg_loss': seg_loss.to(device),  # B,
            'cbl_loss': cbl_loss.to(device),  # B,
            'dice_loss': dice_loss.to(device),  # B,
            'crops': crops,  # Crop字典
        }

    def _generate_gt_labels(self, batch):
        """专门负责生成Ground Truth标签"""
        # batch_size = batch.batch.max().item() + 1 if hasattr(batch, 'batch') else 1
        batch_size = 1
        num_points = batch.pos.shape[0] // batch_size
        gt_labels = batch.y.to(device, dtype=torch.long)  # BN,

        one_hot_labels = (torch.nn.functional.one_hot(gt_labels, num_classes=self.num_classes).
                          reshape(batch_size, num_points, -1).to(device, dtype=torch.long))  # [B,N, 15]

        ground_truth = {
            'gt_labels': gt_labels.to(device),  # BN,
            'onehot_gt_labels': one_hot_labels.to(device)  # BNC
        }

        if hasattr(batch, 'gt_edge_mask'):
            gt_edge_mask = batch.gt_edge_mask.to(device, dtype=torch.float32)
            ground_truth['gt_edge_mask'] = gt_edge_mask.long()  # BN,

        if hasattr(batch, 'gt_tte_mask'):
            gt_tte_mask = batch.gt_tte_mask.to(device, dtype=torch.float32)
            ground_truth['gt_tte_mask'] = gt_tte_mask.long()  # BN,

        if 'gt_edge_mask' not in ground_truth or 'gt_tte_mask' not in ground_truth:
            gt_edge_mask, _ = DynamicRadiusNNEdgeMaskWithPos().generate_edge_masks(batch.pos, gt_labels)
            ground_truth['gt_edge_mask'] = gt_edge_mask.to(device, dtype=torch.float32).long()  # BN,
            ground_truth['gt_tte_mask'] = gt_tte_mask.to(device, dtype=torch.float32).long()  # BN,

        return ground_truth

    def _compute_losses(self, outputs, ground_truth):
        """专门负责损失计算"""
        # 语义分割损失
        seg_outputs = outputs['seg_result'].reshape([-1, self.num_classes])  # BNC -> (B*N, C)
        gt_labels = ground_truth['gt_labels'].view(-1)  # BN -> (B*N,)
        gt_edge_mask = ground_truth.get('gt_edge_mask', None)
        gt_tte_mask = ground_truth.get('gt_tte_mask', None)

        # seg_loss = self.xe_loss(seg_outputs, gt_labels)
        seg_loss = outputs['seg_loss'].to(device)
        dice_loss = outputs['dice_loss'].to(device)

        # # 边界分割损失
        # edge_loss = torch.tensor(0.0).to(device)
        # if 'edge_seg_result' in outputs and outputs['edge_seg_result'] is not None:
        #     edge_seg_result = outputs['edge_seg_result']  # BCN
        #     B, _, N = edge_seg_result.shape
        #
        #     # 计算边界分割loss
        #     edge_logits = edge_seg_result.permute(0, 2, 1).contiguous().view(B * N, 2)  # (B*N, 2)
        #     edge_labels = gt_edge_mask.view(B * N)  # (B*N,)
        #     # edge_loss = self.xe_loss(edge_logits, edge_labels)
        #     edge_loss = self.dice_loss(edge_logits, edge_labels)
        #
        # tte_loss = torch.tensor(0.0).to(device)
        # if 'tte_seg_result' in outputs and outputs['tte_seg_result'] is not None:
        #     tte_seg_result = outputs['tte_seg_result']  # BCN
        #     B, _, N = tte_seg_result.shape
        #
        #     # 计算TTE分割loss
        #     tte_logits = tte_seg_result.permute(0, 2, 1).contiguous().view(B * N, -1)  # (B*N, 15)
        #     tte_labels = gt_tte_mask.view(B * N)  # (B*N,)
        #     # tte_loss = self.xe_loss(tte_logits, tte_labels)
        #     tte_loss = self.dice_loss(tte_logits, tte_labels)

        # CBL损失（如果存在）
        cbl_loss = torch.tensor(0.0).to(device)
        if outputs.get('cbl_loss') is not None and self.add_cbl:
            cbl_loss = outputs['cbl_loss'].to(device)

        # L2正则化损失
        l2_loss = self.regularization_losses().to(device)

        return {
            'seg_loss': seg_loss,
            'dice_loss': dice_loss,
            # 'edge_loss': edge_loss,
            # 'tte_loss': tte_loss,
            'cbl_loss': cbl_loss,
            'l2_loss': l2_loss
        }

    def _aggregate_losses(self, losses):
        """聚合各项损失"""
        seg_loss = losses['seg_loss'].to(device)
        dice_loss = losses['dice_loss'].to(device)
        # edge_loss = losses['edge_loss'].to(device)
        # tte_loss = losses['tte_loss'].to(device)
        cbl_loss = losses['cbl_loss'].to(device)
        l2_loss = losses['l2_loss'].to(device)

        lam_list = self.get_lam_list()

        loss_list = [seg_loss, dice_loss, cbl_loss, l2_loss]

        total_loss = 0.0
        for lam, loss in zip(lam_list, loss_list):
            total_loss += lam * loss

        return total_loss

    def get_lam_list(self):
        lam_edge = 0
        lam_dice = 0.0
        lam_cbl = 0.3
        if self.loss_type == 'SEG_CBL':
            lam_list = [1.0, lam_dice, lam_cbl, 0.001]
        elif self.loss_type == 'SEG':
            lam_list = [1.0, lam_dice, 0.0, 0.001]
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}. "
                             f"Available loss types: TAE_TTE, SEG_TAE_TTE, SEG")
        return lam_list

    def _compute_metrics(self, predictions, ground_truth, **kwargs):
        """专门负责指标计算"""

        return self.metrics_mgr.compute_step_metrics(
            predictions,
            ground_truth,
            **kwargs
        )

    def _perform_step_forward(self, batch):
        """执行完整的前向传播流程"""
        # 1. 数据预处理
        # B = batch.batch.max().item() + 1 if hasattr(batch, 'batch') else 1
        B = 1
        # N = batch.pos.shape[0]
        # x0 = batch.x.to(self.device, dtype=torch.float32)  # NC
        # gt_labels = batch.y.to(self.device, dtype=torch.long)  # N,
        pos = batch.pos.to(self.device, dtype=torch.float32)  # N, 3

        # 2. 模型前向传播，主要训练时使用的是训练数据集的中心点，而测试时使用的是tmc聚类结果的中心点
        outputs = self._forward_pass(batch, is_train=self.training)

        # 3. 生成真实标签
        ground_truth = self._generate_gt_labels(batch)

        # 4. 计算损失
        loss_components = self._compute_losses(outputs, ground_truth)
        total_loss = self._aggregate_losses(loss_components)

        # 5. 计算评估指标（step级别）
        # c2f_idx = batch.c2f_idx.squeeze().to(self.device)
        # outputs_ori, gt_ori = self.map_to_ori(batch, outputs, ground_truth)
        # metrics = self._compute_metrics(outputs_ori, gt_ori, pos=pos[c2f_idx,:])

        metrics = self._compute_metrics(outputs, ground_truth, pos=pos)

        return {
            'loss': total_loss,
            'outputs': outputs,
            'ground_truth': ground_truth,
            'loss_components': loss_components,
            'metrics': metrics
        }

    def map_to_ori(self, batch, outputs, ground_truth):
        # B = batch.batch.max().item() + 1 if hasattr(batch, 'batch') else 1
        B = 1
        N = batch.pos.shape[0]

        seg_result = outputs['seg_result']

        c2f_idx = batch.c2f_idx.squeeze().to(seg_result.device)
        seg_result_onehot = torch.zeros(B, c2f_idx.shape[0], self.num_classes).to(seg_result.device)
        y_pred = seg_result.argmax(dim=-1, keepdim=True)  # [B, N, 1]
        y_pred_ori = y_pred[:, c2f_idx, :]
        seg_result_onehot.scatter_(2, y_pred_ori, 1)  # [B, N, C]

        outputs_ori = {}
        outputs_ori['seg_result'] = seg_result_onehot

        if 'edge_seg_result' in outputs and 'tte_seg_result' in outputs['edge_seg_result']:
            edge_seg_result = outputs['edge_seg_result']
            tte_seg_result = outputs['tte_seg_result']
            outputs_ori['edge_seg_result'] = edge_seg_result[:, :, c2f_idx]
            outputs_ori['tte_seg_result'] = tte_seg_result[:, :, c2f_idx]

        y_ori = batch.y_ori.to(device, dtype=torch.long)  # BN,
        onehot_y_ori = (torch.nn.functional.one_hot(y_ori, num_classes=self.num_classes).
                        reshape(B, -1, self.num_classes).to(device, dtype=torch.long))  # [B,N, 15]
        gt_edge_mask = ground_truth['gt_edge_mask']
        gt_tte_mask = ground_truth['gt_tte_mask']
        gt_edge_mask_ori = gt_edge_mask[c2f_idx]
        gt_tte_mask_ori = gt_tte_mask[c2f_idx]

        gt_ori = {}
        gt_ori['gt_labels'] = y_ori
        gt_ori['onehot_gt_labels'] = onehot_y_ori
        gt_ori['gt_edge_mask'] = gt_edge_mask_ori
        gt_ori['gt_tte_mask'] = gt_tte_mask_ori

        return outputs_ori, gt_ori

    def forward(self, x):
        return self.model(x).to(device)

    def training_step(self, batch, batch_idx):
        outputs = self._perform_step_forward(batch)

        loss = outputs['loss']
        metrics_dict = outputs['metrics']

        # 更新训练指标
        self.metrics_mgr.update(
            loss.item(), metrics_dict, is_train=True
        )

        # 记录日志
        metrics = self.metrics_mgr.get_metrics_dict(
            loss.item(), metrics_dict, is_train=True
        )

        self.log_dict(metrics)
        return loss

    # def validation_step(self, batch, batch_idx=0):
    #     outputs = self._perform_step_forward(batch)
    #
    #     loss = outputs['loss']
    #     loss_components = outputs['loss_components']
    #     seg_loss = loss_components['seg_loss']
    #     dice_loss = loss_components['dice_loss']
    #     # edge_loss = loss_components['edge_loss']
    #     # tte_loss = loss_components['tte_loss']
    #     cbl_loss = loss_components['cbl_loss']
    #     l2_loss = loss_components['l2_loss']
    #
    #     metrics_dict = outputs['metrics']
    #
    #     # 打印详细信息
    #     print(f"epoch:{self.current_epoch}-{batch.filename[0]}, "
    #           f"loss:{loss:.4f}=Lseg:{seg_loss:.4f} + Ldice:{dice_loss:.4f} "
    #           f"+ Lcbl:{cbl_loss:.4f} + L2:{l2_loss:.4f}", end=' ')
    #
    #     # 更新验证指标
    #     self.metrics_mgr.update(
    #         loss.item(), metrics_dict, is_train=False
    #     )
    #
    #     # 记录日志
    #     metrics = self.metrics_mgr.get_metrics_dict(
    #         loss.item(), metrics_dict, is_train=False
    #     )
    #
    #     self.log_dict(metrics)
    #     return metrics

    def validation_step(self, batch, batch_idx=0):
        outputs = self._perform_step_forward(batch)

        loss = outputs['loss']
        loss_components = outputs['loss_components']
        seg_loss = loss_components['seg_loss']
        dice_loss = loss_components['dice_loss']
        # edge_loss = loss_components['edge_loss']
        # tte_loss = loss_components['tte_loss']
        cbl_loss = loss_components['cbl_loss']
        l2_loss = loss_components['l2_loss']

        metrics_dict = outputs['metrics']

        # ========== 新增：计算 Crop 级别指标 ==========
        crops = outputs['outputs'].get('crops', None)
        crop_metrics_dict = {}

        if crops is not None:
            try:
                # 计算当前 batch 的 crop 指标
                all_crop_metrics = CropEval.compute_crop_metrics(
                    crops['crop_gt_binseg_label'],
                    crops['crop_pred_seg_labels'],
                    crops['cluster_gt_seg_labels']
                )

                # 提取平均指标（label_id == 'all'）
                avg_crop_metrics = next((m for m in all_crop_metrics if m.get('label_id') == 'all'), None)

                if avg_crop_metrics:
                    crop_metrics_dict = {
                        'val_crop_ACC': avg_crop_metrics.get('ACC', 0.0),
                        'val_crop_precision': avg_crop_metrics.get('precision', 0.0),
                        'val_crop_recall': avg_crop_metrics.get('recall', 0.0),
                        'val_crop_F1': avg_crop_metrics.get('F1', 0.0),
                        'val_crop_IOU': avg_crop_metrics.get('IOU', 0.0),
                    }

                    # 打印当前 batch 的 crop 指标摘要
                    print(f"| Crop: ACC={crop_metrics_dict['val_crop_ACC']:.4f}, "
                          f"Prec={crop_metrics_dict['val_crop_precision']:.4f}, "
                          f"Rec={crop_metrics_dict['val_crop_recall']:.4f}, "
                          f"F1={crop_metrics_dict['val_crop_F1']:.4f}, "
                          f"IoU={crop_metrics_dict['val_crop_IOU']:.4f}")
            except Exception as e:
                print(f"⚠️  Crop 指标计算失败: {e}")
                crop_metrics_dict = {}
        # ================================================

        # 打印详细信息
        print(f"epoch:{self.current_epoch}-{batch.filename[0]}, "
              f"loss:{loss:.4f}=Lseg:{seg_loss:.4f} + Ldice:{dice_loss:.4f} "
              f"+ Lcbl:{cbl_loss:.4f} + L2:{l2_loss:.4f}", end=' ')

        # 更新验证指标
        self.metrics_mgr.update(
            loss.item(), metrics_dict, is_train=False
        )

        # 记录日志
        metrics = self.metrics_mgr.get_metrics_dict(
            loss.item(), metrics_dict, is_train=False
        )

        # ========== 新增：记录 Crop 指标到日志 ==========
        if crop_metrics_dict:
            metrics.update(crop_metrics_dict)
        # ================================================

        self.log_dict(metrics)
        return metrics


    def predict_step(self, batch, batch_idx=0):
        outputs = self._perform_step_forward(batch)

        batch.outputs = outputs['outputs']
        batch.metrics = outputs['metrics']

        return batch

    def predict_step_forward(self, batch):
        # B = batch.batch.max().item() + 1 if hasattr(batch, 'batch') else 1
        x0 = batch.x.to(self.device, dtype=torch.float32)  # NC
        B = 1

        # 转换为BCN格式
        inputs_bcn = x0.permute([1, 0]).reshape(B, self.input_dim, -1)  # BCN格式

        # 2. 模型前向传播
        outputs = self._forward_pass(batch, is_train=False)

        batch.outputs = outputs

        return batch

    def regularization_losses(self):
        """计算L2正则化损失"""
        l2_lambda = 0.0001
        l2_reg = torch.tensor(0., requires_grad=True)

        for name, param in self.model.named_parameters():
            if 'weight' in name:
                l2_reg = l2_reg + torch.norm(param, 2)

        return l2_lambda * l2_reg

    def on_train_epoch_end(self):
        print("Training epoch ended.")
        optimizer = self.trainer.optimizers[0]
        current_lr = optimizer.param_groups[0]['lr']
        # 打印epoch级别的学习率
        print(f"===== Epoch {self.current_epoch} End, Current LR: {current_lr:.6f} =====")

        torch.cuda.empty_cache()
        print(f"Train Epoch end, GPU内存清理完成: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")

    def on_validation_epoch_end(self):
        self.metrics_mgr.compute(is_train=True)
        self.metrics_mgr.compute(is_train=False)
        self.metrics_mgr._print_epoch_info(self.current_epoch)

        self.metrics_mgr.reset(is_train=True)
        self.metrics_mgr.reset(is_train=False)

        torch.cuda.empty_cache()
        print(f"Val Epoch end, GPU内存清理完成: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")

    def on_train_batch_start(self, batch, batch_idx):
        if batch_idx % 60 == 0:
            torch.cuda.empty_cache()
            print(f"Batch start, GPU内存清理完成: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")

    def on_validation_batch_start(self, batch, batch_idx):
        if batch_idx % 60 == 0:
            torch.cuda.empty_cache()
            print(f"Validation batch start, GPU内存清理完成: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), amsgrad=True, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=5e-5)
        return [optimizer], [scheduler]


class PatchSegNetTrainer(BaseTrainer):
    """TMELFSegNet训练器，继承自BaseTrainer"""

    def __init__(self, config_path="../../configs", config_name="Module_PatchSegNet"):
        super().__init__(config_path, config_name)

    # 获得模型实例
    def build_module(self, model_obj: DictConfig, ckpt_path=None):
        """获取TMELFSegNet模型实例"""
        # 如果指定了检查点路径，则从检查点中加载模型
        if ckpt_path is not None and os.path.exists(ckpt_path):
            try:
                print(f"\n开始加载检查点...")

                module = PatchSegModule.load_from_checkpoint(
                    checkpoint_path=ckpt_path,
                    model_cfg=model_obj,  # 传递配置
                    num_classes=15,
                    strict=False,
                    map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                )
                print(f"✅ 成功加载检查点{ckpt_path}")

                return module
            except Exception as e:
                print(f"\n❌ 未能加载检查点{ckpt_path}: {e}")

        # 否则，创建新的模型实例
        name = model_obj["name"]
        if name == "PatchSegNetModule":
            module = PatchSegModule(model_obj, num_classes=15).to(device)
        else:
            raise ValueError(f"Invalid model name: {name}")

        return module

    # 准备数据
    def prepare_data(self, cfg_obj: DictConfig):
        if cfg_obj.dataset.name == "T3DSPlusSegDataset":
            _, self.train_loader, self.test_loader \
                = T3DSPlusSegDataset.get_dataloaders_from_cfg_obj(cfg_obj)
        elif cfg_obj.dataset.name == "TADPMSegDataset":
            _, self.train_loader, self.test_loader \
                = TADPMSegDataset.get_dataloaders_from_cfg_obj(cfg_obj)


############################################################################
## Training Testing Prediction
############################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config_path = "../../configs"
config_name = "Module_PatchSegNet"


def do_train():
    config_dir = os.path.abspath(config_path)

    # 可变参数
    # loss_type_list = ["SEG_CBL", "SEG", "SEG_TAE_TTE_CBL", "SEG_TAE_TTE"]
    # loss_type_list = ["SEG_TAE_TTE_CBL","SEG_TAE_TTE", "SEG_CBL","SEG"]
    loss_type_list = ["SEG_CBL"]

    for loss_type in loss_type_list:
        # 使用 Hydra compose API 动态覆盖配置
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            # 动态覆盖 loss_type
            cfg = compose(
                config_name=config_name,
                overrides=[
                    f"loss_type={loss_type}",
                ]
            )

            cfg_obj = instantiate_from_config(cfg)
            print(cfg_obj)

            # 创建 trainer
            trainer = PatchSegNetTrainer(config_path=config_path, config_name=config_name)
            trainer.set_cfg_obj(cfg_obj)  # 要用函数更新，否则会报错
            trainer.fit()

            # 验证
            # print(f"\n配置检查 (loss_type={loss_type}):")
            # print(f"  loss_type: {trainer.cfg_obj.loss_type}")
            # print(f"  model.loss_type: {trainer.cfg_obj.model.loss_type}")
            # print(f"  ckpt.exp_name: {trainer.cfg_obj.ckpt.exp_name}")
            # print(f"  logger.exp_args.lt: {trainer.cfg_obj.logger.exp_args.lt}")
            print(f"\n✅ 完成实验：{loss_type}-{trainer.cfg_obj.net_name}-{trainer.cfg_obj.net_para_name}\n")


def do_pred():
    # 1. 创建 trainer
    trainer = PatchSegNetTrainer(config_path=config_path, config_name=config_name)
    cfg_obj = trainer.cfg_obj

    # 2. 获取模型实例
    model_path = cfg_obj.prediction.model_path
    model_name = cfg_obj.prediction.model_name
    ckpt_path = os.path.join(model_path, model_name)

    cfg_obj.dataset.sample_ratio = 1.0

    # 3. 加载模型
    trainer.prepare_trainer(ckpt_path) if not trainer.trainer else None

    # 4. 获取模型实例
    module = trainer.module.to(device)  # 获取模型实例

    # 5. 测试集数据加载
    test_loader = trainer.test_loader

    # 6. 预测并保存结果
    eval_segnets.batch_insseg_pred(ckpt_path, module, test_loader,
                                   save_json=False, map_to_original=True)  # save instance segmentation results
    # batch_crop_pred(ckpt_path, module, test_loader, save_json=True, save_excel=True)               # save binary segmentation results


def test_get_model():
    import torch  # Add this import statement

    model = DTSegNet(num_classes=15, feature_dim=6)
    model.cuda()
    xyz = torch.rand(10000, 6).cuda()
    data = Data(x=xyz, pos=xyz[:, :3])
    data_list = [data]
    # 使用 DataLoader 批量处理
    loader = DataLoader(data_list, batch_size=1, shuffle=False)

    for batch in loader:
        B, C, N = batch.batch_size, batch.x.size(1), batch.x.size(0) // batch.batch_size

        x_bcn = batch.x.view(B, N, C).permute(0, 2, 1)  # 转换为 (B, C, N)
        pos_bcn = batch.pos.view(B, N, 3).permute(0, 2, 1)  # 转换为 (B, 3, N)

        seg_outputs, _ = model(x_bcn)

        # output is B, C, N order
        print(seg_outputs.shape)

    x_bcn = data.x.unsqueeze(0).permute(0, 2, 1)
    flops, params = profile(model, inputs=(x_bcn,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f G, params: %.2f M' % (flops / 1.0e9, params / 1.0e6))


def main():
    # 打印配置信息
    # print_yaml_config(config_path="../../configs", config_name="Module_PatchSegNet")
    # cfg_obj = instantiate_from_yaml_config(config_path="../../configs", config_name="Module_PatchSegNet")
    # print(cfg_obj)

    # 1. 测试数据读取
    # test_reader()
    # test_reader_2()

    # 2. 测试模型读取
    # test_get_model()

    # 3. 测试训练及预测
    do_train()
    # do_pred()

    # test_predict_1()


if __name__ == '__main__':
    main()
