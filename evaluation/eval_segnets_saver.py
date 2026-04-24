import json
from collections import OrderedDict

import numpy as np
import torch

from configs.hydra_yaml_config import *
from datasets import T3DSPlusSegDataset, TADPMSegDataset
from evaluation.ModuleFactory import ModuleFactory, auto_register_modules
from evaluation.seg_evaluation import *
import pandas as pd
from datetime import datetime
import os



def save_evaluation_results_to_excel(
    all_results,  # 改为接收所有结果的列表
    save_path=None
):
    """
    将评估结果保存到 Excel 文件（支持多条记录）

    Args:
        all_results: 列表，每个元素是一个字典，包含：
            {
                'filename': '样本文件名',
                'acc': 准确率,
                'iou': 交并比,
                'dsc': Dice 系数,
                'ppv': 精确率,
                'sen': 召回率,
                'SEM_ACC': 语义分割准确率,
                'boundary_iou': 边界 IoU,
                'background_iou': 背景 IoU,
                'merge_pairs_ious_dict': {合并的 pairs IoUs 字典}
            }
        save_path: Excel 保存路径
    """

    if save_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = f"evaluation_results_{timestamp}.xlsx"

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

    # ========== Sheet 1: 所有样本的详细指标 ==========
    all_records = []

    for result in all_results:
        filename = result.get('filename', 'Unknown')

        # 提取核心指标
        record = {
            'Filename': filename,
            'Accuracy (ACC)': f"{result['acc']:.4f}",
            'IoU': f"{result['iou']:.4f}",
            'Dice Score (DSC)': f"{result['dsc']:.4f}",
            'Precision (PPV)': f"{result['ppv']:.4f}",
            'Sensitivity/Recall (SEN)': f"{result['sen']:.4f}",
            'Semantic Segmentation ACC (SEM_ACC)': f"{result['SEM_ACC']:.4f}",
            'Boundary IoU': f"{result['boundary_iou']:.4f}",
            'Background IoU': f"{result['background_iou']:.4f}",
        }

        # 添加 mIOU 指标
        merge_pairs = result.get('merge_pairs_ious_dict', {})
        for i in range(1, 8):
            key = f'mIOU_{i}_{i+7}'
            value = merge_pairs.get(key, 0)
            record[key] = f"{value:.4f}" if isinstance(value, (int, float)) else str(value)

        all_records.append(record)

    df_details = pd.DataFrame(all_records)

    # ========== Sheet 2: 统计摘要 ==========
    # 计算统计信息
    numeric_cols = ['Accuracy (ACC)', 'IoU', 'Dice Score (DSC)', 'Precision (PPV)',
                    'Sensitivity/Recall (SEN)', 'Semantic Segmentation ACC (SEM_ACC)',
                    'Boundary IoU', 'Background IoU']

    miou_cols = [f'mIOU_{i}_{i+7}' for i in range(1, 8)]

    summary_data = {
        'Metric': [],
        'Mean': [],
        'Std': [],
        'Max': [],
        'Min': []
    }

    # 核心指标统计
    for col in numeric_cols + miou_cols:
        if col in df_details.columns:
            values = pd.to_numeric(df_details[col], errors='coerce')
            summary_data['Metric'].append(col)
            summary_data['Mean'].append(f"{values.mean():.4f}")
            summary_data['Std'].append(f"{values.std():.4f}")
            summary_data['Max'].append(f"{values.max():.4f}")
            summary_data['Min'].append(f"{values.min():.4f}")

    # 添加样本数量信息
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

    # ========== Sheet 3: 每个样本的 mIOU 详情（可选）==========
    miou_records = []
    for result in all_results:
        filename = result.get('filename', 'Unknown')
        merge_pairs = result.get('merge_pairs_ious_dict', {})

        for key, value in merge_pairs.items():
            miou_records.append({
                'Filename': filename,
                'Pair Name': key,
                'IoU Value': f"{value:.4f}" if isinstance(value, (int, float)) else str(value)
            })

    df_miou = pd.DataFrame(miou_records)

    # ========== 保存到 Excel ==========
    try:
        with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
            df_details.to_excel(writer, sheet_name='All Samples', index=False)
            df_summary.to_excel(writer, sheet_name='Statistics', index=False)
            df_miou.to_excel(writer, sheet_name='mIOU Details', index=False)

        print("\n" + "="*70)
        print("📊 评估结果已保存到 Excel")
        print("="*70)
        print(f"保存路径：{os.path.abspath(save_path)}")
        print(f"\n包含以下工作表:")
        print(f"  1. All Samples - 所有样本的详细指标 ({len(all_results)} 条记录)")
        print(f"  2. Statistics - 统计摘要 (均值/标准差/最大/最小)")
        print(f"  3. mIOU Details - 每个样本的 mIOU 详情 ({len(df_miou)} 条)")
        print("="*70)

        return save_path

    except Exception as e:
        print(f"❌ 保存失败：{e}")
        raise


def save_batch_info_to_json(batch, outputs, save_dir, only_save_pred=False):
    """
    保存 batch 的关键信息到 JSON 文件

    Args:
        batch: 输入的 batch 数据
        outputs: 模型输出
        save_dir: 保存目录
    """

    # 获取文件名
    filename = batch.filename[0] if isinstance(batch.filename, list) else batch.filename


    if only_save_pred:
        batch_data = {
            'filename': filename,
            'pos': batch.pos.cpu().numpy().tolist() if hasattr(batch, 'pos') and batch.pos is not None else None,
            'y': batch.y.cpu().numpy().tolist() if hasattr(batch, 'y') and batch.y is not None else None,
        }
    else:
        # 构建保存字典
        batch_data = {
            'filename': filename,

            # Ground Truth 数据
            'y': batch.y.cpu().numpy().tolist() if hasattr(batch, 'y') and batch.y is not None else None,
            'face': batch.face.cpu().numpy().tolist() if hasattr(batch, 'face') and batch.face is not None else None,
            'pos': batch.pos.cpu().numpy().tolist() if hasattr(batch, 'pos') and batch.pos is not None else None,
            'normals': batch.normals.cpu().numpy().tolist() if hasattr(batch,
                                                                       'normals') and batch.normals is not None else None,
            'instances': batch.instances.cpu().numpy().tolist() if hasattr(batch,
                                                                           'instances') and batch.instances is not None else None,
            # 'jaw': batch.jaw.cpu().numpy().tolist() if hasattr(batch, 'jaw') and batch.jaw is not None else None,
            'x': batch.x.cpu().numpy().tolist() if hasattr(batch, 'x') and batch.x is not None else None,
            'y_ori': batch.y_ori.cpu().numpy().tolist() if hasattr(batch, 'y_ori') and batch.y_ori is not None else None,
            'f2c_idx': batch.f2c_idx.cpu().numpy().tolist() if hasattr(batch,
                                                                       'f2c_idx') and batch.f2c_idx is not None else None,
            'c2f_idx': batch.c2f_idx.cpu().numpy().tolist() if hasattr(batch,
                                                                       'c2f_idx') and batch.c2f_idx is not None else None,
            'gt_edge_mask': batch.gt_edge_mask.cpu().numpy().tolist() if hasattr(batch,
                                                                                 'gt_edge_mask') and batch.gt_edge_mask is not None else None,
            'gt_tte_mask': batch.gt_tte_mask.cpu().numpy().tolist() if hasattr(batch,
                                                                               'gt_tte_mask') and batch.gt_tte_mask is not None else None,
            'instance_centroids': batch.instance_centroids.cpu().numpy().tolist() if hasattr(batch,
                                                                                             'instance_centroids') and batch.instance_centroids is not None else None,
            'instance_labels': batch.instance_labels.cpu().numpy().tolist() if hasattr(batch,
                                                                                       'instance_labels') and batch.instance_labels is not None else None,
            'per_instance_count': batch.per_instance_count.cpu().numpy().tolist() if hasattr(batch,
                                                                                             'per_instance_count') and batch.per_instance_count is not None else None,
            'instance_count': batch.instance_count.cpu().numpy().tolist() if hasattr(batch,
                                                                                     'instance_count') and batch.instance_count is not None else None,

        }

    # 安全获取预测结果并进行存在性判断
    if hasattr(outputs, 'outputs'):
        seg_result = outputs.outputs.get('seg_result', None)
        edge_seg_result = outputs.outputs.get('edge_seg_result', None)
        tte_seg_result = outputs.outputs.get('tte_seg_result', None)

        if seg_result is not None:
            pred_classes = torch.argmax(seg_result, dim=-1).squeeze().to(batch.device)
            batch_data['pred_seg'] = pred_classes.cpu().numpy().tolist()
        if edge_seg_result is not None:
            batch_data['pred_edge_seg'] = torch.argmax(edge_seg_result,
                                                            dim=1).squeeze().cpu().numpy().tolist()
        if tte_seg_result is not None:
            batch_data['pred_tte_seg'] = torch.argmax(tte_seg_result,
                                                           dim=1).squeeze().cpu().numpy().tolist()

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 保存为 JSON 文件
    json_path = os.path.join(save_dir, f"{filename}.json")

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(batch_data, f, ensure_ascii=False, indent=2)

    return json_path

def batch_insseg_pred(ckpt_path, module, test_loader, save_json=True, map_to_original=False):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_save_path = Path(ckpt_path).parent
    last_folder_name = Path(base_save_path).name
    # excel 保存目录
    excel_save_path = base_save_path / f"{last_folder_name}_eval_{timestamp}.xlsx"
    # JSON 保存目录
    json_save_dir = base_save_path / f"batch_json_{timestamp}"
    print(f"\n📂 JSON 保存目录：{json_save_dir}")
    # 存储所有样本的结果
    all_results = []
    # 遍历测试集
    for batch_idx, batch in enumerate(test_loader):
        print(f"评估样本 {batch_idx + 1}/{len(test_loader)}: {batch.filename}")

        pred_data = module.predict_step_forward(batch)

        if not map_to_original:
            gt_labels = pred_data.y.to(batch.device)
            pred = pred_data.outputs["seg_result"].to(batch.device)
            pred_classes = torch.argmax(pred, dim=-1).squeeze().to(batch.device)
            gt_edge_mask = pred_data.gt_edge_mask.to(batch.device)
        else:
            gt_labels = pred_data.y_ori.squeeze().to(batch.device)
            pred = pred_data.outputs["seg_result"].to(batch.device)
            pred_classes = torch.argmax(pred, dim=-1).squeeze().to(batch.device)
            pred_classes = pred_classes[batch.c2f_idx].squeeze().to(batch.device)
            gt_edge_mask = pred_data.gt_edge_mask[batch.c2f_idx].squeeze().to(batch.device)

        # segmentation metrics
        outs = cal_multi_cls_metric(gt_labels.cpu().numpy(), pred_classes.cpu().numpy(), pred_classes.cpu().numpy())
        acc, iou, dsc, ppv, sen, SEM_ACC, _ = outs

        # boundary metrics
        iou_outs = calc_bnd_bg_merged_iou(pred_classes, gt_labels, gt_edge_mask, num_classes=15)
        boundary_iou, background_iou, merge_pairs_ious_dict = iou_outs

        # edge metrics
        edge_binary_iou = 0.0
        tte_binary_iou = 0.0

        if pred_data.outputs.get('edge_seg_result') is not None and not map_to_original:
            pred_tae_mask = torch.argmax(pred_data.outputs['edge_seg_result'], dim=1).reshape(-1, )  # 结果形状 (B, N)
            edge_binary_iou = cal_edge_iou(gt_edge_mask.cpu().numpy(), pred_tae_mask.cpu().numpy(), target_class=1)
        if pred_data.outputs.get('tte_seg_result') is not None and not map_to_original:
            pred_tte_mask = torch.argmax(pred_data.outputs['tte_seg_result'], dim=1).reshape(-1, )
            tte_binary_iou = cal_edge_iou(gt_edge_mask.cpu().numpy(), pred_tte_mask.cpu().numpy(), target_class=1)

        # 打印当前样本结果
        print(f"acc: {acc:.4f}, iou={iou:.4f}, dsc={dsc:.4f}, ppv={ppv:.4f}, sen={sen:.4f}, SEM_ACC={SEM_ACC:.4f}")
        print(f"boundary_iou={boundary_iou:.4f}, background_iou={background_iou:.4f}")
        print(f"edge_binary_iou={edge_binary_iou:.4f}, tte_binary_iou={tte_binary_iou:.4f}")

        merge_pairs_str = ", ".join([f"{k}={v:.4f}" if isinstance(v, (int, float)) else f"{k}={v}"
                                     for k, v in merge_pairs_ious_dict.items()])
        print(f"merge_pairs_ious: {merge_pairs_str}")

        # 保存 batch 信息到 JSON
        if save_json:
            try:
                json_path = save_batch_info_to_json(batch, pred_data, json_save_dir)
                print(f"✅ JSON 已保存：{os.path.basename(json_path)}")
            except Exception as e:
                print(f"❌ JSON 保存失败：{e}")

        # 添加到结果列表
        result = {
            'filename': batch.filename[0] if isinstance(batch.filename, list) else batch.filename,
            'acc': acc,
            'iou': iou,
            'dsc': dsc,
            'ppv': ppv,
            'sen': sen,
            'SEM_ACC': SEM_ACC,
            'boundary_iou': boundary_iou,
            'background_iou': background_iou,
            'edge_binary_iou': edge_binary_iou,
            'tte_binary_iou': tte_binary_iou,
            'merge_pairs_ious_dict': merge_pairs_ious_dict
        }
        all_results.append(result)

        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()
            print(f"Validation batch start, GPU内存清理完成: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")


    # 保存所有结果到 Excel
    save_evaluation_results_to_excel(
        all_results=all_results,
        save_path=excel_save_path
    )

# def prepare_data(cfg_obj: DictConfig):
#     if cfg_obj.dataset.name == "T3DSPlusSegDataset":
#         dm, train_loader, test_loader \
#             = T3DSPlusSegDataset.get_dataloaders_from_cfg_obj(cfg_obj)
#     elif cfg_obj.dataset.name == "TADPMSegDataset":
#         dm, train_loader, test_loader \
#             = TADPMSegDataset.get_dataloaders_from_cfg_obj(cfg_obj)
#
#     return dm, train_loader, test_loader
#
#
# def get_modules(cfg_obj: DictConfig, name="DTSegNetModule"):
#     modules_config = {
#         "DTSegNetModule": {
#             "ckpt_path": cfg_obj.predictors.DTSegNetModule.ckpt_cfg.full_path,
#             "model_cfg": cfg_obj.predictors.DTSegNetModule.model_cfg
#         },
#         "PTSegNetModule": {
#             "ckpt_path": cfg_obj.predictors.PTSegNetModule.ckpt_cfg.full_path,
#             "model_cfg": cfg_obj.predictors.PTSegNetModule.model_cfg
#         },
#         "TMELFSegModule": {
#             "ckpt_path": cfg_obj.predictors.TMELFSegModule.ckpt_cfg.full_path,
#             "model_cfg": cfg_obj.predictors.TMELFSegModule.model_cfg
#         },
#     }
#
#     assert name in modules_config.keys()
#     return modules_config[name]
#
#

#
#
# def process_one():
#     print("Current working directory:", os.getcwd())
#
#     config_path = "../configs"
#     config_name = "eval_segnet"
#     cfg = get_yaml_config(config_path=config_path, config_name=config_name)
#     args = cfg.paths
#     ds_args = cfg.dataset
#     print_yaml_cfg_dict(cfg)
#
#     cfg_obj = instantiate_from_config(cfg)
#     print(cfg_obj)
#
#     _, _, test_loader =prepare_data(cfg_obj)
#     print(test_loader)
#
#     auto_register_modules()
#     # DTSegNetModule
#     dtseg_module_cfg = get_modules(cfg_obj, name="DTSegNetModule")
#     dtseg_module = ModuleFactory.load_from_checkpoint(
#         module_name="DTSegNetModule",
#         checkpoint_path=dtseg_module_cfg["ckpt_path"],
#         model_cfg=dtseg_module_cfg["model_cfg"],
#         map_location="cuda"
#     )
#
#     # ptseg_module_cfg = get_modules(cfg_obj, name="PTSegNetModule")
#     # ptseg_module = ModuleFactory.load_from_checkpoint(
#     #     module_name="PTSegNetModule",
#     #     checkpoint_path=ptseg_module_cfg["ckpt_path"],
#     #     model_cfg=ptseg_module_cfg["model_cfg"],
#     #     map_location="cuda"
#     # )
#
#
#     batch = next(iter(test_loader))
#     outputs = dtseg_module.predict_step_forward(batch)
#     print(outputs)
#
#     gt_labels = outputs.y.to(device)
#     pred = outputs.outputs["seg_result"].to(device)  #seg_result
#     pred_classes = torch.argmax(pred, dim=-1).squeeze().to(device)
#     gt_edge_mask = outputs.gt_edge_mask.to(device)
#
#     outs = cal_multi_cls_metric(gt_labels.cpu().numpy(), pred_classes.cpu().numpy(), pred_classes.cpu().numpy())
#     # ACC, IOU, F1, PRECISION, RECALL, SEM_ACC, IOU_arr = outs
#     acc, iou, dsc, ppv, sen,SEM_ACC, _ = outs
#
#     iou_outs = calc_bnd_bg_merged_iou(pred_classes, gt_labels, gt_edge_mask, num_classes=15)
#     boundary_iou, background_iou, merge_pairs_ious_dict = iou_outs
#
#     print(f"acc: {acc:.4f}, iou={iou:.4f}, dsc={dsc:.4f}, ppv={ppv:.4f}, sen={sen:.4f}, SEM_ACC={SEM_ACC:.4f}")
#     print(f"boundary_iou={boundary_iou:.4f}, background_iou={background_iou:.4f}")
#     merge_pairs_str = ", ".join([f"{k}={v:.4f}" if isinstance(v, (int, float)) else f"{k}={v}"
#                                  for k, v in merge_pairs_ious_dict.items()])
#     print(f"merge_pairs_ious: {merge_pairs_str}")
#
#
# def batch_process():
#     print("Current working directory:", os.getcwd())
#
#     config_path = "../configs"
#     config_name = "eval_segnet"
#     cfg = get_yaml_config(config_path=config_path, config_name=config_name)
#     args = cfg.paths
#     ds_args = cfg.dataset
#     print_yaml_cfg_dict(cfg)
#
#     cfg_obj = instantiate_from_config(cfg)
#     print(cfg_obj)
#
#     _, _, test_loader =prepare_data(cfg_obj)
#     print(test_loader)
#
#     auto_register_modules()
#     # DTSegNetModule
#     # name = "DTSegNetModule"
#     name = "PTSegNetModule"
#     # name = "TMELFSegModule"
#     module_cfg = get_modules(cfg_obj, name=name)
#     module = ModuleFactory.load_from_checkpoint(
#         module_name=name,
#         checkpoint_path=module_cfg["ckpt_path"],
#         model_cfg=module_cfg["model_cfg"],
#         map_location="cuda"
#     )
#
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#     base_save_path = Path(module_cfg["ckpt_path"]).parent
#     excel_save_path = base_save_path / f"eval_results_{timestamp}.xlsx"
#
#     # JSON 保存目录
#     json_save_dir = base_save_path / f"batch_json_{timestamp}"
#     print(f"\n📂 JSON 保存目录：{json_save_dir}")
#
#
#     # 存储所有样本的结果
#     all_results = []
#
#     # 遍历测试集
#     for batch_idx, batch in enumerate(test_loader):
#         print(f"评估样本 {batch_idx + 1}/{len(test_loader)}: {batch.filename}")
#
#         outputs = module.predict_step_forward(batch)
#
#         gt_labels = outputs.y.to(device)
#         pred = outputs.outputs["seg_result"].to(device)
#         pred_classes = torch.argmax(pred, dim=-1).squeeze().to(device)
#         gt_edge_mask = outputs.gt_edge_mask.to(device)
#
#         outs = cal_multi_cls_metric(gt_labels.cpu().numpy(), pred_classes.cpu().numpy(), pred_classes.cpu().numpy())
#         acc, iou, dsc, ppv, sen, SEM_ACC, _ = outs
#
#         iou_outs = calc_bnd_bg_merged_iou(pred_classes, gt_labels, gt_edge_mask, num_classes=15)
#         boundary_iou, background_iou, merge_pairs_ious_dict = iou_outs
#
#         # 打印当前样本结果
#         print(f"acc: {acc:.4f}, iou={iou:.4f}, dsc={dsc:.4f}, ppv={ppv:.4f}, sen={sen:.4f}, SEM_ACC={SEM_ACC:.4f}")
#         print(f"boundary_iou={boundary_iou:.4f}, background_iou={background_iou:.4f}")
#         merge_pairs_str = ", ".join([f"{k}={v:.4f}" if isinstance(v, (int, float)) else f"{k}={v}"
#                                      for k, v in merge_pairs_ious_dict.items()])
#         print(f"merge_pairs_ious: {merge_pairs_str}")
#
#         # 保存 batch 信息到 JSON
#         try:
#             json_path = save_batch_info_to_json(batch, outputs, json_save_dir)
#             print(f"✅ JSON 已保存：{os.path.basename(json_path)}")
#         except Exception as e:
#             print(f"❌ JSON 保存失败：{e}")
#
#         # 添加到结果列表
#         result = {
#             'filename': batch.filename[0] if isinstance(batch.filename, list) else batch.filename,
#             'acc': acc,
#             'iou': iou,
#             'dsc': dsc,
#             'ppv': ppv,
#             'sen': sen,
#             'SEM_ACC': SEM_ACC,
#             'boundary_iou': boundary_iou,
#             'background_iou': background_iou,
#             'merge_pairs_ious_dict': merge_pairs_ious_dict
#         }
#         all_results.append(result)
#
#     # 保存所有结果到 Excel
#     save_evaluation_results_to_excel(
#         all_results=all_results,
#         save_path=excel_save_path
#     )
#
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#
# def main():
#     # process_one()
#     batch_process()
#
# if __name__ == '__main__':
#     main()