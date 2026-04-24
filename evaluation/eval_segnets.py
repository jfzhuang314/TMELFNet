import glob
import json
import time
from collections import OrderedDict

import numpy as np
import torch
from scipy import sparse
from scipy.spatial import KDTree
from torch_geometric.data import Batch

import torch.nn.functional as F
from configs.hydra_yaml_config import *
from datasets import T3DSPlusSegDataset, TADPMSegDataset
from datasets.transform.PointSamplingIndexer import NNPointSamplingIndexer
from datasets.transform.TopoMorphCluster import TopoMorphCluster
from datasets.transform.t3dsPlus_transforms import DynamicRadiusNNEdgeMask
from evaluation.ModuleFactory import ModuleFactory, auto_register_modules
from evaluation.eval_TMC import ClusterEval
from evaluation.seg_evaluation import *
import pandas as pd
from datetime import datetime
import os
import external_libs.pv_gen_utils as pvgu

import warnings

from external_libs.mesh_utils import check_n3
from external_libs.mesh_vis import TMCVisPV
import pyvista as pv

warnings.filterwarnings('ignore', category=UserWarning)



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
            pred_classes = torch.argmax(seg_result, dim=-1).squeeze().to(device)
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


def bilateral_label_transfer_fast(pos_coarse, pos_fine, pred_labels,
                                  sigma_spatial=0.5, k_neighbors=20):
    """
    更快的双边滤波实现（使用稀疏矩阵）

    Args:
        pos_coarse: 粗网格点云坐标 (N_c, 3)
        pos_fine: 细网格点云坐标 (N_f, 3)
        pred_labels: 粗网格预测标签 (N_c,)
        sigma_spatial: 空间域标准差
        k_neighbors: 搜索的近邻数量

    Returns:
        ndarray: 细网格的平滑标签 (N_f,)
    """


    if isinstance(pos_coarse, torch.Tensor):
        pos_coarse = pos_coarse.cpu().numpy()
    if isinstance(pos_fine, torch.Tensor):
        pos_fine = pos_fine.cpu().numpy()
    if isinstance(pred_labels, torch.Tensor):
        pred_labels = pred_labels.cpu().numpy()

    N_f = pos_fine.shape[0]
    num_classes = int(np.max(pred_labels) + 1)

    print(f"   🚀 快速双边滤波: {N_f} 个点, {num_classes} 类")

    # 构建KD树
    tree = KDTree(pos_coarse)
    distances, indices = tree.query(pos_fine, k=k_neighbors)

    if len(distances.shape) == 1:
        distances = distances.reshape(-1, 1)
        indices = indices.reshape(-1, 1)

    # 计算权重
    weights = np.exp(-distances ** 2 / (2 * sigma_spatial ** 2))
    weight_sum = np.sum(weights, axis=1, keepdims=True)
    weights = weights / (weight_sum + 1e-8)

    # 使用稀疏矩阵加速投票
    # 构建稀疏权重矩阵 (N_f, N_c)
    row_idx = np.repeat(np.arange(N_f), k_neighbors)
    col_idx = indices.flatten()
    weight_vals = weights.flatten()

    sparse_weights = sparse.csr_matrix(
        (weight_vals, (row_idx, col_idx)),
        shape=(N_f, len(pred_labels))
    )

    # 构建标签矩阵 (N_c, num_classes)
    label_matrix = np.zeros((len(pred_labels), num_classes))
    label_matrix[np.arange(len(pred_labels)), pred_labels] = 1

    # 矩阵乘法得到软投票 (N_f, num_classes)
    soft_votes = sparse_weights.dot(label_matrix)

    # 取最大值
    fine_labels = np.argmax(soft_votes, axis=1)

    return fine_labels


class SegPlotter:
    """分割结果可视化器"""

    @staticmethod
    def check_inputs(merged_batch):
        """检查输入数据是否完整"""
        if merged_batch is None:
            print("❌ merged_batch 为 None")
            return False

        required_attrs = ['pos_ori', 'face_ori', 'y_ori', 'net_pred_results']
        for attr in required_attrs:
            if not hasattr(merged_batch, attr):
                print(f"⚠️ 缺少必需字段: {attr}")
                return False

        if not merged_batch.net_pred_results:
            print("⚠️ net_pred_results 为空")
            return False

        return True

    @staticmethod
    def map_predictions_to_original(pred_classes, c2f_idx=None):
        """
        将预测结果映射到原始点云

        Args:
            pred_classes: 预测类别数组
            c2f_idx: coarse to fine 索引映射

        Returns:
            ndarray: 映射后的预测类别
        """
        if c2f_idx is not None:
            if isinstance(c2f_idx, torch.Tensor):
                c2f_idx = c2f_idx.cpu().numpy()
            if len(c2f_idx.shape) > 1:
                c2f_idx = c2f_idx.squeeze()
            return pred_classes[c2f_idx]
        return pred_classes

    @staticmethod
    def plot_pcd(pos, labels, save_path=None, window_size=None,
                 elevation=0, azimuth=0, margin=20, enable_lighting=False, point_size=3.0):
        """
        比较预测边缘掩码与真实边缘掩码（点云版本，支持交互）

        Args:
            pos: 点云坐标 (N, 3)
            labels: 标签数组 (N,)
            save_path: 保存路径，如果为None则只显示不保存
            window_size: 窗口尺寸 [width, height]，默认 [2880, 1620]
            elevation: 仰角（度），默认 90（俯视）
            azimuth: 方位角（度），默认 0
            margin: 边距（像素），默认 20
            enable_lighting: 是否启用光照，默认 False
            point_size: 点大小，默认 3.0
        """
        if isinstance(pos, torch.Tensor):
            pos = pos.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        if window_size is None:
            window_size = [int(1920 * 1.5), int(1080 * 1.5)]

        pcd = pvgu.np_to_pcd_with_label(pos, labels)

        # 创建绘图器（交互式）
        plotter = pv.Plotter(window_size=window_size, off_screen=False)
        plotter.background_color = 'white'

        # 添加点云（使用RGB颜色）
        plotter.add_points(pcd, scalars='colors', rgb=True,
                           render_points_as_spheres=True, point_size=point_size)

        # 设置视角
        plotter.view_xy()
        plotter.camera.elevation = elevation
        plotter.camera.azimuth = azimuth
        plotter.zoom_camera(1.5)

        plotter.show()
        plotter.close()

        # 如果需要保存，先截图再显示
        if save_path:
            mesh_list = [pcd]
            elevation = 90
            azimuth = 0

            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            TMCVisPV.vis_mesh_grid_separate_render(
                mesh_list,
                rows=1,
                cols=1,
                window_size=window_size,
                elevation=elevation,
                azimuth=azimuth,
                save_path=save_path,
                margin=margin,
                enable_lighting=enable_lighting,
                point_size=point_size,
                line_width=1.0,
            )

            print(f"✅ 点云图已保存: {save_path}")


    @staticmethod
    def plot_seg_result(pos, gt_labels, pred_labels, save_path=None):
        """
        比较预测边缘掩码与真实边缘掩码（点云版本，支持交互）

        Args:
            pos: 点云坐标 (N, 3)
            pred_edge_mask: 预测边缘掩码 (N,)
            gt_edge_mask: 真实边缘掩码 (N,)
            save_path: 保存路径，如果为None则只显示不保存
        """
        if isinstance(pos, torch.Tensor):
            pos = pos.cpu().numpy()
        if isinstance(gt_labels, torch.Tensor):
            gt_labels = gt_labels.cpu().numpy()
        if isinstance(pred_labels, torch.Tensor):
            tae_edge_mask = pred_labels.cpu().numpy()

        pred_pos = pos + np.array([1.5, 0.0, 0.0])

        gt_pcd = pvgu.np_to_pcd_with_label(pos, gt_labels)
        pred_pcd = pvgu.np_to_pcd_with_label(pred_pos, pred_labels)

        window_size = [int(1920 * 1.5), int(1080 * 1.5)]
        # # 创建绘图器（交互式）
        plotter = pv.Plotter(window_size=window_size, off_screen=False)
        plotter.background_color = 'white'

        # 添加点云（使用RGB颜色）
        plotter.add_points(gt_pcd, scalars='colors', rgb=True,
                           render_points_as_spheres=True, point_size=3.0)
        plotter.add_points(pred_pcd, scalars='colors', rgb=True,
                           render_points_as_spheres=True, point_size=3.0)

        # 交互式显示
        plotter.view_xy()
        plotter.zoom_camera(1.5)

        plotter.show()
        plotter.close()

        # 如果需要保存，先截图再显示
        if save_path:
            # 创建带颜色的点云
            mesh_list = [
                gt_pcd,
                pred_pcd,
            ]

            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            # plotter.screenshot(save_path, window_size=[int(1920 * 2), int(1080 * 2)])
            TMCVisPV.vis_mesh_grid_separate_render(
                mesh_list,
                rows=1,
                cols=2,
                window_size=window_size,
                elevation=90,
                azimuth=0,
                save_path=save_path,
                margin=20,
                enable_lighting=False,
                point_size=3.0,
                line_width=1.0,
            )

            print(f"✅ 边缘对比图已保存: {save_path}")

    @staticmethod
    def compare_edge_masks_pointcloud(pos, pred_edge_mask, tae_edge_mask, save_path=None):
        """
        比较预测边缘掩码与真实边缘掩码（点云版本，支持交互）

        Args:
            pos: 点云坐标 (N, 3)
            pred_edge_mask: 预测边缘掩码 (N,)
            gt_edge_mask: 真实边缘掩码 (N,)
            save_path: 保存路径，如果为None则只显示不保存
        """
        if isinstance(pos, torch.Tensor):
            pos = pos.cpu().numpy()
        if isinstance(pred_edge_mask, torch.Tensor):
            pred_edge_mask = pred_edge_mask.cpu().numpy()
        if isinstance(tae_edge_mask, torch.Tensor):
            tae_edge_mask = tae_edge_mask.cpu().numpy()


        # 创建RGB颜色数组
        # colors = np.zeros((len(pos), 3), dtype=np.float32)
        # colors[:,2] = 1
        pred_edge_colors = np.repeat([0, 0, 1.0], len(pos)).reshape(-1, 3)
        pred_edge_colors[pred_edge_mask==1,: ] = [0, 1, 0]
        tae_edge_colors = np.repeat([0, 0, 1.0], len(pos)).reshape(-1, 3)
        tae_edge_colors[tae_edge_mask==1,: ] = [0, 1, 0]
        tae_points = pos + np.array([1.5, 0.0, 0.0])

        pred_edge_pcd = pv.PolyData(pos)
        pred_edge_pcd.point_data['colors'] = pred_edge_colors
        tae_edge_pcd = pv.PolyData(tae_points)
        tae_edge_pcd.point_data['colors'] = tae_edge_colors

        window_size = [int(1920 * 1.5), int(1080 * 1.5)]
        # # 创建绘图器（交互式）
        plotter = pv.Plotter(window_size=window_size, off_screen=False)
        plotter.background_color = 'white'

        # 添加点云（使用RGB颜色）
        plotter.add_points(pred_edge_pcd, scalars='colors', rgb=True,
                           render_points_as_spheres=True, point_size=3.0)
        plotter.add_points(tae_edge_pcd, scalars='colors', rgb=True,
                           render_points_as_spheres=True, point_size=3.0)

        # 交互式显示
        plotter.view_xy()
        plotter.zoom_camera(1.5)

        # if save_path:
        #     os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        #     plotter.screenshot(save_path)

        plotter.show()
        plotter.close()

        # 如果需要保存，先截图再显示
        if save_path:
            # 创建带颜色的点云
            mesh_list = [
                pred_edge_pcd,
                tae_edge_pcd,
            ]

            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            # plotter.screenshot(save_path, window_size=[int(1920 * 2), int(1080 * 2)])
            TMCVisPV.vis_mesh_grid_separate_render(
                mesh_list,
                rows=1,
                cols=2,
                window_size=window_size,
                elevation=90,
                azimuth=0,
                save_path=save_path,
                margin=20,
                enable_lighting=False,
                point_size=3.0,
                line_width=1.0,
            )

            print(f"✅ 边缘对比图已保存: {save_path}")


    @staticmethod
    def plot_seg_result(pos, gt_labels, pred_labels, save_path=None):
        """
        比较预测边缘掩码与真实边缘掩码（点云版本，支持交互）

        Args:
            pos: 点云坐标 (N, 3)
            pred_edge_mask: 预测边缘掩码 (N,)
            gt_edge_mask: 真实边缘掩码 (N,)
            save_path: 保存路径，如果为None则只显示不保存
        """
        if isinstance(pos, torch.Tensor):
            pos = pos.cpu().numpy()
        if isinstance(gt_labels, torch.Tensor):
            gt_labels = gt_labels.cpu().numpy()
        if isinstance(pred_labels, torch.Tensor):
            tae_edge_mask = pred_labels.cpu().numpy()

        gt_pcd = pvgu.np_to_pcd_with_label(pos, gt_labels)
        pred_pcd = pvgu.np_to_pcd_with_label(pos, pred_labels)

        window_size = [int(1920 * 1.5), int(1080 * 1.5)]
        # # 创建绘图器（交互式）
        plotter = pv.Plotter(window_size=window_size, off_screen=False)
        plotter.background_color = 'white'

        # 添加点云（使用RGB颜色）
        plotter.add_points(gt_pcd, scalars='colors', rgb=True,
                           render_points_as_spheres=True, point_size=3.0)
        plotter.add_points(pred_pcd, scalars='colors', rgb=True,
                           render_points_as_spheres=True, point_size=3.0)

        # 交互式显示
        plotter.view_xy()
        plotter.zoom_camera(1.5)

        plotter.show()
        plotter.close()

        # 如果需要保存，先截图再显示
        if save_path:
            # 创建带颜色的点云
            mesh_list = [
                gt_pcd,
                pred_pcd,
            ]

            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            # plotter.screenshot(save_path, window_size=[int(1920 * 2), int(1080 * 2)])
            TMCVisPV.vis_mesh_grid_separate_render(
                mesh_list,
                rows=1,
                cols=2,
                window_size=window_size,
                elevation=90,
                azimuth=0,
                save_path=save_path,
                margin=20,
                enable_lighting=False,
                point_size=3.0,
                line_width=1.0,
            )

            print(f"✅ 边缘对比图已保存: {save_path}")


    @staticmethod
    def plot_segmentation_results(merged_data, save_dir=None, cmap_name='label15',
                                  network_order=None, save_metrics=True):
        """
        可视化所有样本的分割结果

        Args:
            merged_data: merge_predictions_with_raw_data() 返回的数据
            save_dir: 保存目录
            cmap_name: 颜色映射名称
            network_order: 网络渲染顺序列表
            save_metrics: 是否保存评估指标到Excel
        """

        if save_dir is None:
            save_dir = "./segnet_predictions/seg_results"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

        # 默认网络顺序
        if network_order is None:
            network_order = [
                'pointnetpp',
                'teethgnn',
                'tsegnet',
                'ptsegnet',
                'dtsegnet',
                'tmelfseg'
            ]

        success_count = 0
        fail_count = 0

        # 存储所有评估结果
        all_metrics = []

        for filename, merged_batch in merged_data.items():
            try:
                if not SegPlotter.check_inputs(merged_batch):
                    fail_count += 1
                    continue

                # 获取原始几何信息
                pos_ori = merged_batch.pos_ori
                face_ori = merged_batch.face_ori
                y_ori = merged_batch.y_ori.squeeze()
                pos = merged_batch.pos

                if isinstance(pos_ori, torch.Tensor):
                    pos_ori = pos_ori.cpu().numpy()
                if isinstance(face_ori, torch.Tensor):
                    face_ori = face_ori.cpu().numpy()
                if isinstance(y_ori, torch.Tensor):
                    y_ori = y_ori.cpu().numpy()

                # 获取 c2f_idx（如果存在）
                c2f_idx = None
                if hasattr(merged_batch, 'c2f_idx'):
                    c2f_idx = merged_batch.c2f_idx

                # 创建基础网格
                V = check_n3(pos_ori)
                F = check_n3(face_ori)

                # 构建网格列表（按指定顺序）
                mesh_list = []
                titles = []

                # 1. Ground Truth
                gt_mesh = pvgu.np_to_mesh(V, F)
                gt_mesh = pvgu.get_colored_mesh(gt_mesh, y_ori.astype(np.int32), palette_type=cmap_name)

                # 2. 按照指定顺序处理各个网络的预测结果
                net_pred_results = merged_batch.net_pred_results

                for net_name in network_order:
                    # 检查该网络是否存在于预测结果中
                    if net_name not in net_pred_results:
                        print(f"   ⚠️ {net_name}: 未在预测结果中找到，跳过")
                        continue

                    pred_classes = net_pred_results[net_name]

                    if isinstance(pred_classes, torch.Tensor):
                        pred_classes = pred_classes.cpu().numpy()

                    # 判断映射方向并选择合适的方法
                    use_bilateral = False
                    transfer_method = "Index"

                    # 检测是否需要双边滤波（粗→细映射）
                    # if hasattr(merged_batch, 'c2f_idx') and c2f_idx is not None:
                    #     if len(c2f_idx) > len(pred_classes):
                    #         use_bilateral = True
                    #         transfer_method = "Bilateral"

                    # 对于特定网络强制使用双边滤波
                    if net_name in ['tmelfseg']:
                        if len(pos) < len(pos_ori):
                            use_bilateral = True
                            transfer_method = "Bilateral"

                    # 执行标签传递
                    if use_bilateral:
                        bilateral_params = {
                            'sigma_spatial': 0.25,
                            'k_neighbors': 20
                        }

                        pred_mapped = bilateral_label_transfer_fast(
                            pos_coarse=pos,
                            pos_fine=pos_ori,
                            pred_labels=pred_classes,
                            sigma_spatial=bilateral_params['sigma_spatial'],
                            k_neighbors=bilateral_params['k_neighbors']
                        )
                    else:
                        pred_mapped = SegPlotter.map_predictions_to_original(pred_classes, c2f_idx)

                    # 确保长度匹配
                    if len(pred_mapped) != len(V):
                        print(f"   ⚠️ {net_name}: 预测长度 {len(pred_mapped)} != 点数 {len(V)}，跳过")
                        continue

                    # ========== 计算评估指标 ==========
                    if len(pred_mapped) == len(y_ori):
                        # 核心分割指标
                        outs = cal_multi_cls_metric(
                            y_ori,
                            pred_mapped,
                            pred_mapped
                        )
                        acc, iou, dsc, ppv, sen, sem_acc, _ = outs

                        # 边界指标（如果有gt_edge_mask）
                        boundary_iou = 0.0
                        background_iou = 0.0
                        merge_pairs_dict = {}

                        if hasattr(merged_batch, 'gt_edge_mask'):
                            gt_edge_mask = merged_batch.gt_edge_mask
                            if isinstance(gt_edge_mask, torch.Tensor):
                                gt_edge_mask = gt_edge_mask.cpu().numpy()

                            # 确保长度匹配
                            if len(gt_edge_mask) == len(pred_mapped):
                                try:
                                    iou_outs = calc_bnd_bg_merged_iou(
                                        torch.LongTensor(pred_mapped),
                                        torch.LongTensor(y_ori),
                                        torch.LongTensor(gt_edge_mask),
                                        num_classes=15
                                    )
                                    boundary_iou, background_iou, merge_pairs_dict = iou_outs
                                except Exception as e:
                                    print(f"   ⚠️ {net_name}: 边界指标计算失败: {e}")

                        # 记录指标
                        metric_record = {
                            'filename': filename,
                            'network': net_name,
                            'accuracy': acc,
                            'iou': iou,
                            'dice': dsc,
                            'precision': ppv,
                            'recall': sen,
                            'sem_acc': sem_acc,
                            'boundary_iou': boundary_iou,
                            'background_iou': background_iou,
                            'transfer_method': transfer_method,
                        }

                        # 添加 mIOU 指标
                        for key, value in merge_pairs_dict.items():
                            metric_record[key] = value

                        all_metrics.append(metric_record)

                        title_text = f"{net_name}\nAcc={acc:.3f}"
                    else:
                        title_text = f"{net_name}\n(N/A)"

                    # 创建预测网格
                    pred_mesh = pvgu.np_to_mesh(V, F)
                    pred_mesh = pvgu.get_colored_mesh(pred_mesh, pred_mapped.astype(np.int32), palette_type=cmap_name)
                    mesh_list.append(pred_mesh)
                    titles.append(title_text)

                # 将 GT 添加到最后
                mesh_list.append(gt_mesh)
                titles.append('Ground Truth')

                if len(mesh_list) == 0:
                    fail_count += 1
                    continue

                # 计算布局
                cols = len(mesh_list)
                rows = 1

                # 保存路径
                save_path = os.path.join(save_dir, f"{filename}.png")

                # 渲染并保存
                TMCVisPV.vis_mesh_grid_separate_render(
                    mesh_list,
                    rows=rows,
                    cols=cols,
                    window_size=[int(1920 * 1.5), int(1080 * 1.5)],
                    elevation=90,
                    azimuth=0,
                    save_path=save_path,
                    margin=20,
                    enable_lighting=False,
                    point_size=3.0,
                    line_width=1.0,
                    titles=titles if len(titles) == len(mesh_list) else None
                )

                print(f"   ✅ {filename}: {len(mesh_list)} 个网格已保存")
                success_count += 1

            except Exception as e:
                print(f"   ❌ {filename} 可视化失败: {e}")
                import traceback
                traceback.print_exc()
                fail_count += 1

        print(f"\n{'='*70}")
        print(f"📊 可视化统计")
        print(f"{'='*70}")
        print(f"✅ 成功: {success_count} 个样本")
        print(f"❌ 失败: {fail_count} 个样本")
        print(f"📁 保存位置: {save_dir}")
        print(f"{'='*70}\n")

        # ========== 保存评估指标到 Excel ==========
        if save_metrics and all_metrics:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            excel_save_path = os.path.join(save_dir, "csv" ,f"segmentation_metrics_{timestamp}.xlsx")
            os.makedirs(os.path.dirname(excel_save_path), exist_ok=True)
            # 转换为适合Excel的格式
            excel_records = []
            for record in all_metrics:
                excel_row = {
                    'Filename': record['filename'],
                    'Network': record['network'],
                    'Transfer Method': record['transfer_method'],
                    'Accuracy': f"{record['accuracy']:.4f}",
                    'IoU': f"{record['iou']:.4f}",
                    'Dice Score': f"{record['dice']:.4f}",
                    'Precision': f"{record['precision']:.4f}",
                    'Recall': f"{record['recall']:.4f}",
                    'Semantic ACC': f"{record['sem_acc']:.4f}",
                    'Boundary IoU': f"{record['boundary_iou']:.4f}",
                    'Background IoU': f"{record['background_iou']:.4f}",
                }

                # 添加 mIOU 指标
                for key in ['mIOU_1_8', 'mIOU_2_9', 'mIOU_3_10', 'mIOU_4_11',
                           'mIOU_5_12', 'mIOU_6_13', 'mIOU_7_14']:
                    excel_row[key] = f"{record.get(key, 0):.4f}"

                excel_records.append(excel_row)

            df_details = pd.DataFrame(excel_records)

            # 计算统计摘要（按网络分组）
            summary_data = []
            for net_name in network_order:
                net_records = df_details[df_details['Network'] == net_name]
                if len(net_records) == 0:
                    continue

                numeric_cols = ['Accuracy', 'IoU', 'Dice Score', 'Precision', 'Recall',
                               'Semantic ACC', 'Boundary IoU', 'Background IoU']

                for col in numeric_cols:
                    values = pd.to_numeric(net_records[col], errors='coerce')
                    summary_data.append({
                        'Network': net_name,
                        'Metric': col,
                        'Mean': f"{values.mean():.4f}",
                        'Std': f"{values.std():.4f}",
                        'Max': f"{values.max():.4f}",
                        'Min': f"{values.min():.4f}",
                        'Samples': len(values)
                    })

            df_summary = pd.DataFrame(summary_data)

            # 保存到 Excel
            try:
                with pd.ExcelWriter(excel_save_path, engine='openpyxl') as writer:
                    df_details.to_excel(writer, sheet_name='All Results', index=False)
                    df_summary.to_excel(writer, sheet_name='Summary by Network', index=False)

                print(f"\n{'='*70}")
                print(f"📊 评估指标已保存到 Excel")
                print(f"{'='*70}")
                print(f"保存路径：{os.path.abspath(excel_save_path)}")
                print(f"\n包含以下工作表:")
                print(f"  1. All Results - 所有样本和网络的详细指标 ({len(df_details)} 条记录)")
                print(f"  2. Summary by Network - 按网络分组的统计摘要 ({len(df_summary)} 条)")
                print(f"{'='*70}\n")

            except Exception as e:
                print(f"❌ Excel 保存失败：{e}")

def prepare_data(cfg_obj: DictConfig):
    if cfg_obj.dataset.name == "T3DSPlusSegDataset":
        dm, train_loader, test_loader \
            = T3DSPlusSegDataset.get_dataloaders_from_cfg_obj(cfg_obj)
    elif cfg_obj.dataset.name == "TADPMSegDataset":
        dm, train_loader, test_loader \
            = TADPMSegDataset.get_dataloaders_from_cfg_obj(cfg_obj)

    return dm, train_loader, test_loader


def get_modules(cfg_obj: DictConfig, name="DTSegNetModule"):
    modules_config = {
        "DTSegNetModule": {
            "ckpt_path": cfg_obj.predictors.DTSegNetModule.ckpt_cfg.full_path,
            "model_cfg": cfg_obj.predictors.DTSegNetModule.model_cfg
        },
        "PTSegNetModule": {
            "ckpt_path": cfg_obj.predictors.PTSegNetModule.ckpt_cfg.full_path,
            "model_cfg": cfg_obj.predictors.PTSegNetModule.model_cfg
        },
        "TMELFSegModule": {
            "ckpt_path": cfg_obj.predictors.PatchSegModule.ckpt_cfg.full_path,
            "model_cfg": cfg_obj.predictors.PatchSegModule.model_cfg
        },
        "PointnetppModule": {
            "ckpt_path": cfg_obj.predictors.PointnetppModule.ckpt_cfg.full_path,
            "model_cfg": cfg_obj.predictors.PointnetppModule.model_cfg
        },
        "TeethGNNModule": {
            "ckpt_path": cfg_obj.predictors.TeethGNNModule.ckpt_cfg.full_path,
            "model_cfg": cfg_obj.predictors.TeethGNNModule.model_cfg
        },
        "TSegNetModule": {
            "ckpt_path": cfg_obj.predictors.TSegNetModule.ckpt_cfg.full_path,
            "model_cfg": cfg_obj.predictors.TSegNetModule.model_cfg
        }
    }

    assert name in modules_config.keys()
    return modules_config[name]



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
            gt_labels = pred_data.y.to(device)
            pred = pred_data.outputs["seg_result"].to(device)
            pred_classes = torch.argmax(pred, dim=-1).squeeze().to(device)
            gt_edge_mask = pred_data.gt_edge_mask.to(device)
        else:
            gt_labels = pred_data.y_ori.squeeze().to(device)
            pred = pred_data.outputs["seg_result"].to(device)
            pred_classes = torch.argmax(pred, dim=-1).squeeze().to(device)
            pred_classes = pred_classes[batch.c2f_idx].squeeze().to(device)
            gt_edge_mask = pred_data.gt_edge_mask[batch.c2f_idx].squeeze().to(device)

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


def process_one():
    print("Current working directory:", os.getcwd())

    config_path = "../configs"
    config_name = "eval_segnet"
    cfg = get_yaml_config(config_path=config_path, config_name=config_name)
    args = cfg.paths
    ds_args = cfg.dataset
    print_yaml_cfg_dict(cfg)

    cfg_obj = instantiate_from_config(cfg)
    print(cfg_obj)

    _, _, test_loader =prepare_data(cfg_obj)
    print(test_loader)

    auto_register_modules()
    # DTSegNetModule
    dtseg_module_cfg = get_modules(cfg_obj, name="DTSegNetModule")
    dtseg_module = ModuleFactory.load_from_checkpoint(
        module_name="DTSegNetModule",
        checkpoint_path=dtseg_module_cfg["ckpt_path"],
        model_cfg=dtseg_module_cfg["model_cfg"],
        map_location="cuda"
    )

    # ptseg_module_cfg = get_modules(cfg_obj, name="PTSegNetModule")
    # ptseg_module = ModuleFactory.load_from_checkpoint(
    #     module_name="PTSegNetModule",
    #     checkpoint_path=ptseg_module_cfg["ckpt_path"],
    #     model_cfg=ptseg_module_cfg["model_cfg"],
    #     map_location="cuda"
    # )

    data_iter = iter(test_loader)
    next(data_iter)
    batch = next(data_iter)
    outputs = dtseg_module.predict_step_forward(batch)
    print(outputs)

    gt_labels = outputs.y.to(device)
    pred = outputs.outputs["seg_result"].to(device)  #seg_result
    pred_classes = torch.argmax(pred, dim=-1).squeeze().to(device)
    gt_edge_mask = outputs.gt_edge_mask.to(device)
    tae_lf = outputs.outputs.get('edge_seg_result').to(device)
    tae_mask = torch.argmax(tae_lf, dim=1).reshape(-1, )
    tte_lf = outputs.outputs.get('tte_seg_result').to(device)
    tte_mask = torch.argmax(tte_lf, dim=1).reshape(-1, )

    outs = cal_multi_cls_metric(gt_labels.cpu().numpy(), pred_classes.cpu().numpy(), pred_classes.cpu().numpy())
    # ACC, IOU, F1, PRECISION, RECALL, SEM_ACC, IOU_arr = outs
    acc, iou, dsc, ppv, sen,SEM_ACC, _ = outs

    iou_outs = calc_bnd_bg_merged_iou(pred_classes, gt_labels, gt_edge_mask, num_classes=15)
    boundary_iou, background_iou, merge_pairs_ious_dict = iou_outs

    print(f"acc: {acc:.4f}, iou={iou:.4f}, dsc={dsc:.4f}, ppv={ppv:.4f}, sen={sen:.4f}, SEM_ACC={SEM_ACC:.4f}")
    print(f"boundary_iou={boundary_iou:.4f}, background_iou={background_iou:.4f}")
    merge_pairs_str = ", ".join([f"{k}={v:.4f}" if isinstance(v, (int, float)) else f"{k}={v}"
                                 for k, v in merge_pairs_ious_dict.items()])
    print(f"merge_pairs_ious: {merge_pairs_str}")

    # 根据pred_classes生成边缘
    pred_edge_mask,_ = DynamicRadiusNNEdgeMask().generate_edge_masks(batch.pos.to(device),
                                                                   pred_classes)
    # pvgu.print_3d(pvgu.np_to_pcd_with_label(batch.pos.cpu().numpy(),
    #                                         pred_edge_mask.squeeze().cpu().numpy()))
    filename = batch.filename[0] if isinstance(batch.filename, list) else batch.filename

    save_path = os.path.join("tae_pred_edge", f"{filename}_tae_pred_edge.png")
    SegPlotter.compare_edge_masks_pointcloud(
        pos=batch.pos,
        pred_edge_mask=pred_edge_mask,
        tae_edge_mask=tae_mask,
        save_path= save_path
    )

def do_exp_cluster_fusion():
    auto_register_modules()

    print("Current working directory:", os.getcwd())

    cfg_obj = create_cfg_obj(config_path="../configs", config_name="eval_segnet")

    raw_data_path = cfg_obj.paths.raw_data_path
    fps_48000_path = cfg_obj.paths.fps_48000_path
    fps_16000_tmc_path = cfg_obj.paths.fps_16000_tmc_path

    fold_path = cfg_obj.dataset.folds.fold_path
    fold_filename = cfg_obj.dataset.folds.fold_filename
    fold_fullname = os.path.join(fold_path, fold_filename)
    # print(fold_fullname)

    val_pairs = gen_val_files_pair(fold_fullname, fps_16000_tmc_path, fps_48000_path, raw_data_path)
    # print(f"找到 {len(val_pairs)} 个匹配的三元组")
    # print(data_raw, data_fps_48000, data_fps_16000_tmc)

    # DTSegNetModule
    dtseg_module, patchseg_module = load_modules(cfg_obj)

    pair = val_pairs[0]
    data_raw = torch.load(pair[0])
    data_fps_48000 = torch.load(pair[1])
    data_fps_16000_tmc = torch.load(pair[2])


    batch = Batch.from_data_list([data_fps_16000_tmc])
    with torch.no_grad():
        batch_dtseg = dtseg_module.predict_step_forward(batch)
    # print(batch_dtseg, batch_dtseg.filename[0])

    tmc = create_tmc(cfg_obj)
    batch_tmc = tmc(batch)

    y_tmc = batch_tmc.tmc_proposal['outputs']['seg_result']  # [1, 16000, 15]
    y_tmc = torch.argmax(y_tmc, dim=-1).squeeze().cpu().numpy()

    #################################################################
    # 考虑一下合并seg results，提升recall，看看效果如何
    # per_pred_clu_count, pred_cents = ClusterEval.extract_cluster_by_label(batch.pos, y_tmc)
    #
    # merged_cents, match_info = ClusterEval.merge_centroids_hungarian(
    #     cluster_cents=batch_tmc.tmc_proposal['cluster_centroids'],
    #     pred_cents=pred_cents,
    #     distance_threshold=0.08
    # )
    #
    # batch_tmc.tmc_proposal['cluster_centroids'] = merged_cents

    ############################################


    batch_patch = Batch.from_data_list([data_fps_48000])
    batch_patch.tmc_proposal = batch_tmc.tmc_proposal
    with torch.no_grad():
        batch_patch = patchseg_module.predict_step_forward(batch_patch)

    # 置信度排序渲染算法
    pts_coarse = batch_tmc.pos
    pts_fine = batch_patch.pos

    final_labels, _ = conf_sorted_rendering(batch_patch, batch_tmc)  # 渲染后的点云[48000]
    y_ori = batch_patch.y_ori.squeeze().cpu().numpy()                       # ori gt



    ###################### 16000 点 ######################
    # nn映射
    y_map_ori = y_tmc[batch_tmc.c2f_idx.squeeze().cpu().numpy()]
    acc = np.sum(y_map_ori == y_ori) / np.sum(y_map_ori >-1)
    print(f"nn   acc: {acc:.4f}")

    # 双边滤波映射
    y_map_ori = bilateral_label_transfer_fast(pos_coarse=pts_coarse, pos_fine=data_raw.pos.cpu().numpy(),
                                              pred_labels=final_labels,
                                              sigma_spatial=0.3,k_neighbors=6)
    acc = np.sum(y_map_ori == y_ori) / np.sum(y_map_ori >-1)
    print(f"bilateral  acc: {acc:.4f}")

    ###################### 48000 点 ######################
    # nn映射
    y_map_ori = final_labels[batch_patch.c2f_idx.squeeze().cpu().numpy()]
    acc = np.sum(y_map_ori == y_ori) / np.sum(y_map_ori >-1)
    print(f"nn   acc: {acc:.4f}")

    # 双边滤波映射
    y_map_ori = bilateral_label_transfer_fast(pos_coarse=pts_fine, pos_fine=data_raw.pos.cpu().numpy(),
                                              pred_labels=final_labels,
                                              sigma_spatial=0.3,k_neighbors=6)
    acc = np.sum(y_map_ori == y_ori) / np.sum(y_map_ori >-1)
    print(f"bilateral  acc: {acc:.4f}")
    # 4. 渲染
    # SegPlotter.plot_pcd(pts_fine, final_labels.astype(np.int32), save_path=os.path.join('tmp', 'label_final_seg.png'))


def conf_sorted_rendering(batch_patch, batch_tmc):
    pts_coarse = batch_tmc.pos
    pts_fine = batch_patch.pos
    indexer = NNPointSamplingIndexer()
    f2c_idx_nn, c2f_idx_nn = indexer(pts_fine, pts_coarse)
    # 2. 转移实例标签
    coarse_seg_probs = batch_tmc.tmc_proposal['outputs']['seg_result']  # [1, 16000, 15]
    coarse_seg_probs = F.softmax(coarse_seg_probs, dim=-1)
    # 映射到 48k 点
    fine_seg_probs = coarse_seg_probs[:, c2f_idx_nn.squeeze(), :]  # [48000]
    fine_seg_labels = torch.argmax(fine_seg_probs, dim=-1).squeeze()
    # print(pts_fine.shape, fine_seg_probs.shape, fine_seg_labels)
    num_class = fine_seg_probs.shape[-1]
    crops = batch_patch.outputs['crops']
    bin_pred_probs = F.softmax(batch_patch.outputs['crops']['crop_pred_seg_labels'], dim=1).detach().cpu().numpy()
    max_conf, pred_labels = torch.max(fine_seg_probs, dim=-1)
    max_conf = max_conf.squeeze().cpu().numpy()
    pred_labels = pred_labels.squeeze().cpu().numpy()
    crop_pred_labels = np.zeros_like(crops['cluster_gt_seg_labels'].cpu().numpy()).astype(np.int32)
    crop_pred_conf = np.zeros_like(crops['cluster_gt_seg_labels'].cpu().numpy()).astype(np.float32)
    # # 求取单个crop的置信度
    for id in range(crops['cropped_pos'].shape[0]):
        crop_idxs = crops['cropped_nn_indexes'][0, id, :].cpu().numpy()
        crop_pts = pts_fine[crop_idxs, :].float()
        crop_conf = max_conf[crop_idxs]
        crop_labels = pred_labels[crop_idxs]

        crop_bin_pred_probs = bin_pred_probs[id, 1, :]
        crop_mask = (crop_bin_pred_probs > 0.5)
        crop_bin_conf = crop_conf[crop_mask]
        crop_bin_pred_labels = crop_labels[crop_mask]

        # 初始化结果数组
        label_conf_sums = np.zeros(num_class)
        # 使用 np.add.at 进行散点累加 (等同于 scatter_sum)
        np.add.at(label_conf_sums, crop_bin_pred_labels, crop_bin_conf)

        # 计算指标
        total_conf = label_conf_sums.sum()
        max_label_conf = label_conf_sums.max()
        purity = max_label_conf / (total_conf + 1e-6) if total_conf > 0 else 0.0

        crop_bin_label = np.argmax(label_conf_sums)

        # print(id, purity, crop_bin_label)

        crop_pred_labels[id] = crop_bin_label
        crop_pred_conf[id] = purity
    # print(crop_pred_conf, crop_pred_labels)
    sorted_indices = np.argsort(crop_pred_conf)[::-1]
    num_points = batch_patch.pos.shape[0]
    final_labels = np.zeros(num_points, dtype=np.int32)
    processed_mask = np.zeros(num_points, dtype=bool)  # 标记哪些点已经被高置信度的 crop 处理过了

    # 2. 按置信度从高到低遍历 crops
    for i in sorted_indices:
        cls_id = crop_pred_labels[i]
        indices = crops['cropped_nn_indexes'][0, i, :].cpu().numpy()

        crop_bin_pred_probs = bin_pred_probs[i, 1, :]
        crop_mask = (crop_bin_pred_probs > 0.5)

        indices = indices[crop_mask]

        # 【关键逻辑】：只处理那些还没被更高置信度 crop 覆盖的点
        # ~processed_mask[indices] 找出当前 crop 中尚未被赋值的点
        target_mask = ~processed_mask[indices]

        if target_mask.any():
            target_indices = indices[target_mask]
            final_labels[target_indices] = cls_id  # 赋予类别
            processed_mask[target_indices] = True  # 标记为已处理
    # 3. (可选) 处理剩余未覆盖的点，例如继承粗粒度标签
    uncovered = ~processed_mask
    final_labels[uncovered] = 0
    return final_labels, pts_fine


def load_modules(cfg_obj):
    start_time = time.time()
    from model._DilatedToothSegNet.dilated_tooth_seg_module import DilatedToothSegModule
    dtseg_module = DilatedToothSegModule.load_from_checkpoint(
        module_name="DTSegNetModule",
        checkpoint_path=cfg_obj.predictors.DTSegNetModule.ckpt_cfg.full_path,
        model_cfg=cfg_obj.predictors.DTSegNetModule.model_cfg,
        map_location="cuda",
        strict=False,  # 不严格匹配，跳过无用键
        weights_only=True,  # 只加载权重，不加载优化器
        _load_model_state_dict_only=True
    ).eval()
    # print(dtseg_module)
    from model._PatchSegNet.patch_seg_module import PatchSegModule
    patchseg_module = PatchSegModule.load_from_checkpoint(
        module_name="PatchSegModule",
        checkpoint_path=cfg_obj.predictors.PatchSegModule.ckpt_cfg.full_path,
        model_cfg=cfg_obj.predictors.PatchSegModule.model_cfg,
        map_location="cuda",
        strict = False,  # 不严格匹配，跳过无用键
        weights_only = True,  # 只加载权重，不加载优化器
        _load_model_state_dict_only=True
    ).eval()
    # print(patchseg_module)
    end_time = time.time()
    print(f"加载模型耗时：{end_time - start_time:.2f} 秒")

    return dtseg_module, patchseg_module


def gen_val_files_pair(fold_fullname, fps_16000_tmc_path, fps_48000_path, raw_data_path):
    """raw, fps48000,fps_16000_tmc"""
    with open(fold_fullname, 'r', encoding='utf-8') as f:
        val_filenames = f.read().splitlines()  # splitlines() 会自动去掉末尾的换行符
    print(len(val_filenames), val_filenames)
    triplets = []
    for name in val_filenames:
        # 构建三个路径下的 .pt 文件完整路径
        p1 = os.path.join(raw_data_path, f"{name}.pt")
        p2 = os.path.join(fps_48000_path, f"{name}.pt")
        p3 = os.path.join(fps_16000_tmc_path, f"{name}.pt")

        # 检查三个文件是否都存在
        if os.path.exists(p1) and os.path.exists(p2) and os.path.exists(p3):
            triplets.append((p1, p2, p3))
    # triplets 现在包含了所有匹配成功的三元组列表

    return triplets


def create_cfg_obj(config_path="/configs", config_name="eval_segnets"):
    # config_path = "../configs"
    # config_name = "eval_segnet"
    cfg = get_yaml_config(config_path=config_path, config_name=config_name)
    print_yaml_cfg_dict(cfg)
    cfg_obj = instantiate_from_config(cfg)
    print(cfg_obj)

    return cfg_obj


def create_tmc(cfg_obj):
    tmc = TopoMorphCluster(num_classes=cfg_obj.tmc_cluster.num_classes,
                           use_outputs_data=True,
                           seg_predictor=None,
                           filter_type=cfg_obj.tmc_cluster.filter_type,  # bg, tae, tae_tte, tae_tte_gtte
                           mounted_segnet_outputs=cfg_obj.tmc_cluster.mounted_segnet_outputs,
                           mounted_mask_outputs=cfg_obj.tmc_cluster.mounted_mask_outputs,
                           mounted_knn_ica_meshes=cfg_obj.tmc_cluster.mounted_knn_ica_meshes,
                           aggr_type=cfg_obj.tmc_cluster.aggr_type,  # boolean, add
                           small_cluster_size_thre=cfg_obj.tmc_cluster.small_cluster_size_thre,
                           alpha=cfg_obj.tmc_cluster.alpha,
                           k=cfg_obj.tmc_cluster.k,
                           max_iter=cfg_obj.tmc_cluster.max_iter,
                           iter_thred_len_ratio=cfg_obj.tmc_cluster.iter_thred_len_ratio,
                           field_bound_ratio=cfg_obj.tmc_cluster.field_bound_ratio,
                           field_sigma_ratio=cfg_obj.tmc_cluster.field_sigma_ratio)
    return tmc


def batch_exp_cluster_fusion():
    """
    批量执行置信度排序渲染算法 (Confidence-Sorted Rendering) 并评估
    参考 batch_evaluate_all_networks 的指标计算方式
    """
    auto_register_modules()
    print("Current working directory:", os.getcwd())

    cfg_obj = create_cfg_obj(config_path="../configs", config_name="eval_segnet")
    cluster_fusion_path = cfg_obj.paths.cluster_fusion_path

    # 1. 获取数据路径与三元组
    raw_data_path = cfg_obj.paths.raw_data_path
    fps_48000_path = cfg_obj.paths.fps_48000_path
    fps_16000_tmc_path = cfg_obj.paths.fps_16000_tmc_path
    fold_path = cfg_obj.dataset.folds.fold_path
    fold_filename = cfg_obj.dataset.folds.fold_filename
    fold_fullname = os.path.join(fold_path, fold_filename)

    val_pairs = gen_val_files_pair(fold_fullname, fps_16000_tmc_path, fps_48000_path, raw_data_path)
    print(f"✅ 找到 {len(val_pairs)} 个待评估样本")

    # 2. 加载模型
    dtseg_module, patchseg_module = load_modules(cfg_obj)
    tmc = create_tmc(cfg_obj)

    # 3. 准备存储结果
    all_results = []
    save_dir = os.path.join(cluster_fusion_path, "exp_cluster_fusion", datetime.now().strftime("%Y%m%d_%H%M%S"))
    img_save_dir = os.path.join(save_dir, "imgs")
    os.makedirs(img_save_dir, exist_ok=True)

    # 4. 批量处理
    for idx, pair in enumerate(val_pairs):
        try:
            data_raw = torch.load(pair[0])
            data_fps_48000 = torch.load(pair[1])
            data_fps_16000_tmc = torch.load(pair[2])

            filename = os.path.basename(pair[0]).replace('.pt', '')
            print(f"\n[{idx+1}/{len(val_pairs)}] 处理样本: {filename}")

            # --- 核心推理流程 ---
            batch_tmc_input = Batch.from_data_list([data_fps_16000_tmc])
            with torch.no_grad():
                _ = dtseg_module.predict_step_forward(batch_tmc_input)

            batch_tmc = tmc(batch_tmc_input)

            #################################################################
            # 考虑一下合并seg results，提升recall，看看效果如何
            seg_results = batch_tmc.outputs['seg_result']
            per_pred_clu_count, pred_cents = ClusterEval.extract_cluster_by_label(batch_tmc.pos, seg_results)

            merged_cents, match_info = ClusterEval.merge_centroids_hungarian(
                cluster_cents=batch_tmc.tmc_proposal['cluster_centroids'],
                pred_cents=pred_cents,
                distance_threshold=0.08
            )

            batch_tmc.tmc_proposal['cluster_centroids'] = torch.from_numpy(merged_cents)

            ############################################

            batch_patch = Batch.from_data_list([data_fps_48000])
            batch_patch.tmc_proposal = batch_tmc.tmc_proposal

            with torch.no_grad():
                batch_patch = patchseg_module.predict_step_forward(batch_patch)

            # --- 执行置信度排序渲染 ---
            final_labels, pts_fine = conf_sorted_rendering(batch_patch, batch_tmc)

            # --- 准备 GT 数据 ---
            # 注意：这里我们使用 data_raw (原始高分辨率) 作为评估基准
            gt_labels = data_raw.y.squeeze().cpu().numpy()
            gt_edge_mask = data_raw.gt_edge_mask.squeeze().cpu().numpy() if hasattr(data_raw, 'gt_edge_mask') else None

            # 如果 final_labels 是在 48k 点上计算的，而 GT 在原始点云上，需要映射
            # 假设 conf_sorted_rendering 返回的是对应 pts_fine (48k) 的标签
            # 我们需要将 final_labels 映射回 data_raw.pos 才能准确计算边界指标
            # 这里使用双边滤波进行高质量映射用于评估
            pred_mapped_to_raw = bilateral_label_transfer_fast(
                pos_coarse=pts_fine.cpu().numpy() if isinstance(pts_fine, torch.Tensor) else pts_fine,
                pos_fine=data_raw.pos.cpu().numpy(),
                pred_labels=final_labels,
                sigma_spatial=0.3,
                k_neighbors=6
            )

            # --- 计算指标 (参考 batch_evaluate_all_networks) ---
            pred_tensor = torch.LongTensor(pred_mapped_to_raw)
            gt_tensor = torch.LongTensor(gt_labels)

            # 1. 基础分割指标
            outs = cal_multi_cls_metric(gt_labels, pred_mapped_to_raw, pred_mapped_to_raw)
            acc, iou, dsc, ppv, sen, sem_acc, _ = outs

            # 2. 边界与背景指标
            boundary_iou = 0.0
            background_iou = 0.0
            merge_pairs_dict = {}

            if gt_edge_mask is not None and len(gt_edge_mask) == len(pred_mapped_to_raw):
                try:
                    iou_outs = calc_bnd_bg_merged_iou(
                        pred_tensor,
                        gt_tensor,
                        torch.LongTensor(gt_edge_mask),
                        num_classes=15
                    )
                    boundary_iou, background_iou, merge_pairs_dict = iou_outs
                except Exception as e:
                    print(f"   ⚠️ 边界指标计算失败: {e}")

            result = {
                'filename': filename,
                'acc': acc,
                'iou': iou,
                'dsc': dsc,
                'ppv': ppv,
                'sen': sen,
                'SEM_ACC': sem_acc,
                'boundary_iou': boundary_iou,
                'background_iou': background_iou,
                'merge_pairs_ious_dict': merge_pairs_dict
            }
            all_results.append(result)

            print(f"   📊 Acc: {acc:.4f}, IoU: {iou:.4f}, B-IoU: {boundary_iou:.4f}")

            # --- 可视化保存 ---
            save_path = os.path.join(img_save_dir, f"{filename}_pred.png")
            SegPlotter.plot_pcd(
                pos=data_raw.pos,
                labels=pred_mapped_to_raw.astype(np.int32),
                save_path=save_path,
                window_size=[1920, 1080],
                point_size=2.0
            )

        except Exception as e:
            print(f"   ❌ 处理失败: {e}")
            import traceback
            traceback.print_exc()

    # 5. 保存统计结果到 Excel
    if all_results:
        excel_path = os.path.join(save_dir, "cluster_fusion_metrics.xlsx")
        save_evaluation_results_to_excel(all_results, save_path=excel_path)
        print(f"\n🎉 批量评估完成！结果已保存至: {save_dir}")




def batch_process():
    print("Current working directory:", os.getcwd())

    config_path = "../configs"
    config_name = "eval_segnet"
    cfg = get_yaml_config(config_path=config_path, config_name=config_name)
    args = cfg.paths
    ds_args = cfg.dataset
    print_yaml_cfg_dict(cfg)

    cfg_obj = instantiate_from_config(cfg)
    print(cfg_obj)

    _, _, test_loader =prepare_data(cfg_obj)
    print(test_loader)

    auto_register_modules()
    # DTSegNetModule
    # name = "DTSegNetModule"
    name = "PTSegNetModule"
    # name = "TMELFSegModule"
    module_cfg = get_modules(cfg_obj, name=name)
    module = ModuleFactory.load_from_checkpoint(
        module_name=name,
        checkpoint_path=module_cfg["ckpt_path"],
        model_cfg=module_cfg["model_cfg"],
        map_location="cuda"
    )

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_save_path = Path(module_cfg["ckpt_path"]).parent
    excel_save_path = base_save_path / f"eval_results_{timestamp}.xlsx"

    # JSON 保存目录
    json_save_dir = base_save_path / f"batch_json_{timestamp}"
    print(f"\n📂 JSON 保存目录：{json_save_dir}")


    # 存储所有样本的结果
    all_results = []

    # 遍历测试集
    for batch_idx, batch in enumerate(test_loader):
        print(f"评估样本 {batch_idx + 1}/{len(test_loader)}: {batch.filename}")

        outputs = module.predict_step_forward(batch)

        gt_labels = outputs.y.to(device)
        pred = outputs.outputs["seg_result"].to(device)
        pred_classes = torch.argmax(pred, dim=-1).squeeze().to(device)
        gt_edge_mask = outputs.gt_edge_mask.to(device)

        outs = cal_multi_cls_metric(gt_labels.cpu().numpy(), pred_classes.cpu().numpy(), pred_classes.cpu().numpy())
        acc, iou, dsc, ppv, sen, SEM_ACC, _ = outs

        iou_outs = calc_bnd_bg_merged_iou(pred_classes, gt_labels, gt_edge_mask, num_classes=15)
        boundary_iou, background_iou, merge_pairs_ious_dict = iou_outs

        # 打印当前样本结果
        print(f"acc: {acc:.4f}, iou={iou:.4f}, dsc={dsc:.4f}, ppv={ppv:.4f}, sen={sen:.4f}, SEM_ACC={SEM_ACC:.4f}")
        print(f"boundary_iou={boundary_iou:.4f}, background_iou={background_iou:.4f}")
        merge_pairs_str = ", ".join([f"{k}={v:.4f}" if isinstance(v, (int, float)) else f"{k}={v}"
                                     for k, v in merge_pairs_ious_dict.items()])
        print(f"merge_pairs_ious: {merge_pairs_str}")

        # 保存 batch 信息到 JSON
        try:
            json_path = save_batch_info_to_json(batch, outputs, json_save_dir)
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
            'merge_pairs_ious_dict': merge_pairs_ious_dict
        }
        all_results.append(result)

    # 保存所有结果到 Excel
    save_evaluation_results_to_excel(
        all_results=all_results,
        save_path=excel_save_path
    )


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_module_loading():
    """
    测试所有分割模块的加载是否成功

    Returns:
        dict: 包含每个模块加载状态的字典
    """
    print("\n" + "="*70)
    print("🧪 开始测试模块加载")
    print("="*70)

    config_path = "../configs"
    config_name = "eval_segnet"

    try:
        cfg = get_yaml_config(config_path=config_path, config_name=config_name)
        cfg_obj = instantiate_from_config(cfg)
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        return {}

    # 定义要测试的模块列表
    module_names = [
        "DTSegNetModule",
        "PTSegNetModule",
        "TMELFSegModule",
        "PointnetppModule",
        "TeethGNNModule",
        "TSegNetModule"
    ]

    results = {}

    for module_name in module_names:
        print(f"\n{'-'*70}")
        print(f"📦 测试模块: {module_name}")
        print(f"{'-'*70}")

        try:
            # 1. 获取配置
            module_cfg = get_modules(cfg_obj, name=module_name)
            ckpt_path = module_cfg["ckpt_path"]
            model_cfg = module_cfg["model_cfg"]

            print(f"   ✓ 配置获取成功")
            print(f"   📁 检查点路径: {ckpt_path}")

            # 2. 检查检查点文件是否存在
            if not os.path.exists(ckpt_path):
                print(f"   ⚠️  检查点文件不存在: {ckpt_path}")
                results[module_name] = {
                    "status": "FAILED",
                    "error": f"Checkpoint file not found: {ckpt_path}",
                    "ckpt_path": ckpt_path
                }
                continue

            print(f"   ✓ 检查点文件存在")

            # 3. 尝试加载模块
            auto_register_modules()
            module = ModuleFactory.load_from_checkpoint(
                module_name=module_name,
                checkpoint_path=ckpt_path,
                model_cfg=model_cfg,
                map_location="cpu"  # 使用 CPU 避免 GPU 内存问题
            )

            if module is None:
                print(f"   ❌ 模块加载返回 None")
                results[module_name] = {
                    "status": "FAILED",
                    "error": "Module loading returned None",
                    "ckpt_path": ckpt_path
                }
                continue

            print(f"   ✓ 模块加载成功")

            # 4. 验证模块基本属性
            if hasattr(module, 'model'):
                print(f"   ✓ 模块包含 model 属性")

                # 统计模型参数量
                total_params = sum(p.numel() for p in module.model.parameters())
                trainable_params = sum(p.numel() for p in module.model.parameters() if p.requires_grad)
                print(f"   📊 总参数量: {total_params:,}")
                print(f"   📊 可训练参数: {trainable_params:,}")

            if hasattr(module, 'predict_step_forward'):
                print(f"   ✓ 模块包含 predict_step_forward 方法")

            results[module_name] = {
                "status": "SUCCESS",
                "ckpt_path": ckpt_path,
                "total_params": total_params if 'total_params' in locals() else 0,
                "trainable_params": trainable_params if 'trainable_params' in locals() else 0
            }

            print(f"   ✅ {module_name} 加载测试通过")

        except Exception as e:
            print(f"   ❌ 加载失败: {e}")
            import traceback
            traceback.print_exc()

            results[module_name] = {
                "status": "FAILED",
                "error": str(e),
                "ckpt_path": ckpt_path if 'ckpt_path' in locals() else "N/A"
            }

    # 打印总结
    print(f"\n{'='*70}")
    print("📊 测试总结")
    print(f"{'='*70}")

    success_count = sum(1 for r in results.values() if r["status"] == "SUCCESS")
    failed_count = sum(1 for r in results.values() if r["status"] == "FAILED")

    print(f"✅ 成功: {success_count}/{len(module_names)}")
    print(f"❌ 失败: {failed_count}/{len(module_names)}")

    print(f"\n详细结果:")
    for module_name, result in results.items():
        status_icon = "✅" if result["status"] == "SUCCESS" else "❌"
        print(f"  {status_icon} {module_name}: {result['status']}")

        if result["status"] == "SUCCESS":
            print(f"      参数量: {result.get('total_params', 0):,}")
        else:
            print(f"      错误: {result.get('error', 'Unknown')}")

    print(f"{'='*70}\n")

    return results


def test_module_inference():
    """
    测试模块的推理功能（需要数据加载器）

    Returns:
        dict: 包含每个模块推理测试结果的字典
    """
    print("\n" + "="*70)
    print("🧪 开始测试模块推理功能")
    print("="*70)

    config_path = "../configs"
    config_name = "eval_segnet"

    try:
        cfg = get_yaml_config(config_path=config_path, config_name=config_name)
        cfg_obj = instantiate_from_config(cfg)
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        return {}

    # 准备数据
    try:
        _, _, test_loader = prepare_data(cfg_obj)
        batch = next(iter(test_loader))
        print(f"✓ 数据加载成功，batch size: {batch.pos.shape[0]}")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return {}

    # 定义要测试的模块列表
    module_names = [
        # "DTSegNetModule",
        # "PTSegNetModule",
        "TMELFSegModule",
        # "PointnetppModule",
        # "TeethGNNModule",
        # "TSegNetModule"
    ]

    results = {}

    for module_name in module_names:
        print(f"\n{'-'*70}")
        print(f"🔍 测试推理: {module_name}")
        print(f"{'-'*70}")

        try:
            # 加载模块
            module_cfg = get_modules(cfg_obj, name=module_name)
            auto_register_modules()
            module = ModuleFactory.load_from_checkpoint(
                module_name=module_name,
                checkpoint_path=module_cfg["ckpt_path"],
                model_cfg=module_cfg["model_cfg"],
                map_location="cuda" if torch.cuda.is_available() else "cpu"
            )

            if module is None:
                print(f"   ❌ 模块加载失败")
                results[module_name] = {"status": "FAILED", "error": "Module loading failed"}
                continue

            if module_name == "TMELFSegModule":
                tmc = TopoMorphCluster(
                    num_classes=15,
                    predictor_type='DTSegNet',
                    ckpt_path='../common/ckpts/DTSegnet.ckpt',
                    mounted_mask_outputs=False,
                    mounted_knn_ica_meshes=False,
                    mounted_segnet_outputs=True,
                    filter_type='tae_tte_gtte',
                    aggr_type='add'
                )
                # 对 batch 应用 TMC
                if hasattr(batch, 'num_graphs') and batch.num_graphs > 1:
                    test_data = batch[0] if hasattr(batch, '__getitem__') else batch
                else:
                    test_data = batch

                batch = tmc(test_data)

            # 执行推理
            with torch.no_grad():
                outputs = module.predict_step_forward(batch)

            # 验证输出
            if not hasattr(outputs, 'outputs'):
                print(f"   ❌ 输出缺少 outputs 属性")
                results[module_name] = {"status": "FAILED", "error": "Missing outputs attribute"}
                continue

            if 'seg_result' not in outputs.outputs:
                print(f"   ❌ 输出缺少 seg_result")
                results[module_name] = {"status": "FAILED", "error": "Missing seg_result"}
                continue

            seg_result = outputs.outputs['seg_result']
            pred_classes = torch.argmax(seg_result, dim=-1)

            print(f"   ✓ 推理成功")
            print(f"   📊 输入点数: {batch.pos.shape[0]}")
            print(f"   📊 输出形状: {seg_result.shape}")
            print(f"   📊 预测类别数: {len(torch.unique(pred_classes))}")

            num_unique_classes = len(torch.unique(pred_classes))
            print(f"   📊 预测类别数: {num_unique_classes}")

            results[module_name] = {
                "status": "SUCCESS",
                "output_shape": list(seg_result.shape),
                "num_classes": int(num_unique_classes)
            }

            print(f"   ✅ {module_name} 推理测试通过")

        except Exception as e:
            print(f"   ❌ 推理失败: {e}")
            import traceback
            traceback.print_exc()

            results[module_name] = {
                "status": "FAILED",
                "error": str(e)
            }

    # 打印总结
    print(f"\n{'='*70}")
    print("📊 推理测试总结")
    print(f"{'='*70}")

    success_count = sum(1 for r in results.values() if r["status"] == "SUCCESS")
    failed_count = sum(1 for r in results.values() if r["status"] == "FAILED")

    print(f"✅ 成功: {success_count}/{len(module_names)}")
    print(f"❌ 失败: {failed_count}/{len(module_names)}")

    for module_name, result in results.items():
        status_icon = "✅" if result["status"] == "SUCCESS" else "❌"
        print(f"  {status_icon} {module_name}: {result['status']}")

    print(f"{'='*70}\n")

    return results


def test_modules():
    # loading_results = test_module_loading()
    # failed_modules = [name for name, result in loading_results.items()
    #                   if result["status"] == "FAILED"]
    # 如果需要测试推理功能（需要数据和GPU）
    inference_results = test_module_inference()
    # 检查是否有失败的模块
    failed_modules = [name for name, result in inference_results.items()
                      if result["status"] == "FAILED"]
    if failed_modules:
        print(f"\n⚠️  以下模块加载失败，请检查配置和检查点路径:")
        for name in failed_modules:
            print(f"   - {name}")
    else:
        print("\n🎉 所有模块加载成功！")

def evaluate_and_save_segnet_predictions(test_loader, cfg_obj, save_dir=None):
    """
    评估各个分割网络并将预测结果保存到 batch 中

    Args:
        test_loader: 测试数据加载器
        cfg_obj: 配置对象
        save_dir: 保存目录，如果为 None 则使用 cfg_obj.paths.segnet_pred_path

    Returns:
        dict: 包含所有样本预测结果的字典
    """
    from evaluation.ModuleFactory import ModuleFactory, auto_register_modules
    from datasets.transform.TopoMorphCluster import TopoMorphCluster

    print("\n" + "="*70)
    print("🔬 开始多网络实例分割评估")
    print("="*70)

    # 确定保存路径
    if save_dir is None:
        if hasattr(cfg_obj, 'paths') and hasattr(cfg_obj.paths, 'segnet_pred_path'):
            save_dir = cfg_obj.paths.segnet_pred_path
        else:
            save_dir = "./segnet_predictions"

    os.makedirs(save_dir, exist_ok=True)
    print(f"📁 保存路径: {save_dir}")

    # 定义要评估的网络列表
    network_names = [
        "PointnetppModule",
        "TeethGNNModule",
        "TSegNetModule",
        "PTSegNetModule",
        "DTSegNetModule",
        "TMELFSegModule"  # 添加 TMELFSegModule
    ]

    # 预加载所有模块
    print("\n🔄 加载模型...")
    modules = {}
    module_configs = {}

    auto_register_modules()

    for net_name in network_names:
        try:
            module_cfg = get_modules(cfg_obj, name=net_name)
            module = ModuleFactory.load_from_checkpoint(
                module_name=net_name,
                checkpoint_path=module_cfg["ckpt_path"],
                model_cfg=module_cfg["model_cfg"],
                map_location="cuda" if torch.cuda.is_available() else "cpu"
            )
            modules[net_name] = module
            module_configs[net_name] = module_cfg
            print(f"   ✅ {net_name} 加载成功")
        except Exception as e:
            print(f"   ❌ {net_name} 加载失败: {e}")
            modules[net_name] = None

    # 初始化 TMC（用于 TMELFSegModule）
    tmc = None
    try:
        print("\n🔄 初始化 TMC (用于 TMELFSegModule)...")
        tmc = TopoMorphCluster(
            num_classes=15,
            predictor_type='DTSegNet',
            ckpt_path='../common/ckpts/DTSegnet.ckpt',
            mounted_mask_outputs=False,
            mounted_knn_ica_meshes=False,
            mounted_segnet_outputs=True,
            filter_type='tae_tte_gtte',
            aggr_type='add'
        )
        print("   ✅ TMC 初始化成功")
    except Exception as e:
        print(f"   ⚠️ TMC 初始化失败: {e}")
        print("   ⚠️ TMELFSegModule 将无法使用")

    # 遍历测试集
    all_results = {}

    for batch_idx, batch in enumerate(test_loader):
        print(f"\n处理样本 {batch_idx + 1}/{len(test_loader)}: {batch.filename}")

        filename = batch.filename[0] if isinstance(batch.filename, list) else batch.filename

        # 初始化预测结果字典
        net_pred_results = {}

        # 对每个网络进行推理
        for net_name, module in modules.items():
            if module is None:
                print(f"   ⚠️ 跳过 {net_name} (未成功加载)")
                continue

            try:
                print(f"   🔄 推理: {net_name}")

                # 对于 TMELFSegModule，需要先应用 TMC
                test_batch = batch
                if net_name == "TMELFSegModule":
                    if tmc is None:
                        print(f"      ⚠️ 跳过 TMELFSegModule (TMC 未初始化)")
                        continue

                    try:
                        print(f"      🔄 应用 TMC 变换...")
                        # 处理 batch 格式
                        if hasattr(test_batch, 'num_graphs') and test_batch.num_graphs > 1:
                            single_data = test_batch[0] if hasattr(test_batch, '__getitem__') else test_batch
                        else:
                            single_data = test_batch

                        test_batch = tmc(single_data)
                        print(f"      ✓ TMC 变换完成")

                        # 验证 tmc_proposal 是否存在
                        if not hasattr(test_batch, 'tmc_proposal'):
                            print(f"      ⚠️ 警告: tmc_proposal 属性不存在，跳过")
                            continue

                    except Exception as e:
                        print(f"      ❌ TMC 变换失败: {e}")
                        continue

                # 执行推理
                with torch.no_grad():
                    outputs = module.predict_step_forward(test_batch)

                # 提取预测类别
                if hasattr(outputs, 'outputs') and 'seg_result' in outputs.outputs:
                    seg_result = outputs.outputs['seg_result']
                    pred_classes = torch.argmax(seg_result, dim=-1).squeeze().cpu().numpy()

                    # 简化网络名称作为键
                    simple_name = net_name.replace('Module', '').lower()
                    net_pred_results[simple_name] = pred_classes

                    unique_classes = np.unique(pred_classes[pred_classes >= 0])
                    print(f"      ✓ {simple_name}: {len(unique_classes)} 个类别, {len(pred_classes)} 点")
                else:
                    print(f"      ❌ {net_name}: 输出格式错误")

            except Exception as e:
                print(f"      ❌ {net_name} 推理失败: {e}")
                import traceback
                traceback.print_exc()

        # 将预测结果添加到 batch
        batch.net_pred_results = net_pred_results

        # 保存为 .pt 文件
        save_path = os.path.join(save_dir, f"{filename}.pt")

        # # 构建保存字典
        # # 构建保存字典 - 包含所有可用字段
        # save_data = {
        #     'filename': filename,
        #     'net_pred_results': net_pred_results,
        #
        #     # 几何信息
        #     'pos': batch.pos.cpu().numpy() if hasattr(batch, 'pos') else None,
        #     'face': batch.face.cpu().numpy() if hasattr(batch, 'face') else None,
        #     'normals': batch.normals.cpu().numpy() if hasattr(batch, 'normals') else None,
        #     'x': batch.x.cpu().numpy() if hasattr(batch, 'x') else None,
        #
        #     # 标签信息
        #     'y': batch.y.cpu().numpy() if hasattr(batch, 'y') else None,
        #     'y_ori': batch.y_ori.cpu().numpy() if hasattr(batch, 'y_ori') else None,
        #     'gt_edge_mask': batch.gt_edge_mask.cpu().numpy() if hasattr(batch, 'gt_edge_mask') else None,
        #     'gt_tte_mask': batch.gt_tte_mask.cpu().numpy() if hasattr(batch, 'gt_tte_mask') else None,
        #
        #     # 实例信息
        #     'instances': batch.instances.cpu().numpy() if hasattr(batch, 'instances') else None,
        #     'instance_centroids': batch.instance_centroids.cpu().numpy() if hasattr(batch,
        #                                                                             'instance_centroids') else None,
        #     'instance_labels': batch.instance_labels.cpu().numpy() if hasattr(batch, 'instance_labels') else None,
        #     'per_instance_count': batch.per_instance_count.cpu().numpy() if hasattr(batch,
        #                                                                             'per_instance_count') else None,
        #     'instance_count': batch.instance_count.cpu().numpy() if hasattr(batch, 'instance_count') else None,
        #
        #     # 索引映射
        #     'f2c_idx': batch.f2c_idx.cpu().numpy() if hasattr(batch, 'f2c_idx') else None,
        #     'c2f_idx': batch.c2f_idx.cpu().numpy() if hasattr(batch, 'c2f_idx') else None,
        #
        #     # 其他属性
        #     'jaw': batch.jaw.cpu().numpy() if hasattr(batch, 'jaw') else None,
        #     'batch': batch.batch.cpu().numpy() if hasattr(batch, 'batch') else None,
        #
        #     # 时间戳
        #     'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # }

        torch.save(batch, save_path)
        print(f"   💾 已保存: {save_path}")

        # 存储到总结果
        all_results[filename] = batch

    print(f"\n{'='*70}")
    print(f"✅ 完成! 共处理 {len(all_results)} 个样本")
    print(f"📁 保存位置: {save_dir}")
    print(f"{'='*70}\n")

    return all_results


def batch_evaluate_all_networks():
    """
    批量评估所有网络并保存预测结果
    """
    print("Current working directory:", os.getcwd())

    config_path = "../configs"
    config_name = "eval_segnet"
    cfg = get_yaml_config(config_path=config_path, config_name=config_name)
    print_yaml_cfg_dict(cfg)

    cfg_obj = instantiate_from_config(cfg)

    # 准备数据
    _, _, test_loader = prepare_data(cfg_obj)
    print(f"测试集大小: {len(test_loader)}")

    # 执行评估并保存
    results = evaluate_and_save_segnet_predictions(
        test_loader=test_loader,
        cfg_obj=cfg_obj
    )

    # 打印统计信息
    print("\n📊 统计信息:")
    for filename, batch in results.items():
        pred_results = batch.net_pred_results
        print(f"  {filename}:")
        for net_name, pred_classes in pred_results.items():
            unique_classes = np.unique(pred_classes[pred_classes >= 0])
            print(f"    - {net_name}: {len(unique_classes)} 类, {len(pred_classes)} 点")


def load_segnet_predictions(load_dir=None, max_samples=10):
    """
    从文件夹加载所有网络的预测结果

    Args:
        load_dir: 加载目录

    Returns:
        dict: {filename: batch_object}
    """
    if load_dir is None or not os.path.exists(load_dir):
        print(f"❌ 目录不存在: {load_dir}")
        return {}

    pt_files = sorted(glob.glob(os.path.join(load_dir, "*.pt")))
    print(f"📂 加载 {len(pt_files)} 个文件 from {load_dir}")

    all_data = {}
    for i, pt_file in enumerate(pt_files):
        try:
            filename = os.path.basename(pt_file).replace('.pt', '')
            batch = torch.load(pt_file, map_location='cpu')
            all_data[filename] = batch

            print(f"   {i}/{len(pt_files)}: ✓ {filename}: {len(batch.pos)} 个点")
        except Exception as e:
            print(f"⚠️ 加载失败 {os.path.basename(pt_file)}: {e}")

        if i >= max_samples:
            break

    print(f"✅ 成功加载 {len(all_data)} 个样本\n")
    return all_data


def load_raw_data(load_dir=None, max_samples=10):
    """
    加载原始测试数据

    Args:
        load_dir: 数据目录，如果为 None 则从配置获取
        dataset_name: 数据集名称

    Returns:
        dict: {filename: batch_object}
    """
    if load_dir is None or not os.path.exists(load_dir):
        print(f"❌ 目录不存在: {load_dir}")
        return {}

    pt_files = sorted(glob.glob(os.path.join(load_dir, "*.pt")))

    # 过滤掉不需要的文件
    exclude_files = {'pre_filter.pt', 'pre_transform.pt'}
    pt_files = [f for f in pt_files if os.path.basename(f) not in exclude_files]

    print(f"📂 加载 {len(pt_files)} 个文件 from {load_dir}")

    all_data = {}
    for i, pt_file in enumerate(pt_files):
        try:
            filename = os.path.basename(pt_file).replace('.pt', '')
            batch = torch.load(pt_file, map_location='cpu')
            all_data[filename] = batch
            print(f"   {i}/{len(pt_files)}: ✓ {filename}: {len(batch.pos)} 个点")
        except Exception as e:
            print(f"⚠️ 加载失败 {os.path.basename(pt_file)}: {e}")

        if i>=max_samples:
            break

    print(f"✅ 成功加载 {len(all_data)} 个样本\n")
    return all_data


def merge_predictions_with_raw_data(pred_dir=None, raw_data_dir=None, max_samples=10):
    """
    合并预测结果与原始数据

    Args:
        pred_dir: 预测结果目录
        raw_data_dir: 原始数据目录，None则从数据集加载
        dataset_name: 数据集名称

    Returns:
        dict: {filename: merged_batch}
    """
    print("\n" + "=" * 70)
    print("🔄 合并预测结果与原始数据")
    print("=" * 70)

    # 1. 加载预测结果（复用现有函数）
    pred_data = load_segnet_predictions(load_dir=pred_dir, max_samples=max_samples)
    if not pred_data:
        return {}

    # 2. 加载原始数据（复用现有函数）
    raw_data = load_raw_data(load_dir=raw_data_dir, max_samples=max_samples)
    if not raw_data:
        return {}

    # 3. 按文件名匹配并合并
    print(f"\n🔗 开始合并...")
    merged_data = {}
    unmatched_pred = set(pred_data.keys()) - set(raw_data.keys())
    unmatched_raw = set(raw_data.keys()) - set(pred_data.keys())
    matched_files = set(pred_data.keys()) & set(raw_data.keys())

    for filename in matched_files:
        pred_batch = pred_data[filename]
        raw_batch = raw_data[filename]

        # 基于原始batch创建合并batch
        merged_batch = raw_batch

        # 重命名原始字段为 _ori 后缀
        if hasattr(raw_batch, 'pos'):
            merged_batch.pos_ori = raw_batch.pos

        if hasattr(raw_batch, 'face'):
            merged_batch.face_ori = raw_batch.face

        if hasattr(raw_batch, 'y'):
            merged_batch.y_ori = raw_batch.y

        # 添加预测结果
        if hasattr(pred_batch, 'net_pred_results'):
            merged_batch.net_pred_results = pred_batch.net_pred_results

        if hasattr(pred_batch, 'pos'):
            merged_batch.pos = pred_batch.pos

        if hasattr(pred_batch, 'y'):
            merged_batch.y = pred_batch.y

        if hasattr(pred_batch, 'c2f_idx'):
            merged_batch.c2f_idx = pred_batch.c2f_idx.squeeze()

        merged_data[filename] = merged_batch

    # 打印统计
    print(f"\n{'=' * 70}")
    print(f"📊 合并统计")
    print(f"{'=' * 70}")
    print(f"✅ 成功匹配: {len(matched_files)} 个样本")
    print(f"⚠️ 仅有预测: {len(unmatched_pred)} 个")
    print(f"⚠️ 仅有原始: {len(unmatched_raw)} 个")

    if merged_data:
        sample = list(merged_data.values())[0]
        print(f"\n🔍 示例字段:")
        print(f"   网络数: {len(sample.net_pred_results) if hasattr(sample, 'net_pred_results') else 0}")
        print(f"   有 pos_ori: {hasattr(sample, 'pos_ori')}")
        print(f"   有 y_ori: {hasattr(sample, 'y_ori')}")

    print(f"{'=' * 70}\n")
    return merged_data

def test_merge_predictions():
    cfg = get_yaml_config(config_path="../configs", config_name="eval_segnet")
    cfg_obj = instantiate_from_config(cfg)
    segnet_pred_path = cfg_obj.paths.segnet_pred_path
    raw_data_path = cfg_obj.paths.raw_data_path
    merged_data = merge_predictions_with_raw_data(pred_dir=segnet_pred_path,
                                                  raw_data_dir=raw_data_path,
                                                  max_samples=600)
    if merged_data:
        print(f"📋 样本列表: {list(merged_data.keys())[:5]}...")
        save_dir = cfg_obj.paths.seg_results_path + "_" + datetime.now().strftime("%Y%m%d%H%M%S")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            print(f"\n📁 保存路径: {save_dir}")
        # 可视化分割结果
        custom_order = ['pointnetpp', 'teethgnn', 'tsegnet', 'ptsegnet', 'dtsegnet', 'tmelfseg']
        SegPlotter.plot_segmentation_results(merged_data, save_dir=save_dir,
                                             cmap_name="label15", network_order=custom_order)
    # print(merged_data.keys())



def main():

    # do_exp_cluster_fusion()
    batch_exp_cluster_fusion()
    # process_one()
    # batch_process()
    # test_modules()
    # 批量评估所有网络并保存预测结果
    # batch_evaluate_all_networks()

    # all_results = load_segnet_predictions()
    # all_results = load_raw_data()
    # print(all_results)

    # test_merge_predictions()




if __name__ == '__main__':
    main()