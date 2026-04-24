import os
from collections import OrderedDict
from datetime import datetime

import pandas as pd
import torch
from pytorch_lightning.loggers import CSVLogger
from scipy.optimize import linear_sum_assignment
from scipy.spatial import Delaunay
from torch import cdist

from datasets import T3DSPlusSegDataset, TADPMSegDataset

from omegaconf import OmegaConf, DictConfig
from configs.hydra_yaml_config import get_yaml_config, print_yaml_cfg_dict, instantiate_from_config
from datasets.T3DSPlusSegDataset import T3DSPlusSegReader
from evaluation.seg_evaluation import cal_edge_iou, cal_binary_metric_label
# from external_libs import gen_utils as gu
# from external_libs import ops_utils as ou
from datasets.transform.t3dsPlus_transforms import *
from datasets.transform.PointSamplingIndexer import *
from external_libs.mesh_vis import TMCVisPV, PVMeshFactory
from model._DilatedToothSegNet.topo_clustering import (get_filtered_faces, setdiff, interp_contour,
                                                       update_interp_points, label_contour_segment,
                                                       point_to_segment_distance, check_n3)

from common.interfaces import generate_exp_name

from model2._elfnet.ELFNetSeg import PointTransformerNet
from datasets.transform.TopoMorphCluster import *
from external_libs import pv_gen_utils as pvgu


class ClusterEval:
    """ 聚类指标评估 """
    @staticmethod
    def compute_chamfer_distance(gt_cent, pred_cent):
        """计算两个点集之间的Chamfer距离
        gt_cent: tuple of torch.Tensor, 形状为 (N, 3)
        pred_cent: tuple of torch.Tensor, 形状为 (M, 3)
        返回： float, Chamfer距离
        """
        assert gt_cent is not None and pred_cent is not None
        assert (gt_cent.shape[1] == pred_cent.shape[1] == 3) and (gt_cent.shape[1] in [2, 3]), "输入点集必须为3维或2维"

        D = torch.cdist(gt_cent, pred_cent, p=2)
        D_sq = D ** 2

        # 从gt_cent到pred_cent的最小距离
        dist1 = torch.mean(torch.min(D_sq, dim=1)[0])
        # 从pred_cent到gt_cent的最小距离
        dist2 = torch.mean(torch.min(D_sq, dim=0)[0])
        # print("pred_cents to gt_cents:",torch.min(D, dim=0)[0])
        # Chamfer距离定义为双向最小距离的和
        return dist1 + dist2

    @staticmethod
    def compute_minmax_distance(gt_cent, pred_cent):
        """计算预测点集到真实点集的最小距离中最大的距离，即min-max距离；
        主要用以阈值的设定
        gt_cent: tuple of torch.Tensor, 形状为 (N, 3)
        pred_cent: tuple of torch.Tensor, 形状为 (M, 3)
        返回： float, min-max距离
        """
        assert gt_cent is not None and pred_cent is not None
        assert (gt_cent.shape[1] == pred_cent.shape[1] == 3) and (gt_cent.shape[1] in [2, 3]), "输入点集必须为3维或2维"

        D = torch.cdist(gt_cent, pred_cent, p=2)
        p2g_dist = torch.max(torch.min(D, dim=0)[0])
        g2p_dist = torch.max(torch.min(D, dim=1)[0])
        # print("pred_cents to gt_cents:", p2g_dist)
        # print("gt_cents to pred_cents:", g2p_dist)
        # Chamfer距离定义为双向最小距离的和
        return p2g_dist, g2p_dist

    @staticmethod
    def compute_detection_metrics(gt_cent, pred_cent, tooth_sizes, thre_dist=0.10):
        """
        评价检测器性能：计算Precision、Recall、F1-score
        gt_cent, pred_cent: tensor；它们的第一个元素为背景点，不参与计算。
        使用匈牙利匹配算法计算TP、FP、FN，从而计算Precision、Recall、F1-score
        参数:
            pred_cent (torch.Tensor): 检测目标坐标，形状 (N, D)，N=检测数
            gt_cent (torch.Tensor): 真实目标坐标，形状 (M, D)，M=真实数（这里M=10）
            tooth_sizes (torch.Tensor): 每个实例的尺寸，形状 (M-1,)，剔除掉背景
            threshold (float): 距离阈值，小于该值视为有效匹配
        返回:
            metrics (dict): 包含precision、recall、f1、TP、FP、FN的评价结果
        """
        assert gt_cent is not None and pred_cent is not None
        assert (gt_cent.shape[1] == pred_cent.shape[1] == 3) and (gt_cent.shape[1] in [2, 3]), "输入点集必须为3维或2维"

        # 强制转换为 Float32
        if gt_cent.dtype != torch.float32:
            gt_cent = gt_cent.float()
        if pred_cent.dtype != torch.float32:
            pred_cent = pred_cent.float()


        # 检查是否为空张量
        if gt_cent.shape[0] == 1 or pred_cent.shape[0] == 1:
            print("⚠️ Warning: Empty tensor detected in compute_chamfer_distance")
            print(f"  gt_cent shape: {gt_cent.shape}, pred_cent shape: {pred_cent.shape}")
            return {
            "cd": 100.0,
            "p2g_dist": 100.0,  # minimum distance from predicted centroids to gt centroids
            "g2p_dist": 100.0,  # minimum distance from gt centroids to predicted centroids
            "TP": 0,
            "FP": 14,
            "FN": 14,
            "precision": round(0.0, 4),
            "recall": round(0.0, 4),
            "f1": round(0.0, 4)
        }

        # 1. 计算距离矩阵 (N, M)，背景点不参与计算
        dist_matrix = torch.cdist(pred_cent[1:, :], gt_cent[1:, :], p=2)

        # 2. 生成代价矩阵（仅保留＜阈值的匹配，否则设为极大值）
        cost_matrix = dist_matrix.clone()
        cost_matrix[cost_matrix >= thre_dist] = 1e9  # 超过阈值的匹配视为不可行

        # 3. 匈牙利算法做一对一匹配
        pred_idx, gt_idx = linear_sum_assignment(cost_matrix.cpu().numpy())

        # 4. 统计TP（仅保留距离＜阈值的匹配）
        TP = 0
        for p_idx, g_idx in zip(pred_idx, gt_idx):
            if dist_matrix[p_idx, g_idx] < tooth_sizes[g_idx] * 0.5:
                TP += 1

        # 5. 计算FP、FN
        N = pred_cent.shape[0] - 1  # 检测总数
        M = gt_cent.shape[0] - 1  # 真实总数
        FP = N - TP
        FN = M - TP

        # 6. 计算评价指标（避免分母为0）
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # 距离相关的指标
        cd = ClusterEval.compute_chamfer_distance(gt_cent[1:, :], pred_cent[1:, :])
        p2g_dist, g2p_dist = ClusterEval.compute_minmax_distance(gt_cent[1:, :], pred_cent[1:, :])

        return {
            "cd": cd.item(),
            "p2g_dist": p2g_dist.item(),  # minimum distance from predicted centroids to gt centroids
            "g2p_dist": g2p_dist.item(),  # minimum distance from gt centroids to predicted centroids
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4)
        }

    @staticmethod
    def extract_cluster_by_label(pos, seg_results, min_clu_size=30):
        pred_labels = torch.argmax(seg_results, dim=-1).squeeze()
        pred_cents = scatter_mean(pos.to(pred_labels.device), pred_labels, dim=0)
        per_pred_clu_count = scatter_sum(torch.ones_like(pred_labels), pred_labels, dim=0)

        valid_clu = (per_pred_clu_count > min_clu_size)
        pred_cents = pred_cents[valid_clu]
        per_pred_clu_count = per_pred_clu_count[valid_clu]

        # 如果没有背景点（标签0），则插入一个全零向量作为背景中心点
        if 0 not in torch.unique(pred_labels) or (len(valid_clu) > 0 and not valid_clu[0]):
            # 在索引0位置插入全零背景点
            background_point = torch.zeros((1, pos.shape[1]), device=pred_cents.device, dtype=pred_cents.dtype)
            pred_cents = torch.cat([background_point, pred_cents], dim=0)
            per_pred_clu_count = torch.cat([torch.tensor([0], device=per_pred_clu_count.device), per_pred_clu_count],
                                           dim=0)

        return per_pred_clu_count, pred_cents

    def compute_tooth_sizes(points, gt_labels, valid_mask):
        """
        计算每个实例的尺寸（tooth_size）
        Args:
            points: 点云坐标 (N, 3)
            labels: 实例标签 (N,)
            instance_centroids: 实例质心 (num_instances, 3)
            valid_mask: 有效实例掩码
        Returns:
            instance_sizes: 每个实例的尺寸 (num_instances,)
        """
        # 计算每个点到其对应质心的距离平方
        # 首先获取每个点对应的质心索引
        u_instances = torch.unique(gt_labels)
        u_instances = u_instances[u_instances != 0]

        instance_sizes = torch.zeros_like(valid_mask).float()

        for i, l in enumerate(u_instances):
            gt_lbl = gt_labels[gt_labels == l]
            label = torch.unique(gt_lbl)

            assert len(label) == 1
            # compute gt tooth center and size

            gt_verts = points[gt_labels == l,:]
            gt_center = torch.mean(gt_verts, dim=0)

            dist = torch.sqrt(torch.sum((gt_verts - gt_center) ** 2, dim=1))

            instance_sizes[i+1] = dist.max()     # 第0个没有计算，因而索引从1开始

        # print(f"instance_sizes: {instance_sizes[valid_mask].cpu().numpy()}")
        # 只保留有效实例的 size
        return instance_sizes[valid_mask]

    @staticmethod
    def merge_centroids_hungarian(cluster_cents, pred_cents, distance_threshold=0.10):
        """
        使用匈牙利算法合并聚类质心和预测质心
        以cluster为主，未匹配的cluster保留

        Args:
            cluster_cents: TMC聚类质心 (N_cluster, 3)，第0个为背景
            pred_cents: 预测质心 (N_pred, 3)，第0个为背景
            distance_threshold: 匹配距离阈值

        Returns:
            merged_cents: 合并后的质心
            match_info: 匹配信息字典
        """
        from scipy.spatial.distance import cdist as scipy_cdist

        if isinstance(cluster_cents, torch.Tensor):
            cluster_cents = cluster_cents.cpu().numpy()
        if isinstance(pred_cents, torch.Tensor):
            pred_cents = pred_cents.cpu().numpy()

        # 跳过背景（索引0）
        cluster_wo_bg = cluster_cents[1:]
        pred_wo_bg = pred_cents[1:]

        N_cluster = len(cluster_wo_bg)
        N_pred = len(pred_wo_bg)

        print(f"\n🔗 质心合并分析")
        print(f"   Cluster质心数: {N_cluster}")
        print(f"   Pred质心数: {N_pred}")

        # 计算距离矩阵（使用scipy）
        dist_matrix = scipy_cdist(cluster_wo_bg, pred_wo_bg)

        # 匈牙利算法最优匹配
        row_ind, col_ind = linear_sum_assignment(dist_matrix)

        # 过滤掉超过阈值的匹配
        valid_mask = dist_matrix[row_ind, col_ind] < distance_threshold
        row_ind = row_ind[valid_mask]
        col_ind = col_ind[valid_mask]
        matched_distances = dist_matrix[row_ind, col_ind]

        # 统计匹配情况
        matched_cluster_idx = set(row_ind)
        matched_pred_idx = set(col_ind)
        unmatched_cluster_idx = set(range(N_cluster)) - matched_cluster_idx
        unmatched_pred_idx = set(range(N_pred)) - matched_pred_idx

        print(f"   成功匹配: {len(row_ind)} 对")
        print(f"   未匹配Cluster: {len(unmatched_cluster_idx)} 个")
        print(f"   未匹配Pred: {len(unmatched_pred_idx)} 个")
        if len(matched_distances) > 0:
            print(f"   平均匹配距离: {np.mean(matched_distances):.6f}")
        else:
            print(f"   无有效匹配")

        # 构建合并后的质心
        # 策略：对于匹配的，使用cluster质心；未匹配的cluster直接加入
        merged_list = [cluster_cents[0]]  # 保留背景

        # 添加所有cluster质心（包括匹配和未匹配的）
        for i in range(N_cluster):
            merged_list.append(cluster_wo_bg[i])

        # 添加未匹配的pred质心
        for i in unmatched_pred_idx:
            merged_list.append(pred_wo_bg[i])

        # merged_cents = np.array(merged_list)
        merged_cents = np.array(merged_list, dtype=np.float32)
        # 匹配信息
        match_info = {
            'matched_pairs': list(zip(row_ind.tolist(), col_ind.tolist())),
            'matched_distances': matched_distances.tolist(),
            'unmatched_cluster': list(unmatched_cluster_idx),
            'unmatched_pred': list(unmatched_pred_idx),
            'n_matched': len(row_ind),
            'n_unmatched_cluster': len(unmatched_cluster_idx),
            'n_unmatched_pred': len(unmatched_pred_idx)
        }

        print(f"   合并后质心数: {len(merged_cents)-1}")
        print(f"{'='*60}\n")

        return merged_cents, match_info


class TMCPlotter:

    @staticmethod
    def check_inputs(data_qslim=None,tmc_proposal=None):
        # 检查输入参数
        if data_qslim is None or tmc_proposal is None:
            print("data_qslim or tmc_proposal is None")
            return False

        return True

    @staticmethod
    def visualize_centroids_comparison(gt_centroids, pred_centroids, pos, face,
                                       save_path=None, window_size=[int(1920 * 1.5), int(1080 * 1.5)],
                                       elevation=90, azimuth=0):
        """
        可视化真实质心和预测质心的对比，并显示最近距离

        Args:
            gt_centroids: 真实质心 (N_gt, 3)
            pred_centroids: 预测质心 (N_pred, 3)
            pos: 点云坐标 (N, 3)
            face: 面片 (M, 3)
            save_path: 保存路径
            window_size: 窗口大小
            elevation: 仰角（度），90为俯视，0为平视
            azimuth: 方位角（度），0为正X轴方向
        """
        import pyvista as pv

        if isinstance(gt_centroids, torch.Tensor):
            gt_centroids = gt_centroids.cpu().numpy()
        if isinstance(pred_centroids, torch.Tensor):
            pred_centroids = pred_centroids.cpu().numpy()
        if isinstance(pos, torch.Tensor):
            pos = pos.cpu().numpy()
        if isinstance(face, torch.Tensor):
            face = face.cpu().numpy()

        V, F = check_n3(pos), check_n3(face)

        # 创建基础网格（半透明）
        mesh = pvgu.np_to_mesh(V, F)
        mesh.point_data['scalars'] = np.ones(len(V)) * 0.8
        mesh = pvgu.get_colored_mesh_scalar(mesh, mesh.point_data['scalars'],
                                           cmap_name='gray', vmin=0, vmax=1)

        # 创建质心点云
        gt_pcd = pv.PolyData(gt_centroids[1:])  # 跳过背景
        pred_pcd = pv.PolyData(pred_centroids[1:])  # 跳过背景

        # # 计算距离矩阵
        # from scipy.spatial.distance import cdist
        # dist_matrix = cdist(gt_centroids[1:], pred_centroids[1:])
        #
        # # 找到每个GT质心的最近预测质心
        # nearest_pred_idx = np.argmin(dist_matrix, axis=1)
        # nearest_distances = np.min(dist_matrix, axis=1)

        # # 为GT质心着色（根据最近距离）
        # gt_colors = np.zeros((len(gt_centroids)-1, 3))
        # for i in range(len(gt_centroids)-1):
        #     dist = nearest_distances[i]
        #     # 距离越小越绿，越大越红
        #     if dist < 0.05:
        #         gt_colors[i] = [0.0, 1.0, 0.0]  # 绿色
        #     elif dist < 0.10:
        #         gt_colors[i] = [1.0, 1.0, 0.0]  # 黄色
        #     else:
        #         gt_colors[i] = [1.0, 0.0, 0.0]  # 红色
        #
        # gt_pcd.point_data['colors'] = gt_colors
        #
        # # 预测质心用蓝色
        # pred_colors = np.tile([0.0, 1.0, 0.0], (len(pred_centroids)-1, 1))
        # pred_pcd.point_data['colors'] = pred_colors
        #
        # # 绘制连接线（仅显示距离较大的）
        # lines_mesh = pv.PolyData()
        # line_points = []
        # for i in range(len(gt_centroids)-1):
        #     if nearest_distances[i] > 0.05:  # 只显示距离>0.05的连接
        #         line_points.append(gt_centroids[i+1])
        #         line_points.append(pred_centroids[nearest_pred_idx[i]+1])
        #
        # if len(line_points) > 0:
        #     line_points = np.array(line_points)
        #     lines = pv.lines_from_points(line_points)
        #     lines_mesh = lines

        # 创建绘图器
        plotter = pv.Plotter(off_screen=True, window_size=window_size)
        plotter.background_color = 'white'

        # 添加网格
        plotter.add_mesh(mesh, show_edges=False, opacity=0.1)

        # # 添加GT质心（大球）
        # plotter.add_points(gt_pcd,
        #                   render_points_as_spheres=True,
        #                   point_size=15,
        #                   scalars='colors',
        #                   rgb=True)
        #
        # # 添加预测质心（小球）
        # plotter.add_points(pred_pcd,
        #                   render_points_as_spheres=True,
        #                   point_size=10,
        #                   scalars='colors',
        #                   rgb=True)

        # GT Centroids (Large Spheres)
        if gt_centroids is not None and len(gt_centroids) > 0:
            for cent in gt_centroids[1:]:
                # 修复：将 numpy 数组转换为 list
                sphere = pv.Sphere(radius=0.010, center=cent.tolist())
                plotter.add_mesh(sphere, color='red', opacity=0.4)

        # Pred Centroids (Small Spheres)
        if pred_centroids is not None and len(pred_centroids) > 0:
            for cent in pred_centroids[1:]:
                # 修复：将 numpy 数组转换为 list
                sphere = pv.Sphere(radius=0.006, center=cent.tolist())
                plotter.add_mesh(sphere, color='green', opacity=1.0)

        plotter.add_text("Centroids (Red:GT-Large, Green:Pred-Small)", font_size=10)

        # 设置视角
        plotter.view_xy()
        plotter.camera.elevation = elevation
        plotter.camera.azimuth = azimuth

        # 保存
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plotter.screenshot(save_path, window_size=window_size)
            print(f"✅ 质心对比图已保存: {save_path}")

        plotter.close()

        # 打印详细距离信息
        print(f"\n{'='*60}")
        print(f"📊 质心距离分析")
        print(f"{'='*60}")
        print(f"GT质心数: {len(gt_centroids)-1}")
        print(f"预测质心数: {len(pred_centroids)-1}")
        # print(f"平均距离: {avg_dist:.6f}")
        # print(f"最大距离: {max_dist:.6f}")
        # print(f"最小距离: {np.min(nearest_distances):.6f}")
        # print(f"\n距离分布:")
        # print(f"  < 0.05:  {np.sum(nearest_distances<0.05):3d} ({np.sum(nearest_distances<0.05)/len(nearest_distances)*100:.1f}%)")
        # print(f"  0.05-0.10: {np.sum((nearest_distances>=0.05)&(nearest_distances<0.10)):3d} ({np.sum((nearest_distances>=0.05)&(nearest_distances<0.10))/len(nearest_distances)*100:.1f}%)")
        # print(f"  ≥ 0.10:  {np.sum(nearest_distances>=0.10):3d} ({np.sum(nearest_distances>=0.10)/len(nearest_distances)*100:.1f}%)")
        # print(f"{'='*60}\n")
        #
        # return nearest_distances

    @staticmethod
    def visualize_mesh_and_centroids_comparison(
        gt_labels, pred_labels,
        gt_centroids, pred_centroids, cluster_centroids,
        pos, face,
        save_path=None,
        elevation=0, azimuth=0,
        window_size=[2880, 1620]
    ):
        """
        可视化网格及质心对比 (GT vs Pred)
        Args:
            gt_labels: GT 标签 [N]
            pred_labels: Pred 标签 [N]
            gt_centroids: GT 质心 [K, 3]
            pred_centroids: Pred 质心 [M, 3]
            pos: 顶点坐标 [N, 3]
            face: 面索引 [3, F]
            save_path: 保存路径
        """
        import pyvista as pv
        from external_libs import gen_utils as gu

        # 转换数据类型
        if isinstance(pos, torch.Tensor): pos = pos.cpu().numpy()
        if isinstance(face, torch.Tensor): face = face.cpu().numpy().T
        if isinstance(gt_labels, torch.Tensor): gt_labels = gt_labels.cpu().numpy()
        if isinstance(pred_labels, torch.Tensor): pred_labels = pred_labels.cpu().numpy()
        if isinstance(gt_centroids, torch.Tensor): gt_centroids = gt_centroids.cpu().numpy()
        if isinstance(pred_centroids, torch.Tensor): pred_centroids = pred_centroids.cpu().numpy()
        if isinstance(cluster_centroids, torch.Tensor): cluster_centroids = cluster_centroids.cpu().numpy()

        # 创建 Mesh
        gt_mesh = pvgu.np_to_mesh(pos, face)
        gt_mesh = pvgu.get_colored_mesh(gt_mesh, gt_labels, palette_type='label15')

        pred_mesh = pvgu.np_to_mesh(pos, face)
        pred_mesh = pvgu.get_colored_mesh(pred_mesh, pred_labels, palette_type='label15')

        # 创建绘图器 (1行3列)
        plotter = pv.Plotter(shape=(1, 2), window_size=window_size, off_screen=True,border=False)

        # # # --- 1. GT Mesh
        # plotter.subplot(0, 0)
        # plotter.add_mesh(gt_mesh,  show_edges=False, scalars='colors',rgb=True, lighting=False)
        # # plotter.remove_scalar_bar()
        #
        # plotter.view_xy()
        # plotter.camera.elevation = elevation
        # plotter.camera.azimuth = azimuth

        # --- 2. Pred Mesh + Centroids ---
        plotter.subplot(0, 0)
        plotter.add_mesh(pred_mesh, show_edges=False, scalars='colors', rgb=True, lighting=False)
        # plotter.remove_scalar_bar() # 彻底移除当前子图的颜色条

        plotter.view_xy()
        plotter.camera.elevation = elevation
        plotter.camera.azimuth = azimuth


        # --- 3. Centroids Comparison Only ---
        plotter.subplot(0, 1)

        # 添加透明纯色 Mesh 作为背景参考
        ref_mesh = pvgu.np_to_mesh(pos, face)
        plotter.add_mesh(ref_mesh, color='lightgray', opacity=0.1, show_edges=False)
        # plotter.enable_3_lights()

        # GT Centroids (Large Spheres)
        if gt_centroids is not None and len(gt_centroids) > 0:
            for cent in gt_centroids:
                # 修复：将 numpy 数组转换为 list
                sphere = pv.Sphere(radius=0.010, center=cent.tolist())
                plotter.add_mesh(sphere, color='red', opacity=0.3)

        # Pred Centroids (Small Spheres)
        if pred_centroids is not None and len(pred_centroids) > 0:
            for cent in pred_centroids:
                # 修复：将 numpy 数组转换为 list
                sphere = pv.Sphere(radius=0.006, center=cent.tolist())
                plotter.add_mesh(sphere, color='blue', opacity=0.9)

        # Cluster Centroids (Small Spheres)
        if cluster_centroids is not None and len(cluster_centroids) > 0:
            for cent in cluster_centroids:
                # 修复：将 numpy 数组转换为 list
                sphere = pv.Sphere(radius=0.006, center=cent.tolist())
                plotter.add_mesh(sphere, color='green', opacity=0.9)

        plotter.add_text("Centroids (Red:GT-Large, Blue:Pred-Small, Green:Cluster-Small)", font_size=10)
        plotter.view_xy()
        plotter.camera.elevation = elevation
        plotter.camera.azimuth = azimuth

        plotter.show()

        # 保存或显示
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plotter.render() # 关键：先渲染再截图
            plotter.screenshot(save_path)
            print(f"✅ 质心对比图已保存: {save_path}")
        else:
            plotter.show()
        plotter.close()




    @staticmethod
    def plot_lf(data_qslim=None,
                tmc_proposal=None,
                save_path=None,
                window_size=[int(1920 * 1.5), int(1080 * 1.5)],
                cmap_name='jet'):

        if not TMCPlotter.check_inputs(data_qslim, tmc_proposal):
            return

        # 检查必需的logits字段
        required_keys = ['bg_lf', 'tae_lf', 'tte_lf', 'gtte_lf']
        for key in required_keys:
            if key not in tmc_proposal or tmc_proposal[key] is None:
                print(f"⚠️ 警告: tmc_proposal 中缺少 '{key}' 字段")
                return  False



        V,F = data_qslim.pos.cpu().numpy(), data_qslim.face.cpu().numpy()
        V,F = check_n3(V), check_n3(F)

        mesh_qslim = pvgu.np_to_mesh(V, F)

        bg_lf = tmc_proposal['bg_lf'][:,1,:]
        tae_lf = tmc_proposal['tae_lf'][:,1,:]
        tte_lf = tmc_proposal['tte_lf'][:,1,:]
        gtte_lf = tmc_proposal['gtte_lf']

        # 延伸lf
        tte_gtte_lf = tte_lf + gtte_lf
        tae_tte_lf = tae_lf + tte_lf

        vmin,vmax = -5,5
        bg_lf = np.clip(bg_lf, vmin, vmax)
        tae_lf = np.clip(tae_lf, vmin, vmax)
        tte_lf = np.clip(tte_lf, vmin, vmax)
        gtte_lf = np.clip(gtte_lf, vmin, vmax)
        tte_gtte_lf = np.clip(tte_gtte_lf, vmin, vmax)
        tae_tte_lf = np.clip(tae_tte_lf, vmin, vmax)

        # bg_lf = perform_hist_eq(bg_lf, 'linear')
        # tae_lf = perform_hist_eq(tae_lf, 'linear')
        # tte_lf = perform_hist_eq(tte_lf, 'linear')
        # gtte_lf = perform_hist_eq(gtte_lf, 'linear')



        bg_mesh = pvgu.get_colored_mesh_scalar(mesh_qslim, bg_lf, cmap_name=cmap_name, vmin=vmin, vmax=vmax)
        tae_mesh = pvgu.get_colored_mesh_scalar(mesh_qslim, tae_lf, cmap_name=cmap_name, vmin=vmin, vmax=vmax)
        tte_mesh = pvgu.get_colored_mesh_scalar(mesh_qslim, tte_lf, cmap_name=cmap_name, vmin=vmin, vmax=vmax)
        gtte_mesh = pvgu.get_colored_mesh_scalar(mesh_qslim, gtte_lf, cmap_name=cmap_name, vmin=vmin, vmax=vmax)
        tte_gtte_mesh = pvgu.get_colored_mesh_scalar(mesh_qslim, tte_gtte_lf, cmap_name=cmap_name, vmin=vmin, vmax=vmax)
        tae_tte_mesh = pvgu.get_colored_mesh_scalar(mesh_qslim, tae_tte_lf, cmap_name=cmap_name, vmin=vmin, vmax=vmax)


        mesh_list = [bg_mesh, tae_mesh, tte_mesh, gtte_mesh, tte_gtte_mesh, tae_tte_mesh]
        # titles = ['Bg', 'Tta', 'Tte', 'Gtte']

        TMCVisPV.vis_mesh_grid_separate_render(
            mesh_list,
            rows=2,
            cols=int(np.ceil(len(mesh_list)/2)),
            window_size=window_size,
            elevation=90,
            azimuth=0,
            # elevation=45,
            # azimuth=30,
            save_path=save_path,
            margin=20,
            enable_lighting=False,
            # view = 'xy'
        )

    @staticmethod
    def plot_mask_meshes(data_qslim=None,
                        tmc_proposal=None,
                        save_path=None,
                         window_size=[int(1920 * 1.5), int(1080 * 1.5)],
                        cmap_name='jet'):
        """
        使用 PyVista 绘制并保存网格（推荐版本）

        Args:
            V: 顶点数组 (N, 3)
            F: 面片数组 (M, 3)
            bg_mask: 背景掩码
            edge_mask: 边缘掩码
            tae_mask: TAE 掩码
            tte_mask: TTE 掩码
            verts_clu_labels: 顶点聚类标签
            gt_labels: Ground Truth 标签
            pred_labels: 预测标签
            knn_ica_meshes: KNN ICA 网格列表
            save_path: 保存路径
            show_edges: 是否显示网格边
            background: 背景颜色
            window_size: 窗口大小 [width, height]
        """
        if not TMCPlotter.check_inputs(data_qslim, tmc_proposal):
            return

        # 检查必需的logits字段
        required_keys = ['bg_mask', 'tae_mask', 'tte_mask', 'edge_mask','verts_clu_labels']
        for key in required_keys:
            if key not in tmc_proposal or tmc_proposal[key] is None:
                print(f"⚠️ 警告: tmc_proposal 中缺少 '{key}' 字段")
                return  False

        V,F = data_qslim.pos.cpu().numpy(), data_qslim.face.cpu().numpy()
        V,F = check_n3(V), check_n3(F)

        gt_labels = data_qslim.y.cpu().numpy()

        bg_mask = tmc_proposal['bg_mask'].cpu().numpy()
        edge_mask = tmc_proposal['edge_mask'].cpu().numpy()
        tae_mask = tmc_proposal['tae_mask'].cpu().numpy()
        tte_mask = tmc_proposal['tte_mask'].cpu().numpy()
        verts_clu_labels = tmc_proposal['verts_clu_labels']

        # 创建基础网格（使用 PyVista 工厂）
        gt_mesh = PVMeshFactory.create_raw_mesh(V, F)
        raw_mesh = PVMeshFactory.create_raw_mesh(V, F)
        edge_mesh = PVMeshFactory.create_edge_mesh(V, F, edge_mask)
        tooth_mesh = PVMeshFactory.create_tooth_mesh(V, F, bg_mask)
        tae_sep_teeth_mesh = PVMeshFactory.create_sep_teeth_mesh(V, F, bg_mask, tae_mask)
        tae_tte_sep_teeth_mesh = PVMeshFactory.create_sep_teeth_mesh(V, F, bg_mask, tae_mask + tte_mask)
        edge_sep_teeth_mesh = PVMeshFactory.create_sep_teeth_mesh(V, F, bg_mask, edge_mask)

        # 上色（使用 pv_gen_utils）
        if gt_labels is not None:
            gt_mesh = pvgu.get_colored_mesh(gt_mesh, gt_labels.astype(np.int32), 'label15')
        raw_mesh = pvgu.get_colored_mesh(raw_mesh, verts_clu_labels.astype(np.int32), 'gist_ncar')
        tooth_mesh = pvgu.get_colored_mesh(tooth_mesh, verts_clu_labels.astype(np.int32), 'gist_ncar')
        tae_sep_teeth_mesh = pvgu.get_colored_mesh(tae_sep_teeth_mesh, verts_clu_labels.astype(np.int32), 'gist_ncar')
        tae_tte_sep_teeth_mesh = pvgu.get_colored_mesh(tae_tte_sep_teeth_mesh, verts_clu_labels.astype(np.int32),
                                                       'gist_ncar')
        edge_sep_teeth_mesh = pvgu.get_colored_mesh(edge_sep_teeth_mesh, verts_clu_labels.astype(np.int32),
                                                    'gist_ncar')

        # 构建网格列表
        mesh_list = [raw_mesh, gt_mesh, edge_mesh, tooth_mesh, tae_sep_teeth_mesh, tae_tte_sep_teeth_mesh, edge_sep_teeth_mesh]


        if 'knn_ica_meshes' in tmc_proposal and tmc_proposal['knn_ica_meshes'] is not None:
            knn_ica_meshes = tmc_proposal['knn_ica_meshes']
            # 调试输出
            print(f"🔍 knn_ica_meshes 类型：{type(knn_ica_meshes)}")
            for i, mesh in enumerate(knn_ica_meshes):
                if mesh is None:
                    print(f"   ⚠️ knn_ica_meshes[{i}] 是 None")

            # 过滤掉 None 元素
            # 过滤掉 None 元素并转换为 PyVista 对象
            valid_knn_meshes = []
            knn_names = ['alpha', 'bnd_contour', 'contour', 'vectors']

            for idx, (mesh, name) in enumerate(zip(knn_ica_meshes, knn_names)):

                if mesh is not None:
                    try:
                        # 将 Open3D 对象转换为 PyVista 对象
                        pv_mesh = TMCVisPV._o3d_to_pyvista(mesh)
                        if pv_mesh is not None:
                            valid_knn_meshes.append(pv_mesh)
                        print(f"   ✅ {name}: Open3D -> PyVista 转换成功")
                    except Exception as e:
                        print(f"   ⚠️ {name}: 转换失败 - {e}")
                else:
                    print(f"   ⚠️ {name}: 跳过（None）")

            mesh_list.extend(valid_knn_meshes)

        # 可视化
        TMCVisPV.vis_mesh_grid_separate_render(
            mesh_list,
            rows=2,
            cols=int(np.ceil(len(mesh_list)/2)),
            window_size=window_size,
            elevation=90,
            azimuth=0,
            # elevation=45,
            # azimuth=30,
            save_path=save_path,
            margin=20,
            enable_lighting=False,
            point_size=4.0,
            line_width=1.0,
            # view = 'xy'
        )


def save_tmc_evaluation_results_to_excel(
        all_results,
        save_path=None
):
    """
    将 TMC 评估结果保存到 Excel 文件（支持多条记录）

    Args:
        all_results: 列表，每个元素是一个字典，包含：
            {
                'name': '样本文件名',
                'TP': 真阳性,
                'FP': 假阳性,
                'FN': 假阴性,
                'precision': 精确率,
                'recall': 召回率,
                'f1': F1 分数,
                'cd': Chamfer 距离,
                'p2g_dist': 预测到真实的距离,
                'g2p_dist': 真实到预测的距离
            }
        save_path: Excel 保存路径
    """
    if save_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = f"tmc_evaluation_results_{timestamp}.xlsx"

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

    # ========== Sheet 1: 所有样本的详细指标 ==========
    all_records = []

    for result in all_results:
        filename = result.get('name', 'Unknown')

        record = {
            'Filename': filename,
            'TP': result.get('TP', 0),
            'FP': result.get('FP', 0),
            'FN': result.get('FN', 0),
            'Precision': f"{result.get('precision', 0):.4f}",
            'Recall': f"{result.get('recall', 0):.4f}",
            'F1 Score': f"{result.get('f1', 0):.4f}",
            'Chamfer Distance': f"{result.get('cd', 0):.4f}",
            'P2G Distance': f"{result.get('p2g_dist', 0):.4f}",
            'G2P Distance': f"{result.get('g2p_dist', 0):.4f}",
            'Background IOU': f"{result.get('background_iou', 0):.4f}",
            'Edge Binary IOU': f"{result.get('edge_binary_iou', 0):.4f}",
            'TTE  Binary IOU': f"{result.get('tte_binary_iou', 0):.4f}"
        }

        all_records.append(record)

    df_details = pd.DataFrame(all_records)

    # ========== Sheet 2: 统计摘要 ==========
    numeric_cols = ['Precision', 'Recall', 'F1 Score', 'Chamfer Distance',
                    'P2G Distance', 'G2P Distance', 'Background IOU', 'Edge Binary IOU', 'TTE  Binary IOU']

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

    # ========== 保存到 Excel ==========
    try:
        with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
            df_details.to_excel(writer, sheet_name='All Samples', index=False)
            df_summary.to_excel(writer, sheet_name='Statistics', index=False)

        print("\n" + "=" * 70)
        print("📊 TMC 评估结果已保存到 Excel")
        print("=" * 70)
        print(f"保存路径：{os.path.abspath(save_path)}")
        print(f"\n包含以下工作表:")
        print(f"  1. All Samples - 所有样本的详细指标 ({len(all_results)} 条记录)")
        print(f"  2. Statistics - 统计摘要 (均值/标准差/最大/最小)")
        print("=" * 70)

        return save_path

    except Exception as e:
        print(f"❌ 保存失败：{e}")
        raise


def batch_tmc_pred(ckpt_path, version_name, tmc, test_loader, thre_dist=0.10, save_excel=True):
    """
    批量预测并评估 TMC 模型性能

    Args:
        ckpt_path: 模型检查点路径
        version_name: 版本名称
        tmc: TopoMorphCluster 实例
        test_loader: 测试数据加载器
        thre_dist: 距离阈值
        save_excel: 是否保存 Excel 结果
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_save_path = Path(ckpt_path).parent
    last_folder_name = Path(base_save_path).name


    # excel 保存目录
    excel_save_path = base_save_path / f"{version_name}_{timestamp}.xlsx"

    print(f"\n📂 Excel 保存路径：{excel_save_path}")

    all_metrics_list = []
    error_list = []

    for batch_idx, batch in enumerate(test_loader):
        print(f"评估样本 {batch_idx + 1}/{len(test_loader)}: {batch.filename}")

        try:
            data = tmc(batch)

            if not hasattr(data, 'tmc_proposal'):
                error_list.append(f"{batch.filename} 缺少 tmc_proposal")
                continue
            tmc_proposal = data.tmc_proposal

            # edge predictor metrics

            background_iou, edge_binary_iou, tte_binary_iou = compute_edge_metrics(data, tmc_proposal)

            # cluster metrics
            if not hasattr(data, 'tooth_sizes'):
                data.tooth_sizes = ClusterEval.compute_tooth_sizes(data.pos, data.y, data.instance_labels > 0)

            clu_metrics = ClusterEval.compute_detection_metrics(
                data.instance_centroids,
                data.tmc_proposal['cluster_centroids'],
                data.tooth_sizes,
                thre_dist=thre_dist
            )

            clu_metrics['name'] = data.filename[0] if isinstance(data.filename, list) else data.filename

            ordered_metrics = OrderedDict([
                ('name', clu_metrics['name']),
                ('background_iou', float(background_iou)),
                ('edge_binary_iou', float(edge_binary_iou)),
                ('tte_binary_iou', float(tte_binary_iou)),
                ('TP', clu_metrics['TP']),
                ('FP', clu_metrics['FP']),
                ('FN', clu_metrics['FN']),
                ('precision', clu_metrics['precision']),
                ('recall', clu_metrics['recall']),
                ('f1', clu_metrics['f1']),
                ('cd', clu_metrics['cd']),
                ('p2g_dist', clu_metrics['p2g_dist']),
                ('g2p_dist', clu_metrics['g2p_dist']),
            ])

            print(f"TP={clu_metrics['TP']}, FP={clu_metrics['FP']}, FN={clu_metrics['FN']}")
            print(
                f"precision={clu_metrics['precision']:.4f}, recall={clu_metrics['recall']:.4f}, f1={clu_metrics['f1']:.4f}")
            print(
                f"cd={clu_metrics['cd']:.6f}, p2g_dist={clu_metrics['p2g_dist']:.6f}, g2p_dist={clu_metrics['g2p_dist']:.6f}")
            print(
                f"background_iou={background_iou:.4f}, edge_binary_iou={edge_binary_iou:.4f}, tte_binary_iou={tte_binary_iou:.4f}")

            all_metrics_list.append(ordered_metrics)

        except Exception as e:
            error_list.append(batch.filename if hasattr(batch, 'filename') else f"Batch {batch_idx}")
            print(f"❌ Error in batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()

    for error_file in error_list:
        print(f"Error in {error_file}")

    if save_excel and len(all_metrics_list) > 0:
        save_tmc_evaluation_results_to_excel(
            all_results=all_metrics_list,
            save_path=excel_save_path
        )

    avg_metrics = {}
    if len(all_metrics_list) > 0:
        numeric_keys = [key for key in all_metrics_list[0].keys() if key != 'name']

        for key in numeric_keys:
            values = [metrics[key] for metrics in all_metrics_list]
            avg_value = np.mean(values)
            avg_metrics[key] = avg_value
            metric_name = key.split('_')[-1] if '_' in key else key
            print(f"{key}: {avg_value:.6f}")

    return all_metrics_list, avg_metrics




def compute_edge_metrics(data, tmc_proposal):
    background_iou = 0.0
    edge_binary_iou = 0.0
    tte_binary_iou = 0.0
    if 'outputs' in tmc_proposal and tmc_proposal['outputs'].get('edge_seg_result') is not None:
        pred_tae_mask = torch.argmax(tmc_proposal['outputs']['edge_seg_result'], dim=1).reshape(-1, )  # 结果形状 (B, N)
        gt_edge_mask = data.gt_edge_mask.reshape(-1, )
        edge_binary_iou = cal_edge_iou(gt_edge_mask.cpu().numpy(), pred_tae_mask.cpu().numpy(), target_class=1)
    if 'outputs' in tmc_proposal and tmc_proposal['outputs'].get('tte_seg_result') is not None:
        pred_tte_mask = torch.argmax(tmc_proposal['outputs']['tte_seg_result'], dim=1).reshape(-1, )
        gt_tte_mask = data.gt_tte_mask.reshape(-1, )
        tte_binary_iou = cal_edge_iou(gt_tte_mask.cpu().numpy(), pred_tte_mask.cpu().numpy(), target_class=1)
    if 'outputs' in tmc_proposal and tmc_proposal['outputs'].get('bg_seg_result') is not None:
        pred = torch.argmax(tmc_proposal['outputs']['bg_seg_result'], dim=1).reshape(-1, )
        labels = data.y.reshape(-1, )

        background_pred = (pred == 0).squeeze().float()
        background_labels = (labels == 0).squeeze().float()
        _, _, _, _, background_iou, _ = cal_binary_metric_label(gt_labels=background_labels,
                                                                pred_classes=background_pred,
                                                                target_class=1)
    return background_iou, edge_binary_iou, tte_binary_iou


def get_mesh_edge_statistics(mesh):
    """
    使用向量化操作快速计算边长统计
    """
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    vertices, triangles = check_n3(vertices), check_n3(triangles)

    if len(triangles) == 0 or len(vertices) == 0:
        return {'mean_length': 0, 'min_length': 0, 'max_length': 0, 'std_length': 0}

    # 计算KNN平均距离
    knn_avg_dist = compute_knn_average_distance(torch.from_numpy(vertices), 3).cpu().numpy()

    # 获取三角形的三组顶点
    v0 = vertices[triangles[:, 0],:]
    v1 = vertices[triangles[:, 1],:]
    v2 = vertices[triangles[:, 2],:]

    # 计算三条边的长度（向量化）
    l0 = np.linalg.norm(v0 - v1, axis=1)
    l1 = np.linalg.norm(v1 - v2, axis=1)
    l2 = np.linalg.norm(v2 - v0, axis=1)

    # 合并所有边长
    all_edges = np.concatenate([l0, l1, l2])

    # 去除重复边（可选）
    # 构建边索引对（排序以确保唯一性）
    edge_indices = np.concatenate([
        np.sort(np.column_stack([triangles[:, 0], triangles[:, 1]]), axis=1),
        np.sort(np.column_stack([triangles[:, 1], triangles[:, 2]]), axis=1),
        np.sort(np.column_stack([triangles[:, 2], triangles[:, 0]]), axis=1)
    ])

    # 使用 unique 获取唯一边
    unique_edges, unique_idx = np.unique(edge_indices, axis=0, return_index=True)
    unique_lengths = all_edges[unique_idx]

    return {
        'knn_avg_dist': knn_avg_dist,
        'mean_length_ratio': float(np.mean(unique_lengths))/knn_avg_dist,
        'min_length_ratio': float(np.min(unique_lengths))/knn_avg_dist,
        'max_length_ratio': float(np.max(unique_lengths))/knn_avg_dist,
        'std_length_ratio': float(np.std(unique_lengths))/knn_avg_dist,
        'edge_count': len(unique_lengths)
    }


# class TMCVisPV:
#     """PyVista 版本的 TMC 可视化工具"""
#
#     @staticmethod
#     def vis_mesh_grid(mesh_list, rows=None, cols=None, x_offset=1.2, y_offset=1.2,
#                          title_prefix="Mesh",
#                          save_path="vis_mesh_grid.png",
#                          show_edges=False,
#                          background='white',
#                          window_size=[3840, 2160]):
#         """
#         使用 PyVista 将多个 mesh 按网格布局排列并可视化
#
#         Args:
#             mesh_list: mesh 列表，每个元素是 (pyvista.PolyData, title) 元组
#             rows: 行数，如果为 None 则自动计算
#             cols: 列数，如果为 None 则自动计算
#             x_offset: x 方向偏移量
#             y_offset: y 方向偏移量
#             title_prefix: 标题前缀
#             save_path: 保存路径
#             show_edges: 是否显示网格边
#             background: 背景颜色 'white' 或 'black'
#             window_size: 窗口大小 [width, height]
#
#         Returns:
#             pyvista.Plotter 对象
#         """
#         n_meshes = len(mesh_list)
#
#         if n_meshes == 0:
#             print("⚠️ mesh_list 为空")
#             return None
#
#         # 自动计算行列数
#         if rows is None and cols is None:
#             cols = int(np.ceil(np.sqrt(n_meshes)))
#             rows = int(np.ceil(n_meshes / cols))
#         elif rows is None:
#             rows = int(np.ceil(n_meshes / cols))
#         elif cols is None:
#             cols = int(np.ceil(n_meshes / rows))
#
#         print(f"📊 PyVista 网格布局：{rows}行 × {cols}列，共 {n_meshes} 个 mesh")
#
#         # 创建 Plotter
#         plotter = pv.Plotter(off_screen=True, window_size=window_size)
#         plotter.background_color = background
#
#         # 渲染每个 mesh
#         for idx, mesh_item in enumerate(mesh_list):
#             if isinstance(mesh_item, tuple):
#                 mesh, custom_title = mesh_item
#             else:
#                 mesh = mesh_item
#                 custom_title = None
#
#             if mesh is None:
#                 print(f"⚠️ mesh_list 中第 {idx + 1} 个元素为 None")
#                 continue
#
#             # 计算当前 mesh 在网格中的位置
#             row = idx // cols
#             col = idx % cols
#
#             # 计算平移量（以网格中心为原点）
#             center_x = (col - (cols - 1) / 2) * x_offset
#             center_y = (row - (rows - 1) / 2) * (-y_offset)
#
#             # 复制并平移 mesh
#             mesh_copy = mesh.copy()
#             mesh_copy.points += np.array([center_x, center_y, 0])
#
#             # 生成标题
#             if custom_title:
#                 title = f"{title_prefix}_{custom_title}" if title_prefix else custom_title
#             else:
#                 title = f"{title_prefix}_{idx + 1}" if title_prefix else f"Mesh_{idx + 1}"
#
#             # 添加 mesh 到场景
#             if 'colors' in mesh_copy.point_data:
#                 plotter.add_mesh(
#                     mesh_copy,
#                     scalars='colors',
#                     rgb=True,
#                     show_edges=show_edges,
#                     smooth_shading=True
#                 )
#             elif 'labels' in mesh_copy.point_data:
#                 plotter.add_mesh(
#                     mesh_copy,
#                     scalars='labels',
#                     cmap='tab20',
#                     show_edges=show_edges,
#                     smooth_shading=True
#                 )
#             else:
#                 plotter.add_mesh(
#                     mesh_copy,
#                     show_edges=show_edges,
#                     smooth_shading=True
#                 )
#
#             print(f"  [{idx + 1}/{n_meshes}] {title}: 位置 ({row},{col}), 平移 ({center_x:.3f}, {center_y:.3f})")
#
#         # 设置相机视角
#         plotter.view_isometric()
#         plotter.add_axes()
#
#         # 保存图片
#         if save_path:
#             save_dir = os.path.dirname(save_path)
#             if save_dir and not os.path.exists(save_dir):
#                 os.makedirs(save_dir, exist_ok=True)
#
#             plotter.screenshot(save_path, window_size=window_size)
#             print(f"✅ 高清图片已保存：{save_path}")
#             print(f"   分辨率：{window_size[0]}x{window_size[1]} 像素")
#
#         plotter.close()
#         return plotter
#
#     @staticmethod
#     def plot_save_meshes(V, F, bg_mask, edge_mask, tae_mask, tte_mask,
#                               verts_clu_labels,
#                               gt_labels=None,
#                               pred_labels=None,
#                               knn_ica_meshes=None,
#                               save_path="vis_mesh_grid.png",
#                               show_edges=False,
#                               background='white',
#                               window_size=[3840, 2160]):
#         """
#         使用 PyVista 绘制并保存网格（推荐版本）
#
#         Args:
#             V: 顶点数组 (N, 3)
#             F: 面片数组 (M, 3)
#             bg_mask: 背景掩码
#             edge_mask: 边缘掩码
#             tae_mask: TAE 掩码
#             tte_mask: TTE 掩码
#             verts_clu_labels: 顶点聚类标签
#             gt_labels: Ground Truth 标签
#             pred_labels: 预测标签
#             knn_ica_meshes: KNN ICA 网格列表
#             save_path: 保存路径
#             show_edges: 是否显示网格边
#             background: 背景颜色
#             window_size: 窗口大小 [width, height]
#         """
#         factory = PVMeshFactory
#
#         # 创建基础网格
#         gt_mesh = factory.create_raw_mesh(V, F)
#         pred_mesh = factory.create_raw_mesh(V, F)
#         raw_mesh = factory.create_raw_mesh(V, F)
#         edge_mesh = factory.create_edge_mesh(V, F, edge_mask)
#         tooth_mesh = factory.create_tooth_mesh(V, F, bg_mask)
#         tae_sep_teeth_mesh = factory.create_sep_teeth_mesh(V, F, bg_mask, tae_mask)
#         tae_tte_sep_teeth_mesh = factory.create_sep_teeth_mesh(V, F, bg_mask, tae_mask + tte_mask)
#         edge_sep_teeth_mesh = factory.create_sep_teeth_mesh(V, F, bg_mask, edge_mask)
#
#         # 上色（使用 pv_gen_utils）
#         if gt_labels is not None:
#             gt_mesh = pvgu.get_colored_mesh(gt_mesh, gt_labels.astype(np.int32), 'label15')
#         if pred_labels is not None:
#             pred_mesh = pvgu.get_colored_mesh(pred_mesh, pred_labels.astype(np.int32), 'label15')
#         raw_mesh = pvgu.get_colored_mesh(raw_mesh, verts_clu_labels.astype(np.int32), 'label15')
#         tooth_mesh = pvgu.get_colored_mesh(tooth_mesh, verts_clu_labels.astype(np.int32), 'label15')
#         tae_sep_teeth_mesh = pvgu.get_colored_mesh(tae_sep_teeth_mesh, verts_clu_labels.astype(np.int32), 'label15')
#         tae_tte_sep_teeth_mesh = pvgu.get_colored_mesh(tae_tte_sep_teeth_mesh, verts_clu_labels.astype(np.int32),
#                                                        'label15')
#         edge_sep_teeth_mesh = pvgu.get_colored_mesh(edge_sep_teeth_mesh, verts_clu_labels.astype(np.int32), 'label15')
#
#         # 构建网格列表
#         mesh_list = [
#             (raw_mesh, 'raw'),
#             (gt_mesh, 'gt'),
#             (pred_mesh, 'pred'),
#             (edge_mesh, 'edge'),
#             (tooth_mesh, 'tooth'),
#             (edge_sep_teeth_mesh, 'sep_teeth')
#         ]
#
#         if knn_ica_meshes:
#             mesh_list.extend([
#                 (knn_ica_meshes[0], 'alpha'),
#                 (knn_ica_meshes[1], 'bnd_contour'),
#                 (knn_ica_meshes[2], 'contour'),
#                 (knn_ica_meshes[3], 'vectors'),
#             ])
#
#         # 计算偏移量
#         V_range = V.max(axis=0) - V.min(axis=0)
#         x_offset = V_range[0] * 1.1
#         y_offset = V_range[1] * 1.1
#
#         # 可视化
#         TMCVisPV.vis_mesh_grid(
#             mesh_list,
#             save_path=save_path,
#             x_offset=x_offset,
#             y_offset=y_offset,
#             show_edges=show_edges,
#             background=background,
#             window_size=window_size
#         )


def eval_TMC():
    print("Current working directory:", os.getcwd())

    config_path = "../configs"
    config_name = "module_TMC"
    cfg = get_yaml_config(config_path=config_path, config_name=config_name)
    args = cfg.paths
    ds_args = cfg.dataset
    print_yaml_cfg_dict(cfg)

    # 1. read data
    reader = T3DSPlusSegReader(is_filter_small_instance=ds_args.is_filter_small_instance,
                               filter_small_instance_size=ds_args.filter_small_instance_size,
                               verbose=True)

    # 难分样例：ANLLPLV7， B5GFZIRW，
    # 01A6HAN6   0TMOBYXS JJEIMAPR、ameziani、00OMSZGW(姿态严重)， WRT7Y7P9，MGYSCSI9
    # EJWZZZRF, patient1 014F9HTN, 017APYC4 01M10NMV 01MAU84T 2CE5VDAE ANLLPLV7 B5GFZIRW DG27PDD4
    # IU0UV8E_upper OTKQ5J9_upper, CE5VDAE_upper, PLDXK7R_upper, CQZX40I5_upper,

    # 01KTRG9K_lower 0OF8OOCX_lower 1QI2XWMD_lower IKZNMT4Z_upper 01K17AN8_lower 4G9LHQ2X_lower
    # 特别案例 01K3G866_lower B5GFZIRW_lower FHA6N618_lower
    # 01M10NMV_lower 4J24X0ES_upper AVI68NCT_upper B5GFZIRW_lower CD81F1F4_upper CQZX40I5_upper
    # E552BK9X_upper  GNR6QR3P_lower TVSR5QBQ_lower U9UBDO05_upper VB7GYS2O_upper Y48DURWV_upper ZD89X7G1_lower

    # name = "01KTRG9K_lower"
    name = "E23G704K_upper"
    LU = name.split('_')[1]
    patentt_id = name.split('_')[0]
    obj_name = f"{LU}/{patentt_id}/{patentt_id}_{LU}.obj"
    json_name = f"{LU}/{patentt_id}/{patentt_id}_{LU}.json"

    obj_filename = os.path.join(args.source_obj_data_path, obj_name)
    seg_json_filename = os.path.join(args.source_seg_json_data_path, json_name)

    data = reader(obj_filename, seg_json_filename)
    print(data.keys())

    # 2. pre_transform
    pre_transform = instantiate_from_config(cfg.pre_transform_obj)
    # print(pre_transform)
    data = pre_transform(data)
    batch = Batch.from_data_list([data])

    save_lf_path = "tmp/logtis_field"
    # 3. TMC
    tmc = TopoMorphCluster(num_classes=15,
                           predictor_type='DTSegNet',
                           ckpt_path='../common/ckpts/DTSegnet.ckpt',
                           # ckpt_path='../../../common/DTSegNet-0.025ratio.ckpt',
                           mounted_mask_outputs= True,
                           mounted_knn_ica_meshes=True,
                           save_lf_path=None,
                           net_para_name='seg_base',
                           filter_type='tae_tte_gtte',  # bg, tae, tae_tte, tae_tte_gtte, tae_tte_ptae
                           aggr_type='add',  # boolean, add
                           small_cluster_size_thre = 30,
                           alpha = 0.12,
                           k = 3,
                           max_iter = 5,
                           iter_thred_len_ratio = 0.01,
                           field_bound_ratio = 1.5,
                           field_sigma_ratio = 1)
    # TMC聚类
    data = tmc(batch[0])

    # print(data.tmc_proposal)

    # 聚类评估
    # eval_topo_morph_cluster(cluster_centroids, cluster_labels, data)
    print(data.instance_centroids.shape, data.instance_labels, data.instance_count)

    thre_dist = 0.08

    if not hasattr(data, 'tooth_sizes'):
        data.tooth_sizes = ClusterEval.compute_tooth_sizes(data.pos, data.y, data.instance_labels > 0)

    cluster_cent = data.tmc_proposal['cluster_centroids']
    # 聚类评估
    clu_metrics = ClusterEval.compute_detection_metrics(data.instance_centroids,
                                                        cluster_cent,
                                                        data.tooth_sizes,
                                                        thre_dist=thre_dist)

    print(clu_metrics)

    # 方法2：直接根据实例分割结果直接用y预测聚类结果
    batch = Batch.from_data_list([data])
    with torch.no_grad():
        pred_data = tmc._seg_predictor(batch)
        # pred_data = pred_data.cpu()

    seg_results = pred_data.outputs['seg_result']
    per_pred_clu_count, pred_cents = ClusterEval.extract_cluster_by_label(pred_data.pos, seg_results)

    pred_metrics = ClusterEval.compute_detection_metrics(data.instance_centroids.to(pred_cents.device),
                                                        pred_cents,
                                                        tooth_sizes=data.tooth_sizes,
                                                        thre_dist=thre_dist)

    print(per_pred_clu_count,pred_metrics)

    # 基于拓扑聚类结果评估

    logger = CSVLogger(
        save_dir="./logs/tmc_experiments",
        name="tmc_metrics",
        version=1
    )
    clu_metrics['name'] = data.filename[0] if isinstance(data.filename, list) else data.filename

    # 使用 OrderedDict 控制列的顺序
    ordered_metrics = OrderedDict([
        ('01_name', clu_metrics['name']),
        ('03_TP', clu_metrics['TP']),
        ('04_FP', clu_metrics['FP']),
        ('05_FN', clu_metrics['FN']),
        ('06_precision', clu_metrics['precision']),
        ('07_recall', clu_metrics['recall']),
        ('08_f1', clu_metrics['f1']),
        ('09_cd', clu_metrics['cd']),
        ('13_TP', pred_metrics['TP']),
        ('14_FP', pred_metrics['FP']),
        ('15_FN', pred_metrics['FN']),
        ('16_precision', pred_metrics['precision']),
        ('17_recall', pred_metrics['recall']),
        ('18_f1', pred_metrics['f1']),
        ('19_cd', pred_metrics['cd']),
    ])

    logger.log_metrics(ordered_metrics)
    logger.save()

    # V = data.data_qslim.pos.cpu().numpy()
    # F = data.data_qslim.face.cpu().numpy()
    # bg_mask = data.tmc_proposal['bg_mask']
    # tae_mask = data.tmc_proposal['tae_mask']
    # tte_mask = data.tmc_proposal['tte_mask']
    # edge_mask = data.tmc_proposal['edge_mask']
    # verts_clu_labels = data.tmc_proposal['verts_clu_labels']
    # knn_ica_meshes = data.tmc_proposal['knn_ica_meshes']

    # V,F = check_n3(V), check_n3(F)
    # TMCVisPV.plot_save_meshes(V, F, bg_mask, edge_mask, tae_mask, tte_mask, verts_clu_labels,
    #                  knn_ica_meshes=data.tmc_proposal['knn_ica_meshes'],
    #                  window_size=(3840, 2160 ),
    #                  save_path="vis_mesh_grid.png")
    filename = data.filename[0] if isinstance(data.filename, list) else data.filename
    save_path = os.path.join(save_lf_path, filename+'.png')
    if not exists(save_lf_path):
        os.makedirs(save_lf_path)
    TMCPlotter.plot_lf(data.data_qslim, data.tmc_proposal, save_path=save_path)

    save_mask_path = "tmp/mask_mesh"
    if not exists(save_mask_path):
        os.makedirs(save_mask_path)
    save_path = os.path.join(save_mask_path, filename+'.png')
    TMCPlotter.plot_mask_meshes(data.data_qslim, data.tmc_proposal,
                                save_path=save_path, window_size=(3840, 2160))


def test_CropBatchGenerator():
    print("Current working directory:", os.getcwd())

    config_path = "../configs"
    config_name = "test_TMC"
    # config_name = "test_T3DSPlusTransforms"
    cfg = get_yaml_config(config_path=config_path, config_name=config_name)
    args = cfg.paths
    ds_args = cfg.dataset
    print_yaml_cfg_dict(cfg)

    # print_yaml_config(config_path=config_path, config_name=config_name)

    reader = T3DSPlusSegReader(is_filter_small_instance=ds_args.is_filter_small_instance,
                               filter_small_instance_size=ds_args.filter_small_instance_size,
                               verbose=True)

    # argmin存在缺失的样本：  0EAKT1CU_lower 013FHA7K_upper

    name = "013FHA7K_upper"
    LU = name.split('_')[1]
    patentt_id = name.split('_')[0]
    obj_name = f"{LU}/{patentt_id}/{patentt_id}_{LU}.obj"
    json_name = f"{LU}/{patentt_id}/{patentt_id}_{LU}.json"

    obj_filename = os.path.join(args.source_obj_data_path, obj_name)
    seg_json_filename = os.path.join(args.source_seg_json_data_path, json_name)

    data = reader(obj_filename, seg_json_filename)
    print(data.keys())

    # print(cfg.pre_transform_obj)
    #
    # 先做规整化及PCA运算
    pre_transform = instantiate_from_config(cfg.pre_transform_obj)
    print(pre_transform)

    data = pre_transform(data)

    batch = Batch.from_data_list([data])

    # TMC聚类
    # tmc = TopoMorphCluster(num_classes=15,
    #              predictor_type='DTSegNet',
    #              ckpt_path='../../../common/DTSegNet.ckpt',
    #              # ckpt_path='../../../common/DTSegNet-0.025ratio.ckpt',
    #              net_para_name='seg_base',
    #              mounted_segnet_outputs=True,
    #              filter_type='tae_tte_gtte',  # bg, tae, tae_tte, tae_tte_gtte, tae_tte_ptae
    #              aggr_type='add',  # boolean, add
    #              small_cluster_size_thre = 30,
    #              alpha = 0.14,
    #              k = 3,
    #              max_iter = 5,
    #              iter_thred_len_ratio = 0.01,
    #              field_bound_ratio = 2,
    #              field_sigma_ratio = 1)
    #
    # data = tmc(batch[0])

    # 上一个网络的预测输出
    seg_result = data.outputs['seg_result']
    pred_seg_labels = torch.argmax(seg_result, dim=-1).view(-1)  # [1]
    print("pred_seg_labels.shape:", pred_seg_labels.shape)

    print(data.instance_centroids.shape, data.instance_labels, data.instance_count)

    # 裁剪批次
    outs = TMCUtils.build_crop_batch(data)
    crop_nn_indexes = outs['crop_nn_indexes']  # (1, clu, crop)
    cropped_feats = outs['crop_feats']
    cropped_pos = outs['crop_pos']
    crop_gt_binseg_label = outs['crop_gt_binseg_label']
    crop_gt_seg_label = outs['crop_gt_seg_label']
    cluster_gt_seg_labels = outs['cluster_gt_seg_labels']


    print(crop_nn_indexes.shape, cropped_feats.shape, cropped_pos.shape, crop_gt_binseg_label.shape,
          crop_gt_seg_label.shape, cluster_gt_seg_labels.shape)

    # 语义分割网络
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = [cropped_feats.float().to(device), crop_gt_binseg_label.long().to(device)]
    ins_segnet_model = PointTransformerNet(num_classes=2, num_feat=6).to(device)
    bin_outs = ins_segnet_model(inputs)
    print(bin_outs['cbl_loss'].shape, bin_outs['sem_1'].shape, bin_outs['first_features'].shape)

    # # 二值语义标签投票
    data.y_binseg = TMCUtils.map_crop2ori_label(bin_outs['sem_1'], crop_nn_indexes,  data.pos.shape[0])
    # data.y_binseg = (pred_seg_labels>0).view(-1)
    # # 二值分割网络结合语义分割网络投票
    # onehot = F.one_hot(crop_gt_binseg_label.view(-1), num_classes=2).reshape(-1, bin_outs['sem_1'].shape[2], 2)
    # onehot = onehot.permute(0, 2, 1)
    # data.y_voteseg = vote_seg_label(onehot, pred_seg_labels, crop_nn_indexes, num_classes=15)

    data.y_voteseg = TMCUtils.vote_seg_label(bin_outs['sem_1'], pred_seg_labels, crop_nn_indexes, num_classes=15)

    TMCUtils.vis_crops(cluster_gt_seg_labels, crop_gt_binseg_label, cropped_pos, data)




def prepare_data(cfg_obj: DictConfig, only_val=False):
    if cfg_obj.dataset.name == "T3DSPlusSegDataset":
        dm, train_loader, test_loader \
            = T3DSPlusSegDataset.get_dataloaders_from_cfg_obj(cfg_obj,only_val=only_val)
    elif cfg_obj.dataset.name == "TADPMSegDataset":
        dm, train_loader, test_loader \
            = TADPMSegDataset.get_dataloaders_from_cfg_obj(cfg_obj,only_val=only_val)

    return dm, train_loader, test_loader

def eval_batch_TMC():
    print("Current working directory:", os.getcwd())

    config_path = "../configs"
    config_name = "module_TMC"
    cfg = get_yaml_config(config_path=config_path, config_name=config_name)
    print_yaml_cfg_dict(cfg)

    cfg_obj = instantiate_from_config(cfg)

    version_name = generate_exp_name(cfg_obj.logger.exp_args, add_timestamp=False,only_vals= True)
    print(version_name)

    _, train_loader, test_loader =prepare_data(cfg_obj)

    print(train_loader, test_loader)

    # 创建logger
    logger = CSVLogger(
        save_dir=cfg_obj.logger.save_dir,
        name=cfg_obj.logger.exp_name,
        version=version_name
    )

    save_lf_path = os.path.join(cfg_obj.paths.images_path, version_name, "logtis_field")

    # 创建TMC
    tmc = TopoMorphCluster(num_classes=cfg_obj.tmc_cluster.num_classes,
                 predictor_type=cfg_obj.tmc_cluster.predictor_type,
                 ckpt_path=cfg_obj.tmc_cluster.ckpt_path,
                 net_para_name=cfg_obj.tmc_cluster.net_para_name,
                 filter_type=cfg_obj.tmc_cluster.filter_type,  # bg, tae, tae_tte, tae_tte_gtte
                 mounted_segnet_outputs=True,
                 mounted_mask_outputs=True,
                 mounted_knn_ica_meshes=True,
                 save_lf_path=save_lf_path,
                 aggr_type=cfg_obj.tmc_cluster.aggr_type,  # boolean, add
                 small_cluster_size_thre = cfg_obj.tmc_cluster.small_cluster_size_thre,
                 alpha = cfg_obj.tmc_cluster.alpha,
                 k = cfg_obj.tmc_cluster.k,
                 max_iter = cfg_obj.tmc_cluster.max_iter,
                 iter_thred_len_ratio = cfg_obj.tmc_cluster.iter_thred_len_ratio,
                 field_bound_ratio = cfg_obj.tmc_cluster.field_bound_ratio,
                 field_sigma_ratio = cfg_obj.tmc_cluster.field_sigma_ratio)

    all_metrics_list = []
    thre_dist = cfg_obj.tmc_cluster.thre_dist

    error_list = []

    for batch in test_loader:
        data = tmc(batch[0])

        try:
            if not hasattr(data, 'tooth_sizes'):
                data.tooth_sizes = ClusterEval.compute_tooth_sizes(data.pos, data.y, data.instance_labels > 0)

            # 聚类评估
            clu_metrics = ClusterEval.compute_detection_metrics(data.instance_centroids,
                                                                data.tmc_proposal['cluster_centroids'],
                                                                data.tooth_sizes,
                                                                thre_dist=thre_dist)
            # print(clu_metrics)

            # 直接用y预测聚类结果
            with torch.no_grad():
                pred_data = tmc._seg_predictor(batch)

            seg_results = pred_data.outputs['seg_result']
            per_pred_clu_count, pred_cents = ClusterEval.extract_cluster_by_label(pred_data.pos, seg_results)
            pred_metrics = ClusterEval.compute_detection_metrics(data.instance_centroids.to(pred_cents.device),
                                                                 pred_cents,
                                                                 data.tooth_sizes,
                                                                 thre_dist=thre_dist)
            # print(per_pred_clu_count, pred_metrics)
            # 基于拓扑聚类结果评估
            clu_metrics['name'] = data.filename[0] if isinstance(data.filename, list) else data.filename
            # 使用 OrderedDict 控制列的顺序
            ordered_metrics = OrderedDict([
                ('01_name', clu_metrics['name']),
                ('03_TP', clu_metrics['TP']),
                ('04_FP', clu_metrics['FP']),
                ('05_FN', clu_metrics['FN']),
                ('06_precision', clu_metrics['precision']),
                ('07_recall', clu_metrics['recall']),
                ('08_f1', clu_metrics['f1']),
                ('09_cd', clu_metrics['cd']),
                ('13_TP', pred_metrics['TP']),
                ('14_FP', pred_metrics['FP']),
                ('15_FN', pred_metrics['FN']),
                ('16_precision', pred_metrics['precision']),
                ('17_recall', pred_metrics['recall']),
                ('18_f1', pred_metrics['f1']),
                ('19_cd', pred_metrics['cd']),
            ])

            logger.log_metrics(ordered_metrics)

            f2q_idx, _ = NNPointSamplingIndexer()(data.pos.cpu(), data.data_qslim.pos.cpu())
            # pred_labels = torch.argmax(seg_results, dim=-1).squeeze()

            filename = data.filename[0] if isinstance(data.filename, list) else data.filename

            save_lf_path = os.path.join(cfg_obj.paths.images_path, version_name, 'lf')
            save_path = os.path.join(save_lf_path, filename + '.png')
            if not exists(save_lf_path):
                os.makedirs(save_lf_path)
            TMCPlotter.plot_lf(data.data_qslim, data.tmc_proposal,
                               save_path=save_path, window_size=[int(1920 * 1.5), int(1080 * 1.5)])

            save_mask_path = os.path.join(cfg_obj.paths.images_path, version_name, 'mask')
            if not exists(save_mask_path):
                os.makedirs(save_mask_path)
            save_path = os.path.join(save_mask_path, filename + '.png')
            TMCPlotter.plot_mask_meshes(data.data_qslim, data.tmc_proposal,
                                        save_path=save_path, window_size=[int(1920 * 1.5), int(1080 * 1.5)])



            all_metrics_list.append(ordered_metrics)
        except Exception as e:
            # print(f"Error in {data.filename}")
            error_list.append(data.filename)
            print(e)

    for error_file in error_list:
        print(f"Error in {error_file}")


    # 计算平均指标
    # 提取所有数值型指标并计算平均值
    numeric_keys = [key for key in ordered_metrics.keys() if key != '01_name']
    avg_metrics = {}

    for key in numeric_keys:
        values = [metrics[key] for metrics in all_metrics_list]
        avg_value = np.mean(values)
        avg_metrics[key] = avg_value

        # 打印平均值
        metric_name = key.split('_')[1] if '_' in key else key
        print(f"{key}: {avg_value:.4f}")

    # 创建平均指标的 OrderedDict
    avg_ordered_metrics = OrderedDict([
                                          ('01_name', 'AVERAGE'),
                                      ] + [(key, avg_metrics[key]) for key in numeric_keys])

    # 记录平均指标
    logger.log_metrics(avg_ordered_metrics)
    # logger.save()
    logger.save()


def eval_batch_TMC_v2():
    print("Current working directory:", os.getcwd())

    config_path = "../configs"
    config_name = "module_TMC"
    cfg = get_yaml_config(config_path=config_path, config_name=config_name)
    print_yaml_cfg_dict(cfg)

    cfg_obj = instantiate_from_config(cfg)

    version_name = generate_exp_name(cfg_obj.logger.exp_args, add_timestamp=True,only_vals= True)
    print(version_name)

    _, train_loader, test_loader =prepare_data(cfg_obj, only_val=True)

    print(train_loader, test_loader)


    save_lf_path = os.path.join(cfg_obj.paths.images_path, version_name, "logtis_field")
    save_logger_path = os.path.join(cfg_obj.paths.images_path, version_name)

    # 创建logger
    logger = CSVLogger(
        save_dir= cfg_obj.paths.images_path,
        name= version_name,
        version="csv"
    )

    # 创建TMC
    tmc = TopoMorphCluster(num_classes=cfg_obj.tmc_cluster.num_classes,
                 predictor_type=cfg_obj.tmc_cluster.predictor_type,
                 ckpt_path=cfg_obj.tmc_cluster.ckpt_path,
                 net_para_name=cfg_obj.tmc_cluster.net_para_name,
                 filter_type=cfg_obj.tmc_cluster.filter_type,  # bg, tae, tae_tte, tae_tte_gtte
                 mounted_segnet_outputs=True,
                 mounted_mask_outputs=True,
                 mounted_knn_ica_meshes=True,
                 save_lf_path=save_lf_path,
                 aggr_type=cfg_obj.tmc_cluster.aggr_type,  # boolean, add
                 small_cluster_size_thre = cfg_obj.tmc_cluster.small_cluster_size_thre,
                 alpha = cfg_obj.tmc_cluster.alpha,
                 k = cfg_obj.tmc_cluster.k,
                 max_iter = cfg_obj.tmc_cluster.max_iter,
                 iter_thred_len_ratio = cfg_obj.tmc_cluster.iter_thred_len_ratio,
                 field_bound_ratio = cfg_obj.tmc_cluster.field_bound_ratio,
                 field_sigma_ratio = cfg_obj.tmc_cluster.field_sigma_ratio)

    all_metrics_list = []
    thre_dist = cfg_obj.tmc_cluster.thre_dist

    error_list = []

    for batch in test_loader:

        # filename = batch.filename[0] if isinstance(batch.filename, list) else batch.filename
        # if filename not in ["1065_L_Ori"]:
        #     continue

        data = tmc(batch[0])

        try:
            if not hasattr(data, 'tooth_sizes'):
                data.tooth_sizes = ClusterEval.compute_tooth_sizes(data.pos, data.y, data.instance_labels > 0)

            # ========== 1. TMC聚类评估 ==========
            clu_metrics = ClusterEval.compute_detection_metrics(data.instance_centroids,
                                                                data.tmc_proposal['cluster_centroids'],
                                                                data.tooth_sizes,
                                                                thre_dist=thre_dist)
            cluster_cents = data.tmc_proposal['cluster_centroids']
            # ========== 2. 直接用y预测聚类结果 ==========
            with torch.no_grad():
                pred_data = tmc._seg_predictor(batch)

            seg_results = pred_data.outputs['seg_result']
            per_pred_clu_count, pred_cents = ClusterEval.extract_cluster_by_label(pred_data.pos, seg_results)
            pred_metrics = ClusterEval.compute_detection_metrics(data.instance_centroids.to(pred_cents.device),
                                                                 pred_cents,
                                                                 data.tooth_sizes,
                                                                 thre_dist=thre_dist)

            # ========== 3. 合并质心（以cluster为主）==========
            merged_cents, match_info = ClusterEval.merge_centroids_hungarian(
                cluster_cents=data.tmc_proposal['cluster_centroids'],
                pred_cents=pred_cents,
                distance_threshold=thre_dist
            )

            # 确保类型为 Float32
            if isinstance(merged_cents, np.ndarray):
                merged_cents = torch.from_numpy(merged_cents).float().to(data.instance_centroids.device)
            elif isinstance(merged_cents, torch.Tensor):
                merged_cents = merged_cents.float().to(data.instance_centroids.device)

            # 使用合并后的质心进行评估
            merged_metrics = ClusterEval.compute_detection_metrics(
                data.instance_centroids,
                merged_cents,
                data.tooth_sizes,
                thre_dist=thre_dist
            )

            # ========== 4. 记录指标 ==========
            clu_metrics['name'] = data.filename[0] if isinstance(data.filename, list) else data.filename

            ordered_metrics = OrderedDict([
                ('01_name', clu_metrics['name']),
                # TMC聚类指标
                ('03_TMC_TP', clu_metrics['TP']),
                ('04_TMC_FP', clu_metrics['FP']),
                ('05_TMC_FN', clu_metrics['FN']),
                ('06_TMC_precision', clu_metrics['precision']),
                ('07_TMC_recall', clu_metrics['recall']),
                ('08_TMC_f1', clu_metrics['f1']),
                ('09_TMC_cd', clu_metrics['cd']),
                # Pred分割指标
                ('13_Pred_TP', pred_metrics['TP']),
                ('14_Pred_FP', pred_metrics['FP']),
                ('15_Pred_FN', pred_metrics['FN']),
                ('16_Pred_precision', pred_metrics['precision']),
                ('17_Pred_recall', pred_metrics['recall']),
                ('18_Pred_f1', pred_metrics['f1']),
                ('19_Pred_cd', pred_metrics['cd']),
                # 合并后指标
                ('23_Merged_TP', merged_metrics['TP']),
                ('24_Merged_FP', merged_metrics['FP']),
                ('25_Merged_FN', merged_metrics['FN']),
                ('26_Merged_precision', merged_metrics['precision']),
                ('27_Merged_recall', merged_metrics['recall']),
                ('28_Merged_f1', merged_metrics['f1']),
                ('29_Merged_cd', merged_metrics['cd']),
                # 匹配信息
                ('30_n_matched', match_info['n_matched']),
                ('31_n_unmatched_cluster', match_info['n_unmatched_cluster']),
                ('32_n_unmatched_pred', match_info['n_unmatched_pred']),
            ])

            logger.log_metrics(ordered_metrics)

            # ========== 5. 可视化 ==========
            f2q_idx, _ = NNPointSamplingIndexer()(data.pos.cpu(), data.data_qslim.pos.cpu())

            filename = data.filename[0] if isinstance(data.filename, list) else data.filename

            # 保存logits field可视化
            save_lf_path = os.path.join(cfg_obj.paths.images_path, version_name, 'lf')
            save_path = os.path.join(save_lf_path, filename + '.png')
            if not exists(save_lf_path):
                os.makedirs(save_lf_path)

            # TMCPlotter.plot_lf(data.data_qslim, data.tmc_proposal,
            #                    save_path=save_path, window_size=[int(1920 * 1.5), int(1080 * 1.5)])

            # 保存mask可视化
            save_mask_path = os.path.join(cfg_obj.paths.images_path, version_name, 'mask')
            if not exists(save_mask_path):
                os.makedirs(save_mask_path)
            save_path = os.path.join(save_mask_path, filename + '.png')

            # TMCPlotter.plot_mask_meshes(data.data_qslim, data.tmc_proposal,
            #                             save_path=save_path, window_size=[int(1920 * 1.5), int(1080 * 1.5)])

            # 保存质心对比可视化
            if hasattr(data.data_qslim, 'face') and data.data_qslim.face is not None:
                try:
                    centroid_save_dir = os.path.join(cfg_obj.paths.images_path, version_name, 'centroids')
                    os.makedirs(centroid_save_dir, exist_ok=True)
                    merged_centroid_save_path = os.path.join(centroid_save_dir, f"{filename}_merged_centroids.png")

                    pred_classes = torch.argmax(seg_results, dim=-1).squeeze()
                    pred_centroid_save_path = os.path.join(centroid_save_dir, f"{filename}_pred_centroids.png")
                    # TMCPlotter.visualize_mesh_and_centroids_comparison(
                    #     gt_centroids=data.instance_centroids[1:, :],
                    #     pred_centroids=pred_cents[1:, :].to(data.instance_centroids.device),
                    #     cluster_centroids=cluster_cents[1:, :].to(data.instance_centroids.device),
                    #     pos=data.data_qslim.pos,
                    #     face=data.data_qslim.face,
                    #     gt_labels=data.data_qslim.y,
                    #     pred_labels= pred_classes[f2q_idx.squeeze()],
                    #     save_path=pred_centroid_save_path,
                    #     window_size=[int(1920 * 1.5), int(1080 * 1.5)],
                    #     elevation=0,
                    #     azimuth=0)
                except Exception as e:
                    print(f"   ⚠️ 质心可视化失败: {e}")

            all_metrics_list.append(ordered_metrics)

            # 打印当前样本的详细指标
            print(f"\n{'='*70}")
            print(f"📊 样本: {filename}")
            print(f"{'='*70}")
            print(f"--- TMC聚类 ---")
            print(f"  TP={clu_metrics['TP']:2d}, FP={clu_metrics['FP']:2d}, FN={clu_metrics['FN']:2d}")
            print(f"  P={clu_metrics['precision']:.4f}, R={clu_metrics['recall']:.4f}, F1={clu_metrics['f1']:.4f}")
            print(f"--- Pred分割 ---")
            print(f"  TP={pred_metrics['TP']:2d}, FP={pred_metrics['FP']:2d}, FN={pred_metrics['FN']:2d}")
            print(f"  P={pred_metrics['precision']:.4f}, R={pred_metrics['recall']:.4f}, F1={pred_metrics['f1']:.4f}")
            print(f"--- 合并后 ---")
            print(f"  TP={merged_metrics['TP']:2d}, FP={merged_metrics['FP']:2d}, FN={merged_metrics['FN']:2d}")
            print(f"  P={merged_metrics['precision']:.4f}, R={merged_metrics['recall']:.4f}, F1={merged_metrics['f1']:.4f}")
            print(f"  CD={merged_metrics['cd']:.6f}")
            print(f"--- 匹配统计 ---")
            print(f"  匹配对数: {match_info['n_matched']}")
            print(f"  未匹配Cluster: {match_info['n_unmatched_cluster']}")
            print(f"  未匹配Pred: {match_info['n_unmatched_pred']}")
            print(f"{'='*70}\n")

        except Exception as e:
            error_list.append(data.filename)
            print(f"❌ Error in {data.filename}: {e}")
            import traceback
            traceback.print_exc()

    for error_file in error_list:
        print(f"Error in {error_file}")


    # ========== 计算平均指标 ==========
    numeric_keys = [key for key in ordered_metrics.keys() if key != '01_name']
    avg_metrics = {}

    print(f"\n{'='*70}")
    print(f"📊 整体统计（{len(all_metrics_list)} 个样本）")
    print(f"{'='*70}")

    for key in numeric_keys:
        values = [metrics[key] for metrics in all_metrics_list]
        avg_value = np.mean(values)
        avg_metrics[key] = avg_value

        # 分组打印
        if key.startswith('03_') or key.startswith('06_') or key.startswith('08_'):
            prefix = "TMC"
        elif key.startswith('13_') or key.startswith('16_') or key.startswith('18_'):
            prefix = "Pred"
        elif key.startswith('23_') or key.startswith('26_') or key.startswith('28_'):
            prefix = "Merged"
        else:
            prefix = ""

        if prefix:
            metric_name = key.split('_', 1)[1]
            print(f"  {prefix} {metric_name}: {avg_value:.4f}")

    # 创建平均指标的 OrderedDict
    avg_ordered_metrics = OrderedDict([
                                          ('01_name', 'AVERAGE'),
                                      ] + [(key, avg_metrics[key]) for key in numeric_keys])

    # 记录平均指标
    logger.log_metrics(avg_ordered_metrics)
    logger.save()

    print(f"{'='*70}\n")


def test_batch_CropBatchGenerator():
    print("Current working directory:", os.getcwd())

    config_path = "../configs"
    config_name = "module_TMC"
    cfg = get_yaml_config(config_path=config_path, config_name=config_name)
    args = cfg.paths
    ds_args = cfg.dataset
    print_yaml_cfg_dict(cfg)

    cfg_obj = instantiate_from_config(cfg)

    print(cfg_obj)

    version_name = generate_exp_name(cfg_obj.logger.exp_args)
    print(version_name)

    _, train_loader, test_loader =prepare_data(cfg_obj)

    print(train_loader, test_loader)


    # 创建TMC
    tmc = TopoMorphCluster(num_classes=cfg_obj.tmc_cluster.num_classes,
                 predictor_type=cfg_obj.tmc_cluster.predictor_type,
                 ckpt_path=cfg_obj.tmc_cluster.ckpt_path,
                 net_para_name=cfg_obj.tmc_cluster.net_para_name,
                 filter_type=cfg_obj.tmc_cluster.filter_type,  # bg, tae, tae_tte, tae_tte_gtte
                 mounted_segnet_outputs= True,
                 mounted_mask_outputs= True,
                 mounted_knn_ica_meshes= True,
                 aggr_type=cfg_obj.tmc_cluster.aggr_type,  # boolean, add
                 small_cluster_size_thre = cfg_obj.tmc_cluster.small_cluster_size_thre,
                 alpha = cfg_obj.tmc_cluster.alpha,
                 k = cfg_obj.tmc_cluster.k,
                 max_iter = cfg_obj.tmc_cluster.max_iter,
                 iter_thred_len_ratio = cfg_obj.tmc_cluster.iter_thred_len_ratio,
                 field_bound_ratio = cfg_obj.tmc_cluster.field_bound_ratio,
                 field_sigma_ratio = cfg_obj.tmc_cluster.field_sigma_ratio)

    is_train = False
    loader = train_loader if is_train else test_loader
    subflod = 'train' if is_train else 'test'

    for batch in loader:

        start = time.time()
        data = tmc(batch[0])
        end = time.time()
        print(f"tmc 耗时：{end - start:.4f}秒")

        # data = batch
        tmc_proposal = data.tmc_proposal
        print(tmc_proposal['cluster_centroids'].shape,
              tmc_proposal['cluster_labels'],
              tmc_proposal['per_cluster_count'],
              tmc_proposal['cluster_num'])

        gt_cent = data.instance_centroids
        pred_cent = tmc_proposal['cluster_centroids']

        # 注意：0 为背景类，不参与计算
        dist_matrix = torch.cdist(pred_cent[1:], gt_cent[1:],  p=2)
        sqd_argmin = torch.argmin(dist_matrix, dim=1) # 最近邻匹配，容易出现多个预测点对应同一个 gt 点的情况
        cluster_gt_seg_labels = data.instance_labels[sqd_argmin+1]  # 注意，0 为背景类，不参与计算
        pred_cent_wo_bg = pred_cent[1:]
        # dist_np = dist_matrix.cpu().numpy()
        # row_ind, col_ind = linear_sum_assignment(dist_np)  # 使用匈牙利算法进行最优一对一匹配
        # distance_threshold = 0.10  # 根据实际数据尺度调整， 只保留有效的匹配
        # valid_mask = dist_np[row_ind, col_ind] < distance_threshold
        # row_ind = row_ind[valid_mask]
        # col_ind = col_ind[valid_mask]
        # cluster_gt_seg_labels = data.instance_labels[col_ind + 1]  # row_ind 是预测点的索引，col_ind 是对应的 gt 点索引
        # pred_cent_wo_bg = pred_cent[row_ind + 1]

        center_points = pred_cent_wo_bg.reshape(1,-1, 3) # B, clu， 3
        l0_xyz = data.pos.reshape(1,-1, 3)                                     # B, N, 3
        feat = data.tmc_proposal['outputs']['cbl_inputs'][1] # 提取输入特征，BN,32              # BN, 32
        feat = feat.reshape(1,-1, 32)                                          # B, N, 32
        gt_labels = data.y.reshape(1,-1,1)
        nn_crop_indexes = ou.get_nearest_neighbor_idx(pvgu.torch_to_numpy(l0_xyz),
                                                    pvgu.torch_to_numpy(center_points), 2048)


        cropped_pos = ou.get_indexed_features(l0_xyz.permute(0, 2, 1), nn_crop_indexes) # clu, 3,  crop
        cropped_feats = ou.get_indexed_features(feat.permute(0, 2, 1), nn_crop_indexes) # clu, 32, crop
        # ddf = get_ddf(cropped_pos[:, :3, :].permute(0, 2, 1), center_points)  # B, 1, crop

        cropped_feats = torch.cat([cropped_pos[:, :3, :], cropped_feats], axis=1) # clu, 36, crop
        crop_gt_seg_label = ou.get_indexed_features(gt_labels.permute(0, 2, 1), nn_crop_indexes) # clu,1,crop

        # crop的二进制掩码
        crop_gt_binseg_label = torch.zeros_like(crop_gt_seg_label)
        for i in range(crop_gt_seg_label.shape[0]):
            mask = (crop_gt_seg_label[i, 0]==cluster_gt_seg_labels[i])
            crop_gt_binseg_label[i, 0, mask] = 1

        # 网格化排列 crops 点云
        crops = []
        xy_offset = 0.5  # x,y 轴方向平移步长
        # cols = int(np.ceil(np.sqrt(cropped_pos.shape[0])))  # 网格列数
        cols = 4  # 网格列数

        # 动态计算平移距离，避免重叠
        for i in range(cropped_pos.shape[0]):

            label_i = cluster_gt_seg_labels[i].cpu().numpy()
            j = label_i -1 # 注意，0 为背景类，不参与计算，因而cluster_gt_seg_labels序号从1开始
            j = j + 1 if j > 6 else j

            crop_pcd = pvgu.np_to_pcd_with_label(
                cropped_pos[i].cpu().T.numpy(),
                crop_gt_binseg_label[i].cpu().T.numpy() * label_i
            )

            # 1. 先减去中心点（归一化到原点）
            points = np.asarray(crop_pcd.points)
            center = np.mean(points, axis=0)
            # crop_pcd.translate(-center)
            crop_pcd.points = crop_pcd.points + np.array(-center)

            # 网格状排列：按行列计算 x,y 偏移
            row = j // cols - cols//2
            col = j % cols - cols//2
            translate_x = col * xy_offset
            translate_y = row * xy_offset
            crop_pcd.points = crop_pcd.points + np.array([translate_x, translate_y, 0])
            crops.append(crop_pcd)


        crop_pcd = pvgu.np_to_pcd_with_label(
            data.pos.cpu().numpy(),
            data.y.cpu().numpy()
        )
        crop_pcd.points = crop_pcd.points + np.array([-cols*xy_offset*1.2 ,0,0])
        crops.append(crop_pcd)

        # version_name = generate_exp_name(cfg_obj.logger.exp_args)
        # print(version_name)

        file_name = f"{data.filename}_{gt_cent.shape[0]-1}gt_{pred_cent_wo_bg.shape[0]}crop.png"
        save_path = os.path.join(cfg_obj.paths.images_path+'_crop', version_name, file_name)

        pvgu.print_3d_args(*crops,save_path=save_path, window_size=[int(3840*2), int(2160*2)])

def do_batch_save_Crops():
    print("Current working directory:", os.getcwd())

    config_path = "../configs"
    config_name = "module_TMC"
    cfg = get_yaml_config(config_path=config_path, config_name=config_name)
    args = cfg.paths
    ds_args = cfg.dataset
    print_yaml_cfg_dict(cfg)

    cfg_obj = instantiate_from_config(cfg)

    print(cfg_obj)

    version_name = generate_exp_name(cfg_obj.logger.exp_args, add_timestamp=False, only_vals=True)
    print(version_name)

    _, train_loader, test_loader =prepare_data(cfg_obj)

    print(train_loader, test_loader)


    # 创建TMC
    tmc = TopoMorphCluster(num_classes=cfg_obj.tmc_cluster.num_classes,
                 predictor_type=cfg_obj.tmc_cluster.predictor_type,
                 ckpt_path=cfg_obj.tmc_cluster.ckpt_path,
                 net_para_name=cfg_obj.tmc_cluster.net_para_name,
                 filter_type=cfg_obj.tmc_cluster.filter_type,  # bg, tae, tae_tte, tae_tte_gtte
                 mounted_segnet_outputs= cfg_obj.tmc_cluster.mounted_segnet_outputs,
                 mounted_mask_outputs=cfg_obj.tmc_cluster.mounted_mask_outputs,
                 mounted_knn_ica_meshes=cfg_obj.tmc_cluster.mounted_knn_ica_meshes,
                 aggr_type=cfg_obj.tmc_cluster.aggr_type,  # boolean, add
                 small_cluster_size_thre = cfg_obj.tmc_cluster.small_cluster_size_thre,
                 alpha = cfg_obj.tmc_cluster.alpha,
                 k = cfg_obj.tmc_cluster.k,
                 max_iter = cfg_obj.tmc_cluster.max_iter,
                 iter_thred_len_ratio = cfg_obj.tmc_cluster.iter_thred_len_ratio,
                 field_bound_ratio = cfg_obj.tmc_cluster.field_bound_ratio,
                 field_sigma_ratio = cfg_obj.tmc_cluster.field_sigma_ratio)

    is_train = False
    loader = train_loader if is_train else test_loader
    subflod = 'train' if is_train else 'test'


    # 测试样本需要同TMC创建聚类结果
    for batch in test_loader:

        # 判断是否已存在聚类结果
        file_name = f"{batch[0].filename}.pt"
        save_path = os.path.join(cfg_obj.paths.tmc_path, version_name,file_name)
        print("save path:", save_path)

        if os.path.exists(save_path):
            continue


        start = time.time()
        data = tmc(batch[0])
        end = time.time()
        print(f"tmc 耗时：{end - start:.4f}秒")

        # data = batch
        tmc_proposal = data.tmc_proposal
        print('cluster_centroids:', tmc_proposal['cluster_centroids'].shape,
              'cluster_labels:', tmc_proposal['cluster_labels'],
              'per_cluster_count:', tmc_proposal['per_cluster_count'],
              'cluster_num:', tmc_proposal['cluster_num'])

        if 'outputs' in tmc_proposal and tmc_proposal['outputs'] is not None:

            del tmc_proposal['outputs']['edge_seg_result']
            del tmc_proposal['outputs']['tte_seg_result']
            tmc_proposal['outputs']['cbl_inputs'][1] = None

            print('output_seg_result:', tmc_proposal['outputs']['seg_result'].shape,
                  # 'output_edge_seg_result:', tmc_proposal['outputs']['edge_seg_result'].shape,
                  # 'output_tte_seg_result:', tmc_proposal['outputs']['tte_seg_result'].shape,
                  'output_cbl_inputs_pos:', tmc_proposal['outputs']['cbl_inputs'][0].shape,
                  # 'output_cbl_inputs_feat:', c.shape,
                  'output_cbl_inputs_offset:', tmc_proposal['outputs']['cbl_inputs'][2].shape
                  )

        # 保存聚类结果
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 确保目录存在
        torch.save(tmc_proposal, save_path)

    # for batch in train_loader:
    #
    #     # 判断是否已存在聚类结果
    #     file_name = f"{batch[0].filename}.pt"
    #     save_path = os.path.join(cfg_obj.paths.tmc_path, version_name,file_name)
    #     print("save path:", save_path)
    #
    #     if os.path.exists(save_path):
    #         continue
    #
    #     # 训练数据集无需创建聚类结果
    #     pred_data = tmc._seg_predictor(batch[0])
    #     tmc_proposal = {}
    #     tmc_proposal['outputs'] = pred_data.outputs
    #
    #     if 'outputs' in tmc_proposal and tmc_proposal['outputs'] is not None:
    #         del tmc_proposal['outputs']['edge_seg_result']
    #         del tmc_proposal['outputs']['tte_seg_result']
    #         tmc_proposal['outputs']['cbl_inputs'][1] = None
    #
    #
    #     torch.save(tmc_proposal, save_path)


def test_load_tmc_proposal():
    print("Current working directory:", os.getcwd())

    config_path = "../configs"
    config_name = "test_TMC"
    cfg = get_yaml_config(config_path=config_path, config_name=config_name)
    args = cfg.paths
    ds_args = cfg.dataset
    print_yaml_cfg_dict(cfg)

    cfg_obj = instantiate_from_config(cfg)

    print(cfg_obj)

    props_loader = cfg_obj.cluster.load_tmc_props

    data = Data()
    data.filename = "3EIXXI7Z_upper"
    data = props_loader(data)
    tmc_proposal = data.tmc_proposal

    print('cluster_centroids:', tmc_proposal['cluster_centroids'].shape,
          'cluster_labels:', tmc_proposal['cluster_labels'],
          'per_cluster_count:', tmc_proposal['per_cluster_count'],
          'cluster_num:', tmc_proposal['cluster_num'])

    if 'outputs' in tmc_proposal and tmc_proposal['outputs'] is not None:
        print('output_seg_result:', tmc_proposal['outputs']['seg_result'].shape,
              'output_cbl_inputs_pos:', tmc_proposal['outputs']['cbl_inputs'][0].shape,
              'output_cbl_inputs_offset:', tmc_proposal['outputs']['cbl_inputs'][2].shape
              )

def merge_centroids():
    print("Current working directory:", os.getcwd())

    config_path = "../configs"
    config_name = "module_TMC"
    cfg = get_yaml_config(config_path=config_path, config_name=config_name)
    args = cfg.paths
    ds_args = cfg.dataset
    print_yaml_cfg_dict(cfg)

    cfg_obj = instantiate_from_config(cfg)

    # 1. read data
    reader = T3DSPlusSegReader(is_filter_small_instance=ds_args.is_filter_small_instance,
                               filter_small_instance_size=ds_args.filter_small_instance_size,
                               verbose=True)

    # 难分样例：ANLLPLV7， B5GFZIRW，
    # 01A6HAN6   0TMOBYXS JJEIMAPR、ameziani、00OMSZGW(姿态严重)， WRT7Y7P9，MGYSCSI9
    # EJWZZZRF, patient1 014F9HTN, 017APYC4 01M10NMV 01MAU84T 2CE5VDAE ANLLPLV7 B5GFZIRW DG27PDD4
    # IU0UV8E_upper OTKQ5J9_upper, CE5VDAE_upper, PLDXK7R_upper, CQZX40I5_upper,

    # 01KTRG9K_lower 0OF8OOCX_lower 1QI2XWMD_lower IKZNMT4Z_upper 01K17AN8_lower 4G9LHQ2X_lower
    # 特别案例 01K3G866_lower B5GFZIRW_lower FHA6N618_lower
    # 01M10NMV_lower 4J24X0ES_upper AVI68NCT_upper B5GFZIRW_lower CD81F1F4_upper CQZX40I5_upper
    # E552BK9X_upper  GNR6QR3P_lower TVSR5QBQ_lower U9UBDO05_upper VB7GYS2O_upper Y48DURWV_upper ZD89X7G1_lower

    # name = "01KTRG9K_lower"
    name = "01M6JVVN_upper"
    LU = name.split('_')[1]
    patentt_id = name.split('_')[0]
    obj_name = f"{LU}/{patentt_id}/{patentt_id}_{LU}.obj"
    json_name = f"{LU}/{patentt_id}/{patentt_id}_{LU}.json"

    obj_filename = os.path.join(args.source_obj_data_path, obj_name)
    seg_json_filename = os.path.join(args.source_seg_json_data_path, json_name)

    data = reader(obj_filename, seg_json_filename)
    print(data.keys())

    # 2. pre_transform
    pre_transform = instantiate_from_config(cfg.pre_transform_obj)
    # print(pre_transform)
    data = pre_transform(data)
    batch = Batch.from_data_list([data])

    save_lf_path = "tmp/logtis_field"
    # 3. TMC
    tmc = TopoMorphCluster(num_classes=cfg_obj.tmc_cluster.num_classes,
                 predictor_type=cfg_obj.tmc_cluster.predictor_type,
                 ckpt_path=cfg_obj.tmc_cluster.ckpt_path,
                 net_para_name=cfg_obj.tmc_cluster.net_para_name,
                 filter_type=cfg_obj.tmc_cluster.filter_type,  # bg, tae, tae_tte, tae_tte_gtte
                 mounted_segnet_outputs=True,
                 mounted_mask_outputs=True,
                 mounted_knn_ica_meshes=True,
                 save_lf_path=save_lf_path,
                 aggr_type=cfg_obj.tmc_cluster.aggr_type,  # boolean, add
                 small_cluster_size_thre = cfg_obj.tmc_cluster.small_cluster_size_thre,
                 alpha = cfg_obj.tmc_cluster.alpha,
                 k = cfg_obj.tmc_cluster.k,
                 max_iter = cfg_obj.tmc_cluster.max_iter,
                 iter_thred_len_ratio = cfg_obj.tmc_cluster.iter_thred_len_ratio,
                 field_bound_ratio = cfg_obj.tmc_cluster.field_bound_ratio,
                 field_sigma_ratio = cfg_obj.tmc_cluster.field_sigma_ratio)

    # TMC聚类
    data = tmc(batch[0])

    # print(data.tmc_proposal)

    # 聚类评估
    # eval_topo_morph_cluster(cluster_centroids, cluster_labels, data)
    print(data.instance_centroids.shape, data.instance_labels, data.instance_count)

    thre_dist = 0.08

    if not hasattr(data, 'tooth_sizes'):
        data.tooth_sizes = ClusterEval.compute_tooth_sizes(data.pos, data.y, data.instance_labels > 0)

    cluster_cents = data.tmc_proposal['cluster_centroids']
    # 聚类评估
    clu_metrics = ClusterEval.compute_detection_metrics(data.instance_centroids,
                                                        cluster_cents,
                                                        data.tooth_sizes,
                                                        thre_dist=thre_dist)

    print(clu_metrics)

    # 方法2：直接根据实例分割结果直接用y预测聚类结果
    batch = Batch.from_data_list([data])
    with torch.no_grad():
        pred_data = tmc._seg_predictor(batch)
        # pred_data = pred_data.cpu()

    seg_results = pred_data.outputs['seg_result']
    per_pred_clu_count, pred_cents = ClusterEval.extract_cluster_by_label(pred_data.pos, seg_results)

    pred_metrics = ClusterEval.compute_detection_metrics(data.instance_centroids.to(pred_cents.device),
                                                        pred_cents,
                                                        tooth_sizes=data.tooth_sizes,
                                                        thre_dist=thre_dist)

    print(per_pred_clu_count,pred_metrics)

    # 合并质心（以cluster为主）
    merged_cents, match_info = ClusterEval.merge_centroids_hungarian(
        cluster_cents=data.tmc_proposal['cluster_centroids'],
        pred_cents=pred_cents,
        distance_threshold=0.10
    )

    print(merged_cents, match_info)

    # 可视化质心对比
    try:
        centroid_save_dir = os.path.join("", "centroid_vis")
        os.makedirs(centroid_save_dir, exist_ok=True)
        filename = data.filename[0] if isinstance(data.filename, list) else data.filename
        centroid_save_path = os.path.join(centroid_save_dir, f"{filename}_centroids.png")

        TMCPlotter.visualize_centroids_comparison(
            gt_centroids=cluster_cents,
            pred_centroids=pred_cents,
            pos=data.data_qslim.pos,
            face=data.data_qslim.face,
            save_path=centroid_save_path
        )
    except Exception as e:
        print(f"⚠️ 质心可视化失败: {e}")


def main():
    eval_TMC()
    # eval_batch_TMC()
    # eval_batch_TMC_v2()
    # test_CropBatchGenerator()
    # test_batch_CropBatchGenerator()
    # do_batch_save_Crops()
    # test_load_tmc_proposal()
    # merge_centroids()

if __name__ == "__main__":
    main()


