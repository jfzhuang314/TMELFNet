from copy import deepcopy
from pathlib import Path
from typing import List, Tuple, Union, Any, Optional, Dict, Callable

import torch
from numpy._typing import NDArray, ArrayLike
from scipy.sparse import coo_matrix, triu
from sklearn.decomposition import PCA
from torch import Tensor
from torch_geometric.transforms import BaseTransform, FaceToEdge
import torch_geometric.transforms as T

from common.PredictorFactory import PredictorFactory
from datasets.misc.PreTransform import *
from torch_geometric.data import Data, Batch
from torch_scatter import scatter_mean, scatter_sum
import random

from sklearn.neighbors import NearestNeighbors
import numpy as np

from datasets.transform.PointSamplingIndexer import KNNPointSamplingIndexer, NNPointSamplingIndexer
from external_libs import gen_utils as gu
from torch_geometric.nn import radius_graph
from torch_scatter import scatter_max

from utils.points_utils import compute_polar_angles, compute_tae_gt_bin_labels_ang, compute_tte_gt_bin_labels, \
    compute_knn_average_distance, compute_knn_average_distance_multioutput
from torch.nn import functional as F


#region utils

class MapUtils:
    @staticmethod
    def seg_label48_to_label15(labels, jaw):
        """
        将48分类标签转换为15分类标签
        labels48从0~48，其中0~10，19,20，29,30，39,40，49,50，属于背景，11~28属于upper，31~48属于lower；
        由于upper和lower的标签是不连续的，所以需要根据jaw来进行转换；18,28,38,48属于第三磨牙，通常设为背景；
        :param labels: 48分类标签
        :param jaw: 牙位，'upper'或'lower'
        :return: 15分类标签
        """
        labels = (labels - 20) if jaw == 'lower' else labels
        labels[labels <= 10] = 0 # 小于等于10的全部置为0
        labels[np.isin(labels % 10, [0, 8, 9])]= 0  # 18,28,38,48全部置为0
        labels[labels // 10 == 1] %= 10  # 11~17全部置为1,2,3,4...
        labels[labels // 10 == 2] = (labels[labels // 10 == 2] % 10) + 7  # 21,22,23,24全部置为8,9,10,11
        # labels[labels < 0] = 0
        return labels

    @staticmethod
    def lm_labels_str_to_int(lm_labels):
        """
        将landmark类别字符串转换为整数
        :param lm_labels: landmark类别字符串数组
        :return: landmark类别整数数组
        """
        # 定义landmark类别, 共6类;从字符串到整数的映射。
        landmark_dict = {
            'Mesial': 0,
            'Distal': 1,
            'FacialPoint': 2,
            'OuterPoint': 3,
            'InnerPoint': 4,
            'Cusp': 5,
        }

        return np.array(landmark_dict[lm_labels])


class AdjustMeshVertsTransform(T.BaseTransform):
    """
    辅助类：用于调整网格顶点数的变换类。
    """
    def __init__(self, target_vertices=10000):
        self.target_vertices = target_vertices

    def __call__(self, data):
        """
        调用基类的 __call__ 方法进行浅拷贝，然后调用 forward 方法执行实际的变换操作。

        :param data: 输入的 3D 数据，包含 'mesh' 键
        :return: 经过变换后的数据
        """
        return super().__call__(data)

    @staticmethod
    def adjust_mesh_vertices(mesh, target_vertices=10000):
        """
            调整 TriangleMesh 格式网格的顶点数到指定数量。

            参数:
            mesh (open3d.geometry.TriangleMesh): 输入的网格。
            target_vertices (int): 目标顶点数，默认为 10000。

            返回:
            open3d.geometry.TriangleMesh: 调整顶点数后的网格。
            """
        # 第一步先降低到10000点以下
        current_vertices = len(mesh.vertices)
        if current_vertices == target_vertices:
            return mesh

        if current_vertices > target_vertices:
            # 顶点数多于目标值，进行简化操作，降低到目标顶点数以下
            target_triangles = target_vertices * 2
            simplified_mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangles)
            while len(simplified_mesh.vertices) >= target_vertices:
                target_triangles = max(target_triangles - 100, 100)
                simplified_mesh = simplified_mesh.simplify_quadric_decimation(
                    target_number_of_triangles=target_triangles)

            mesh = simplified_mesh

        # 第二步：选择面积最大的三角形进行细分，直到顶点数达到目标值
        current_vertices = len(mesh.vertices)
        if current_vertices == target_vertices:
            return mesh

        if current_vertices < target_vertices:
            # 顶点数少于目标值，选择面积最大的那些三角形进行细分
            triangles = np.asarray(mesh.triangles)
            vertices = np.asarray(mesh.vertices)
            triangle_areas = []
            for triangle in triangles:
                v1 = vertices[triangle[0]]
                v2 = vertices[triangle[1]]
                v3 = vertices[triangle[2]]
                area = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))
                triangle_areas.append(area)
            sorted_indices = np.argsort(triangle_areas)[::-1]

            subdivided_mesh = mesh
            while len(subdivided_mesh.vertices) < target_vertices:
                for index in sorted_indices:
                    vertex_indices = triangles[index]
                    v1 = vertices[vertex_indices[0]]
                    v2 = vertices[vertex_indices[1]]
                    v3 = vertices[vertex_indices[2]]
                    midpoint1 = (v1 + v2) / 2
                    midpoint2 = (v2 + v3) / 2
                    midpoint3 = (v3 + v1) / 2
                    new_vertices = np.vstack([vertices, midpoint1, midpoint2, midpoint3])
                    new_triangles = []
                    for i, triangle in enumerate(triangles):
                        if i == index:
                            new_triangles.extend([
                                [vertex_indices[0], len(vertices), len(vertices) + 2],
                                [len(vertices), vertex_indices[1], len(vertices) + 1],
                                [len(vertices) + 2, len(vertices) + 1, vertex_indices[2]],
                                [len(vertices), len(vertices) + 1, len(vertices) + 2]
                            ])
                        else:
                            new_triangles.append(triangle)
                    subdivided_mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
                    subdivided_mesh.triangles = o3d.utility.Vector3iVector(np.array(new_triangles))
                    vertices = new_vertices
                    triangles = np.array(new_triangles)
                    if len(subdivided_mesh.vertices) >= target_vertices:
                        break

            # 第三步：细分后顶点数多于目标值，处理超出部分
            num_excess_vertices = len(subdivided_mesh.vertices) - target_vertices
            if num_excess_vertices > 0:
                # 获取边界点
                # boundary_vertices = np.asarray(subdivided_mesh.get_boundary_vertices())
                boundary_vertices = MeshProc.get_boundary_points(subdivided_mesh)
                if len(boundary_vertices) == 0:
                    # 没有边界点，随机删除超出的顶点数
                    all_vertices = np.arange(len(subdivided_mesh.vertices))
                    vertices_to_remove = random.sample(list(all_vertices), num_excess_vertices)
                    subdivided_mesh.remove_vertices_by_index(vertices_to_remove)
                else:
                    # 有边界点，优先删除边界点
                    if len(boundary_vertices) >= num_excess_vertices:
                        vertices_to_remove = random.sample(list(boundary_vertices), num_excess_vertices)
                        subdivided_mesh.remove_vertices_by_index(vertices_to_remove)
                    else:
                        vertices_removed = len(boundary_vertices)
                        subdivided_mesh.remove_vertices_by_index(boundary_vertices)
                        # 构建邻接矩阵
                        adjacency_matrix = o3d.geometry.TriangleMesh.get_adjacency_matrix(subdivided_mesh)
                        k = 1
                        while vertices_removed < num_excess_vertices:
                            # 计算 k 阶邻域
                            k_hop_neighbors = set()
                            current_neighbors = set(boundary_vertices)
                            for _ in range(k):
                                new_neighbors = set()
                                for vertex in current_neighbors:
                                    for neighbor in np.where(adjacency_matrix[vertex])[0]:
                                        new_neighbors.add(neighbor)
                                current_neighbors = new_neighbors - current_neighbors
                                k_hop_neighbors.update(current_neighbors)
                            k_hop_neighbors = list(k_hop_neighbors)
                            if len(k_hop_neighbors) >= num_excess_vertices - vertices_removed:
                                vertices_to_remove = random.sample(k_hop_neighbors,
                                                                   num_excess_vertices - vertices_removed)
                                subdivided_mesh.remove_vertices_by_index(vertices_to_remove)
                                vertices_removed = num_excess_vertices
                            else:
                                vertices_removed += len(k_hop_neighbors)
                                subdivided_mesh.remove_vertices_by_index(k_hop_neighbors)
                            k += 1

            return subdivided_mesh

    def forward(self, mesh):
        """
        调用 adjust_mesh_vertices 函数调整网格的顶点数。

        :param data: 输入的 3D 数据，包含 'mesh' 键
        :return: 经过变换后的数据
        """
        # org_mesh = data
        if mesh is None:
            return mesh

        mesh = mesh.remove_degenerate_triangles()
        mesh = mesh.remove_duplicated_vertices()
        mesh = mesh.remove_duplicated_triangles()

        mesh = AdjustMeshVertsTransform.adjust_mesh_vertices(mesh, self.target_vertices)
        return mesh


class TriMeshToPy(T.BaseTransform):
    """
    辅助类：将 TriangleMesh 转换为 PyTorch Geometric 的 Data 对象，主要是edge_index。
    """
    def __init__(self, compute_vertex_normals=True,
                 compute_vertex_colors=False,
                 compute_adjacency_list=True,
                 compute_triangle_normals=False,
                 compute_triangle_adjacency=False,
                 compute_triangle_uvs = False
                 ):
        self.comp_vertex_normals = compute_vertex_normals
        self.comp_vertex_colors = compute_vertex_colors
        self.comp_adjacency_list = compute_adjacency_list
        self.comp_triangle_normals = compute_triangle_normals
        self.comp_triangle_adjacency = compute_triangle_adjacency
        self.comp_triangle_uvs = compute_triangle_uvs

    def __call__(self, mesh):
        """
        调用基类的 __call__ 方法进行浅拷贝，然后调用 forward 方法执行实际的变换操作。
        :param mesh: 输入的 TriangleMesh 对象
        :return: 经过变换后的 PyTorch Geometric 的 Data 对象
        """
        return super().__call__(mesh)

    @staticmethod
    def convert_adjacency_list(mesh):
        """
        将 TriangleMesh 的邻接列表转换为 2xM 形式的边索引矩阵。
        :param mesh: 输入的 TriangleMesh 对象
        :return: 2xM 形式的边索引矩阵，其中 M 是边的数量
        """
        num_vertices = len(mesh.adjacency_list)
        # 平铺 mesh.adjacency_list
        all_neighbors = [neighbor for sublist in mesh.adjacency_list for neighbor in sublist]
        # 生成 row 列表
        rows = [i for i, sublist in enumerate(mesh.adjacency_list) for _ in range(len(sublist))]

        # 创建初始的稀疏矩阵，不考虑边的重复问题
        sparse_matrix = coo_matrix((np.ones(len(rows)), (rows, all_neighbors)), shape=(num_vertices, num_vertices))

        # 提取上三角矩阵来获取边
        upper_triangular = triu(sparse_matrix, k=1)
        edges = np.vstack((upper_triangular.row, upper_triangular.col))

        return edges
    def forward(self, mesh):
        """
        执行实际的变换操作，将 TriangleMesh 对象转换为 PyTorch Geometric 的 Data 对象。
        :param mesh: 输入的 TriangleMesh 对象
        :return: 经过变换后的 PyTorch Geometric 的 Data 对象
        """
        data = dict()
        # 获取顶点数据
        data['V'] = np.asarray(mesh.vertices)
        # 获取三角形数据
        data['F'] = np.asarray(mesh.triangles)
        # 计算顶点法向量
        if self.comp_vertex_normals:
            mesh.compute_vertex_normals()
            data['vNormal'] = np.asarray(mesh.vertex_normals)
        # # 计算边连接矩阵
        # if self.comp_adjacency_list:
        #     mesh.compute_adjacency_list()
        #     data['edge_index'] = TriMeshToPy.convert_adjacency_list(mesh)

        return data

#endregion utils

class ToTensorTuple(object):
    """将输入元组中的numpy数组或tensor转换为统一的tensor格式"""

    def __init__(self) -> None:
        # 构建类型映射表
        self.type_mapping = {
            # numpy类型 -> torch类型
            bool: torch.bool,
            np.bool_: torch.bool,
            int: torch.int64,
            np.int16: torch.int16,
            np.uint16: torch.int32,
            np.int32: torch.int32,
            np.int64: torch.int64,
            float: torch.float64,
            np.float32: torch.float32,
            np.float64: torch.float64
        }

    def __call__(
            self,
            *tuples: Tuple[Union[NDArray[Any], torch.Tensor]],  # Changed from TensorType to torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        result = []
        for t in tuples:
            if isinstance(t, torch.Tensor):
                # 如果是tensor直接追加
                result.append(t)
            elif isinstance(t, np.ndarray):
                # 获取最接近的映射类型
                dtype = t.dtype.type if hasattr(t.dtype, 'type') else type(t.item(0))
                torch_type = self.type_mapping.get(dtype)
                if torch_type is not None:
                    result.append(torch.tensor(t, dtype=torch_type))
                elif dtype == str:
                    result.append(t)
                else:
                    raise ValueError(f'Unsupported dtype: {dtype}')
        return tuple(result)

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'


class ToNumpyTuple(object):
    """将输入元组中的tensor或numpy数组转换为统一的numpy数组格式"""
    def __init__(self) -> None:
        # 构建类型映射表
        self.type_mapping = {
            # torch类型 -> numpy类型
            torch.bool: np.bool_,
            torch.uint8: np.uint8,
            torch.int16: np.int16,
            torch.int32: np.int32,
            torch.int64: np.int64,
            torch.float16: np.float16,
            torch.float32: np.float32,
            torch.float64: np.float64
        }

    def __call__(
            self,
            *tuples: Tuple[Union[NDArray[Any], torch.Tensor]],  # Changed from TensorType to torch.Tensor
    ) -> Tuple[NDArray[Any], ...]:
        result = []
        for t in tuples:
            if isinstance(t, np.ndarray):
                result.append(t)
            elif isinstance(t, torch.Tensor):
                # 获取最接近的映射类型
                torch_type = t.dtype
                numpy_type = self.type_mapping.get(torch_type)
                if numpy_type is not None:
                    result.append(t.numpy().astype(numpy_type))
                elif torch_type == str:
                    result.append(t)
                else:
                    raise ValueError(f'Unsupported dtype: {torch_type}')

        return tuple(result)

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'


class PreprocessBatch(BaseTransform):
    def __init__(
            self,

    ) -> None:
        super().__init__()


    def __call__(
            self,
            data: Data,
    ) -> Data:

        if hasattr(data, 'face'): del data.face
        if hasattr(data, 'edge_index'): del data.edge_index
        if hasattr(data, 'y_ori'): del data.y_ori
        if hasattr(data, 'c2f_idx'): del data.c2f_idx
        if hasattr(data, 'f2c_idx'): del data.f2c_idx
        if hasattr(data, 'data_qslim'): del data.data_qslim



        return data


class EnsureNumPoints(BaseTransform):
    """
    确保点云具有固定的顶点数。
    - 如果点数不足，通过随机重复采样补齐。
    - 如果点数过多，通过 FPS 算法下采样。
    """

    def __init__(self, num_points: int = 16000):
        super().__init__()
        self.num_points = num_points

    def __call__(self, data: Data) -> Data:
        current_points = data.pos.shape[0]

        # 情况1：点数不足，需要补齐 (Upsampling)
        if current_points < self.num_points:
            padding_count = self.num_points - current_points
            # 随机选择索引进行重复
            upsample_indices = torch.randint(0, current_points, (padding_count,), device=data.pos.device)

            # 补齐所有相关属性
            data.pos = torch.cat([data.pos, data.pos[upsample_indices]], dim=0)
            if hasattr(data, 'x'):
                data.x = torch.cat([data.x, data.x[upsample_indices]], dim=0)
            if hasattr(data, 'normals'):
                data.normals = torch.cat([data.normals, data.normals[upsample_indices]], dim=0)
            if hasattr(data, 'y'):
                data.y = torch.cat([data.y, data.y[upsample_indices]], dim=0)
            if hasattr(data, 'instances'):
                data.instances = torch.cat([data.instances, data.instances[upsample_indices]], dim=0)
            if hasattr(data, 'gt_edge_mask'):
                data.gt_edge_mask = torch.cat([data.gt_edge_mask, data.gt_edge_mask[upsample_indices]], dim=0)
            if hasattr(data, 'gt_tte_mask'):
                data.gt_tte_mask = torch.cat([data.gt_tte_mask, data.gt_tte_mask[upsample_indices]], dim=0)

        # 情况2：点数过多，需要下采样 (Downsampling)
        elif current_points > self.num_points:
            # 使用 FPS 采样获取索引 (假设 gu.resample_pcd 可用，或者使用 pointnet2_utils)
            # 这里为了通用性，我们使用简单的随机采样作为备选，或者调用你已有的 gu 工具
            import numpy as np
            pos_np = data.pos.cpu().numpy()
            # 简单起见，这里使用随机采样。如果需要严格的 FPS，请替换为 gu.resample_pcd 逻辑
            indices = np.random.choice(current_points, self.num_points, replace=False)
            indices = torch.from_numpy(indices).to(data.pos.device)

            data.pos = data.pos[indices]
            if hasattr(data, 'x'): data.x = data.x[indices]
            if hasattr(data, 'normals'): data.normals = data.normals[indices]
            if hasattr(data, 'y'): data.y = data.y[indices]
            if hasattr(data, 'instances'): data.instances = data.instances[indices]
            if hasattr(data, 'gt_edge_mask'): data.gt_edge_mask = data.gt_edge_mask[indices]
            if hasattr(data, 'gt_tte_mask'): data.gt_tte_mask = data.gt_tte_mask[indices]

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_points={self.num_points})'


class RelKnnDistFeature(BaseTransform):
    """
    计算每个顶点的相对KNN距离，并将其作为特征追加到 data.x 中。
    同时保留 rel_knn_dist 字段到 data 中，便于后续分析和可视化。

    相对KNN距离 = 每个点的KNN平均距离 / 全局平均KNN距离
    - 值 > 1: 该点比平均更稀疏
    - 值 < 1: 该点比平均更密集
    """

    def __init__(self, k=8, eps=1e-8) -> None:
        """
        Args:
            k: KNN的邻居数量，默认 8
            eps: 防止除以零的小值，默认 1e-8
        """
        super().__init__()
        self.k = k
        self.eps = eps

    def __call__(self, data: Data) -> Data:
        """
        计算相对KNN距离并添加到数据中

        Args:
            data: PyG Data对象，必须包含 pos 属性

        Returns:
            添加了 rel_knn_dist 特征的 Data 对象
        """
        pos = data.pos

        # 计算每个点的KNN平均距离和全局平均距离
        per_point_avg_distances, global_avg_distance = compute_knn_average_distance_multioutput(pos, self.k)

        # 计算相对KNN距离
        rel_knn_dist = per_point_avg_distances / (global_avg_distance + self.eps)

        # 将相对KNN距离追加到 data.x
        if hasattr(data, 'x') and data.x is not None and not hasattr(data, 'rel_knn_dist'):
            # 如果 x 已存在，追加为新特征
            rel_knn_dist_expanded = rel_knn_dist.unsqueeze(-1)  # [N, 1]
            data.x = torch.cat([data.x, rel_knn_dist_expanded], dim=-1)

            # 保留 rel_knn_dist 字段到 data 中，便于后续使用
            data.rel_knn_dist = rel_knn_dist
            data.avg_knn_dist = global_avg_distance

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(k={self.k}, eps={self.eps})'


class RandomZAxisRotate(BaseTransform):
    def __init__(
            self,
            max_degrees: float = 45,
            rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__()
        self.max_angle = max_degrees / 180 * np.pi
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(
            self,
            data: Data,
    ) -> Data:


        angle = self.rng.uniform(-self.max_angle, self.max_angle)
        cosval, sinval = np.cos(angle), np.sin(angle)

        R = torch.tensor([
            [cosval, -sinval, 0],
            [sinval, cosval, 0],
            [0, 0, 1],
        ], dtype=data.pos.dtype, device=data.pos.device)

        data.pos = data.pos @ R.T

        if hasattr(data, 'normals'):
            data.normals = data.normals @ R.T

        if hasattr(data, 'landmark_coords'):
            data.landmark_coords = data.landmark_coords @ R.T
        if hasattr(data, 'instance_centroids'):
            data.instance_centroids = data.instance_centroids @ R.T

        return data


class RandomScale(T.BaseTransform):

    def __init__(
            self,
            low: float = 0.95,
            high: float = 1.05,
            rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.low = low
        self.high = high
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(
            self,
            data: Data,
    ) -> Data:
        scale = self.rng.uniform(self.low, self.high)
        data.pos = data.pos * scale

        if hasattr(data, 'landmark_coords'):
            data.landmark_coords = data.landmark_coords * scale
        if hasattr(data, 'instance_centroids'):
            data.instance_centroids = data.instance_centroids * scale

        return data

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    low: {self.low},',
            f'    high: {self.high},',
            ')',
        ])


class YMinMaxNormalize(T.BaseTransform):
    """
    对Y轴坐标进行归一化处理，将其映射到[-1, 1]范围。该归一化方法采用TGN的归一化方法，具体为：
    (y - y_min) / (y_max - y_min) * 2 - 1；我们在实验过程中发现，如果z轴减去z轴的最大值，可以适当提升性能。
    """

    def __init__(
            self,
            y_axis_min: float = -36.9843781139949,
            y_axis_max: float = 33.15232091532151,
    ) -> None:
        self.y_axis_min = y_axis_min
        self.y_axis_max = y_axis_max
        self.scale_factor = 2.0 / (y_axis_max - y_axis_min)

    def __call__(
            self,
            data: Data,
    ) -> Data:
        # 获取点云坐标
        points = data.pos  # 使用pos属性
        # 减去均值
        mean_offset = torch.mean(points, dim=0)
        points_centered = points - mean_offset
        # 应用Y轴范围归一化
        points_normalized = torch.clone(points_centered)
        # 使用torch操作替代numpy操作
        y_range = self.y_axis_max - self.y_axis_min
        points_normalized[:, :3] = ((points_centered[:, :3] - self.y_axis_min) / y_range) * 2 - 1

        # Z轴减去最大值
        z_max = torch.max(points_normalized[:, 2])
        points_normalized[:, 2] = points_normalized[:, 2] - z_max

        # 更新pos
        data.pos = points_normalized

        # 如果存在landmark_coords，也进行相同的变换
        if hasattr(data, 'landmark_coords'):
            landmarks = data.landmark_coords

            # 应用相同的中心化
            landmarks_centered = landmarks - mean_offset
            # 应用相同的归一化
            landmarks_normalized = torch.clone(landmarks_centered)
            landmarks_normalized[:, :3] = ((landmarks_centered[:, :3] - self.y_axis_min) / y_range) * 2 - 1
            # Z轴减去最大值（使用点云的z_max，保持一致性）
            landmarks_normalized[:, 2] = landmarks_normalized[:, 2] - z_max

            data.landmark_coords = landmarks_normalized

        return data

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    y_axis_min={self.y_axis_min},',
            f'    y_axis_max={self.y_axis_max},',
            ')',
        ])


class MeanScaleNormalize(T.BaseTransform):
    """
    对Y轴坐标进行归一化处理，将其映射到[-1, 1]范围。该归一化方法采用TGN的归一化方法，具体为：
    xyz / scale；我们在preprocess实验中对该归一化进行了测试。我们在实验过程中发现，如果z轴减去z轴的最大值，可以适当提升性能。
    """

    def __init__(
            self,
            scale: float = 60.0,
            zflip: bool = False,
            subzmax: bool = True,  # 是否减去z轴的最大值
    ) -> None:
        self.scale = scale
        self.scale_factor = 1.0 / scale
        self.zflip = zflip
        self.subzmax = subzmax

    def __call__(
            self,
            data: Data,
    ) -> Data:
        # 获取点云坐标
        points = data.pos  # 使用pos属性
        # 减去均值
        center = torch.mean(points, dim=0)
        points_centered = points - center
        # 应用Y轴范围归一化
        points_normalized = torch.clone(points_centered)
        # 使用torch操作替代numpy操作
        points_normalized[:, :3] = points_centered[:, :3] * self.scale_factor

        # 对于TADPM数据集，lower jaw需要z轴翻转
        should_flip = self.zflip and data.jaw == 'lower'
        points_normalized[:, 2] = center[2] - points_normalized[:, 2] if should_flip else points_normalized[:, 2]

        # Z轴减去最大值
        z_max = torch.max(points_normalized[:, 2])
        points_normalized[:, 2] = points_normalized[:, 2] - z_max if self.subzmax else points_normalized[:, 2]

        # 更新pos
        data.pos = points_normalized

        # 如果存在landmark_coords，也进行相同的变换
        if hasattr(data, 'landmark_coords'):
            landmarks = data.landmark_coords

            # 应用相同的中心化
            landmarks_centered = landmarks - center
            # 应用相同的归一化
            landmarks_normalized = torch.clone(landmarks_centered)
            landmarks_normalized[:, :3] = landmarks_centered[:, :3] * self.scale_factor

            landmarks_normalized[:, 2] = center[2] - landmarks_normalized[:, 2] if should_flip else landmarks_normalized[:, 2]
            # Z轴减去最大值（使用点云的z_max，保持一致性）
            landmarks_normalized[:, 2] = landmarks_normalized[:, 2] - z_max if self.subzmax else landmarks_normalized[:, 2]

            data.landmark_coords = landmarks_normalized

        return data

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    scale={self.scale_factor},',
            f'    zflip={self.zflip},',
            ')',
        ])


class PoseNormalize(T.BaseTransform):
    """
    对点云和法线进行归一化处理，将其映射到主坐标轴上。该归一化方法采用TGN的归一化方法，具体为：
    1. 对点云进行PCA分析，得到主坐标轴；
    2. 如果主坐标轴的z轴方向与法线的z轴方向相反，将主坐标轴绕y轴旋转180度；
    3. 如果点云在主坐标轴上的z轴方向与法线的z轴方向相反，将主坐标轴绕y轴旋转180度；
    4. 将点云和法线分别绕主坐标轴旋转，使它们映射到主坐标轴上。
    """
    def __call__(
            self,
            data: Data,
    ) -> Data:
        points = data.pos
        normals = data.normals

        # 将tensor转换为numpy以便进行PCA计算
        points_np = points.detach().cpu().numpy()
        normals_np = normals.detach().cpu().numpy()

        pca = PCA()
        pca.fit(points_np)
        R = pca.components_

        # disallow reflections
        if np.linalg.det(R) < 0:
            R = np.array([
                [-1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]) @ R

        # rotate 180 degrees around y-axis if points are upside down
        if (normals_np @ R.T)[:, 2].mean() < 0:
            R = np.array([
                [-1, 0, 0],
                [0, 1, 0],
                [0, 0, -1],
            ]) @ R

        # 将旋转矩阵转换为tensor
        R_tensor = torch.from_numpy(R).to(points.dtype).to(points.device)

        # rotate points to principal axes of decreasing explained variance
        data.pos = points @ R_tensor.T
        data.normals = normals @ R_tensor.T

        if hasattr(data, 'landmark_coords'):
            data.landmark_coords = data.landmark_coords @ R_tensor.T

        # 创建变换矩阵affine
        T = torch.eye(4, dtype=points.dtype, device=points.device)
        T[:3, :3] = R_tensor

        # 更新affine变换矩阵
        if hasattr(data, 'affine'):
            data.affine = T @ data.affine
        else:
            data.affine = T

        data.x = torch.cat([data.pos, data.normals], dim=-1)

        return data

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'

class TadpmPoseNormalize(T.BaseTransform):
    """
    为TADPM数据集专门设计的姿态调整方法；其具体步骤如下：
    从data.filename提取patient_id,jaw。一个典型的filename为：0001_L_Ori.stl；
    1. 对于patient_id小于等于227的样本，lower jaw需要z轴翻转，即data.pos[:, 2] = center[2] - data.pos[:, 2]；
    2. 对于patient_id大于227的样本
    （1）upper jaw需要z轴翻转，即data.pos[:, 2] = center[2] - data.pos[:, 2]；
    （2）无论是lower还是upper，都要围绕z轴旋转180度，即data.pos = torch.rotate_around_z(data.pos, 180)；
    """
    def __call__(self, data):
        """
        对TADPM数据集进行姿态调整
        Args:
            data: 包含pos和filename属性的数据对象
        Returns:
            调整后的数据对象
        """
        # 从filename提取patient_id和jaw信息
        patient_id, jaw = self._extract_patient_info(data.filename)

        points = data.pos
        normals = data.normals

        R = np.eye(3)
        Ry_180 = np.array([
                    [-1, 0, 0],
                    [0, 1, 0],
                    [0, 0, -1],
                ])
        Rz_180 = np.array([
                    [-1, 0, 0],
                    [0, -1, 0],
                    [0, 0, 1],
                ])

        if patient_id <= 277:
            R = Ry_180 @ R if jaw == 'lower' else R  # lower jaw需要z轴翻转
        else:
            R = Ry_180 @ R if jaw == 'upper' else R  # upper jaw需要z轴翻转
            R = Rz_180 @ R # 无论lower还是upper，都需要绕z轴旋转180度

        # 将旋转矩阵转换为tensor
        R_tensor = torch.from_numpy(R).to(points.dtype).to(points.device)

        # rotate points to principal axes of decreasing explained variance
        data.pos = points @ R_tensor.T
        data.normals = normals @ R_tensor.T

        if hasattr(data, 'landmark_coords'):
            data.landmark_coords = data.landmark_coords @ R_tensor.T

        # 创建变换矩阵affine
        T = torch.eye(4, dtype=points.dtype, device=points.device)
        T[:3, :3] = R_tensor

        # 更新affine变换矩阵
        if hasattr(data, 'affine'):
            data.affine = T @ data.affine
        else:
            data.affine = T

        # 重新标准化
        normalizator = MeanScaleNormalize(scale=1.0, zflip=False, subzmax=True)
        data = normalizator(data)

        return data

    def _extract_patient_info(self, filename):
        """
        从文件名提取patient_id和jaw信息
        Args:
            filename: 文件名，如 "0001_L_Ori.stl"
        Returns:
            tuple: (patient_id, jaw_type)
        """
        filename = Path(filename).name
        parts = filename.split('_')

        if len(parts) >= 3:
            patient_id = int(parts[0])  # 提取数字部分
            jaw_indicator = parts[1]  # 提取L或U

            jaw = 'lower' if jaw_indicator == 'L' else 'upper'
            return patient_id, jaw
        else:
            raise ValueError(f"无法解析文件名格式: {filename}")

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'


class XoYPCAPoseNormalize:
    """基于XOY平面PCA的姿态归一化类"""

    def __call__(self, data):
        """
        对点云数据进行姿态归一化处理
        Args:
            data: 包含点云数据的对象，需有pos属性
        Returns:
            处理后的数据对象
        """
        # 提取点云数据
        points = data.pos
        normals = data.normals

        # if hasattr(points, 'detach'):
        #     points_np = points.detach().cpu().numpy()
        # else:
        #     points_np = np.array(points)
        points_np = points.detach().cpu().numpy()
        normals_np = normals.detach().cpu().numpy()


        # 执行PCA姿态归一化
        R = self._get_rot_mat(points_np,normals_np)



        # 将旋转矩阵转换为tensor
        R_tensor = torch.from_numpy(R).to(points.dtype).to(points.device)


        data.pos = points @ R_tensor.T
        data.normals = normals @ R_tensor.T

        if hasattr(data, 'landmark_coords'):
            data.landmark_coords = data.landmark_coords @ R_tensor.T

        # 创建变换矩阵affine
        T = torch.eye(4, dtype=points.dtype, device=points.device)
        T[:3, :3] = R_tensor

        # 更新affine变换矩阵
        if hasattr(data, 'affine'):
            data.affine = T @ data.affine
        else:
            data.affine = T

        # 重新标准化
        normalizator = MeanScaleNormalize(scale=1.0, zflip=False, subzmax=True)
        data = normalizator(data)

        data.x = torch.cat([data.pos, data.normals], dim=-1)

        return data

    def _get_rot_mat(self, points_np, normals_np):
        """执行PCA姿态归一化"""
        # 1. 学习XOY平面的主成分
        points_2d = points_np[:, :2]  # 提取XOY平面坐标
        pca_2d = PCA(n_components=2)
        pca_2d.fit(points_2d)

        # 2. 构造3D主成分矩阵
        rotation_matrix = np.zeros((3, 3))
        rotation_matrix[:2, :2] = pca_2d.components_  # XOY平面主成分
        rotation_matrix[2, 2] = 1.0  # Z轴分量

        # rotation_matrix = np.eye(3)
        # 3. 构造180度绕Z轴旋转矩阵
        Rz_180 = np.array([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1],
        ])

        Rz_270 = np.array([
            [0, 1, 0],
            [-1, 0, 0],
            [0, 0, 1],
        ])

        Ry_180 = np.array([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, -1],
        ])

        pca = PCA()
        pca.fit(points_np)
        R_3d = pca.components_

        # 检查并修正z轴方向
        if self._should_flip_z_axis(normals_np, R_3d):
            print("需要180°绕Y轴旋转")
            rotation_matrix = Ry_180 @ rotation_matrix  # 180°绕Y轴旋转

        # 通常在绝大多数情况下，det为正；在少数情况下，det为负，这时需要将主成分矩阵进行成分交换；
        # 此时，开口形状将朝向左右。为了让开口形状向上，需要将整体形状旋转±90°。后续再通过检测开口方向进行调整。左右
        if np.linalg.det(rotation_matrix) < 0:
            rotation_matrix[:2, :2] = pca_2d.components_[::-1, :]
            rotation_matrix = Rz_270 @ rotation_matrix

        # 4. 应用变换到原始点云
        centroid = np.mean(points_np, axis=0)
        points_centered = points_np - centroid

        # 5. 应用旋转变换
        points_transformed = points_centered @ rotation_matrix.T

        # 6. 检查曲线开口方向并调整
        if not self._is_curve_opening_up(points_transformed):
            rotation_matrix = Rz_180 @ rotation_matrix       # 先pca后旋转z


        return rotation_matrix

    def _is_curve_opening_up(self, points_transformed):
        """
        判断XOY平面内曲线是否开口向上
        Args:
            points_transformed: 变换后的3D点云 (N, 3)
        Returns:
            bool: True表示开口向上，False表示开口向下
        """
        # 提取XOY平面坐标
        x = points_transformed[:, 0]
        y = points_transformed[:, 1]

        # 找到数据的主要延伸方向
        cov_matrix = np.cov(x, y)
        eigenvals, eigenvecs = np.linalg.eig(cov_matrix)

        # 选择主要延伸方向
        main_direction = eigenvecs[:, np.argmax(eigenvals)]

        # 将点投影到主轴方向作为自变量
        t = x * main_direction[0] + y * main_direction[1]

        # 垂直方向作为因变量
        perpendicular_direction = np.array([-main_direction[1], main_direction[0]])
        z = x * perpendicular_direction[0] + y * perpendicular_direction[1]

        # 拟合二次关系: z = a*t² + b*t + c
        A = np.column_stack([
            t**2,
            t,
            np.ones_like(t)
        ])

        coeffs, residuals, rank, s = np.linalg.lstsq(A, z, rcond=None)
        a = coeffs[0]  # 二次项系数

        # 返回布尔值：True表示开口向上(a>0)，False表示开口向下(a<0)
        return a > 0

    def _should_flip_z_axis(self, normals_np, R):
        """
        判断是否需要翻转Z轴方向
        """
        third_component = R[2]
        dot_products = np.dot(normals_np, third_component)
        return np.mean(dot_products) < 0

    def __repr__(self):
        return self.__class__.__name__ + '()'




# 未经测试，谨慎使用
class ZScoreNormalize(T.BaseTransform):
    def __init__(
            self,
            mean: Optional[Union[float, ArrayLike]] = None,
            std: Optional[Union[float, ArrayLike]] = None,
    ) -> None:
        super().__init__()
        self.mean_ = mean
        self.std_ = std

    def mean(
            self,
            pos: torch.Tensor,
    ) -> Union[float, torch.Tensor]:
        if self.mean_ is None:
            return pos.mean(dim=0)

        return torch.tensor(self.mean_) if isinstance(self.mean_, (int, float, list, np.ndarray)) else self.mean_

    def std(
            self,
            pos: torch.Tensor,
    ) -> Union[float, torch.Tensor]:
        if self.std_ is None:
            return pos.std(dim=0)

        return torch.tensor(self.std_) if isinstance(self.std_, (int, float, list, np.ndarray)) else self.std_

    def __call__(
            self,
            data: Data,
    ) -> Data:
        # 对 pos 属性进行 Z-Score 归一化
        if hasattr(data, 'pos') and data.pos is not None:
            pos = data.pos
            mean = self.mean(pos)
            std = self.std(pos)
            # 避免除以零
            std = torch.where(std == 0, torch.ones_like(std), std)

            # 更新 pos
            data.pos = (pos - mean) / std

            # 如果存在 affine 属性，需要更新 affine 变换
            if hasattr(data, 'affine') and data.affine is not None:
                # 创建仿射变换矩阵，先平移后缩放
                affine_translation = torch.eye(4, dtype=data.affine.dtype, device=data.affine.device)
                affine_translation[:3, 3] = -mean

                affine_scale = torch.eye(4, dtype=data.affine.dtype, device=data.affine.device)
                affine_scale[0, 0] = 1.0 / std[0]
                affine_scale[1, 1] = 1.0 / std[1]
                affine_scale[2, 2] = 1.0 / std[2]

                # 合并仿射变换
                new_affine = affine_scale @ affine_translation @ data.affine
                data.affine = new_affine

        # 如果存在 landmark_coord 属性，也需要归一化
        if hasattr(data, 'landmark_coord') and data.landmark_coord is not None:
            landmark_coord = data.landmark_coord
            mean = self.mean(landmark_coord)
            std = self.std(landmark_coord)
            # 避免除以零
            std = torch.where(std == 0, torch.ones_like(std), std)
            data.landmark_coord = (landmark_coord - mean) / std

        return data

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    mean={self.mean_},',
            f'    std={self.std_},',
            ')',
        ])


class XYZMinMaxNormalize(T.BaseTransform):
    """
    对XYZ轴坐标进行归一化处理，使所有点相对于质心居中，并缩放到单位球内。
    使用 pc_normalize 方法进行标准化：减去质心，然后除以最大距离。
    """

    def __init__(self) -> None:
        pass

    def __call__(
            self,
            data: Data,
    ) -> Data:
        # 获取点云坐标
        points = data.pos  # 使用pos属性

        # 计算质心并居中
        centroid = torch.mean(points, dim=0)
        points_centered = points - centroid

        # 计算每个点到原点（居中后）的距离
        distances = torch.sqrt(torch.sum(points_centered ** 2, dim=1))

        # 找到最大距离
        max_distance = torch.max(distances)

        # 避免除以零
        if max_distance > 0:
            # 归一化到单位球内
            points_normalized = points_centered / max_distance
        else:
            points_normalized = points_centered

        # 更新pos
        data.pos = points_normalized

        # 如果存在landmark_coords，也进行相同的变换
        if hasattr(data, 'landmark_coords'):
            landmarks = data.landmark_coords
            # 应用相同的质心偏移
            landmarks_centered = landmarks - centroid
            # 应用相同的缩放
            if max_distance > 0:
                landmarks_normalized = landmarks_centered / max_distance
            else:
                landmarks_normalized = landmarks_centered
            data.landmark_coords = landmarks_normalized

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

class InstanceCentroids(T.BaseTransform):
    """
    returns：
        instance_centroids: 实例质心坐标，包含背景0
        instance_labels: 实例标签，包含背景0
        instance_count: 实例数量，包含背景0
    """
    def __call__(
            self,
            data: Data,
    ) -> Data:
        # no-op if ground-truth data is unavailable
        if not hasattr(data, 'y') or 'instances' not in data:
            return data

        points = data.pos
        labels = data.y
        instances = data.instances

        instance_centroids = scatter_mean(
            src=points,
            index=labels,
            dim=0,
        )

        instance_labels = scatter_max(
            src=labels,
            index=labels,
            dim=0,
        )[0]

        per_instance_count = scatter_sum(
            src=torch.ones_like(labels),
            index=labels,
            dim=0,
        )

        valid_mask = (per_instance_count > 0).squeeze()

        valid_mask = (per_instance_count > 0).squeeze()

        # ========== 确保背景类（索引0）存在 ==========
        # 先过滤出有效实例
        valid_centroids = instance_centroids[valid_mask]
        valid_labels = instance_labels[valid_mask]
        valid_counts = per_instance_count[valid_mask]

        # 检查背景类（标签0）是否在有效实例中
        has_background = torch.any(valid_labels == 0).item()

        if not has_background:
            # 背景类不存在，需要在最前面插入零向量
            zero_centroid = torch.zeros((1, points.shape[1]), dtype=points.dtype, device=points.device)
            zero_label = torch.tensor([0], dtype=valid_labels.dtype, device=valid_labels.device)
            zero_count = torch.tensor([0], dtype=valid_counts.dtype, device=valid_counts.device)

            # 在开头插入背景类的零向量
            instance_centroids = torch.cat([zero_centroid, valid_centroids], dim=0)
            instance_labels = torch.cat([zero_label, valid_labels], dim=0)
            per_instance_count = torch.cat([zero_count, valid_counts], dim=0)
        else:
            # 背景类存在，直接使用过滤后的结果
            instance_centroids = valid_centroids
            instance_labels = valid_labels
            per_instance_count = valid_counts

        data.instance_centroids = instance_centroids
        data.instance_labels = instance_labels
        data.per_instance_count = per_instance_count
        data.instance_count = instance_labels.shape[0]


        return data


    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'


def process_index_array(index_array, name="index_array"):
    if index_array is not None:
        if index_array.ndim == 3:
            return torch.from_numpy(index_array.transpose([0, 2, 1]))# 三维数组，直接转置
        elif index_array.ndim == 2:
            # 二维数组，扩展为三维后转置
            return torch.from_numpy(index_array.transpose([1, 0]))
        else:
            raise ValueError(f"Unexpected dimension for {name}: {index_array.ndim}")
    else:
        return None

class FPSSample(T.BaseTransform):

    """
    原始数据直接转换为FPS24000格式的变换类。不针对mesh做任何操作
    """
    def __init__(self,
                 num_classes=15,
                 num_fps_points=24000,
                 sam_indexer: Optional[Callable[..., Any]] = None,
                 k = 8,
                 edge_sam_indexer: Optional[Callable[..., Any]] = None):
        self.num_classes = num_classes
        self.num_fps_points = num_fps_points
        self.sam_indexer = sam_indexer
        self.edge_sam_indexer = edge_sam_indexer
        self.k = k
    def __call__(self, data):
        return self.forward(data)

    def forward(self, data):

        vertices = data.pos.numpy()
        normals = data.normals.numpy()
        labels = data.y.numpy()

        labeled_vertices = np.concatenate([vertices, normals, labels], axis=1)
        if labeled_vertices.shape[0] > self.num_fps_points:
            labeled_vertices = gu.resample_pcd([labeled_vertices], self.num_fps_points, "fps")[0]

        # 分离顶点、法向量、标签
        vertices_fps = labeled_vertices[:,:3]  # fps的顶点坐标
        normals_fps = labeled_vertices[:,3:6]  # fps的法向量
        labels_fps = labeled_vertices[:,6]     # fps的标签

        # 构建f2c_idx, c2f_idx
        f2c_idx, c2f_idx = self.sam_indexer(vertices, vertices_fps) \
            if self.sam_indexer is not None else (None, None)

        # 原始形状需要保留的信息：
        data.y_ori = torch.from_numpy(labels).long()    # 原始标签，后续用于指标评价

        # 丢掉原始顶点，只保留fps顶点
        data.pos = torch.from_numpy(vertices_fps)
        data.normals = torch.from_numpy(normals_fps)
        data.y = torch.from_numpy(labels_fps).long()
        data.x = torch.from_numpy(labeled_vertices[:,:6])
        data.f2c_idx = process_index_array(f2c_idx, name="f2c_idx")
        data.c2f_idx = process_index_array(c2f_idx, name="c2f_idx")

        data.instances = data.instances[data.f2c_idx.T].squeeze()

        # 处理gt_edge_mask
        if hasattr(data, 'gt_edge_mask'):
            if self.edge_sam_indexer is None:
                self.edge_sam_indexer = KNNPointSamplingIndexer(k=self.k)
            f2c_idx, c2f_idx = self.edge_sam_indexer(vertices, vertices_fps)  # (M, k)
            # new_gt_edge_mask = torch.zeros(vertices_fps.shape[0], dtype=torch.long, device=data.gt_edge_mask.device)
            new_gt_edge_mask = data.gt_edge_mask[f2c_idx]
            new_gt_edge_mask = torch.max(new_gt_edge_mask, dim=1).values
            data.gt_edge_mask = new_gt_edge_mask.squeeze()

            # data.gt_tte_mask = data.gt_edge_mask & (data.y != 0)  # 只保留标签非0的点

        if hasattr(data, 'gt_tte_mask'):
            if self.edge_sam_indexer is None:
                self.edge_sam_indexer = KNNPointSamplingIndexer(k=self.k)
            f2c_idx, c2f_idx = self.edge_sam_indexer(vertices, vertices_fps)  # (M, k)
            new_gt_tte_mask = data.gt_tte_mask[f2c_idx]
            new_gt_tte_mask = torch.max(new_gt_tte_mask, dim=1).values
            data.gt_tte_mask = new_gt_tte_mask.squeeze()


        delattr(data, 'face') if hasattr(data, 'face') else None # 删除face属性
        delattr(data, 'edge_index') if hasattr(data, 'edge_index') else None # 删除face属性


        return data

    def __repr__(self) -> str:
        return (f"FPSSample(num_classes={self.num_classes}, "
                f"num_fps_points={self.num_fps_points}, "
                f"sam_indexer={self.sam_indexer})")




class QSlimSample(T.BaseTransform):
    """
    原始数据直接转换为QSlim格式的变换类。不针对mesh做任何操作
    """
    def __init__(self, num_classes=15,
                 tar_vert_num=10000,
                 sam_indexer: Optional[Callable[..., Any]] = None,
                 k = 8,
                 edge_sam_indexer: Optional[Callable[..., Any]] = None):
        self.num_classes = num_classes
        self.tar_vert_num = tar_vert_num
        self.sam_indexer = sam_indexer
        self.edge_sam_indexer = edge_sam_indexer
        self.k = k

    def __call__(self, data):
        return self.forward(data)

    def forward(self, data):
        vertices = data.pos.numpy()
        normals = data.normals.numpy()
        labels = data.y.numpy()
        filename = data.filename

        # qslim算法
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        #  data.face 为 [3,num_faces]，需要转置为 [num_faces,3]
        mesh.triangles = o3d.utility.Vector3iVector(data.face.T.numpy())

        # 使用QSlim算法将顶点数调整为10000
        mesh = AdjustMeshVertsTransform(target_vertices=self.tar_vert_num)(mesh)

        # 将mesh转为pydata2
        pydata2 = TriMeshToPy()(mesh)

        # 构建f2c_idx, c2f_idx
        f2c_idx, c2f_idx = self.sam_indexer(vertices, pydata2['V']) \
            if self.sam_indexer is not None else (None, None)

        # 原始形状需要保留的信息：
        data.y_ori = torch.from_numpy(labels).long()    # 原始标签，后续用于指标评价

        # 类标转移，使用NN算法
        data.f2c_idx = process_index_array(f2c_idx, name="f2c_idx")
        data.c2f_idx = process_index_array(c2f_idx, name="c2f_idx")

        # 类标转移相关的两个属性
        data.instances = data.instances[data.f2c_idx.T].squeeze()
        data.y = data.y[data.f2c_idx.T].squeeze()

        # 舍弃原始数据，替换为QSlim数据
        data.pos = torch.from_numpy(pydata2['V'])
        data.normals = torch.from_numpy(pydata2['vNormal'])
        data.x = torch.from_numpy(np.concatenate((pydata2['V'], pydata2['vNormal']), axis=1))
        # data.y = torch.from_numpy(labels[down_idx]).long()   # 使用NN作为预测标签
        data.face = torch.from_numpy(pydata2['F']).T.long()   # 记得转置，[3,num_faces]
        # data.f2c_idx = torch.from_numpy(f2c_idx) if f2c_idx is not None else None # f2c_idx: [N, 1]
        # data.c2f_idx = torch.from_numpy(c2f_idx) if c2f_idx is not None else None # c2f_idx: [N, 1]
        data = FaceToEdge(remove_faces=False)(data) # 生成edge_index

        if hasattr(data, 'gt_edge_mask'):
            if self.edge_sam_indexer is None:
                self.edge_sam_indexer = KNNPointSamplingIndexer(k=self.k)
            f2c_idx, c2f_idx = self.edge_sam_indexer(vertices, data.pos)  # (M, k)
            # new_gt_edge_mask = torch.zeros(vertices_fps.shape[0], dtype=torch.long, device=data.gt_edge_mask.device)
            new_gt_edge_mask = data.gt_edge_mask[f2c_idx]
            new_gt_edge_mask = torch.max(new_gt_edge_mask, dim=1).values
            data.gt_edge_mask = new_gt_edge_mask.squeeze()

        if hasattr(data, 'gt_tte_mask'):
            if self.edge_sam_indexer is None:
                self.edge_sam_indexer = KNNPointSamplingIndexer(k=self.k)
            f2c_idx, c2f_idx = self.edge_sam_indexer(vertices, data.pos)  # (M, k)
            new_gt_tte_mask = data.gt_tte_mask[f2c_idx]
            new_gt_tte_mask = torch.max(new_gt_tte_mask, dim=1).values
            data.gt_tte_mask = new_gt_tte_mask.squeeze()


        return data

class FPSQSlimSample(T.BaseTransform):
    """
    原始数据直接转换为FPS24000格式的变换类。不针对mesh做任何操作
    """
    def __init__(self, num_classes=15,
                 num_fps_points=16000,
                 tar_vert_num=10000,
                 fps_sam_indexer: Optional[Callable[..., Any]] = None,
                 qslim_sam_indexer: Optional[Callable[..., Any]] = None):
        self.num_classes = num_classes
        self.num_fps_points = num_fps_points
        self.tar_vert_num = tar_vert_num
        self.fps_sam_indexer = fps_sam_indexer
        self.qslim_sam_indexer = qslim_sam_indexer

    def __call__(self, data):
        return self.forward(data)

    def forward(self, data):

        data_copy = data.clone() # 数据拷贝，避免数据被修改
        # FPS
        fps_sampler = FPSSample(num_classes=self.num_classes,
                                num_fps_points=self.num_fps_points,
                                sam_indexer=self.fps_sam_indexer)
        data_fps = fps_sampler(data)

        # QSLIM
        qslim_sampler = QSlimSample(num_classes=self.num_classes,
                                    tar_vert_num=self.tar_vert_num,
                                    sam_indexer=self.qslim_sam_indexer)
        data_qslim = qslim_sampler(data_copy)

        data_fps.data_qslim = data_qslim
        return data_fps

class MountedQSlimSample(T.BaseTransform):
    """
    挂载QSlimSample结果到data.data_qslim对象下
    """
    def __init__(self, num_classes=15,
                 tar_vert_num=10000,
                 sam_indexer: Optional[Callable[..., Any]] = None):
        self.num_classes = num_classes
        self.tar_vert_num = tar_vert_num
        self.sam_indexer =sam_indexer

    def __call__(self, data):
        return self.forward(data)

    def forward(self, data):
        vertices = data.pos.numpy()
        normals = data.normals.numpy()
        labels = data.y.numpy()
        filename = data.filename

        # qslim算法
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        #  data.face 为 [3,num_faces]，需要转置为 [num_faces,3]
        mesh.triangles = o3d.utility.Vector3iVector(data.face.T.numpy())

        # 使用QSlim算法将顶点数调整为10000
        # mesh = AdjustMeshVertsTransform(target_vertices=self.tar_vert_num)(mesh)
        target_triangles = self.tar_vert_num * 2
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangles)

        # 将mesh转为pydata2
        pydata2 = TriMeshToPy()(mesh)

        # 构建f2c_idx, c2f_idx
        f2c_idx, c2f_idx = self.sam_indexer(vertices, pydata2['V']) \
            if self.sam_indexer is not None else (None, None)

        # 原始形状需要保留的信息：
        data_qslim = deepcopy(data)
        data_qslim.y_ori = torch.from_numpy(labels).long()    # 原始标签，后续用于指标评价

        # # 类标转移，使用NN算法
        # data_qslim.f2c_idx = process_index_array(f2c_idx, name="f2c_idx")
        # data_qslim.c2f_idx = process_index_array(c2f_idx, name="c2f_idx")

        # 类标转移相关的两个属性
        data_qslim.instances = data.instances[f2c_idx.T].squeeze()
        data_qslim.y = data.y[f2c_idx.T].squeeze()

        # 舍弃原始数据，替换为QSlim数据
        data_qslim.pos = torch.from_numpy(pydata2['V'])
        data_qslim.normals = torch.from_numpy(pydata2['vNormal'])
        # data_qslim.x = torch.from_numpy(np.concatenate((pydata2['V'], pydata2['vNormal']), axis=1))
        data_qslim.face = torch.from_numpy(pydata2['F']).T.long()   # 记得转置，[3,num_faces]
        # data = FaceToEdge(remove_faces=False)(data) # 生成edge_index

        delattr(data_qslim, 'edge_index') if hasattr(data_qslim, 'edge_index') else None

        data.data_qslim = data_qslim

        return data


class BAFPSSample(T.BaseTransform):
    """
    Boundary aware FPS Sample (BAFPS)
    结合 FPS 和模型预测权重的自适应采样
    策略：
    1. 使用 FPS 保证空间均匀性（占 init_ratio）
    2. 使用模型预测权重进行重要性采样（占 1-init_ratio）
    3. 权重基于边缘置信度和语义类别
    返回的对象与 FPSSample 完全一致
    """

    def __init__(self,
                 num_classes=15,
                 num_fps_points=24000,
                 sam_indexer: Optional[Callable[..., Any]] = None,
                 k=8,
                 edge_sam_indexer: Optional[Callable[..., Any]] = None,
                 predictor_type='DTSegNet',
                 ckpt_path='../../../common/DTSegNet.ckpt',
                 net_para_name='seg_base',
                 init_ratio=0.8,
                 weights_type='default',
                 device=None,
                 disp_info=False):
        """
        初始化 BAFPS 采样器
        Args:
            num_classes: 类别数
            num_fps_points: 目标采样点数
            sam_indexer: 采样索引器
            k: KNN 的 k 值
            edge_sam_indexer: 边缘采样索引器
            predictor_type: 预测器类型 ('PTSegNet' 或 'DTSegNet')
            ckpt_path: 模型检查点路径
            net_para_name: 网络参数名称
            init_ratio: 第一轮 FPS 保留比例 (0~1)
                       FPS 采样点数 = npoint * init_ratio
                       IS 采样点数 = npoint * (1 - init_ratio)
            weights_type: 权重类型
                - 'default': 默认权重（边缘 + 语义不确定性）
                - 'edge_only': 只使用边缘权重
                - 'semantic_only': 只使用语义权重
                - 'custom': 自定义权重函数
            device: 设备（None 则自动选择）
        """
        # 调用父类初始化（完全仿照 FPSSample）
        super().__init__()

        # FPSSample的参数
        self.num_classes = num_classes
        self.num_fps_points = num_fps_points

        self.k = k
        self.sam_indexer = NNPointSamplingIndexer() if sam_indexer is None else sam_indexer
        self.edge_sam_indexer = KNNPointSamplingIndexer(self.k) if edge_sam_indexer is None else edge_sam_indexer

        # BAFPS 特有参数
        self.predictor_type = predictor_type
        self.ckpt_path = ckpt_path
        self.net_para_name = net_para_name
        self.init_ratio = init_ratio
        self.weights_type = weights_type
        self.device = device

        # 验证参数
        if not (0 < init_ratio <= 1):
            raise ValueError(f"init_ratio 必须在 (0, 1] 范围内，当前为 {init_ratio}")

        # 调试信息
        self.disp_info = disp_info

        # 延迟加载预测器
        self._predictor = None

    @property
    def predictor(self):
        """懒加载预测器"""
        if self._predictor is None:
            try:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self._predictor = PredictorFactory.get_predictor(
                    predictor_type=self.predictor_type,
                    ckpt_path=self.ckpt_path,
                    net_para_name=self.net_para_name,
                    device=device
                )
            except Exception as e:
                print(f"❌ 预测器加载失败：{str(e)}")
                raise RuntimeError(f"BAFPSSample 无法加载预测器：{str(e)}") from e

        return self._predictor

    def __call__(self, data):
        """
        对数据进行 BAFPS 采样（返回与 FPSSample 相同的对象结构）
        Args:
            data: PyG Data 对象
        Returns:
            data: 处理后的 Data 对象（与 FPSSample 返回的格式一致）
        """
        fps_sampler = FPSSample(num_classes=self.num_classes,
                               num_fps_points=self.num_fps_points,
                               sam_indexer=self.sam_indexer)
        data_fps = deepcopy(data)
        data_fps = fps_sampler(data_fps)
        batch = Batch.from_data_list([data_fps])
        with torch.no_grad():
            pred_data = self.predictor.predict_step_forward(batch)
            pred_data = pred_data.cpu()
            # print(pred_data)

        ba_weights, ba_edge_y = self._compute_importance_weights(pred_data) # (N,)

        weights_ori = ba_weights[pred_data.c2f_idx]
        ba_fps_idxs = gu.ba_fps(data.pos.cpu().numpy(),
                                   npoint=self.num_fps_points,
                                   init_ratio=self.init_ratio,
                                   weights= weights_ori.cpu().numpy())

        # ========== 步骤 6: 应用采样（仿照 FPSSample）==========
        # 使用采样索引提取数据
        # 分离顶点、法向量、标签
        pos_ba_fps = data.pos[ba_fps_idxs]
        normals_ba_fps = data.normals[ba_fps_idxs]
        labels_ba_fps = data.y[ba_fps_idxs]
        x_ba_fps = torch.cat([pos_ba_fps, normals_ba_fps], dim=-1)

        # 构建 f2c_idx, c2f_idx（与 FPSSample 一致）
        f2c_idx, c2f_idx = self.sam_indexer(data.pos, pos_ba_fps) \
            if self.sam_indexer is not None else (None, None)

        data_bafps = deepcopy(data_fps)

        # 原始形状需要保留的信息
        data_bafps.y_ori = data.y

        # 更新为采样后的数据
        data_bafps.pos = pos_ba_fps
        data_bafps.normals = normals_ba_fps
        data_bafps.y = labels_ba_fps
        data_bafps.x = x_ba_fps
        data_bafps.f2c_idx = process_index_array(f2c_idx, name="f2c_idx")
        data_bafps.c2f_idx = process_index_array(c2f_idx, name="c2f_idx")

        # 权值信息及伪标签
        data_bafps.ba_weights = ba_weights
        data_bafps.ba_y = ba_edge_y

        # 处理 instances
        if hasattr(data, 'instances'):
            data_bafps.instances = data.instances[data_bafps.f2c_idx.T].squeeze()

        # 处理 gt_edge_mask（与 FPSSample 一致）
        if hasattr(data, 'gt_edge_mask'):
            if self.edge_sam_indexer is None:
                self.edge_sam_indexer = KNNPointSamplingIndexer(k=self.k)
            f2c_idx_edge, _ = self.edge_sam_indexer(data.pos, pos_ba_fps)
            new_gt_edge_mask = data.gt_edge_mask[f2c_idx_edge]
            new_gt_edge_mask = torch.max(new_gt_edge_mask, dim=1).values
            data_bafps.gt_edge_mask = new_gt_edge_mask.squeeze()

        # 处理 gt_tte_mask（与 FPSSample 一致）
        if hasattr(data, 'gt_tte_mask'):
            if self.edge_sam_indexer is None:
                self.edge_sam_indexer = KNNPointSamplingIndexer(k=self.k)
            f2c_idx_edge, _ = self.edge_sam_indexer(data.pos, pos_ba_fps)
            new_gt_tte_mask = data.gt_tte_mask[f2c_idx_edge]
            new_gt_tte_mask = torch.max(new_gt_tte_mask, dim=1).values
            data_bafps.gt_tte_mask = new_gt_tte_mask.squeeze()

        # 删除 face 和 edge_index
        delattr(data_bafps, 'face') if hasattr(data, 'face') else None
        delattr(data_bafps, 'edge_index') if hasattr(data, 'edge_index') else None


        return data_bafps


    def _compute_importance_weights(self, pred_data):
        """计算重要性采样权重"""

        bg_flag, tae_flag, tte_flag, tae_elf, tte_elf, tae_elf_sm, tte_elf_sm = (
            self.get_elfs(pred_data))

        # 根据不同的 weights_type 计算权重
        if self.weights_type == 'default':
            weights = self._compute_default_weights(bg_flag, tae_flag, tte_flag, tae_elf_sm, tte_elf_sm)
        else:
            raise ValueError(f"未知的 weights_type: {self.weights_type}")

        fake_edge_y = self.compute_fake_edge_label(bg_flag, tae_elf_sm, tae_flag, tte_elf_sm, tte_flag)

        return weights,fake_edge_y

    def compute_fake_edge_label(self, bg_flag, tae_elf_sm, tae_flag, tte_elf_sm, tte_flag):
        y = torch.ones(bg_flag.shape[0])
        y[bg_flag == 0] = 0
        y[bg_flag > 0] = 1
        y[tae_flag > 0] = y[tae_flag > 0] + tae_elf_sm[0, 1, :].squeeze()[tae_flag > 0] * 5
        y[tte_flag > 0] = y[tte_flag > 0] + 5 + tte_elf_sm[0, 1, :].squeeze()[tte_flag > 0] * 5

        return  y

    def _compute_default_weights(self, bg_flag, tae_flag, tte_flag,tae_elf_sm, tte_elf_sm):
        """默认权重：结合边缘置信度和语义不确定性
        权值规则：背景：1.0，非背景：10.0，
        tae边缘：10.0 + 990.0 * tae_elf_sm[1, :]，
        边缘：1000.0 * 1990.0 * tte_elf_sm[1, :]
        """
        weights = torch.ones(bg_flag.shape[0])
        weights[bg_flag == 0] = 1.0
        weights[bg_flag > 0] = 10.0
        weights[tae_flag > 0] = (weights[tae_flag > 0] + 990.0 *
                                  tae_elf_sm[0, 1, :].squeeze()[tae_flag > 0])

        weights[tte_flag > 0] = (weights[tte_flag > 0] + 1990.0
                                 * tte_elf_sm[0, 1, :].squeeze()[tte_flag > 0])

        return weights

    def get_elfs(self, pred_data):
        bg_flag = torch.argmax(pred_data.outputs['seg_result'], dim=-1) > 0  # 0:bf, 1:tooth
        tae_elf = pred_data.outputs['edge_seg_result']
        tte_elf = pred_data.outputs['tte_seg_result']
        tae_elf_sm = F.softmax(tae_elf, dim=1)
        tte_elf_sm = F.softmax(tte_elf, dim=1)
        tae_flag = torch.argmax(pred_data.outputs['edge_seg_result'], dim=1) > 0
        tte_flag = torch.argmax(pred_data.outputs['tte_seg_result'], dim=1) > 0
        # edge_flag = (tae_flag + tte_flag) > 0
        bg_flag, tae_flag, tte_flag = bg_flag.view(-1), tae_flag.view(-1), tte_flag.view(-1)  # N,
        return bg_flag, tae_flag, tte_flag, tae_elf, tte_elf, tae_elf_sm, tte_elf_sm

    def __repr__(self) -> str:
        return (f"BAFPFSSample(num_classes={self.num_classes}, "
                f"num_fps_points={self.num_fps_points}, "
                f"sam_indexer={self.sam_indexer}, "
                f"init_ratio={self.init_ratio}, "
                f"predictor_type={self.predictor_type}, "
                f"weights_type={self.weights_type})")



class FixedRadiusNNEdgeMask(T.BaseTransform):
    def __init__(self,
                 r: float = 0.03,
                 max_num_neighbors: int = 16):
        """
        初始化FixedRadiusNNEdgeGenerator
        :param r: 半径，用于确定邻域范围
        :param max_num_neighbors: 最大邻居数，用于限制邻域大小
        """
        self.r = r
        self.max_num_neighbors = max_num_neighbors

    def __call__(self, data):
        """
        识别边缘点并将结果存储在gt_edge_mask属性中
        :param data: 包含pos和labels属性的数据对象
        :return: 添加了gt_edge_mask属性的数据对象
        """
        pos = data.pos
        labels = data.y.squeeze() if data.y.dim() > 1 else data.y

        # 使用radius_graph函数计算邻域图
        edge_index = radius_graph(pos, r=self.r, max_num_neighbors=self.max_num_neighbors, loop=False)

        # 获取源节点和目标节点
        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]

        # 获取源节点和目标节点的标签
        src_labels = labels[src_nodes]
        dst_labels = labels[dst_nodes]

        # 检查每个节点是否与其邻居有不同的标签
        # 如果某个节点的邻居中有不同标签的节点，则该节点是边缘点
        diff_labels = src_labels != dst_labels

        # 对于每个源节点，检查是否有至少一个邻居具有不同的标签
        # 使用scatter_max来聚合不同标签的信息
        edge_mask = torch.zeros(pos.size(0), dtype=torch.long, device=pos.device)
        scatter_max(diff_labels.long(), src_nodes, out=edge_mask)

        # 将结果存储在gt_edge_mask属性中
        data.gt_edge_mask = edge_mask

        return data

    def __repr__(self) -> str:
        """自定义打印格式"""
        return f"{self.__class__.__name__}(r={self.r})"


class DynamicRadiusNNEdgeMask(T.BaseTransform):
    def __init__(self,
                 rmin: float = 1.8,
                 rmax: float = 2.2,
                 k: int = 2,
                 disp_avg_dist: bool = True,
                 eliminate_tae_bg: bool = False,
                 max_num_neighbors: int = 16):
        """
        初始化DynamicRadiusNNEdgeMask
        :param rmin: 最小半径
        :param rmax: 最大半径
        :param max_num_neighbors: 最大邻居数，用于限制邻域大小
        """
        self.rmin = rmin
        self.rmax = rmax
        self.k = k
        self.max_num_neighbors = max_num_neighbors
        self.disp_avg_dist = disp_avg_dist
        self.eliminate_tae_bg = eliminate_tae_bg

    def __call__(self, data):
        """
        识别边缘点并将结果存储在gt_edge_mask属性中
        :param data: 包含pos和labels属性的数据对象
        :return: 添加了gt_edge_mask属性的数据对象
        """
        pos = data.pos
        labels = data.y.squeeze() if data.y.dim() > 1 else data.y

        tae_gt_bin_labels, tte_gt_bin_labels = self.generate_edge_masks(pos, labels)

        data.gt_edge_mask = tae_gt_bin_labels
        data.gt_tte_mask = tte_gt_bin_labels


        return data

    def generate_edge_masks(self, pos, labels):
        knn_avg_dist = compute_knn_average_distance(pos, self.k)
        if self.disp_avg_dist:
            print(f"average distance: {knn_avg_dist}")
        # 计算极角
        _, abs_ang = compute_polar_angles(pos)
        radius = (self.rmin + abs_ang * (self.rmax - self.rmin)) * knn_avg_dist
        tae_gt_bin_labels = compute_tae_gt_bin_labels_ang(pos, labels, r=radius)
        tte_gt_bin_labels = compute_tte_gt_bin_labels(pos, labels, r=radius)

        if self.eliminate_tae_bg:
            print(f"eliminate tae bg:{torch.sum(tae_gt_bin_labels[labels == 0])}")
            tae_gt_bin_labels[labels == 0] = 0

        return tae_gt_bin_labels, tte_gt_bin_labels


class KNNThreEdgeMask(T.BaseTransform):
    def __init__(self,
                 k: int = 8,
                 threshold_ratio: float = 0.5):
        """
        基于K近邻的边界点检测类，参考你提供的compute_boundary_mask逻辑

        :param k: 近邻数量
        :param threshold_ratio: 阈值比例，用于判断是否为边界点
        """
        self.k = k
        self.threshold_ratio = threshold_ratio

    def __call__(self, data):
        """
        识别边界点并将结果存储在boundary_mask和tte_boundary_mask属性中
        :param data: 包含pos和y属性的数据对象
        :return: 添加了boundary_mask和tte_boundary_mask属性的数据对象
        """
        pos = data.pos
        labels = data.y.squeeze() if data.y.dim() > 1 else data.y

        # 计算普通的边界mask (gt_edge_mask)
        edge_mask, tte_mask = self.compute_boundary_mask(pos, labels, self.k, self.threshold_ratio)

      # 根据配置决定返回格式
        data.gt_edge_mask = torch.from_numpy(edge_mask)
        data.gt_tte_mask = torch.from_numpy(tte_mask)

        return data

    def compute_boundary_mask(self, pos, labels, k, threshold_ratio):
        """
        向量化计算边界点mask，参考你提供的逻辑

        :param pos: 点坐标，形状 [N, 3]
        :param labels: 点标签，形状 [N,]
        :param k: 近邻数量
        :param threshold_ratio: 阈值比例
        :return: 边界点mask，bool数组，True表示边界点
        """
        # 转换为numpy进行计算
        pos_np = pos.cpu().numpy() if torch.is_tensor(pos) else pos
        labels_np = labels.cpu().numpy() if torch.is_tensor(labels) else labels

        # 构建近邻搜索器
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(pos_np)
        _, indices = nbrs.kneighbors(pos_np)  # (N, k+1)

        # 1. 计算tae
        # 获取邻居标签
        neighbor_labels = labels_np[indices]  # (N, k+1)
        center_labels = labels_np[:, np.newaxis]  # (N, 1)
        # 计算标签差异
        equal_mask = (neighbor_labels == center_labels)  # (N, k+1)
        equal_mask = equal_mask[:, 1:]  # 排除自己，shape=(N, k)
        # 计算不同标签的数量
        diff_count = np.sum(~equal_mask, axis=1)  # shape=(N,)
        # 根据阈值判断是否为边界点
        boundary_mask = diff_count > (k * threshold_ratio)

        # 2. 计算tte边界点
        neighbor_labels_k = neighbor_labels[:, 1:]  # (N, k)
        nz_mask = (neighbor_labels != 0)  # (N, k+1)
        nz_mask_k = nz_mask[:, 1:]        # (N, k)

        # 找非零值的最小值
        labels_for_min = np.where(nz_mask_k, neighbor_labels_k, np.inf)  # 零值设为inf
        nz_min_values = np.min(labels_for_min, axis=1)  # 找最小的非零值
        nz_min_values = np.where(nz_min_values == np.inf, 0, nz_min_values)  # 全零时设为0

        # 找非零值的最大值
        labels_for_max = np.where(nz_mask_k, neighbor_labels_k, -np.inf)  # 零值设为-inf
        nz_max_values = np.max(labels_for_max, axis=1)  # 找最大的非零值
        nz_max_values = np.where(nz_max_values == -np.inf, 0, nz_max_values)  # 全零时设为0

        # 判断TTE边界条件
        all_zero = (nz_max_values == 0)  # 邻域全零
        same_nonzero = (nz_max_values == nz_min_values)  # 非零标签相同

        # TTE边界点：不是全零且非零标签不全相同
        tte_mask = ~(all_zero | same_nonzero) & boundary_mask

        return boundary_mask, tte_mask


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'k={self.k}, '
                f'threshold_ratio={self.threshold_ratio}, '
                f'return_numpy={self.return_numpy})')

    # ... existing code ...

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'k={self.k}, '
                f'threshold_ratio={self.threshold_ratio}')

# ... existing code ...




class FaceInconsistEdgeMask(T.BaseTransform):

    def __call__(self, data):
        """
        识别面片标签不一致的边缘点并将结果存储在gt_edge_mask属性中
        :param data: 包含pos, face和y属性的数据对象
        :return: 添加了gt_edge_mask属性的数据对象
        """
        if not hasattr(data, 'pos') or not hasattr(data, 'face') or not hasattr(data, 'y'):
            raise ValueError("Data object must have 'pos', 'face', and 'y' attributes")

        pos = data.pos
        faces = data.face  # [3, num_faces] 或 [num_faces, 3]
        labels = data.y.squeeze() if data.y.dim() > 1 else data.y

        # 确保faces是[3, num_faces]的格式
        if faces.shape[0] != 3:
            faces = faces.t()  # 转置为[3, num_faces]

        num_vertices = pos.size(0)
        # 获取每个面片的三个顶点索引
        face_vertices = faces  # [3, num_faces]
        # 获取每个面片的三个顶点的标签
        f1_labels = labels[face_vertices[0]]  # [num_faces]
        f2_labels = labels[face_vertices[1]]  # [num_faces]
        f3_labels = labels[face_vertices[2]]  # [num_faces]

        # 初始化边缘掩码，所有点初始为0
        edge_mask = torch.zeros(num_vertices, dtype=torch.long, device=pos.device)
        # 检查每个面片内的标签是否一致,如果三个点的标签不完全相同，则这个面片包含边缘点
        same_labels = (f1_labels == f2_labels) & (f2_labels == f3_labels)
        inconsistent_faces = ~same_labels  # [num_faces],不一致的面片（即包含边缘点的面片）的索引
        inconsistent_face_vertices = face_vertices[:, inconsistent_faces]  # [3, num_inconsistent_faces]
        edge_vertex_indices = inconsistent_face_vertices.flatten().unique() # 将这些顶点标记为边缘点（设置为1）, 展平顶点索引并去重
        edge_mask[edge_vertex_indices] = 1 # 将对应的边缘掩码位置为1
        # 将结果存储在gt_edge_mask属性中
        data.gt_edge_mask = edge_mask   #  [num_vertices], 边缘点对应的掩码, 1表示是边缘点, 0表示不是边缘点

        # tte 边缘点
        tte_mask = torch.zeros(num_vertices, dtype=torch.long, device=pos.device)
        # 检查每个面片内的标签是否不一致,如果三个点的标签不完全相同，则这个面片包含边缘点
        diff_labels1 = (f1_labels != f2_labels) | (f2_labels != f3_labels) | (f3_labels != f1_labels)
        diff_labels2 = (f1_labels != 0) & (f2_labels != 0) & (f3_labels != 0)
        diff_labels31 = ((f1_labels == 0) & (f2_labels > 0) & (f3_labels > 0) & (f2_labels != f3_labels))
        diff_labels32 = ((f2_labels == 0) & (f1_labels > 0) & (f3_labels > 0) & (f3_labels != f1_labels))
        diff_labels33 = ((f3_labels == 0) & (f2_labels > 0) & (f1_labels > 0) & (f1_labels != f2_labels))
        diff_labels3 = diff_labels31 | diff_labels32 | diff_labels33
        inconsistent_faces = diff_labels1 & (diff_labels2 | diff_labels3)  # [num_faces],不一致的面片（即包含边缘点的面片）的索引
        inconsistent_face_vertices = face_vertices[:, inconsistent_faces]  # [3, num_inconsistent_faces]
        edge_vertex_indices = inconsistent_face_vertices.flatten().unique() # 将这些顶点标记为边缘点（设置为1）, 展平顶点索引并去重
        tte_mask[edge_vertex_indices] = 1 # 将对应的边缘掩码位置为1
        # tte_mask = edge_mask & (labels != 0)  # 只保留标签非0的点
        data.gt_tte_mask = tte_mask   #  [num_vertices], 边缘点对应的掩码, 1表示是边缘点, 0表示不是边缘点


        return data

    def __repr__(self) -> str:
        """自定义打印格式"""
        return f"{self.__class__.__name__}()"

class RadiiEdgeIndex(T.BaseTransform):
    """
    根据不同半径值生成和存储edge_index的变换类
    """
    def __init__(self, radii: List[float], max_num_neighbors: int = 32):
        """
        初始化
        :param radii: 半径列表
        :param max_num_neighbors: 最大邻居数
        """
        self.radii = radii
        self.max_num_neighbors = max_num_neighbors

    def __call__(self, data: Data) -> Dict[float, torch.Tensor]:
        """
        为给定的data对象生成不同半径下的edge_index
        :param data: 包含pos属性的数据对象
        :return: 字典，key为半径，value为对应的edge_index
        """
        radii_edge_index = self.forward(data)
        data.radii_edge_index = radii_edge_index
        return data

    def forward(self, data: Data) -> Dict[float, torch.Tensor]:
        """
        生成不同半径下的edge_index
        :param data: 包含pos属性的数据对象
        :return: 字典，key为半径，value为对应的edge_index
        """
        pos = data.pos
        batch = data.batch if hasattr(data, 'batch') else None

        edge_index_dict = {}

        for r in self.radii:
            edge_index = radius_graph(
                pos,
                r=r,
                batch=batch,
                max_num_neighbors=self.max_num_neighbors
            )
            edge_index_dict[r] = edge_index

        return edge_index_dict



#################################################################
# Test Functions
#################################################################

def test_ToTensorTuple_ToNumpyTuple():
    # 创建转换器实例
    tensor_converter = ToTensorTuple()
    numpy_converter = ToNumpyTuple()
    # 示例数据
    np_array1 = np.random.rand(1, 1)  # numpy数组
    np_array2 = np.random.rand(2, 2)
    torch_tensor1 = torch.rand(5, 3)  # torch张量
    torch_tensor2 = torch.rand(5)

    # 打印原始dtype
    print(f"原始numpy数组1 dtype: {np_array1.dtype}")
    print(f"原始numpy数组2 dtype: {np_array2.dtype}")
    print(f"原始torch张量1 dtype: {torch_tensor1.dtype}")
    print(f"原始torch张量2 dtype: {torch_tensor2.dtype}\n")

    # 转换为tensor
    tensor_result = tensor_converter(np_array1, np_array2)
    print(f"Numpy转Tensor结果类型: {type(tensor_result[0])}, dtype: {tensor_result[0].dtype}")
    print(f"Numpy转Tensor结果类型: {type(tensor_result[1])}, dtype: {tensor_result[1].dtype}\n")

    # 转换为numpy
    numpy_result = numpy_converter(torch_tensor1[0], torch_tensor2[1])
    print(f"Tensor转Numpy结果类型: {type(numpy_result[0])}, dtype: {numpy_result[0].dtype}")
    print(f"Tensor转Numpy结果类型: {type(numpy_result[1])}, dtype: {numpy_result[1].dtype}\n")

    # 混合类型转换示例
    mixed_result = tensor_converter(np_array1, np_array2, torch_tensor1, torch_tensor2)
    print(f"混合输入转换结果1 dtype: {mixed_result[0].dtype}")
    print(f"混合输入转换结果2 dtype: {mixed_result[1].dtype}")
    print(f"混合输入转换结果3 dtype: {mixed_result[2].dtype}")
    print(f"混合输入转换结果4 dtype: {mixed_result[3].dtype}\n")

    mixed_numpy_result = numpy_converter(np_array1, torch_tensor1, np_array2, torch_tensor2)
    print(f"Numpy转换器混合结果1 dtype: {mixed_numpy_result[0].dtype}")
    print(f"Numpy转换器混合结果2 dtype: {mixed_numpy_result[1].dtype}")
    print(f"Numpy转换器混合结果3 dtype: {mixed_numpy_result[2].dtype}")
    print(f"Numpy转换器混合结果4 dtype: {mixed_numpy_result[3].dtype}")


def test_KNNThreEdgeMask():
    """
    测试完整的TTE边界点检测算法
    """
    # 模拟真实数据
    pos = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0],  # 牙齿A区域
        [2, 0, 0], [2, 1, 0],  # 牙齿B区域
        [1, 2, 0],  # 背景区
        [3, 0, 0], [3, 1, 0],  # 牙齿C区域
        [2, 2, 0],  # 背景区
        [0, 2, 0]  # 牙齿A区域
    ])

    labels = np.array([1, 1, 1, 2, 2, 0, 3, 3, 0, 1])

    print("=== 完整TTE算法测试 ===")

    # 1. 构建近邻关系
    nbrs = NearestNeighbors(n_neighbors=4, algorithm='auto').fit(pos)
    _, indices = nbrs.kneighbors(pos)

    neighbor_labels = labels[indices]
    center_labels = labels[:, np.newaxis]

    # 2. 计算普通边界点
    equal_mask = (neighbor_labels == center_labels)
    equal_mask_k = equal_mask[:, 1:]  # 排除自己
    diff_count = np.sum(~equal_mask_k, axis=1)
    boundary_mask = diff_count > 2  # k=3, 阈值>1.5

    print("普通边界点:", np.where(boundary_mask)[0])

    # 3. 计算TTE边界点
    neighbor_labels_k = neighbor_labels[:, 1:]
    nz_mask_k = (neighbor_labels_k != 0)

    # 找非零值的最小值
    labels_for_min = np.where(nz_mask_k, neighbor_labels_k, np.inf)
    nz_min_values = np.min(labels_for_min, axis=1)
    nz_min_values = np.where(nz_min_values == np.inf, 0, nz_min_values)

    # 找非零值的最大值
    labels_for_max = np.where(nz_mask_k, neighbor_labels_k, -np.inf)
    nz_max_values = np.max(labels_for_max, axis=1)
    nz_max_values = np.where(nz_max_values == -np.inf, 0, nz_max_values)

    # TTE条件判断
    all_zero = (nz_max_values == 0)
    same_nonzero = (nz_max_values == nz_min_values)
    tte_mask = ~(all_zero | same_nonzero) & boundary_mask

    print("TTE边界点:", np.where(tte_mask)[0])

    # 详细分析结果
    print("\n=== 详细分析 ===")
    for i in range(len(pos)):
        print(f"点{i}: 标签={labels[i]}, 普通边界={'是' if boundary_mask[i] else '否'}, "
              f"TTE边界={'是' if tte_mask[i] else '否'}")


def main():
    # test_ToTensorTuple_ToNumpyTuple()
    test_KNNThreEdgeMask()

if __name__ == '__main__':
    main()
