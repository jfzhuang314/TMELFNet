# PointSamplingIndexer 类设计

import torch
from abc import ABC, abstractmethod
import time
import torch
from sklearn.neighbors import NearestNeighbors
import numpy as np
from torch_geometric.transforms import BaseTransform
import torch_geometric.transforms as T

#region utils
def sklearn_nn_search(pts: np.ndarray, pts_fps: np.ndarray, k: int = 1):
    """
    使用 sklearn 的最近邻搜索算法
    """
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(pts_fps)
    distances, indices = nbrs.kneighbors(pts)

    return indices.flatten()

#endregion utils


class PointSamplingIndexer(T.BaseTransform):
    """点云采样索引生成器基类"""

    @abstractmethod
    def generate_indices(self, pts: np.ndarray, pts_fps: np.ndarray) -> tuple:
        """
        生成上采样索引和下采样索引

        Args:
            pts: 原始点云数据 (N, 3)
            pts_fps: FPS采样后的点云数据 (M, 3)

        Returns:
            tuple: (ud_idx, du_idx) 分别为上采样索引和下采样索引
        """
        pass

    def __call__(self, pts: np.ndarray, pts_fps: np.ndarray):
        return self.generate_indices(pts, pts_fps)

class NNPointSamplingIndexer(PointSamplingIndexer):
    """基于KD树最近邻映射的点采样索引生成器"""
    def generate_indices(self, pts: np.ndarray, pts_fps: np.ndarray) -> tuple:
        """
        使用KD树算法生成上采样索引和下采样索引
        时间复杂度从 O(N*M) 降至 O(N*log(M) + M*log(M))
        params:
            pts: 原始点云数据 (N, 3)
            pts_fps: FPS采样后的点云数据 (M, 3)
        Returns:
            tuple: (f2c_idx, c2f_idx) 分别为上采样索引和下采样索引
        """
        # 上采样索引，从采样点到原始点索引。例：pts[c2f_idx]
        corse_nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(pts_fps)
        _, c2f_idx = corse_nbrs.kneighbors(pts)  # (M, 1)->(N, 1)。例：pts_fps[c2f_idx]
        # c2f_idx = torch.from_numpy(c2f_idx.flatten()).to(pts.device)

        # 下采样索引，从原始点到采样点索引。例：pts_fps[f2c_idx]
        fined_nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(pts)
        _, f2c_idx = fined_nbrs.kneighbors(pts_fps)  # (N, 1)->(M, 1)。例：pts[f2c_idx]
        # f2c_idx = torch.from_numpy(f2c_idx.flatten()).to(pts.device)

        return f2c_idx, c2f_idx
    def __repr__(self) -> str:
        return f"NNPointSamplingIndexer"



class KNNPointSamplingIndexer(PointSamplingIndexer):
    """基于KD树K近邻映射的点采样索引生成器"""

    def __init__(self, k: int = 3):
        self.k = k
    def generate_indices(self, pts: np.ndarray, pts_fps: np.ndarray) -> tuple:
        """
        使用KD树算法生成K近邻索引
        """
        # 获取每个原始点的K个最近采样点（上采样）
        coarse_nbrs = NearestNeighbors(n_neighbors=min(self.k, pts_fps.shape[0]),
                               algorithm='kd_tree').fit(pts_fps)
        _, c2f_idx = coarse_nbrs.kneighbors(pts)  # (N, k)
        # c2f_idx = torch.from_numpy(c2f_idx).to(pts.device)

        # 获取每个采样点的最近原始点（下采样）
        fined_nbrs = NearestNeighbors(n_neighbors=min(self.k, pts.shape[0]),
                                algorithm='kd_tree').fit(pts)
        _, f2c_idx = fined_nbrs.kneighbors(pts_fps)  # (M, k)

        return f2c_idx, c2f_idx


class NNPointSamplingIndexerCDist(PointSamplingIndexer):
    """基于最近邻映射的点采样索引生成器"""

    def generate_indices(self, pts: np.ndarray, pts_fps: np.ndarray) -> tuple:
        """
        使用最近邻映射生成上采样索引和下采样索引

        Args:
            pts: 原始点云数据 (N, 3)
            pts_fps: FPS采样后的点云数据 (M, 3)

        Returns:
            tuple: (ud_idx, du_idx)
                ud_idx: 上采样索引，用于将采样点扩展回原始点数
                du_idx: 下采样索引，用于从原始点获取采样点索引
        """
        # 计算原始点到采样点的距离矩阵
        dists = torch.cdist(pts, pts_fps)  # (N, M)， 22s
        # 对于每个原始点，找到最近的采样点索引（用于上采样）
        c2f_idx = torch.argmin(dists, dim=1)  # (N,)
        # 下采样索引就是采样点在原始点中的索引
        # 通过找到每个采样点在原始点中的最近邻索引来确定
        f2c_idx = torch.argmin(dists, dim=0)  # (M,)

        return f2c_idx, c2f_idx

class KNNPointSamplingIndexerCDist(PointSamplingIndexer):
    """基于K近邻映射的点采样索引生成器"""

    def __init__(self, k: int = 3):
        """
        Args:
            k: K近邻的数量
        """
        self.k = k

    def generate_indices(self, pts: np.ndarray, pts_fps: np.ndarray) -> tuple:
        """
        使用K近邻映射生成上采样索引和下采样索引

        Args:
            pts: 原始点云数据 (N, 3)
            pts_fps: FPS采样后的点云数据 (M, 3)

        Returns:
            tuple: (ud_idx, du_idx)
                ud_idx: 上采样索引，形状为 (N, k)，每个原始点对应k个最近的采样点
                du_idx: 下采样索引，形状为 (M,)，每个采样点对应原始点中的索引
        """
        # 计算原始点到采样点的距离矩阵
        dists = torch.cdist(pts, pts_fps)  # (N, M), 22s

        # 对于每个原始点，找到K个最近的采样点索引（用于上采样）
        c2f_idx = torch.topk(dists, self.k, dim=1, largest=False)[1]  # (N, k)

        # 下采样索引是每个采样点在原始点中的最近邻索引
        f2c_idx = torch.argmin(dists, dim=0)  # (M,)

        return f2c_idx, c2f_idx
def example_usage():
    """
    使用示例，包含时间测量
    测试结果：
    1. CDist方法。
    2. KDTree方法。

    Args:
        pts: 原始点云数据 (N, 3)
        pts_fps: FPS采样后的点云数据 (M, 3)
    """
    # 假设有点云数据
    pts = np.random.randn(100000, 3)  # 原始点云
    pts_fps = np.random.randn(24000, 3)  # FPS采样后的点云

    # 测试 NN 方法
    nn_indexer = NNPointSamplingIndexer()
    start_time = time.perf_counter()
    f2c_idx_nn, c2f_idx_nn = nn_indexer.generate_indices(pts, pts_fps)
    nn_time = time.perf_counter() - start_time

    print(f"NN 方法执行时间: {nn_time:.4f} 秒")
    print(f"NN - f2c_idx shape: {f2c_idx_nn.shape}, c2f_idx shape: {c2f_idx_nn.shape}")
    print(f"NN - f2c_idx: {pts[f2c_idx_nn].transpose([0,2,1]).shape}, c2f_idx: {pts_fps[c2f_idx_nn].transpose([0,2,1]).shape}")
    # print(f"NN - f2c_idx: {pts[f2c_idx_nn]}")

    # 测试 KNN 方法
    knn_indexer = KNNPointSamplingIndexer(k=5)
    start_time = time.perf_counter()
    f2c_idx_knn, c2f_idx_knn = knn_indexer.generate_indices(pts, pts_fps)
    knn_time = time.perf_counter() - start_time

    print(f"KNN 方法执行时间: {knn_time:.4f} 秒")
    print(f"KNN - f2c_idx shape: {f2c_idx_knn.shape}, c2f_idx shape: {c2f_idx_knn.shape}")
    print(f"KNN - f2c_idx: {pts[f2c_idx_knn].transpose([0,2,1]).shape}, c2f_idx: {pts_fps[c2f_idx_knn].transpose([0,2,1]).shape}")
    # print(f"NN - f2c_idx: {pts[f2c_idx_nn]}")


    # NN 方法执行时间: 0.3867 秒
    # NN - f2c_idx shape: torch.Size([24000]), c2f_idx shape: torch.Size([100000])
    # NN - f2c_idx: torch.Size([24000, 3]), c2f_idx: torch.Size([100000, 3])
    # KNN 方法执行时间: 0.6090 秒
    # KNN - f2c_idx shape: torch.Size([24000, 5]), c2f_idx shape: torch.Size([100000, 5])
    # KNN - f2c_idx: torch.Size([24000, 5, 3]), c2f_idx: torch.Size([100000, 5, 3])


if __name__ == "__main__":
    example_usage()