from typing import Dict, Literal, TypedDict, Union, Callable, Optional

from .PointSamplingIndexer import NNPointSamplingIndexer
from .t3dsPlus_transforms import FPSSample, QSlimSample, FPSQSlimSample, YMinMaxNormalize, \
    PoseNormalize, InstanceCentroids
import torch_geometric.transforms as T


class PrepCfg(TypedDict):
    """预处理配置类型定义"""
    transform: Optional[T.BaseTransform]
    pre_transform: Optional[T.BaseTransform]
    pre_filter: Optional[Callable]

class TXStyles:
    """Transform样式管理器，提供预定义的Transform组合"""

    # 预定义的Transform样式
    DEFAULT_NORMALIZATION = T.Compose([
        YMinMaxNormalize(),
        PoseNormalize(),
        InstanceCentroids(),
    ])

    MINIMAL_NORMALIZATION = T.Compose([
        YMinMaxNormalize(),
    ])

    # ZSCORE_NORMALIZATION = T.Compose([
    #     ZSCORENormalize(),
    #     PoseNormalize(),
    #     InstanceCentroids(),
    # ])

    FULL_NORMALIZATION = T.Compose([
        YMinMaxNormalize(),
        PoseNormalize(),
        InstanceCentroids(),
    ])

    FPS_24000_QSLIM_10000 = T.Compose([
        YMinMaxNormalize(),
        PoseNormalize(),
        FPSQSlimSample(num_classes=15, num_fps_points=24000, tar_vert_num=10000,
                       fps_sam_indexer=NNPointSamplingIndexer(),
                       qslim_sam_indexer=NNPointSamplingIndexer()),
        InstanceCentroids(),
    ])

    FPS_24000 = T.Compose([
        YMinMaxNormalize(),
        PoseNormalize(),
        FPSSample(num_classes=15, num_fps_points=24000, sam_indexer=NNPointSamplingIndexer()),
        InstanceCentroids(),
    ])

    FPS_1000 = T.Compose([
        YMinMaxNormalize(),
        PoseNormalize(),
        FPSSample(num_classes=15, num_fps_points=1000, sam_indexer=NNPointSamplingIndexer()),
        InstanceCentroids(),
    ])

    QSLIM_10000 = T.Compose([
        YMinMaxNormalize(),
        PoseNormalize(),
        QSlimSample(num_classes=15, tar_vert_num=10000, sam_indexer=NNPointSamplingIndexer()),
        InstanceCentroids(),
    ])

    QSLIM_24000 = T.Compose([
        YMinMaxNormalize(),
        PoseNormalize(),
        QSlimSample(num_classes=15, tar_vert_num=10000),
        InstanceCentroids(),
    ])



class PrepCombs:
    """
    预处理组合管理类，提供类型安全的预处理配置访问
    """
    # 内部存储所有预处理组合 - 这是唯一的数据源
    _COMBS: Dict[str, PrepCfg] = {
        "comb_default": {
            "transform": None,
            "pre_transform": TXStyles.DEFAULT_NORMALIZATION,
            "pre_filter": None
        },
        "comb_origin": {
            "transform": None,
            "pre_transform": TXStyles.DEFAULT_NORMALIZATION,
            "pre_filter": None
        },
        "comb_fps_10000": {
            "transform": None,
            "pre_transform": FPSSample(num_classes=15, num_fps_points=10000),
            "pre_filter": None
        },
        "comb_fps_24000": {
            "transform": None,
            "pre_transform": FPSSample(num_classes=15, num_fps_points=24000),
            "pre_filter": None
        },
        "comb_fps_24000_qslim_10000": {
            "transform": None,
            "pre_transform": FPSQSlimSample(num_classes=15, num_fps_points=24000, tar_vert_num=10000),
            "pre_filter": None
        },
        "comb_qslim_10000": {
            "transform": None,
            "pre_transform": QSlimSample(num_classes=15, tar_vert_num=10000),
            "pre_filter": None
        },
        "comb_qslim_5000": {
            "transform": None,
            "pre_transform": QSlimSample(num_classes=15, tar_vert_num=5000),
            "pre_filter": None
        }
    }

    # 基于 _COMBS 的键动态生成类型
    COMBS = Literal[tuple(_COMBS.keys())]

    @classmethod
    def get_comb(cls, comb_name: COMBS) -> PrepCfg:
        """
        获取指定名称的预处理配置
        :param comb_name: 预处理组合名称
        :return: 预处理配置字典
        """
        if comb_name not in cls._COMBS:
            raise ValueError(f"Unknown combination name: {comb_name}")
        return cls._COMBS[comb_name]

    @classmethod
    def get_all_combs(cls) -> Dict[COMBS, PrepCfg]:
        """获取所有预处理组合配置"""
        return cls._COMBS.copy()


# 兼容旧代码的全局变量
prep_combs = PrepCombs._COMBS
