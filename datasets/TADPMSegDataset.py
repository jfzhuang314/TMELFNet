"""
本数据集类采用TGN方式加载数据集
接口要求：
1. def load_dataset(sample_ratio=0.1, train_val_ratio=0.8, comb_name ='comb_default', is_full = False, retain_file_path=None, exclude_file_path=None,rseed = 42):
    加载数据集
    :param sample_ratio: 采样比例
    :param train_val_ratio: 训练集比例
    :param comb_name: 组合名称，包括origin、qslim、fps、bdl等。每种组合对应不同的预处理方式，需在相关类体中进行详尽说明。
    :param is_full: 是否使用全部数据（即upper及lower）
    :param retain_file_path: 保留文件路径
    :param exclude_file_path: 排除文件路径
    :param rseed: 随机种子
    :return: 训练集、验证集、测试集
"""
import re
from random import random

from omegaconf import DictConfig
from torch.utils.data._utils.collate import default_collate
from torch_geometric.loader import DataLoader

from datasets.transform.t3dsPlus_transforms import *
from datasets.transform.teethland_transforms import ToTensor
from datasets.transform.PrepCombs import *

import json
import os
import numpy as np
import torch
from configs.hydra_yaml_config import get_yaml_config, print_yaml_cfg_dict, print_yaml_config, \
    instantiate_from_config, instantiate_from_yaml_config
import open3d as o3d
from scipy.spatial import KDTree
from datasets.BaseSegDataset import BaseSegDataset, BaseSegDataModule, BaseSegFileCollector, BaseSegFileSplitter, \
    DataJsonWriter, DataPrinter

"""
几点说明：
1. "E:\mesh_datasets\3DTeeth\data"中除了原始json文件外，不能新增其他json文件，否则会导致数据集加载错误。
2. 如果有新增其他json文件，请放置到save_jsondata_path文件夹中，不能是save_data_path的子文件夹。
"""
class TADPMSegReader:
    def __init__(self,
                 is_filter_small_instance: bool = True,
                 filter_small_instance_size: int = 30,
                 verbose: bool = True):
        """
        初始化 TADPMSegReader
        :param is_filter_small_instance: 是否过滤小实例
        :param filter_small_instance_size: 过滤小实例的大小
        :param verbose: 是否打印详细信息
        """
        self.is_filter_small_instance = is_filter_small_instance
        self.filter_small_instance_size = filter_small_instance_size
        self.verbose = verbose

    def __call__(self, stl_file: str, json_file: str):
        """
        读取STL文件及JSON元数据
        :param stl_file: STL文件路径
        :param json_file: JSON元数据文件路径
        :param jaw: 颌骨类型
        :return: 包含顶点、类标、实例等信息的Data对象
        """
        try:
            # 读取STL网格数据
            vertices, faces, normals = self.read_stl(stl_file)
            # 获取颌骨类型
            jaw = self.get_jaw_from_path(stl_file)

            # 读取分割标注
            labels, instances = self.read_seg_json(json_file, vertices, jaw)
            # 过滤小实例
            labels, instances = self.filter_small_instance(labels, instances)

            # 打印相关信息
            if self.verbose and self.is_filter_small_instance:
                self.print_msg(json_file, labels, instances)

            # 构建数据字典
            vertices = np.asarray(vertices)
            triangles = np.asarray(faces)
            normals = np.asarray(normals)
            # normals = np.asarray(vertices[:, 3:6]) if len(vertices[:, 3:6]) > 0 else np.zeros_like(vertices)

            # 转换为data_dict字典，字段上匹配Data类的字段
            data_dict = {}
            data_dict['pos'] = np.array(vertices[:, :3])                            # 顶点坐标 [num_vertices, 3], float32
            data_dict['face'] = np.array(triangles).T                               # 三角形索引 [3,num_faces], int32
            data_dict['normals'] = np.array(normals)                               # 顶点法线 [num_vertices, 3], float32
            data_dict['y'] = np.array(labels)                                      # 顶点类标 [num_vertices, 1], int64
            data_dict['instances'] = np.array(instances)                           # 顶点实例 [num_vertices], int32
            data_dict['filename'] = os.path.basename(stl_file).split('.')[0]       # 文件名 [1], str
            data_dict['jaw'] = jaw  # 颌骨类型 [1], str

            # pyG需要转为torch tensor
            data_tensor = ToTensor()(**data_dict)                                  # 转换为torch tensor
            data = Data(**data_tensor)                                             # 转换为Data类
            # data.edge_index = FaceToEdge()(data.face.T)

            # # 修正：先确保 face 是 [3, N] 格式，然后转置为 [N, 3] 再转回来
            # face_for_conversion = data.face  # 转为 [N, 3] 格式
            temp_data = Data(pos=data.pos, face=data.face)  # 创建临时数据对象
            converted_edges = FaceToEdge()(temp_data)  # 应用转换
            data.edge_index = converted_edges.edge_index  # 获取边索引

            return data

            return data
        except Exception as e:
            print(f"读取文件时发生错误: stl_file={stl_file}, json_file={json_file}")
            print(f"错误详情: {str(e)}")
            raise

    def get_jaw_from_path(self,file_path: str) -> str:
        """从文件路径提取颌骨类型"""
        filename = Path(file_path).name

        if '_U_Ori.stl' in filename:
            return 'upper'
        elif '_L_Ori.stl' in filename:
            return 'lower'
        else:
            raise ValueError(f"无法识别文件名格式: {filename}")

    def _check_file_exists(self, file_path: str) -> str:
        """校验文件是否存在，不存在则抛出异常"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        return file_path

    def _validate_jaw_type(self, jaw: str) -> str:
        """校验颌骨类型是否为upper/lower，保证数据合法性"""
        if jaw not in ["upper", "lower", "L", "U"]:
            raise ValueError(f"无效的颌骨类型: {jaw}，仅支持'upper'/'lower'/'L'/'U'")
        return jaw

    def read_stl(self, stl_file: str):
        """
        读取STL网格文件
        :param stl_file: STL文件路径
        :return: 预处理后的网格对象
        """
        try:
            self._check_file_exists(stl_file)

            # 加载STL网格
            mesh = o3d.io.read_triangle_mesh(stl_file, print_progress=False)

            if mesh.is_empty():
                raise ValueError(f"加载的网格为空: {stl_file}")

            # 预处理：修复网格
            mesh.remove_duplicated_vertices()  # 移除重复顶点
            mesh.remove_duplicated_triangles()  # 移除重复三角面
            mesh.compute_vertex_normals()  # 计算法向量
            mesh.compute_triangle_normals()

            vertices = np.asarray(mesh.vertices)    # 模型所有顶点的坐标（#V×3）
            triangles = np.asarray(mesh.triangles)  # 模型所有三角面的索引（#F×3）
            normals = np.asarray(mesh.vertex_normals)

            return vertices, triangles,normals
        except Exception as e:
            if self.verbose:
                print(f"读取STL文件失败: {stl_file}, 错误: {str(e)}")
            raise

    def read_seg_json(self, json_file: str, vertices, jaw: str):
        """
        读取分割JSON元数据
        :param json_file: JSON元数据文件路径
        :param num_vertices: 网格顶点数量，用于初始化标签和实例数组
        :return: seg_metadata, labels, instances
        """
        try:
            self._check_file_exists(json_file)

            # 加载JSON元数据
            with open(json_file, "r", encoding="utf-8") as f:
                seg_metadata = json.load(f)

            # 初始化标签和实例数组
            num_vertices = len(vertices)
            labels = np.zeros(num_vertices, dtype=np.int64)
            instances = np.zeros(num_vertices, dtype=np.int32)

            # 构建KDTree用于近邻查找
            tree = KDTree(vertices)

            # 遍历分割数据
            segmentation = seg_metadata.get("segmentation", {})
            # 遍历每个牙位，给对应顶点上色
            for tooth_id, data in segmentation.items():
                # 获取该牙位对应的顶点坐标列表
                tooth_vertices = np.array(data["vertices"])

                # 使用KDTree快速查找最近的顶点索引
                distances, indices = tree.query(tooth_vertices, k=1, distance_upper_bound=1e-4)
                valid_indices = indices[distances != np.inf]
                instances[valid_indices] = tooth_id

            # 获取颌骨类型
            jaw = self._validate_jaw_type(jaw)

            # 类标映射, 48类映射到15类
            labels = MapUtils.seg_label48_to_label15(instances, jaw)

            return labels.reshape(-1, 1), instances
        except Exception as e:
            if self.verbose:
                print(f"读取JSON分割文件失败: {json_file}, 错误: {str(e)}")
            raise

    def filter_small_instance(self, labels: np.ndarray, instances: np.ndarray):
        """
        过滤小实例
        :param labels: 标签数组
        :param instances: 实例数组
        :return: 过滤后的labels, instances
        """
        try:
            if self.is_filter_small_instance:
                if labels is None or instances is None:
                    raise ValueError("Labels或Instances为None")

                if len(labels) != len(instances):
                    raise ValueError(f"Labels和Instances长度不匹配: labels_len={len(labels)}, instances_len={len(instances)}")

                _, instances, counts = np.unique(instances, return_inverse=True, return_counts=True)
                mask = (counts < self.filter_small_instance_size)[instances]
                labels[mask] = 0
                instances[mask] = 0
                _, instances = np.unique(instances, return_inverse=True)

            return labels, instances
        except Exception as e:
            if self.verbose:
                print(f"过滤小实例时发生错误: {str(e)}")
            raise

    def print_msg(self, json_file: str, labels: np.ndarray,instances: np.ndarray):
        """
        打印相关信息
        :param json_file: json文件路径
        :param labels: 标签数组
        :param instances: 实例数组
        """
        try:
            unique_instances = np.unique(instances)
            unique_labels = np.unique(labels)

            if self.verbose:
                print(f"{json_file} - 标签数: {len(unique_labels)}, 实例数: {len(unique_instances)}")
                print(f"实例统计: {np.bincount(instances.flatten()) if len(unique_instances) <= 20 else '实例过多，不显示详细统计'}")
        except Exception as e:
            if self.verbose:
                print(f"打印信息时发生错误: {str(e)}")



    def __repr__(self) -> str:
        """自定义打印格式，方便调试"""
        return (
            f"{self.__class__.__name__}(\n"
            f"  is_filter_small_instance={self.is_filter_small_instance},\n"
            f"  filter_small_instance_size={self.filter_small_instance_size}\n"
            f")"
        )

class TADPMSegDataset(BaseSegDataset):
    def __init__(self, root: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 pretrans_name: Optional[str] = None,
                 files: Optional[Tuple[List[Path], List[Path], List[Path]]] = None,
                 is_filter_small_instance: Optional[bool] = True,
                 filter_small_instance_size: Optional[int] = 30,
                 recompute: bool = False
                 ):
        """
        初始化 T3DSPlusDataset 类。
        :param root: 数据集根目录
        :param transform: 转换函数
        :param pre_transform: 预处理转换函数
        :param pre_filter: 预处理过滤函数
        :param pretrans_name: 组合名称，用以保存pre_transform后的文件
        :param files: 待处理文件名列表,为三个列表,分别为obj,seg_json,lm_json文件列表
        :param is_filter_small_instance: 是否过滤小实例
        :param filter_small_instance_size: 过滤小实例的大小
        """


        reader = TADPMSegReader(is_filter_small_instance=is_filter_small_instance,
                                          filter_small_instance_size=filter_small_instance_size)

        super(TADPMSegDataset, self).__init__(root, transform,
                                              pre_transform,
                                              pre_filter,
                                              pretrans_name,
                                              files,
                                              is_filter_small_instance,
                                              filter_small_instance_size,
                                              recompute,
                                              reader)

        # 初始化T3DSPlusSegReader

        assert self.mesh_reader is not None

    @property
    def processed_dir(self):
        return os.path.join(self.root, TADPMSegDataset.__name__, 'processed', self.pretrans_name)

########################################################################
# 主DataModule类

class TADPMSegDataModule(BaseSegDataModule):
    def __init__(self,
                 sample_ratio=0.01,  # 采样比例
                 save_data_root: str = r"data/processed",  # T3DSPlus数据集保存路径root
                 pretrans_name='comb_default',  # 组合名称，用以保存pre_transform后的文件
                 pre_transform: Optional[Callable] = None,  # 预处理转换函数
                 pre_filter: Optional[Callable] = None,
                 transform: Optional[Callable] = None,
                 val_transform: Optional[Callable] = None,
                 rseed=42,
                 batch_size=1,
                 num_workers=4,
                 pin_memory: bool = True,
                 persistent_workers: bool = True,
                 recompute: bool = False,
                 is_filter_small_instance: bool = True,
                 filter_small_instance_size: int = 30,
                 file_collector: Optional[Callable[..., Any]] = None,  # 文件收集器
                 file_splitter: Optional[Callable[..., Any]] = None  # 文件划分器
                 ):

        super(TADPMSegDataModule, self).__init__(sample_ratio=sample_ratio,
                                                 save_data_root=save_data_root,
                                                 pretrans_name=pretrans_name,
                                                 pre_transform=pre_transform,
                                                 pre_filter=pre_filter,
                                                 transform=transform,
                                                 val_transform=val_transform,
                                                 rseed=rseed,
                                                 batch_size=batch_size,
                                                 num_workers=num_workers,
                                                 pin_memory=pin_memory,
                                                 persistent_workers=persistent_workers,
                                                 recompute=recompute,
                                                 is_filter_small_instance=is_filter_small_instance,
                                                 filter_small_instance_size=filter_small_instance_size,
                                                 file_collector=file_collector,
                                                 file_splitter=file_splitter)


    def setup(self, stage=None):
        """
        设置数据集（每个GPU都会执行）
        stage: 'fit'/'validate'/'test'/'predict'
        """

        # 收集所有匹配的文件
        if self.file_collector is not None:
            obj_files, seg_json_files = self.file_collector.collect()
        else:
            obj_files, seg_json_files = [], [], []

        # 划分训练集和验证集
        if self.file_splitter is not None:
            train_files_fold, val_files_fold = self.file_splitter.split(
                obj_files, seg_json_files
            )
        else:
            train_files_fold, val_files_fold = [], []

        # 根据sample_ratio对训练集和验证集进行采样
        if self.sample_ratio < 1.0:
            # 设置随机种子以确保可重复性
            random.seed(self.rseed)
            np.random.seed(self.rseed)

            # 对训练集进行采样
            if len(train_files_fold) > 0:
                train_sample_size = max(1, int(len(train_files_fold) * self.sample_ratio))
                train_indices = random.sample(range(len(train_files_fold)), train_sample_size)
                train_files_fold = [train_files_fold[i] for i in train_indices]
            else:
                print("警告: 训练集为空，无法进行采样")

            # 对验证集进行采样
            if len(val_files_fold) > 0:
                val_sample_size = max(1, int(len(val_files_fold) * self.sample_ratio))
                val_indices = random.sample(range(len(val_files_fold)), val_sample_size)
                val_files_fold = [val_files_fold[i] for i in val_indices]
            else:
                print("警告: 验证集为空，无法进行采样")

        # 根据阶段设置数据集
        if stage == 'fit' or stage is None:
            self.train_ds = TADPMSegDataset(
                root= self.save_data_root,
                transform= self.transform,
                pre_transform= self.pre_transform,
                pre_filter= self.pre_filter,
                pretrans_name= self.pretrans_name,
                files= train_files_fold,
                is_filter_small_instance= self.is_filter_small_instance,
                filter_small_instance_size= self.filter_small_instance_size,

                recompute= self.recompute,

            )
            self.val_ds = TADPMSegDataset(
                root= self.save_data_root,
                transform= self.val_transform,
                pre_transform= self.pre_transform,
                pre_filter= self.pre_filter,
                pretrans_name= self.pretrans_name,
                files= val_files_fold,
                is_filter_small_instance=self.is_filter_small_instance,
                filter_small_instance_size=self.filter_small_instance_size,
                recompute= self.recompute,

            )

        if stage == 'test' or stage is None:
            # 对于测试集，我们暂时使用验证集文件（可以根据需要修改）
            self.test_ds = TADPMSegDataset(
                root= self.save_data_root,
                transform= None,
                pre_transform= self.pre_transform,
                pre_filter= self.pre_filter,
                pretrans_name= self.pretrans_name,
                files= val_files_fold,
                is_filter_small_instance=self.is_filter_small_instance,
                filter_small_instance_size=self.filter_small_instance_size,
                recompute= self.recompute
            )

    def train_dataloader(self):
        """返回训练数据加载器"""
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,  # 训练集打乱
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,  # 使用传入的pin_memory参数
            persistent_workers=self.persistent_workers,  # 使用传入的persistent_workers参数
            collate_fn=self.collate_fn,
            drop_last=True  # 可选：丢弃最后一个不完整的批次
        )

    def val_dataloader(self):
        """返回验证数据加载器"""
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,  # 验证集不打乱
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,  # 使用传入的pin_memory参数
            collate_fn=self.collate_fn
        )

    def test_dataloader(self):
        """返回测试数据加载器"""
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,  # 测试集不打乱
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,  # 使用传入的pin_memory参数
            persistent_workers=self.persistent_workers,  # 使用传入的persistent_workers参数
            collate_fn=self.collate_fn
        )

#     def collate_fn(self, batch):
#         # 设定模型要求的固定点数
#         print(">>> Custom collate_fn is called! <<<")  # 加上这句
#
#         target_points = 16000
#         if hasattr(self, 'pretrans_name') and self.pretrans_name:
#             match = re.search(r'(\d+)', self.pretrans_name)
#             if match:
#                 target_points = int(match.group(1))
#
#         aligned_batch = []
#
#         for data in batch:
#             current_points = data.pos.shape[0]
#
#             if current_points < target_points:
#                 # 上采样：先取所有点，再随机填充剩余部分
#                 data = self.up_sampling(data, current_points, target_points)
#
#             aligned_batch.append(data)
#
#         # 使用 PyG 的标准方式合并已对齐的数据
#         return Batch.from_data_list(aligned_batch)
#
#     def up_sampling(self, data, current_points, target_points):
#         extra_needed = target_points - current_points
#         idx = torch.cat([
#             torch.arange(current_points),
#             torch.randint(0, current_points, (extra_needed,))
#         ])
#         # 根据索引对齐所有与点相关的属性
#         data.pos = data.pos[idx]
#         if hasattr(data, 'x') and data.x is not None:
#             data.x = data.x[idx]
#         if hasattr(data, 'y') and data.y is not None:
#             data.y = data.y[idx]
#         if hasattr(data, 'normals') and data.normals is not None:
#             data.normals = data.normals[idx]
#         if hasattr(data, 'instances') and data.instances is not None:
#             data.instances = data.instances[idx]
#         # 注意：点数改变后，原有的 edge_index 或 face 通常会失效
#         # 建议在模型 forward 时重新构建 KNN 图，或者在这里清空让模型处理
#         if hasattr(data, 'edge_index'):
#             del data.edge_index
#         if hasattr(data, 'face'):
#             del data.face
#
#         return data





def load_datasets_from_cfg(config_path = "../configs",config_name = "test_TADPMSegDataset"):
    cfg = get_yaml_config(config_path=config_path, config_name=config_name)
    print_yaml_config(config_path=config_path, config_name=config_name)

    file_collecter = BaseSegFileCollector(
        obj_data_path=cfg.paths.source_obj_data_path,
        seg_json_data_path=cfg.paths.source_seg_json_data_path,
        ext="stl",
        exclude=cfg.dataset.exclude_files
    )
    file_splitter = BaseSegFileSplitter(
        fold_path=cfg.dataset.folds.fold_path,
        fold_filename=cfg.dataset.folds.fold_filename,
        include_val_as_train=False
    )

    pre_transform = instantiate_from_config(cfg.dataset.pre_transform_obj)
    transform = instantiate_from_config(cfg.dataset.transform_obj)
    val_transform = instantiate_from_config(cfg.dataset.val_transform_obj)
    pre_filter = instantiate_from_config(cfg.dataset.pre_filter_obj)

    dm = TADPMSegDataModule(
        save_data_root=cfg.paths.save_data_path,
        pretrans_name=cfg.dataset.pretrans_name,
        pre_transform=pre_transform,
        transform=transform,
        val_transform=val_transform,
        pre_filter=pre_filter,
        sample_ratio=cfg.dataset.sample_ratio,
        batch_size=cfg.dataset.batch_size,
        rseed=cfg.dataset.rseed,
        num_workers=cfg.dataset.num_workers,
        pin_memory=cfg.dataset.pin_memory,
        persistent_workers=cfg.dataset.persistent_workers,
        recompute=cfg.dataset.recompute,
        is_filter_small_instance=cfg.dataset.is_filter_small_instance,
        filter_small_instance_size=cfg.dataset.filter_small_instance_size,
        file_collector=file_collecter,
        file_splitter=file_splitter)

    dm.setup(stage='fit')

    return dm, dm.train_dataloader(), dm.val_dataloader()


def get_dataloaders_from_cfg_obj(cfg: DictConfig, only_val=False):

    file_collecter = BaseSegFileCollector(
        obj_data_path=cfg.paths.source_obj_data_path,
        seg_json_data_path=cfg.paths.source_seg_json_data_path,
        ext="stl",
        exclude=cfg.dataset.exclude_files
    )
    file_splitter = BaseSegFileSplitter(
        fold_path=cfg.dataset.folds.fold_path,
        fold_filename=cfg.dataset.folds.fold_filename,
        include_val_as_train=False
    )

    pre_transform = cfg.dataset.pre_transform_obj
    transform = cfg.dataset.transform_obj
    val_transform = cfg.dataset.val_transform_obj
    pre_filter = cfg.dataset.pre_filter_obj

    dm = TADPMSegDataModule(
        save_data_root=cfg.paths.save_data_path,
        pretrans_name=cfg.dataset.pretrans_name,
        pre_transform=pre_transform,
        transform=transform,
        val_transform=val_transform,
        pre_filter=pre_filter,
        sample_ratio=cfg.dataset.sample_ratio,
        batch_size=cfg.dataset.batch_size,
        rseed=cfg.dataset.rseed,
        num_workers=cfg.dataset.num_workers,
        pin_memory=cfg.dataset.pin_memory,
        persistent_workers=cfg.dataset.persistent_workers,
        recompute=cfg.dataset.recompute,
        is_filter_small_instance=cfg.dataset.is_filter_small_instance,
        filter_small_instance_size=cfg.dataset.filter_small_instance_size,
        file_collector=file_collecter,
        file_splitter=file_splitter)



    if only_val:
        dm.setup(stage='test')
        return dm, None, dm.test_dataloader()
    else:
        dm.setup(stage='fit')
        return dm, dm.train_dataloader(), dm.val_dataloader()

#########################################################################
#region tests


def test_TADPMDataModule(cfg):

    # comb_name = "comb_default"
    comb_name = cfg.pretrans_comb_name
    file_collecter = BaseSegFileCollector(
        obj_data_path=cfg.paths.source_obj_data_path,
        seg_json_data_path=cfg.paths.source_seg_json_data_path,
        exclude = cfg.dataset.exclude_files
    )
    file_splitter= BaseSegFileSplitter(
        fold_path=cfg.folds.fold_path,
        fold_filename=cfg.folds.fold_filename,
        include_val_as_train=False
    )

    dm = TADPMSegDataModule(
                            save_data_root=cfg.paths.save_data_path,
                            sample_ratio=0.02,
                            batch_size=1,
                            rseed=42,
                            # num_workers=0,
                            # persistent_workers=False,
                            file_collector=file_collecter,
                            file_splitter=file_splitter,
                            comb_name=cfg.dataset.comb_name)
    dm.setup(stage='fit')

    # dataloader batch获取
    train_dl = dm.train_dataloader()
    # 从dataloader中获取一个batch
    batch = next(iter(train_dl))
    print(batch) # 包含BNC格式的点云数据和标签

    # dataset逐一获取
    train_ds = dm.train_ds
    print(len(train_ds))
    pdata = train_ds[0]
    # pdata.face = pdata.face.T
    print(pdata) # BNC格式
    data_dict = pdata.to_dict()
    for key in data_dict.keys():
        print(key)

    pdata2 = train_ds[1]
    # pdata2.face = pdata2.face.T
    print(pdata2) # BNC格式
    data_dict = pdata2.to_dict()
    for key in data_dict.keys():
        print(key)

    # batch 合并
    exclude_keys = ['landmark_coords', 'landmark_classes','affine','instance_centroids','instance_labels']
    data_list = [pdata, pdata2]
    batch = Batch.from_data_list(data_list,exclude_keys=exclude_keys)
    keys = pdata.keys()

    for key in keys:
        # Handle variable-length data (points, faces, normals, etc.)
        if key in ['pos', 'face', 'normals', 'y', 'x','instances','edge_index']:
            continue
        # Handle landmark data separately
        elif key in ['landmark_coords', 'landmark_classes', 'affine', 'instance_centroids', 'instance_labels']:
            batch[key] = [data[key] for data in data_list]
        # Handle scalar/meta data
        else:
            batch[key] = default_collate([data[key] for data in data_list])
    print(batch) # 包含BNC格式的点云数据和标签


    # gu.print_3d(gu.np_to_pcd_with_label(pdata.pos.numpy(), pdata.y.numpy()))
    gu.print_3d(gu.np_to_pcd(pdata.pos.numpy()))


    # 示例用法
    writer = DataJsonWriter()
    json_dir = os.path.join(cfg.paths.save_jsondata_path, TADPMSegDataset.__name__, 'processed', cfg.dataset.comb_name)
    writer(pdata, root=json_dir)

def test_TADPMSegReader():
    def calculate_tensor_memory(data):
        """
        计算 Data 对象中所有 tensor 的总内存大小
        """
        total_memory = 0

        for key, value in data.items():
            if torch.is_tensor(value):
                memory_bytes = value.element_size() * value.nelement()
                total_memory += memory_bytes
                print(f"{key}: {memory_bytes} 字节 ({memory_bytes / 1024:.2f} KB)")

        print(f"所有 tensor 总内存: {total_memory} 字节 ({total_memory / 1024:.2f} KB)")
        return total_memory

    config_path = "../configs"
    config_name = "test_TADPMSegDataset"
    cfg = get_yaml_config(config_path=config_path, config_name=config_name)
    args = cfg.paths
    ds_args = cfg.dataset

    print_yaml_cfg_dict(cfg)

    reader = TADPMSegReader(is_filter_small_instance= ds_args.is_filter_small_instance,
                            filter_small_instance_size=ds_args.filter_small_instance_size,
                            verbose= True)

    filename = "0001/ori/0001_L_Ori"
    filename = "0001/ori/0001_U_Ori"
    # filename = "1180/ori/1180_L_Ori"
    filename = "1180/ori/1180_U_Ori"

    obj_filename = os.path.join(args.source_obj_data_path, filename + ".stl")
    seg_json_filename = os.path.join(args.source_seg_json_data_path, filename + ".json")

    data = reader(obj_filename, seg_json_filename)
    print(data.keys())
    # print(calculate_tensor_memory(data))
    print(data)

    # 修正后的代码
    transform = T.Compose([
        MeanScaleNormalize(60.0, False, True),
        XoYPCAPoseNormalize()
    ])

    data = transform(data)

    vis = instantiate_from_config(cfg.vis.qslim_10000_mesh_label_lighton_saver)
    vis.window_visible = True
    vis.save_file = False
    vis(data)


def test_BaseSegFileCollector():
    config_path = "../configs"
    config_name = "test_TADPMSegDataset"
    cfg = get_yaml_config(config_path=config_path, config_name=config_name)
    print_yaml_config(config_path=config_path, config_name=config_name)

    # pretrans_name = "comb_default"
    file_collecter = BaseSegFileCollector(
        obj_data_path=cfg.paths.source_obj_data_path,
        seg_json_data_path=cfg.paths.source_seg_json_data_path,
        ext="stl",
        exclude=cfg.dataset.exclude_files
    )
    file_splitter = BaseSegFileSplitter(
        fold_path=cfg.dataset.folds.fold_path,
        fold_filename=cfg.dataset.folds.fold_filename,
        include_val_as_train=False
    )

    train_files, val_files = file_collecter.collect()
    train_files_fold, val_files_fold = file_splitter.split(train_files, val_files)

    print(f"基于fold文件的划分结果:")

    print(f"训练集文件数: {len(train_files_fold)}")
    for train_file in train_files_fold:
        print(train_file)

    print(f"验证集文件数: {len(val_files_fold)}")
    for val_file in val_files_fold:
        print(val_file)


def test_TADPMSegDataModule():
    config_path = "../configs"
    config_name = "module_CrossTooth"
    cfg = get_yaml_config(config_path=config_path, config_name=config_name)
    print_yaml_config(config_path=config_path, config_name=config_name)

    # pretrans_name = "comb_default"
    file_collecter = BaseSegFileCollector(
        obj_data_path=cfg.paths.source_obj_data_path,
        seg_json_data_path=cfg.paths.source_seg_json_data_path,
        ext="stl",
        exclude=cfg.dataset.exclude_files
    )
    file_splitter = BaseSegFileSplitter(
        fold_path=cfg.dataset.folds.fold_path,
        fold_filename=cfg.dataset.folds.fold_filename,
        include_val_as_train=False
    )

    obj_files, seg_json_files = file_collecter.collect()
    for obj_file, seg_json_file in zip(obj_files, seg_json_files):
        print(obj_file, seg_json_file)

    train_files, val_files = file_splitter.split(obj_files, seg_json_files)
    for train_file in train_files:
        print(train_file)
    for val_file in val_files:
        print(val_file)

    pre_transform = instantiate_from_config(cfg.dataset.pre_transform_obj)
    transform = instantiate_from_config(cfg.dataset.transform_obj)
    val_transform = instantiate_from_config(cfg.dataset.val_transform_obj)
    pre_filter = instantiate_from_config(cfg.dataset.pre_filter_obj)

    dm = TADPMSegDataModule(
        save_data_root=cfg.paths.save_data_path,
        pretrans_name=cfg.dataset.pretrans_name,
        pre_transform=pre_transform,
        transform=transform,
        val_transform=val_transform,
        pre_filter=pre_filter,
        sample_ratio=cfg.dataset.sample_ratio,
        batch_size=2,
        rseed=cfg.dataset.rseed,
        # num_workers=0,
        # persistent_workers=False,
        file_collector=file_collecter,
        file_splitter=file_splitter)
    dm.setup(stage='fit')

    # dataloader batch获取
    train_dl = dm.train_dataloader()
    # 从dataloader中获取一个batch
    batch = next(iter(train_dl))
    print(batch)  # 包含BNC格式的点云数据和标签

    # dataset逐一获取
    train_ds = dm.train_ds
    print(len(train_ds))
    pdata = train_ds[0]

    pdata2 = train_ds[1]

    DataPrinter(pdata).print()
    DataPrinter(pdata2).print()

    # 可视化
    # gu.print_3d(gu.np_to_pcd(pdata.pos.numpy()))

    # 示例用法
    writer = DataJsonWriter()
    json_dir = os.path.join(cfg.paths.save_jsondata_path, TADPMSegDataset.__name__, 'processed',
                            cfg.dataset.pretrans_name)
    writer(pdata, root=json_dir)

def test_load_datasets_from_cfg():
    config_path = "../configs"
    config_name = "test_TADPMSegDataset"
    dm,train_loader, val_loader = load_datasets_from_cfg(config_path=config_path, config_name=config_name)

    batch = next(iter(train_loader))
    print(batch)

    # dataset逐一获取
    train_ds = dm.train_ds
    print(len(train_ds))
    pdata = train_ds[0]
    pdata2 = train_ds[1]

    DataPrinter(pdata).print()
    DataPrinter(pdata2).print()

    # gu.print_3d(gu.np_to_pcd(pdata.pos.numpy()))

    # 示例用法
    cfg = get_yaml_config(config_path=config_path, config_name=config_name)
    writer = DataJsonWriter()
    json_dir = os.path.join(cfg.paths.save_jsondata_path, TADPMSegDataset.__name__, 'processed',
                            cfg.dataset.pretrans_name)
    writer(pdata, root=json_dir)

def test_get_dataloaders_from_cfg_obj():
    config_path = "../configs"
    config_name = "Module_CrossTooth"
    # cfg = get_yaml_config(config_path=config_path, config_name=config_name)
    # print_yaml_config(config_path=config_path, config_name=config_name)
    cfg_obj = instantiate_from_yaml_config(config_path=config_path, config_name=config_name)
    print(cfg_obj)

    # file_collecter = BaseSegFileCollector(
    #     obj_data_path=cfg_obj.paths.source_obj_data_path,
    #     seg_json_data_path=cfg_obj.paths.source_seg_json_data_path,
    #     ext="stl",
    #     exclude=cfg_obj.dataset.exclude_files
    # )
    # file_splitter = BaseSegFileSplitter(
    #     fold_path=cfg_obj.dataset.folds.fold_path,
    #     fold_filename=cfg_obj.dataset.folds.fold_filename,
    #     include_val_as_train=False
    # )
    #
    # train_files, val_files = file_collecter.collect()
    # train_files_fold, val_files_fold = file_splitter.split(train_files, val_files)
    #
    # print(f"基于fold文件的划分结果:")
    #
    # print(f"训练集文件数: {len(train_files_fold)}")
    # for train_file in train_files_fold:
    #     print(train_file)
    #
    # print(f"验证集文件数: {len(val_files_fold)}")
    # for val_file in val_files_fold:
    #     print(val_file)


    dm, train_loader, test_loader = get_dataloaders_from_cfg_obj(cfg_obj)


    batch = next(iter(train_loader))
    print(batch)

    # dataset逐一获取
    train_ds = dm.train_ds
    print(len(train_ds))
    pdata = train_ds[0]
    pdata2 = train_ds[1]

    DataPrinter(pdata).print()
    DataPrinter(pdata2).print()

#endregion tests


def main():
    # test_TADPMSegReader()
    # test_BaseSegFileCollector()
    test_TADPMSegDataModule()
    # test_load_datasets_from_cfg()
    # test_get_dataloaders_from_cfg_obj()

if __name__ == '__main__':

    main()
