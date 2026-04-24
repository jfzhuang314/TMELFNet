"""
本数据集类采用TGN方式加载数据集
接口要求：
1. def load_dataset(sample_ratio=0.1, train_val_ratio=0.8, pretrans_name ='comb_default', is_full = False, retain_file_path=None, exclude_file_path=None,rseed = 42):
    加载数据集
    :param sample_ratio: 采样比例
    :param train_val_ratio: 训练集比例
    :param pretrans_name: 组合名称，包括origin、qslim、fps、bdl等。每种组合对应不同的预处理方式，需在相关类体中进行详尽说明。
    :param is_full: 是否使用全部数据（即upper及lower）
    :param retain_file_path: 保留文件路径
    :param exclude_file_path: 排除文件路径
    :param rseed: 随机种子
    :return: 训练集、验证集、测试集
"""

from functools import lru_cache
from random import random

from matplotlib import pyplot as plt
from torch.utils.data._utils.collate import default_collate
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

from common.DataModuleFileInterface import FileSplitterInterface, FileCollectorInterface
import pytorch_lightning as pl
from datasets.transform.t3dsPlus_transforms import *
from datasets.transform.teethland_transforms import ToTensor
from datasets.transform.PrepCombs import *

from skmultilearn.model_selection import IterativeStratification
import json
import numpy as np
import torch
# from datasets.configs.get_yaml_config import get_args_from_config, print_yaml_config, get_yaml_config
from configs.hydra_yaml_config import *
import external_libs.pv_gen_utils as pvgu

"""
几点说明：
1. "E:\mesh_datasets\3DTeeth\data"中除了原始json文件外，不能新增其他json文件，否则会导致数据集加载错误。
2. 如果有新增其他json文件，请放置到save_jsondata_path文件夹中，不能是save_data_path的子文件夹。
"""


# parser = argparse.ArgumentParser()
# parser.add_argument('--source_obj_data_path', default=r"E:\mesh_datasets\3DTeeth\data", type=str, help="data path in which original .obj data are saved")
# parser.add_argument('--source_seg_json_data_path', default=r"E:\mesh_datasets\3DTeeth\data", type=str, help="data path in which original seg.json data are saved")
# parser.add_argument('--source_landmarks_json_data_path', default=r"E:\mesh_datasets\3DTeeth\3DTeethLand_landmarks_train", type=str, help="data path in which original landmarks.json data are saved")
# parser.add_argument('--save_data_path', default=r"E:\mesh_datasets\3DTeeth\data", type=str, help="data path in which processed data will be saved")
# parser.add_argument('--save_jsondata_path', default=r"E:\mesh_datasets\3DTeeth\jsondata", type=str, help="json data path in which processed data will be saved")
# args = parser.parse_args()

class T3DSPlusSegReader:
    def __init__(self,
                 is_filter_small_instance: bool = True,
                 filter_small_instance_size: int = 30,
                 verbose: bool = True):
        """
        初始化 T3DSPlusReader
        :param is_filter_small_instance: 是否过滤小实例
        :param filter_small_instance_size: 过滤小实例的大小
        """
        self.is_filter_small_instance = is_filter_small_instance
        self.filter_small_instance_size = filter_small_instance_size
        self.verbose = verbose

    def __call__(self, obj_file: str, seg_json_file: str):
        """
        读取obj及json文件
        :param obj_file: obj文件路径
        :param seg_json_file: seg json文件路径
        :return: 包含顶点、类标、实例、关键点坐标、关键点类别等信息的Data对象
        """
        # 读取网格数据
        try:
            # 读取网格数据
            vertices, org_mesh = self.read_mesh(obj_file)
            # 读取分割标注
            seg_annotation, labels, instances = self.read_seg_json(seg_json_file)
            # 过滤小实例
            labels, instances = self.filter_small_instance(labels, instances)
            # 打印相关信息
            if self.verbose and self.is_filter_small_instance:
                self.print_msg(seg_json_file, labels, instances, seg_annotation)

            # 转换为data_dict字典，字段上匹配Data类的字段
            data_dict = {}
            data_dict['pos'] = np.array(vertices[:, :3])  # 顶点坐标 [num_vertices, 3], float32
            data_dict['face'] = np.array(org_mesh.triangles).T  # 三角形索引 [3,num_faces], int32
            data_dict['normals'] = np.array(vertices[:, 3:6])  # 顶点法线 [num_vertices, 3], float32
            data_dict['y'] = np.array(labels)  # 顶点类标 [num_vertices, 1], int64
            data_dict['instances'] = np.array(instances)  # 顶点实例 [num_vertices], int32
            data_dict['filename'] = os.path.basename(obj_file).split('.')[0]  # 文件名 [1], str
            data_dict['jaw'] = seg_annotation['jaw']  # 颌骨类型 [1], str ('upper' or 'lower')

            # pyG需要转为torch tensor
            data_tensor = ToTensor()(**data_dict)  # 转换为torch tensor
            data = Data(**data_tensor)  # 转换为Data类

            # edge_index 属性
            temp_data = Data(pos=data.pos, face=data.face)  # 创建临时数据对象
            converted_edges = FaceToEdge()(temp_data)  # 应用转换
            data.edge_index = converted_edges.edge_index  # 获取边索引

            return data
        except Exception as e:
            print(f"读取文件时发生错误: obj_file={obj_file}, seg_json_file={seg_json_file}")
            print(f"错误详情: {str(e)}")
            raise

    def _check_file_exists(self, file_path: str) -> str:
        """校验文件是否存在，不存在则抛出异常"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        return file_path

    def _validate_jaw_type(self, jaw: str) -> str:
        """校验颌骨类型是否为upper/lower，保证数据合法性"""
        if jaw not in ["upper", "lower"]:
            raise ValueError(f"无效的颌骨类型: {jaw}，仅支持'upper'或'lower'")
        return jaw

    def read_mesh(self, obj_file: str):
        """
        读取网格文件
        :param obj_file: obj文件路径
        :return: vertices, org_mesh
        """
        vertices, org_mesh = gu.read_txt_obj_ls(obj_file, ret_mesh=True, use_tri_mesh=False)
        return vertices, org_mesh

    def read_seg_json(self, seg_json_file: str):
        """
        读取分割json文件
        :param seg_json_file: seg json文件路径
        :return: seg_annotation, labels, instances
        """
        seg_annotation = gu.load_json(seg_json_file)
        id_patient = seg_annotation['id_patient']
        jaw = self._validate_jaw_type(seg_annotation['jaw'])
        labels = np.array(seg_annotation['labels']).reshape(-1, 1)
        instances = np.array(seg_annotation['instances']).reshape(-1)
        # 类标映射, 48类映射到15类
        labels = MapUtils.seg_label48_to_label15(labels, seg_annotation['jaw'])

        return seg_annotation, labels, instances

    def filter_small_instance(self, labels: np.ndarray, instances: np.ndarray):
        """
        过滤小实例
        :param labels: 标签数组
        :param instances: 实例数组
        :return: 过滤后的labels, instances
        """
        if self.is_filter_small_instance:
            # 获取每个唯一实例的计数,inverse_instances、counts是数组索引编号，方便进行索引掩码操作
            unique_instances, inverse_instances, counts = np.unique(instances,
                                                                    return_inverse=True, return_counts=True)

            counts_mask = counts < self.filter_small_instance_size  # 小于阈值的实例掩码
            small_instance_mask = counts_mask[inverse_instances]  # 生成小实例的实例掩码

            # 将小实例的标签和实例ID都设为0（背景）
            labels[small_instance_mask] = 0
            instances[small_instance_mask] = 0
            # inverse_instances[small_instance_mask] = 0
            # 重新映射实例ID，确保实例ID连续

        return labels, instances

    def print_msg(self, seg_json_file: str, labels: np.ndarray,
                  instances: np.ndarray, seg_annotation: dict):
        """
        打印相关信息
        :param seg_json_file: seg json文件路径
        :param labels: 标签数组
        :param instances: 实例数组
        :param seg_annotation: 分割标注
        """
        if labels.sum() != sum(seg_annotation['labels']):
            unique_instances, inverse_instances, counts = np.unique(instances, return_inverse=True, return_counts=True)
            removed_labels = set(seg_annotation['labels']) - set(labels.flatten().tolist())
            # print(seg_json_file, 'counts', np.sort(counts), 'removed', removed_labels)
            print(seg_json_file, 'counts', np.sort(counts), 'removed', removed_labels,
                  'instances_unique', unique_instances)

    def __repr__(self) -> str:
        """自定义打印格式，方便调试"""
        return (
            f"{self.__class__.__name__}(\n"
            f"  is_filter_small_instance={self.is_filter_small_instance},\n"
            f"  filter_small_instance_size={self.filter_small_instance_size}\n"
            f")"
        )


class T3DSPlusSegDataset(Dataset):
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
        self.root = root
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.pretrans_name = pretrans_name

        # 过滤小实例
        self.is_filter_small_instance = is_filter_small_instance
        self.filter_small_instance_size = filter_small_instance_size

        # 正确处理文件列表
        if files and len(files) > 0:
            # files是一个元组列表 [(obj, seg, landmark), ...]
            self.obj_files = [f[0] for f in files]
            self.seg_json_files = [f[1] for f in files]
        else:
            self.obj_files = []
            self.seg_json_files = []

        assert len(self.obj_files) == len(self.seg_json_files)

        # 初始化T3DSPlusSegReader
        self.mesh_reader = T3DSPlusSegReader(is_filter_small_instance, filter_small_instance_size)
        assert self.mesh_reader is not None

        # 需设定是否重新计算，用以调试测试代码
        super(T3DSPlusSegDataset, self).__init__(root,
                                                 force_reload=recompute,
                                                 pre_transform=pre_transform,
                                                 transform=transform,
                                                 pre_filter=pre_filter)

    def len(self):
        # 根据obj_files决定数据集长度
        return len(self.obj_files)

    @property
    def processed_dir(self):
        return os.path.join(self.root, T3DSPlusSegDataset.__name__, 'processed', self.pretrans_name)

    @property
    def processed_file_names(self):
        pt_filenames = []
        for obj_file in self.obj_files:
            filename = Path(obj_file).stem
            pt_filenames.append(f"{filename}.pt")
        return pt_filenames

    @lru_cache(maxsize=256)
    def _read_data(self, idx):
        """
        读取obj及json文件
        :param idx: 数据索引
        :return: 包含顶点、类标、实例、关键点坐标、关键点类别等信息的Data对象
        """
        # 读取obj及json文件
        obj_file = self.obj_files[idx]
        seg_json_file = self.seg_json_files[idx]
        # lm_json_file = self.lm_json_files[idx]

        data = self.mesh_reader(obj_file, seg_json_file)
        return data

    @lru_cache(maxsize=512)
    def _load_mesh_pt(self, idx):
        file_name = self.processed_file_names[idx]
        path = os.path.join(self.processed_dir, file_name)
        return torch.load(path, weights_only=False)  # 仅加载当前样本

    @lru_cache(maxsize=512)
    def get(self, idx):
        data = self._load_mesh_pt(idx)

        if self.transform is not None:
            data = self.transform(data)
        return data

    def process(self):
        for idx, file_name in enumerate(self.processed_file_names):
            processed_path = os.path.join(self.processed_dir, file_name)
            # 考虑是否重新计算预处理文件，主要用以调试测试代码
            if not os.path.exists(processed_path) or self.force_reload:

                # 获取原始数据
                raw_data = self._read_data(idx)  # mesh_obj

                # 应用 pre_transform
                if self.pre_transform is not None:
                    processed_data = self.pre_transform(raw_data)
                else:
                    processed_data = raw_data

                # 应用 pre_filter
                if self.pre_filter is not None and not self.pre_filter(processed_data):
                    continue

                torch.save(processed_data, processed_path)
                print(f"Saved {file_name} to {processed_path}")
            else:
                print(f"{file_name} already exists in {processed_path}, skipping...")


########################################################################
# 主DataModule类
# ... existing code ...
class T3DSPlusSegDataModule(pl.LightningDataModule):
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
        super().__init__()
        # self.args = args
        self.sample_ratio = sample_ratio
        self.pretrans_name = pretrans_name
        self.save_data_root = save_data_root
        self.rseed = rseed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.file_collector: Optional[FileCollectorInterface] = file_collector
        self.file_splitter: Optional[FileSplitterInterface] = file_splitter

        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.transform = transform
        self.val_transform = val_transform

        # 初始化数据集存储
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

        self.is_filter_small_instance = is_filter_small_instance
        self.filter_small_instance_size = filter_small_instance_size
        self.recompute = recompute
        # 固定随机种子
        self._set_seeds()

    def _set_seeds(self):
        """设置随机种子以确保可重复性"""
        random.seed(self.rseed)
        np.random.seed(self.rseed)

    def prepare_data(self):
        # 如果需要下载/预处理数据，放在这里
        pass

    def setup(self, stage=None):
        """
        设置数据集（每个GPU都会执行）
        stage: 'fit'/'validate'/'test'/'predict'
        """
        # 获取预处理函数
        # pre_transform = self.pre_transform
        # pre_filter = self.pre_filter
        # transform = self.transform

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
            self.train_ds = T3DSPlusSegDataset(
                root= self.save_data_root,
                transform= self.transform,
                pre_transform= self.pre_transform,
                pre_filter= self.pre_filter,
                pretrans_name= self.pretrans_name,
                files= train_files_fold,
                is_filter_small_instance= self.is_filter_small_instance,
                filter_small_instance_size= self.filter_small_instance_size,
                recompute= self.recompute
            )
            self.val_ds = T3DSPlusSegDataset(
                root= self.save_data_root,
                transform= self.val_transform,
                pre_transform= self.pre_transform,
                pre_filter= self.pre_filter,
                pretrans_name= self.pretrans_name,
                files= val_files_fold,
                is_filter_small_instance=self.is_filter_small_instance,
                filter_small_instance_size=self.filter_small_instance_size,
                recompute= self.recompute
            )

        if stage == 'test' or stage is None:
            # 对于测试集，我们暂时使用验证集文件（可以根据需要修改）
            self.test_ds = T3DSPlusSegDataset(
                root= self.save_data_root,
                transform= self.val_transform,
                pre_transform= self.pre_transform,
                pre_filter= self.pre_filter,
                pretrans_name= self.pretrans_name,
                files= val_files_fold,
                is_filter_small_instance=self.is_filter_small_instance,
                filter_small_instance_size=self.filter_small_instance_size,
                recompute= self.recompute
            )

    def collate_fn(self, batch):
        # Create a dictionary to hold the collated data
        collated = {}

        # Get all keys from the first sample
        keys = batch[0].keys()

        for key in keys:
            # Handle variable-length data (points, faces, normals, etc.)
            if key in ['pos', 'face', 'normals', 'y', 'instances']:
                # Stack as lists of tensors (won't try to make them equal size)
                collated[key] = [data[key] for data in batch]
            # Handle landmark data separately
            elif key in ['landmark_coords', 'landmark_classes']:
                collated[key] = [data[key] for data in batch]
            # Handle scalar/meta data
            else:
                collated[key] = default_collate([data[key] for data in batch])

        return collated

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
            persistent_workers=self.persistent_workers,  # 使用传入的persistent_workers参数
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

    def predict_dataloader(self):
        """返回验证数据加载器"""
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,  # 验证集不打乱
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,  # 使用传入的pin_memory参数
            collate_fn=self.collate_fn
        )

    def teardown(self, stage=None):
        """清理资源"""
        pass


########################################################################
# utils
class DataJsonWriter:
    def __call__(self, data, root):
        """
        将Data对象写入json文件。当前仅支持写入tensor, numpy数组, 列表, 一级字典等可序列化类型。
        :param data: Data对象，包含filename属性
        :param root: json文件路径根目录
        :return: None
        """
        # 拼接输出路径
        output_path = os.path.join(root, data.filename + ".json")

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 转换Data对象为可序列化的字典
        data_dict = {}
        for key, value in data:
            if isinstance(value, torch.Tensor):
                data_dict[key] = value.cpu().numpy().tolist()
            elif isinstance(value, np.ndarray):
                data_dict[key] = value.tolist()
            elif isinstance(value, dict):
                # 递归处理嵌套字典
                data_dict[key] = {
                    str(k): v.cpu().numpy().tolist() if isinstance(v, torch.Tensor)
                    else v.tolist() if isinstance(v, np.ndarray)
                    else list(v) if isinstance(v, (list, tuple))
                    else v
                    for k, v in value.items()
                }
            else:
                data_dict[key] = value

        # 写入json文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data_dict, f, ensure_ascii=False, indent=2)


class T3DSPlusSegFileCollector(FileCollectorInterface):
    """
    网格文件收集器类，用于从数据集目录中提取匹配的 obj 和 json 文件
    """

    def __init__(self,
                 obj_data_path: str,
                 seg_json_data_path: str,
                 exclude: List[str] = []):
        """
        初始化文件收集器
        :param obj_data_path: obj 文件所在目录
        :param seg_json_data_path: seg.json 文件所在目录
        :param exclude: 需要排除的文件名列表
        """
        self.obj_data_path = Path(obj_data_path)
        self.seg_json_data_path = Path(seg_json_data_path)
        self.exclude = exclude

    def collect(self) -> Tuple[List[str], List[str], List[str]]:
        """
        收集匹配的文件列表
        :return: (obj_files, seg_json_files, landmarks_json_files) 三个文件列表
        """
        # 收集所有文件
        obj_files = sorted(self.obj_data_path.glob('**/*.obj'))
        seg_files = sorted(self.seg_json_data_path.glob('**/*.json'))
        # landmark_files = sorted(self.landmarks_json_data_path.glob('**/*.json'))

        # 提取stem并确保三者一致
        obj_stems = set(f.stem for f in obj_files)
        seg_stems = set(f.stem for f in seg_files)

        # 只保留三个集合的交集并排除exclude的指定文件
        common_stems = obj_stems & seg_stems
        common_stems = common_stems - set(self.exclude)

        # 构建最终的文件列表
        result_obj_files = []
        result_seg_json_files = []
        # result_landmarks_json_files = []

        for stem in sorted(common_stems):
            # 获取对应的obj文件
            obj_file = next((f for f in obj_files if f.stem == stem), None)
            # 获取对应的seg文件
            seg_file = next((f for f in seg_files if f.stem == stem), None)

            if obj_file and seg_file:
                result_obj_files.append(str(obj_file))
                result_seg_json_files.append(str(seg_file))

        return (result_obj_files, result_seg_json_files)


class T3DSPlusSegFileSplitter(FileSplitterInterface):
    """
    T3DSPlus数据集文件划分器
    """

    def __init__(self,
                 fold_path: Optional[str] = None,
                 fold_filename: Optional[str] = None,
                 include_val_as_train: bool = False):
        """
        初始化文件划分器
        :param obj_files: obj文件列表
        :param seg_json_files: seg json文件列表
        :param landmarks_json_files: landmarks json文件列表
        :param fold_path: 验证集fold文件路径
        :param fold_filename: fold文件名
        :param include_val_as_train: 是否将验证集也包含在训练集中
        """
        self.fold_path = fold_path
        self.fold_filename = fold_filename
        self.include_val_as_train = include_val_as_train

    def _split_by_fold_file(self,
                            obj_files: List[str],
                            seg_json_files: List[str]) -> Tuple[List, List]:
        """
        根据指定的fold文件进行划分
        """
        with open(os.path.join(self.fold_path, self.fold_filename), 'r') as f:
            val_mesh_files = [l.strip() for l in f.readlines() if l.strip()]
        # 提取stem用于匹配
        val_mesh_stems = [Path(f).stem for f in val_mesh_files]
        files = list(zip(obj_files, seg_json_files))
        train_files = [fs for fs in files if Path(fs[0]).stem not in val_mesh_stems]
        val_files = [fs for fs in files if Path(fs[0]).stem in val_mesh_stems]

        if self.include_val_as_train:
            train_files += val_files

        return train_files, val_files

    def _split_by_subject_stratification(self,
                                         obj_files: List[str],
                                         seg_json_files: List[str]) -> Tuple[List, List]:
        """
        使用基于受试者的分层划分
        """
        files = list(zip(obj_files, seg_json_files))

        # 确定每个受试者的类别和文件
        subject_idxs = {}
        subject_files = [[] for _ in range(len(files) * 2)]  # 预分配足够空间
        subject_labels = np.zeros((len(files) * 2, 86))  # 预分配足够空间

        valid_subject_count = 0
        for file_idx, case_files in enumerate(files):
            try:
                with open(case_files[1], 'r') as f:  # seg_json_files
                    annotation = json.load(f)

                # 从文件名提取受试者ID
                subject = case_files[0].split(os.sep)[-1].split('_')[0]

                # 处理标签
                labels = np.array(annotation['labels'])
                if 'instances' in annotation:
                    _, instances, counts = np.unique(
                        annotation['instances'], return_inverse=True, return_counts=True,
                    )
                    labels[(counts < 30)[instances]] = 0
                labels = np.unique(labels)

                # 记录受试者信息
                subject_idx = subject_idxs.setdefault(subject, len(subject_idxs))
                subject_files[subject_idx].append(case_files)

                # 确保标签索引不会越界
                valid_labels = labels[labels < 86]
                subject_labels[subject_idx, valid_labels] = 1
                valid_subject_count = max(valid_subject_count, subject_idx + 1)

            except Exception as e:
                print(f"处理文件 {case_files[0]} 时出错: {e}")
                continue

        # 获取有效文件和类别
        subject_num = len(subject_idxs)
        if subject_num > 0:
            subject_files = subject_files[:subject_num]
            subject_labels = subject_labels[:subject_num]

            # 使用分层策略划分受试者
            if subject_num >= 5:  # 至少需要5个受试者才能进行5折交叉验证
                stratifier = IterativeStratification(n_splits=5, order=2, )
                splits = list(stratifier.split(
                    X=np.array(subject_files, dtype=object),
                    y=subject_labels[:subject_num],
                ))

                # write folds to storage for documentation
                for i, (_, fold_idxs) in enumerate(splits):
                    fold_file_path = os.path.join(self.fold_path, f'cityu_fold_{i}.txt')
                    with open(fold_file_path, 'w') as f:
                        for subject_idx in fold_idxs:
                            for fs in subject_files[subject_idx]:
                                f.write(Path(fs[0]).name + '\n')

                # 使用第一折作为默认划分
                train_idxs, val_idxs = splits[0]
                train_files = [f for i in train_idxs for f in subject_files[i]]
                val_files = [f for i in val_idxs for f in subject_files[i]]

                if self.include_val_as_train:
                    train_files += val_files

                return train_files, val_files
        return [], []

    def _split_randomly(self,
                        obj_files: List[str],
                        seg_json_files: List[str]) -> Tuple[List, List]:
        """
        随机划分文件列表为训练集和验证集
        """
        files = list(zip(obj_files, seg_json_files))

        total_count = len(files)
        train_count = int(0.8 * total_count)

        # 随机打乱后划分
        indices = list(range(total_count))
        random.shuffle(indices)

        train_indices = indices[:train_count]
        val_indices = indices[train_count:]

        train_files = [files[i] for i in train_indices]
        val_files = [files[i] for i in val_indices]

        return train_files, val_files

    def split(self,
              obj_files: List[str],
              seg_json_files: List[str]) -> Tuple[List, List]:
        """
        划分文件列表为训练集和验证集
        """
        files = list(zip(obj_files, seg_json_files))

        if len(files) <= 1:
            return files, []

        paras = (obj_files, seg_json_files)
        # 如果指定了fold文件，则按该文件划分
        if self.fold_path is not None and self.fold_filename is not None:
            return self._split_by_fold_file(*paras)

        # 如果没有指定fold文件，使用基于受试者的分层划分
        try:
            train_files, val_files = self._split_by_subject_stratification(*paras)
            if train_files and val_files:  # 如果分层划分成功
                return train_files, val_files
        except ImportError:
            print("skmultilearn未安装，使用简单随机划分")
        except Exception as e:
            print(f"分层划分出错: {e}，使用简单随机划分")

        # 如果上述方法失败，回退到简单的随机划分
        return self._split_randomly(*paras)


########################################################################
# functions
def collect_matched_mesh_files(source_obj_data_path,
                               source_seg_json_data_path,
                               exclude: List[str] = []):
    """
    从 3DSPlus 数据集目录中提取对应的 obj 和 json 文件名。
    这是该数据的第一步，用于提取所有匹配的 obj, seg.json, landmarks.json 文件。
    :param source_obj_data_path: obj 文件所在目录
    :param source_seg_json_data_path: seg.json 文件所在目录
    :param source_landmarks_json_data_path: landmarks.json 文件所在目录
    :param exclude: 需要排除的文件名列表
    :return: 包含 obj, seg.json, landmarks.json 文件名的元组
    """

    # 获取所有文件
    obj_root = Path(source_obj_data_path)
    seg_root = Path(source_seg_json_data_path)

    # 收集obj文件
    obj_files = sorted(obj_root.glob('**/*.obj'))
    # 收集seg json文件
    seg_files = sorted(seg_root.glob('**/*.json'))

    # 提取stem并确保三者一致
    obj_stems = set(f.stem for f in obj_files)
    seg_stems = set(f.stem for f in seg_files)

    # 只保留三个集合的交集并排除exclude的指定文件
    common_stems = obj_stems & seg_stems
    common_stems = common_stems - set(exclude)

    # 构建最终的文件列表
    result_obj_files = []
    result_seg_json_files = []  # 存储 seg json 文件
    result_landmarks_json_files = []  # 存储 landmarks json 文件

    for stem in sorted(common_stems):
        # 获取对应的obj文件
        obj_file = next((f for f in obj_files if f.stem == stem), None)
        # 获取对应的seg文件
        seg_file = next((f for f in seg_files if f.stem == stem), None)

        if obj_file and seg_file:
            result_obj_files.append(str(obj_file))
            # 分别存储 seg 文件
            result_seg_json_files.append(str(seg_file))

    # 返回三个独立的文件列表
    return (result_obj_files, result_seg_json_files)

# ... existing code ...
def split_files(obj_files, seg_json_files,
                fold_path=None, fold_filename=None, include_val_as_train=False):
    """
    将匹配的文件列表划分为训练集和验证集
    :param obj_files: obj文件列表
    :param seg_json_files: seg json文件列表
    :param fold_path: 验证集fold文件路径（如果指定，则按照该文件划分）
    :param include_val_as_train: 是否将验证集也包含在训练集中
    :return: (train_files, val_files) 其中每个元素都是 (obj_file, seg_file, landmark_file) 的元组列表
    """
    # 将三个文件列表合并成一个元组列表
    files = list(zip(obj_files, seg_json_files))

    if len(files) <= 1:
        return files, []

    # 如果指定了fold文件，则按该文件划分
    if fold_path is not None and fold_filename is not None:
        with open(os.path.join(fold_path, fold_filename), 'r') as f:
            val_mesh_files = [l.strip() for l in f.readlines() if l.strip()]
        # 提取stem用于匹配
        val_mesh_stems = [Path(f).stem for f in val_mesh_files]
        train_files = [fs for fs in files if Path(fs[0]).stem not in val_mesh_stems]
        val_files = [fs for fs in files if Path(fs[0]).stem in val_mesh_stems]

        if include_val_as_train:
            train_files += val_files

        return train_files, val_files

    # 如果没有指定fold文件，使用基于受试者的分层划分
    try:
        # 确定每个受试者的类别和文件
        subject_idxs = {}
        subject_files = [[] for _ in range(len(files) * 2)]  # 预分配足够空间
        subject_labels = np.zeros((len(files) * 2, 86))  # 预分配足够空间

        valid_subject_count = 0
        for file_idx, case_files in enumerate(files):
            try:
                with open(case_files[1], 'r') as f:  # seg_json_files
                    annotation = json.load(f)

                # 从文件名提取受试者ID
                subject = case_files[0].split(os.sep)[-1].split('_')[0]

                # 处理标签
                labels = np.array(annotation['labels'])
                if 'instances' in annotation:
                    _, instances, counts = np.unique(
                        annotation['instances'], return_inverse=True, return_counts=True,
                    )
                    labels[(counts < 30)[instances]] = 0
                labels = np.unique(labels)

                # 记录受试者信息
                subject_idx = subject_idxs.setdefault(subject, len(subject_idxs))
                subject_files[subject_idx].append(case_files)

                # 确保标签索引不会越界
                valid_labels = labels[labels < 86]
                subject_labels[subject_idx, valid_labels] = 1
                valid_subject_count = max(valid_subject_count, subject_idx + 1)

            except Exception as e:
                print(f"处理文件 {case_files[0]} 时出错: {e}")
                continue

        # 获取有效文件和类别
        subject_num = len(subject_idxs)
        if subject_num > 0:
            subject_files = subject_files[:subject_num]
            subject_labels = subject_labels[:subject_num]

            # 使用分层策略划分受试者
            if subject_num >= 5:  # 至少需要5个受试者才能进行5折交叉验证
                stratifier = IterativeStratification(n_splits=5, order=2, )
                splits = list(stratifier.split(
                    X=np.array(subject_files, dtype=object),
                    y=subject_labels[:subject_num],
                ))

                # write folds to storage for documentation
                for i, (_, fold_idxs) in enumerate(splits):
                    fold_file_path = os.path.join(fold_path, f'cityu_fold_{i}.txt')
                    with open(fold_file_path, 'w') as f:
                        for subject_idx in fold_idxs:
                            for fs in subject_files[subject_idx]:
                                f.write(Path(fs[0]).name + '\n')

                # 使用第一折作为默认划分
                train_idxs, val_idxs = splits[0]
                train_files = [f for i in train_idxs for f in subject_files[i]]
                val_files = [f for i in val_idxs for f in subject_files[i]]

                if include_val_as_train:
                    train_files += val_files

                return train_files, val_files
    except ImportError:
        print("skmultilearn未安装，使用简单随机划分")
    except Exception as e:
        print(f"分层划分出错: {e}，使用简单随机划分")

    # 如果上述方法失败，回退到简单的随机划分
    total_count = len(files)
    train_count = int(0.8 * total_count)

    # 随机打乱后划分
    indices = list(range(total_count))
    random.shuffle(indices)

    train_indices = indices[:train_count]
    val_indices = indices[train_count:]

    train_files = [files[i] for i in train_indices]
    val_files = [files[i] for i in val_indices]

    return train_files, val_files


def load_datasets(sample_ratio=0.1,
                  fold_path=r"folds",
                  fold_filename=r"cityu_fold_0.txt",
                  source_obj_data_path=None,
                  source_seg_json_data_path=None,
                  exclude: List[str] = [],
                  pretrans_name='comb_default',
                  rseed=42,
                  args=None):
    obj_files, seg_json_files, landmarks_json_files = collect_matched_mesh_files(
        source_obj_data_path=source_obj_data_path,
        source_seg_json_data_path=source_seg_json_data_path,
        exclude=exclude
    )

    # 方法2: 使用指定的fold文件进行划分
    # 假设我们有一个fold文件，其中指定了哪些文件应该属于验证集
    fold_path = fold_path
    fold_filename = fold_filename

    train_files_fold, val_files_fold = split_files(
        obj_files, seg_json_files, landmarks_json_files,
        fold_path=fold_path,
        fold_filename=fold_filename
    )

    # 根据sample_ratio对训练集和验证集进行采样
    if sample_ratio < 1.0:
        # 设置随机种子以确保可重复性
        random.seed(rseed)
        np.random.seed(rseed)

        # 对训练集进行采样
        if len(train_files_fold) > 0:
            train_sample_size = max(1, int(len(train_files_fold) * sample_ratio))
            train_indices = random.sample(range(len(train_files_fold)), train_sample_size)
            train_files_fold = [train_files_fold[i] for i in train_indices]
        else:
            print("警告: 训练集为空，无法进行采样")

        # 对验证集进行采样
        if len(val_files_fold) > 0:
            val_sample_size = max(1, int(len(val_files_fold) * sample_ratio))
            val_indices = random.sample(range(len(val_files_fold)), val_sample_size)
            val_files_fold = [val_files_fold[i] for i in val_indices]
        else:
            print("警告: 验证集为空，无法进行采样")

    pre_transform = prep_combs[pretrans_name]['pre_transform']
    pre_filter = prep_combs[pretrans_name]['pre_filter']
    transform = None
    train_ds = T3DSPlusSegDataset(args.save_data_path, transform, pre_transform, pre_filter, pretrans_name,
                                  train_files_fold)

    val_ds = T3DSPlusSegDataset(args.save_data_path, transform, pre_transform, pre_filter, pretrans_name,
                                val_files_fold)

    # test_ds = T3DSPlusDataset(args.save_data_path, pre_transform, pre_filter, pretrans_name,
    #                          )
    test_ds = []

    return train_ds, val_ds, test_ds


def load_datasets_from_cfg_obj(cfg: DictConfig):

    file_collecter = T3DSPlusSegFileCollector(
        obj_data_path=cfg.paths.source_obj_data_path,
        seg_json_data_path=cfg.paths.source_seg_json_data_path,
        exclude=cfg.dataset.exclude_files
    )
    file_splitter = T3DSPlusSegFileSplitter(
        fold_path=cfg.dataset.folds.fold_path,
        fold_filename=cfg.dataset.folds.fold_filename,
        include_val_as_train=False
    )

    # pre_transform = instantiate_from_config(cfg.dataset.pre_transform_obj)
    # transform = instantiate_from_config(cfg.dataset.transform_obj)
    # pre_filter = instantiate_from_config(cfg.dataset.pre_filter_obj)

    pre_transform = cfg.dataset.pre_transform_obj
    transform = cfg.dataset.transform_obj
    pre_filter = cfg.dataset.pre_filter_obj

    dm = T3DSPlusSegDataModule(
        save_data_root=cfg.paths.save_data_path,
        pretrans_name=cfg.dataset.pretrans_name,
        pre_transform=pre_transform,
        transform=transform,
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

def load_datasets_from_cfg(config_path = "../configs",config_name = "test_T3DSPlusDataset"):
    cfg = get_yaml_config(config_path=config_path, config_name=config_name)
    print_yaml_config(config_path=config_path, config_name=config_name)

    return load_datasets_from_cfg_obj(cfg)
    # file_collecter = T3DSPlusSegFileCollector(
    #     obj_data_path=cfg.paths.source_obj_data_path,
    #     seg_json_data_path=cfg.paths.source_seg_json_data_path,
    #     exclude=cfg.dataset.exclude_files
    # )
    # file_splitter = T3DSPlusSegFileSplitter(
    #     fold_path=cfg.dataset.folds.fold_path,
    #     fold_filename=cfg.dataset.folds.fold_filename,
    #     include_val_as_train=False
    # )
    #
    # pre_transform = instantiate_from_config(cfg.dataset.pre_transform_obj)
    # transform = instantiate_from_config(cfg.dataset.transform_obj)
    # pre_filter = instantiate_from_config(cfg.dataset.pre_filter_obj)
    #
    # dm = T3DSPlusSegDataModule(
    #     save_data_root=cfg.paths.save_data_path,
    #     pretrans_name=cfg.dataset.pretrans_name,
    #     pre_transform=pre_transform,
    #     transform=transform,
    #     pre_filter=pre_filter,
    #     sample_ratio=cfg.dataset.sample_ratio,
    #     batch_size=cfg.dataset.batch_size,
    #     rseed=cfg.dataset.rseed,
    #     num_workers=cfg.dataset.num_workers,
    #     pin_memory=cfg.dataset.pin_memory,
    #     persistent_workers=cfg.dataset.persistent_workers,
    #     recompute=cfg.dataset.recompute,
    #     is_filter_small_instance=cfg.dataset.is_filter_small_instance,
    #     filter_small_instance_size=cfg.dataset.filter_small_instance_size,
    #     file_collector=file_collecter,
    #     file_splitter=file_splitter)
    #
    # dm.setup(stage='fit')
    #
    # return dm, dm.train_dataloader(), dm.val_dataloader()

def get_dataloaders_from_cfg_obj(cfg: DictConfig, only_val=False):

    file_collecter = T3DSPlusSegFileCollector(
        obj_data_path=cfg.paths.source_obj_data_path,
        seg_json_data_path=cfg.paths.source_seg_json_data_path,
        exclude=cfg.dataset.exclude_files
    )
    file_splitter =  T3DSPlusSegFileSplitter(
        fold_path=cfg.dataset.folds.fold_path,
        fold_filename=cfg.dataset.folds.fold_filename,
        include_val_as_train=False
    )

    pre_transform = cfg.dataset.pre_transform_obj
    transform = cfg.dataset.transform_obj
    val_transform = cfg.dataset.val_transform_obj
    pre_filter = cfg.dataset.pre_filter_obj

    dm = T3DSPlusSegDataModule(
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


class DataPrinter:
    """数据打印类，用于打印data对象的属性信息"""

    def __init__(self, data):
        self.data = data

    def print(self):
        """打印data对象的所有属性信息

        Args:
            data: 数据对象，通常包含pos等属性

        Returns:
            None
        """
        print("=" * 50)
        print("Data Object Attributes:")
        print("=" * 50)

        # 遍历data对象的所有属性
        for attr_name in dir(self.data):
            # 跳过私有属性和方法
            if not attr_name.startswith('_'):
                try:
                    attr_value = getattr(self.data, attr_name)
                    # 只处理非方法的属性
                    if not callable(attr_value):
                        # 根据属性类型输出不同信息
                        if isinstance(attr_value, (int, float, bool)):  # scalar类型
                            print(f"{attr_name}: {attr_value}")
                        elif isinstance(attr_value, torch.Tensor):  # tensor类型
                            print(f"{attr_name}: Tensor with shape {attr_value.shape}")
                        elif isinstance(attr_value, str):  # 字符串类型
                            print(f"{attr_name}: {attr_value}")
                        elif isinstance(attr_value, (list, tuple)):  # 列表或元组
                            print(f"{attr_name}: {type(attr_value).__name__} with length {len(attr_value)}")
                        elif isinstance(attr_value, np.ndarray):  # numpy数组
                            print(f"{attr_name}: numpy.ndarray with shape {attr_value.shape}")
                        else:  # 其他类型
                            print(f"{attr_name}: {type(attr_value).__name__}")
                except:
                    # 某些属性可能无法访问，跳过
                    continue

        print("\n" + "=" * 50)
        print("Position Information (data.pos):")
        print("=" * 50)

        # 输出data.pos的详细信息
        if hasattr(self.data, 'pos') and self.data.pos is not None:
            pos = self.data.pos

            # 确保pos是numpy数组或tensor
            if isinstance(pos, torch.Tensor):
                pos_array = pos.detach().cpu().numpy()
            else:
                pos_array = np.array(pos)

            # 计算边界框信息
            xmin = np.min(pos_array[:, 0])
            xmax = np.max(pos_array[:, 0])
            ymin = np.min(pos_array[:, 1])
            ymax = np.max(pos_array[:, 1])
            zmin = np.min(pos_array[:, 2])
            zmax = np.max(pos_array[:, 2])

            # 计算中心点坐标
            center_x = (xmin + xmax) / 2
            center_y = (ymin + ymax) / 2
            center_z = (zmin + zmax) / 2

            print(f"Shape: {pos_array.shape}")
            print(f"X range: [{xmin:.6f}, {xmax:.6f}]")
            print(f"Y range: [{ymin:.6f}, {ymax:.6f}]")
            print(f"Z range: [{zmin:.6f}, {zmax:.6f}]")
            print(f"Center point: ({center_x:.6f}, {center_y:.6f}, {center_z:.6f})")
            print(f"Bounding box size: ({xmax - xmin:.6f}, {ymax - ymin:.6f}, {zmax - zmin:.6f})")
        else:
            print("data.pos attribute not found or is None")

        print("=" * 50)


#########################################################################
# region tests
def test_load_datasets():
    train_ds, val_ds, test_ds = load_datasets(fold_path=r"folds",
                                              fold_filename=r"cityu_fold_0.txt",
                                              pretrans_name="comb_fps_24000_qslim_10000",
                                              sample_ratio=0.02,
                                              source_obj_data_path=None,
                                              source_seg_json_data_path=None,
                                              rseed=42)
    print(len(train_ds), len(val_ds), len(test_ds))
    pdata = train_ds[1]
    # gu.print_3d(gu.np_to_pcd_with_label(pdata.pos.numpy(), pdata.y.numpy()))
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(pdata.data_qslim.pos)
    mesh.triangles = o3d.utility.Vector3iVector(pdata.data_qslim.faces)
    # gu.print_3d(mesh)

    # landmark可视化 - 使用更大的球体和文本标签
    if hasattr(pdata, 'landmark_coords') and hasattr(pdata, 'landmark_classes'):
        # 创建增强版可视化几何体列表
        geometries = []

        # 关键点类别名称
        landmark_class_names = ['Mesial', 'Distal', 'FacialPoint', 'OuterPoint', 'InnerPoint', 'Cusp']
        landmark_colors = {
            0: [1, 0, 0],  # Mesial - 红色
            1: [0, 1, 0],  # Distal - 绿色
            2: [0, 0, 1],  # FacialPoint - 蓝色
            3: [1, 1, 0],  # OuterPoint - 黄色
            4: [1, 0, 1],  # InnerPoint - 品红
            5: [0, 1, 1]  # Cusp - 青色
        }

        # 为每个关键点创建更大的球体表示
        for i, (coord, class_id) in enumerate(zip(pdata.landmark_coords, pdata.landmark_classes)):
            # 创建更大的球体（增大显示效果）
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)  # 增大半径到1.0
            sphere.translate(coord.numpy())
            sphere.paint_uniform_color(landmark_colors.get(class_id.item(), [0.5, 0.5, 0.5]))
            sphere.compute_vertex_normals()
            geometries.append(sphere)

        # 如果有原始网格数据，也一并显示
        if hasattr(pdata, 'pos'):
            original_pcd = o3d.geometry.PointCloud()
            original_pcd.points = o3d.utility.Vector3dVector(pdata.pos.numpy())
            colored_pcd = gu.np_to_pcd_with_label(pdata.pos.numpy(), pdata.y.numpy())
            original_pcd.colors = colored_pcd.colors
            # original_pcd.paint_uniform_color([0.8, 0.8, 0.8])  # 浅灰色
            geometries.append(original_pcd)

        print("显示增强版landmarks（更大球体）...")
        o3d.visualization.draw_geometries(geometries)


def visualize_mesh(points, triangles, colors=None, instances=None, landmark_coords=None, landmark_classes=None):
    """使用 Open3D 可视化 mesh，包括语义信息和关键点"""
    # 创建几何体列表
    geometries = []

    # 创建主网格
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    if colors is not None:
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    else:
        mesh.paint_uniform_color([0.7, 0.7, 0.7])  # 默认灰色

    mesh.compute_vertex_normals()
    geometries.append(mesh)

    # 如果提供了实例信息，则为每个实例创建不同的颜色
    if instances is not None:
        # 为每个实例生成不同颜色
        unique_instances = np.unique(instances)
        instance_colors = {}
        for i, inst_id in enumerate(unique_instances):
            if inst_id == 0:  # 背景或未标记区域
                instance_colors[inst_id] = [0.7, 0.7, 0.7]  # 灰色
            else:
                # 使用不同的颜色表示不同实例
                hue = (i * 0.618034) % 1  # 黄金比例近似值，使颜色尽可能分散
                instance_colors[inst_id] = plt.cm.hsv(hue)[:3]

        # 创建实例着色的点云
        instance_colored_points = np.zeros_like(points)
        for i, inst_id in enumerate(instances):
            instance_colored_points[i] = instance_colors[inst_id]

        instance_point_cloud = o3d.geometry.PointCloud()
        instance_point_cloud.points = o3d.utility.Vector3dVector(points)
        instance_point_cloud.colors = o3d.utility.Vector3dVector(instance_colored_points)
        geometries.append(instance_point_cloud)

    # 如果提供了关键点信息，则绘制关键点
    if landmark_coords is not None and landmark_classes is not None:
        # 关键点类别对应的颜色
        landmark_class_names = ['Mesial', 'Distal', 'FacialPoint', 'OuterPoint', 'InnerPoint', 'Cusp']
        landmark_colors = {
            0: [1, 0, 0],  # Mesial - 红色
            1: [0, 1, 0],  # Distal - 绿色
            2: [0, 0, 1],  # FacialPoint - 蓝色
            3: [1, 1, 0],  # OuterPoint - 黄色
            4: [1, 0, 1],  # InnerPoint - 品红
            5: [0, 1, 1],  # Cusp - 青色
        }

        # 绘制关键点球体
        for coord, class_id in zip(landmark_coords, landmark_classes):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.6)
            sphere.translate(coord)
            sphere.paint_uniform_color(landmark_colors.get(class_id, [0.5, 0.5, 0.5]))  # 默认灰色
            geometries.append(sphere)

        # 绘制关键点标签
        for i, (coord, class_id) in enumerate(zip(landmark_coords, landmark_classes)):
            label = f"{landmark_class_names[class_id]}({i})"
            # 注意：Open3D当前版本不直接支持文本标签，此处仅为示意
            # 在实际应用中可以考虑使用其他库如matplotlib进行标注

    # 显示所有几何体
    o3d.visualization.draw_geometries(geometries)


def test_collect_matched_mesh_files(args):
    """
    collect_matched_mesh_files函数的调用示例
    """
    # 定义数据路径
    source_obj_data_path = args.source_obj_data_path
    source_seg_json_data_path = args.source_seg_json_data_path
    exclude = [
        # # braces
        # '017U3R3T_upper',
        # '01A91JH6_lower',
        # '01A91JH6_upper',
        # '01HMF5HV_lower',
        # '01HMF5HV_upper',
        # 'E23G704K_lower',
        # 'E23G704K_upper',
        # # chaos
        # '016FSM14_lower',
        # '01FPTYH2_lower',
        # missing tooth segmentation
        'K4BAII5F_upper',
        '87N5YSES_upper',
        '67PV9M7X_lower',
        # missing more than one landmark
        'S0AON6PZ_lower',
    ]

    # 调用函数收集匹配的文件
    obj_files, seg_json_files, landmarks_json_files = collect_matched_mesh_files(
        source_obj_data_path=source_obj_data_path,
        source_seg_json_data_path=source_seg_json_data_path,
        exclude=exclude  # 可以指定要排除的文件名列表
    )

    # 输出结果信息
    print(f"找到 {len(obj_files)} 个匹配的文件组")
    print("前5个匹配的文件:")
    for i in range(min(5, len(obj_files))):
        print(f"  Obj: {os.path.abspath(obj_files[i])}")
        print(f"  Seg: {os.path.abspath(seg_json_files[i])}")
        print(f"  Landmark: {os.path.abspath(landmarks_json_files[i])}")
        print()

    return obj_files, seg_json_files, landmarks_json_files


# ... existing code ...

def test_split_files(args):
    """
    split_files函数的调用示例
    """
    # 首先需要获取匹配的文件列表
    obj_files, seg_json_files, landmarks_json_files = collect_matched_mesh_files(
        source_obj_data_path=args.source_obj_data_path,
        source_seg_json_data_path=args.source_seg_json_data_path,
        exclude=args.exclude_files
    )

    # 方法1: 使用默认的随机划分（80%训练，20%验证）
    train_files, val_files = split_files(obj_files, seg_json_files, landmarks_json_files)

    print(f"随机划分结果:")
    print(f"训练集文件数: {len(train_files)}")
    print(f"验证集文件数: {len(val_files)}")
    print()

    # 方法2: 使用指定的fold文件进行划分
    # 假设我们有一个fold文件，其中指定了哪些文件应该属于验证集
    fold_path = r"folds"
    fold_filename = None

    train_files_fold, val_files_fold = split_files(
        obj_files, seg_json_files, landmarks_json_files,
        fold_path=fold_path,
        fold_filename=fold_filename
    )

    print(f"基于fold文件的划分结果:")
    print(f"训练集文件数: {len(train_files_fold)}")
    print(f"验证集文件数: {len(val_files_fold)}")
    print()

    # 方法3: 使用fold文件划分，并将验证集也包含在训练集中
    train_files_include_val, val_files_include_val = split_files(
        obj_files, seg_json_files, landmarks_json_files,
        fold_path=fold_path,
        fold_filename=fold_filename,
        include_val_as_train=True  # 将验证集也包含在训练集中
    )

    print(f"包含验证集的训练集划分结果:")
    print(f"训练集文件数: {len(train_files_include_val)}")
    print(f"验证集文件数: {len(val_files_include_val)}")
    print()

    # 打印一些示例文件信息
    if train_files:
        print("训练集示例文件:")
        for i in range(min(3, len(train_files))):
            obj_file, seg_file, landmark_file = train_files[i]
            print(f"  Obj: {os.path.abspath(obj_file)}")
            print(f"  Seg: {os.path.abspath(seg_file)}")
            print(f"  Landmark: {os.path.abspath(landmark_file)}")
            print()

    return train_files, val_files


def test_T3DSPlusSegFileCollector():
    config_path = "./configs"
    config_name = "test_T3DSPlusDataset"
    cfg = get_yaml_config(config_path=config_path, config_name=config_name)
    print_yaml_config(config_path=config_path, config_name=config_name)

    # pretrans_name = "comb_default"
    file_collecter = T3DSPlusSegFileCollector(
        obj_data_path=cfg.paths.source_obj_data_path,
        seg_json_data_path=cfg.paths.source_seg_json_data_path,
        exclude=cfg.dataset.exclude_files
    )
    file_splitter = T3DSPlusSegFileSplitter(
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


def test_T3DSPlusSegReader():
    print("Current working directory:", os.getcwd())

    config_path = "../configs"
    config_name = "test_T3DSPlusDataset"
    cfg = get_yaml_config(config_path=config_path, config_name=config_name)
    # cfg = instantiate_from_yaml_config(config_path=config_path, config_name=config_name)

    args = cfg.paths
    ds_args = cfg.dataset
    print_yaml_cfg_dict(cfg)

    cfg = instantiate_from_yaml_config(config_path=config_path, config_name=config_name)
    # print_yaml_config(config_path=config_path, config_name=config_name)

    reader = T3DSPlusSegReader(is_filter_small_instance=ds_args.is_filter_small_instance,
                               filter_small_instance_size=ds_args.filter_small_instance_size,
                               verbose=True)
    # JJEIMAPR_lower
    obj_filename = os.path.join(args.source_obj_data_path, "lower/JJEIMAPR/JJEIMAPR_lower.obj")
    seg_json_filename = os.path.join(args.source_seg_json_data_path, "lower/JJEIMAPR/JJEIMAPR_lower.json")
    # lm_json_filename = os.path.join(args.source_landmarks_json_data_path, "lower/01A6HAN6/01A6HAN6_lower__kpt.json")

    data = reader(obj_filename, seg_json_filename)
    print(data.keys())

    # print(cfg.pre_transform_obj)


    pre_transform = cfg.dataset.pre_transform_obj
    transform = cfg.dataset.transform_obj
    pre_filter = cfg.dataset.pre_filter_obj

    print(pre_transform, transform, pre_filter)

    data = pre_transform(data)
    print(data.keys())

    # vis = instantiate_from_config(cfg.vis.qslim_10000_mesh_label_lighton_saver)
    # vis.window_visible = True
    # vis.save_file = False
    # vis(data)

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(data.pos.numpy())
    # if hasattr(data, 'normals'):
    #     pcd.normals = o3d.utility.Vector3dVector(data.normals.numpy())
    #     pcd.normalize_normals()
    # o3d.visualization.draw_geometries([pcd], point_show_normal=True)

def test_T3DSPlusSegDataModule():
    config_path = "../configs"
    config_name = "test_T3DSPlusDataset"
    cfg = get_yaml_config(config_path=config_path, config_name=config_name)
    print_yaml_config(config_path=config_path, config_name=config_name)

    # pretrans_name = "comb_default"
    file_collecter = T3DSPlusSegFileCollector(
        obj_data_path=cfg.paths.source_obj_data_path,
        seg_json_data_path=cfg.paths.source_seg_json_data_path,
        exclude=cfg.dataset.exclude_files
    )
    file_splitter = T3DSPlusSegFileSplitter(
        fold_path=cfg.dataset.folds.fold_path,
        fold_filename=cfg.dataset.folds.fold_filename,
        include_val_as_train=False
    )

    # pre_transform = cfg.pretrans_combs[cfg.pretrans_name]
    # transform = cfg.trans_combs[cfg.transforms_name]
    # pre_filter = cfg.pre_filters[cfg.prefilter_name]

    pre_transform = instantiate_from_config(cfg.dataset.pre_transform_obj)
    transform = instantiate_from_config(cfg.dataset.transform_obj)
    pre_filter = instantiate_from_config(cfg.dataset.pre_filter_obj)

    dm = T3DSPlusSegDataModule(
        save_data_root=cfg.paths.save_data_path,
        pretrans_name=cfg.dataset.pretrans_name,
        pre_transform=pre_transform,
        transform=transform,
        pre_filter=pre_filter,
        sample_ratio=0.02,
        batch_size=1,
        rseed=42,
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
    # pdata.face = pdata.face.T
    # print(pdata)  # BNC格式
    # data_dict = pdata.to_dict()
    # for key in data_dict.keys():
    #     print(key)

    pdata2 = train_ds[1]
    # pdata2.face = pdata2.face.T
    # print(pdata2)  # BNC格式
    # data_dict = pdata2.to_dict()
    # for key in data_dict.keys():
    #     print(key)
    DataPrinter(pdata).print()
    DataPrinter(pdata2).print()

    # 可视化
    gu.print_3d(gu.np_to_pcd(pdata.pos.numpy()))

    # 示例用法
    writer = DataJsonWriter()
    json_dir = os.path.join(cfg.paths.save_jsondata_path, T3DSPlusSegDataset.__name__, 'processed',
                            cfg.dataset.pretrans_name)
    writer(pdata, root=json_dir)

def test_load_datasets_from_cfg():
    config_path = "../configs"
    config_name = "test_T3DSPlusDataset"
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
    json_dir = os.path.join(cfg.paths.save_jsondata_path, T3DSPlusSegDataset.__name__, 'processed',
                            cfg.dataset.pretrans_name)
    writer(pdata, root=json_dir)

#----------------------------测试transforms-------------------------------------#
def test_VisRenderSaver():
    # print("Current working directory:", os.getcwd())
    config_path = "../configs"
    config_name = "test_T3DSPlusDataset"
    cfg = get_yaml_config(config_path=config_path, config_name=config_name)
    args = cfg.paths
    ds_args = cfg.dataset
    print_yaml_cfg_dict(cfg)

    # print_yaml_config(config_path=config_path, config_name=config_name)

    reader = T3DSPlusSegReader(is_filter_small_instance=ds_args.is_filter_small_instance,
                               filter_small_instance_size=ds_args.filter_small_instance_size,
                               verbose=True)

    obj_filename = os.path.join(args.source_obj_data_path, "lower/01A6HAN6/01A6HAN6_lower.obj")
    seg_json_filename = os.path.join(args.source_seg_json_data_path, "lower/01A6HAN6/01A6HAN6_lower.json")
    # lm_json_filename = os.path.join(args.source_landmarks_json_data_path, "lower/01A6HAN6/01A6HAN6_lower__kpt.json")

    data = reader(obj_filename, seg_json_filename)
    print(data.keys())

    # 测试 transforms
    pre_transform = instantiate_from_config(cfg.dataset.pre_transform_obj)
    transform = instantiate_from_config(cfg.dataset.transform_obj)
    pre_filter = instantiate_from_config(cfg.dataset.pre_filter_obj)

    print(pre_transform, transform, pre_filter)

    data = pre_transform(data)
    print(data.keys())

    # 先测试reader
    # renderer = VisRenderSaver(
    #     save_file=True,  # 保存渲染结果
    #     window_visible=True,  # 后台渲染（不弹窗口）
    #     window_size=(800, 600),
    #     root=cfg.paths.save_imgs_path,
    #     classname="T3DSPlusSegDataset",
    #     data_mode="point",  # 网格模式
    #     color_mode="uniform_color",  # 标签颜色模式
    #     uniform_color=(0.5, 0.5, 0.5),  # 默认纯色
    #     # 视角配置（Open3D原生）
    #     view_mode="top",
    #     # Open3D原生RenderOption参数
    #     point_size=2.0,
    #     mesh_show_wireframe=False,  # Open3D原生布尔值
    #     mesh_color_option=1,  # Open3D原生值：3-Vertex（纯顶点色）
    #     mesh_shade_option=0,  # Open3D原生值：1-Smooth（高斯平滑）
    #     light_on=True,
    # )
    # 测试各种vis
    # renderer = instantiate_from_config(cfg.vis.mesh_uniform_saver)
    # data = renderer(data)
    # renderer = instantiate_from_config(cfg.vis.mesh_label_lightoff_saver)
    # data = renderer(data)
    # renderer = instantiate_from_config(cfg.vis.mesh_label_lighton_saver)
    # data = renderer(data)

    # renderer = instantiate_from_config(cfg.vis.point_edge_saver)
    # data = renderer(data)
    #
    # renderer = instantiate_from_config(cfg.vis.vis_render_saver)
    # data = renderer(data)

    # renderer = instantiate_from_config(cfg.vis.instcent_render_saver)
    # data = renderer(data)

    renderer = instantiate_from_config(cfg.vis.ray_blend_render_saver)
    data = renderer(data)

    print(data.keys())


def test_pv_gu():
    print("Current working directory:", os.getcwd())

    config_path = "../configs"
    config_name = "test_T3DSPlusDataset"
    cfg = get_yaml_config(config_path=config_path, config_name=config_name)
    # cfg = instantiate_from_yaml_config(config_path=config_path, config_name=config_name)

    args = cfg.paths
    ds_args = cfg.dataset
    print_yaml_cfg_dict(cfg)

    cfg = instantiate_from_yaml_config(config_path=config_path, config_name=config_name)
    # print_yaml_config(config_path=config_path, config_name=config_name)

    reader = T3DSPlusSegReader(is_filter_small_instance=ds_args.is_filter_small_instance,
                               filter_small_instance_size=ds_args.filter_small_instance_size,
                               verbose=True)
    # JJEIMAPR_lower
    obj_filename = os.path.join(args.source_obj_data_path, "lower/JJEIMAPR/JJEIMAPR_lower.obj")
    seg_json_filename = os.path.join(args.source_seg_json_data_path, "lower/JJEIMAPR/JJEIMAPR_lower.json")
    # lm_json_filename = os.path.join(args.source_landmarks_json_data_path, "lower/01A6HAN6/01A6HAN6_lower__kpt.json")

    data = reader(obj_filename, seg_json_filename)
    # print(data.keys())
    # print(cfg.pre_transform_obj)

    pre_transform = cfg.dataset.pre_transform_obj
    transform = cfg.dataset.transform_obj
    pre_filter = cfg.dataset.pre_filter_obj
    # print(pre_transform, transform, pre_filter)

    data = pre_transform(data)
    print(data.keys())

    # ===================== 使用函数进行可视化 =====================
    face_np = data.face.numpy().T if data.face.ndim == 2 and data.face.shape[0] == 3 else data.face.numpy()
    pvgu.print_3d_args(
        pvgu.np_to_pcd(data.pos.numpy(), offset=[1.0,0,0]),
        pvgu.np_to_mesh(data.pos.numpy(), face_np),
        off_screen=False
    )

    # ==================== 使用DataVisualizer进行可视化 ====================
    visualizer = pvgu.DataVisualizer(data)

    # # 1. 快速可视化（自动检测）
    # print("\n=== 快速可视化点云 ===")
    # visualizer.quick_vis(mode='point', anti_aliasing=False, background='white', window_size=[1200, 900])
    #
    # # 2. 可视化带标签的网格
    # print("\n=== 可视化带标签的网格 ===")
    # visualizer.visualize_mesh(show_edges=False, background='white',
    #                             anti_aliasing=True,
    #                             multi_samples=16,  # 最高质量
    #                             line_smoothing=True,
    #                             point_smoothing=True,
    #                             polygon_smoothing=True,
    #                             azimuth= 45,
    #                             elevation= 45,
    #                             window_size=[1200, 900])
    #
    # # 3. 可视化实例
    # print("\n=== 可视化实例分割 ===")
    # visualizer.visualize_mesh_with_instances(show_edges=False, background='white')
    #
    # # 4. 保存多视角图片
    # print("\n=== 保存多视角渲染 ===")
    # visualizer.visualize_multi_view(
    #     views=['xy', 'iso'],
    #     background='white',
    #     anti_aliasing=True,
    #     multi_samples=8,
    #     window_size=[1920, 1080],
    #     save_dir='./vis_output',
    #     filename='JJEIMAPR_lower'
    # )

    # 5. 保存单张高质量渲染图
    print("\n=== 保存高质量渲染图 ===")
    visualizer.visualize_mesh(
        show_edges=False,
        background='white',
        anti_aliasing=True,
        multi_samples=16,  # 最高质量
        line_smoothing=True,
        point_smoothing=True,
        polygon_smoothing=True,
        # view_vector=[1,-1,1],
        # azimuth=90,
        # elevation=0,
        view = 'xy',
        dpi=300,
        window_size=[int(3840*2), int(2160*2)],
        save_path='./vis_output/high_quality.png'
    )


def test_arap():
    config_path = "./configs"
    config_name = "test_T3DSPlusDataset"
    # cfg = get_yaml_config(config_path=config_path, config_name=config_name)
    cfg = instantiate_from_yaml_config(config_path=config_path, config_name=config_name)
    args = cfg.paths
    ds_args = cfg.dataset
    print_yaml_cfg_dict(cfg)

    # print_yaml_config(config_path=config_path, config_name=config_name)

    reader = T3DSPlusSegReader(is_filter_small_instance=ds_args.is_filter_small_instance,
                               filter_small_instance_size=ds_args.filter_small_instance_size,
                               verbose=True)

    obj_filename = os.path.join(args.source_obj_data_path, "lower/01A6HAN6/01A6HAN6_lower.obj")
    seg_json_filename = os.path.join(args.source_seg_json_data_path, "lower/01A6HAN6/01A6HAN6_lower.json")
    # lm_json_filename = os.path.join(args.source_landmarks_json_data_path, "lower/01A6HAN6/01A6HAN6_lower__kpt.json")

    data = reader(obj_filename, seg_json_filename)
    print(data.keys())

    # 测试 transforms
    pre_transform = instantiate_from_config(cfg.dataset.pre_transform_obj)
    transform = instantiate_from_config(cfg.dataset.transform_obj)
    pre_filter = instantiate_from_config(cfg.dataset.pre_filter_obj)

    print(pre_transform, transform, pre_filter)

    data = pre_transform(data)
    print(data.keys())

    # renderer = instantiate_from_config(cfg.vis.qslim_10000_mesh_label_lighton_saver)
    # data = renderer(data)
    #
    # renderer = instantiate_from_config(cfg.vis.qslim_10000_mesh_edge_lighton_saver)
    # data = renderer(data)
    #
    # renderer = instantiate_from_config(cfg.vis.qslim_10000_mesh_instcent_render_saver)
    # data = renderer(data)

    renderer = instantiate_from_config(cfg.vis.spee_curv_render_saver)
    data = renderer(data)

# endregion tests


def main():
    # config_path = "./configs"
    # config_name = "test_T3DSPlusDataset"
    # args = get_args_from_config(config_path=config_path, config_name=config_name)
    # print_yaml_config(config_path=config_path, config_name=config_name)
    # cfg = get_yaml_config(config_path=config_path, config_name=config_name)

    # test_load_basename()
    # test_sample_generator()
    # test_extract_filenames_from_basenames()
    # test_sample_generator_case2()
    # test_label48_to_label15()
    # combs
    # test_comb_origin()
    # test_comb_fps_24000()
    # test_comb_qslim_10000()
    # test_comb_qslim_5000()
    # test_comb_fps_24000_qslim_10000()
    # test_load_datasets(
    #     fold_path=r"folds",
    #     fold_filename=r"cityu_fold_0.txt",
    #     pretrans_name="comb_fps_24000_qslim_10000",
    #     sample_ratio=0.02,
    #     source_obj_data_path=args.source_obj_data_path,
    #     source_seg_json_data_path=args.source_seg_json_data_path,
    #     source_landmarks_json_data_path=args.source_landmarks_json_data_path,
    #     rseed=42,
    #     args = args
    # )

    # test_collect_matched_mesh_files(args)
    # test_split_files(args)
    # test_T3DSPlusSegFileCollector()
    # test_T3DSPlusSegReader()
    # test_T3DSPlusSegDataModule()
    # test_load_datasets_from_cfg()
    # test_VisRenderSaver()
    test_pv_gu()
    # test_arap()

if __name__ == '__main__':
    main()
