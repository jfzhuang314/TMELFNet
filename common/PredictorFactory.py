import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import torch
from omegaconf import OmegaConf, DictConfig
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader


# ========== 【抽象基类】BasePredictor ==========

class BasePredictor(ABC):
    """
    预测器抽象基类
    所有模型预测器都必须继承此类
    """

    def __init__(self, module, device: torch.device):
        """
        初始化预测器

        Args:
            module: PyTorch Lightning Module
            device: 设备
        """
        self.module = module
        self.device = device
        self.module.eval()  # 设置为评估模式

    @abstractmethod
    def predict_step_forward(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """
        执行单步前向预测（必须由子类实现）

        Args:
            batch: 输入批次数据

        Returns:
            outputs: 模型输出字典
        """
        pass

    def predict_batch(self, data_list: List[Data], batch_size: int = 1) -> List[Dict[str, torch.Tensor]]:
        """
        批量预测（默认实现，可被子类重写）

        Args:
            data_list: Data 对象列表
            batch_size: 批次大小

        Returns:
            all_outputs: 所有输出列表
        """
        loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)
        all_outputs = []

        for batch in loader:
            outputs = self.predict_step_forward(batch.to(self.device))
            all_outputs.append(outputs)

        return all_outputs

    def to(self, device: torch.device) -> 'BasePredictor':
        """
        将模型移动到指定设备

        Args:
            device: 目标设备

        Returns:
            self
        """
        self.module = self.module.to(device)
        self.device = device
        return self

    def eval(self) -> 'BasePredictor':
        """设置为评估模式"""
        self.module.eval()
        return self

    def train(self) -> 'BasePredictor':
        """设置为训练模式"""
        self.module.train()
        return self

    @property
    def is_training(self) -> bool:
        """是否在训练模式"""
        return self.module.training

    def __repr__(self):
        return f"{self.__class__.__name__}(device={self.device})"


# ========== 【具体实现】PTSegNetPredictor ==========

class PTSegNetPredictor(BasePredictor):
    """
    PTSegNetModule 的预测器
    """

    def __init__(self, module, device: torch.device):
        super().__init__(module, device)

    def predict_step_forward(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """执行前向预测"""
        batch = batch.to(self.device)

        with torch.no_grad():
            pred_data = self.module.predict_step_forward(batch)
            return pred_data

    @staticmethod
    def create_module(ckpt_path: str, model_cfg: DictConfig, num_classes: int,
                     map_location: torch.device, strict: bool = False):
        """
        创建模块的静态方法（方便工厂类调用）

        Args:
            ckpt_path: 检查点路径
            model_cfg: 模型配置
            num_classes: 类别数
            map_location: 设备映射
            strict: 是否严格匹配

        Returns:
            module: PTSegNetModule
        """
        from model._cross_tooth import PTSegNetModule

        return PTSegNetModule.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            model_cfg=model_cfg,
            num_classes=num_classes,
            strict=strict,
            map_location=map_location
        )


# ========== 【具体实现】DTSegNetPredictor ==========

class DTSegNetPredictor(BasePredictor):
    """
    DTSegNetModule的预测器
    """

    def __init__(self, module, device: torch.device):
        super().__init__(module, device)

    def predict_step_forward(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """执行前向预测"""
        batch = batch.to(self.device)

        with torch.no_grad():
            pred_data = self.module.predict_step_forward(batch)
            return pred_data


    @staticmethod
    def create_module(ckpt_path: str, model_cfg: DictConfig, num_classes: int,
                     map_location: torch.device, strict: bool = False):
        """
        创建 DTSegNetModule

        Args:
            ckpt_path: 检查点路径
            model_cfg: 模型配置
            num_classes: 类别数
            map_location: 设备映射
            strict: 是否严格匹配

        Returns:
            module: DTSegNetModule
        """
        try:
            from model._DilatedToothSegNet.dilated_tooth_seg_module import DilatedToothSegModule

            return DilatedToothSegModule.load_from_checkpoint(
                checkpoint_path=ckpt_path,
                model_cfg=model_cfg,
                num_classes=num_classes,
                strict=strict,
                map_location=map_location
            )
        except ImportError as e:
            raise ImportError(f"无法导入 DTSegNetModule: {str(e)}")


# ========== 【工厂类】PredictorFactory ==========

class PredictorRegistry:
    """
    预测器注册表
    用于管理所有可用的预测器类型
    """

    _registry: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(cls, name: str, predictor_class: type, config_creator: callable = None):
        """
        注册预测器类型

        Args:
            name: 注册名称
            predictor_class: 预测器类
            config_creator: 配置创建函数（可选）
        """
        if not issubclass(predictor_class, BasePredictor):
            raise TypeError(f"{predictor_class.__name__} 必须继承自 BasePredictor")

        cls._registry[name] = {
            'class': predictor_class,
            'config_creator': config_creator
        }

        # print(f"✅ 已注册预测器：{name} -> {predictor_class.__name__}")

    @classmethod
    def get(cls, name: str) -> Dict[str, Any]:
        """
        获取注册的预测器信息

        Args:
            name: 注册名称

        Returns:
            dict: 包含 class 和 config_creator 的字典
        """
        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise KeyError(f"未知的预测器类型：{name}\n可用的类型：{available}")

        return cls._registry[name]

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """检查是否已注册"""
        return name in cls._registry

    @classmethod
    def list_all(cls) -> List[str]:
        """列出所有已注册的预测器类型"""
        return list(cls._registry.keys())


# ========== 【具体实现】BinDTSegNetPredictor ==========

class BinDTSegNetPredictor(BasePredictor):
    """
    BinDTSegNetModule 的预测器
    """

    def __init__(self, module, device: torch.device):
        super().__init__(module, device)

    def predict_step_forward(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """执行前向预测"""
        batch = batch.to(self.device)

        with torch.no_grad():
            pred_data = self.module.predict_step_forward(batch)
            return pred_data


    @staticmethod
    def create_module(ckpt_path: str, model_cfg: DictConfig, num_classes: int,
                     map_location: torch.device, strict: bool = False):
        """
        创建 BinDTSegNetModule

        Args:
            ckpt_path: 检查点路径
            model_cfg: 模型配置
            num_classes: 类别数
            map_location: 设备映射
            strict: 是否严格匹配

        Returns:
            module: BinDTSegNetModule
        """
        try:
            from model._EdgeSegNet.bin_dtseg_module import BinDTSegModule

            return BinDTSegModule.load_from_checkpoint(
                checkpoint_path=ckpt_path,
                model_cfg=model_cfg,
                num_classes=num_classes,
                strict=strict,
                map_location=map_location
            )
        except ImportError as e:
            raise ImportError(f"无法导入 BinDTSegNetModule: {str(e)}")



# ========== 【单例缓存】ModelModuleCache ==========

class PredictorCache:
    """
    模型模块缓存（单例模式）
    """

    _instance = None
    _cache: Dict[str, BasePredictor] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_or_create(self, key: str, creator_func) -> BasePredictor:
        """获取或创建模型预测器"""
        if key not in self._cache:
            # print(f"\n🔧 创建新预测器：{key}")
            self._cache[key] = creator_func()
        else:
            print(f"\n✅ 使用缓存的预测器：{key}")

        return self._cache[key]

    def clear(self):
        """清空缓存"""
        self._cache.clear()
        print("🗑️ 已清空模型缓存")

    def keys(self) -> List[str]:
        """返回所有缓存的键"""
        return list(self._cache.keys())


# ========== 【主工厂类】PredictorFactory ==========

class PredictorFactory:
    """
    预测器工厂类（支持扩展）
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def initialize_default_predictors(cls):
        """
        初始化默认的预测器类型
        应该在程序启动时调用一次
        """
        # 注册 PTSegNet 预测器
        PredictorRegistry.register(
            name='PTSegNet',
            predictor_class=PTSegNetPredictor,
            config_creator=cls._create_ptsegnet_config
        )

        # 注册 DTSegNet 预测器
        PredictorRegistry.register(
            name='DTSegNet',
            predictor_class=DTSegNetPredictor,
            config_creator=cls._create_dtsegnet_config
        )

        # 注册 BinDTSegNet 预测器
        PredictorRegistry.register(
            name='BinDTSegNet',
            predictor_class=BinDTSegNetPredictor,
            config_creator=cls._create_bindtsegnet_config
        )

        # print(f"✅ 已初始化默认预测器：{PredictorRegistry.list_all()}")

    @staticmethod
    def _create_ptsegnet_config(net_para_name: str = 'seg50', **kwargs) -> DictConfig:
        """创建 PTSegNet 配置"""
        networks_db = {
            'seg50': {
                'in_channels': 6,
                'num_classes': 15,
                'add_cbl': True,
                'enable_pic_feat': False,
                'pretrain': False,
                'planes': [32, 64, 128, 256, 512],
                'blocks': [2, 3, 4, 6, 3],
                'stride': [1, 4, 4, 4, 4],
                'nsample': [8, 16, 32, 32, 32],
                'share_planes': 8,
            },
            'seg_23463': {
                'in_channels': 6,
                'num_classes': 15,
                'add_cbl': True,
                'enable_pic_feat': False,
                'pretrain': False,
                'planes': [32, 64, 128, 256],
                'blocks': [2, 3, 4, 3],
                'stride': [1, 4, 4, 4],
                'nsample': [8, 16, 32, 32],
                'share_planes': 8,
            },
        }

        if net_para_name not in networks_db:
            raise ValueError(f"未知的 PTSegNet 配置：{net_para_name}")

        model_cfg_dict = {
            'name': 'PTSegNetModule',
            'net_name': 'PTSegNet',
            'net_para_name': net_para_name,
            'loss_type': kwargs.get('loss_type', 'SEG_TAE_TTE_CBL'),
            'add_cbl': 'CBL' in kwargs.get('loss_type', 'SEG_TAE_TTE_CBL'),
            'version': '1.0',
            'num_classes': kwargs.get('num_classes', 15),
            'feature_dim': kwargs.get('feature_dim', 6),
            'lr': 0.001,
            'num_epochs': 30,
            'num_batches_to_print': 150,
            'k': 8,
            'lamb': 0.1,
            'dropout': 0.3,
            'compute_edge_binary_iou': True
        }

        model_cfg_obj = OmegaConf.create(model_cfg_dict)
        OmegaConf.update(model_cfg_obj, 'net_params', networks_db[net_para_name])

        return model_cfg_obj

    @staticmethod
    def _create_dtsegnet_config(net_para_name: str = 'seg_base', **kwargs) -> DictConfig:
        """
        创建 DTSegNet 配置

        Args:
            net_para_name: 网络参数名称
                - 'seg_base': 基础分割配置
            **kwargs: 其他参数

        Returns:
            model_cfg_obj: OmegaConf DictConfig 对象
        """
        # DTSegNet 的网络配置数据库
        networks_db = {
            'seg_base': {
                # 边缘图卷积块配置
                'edge_conv_blocks': [
                    {'in_channels': 6, 'out_channels': 24, 'hidden_channels': 24,
                     'kernel_size': 32, 'edge_function': 'local_global'},
                    {'in_channels': 24, 'out_channels': 24, 'hidden_channels': 24,
                     'kernel_size': 32, 'edge_function': 'local_global'},
                    {'in_channels': 24, 'out_channels': 24, 'hidden_channels': 24,
                     'kernel_size': 32, 'edge_function': 'local_global'},
                ],

                # 膨胀卷积块配置
                'dilated_conv_blocks': [
                    {'in_channels': 60, 'out_channels': 60, 'hidden_channels': 60,
                     'kernel_size': 32, 'edge_function': 'local_global', 'dilation_rate': 200},
                    {'in_channels': 60, 'out_channels': 60, 'hidden_channels': 60,
                     'kernel_size': 32, 'edge_function': 'local_global', 'dilation_rate': 900},
                    {'in_channels': 60, 'out_channels': 60, 'hidden_channels': 60,
                     'kernel_size': 32, 'edge_function': 'local_global', 'dilation_rate': 2700},
                ],

                # 隐藏层配置
                'hidden_layers': {
                    'local_feature_compression': {'in_channels': 72, 'out_channels': 60},
                    'global_feature_compression': {'in_channels': 240, 'out_channels': 1024},
                    'cbl_feature_compression': {'in_channels': 256, 'out_channels': 32},
                },

                # 残差块配置
                'residual_blocks': [
                    {'in_channels': 1024, 'hidden_channels': 512, 'out_channels': 512},
                    {'in_channels': 512, 'hidden_channels': 256, 'out_channels': 256},
                ],

                # 输出头配置
                'output_heads': {
                    'semantic_segmentation': {'in_channels': 256, 'out_channels': 15},
                    'edge_segmentation': {'in_channels': 256, 'out_channels': 2},
                    'tte_segmentation': {'in_channels': 256, 'out_channels': 2},
                },

                # STN 变换
                'stn_transform': {
                    'input_channels': 6,
                    'layers': [64, 128, 1024]
                },

                # 特征注意力
                'feature_attention': {
                    'in_channels': 1024,
                    'activation': 'sigmoid'
                },

                # 通用配置
                'num_classes': '15',  # 引用父级配置
                'feature_dim': '6',  # 引用父级配置
                'add_cbl': 'True',  # 引用父级配置
            }
        }

        # 检查配置是否存在
        if net_para_name not in networks_db:
            available_configs = list(networks_db.keys())
            raise ValueError(
                f"未知的 DTSegNet 配置：{net_para_name}\n"
                f"可用的配置：{available_configs}"
            )

        # 构建完整的 model 配置
        model_cfg_dict = {
            'name': 'DTSegNetModule',
            'net_name': 'DTSegNet',
            'net_para_name': net_para_name,
            'loss_type': kwargs.get('loss_type', 'SEG_TAE_TTE'),
            'add_cbl': 'CBL' in kwargs.get('loss_type', 'SEG_TAE_TTE'),
            'version': '1.0',
            'num_classes': kwargs.get('num_classes', 15),
            'feature_dim': kwargs.get('feature_dim', 6),
            'lr': kwargs.get('lr', 0.001),
            'num_epochs': kwargs.get('num_epochs', 30),
            'num_batches_to_print': kwargs.get('num_batches_to_print', 150),
            'k': kwargs.get('k', 8),
            'lamb': kwargs.get('lamb', 0.1),
            'dropout': kwargs.get('dropout', 0.3),
            'compute_edge_binary_iou': kwargs.get('compute_edge_binary_iou', True),
            # DTSegNet 特有配置
            # 'use_stn': kwargs.get('use_stn', True),
            # 'use_cbl': kwargs.get('use_cbl', False),
        }

        # 创建 OmegaConf 对象
        model_cfg_obj = OmegaConf.create(model_cfg_dict)

        # 添加网络参数配置
        OmegaConf.update(model_cfg_obj, 'net_params', networks_db[net_para_name])
        model_cfg_obj.net_params.num_classes = model_cfg_obj.num_classes
        model_cfg_obj.net_params.feature_dim = model_cfg_obj.feature_dim
        model_cfg_obj.net_params.add_cbl = model_cfg_obj.add_cbl

        # print(f"\n✅ DTSegNet 配置创建成功:")
        # print(f"  net_para_name: {net_para_name}")
        # print(f"  edge_conv_blocks: {len(networks_db[net_para_name]['edge_conv_blocks'])} 层")
        # print(f"  dilated_conv_blocks: {len(networks_db[net_para_name]['dilated_conv_blocks'])} 层")
        # print(f"  residual_blocks: {len(networks_db[net_para_name]['residual_blocks'])} 层")
        # print(f"  output_heads: {list(networks_db[net_para_name]['output_heads'].keys())}")

        return model_cfg_obj


    @staticmethod
    def _create_bindtsegnet_config(net_para_name: str = 'seg_bin', **kwargs) -> DictConfig:
        """
        创建 BinDTSegNet 配置

        Args:
            net_para_name: 网络参数名称
                - 'seg_bin': 二分类分割配置
            **kwargs: 其他参数

        Returns:
            model_cfg_obj: OmegaConf DictConfig 对象
        """
        # BinDTSegNet 的网络配置数据库（与 DTSegNet 类似，但使用 seg_bin 配置）
        networks_db = {
            'seg_bin_1536': {
                # 边缘图卷积块配置（轻量版）
                'edge_conv_blocks': [
                    {'in_channels': 6, 'out_channels': 20, 'hidden_channels': 20,
                     'kernel_size': 8, 'edge_function': 'local_global'},
                    {'in_channels': 20, 'out_channels': 20, 'hidden_channels': 20,
                     'kernel_size': 24, 'edge_function': 'local_global'},
                    {'in_channels': 20, 'out_channels': 20, 'hidden_channels': 20,
                     'kernel_size': 56, 'edge_function': 'local_global'},
                ],

                # 隐藏层配置
                'hidden_layers': {
                    'local_feature_compression': {'in_channels': 60, 'out_channels': 48},
                    'global_feature_compression': {'in_channels': 192, 'out_channels': 512},
                    'cbl_feature_compression': {'in_channels': 256, 'out_channels': 32},
                },

                # 膨胀卷积块配置
                'dilated_conv_blocks': [
                    {'in_channels': 48, 'out_channels': 48, 'hidden_channels': 48,
                     'kernel_size': 12, 'edge_function': 'local_global', 'dilation_rate': 32},
                    {'in_channels': 48, 'out_channels': 48, 'hidden_channels': 48,
                     'kernel_size': 24, 'edge_function': 'local_global', 'dilation_rate': 256},
                    {'in_channels': 48, 'out_channels': 48, 'hidden_channels': 48,
                     'kernel_size': 48, 'edge_function': 'local_global', 'dilation_rate': 1536},
                ],

                # 残差块配置
                'residual_blocks': [
                    {'in_channels': 512, 'hidden_channels': 512, 'out_channels': 512},
                    {'in_channels': 512, 'hidden_channels': 256, 'out_channels': 256},
                ],

                # 输出头配置
                'output_heads': {
                    'semantic_segmentation': {'in_channels': 256, 'out_channels': 15},
                    'bg_segmentation': {'in_channels': 256, 'out_channels': 2},
                    'edge_segmentation': {'in_channels': 256, 'out_channels': 2},
                    'tte_segmentation': {'in_channels': 256, 'out_channels': 2},
                },

                # STN 变换
                'stn_transform': {
                    'input_channels': 6,
                    'layers': [64, 128, 512]
                },

                # 特征注意力
                'feature_attention': {
                    'in_channels': 512,
                    'activation': 'sigmoid'
                },

                # 通用配置
                'num_classes': '15',
                'feature_dim': '6',
                'add_cbl': 'True',
            }
        }

        # 检查配置是否存在
        if net_para_name not in networks_db:
            available_configs = list(networks_db.keys())
            raise ValueError(
                f"未知的 BinDTSegNet 配置：{net_para_name}\n"
                f"可用的配置：{available_configs}"
            )

        # 构建完整的 model 配置
        model_cfg_dict = {
            'name': 'BinDTSegNetModule',
            'net_name': 'BinDTSegNet',
            'net_para_name': net_para_name,
            'loss_type': kwargs.get('loss_type', 'BG_TAE_TTE'),
            'add_cbl': 'CBL' in kwargs.get('loss_type', 'BG_TAE_TTE'),
            'version': '1.0',
            'num_classes': kwargs.get('num_classes', 15),
            'feature_dim': kwargs.get('feature_dim', 6),
            'lr': kwargs.get('lr', 0.001),
            'num_epochs': kwargs.get('num_epochs', 30),
            'num_batches_to_print': kwargs.get('num_batches_to_print', 150),
            'k': kwargs.get('k', 8),
            'lamb': kwargs.get('lamb', 0.1),
            'dropout': kwargs.get('dropout', 0.3),
            'compute_edge_binary_iou': kwargs.get('compute_edge_binary_iou', True),
        }


        # 检查配置是否存在
        if net_para_name not in networks_db:
            available_configs = list(networks_db.keys())
            raise ValueError(
                f"未知的 BinDTSegNet 配置：{net_para_name}\n"
                f"可用的配置：{available_configs}"
            )


        # 创建 OmegaConf 对象
        model_cfg_obj = OmegaConf.create(model_cfg_dict)

        # 添加网络参数配置
        OmegaConf.update(model_cfg_obj, 'net_params', networks_db[net_para_name])
        model_cfg_obj.net_params.num_classes = model_cfg_obj.num_classes
        model_cfg_obj.net_params.feature_dim = model_cfg_obj.feature_dim
        model_cfg_obj.net_params.add_cbl = model_cfg_obj.add_cbl

        return model_cfg_obj

    @classmethod
    def get_predictor(cls,
                      predictor_type: str,
                      ckpt_path: str,
                      net_para_name: str,
                      device: Optional[torch.device] = None,
                      **kwargs) -> BasePredictor:
        """
        获取或创建预测器（单例模式）
        Args:
            predictor_type: 预测器类型（如 'PTSegNet', 'DTSegNet'）
            ckpt_path: 检查点路径
            net_para_name: 网络参数名称
            device: 设备（None 则自动选择）
            **kwargs: 其他参数传递给配置创建函数
        Returns:
            predictor: BasePredictor 实例
        Raises:
            FileNotFoundError: 当检查点文件不存在时
            RuntimeError: 当模型加载失败时
            ValueError: 当预测器类型未注册时
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"检查点文件不存在：{ckpt_path}\n"
                f"请确认路径是否正确，或使用绝对路径"
            )

        # 检查文件是否为空
        if os.path.getsize(ckpt_path) == 0:
            raise RuntimeError(f"检查点文件为空：{ckpt_path}")

        # print(f"\n✅ 检查点验证通过:")
        # print(f"   路径：{ckpt_path}")
        # print(f"   大小：{os.path.getsize(ckpt_path) / 1024 / 1024:.2f} MB")

        # ========== 生成缓存键 ==========
        # 使用绝对路径生成键，避免相对路径问题
        abs_ckpt_path = os.path.abspath(ckpt_path)
        cache_key = f"{predictor_type}:{net_para_name}:{abs_ckpt_path}"

        cache = PredictorCache()

        try:
            # 如果缓存中已有，直接返回
            if cache_key in cache._cache:
                # print(f"\n✅ 使用缓存的预测器：{cache_key}")
                return cache._cache[cache_key]

            # 尝试创建新的预测器
            # print(f"\n🔧 开始创建预测器：{cache_key}")

            # 先创建对象，成功后再放入缓存
            predictor = cls._create_predictor(
                predictor_type=predictor_type,
                ckpt_path=abs_ckpt_path,  # 使用绝对路径
                net_para_name=net_para_name,
                device=device,
                **kwargs
            )

            # 只有成功创建后才放入缓存
            cache._cache[cache_key] = predictor

            # print(f"\n{'=' * 70}")
            # print(f"✅ 预测器创建并缓存成功：{predictor}")
            # print(f"{'=' * 70}\n")

            return predictor

        except FileNotFoundError as e:
            # 文件不存在错误，重新抛出
            print(f"\n❌ 文件错误：{str(e)}")
            raise

        except RuntimeError as e:
            # 运行时错误（如加载失败），重新抛出
            print(f"\n❌ 模型加载失败：{str(e)}")
            raise

        except Exception as e:
            # 其他未知错误
            error_msg = (
                f"\n创建预测器时发生错误:\n"
                f"  类型：{predictor_type}\n"
                f"  路径：{ckpt_path}\n"
                f"  参数：{net_para_name}\n"
                f"  错误：{str(e)}\n"
                f"\n可能的原因:\n"
                f"  1. 检查点文件损坏\n"
                f"  2. 配置不匹配\n"
                f"  3. 缺少依赖项\n"
            )
            print(error_msg)
            raise RuntimeError(error_msg) from e

    @classmethod
    def _create_predictor(cls,
                          predictor_type: str,
                          ckpt_path: str,
                          net_para_name: str,
                          device: torch.device,
                          **kwargs) -> BasePredictor:
        """
        内部方法：创建预测器（不处理缓存）
        Returns:
            predictor: BasePredictor 实
        Raises:
            RuntimeError: 当创建失败时
        """
        # print(f"\n{'=' * 70}")
        # print(f"创建预测器")
        # print(f"{'=' * 70}")
        # print(f"  类型：{predictor_type}")
        # print(f"  检查点：{ckpt_path}")
        # print(f"  参数：{net_para_name}")
        # print(f"  设备：{device}")

        # 1. 获取注册的预测器信息
        reg_info = PredictorRegistry.get(predictor_type)
        predictor_class = reg_info['class']
        config_creator = reg_info.get('config_creator')

        # 2. 创建配置（如果有配置创建函数）
        if config_creator:
            try:
                model_cfg = config_creator(net_para_name=net_para_name, **kwargs)
                # print(f"\n✅ 配置创建成功")
                # print(f"  loss_type: {model_cfg.loss_type}")
                # print(f"  num_classes: {model_cfg.num_classes}")
            except Exception as cfg_error:
                raise RuntimeError(f"创建配置失败：{str(cfg_error)}") from cfg_error
        else:
            model_cfg = kwargs.get('model_cfg', None)

        # 3. 加载模型
        try:
            # print(f"\n正在加载模型...")

            module = predictor_class.create_module(
                ckpt_path=ckpt_path,
                model_cfg=model_cfg,
                num_classes=model_cfg.num_classes if model_cfg else 15,
                map_location=device,
                strict=False
            )

            # 验证模型是否成功加载
            param_count = sum(p.numel() for p in module.parameters())
            # print(f"\n✅ 模型加载成功")
            # print(f"  参数量：{param_count:,}")
            # print(f"  设备：{next(module.parameters()).device}")

            # 验证模型是否在正确的设备上
            if not cls._is_same_device(device, next(module.parameters()).device):
                raise RuntimeError(
                    f"模型设备不匹配：期望 {device}, 实际 {next(module.parameters()).device}"
                )

        except FileNotFoundError:
            # 文件不存在，直接抛出
            raise

        except Exception as load_error:
            raise RuntimeError(
                f"加载模型失败：{ckpt_path}\n"
                f"错误详情：{str(load_error)}"
            ) from load_error

        # 4. 创建预测器包装
        try:
            predictor = predictor_class(module, device)
            # print(f"\n✅ 预测器包装创建成功")
        except Exception as wrap_error:
            raise RuntimeError(f"创建预测器包装失败：{str(wrap_error)}") from wrap_error

        # print(f"\n{'=' * 70}")
        # print(f"预测器创建完成")
        # print(f"{'=' * 70}\n")

        return predictor

    @staticmethod
    def _is_same_device(device1, device2):
        # cuda 和 cuda:0 应该被认为是相同的
        if device1.type != device2.type:
            return False

        if device1.type == 'cuda':
            idx1 = device1.index if device1.index is not None else 0
            idx2 = device2.index if device2.index is not None else 0
            return idx1 == idx2

        return True

    @classmethod
    def register_custom_predictor(cls, name: str, predictor_class: type,
                                 config_creator: callable = None):
        """
        注册自定义预测器类型

        Args:
            name: 注册名称
            predictor_class: 预测器类
            config_creator: 配置创建函数
        """
        PredictorRegistry.register(name, predictor_class, config_creator)
        # print(f"✅ 已注册自定义预测器：{name}")


# ========== 【使用示例】如何添加新的预测器 ==========

"""
# 示例：添加一个新的预测器类型

class CustomSegNetPredictor(BasePredictor):
    '''自定义分割网络预测器'''
    
    def __init__(self, module, device):
        super().__init__(module, device)
    
    def predict_step_forward(self, batch):
        # 自定义预测逻辑
        with torch.no_grad():
            outputs = self.module.custom_forward(batch)
        return outputs
    
    @staticmethod
    def create_module(ckpt_path, model_cfg, num_classes, map_location, strict=False):
        from model.custom_module import CustomSegNetModule
        return CustomSegNetModule.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            model_cfg=model_cfg,
            num_classes=num_classes,
            strict=strict,
            map_location=map_location
        )

# 注册到工厂
def create_custom_config(net_para_name, **kwargs):
    # 创建配置的逻辑
    return OmegaConf.create({...})

PredictorFactory.register_custom_predictor(
    name='CustomSegNet',
    predictor_class=CustomSegNetPredictor,
    config_creator=create_custom_config
)

# 使用
predictor = PredictorFactory.get_predictor(
    predictor_type='CustomSegNet',
    ckpt_path='custom.ckpt',
    net_para_name='custom_v1'
)
"""


# ========== 【工具函数】==========

def get_available_predictors() -> List[str]:
    """获取所有可用的预测器类型"""
    return PredictorRegistry.list_all()


def clear_model_cache():
    """清空模型缓存"""
    PredictorCache().clear()


# 自动初始化默认预测器
PredictorFactory.initialize_default_predictors()

if __name__ == '__main__':
    pt_predictor = PredictorFactory.get_predictor(
        predictor_type='PTSegNet',
        ckpt_path='PTSeg50.ckpt',
        net_para_name='seg50'
    )

    pt_predictor = PredictorFactory.get_predictor(
        predictor_type='PTSegNet',
        ckpt_path='PTSeg50.ckpt',
        net_para_name='seg50'
    )

    # 使用 DTSegNet
    dt_predictor = PredictorFactory.get_predictor(
        predictor_type='DTSegNet',
        ckpt_path='ckpts/DTSegnet.ckpt',
        net_para_name='seg_base'
    )


    # 使用 DTSegNet
    bindt_predictor = PredictorFactory.get_predictor(
        predictor_type='BinDTSegNet',
        ckpt_path='BinDTSeg-0.1ratio.ckpt',
        net_para_name='seg_bin_1536'
    )

    # 查看可用类型
    print(get_available_predictors())  # ['PTSegNet', 'DTSegNet']
