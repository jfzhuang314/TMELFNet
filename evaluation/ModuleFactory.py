"""
简化的 Module 生成工厂
用于根据配置动态创建不同的模型模块
"""

import os
import torch
from typing import Optional, Dict, Type
from omegaconf import DictConfig

from configs.hydra_yaml_config import *


class ModuleFactory:
    """Module 工厂类 - 简化版"""

    # 模块注册表
    _modules = {}

    @classmethod
    def register(cls, name: str, module_class: Type):
        """注册模块类"""
        cls._modules[name] = module_class

    @classmethod
    def create(cls,
               module_name: str,
               model_cfg: DictConfig,
               num_classes: int = 15,
               **kwargs) -> torch.nn.Module:
        """
        创建模块实例

        :param module_name: 模块名称 (如 "DTSegModule")
        :param model_cfg: 模型配置
        :param num_classes: 类别数
        :param kwargs: 其他参数
        :return: 模块实例
        """
        if module_name not in cls._modules:
            available = list(cls._modules.keys())
            raise ValueError(f"未知模块：{module_name}, 可用：{available}")

        module_class = cls._modules[module_name]

        try:
            module = module_class(model_cfg, num_classes=num_classes, **kwargs)
            print(f"✅ 创建成功：{module_name}")
            return module
        except Exception as e:
            raise RuntimeError(f"创建失败 {module_name}: {e}")

    @classmethod
    def load_from_checkpoint(cls,
                           module_name: str,
                           checkpoint_path: str,
                           model_cfg: DictConfig = None,
                           map_location: str = "cuda",
                           strict: bool = False) -> torch.nn.Module:
        """
        从检查点加载模块

        :param module_name: 模块名称
        :param checkpoint_path: 检查点路径
        :param model_cfg: 模型配置（可选）
        :param map_location: 设备映射
        :param strict: 严格加载
        :return: 模块实例
        """
        if module_name not in cls._modules:
            raise ValueError(f"未知模块：{module_name}")

        module_class = cls._modules[module_name]

        # 检查文件是否存在
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"检查点不存在：{checkpoint_path}")

        try:
            # 尝试带 config 加载
            if model_cfg is not None:
                module = module_class.load_from_checkpoint(
                    checkpoint_path=checkpoint_path,
                    model_cfg=model_cfg,
                    strict=strict,
                    map_location=torch.device(map_location)
                )
            else:
                # 直接加载
                module = module_class.load_from_checkpoint(
                    checkpoint_path=checkpoint_path,
                    strict=strict,
                    map_location=torch.device(map_location)
                )

            print(f"✅ 加载成功：{module_name} from {os.path.basename(checkpoint_path)}")
            return module

        except Exception as e:
            raise RuntimeError(f"加载检查点失败 {module_name}: {e}")


# ========== 自动注册模块 ==========
def auto_register_modules():
    """自动注册所有可用的模块"""

    # 延迟导入，避免循环依赖
    try:
        from model._cross_tooth.cross_tooth_module import PTSegNetModule
        ModuleFactory.register("PTSegNetModule", PTSegNetModule)
    except ImportError as e:
        print(f"⚠️  无法导入 PTSegNetModule: {e}")

    try:
        from model._DilatedToothSegNet.dilated_tooth_seg_module import DilatedToothSegModule
        ModuleFactory.register("DTSegNetModule", DilatedToothSegModule)
    except ImportError as e:
        print(f"⚠️  无法导入 DilatedToothSegModule: {e}")

    # try:
    #     from model._TSegNet.tsegnet_module import TSegModule
    #     ModuleFactory.register("TSegModule", TSegModule)
    # except ImportError as e:
    #     print(f"⚠️  无法导入 TSegModule: {e}")

    try:
        from model._TMELFNet.tmelf_seg_module import TMELFSegModule
        ModuleFactory.register("TMELFSegModule", TMELFSegModule)
    except ImportError as e:
        print(f"⚠️  无法导入 TMELFSegModule: {e}")

    try:
        from model._BinDTSegNet.bin_dtseg_module import BinDTSegModule
        ModuleFactory.register("BinDTSegModule", BinDTSegModule)
    except ImportError as e:
        print(f"⚠️  无法导入 BinDTSegModule: {e}")

    try:
        from model._pointnetpp.pointnetpp_module import PointnetPPModule
        ModuleFactory.register("PointnetppModule", PointnetPPModule)
    except ImportError as e:
        print(f"⚠️  无法导入 PointnetppModule: {e}")

    try:
        from model._TeethGNN.teethgnn_module import TeethGNNModule
        ModuleFactory.register("TeethGNNModule", TeethGNNModule)
    except ImportError as e:
        print(f"⚠️  无法导入 TeethGNNModule: {e}")

    try:
        from model._TSegNet.tsegnet_module import TSegModule
        ModuleFactory.register("TSegNetModule", TSegModule)
    except ImportError as e:
        print(f"⚠️  无法导入 TSegNetModule: {e}")

    try:
        from model._PatchSegNet.patch_seg_module import PatchSegModule
        ModuleFactory.register("PatchSegModule", PatchSegModule)
    except ImportError as e:
        print(f"⚠️  无法导入 PatchSegModule: {e}")



    print(f"📦 已注册模块：{list(ModuleFactory._modules.keys())}")


# 自动执行注册
# auto_register_modules()


# ========== 使用示例 ==========
if __name__ == "__main__":
    from omegaconf import OmegaConf

    print("="*60)
    print("ModuleFactory 使用示例")
    print("="*60)

    # 示例 1: 从配置创建
    config_path = "../configs"
    config_name = "eval_segnet"
    cfg = get_yaml_config(config_path=config_path, config_name=config_name)
    args = cfg.paths
    ds_args = cfg.dataset
    print_yaml_cfg_dict(cfg)

    cfg_obj = instantiate_from_config(cfg)
    # print(cfg_obj)

    print(cfg_obj.predictors.DTSegNetModule.model_cfg)
    print(cfg_obj.predictors.DTSegNetModule.ckpt_cfg.full_path)

    models_config = {
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
        "PatchSegModule": {
            "ckpt_path": cfg_obj.predictors.PatchSegModule.ckpt_cfg.full_path,
            "model_cfg": cfg_obj.predictors.PatchSegModule.model_cfg
        }
    }

    auto_register_modules()
    # 从检查点加载
    # module = ModuleFactory.load_from_checkpoint(
    #     module_name="DTSegNetModule",
    #     checkpoint_path=models_config["DTSegNetModule"]["ckpt_path"],
    #     model_cfg=models_config["DTSegNetModule"]["model_cfg"],
    #     map_location="cuda"
    # )
    #
    # module = ModuleFactory.load_from_checkpoint(
    #     module_name="PTSegNetModule",
    #     checkpoint_path=models_config["PTSegNetModule"]["ckpt_path"],
    #     model_cfg=models_config["PTSegNetModule"]["model_cfg"],
    #     map_location="cuda"
    # )

    # module = ModuleFactory.load_from_checkpoint(
    #     module_name="TMELFSegModule",
    #     checkpoint_path=models_config["TMELFSegModule"]["ckpt_path"],
    #     model_cfg=models_config["TMELFSegModule"]["model_cfg"],
    #     map_location="cuda"
    # )

    module = ModuleFactory.load_from_checkpoint(
        module_name="PatchSegModule",
        checkpoint_path=models_config["PatchSegModule"]["ckpt_path"],
        model_cfg=models_config["PatchSegModule"]["model_cfg"],
        map_location="cuda"
    )