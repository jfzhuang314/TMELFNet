import hydra
from omegaconf import DictConfig, OmegaConf


import hydra
from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf
import os
from pathlib import Path
from typing import Optional


def get_yaml_config(config_path: str = "../configs", config_name: str = "config", version_base: str = "1.2") -> DictConfig:
    """
    获取Hydra配置的通用函数
    """
    # 转换为绝对路径
    abs_config_path = os.path.abspath(config_path)
    abs_config_path = Path(abs_config_path)

    if not abs_config_path.is_dir():
        raise ValueError(f"配置路径无效: {abs_config_path}")

    # 检查是否已经在Hydra环境中
    if hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        try:
            cfg = hydra.compose(config_name=config_name)
            return cfg
        except Exception as e:
            print(f"[WARNING] Hydra环境中compose失败: {e}")

    # 不在Hydra环境或compose失败，重新初始化
    print(f"[INFO] 初始化Hydra配置目录: {abs_config_path}")
    try:
        with hydra.initialize_config_dir(config_dir=str(abs_config_path), version_base=version_base):
            cfg = hydra.compose(config_name=config_name)
            return cfg
    except Exception as init_e:
        print(f"[ERROR] 初始化配置目录失败: {init_e}")
        raise


def get_args_from_config(config_path: str = "./configs", config_name: str = "test_T3DSPlusDataset",
                         version_base: str = None, print_yaml: bool = False):
    """
    从配置获取参数对象，可选择是否打印 YAML 配置
    Args:
        config_path: 配置文件路径
        config_name: 配置文件名
        version_base: Hydra 版本基础
        print_yaml: 是否打印 YAML 配置内容
    Returns:
        配置的 paths 部分
    """
    cfg = get_yaml_config(config_path, config_name, version_base)

    if print_yaml:
        print("YAML Configuration:")
        print(OmegaConf.to_yaml(cfg))
        print("-" * 50)

    return cfg.paths  # YAML中定义的是paths

# 打印YAML配置，提供了两种打印方法
def print_yaml_config(config_path="./configs", config_name="test_T3DSPlusDataset", version_base: str = None):
    cfg = get_yaml_config(config_path, config_name, version_base)
    print("YAML Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("-" * 50)

def print_yaml_cfg_dict(cfg):
    print("YAML Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("-" * 50)

def instantiate_from_config(config_obj):
    """
    从配置对象实例化对象的通用工具函数
    :param config_obj: 配置对象，可以是字符串、OmegaConf对象或字典
    :return: 实例化后的对象
    """
    # 检查是否已经是实例化的对象
    # if hasattr(config_obj, '__class__') and not isinstance(config_obj, (str, dict)):
    #     # 如果对象已经有 __call__ 方法，说明可能是变换对象，直接返回
    #     if hasattr(config_obj, '__call__'):
    #         return config_obj
    #     # 如果是普通的已实例化对象，也直接返回
    #     return config_obj


    if isinstance(config_obj, str):
        # 如果是字符串，尝试将其解析为OmegaConf对象
        return hydra.utils.instantiate(OmegaConf.create(config_obj))
    elif OmegaConf.is_config(config_obj):
        # 如果已经是OmegaConf配置对象
        return hydra.utils.instantiate(config_obj)
    else:
        # 如果是普通对象或字典
        return hydra.utils.instantiate(config_obj)

def instantiate_from_yaml_config(config_path: str = "../configs", config_name: str = "config", version_base: str = "1.2"):
    cfg = get_yaml_config(config_path, config_name, version_base)
    cfg_obj = instantiate_from_config(cfg)
    return cfg_obj


if __name__ == "__main__":
    # 使用
    args = get_args_from_config(config_path="", config_name="test_T3DSPlusDataset", print_yaml=False)

    cfg = get_yaml_config(config_path="", config_name="test_T3DSPlusDataset")
    print_yaml_config(config_path="", config_name="test_T3DSPlusDataset")

    print(cfg.pre_transform_obj)
    # 打印所有路径配置
    print(f"source_obj_data_path: \"{args.source_obj_data_path}\"")
    print(f"source_seg_json_data_path: \"{args.source_seg_json_data_path}\"")
    print(f"source_landmarks_json_data_path: \"{args.source_landmarks_json_data_path}\"")
    print(f"save_data_path: \"{args.save_data_path}\"")
    print(f"save_jsondata_path: \"{args.save_jsondata_path}\"")