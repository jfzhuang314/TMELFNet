from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Union, Optional

from pytorch_lightning.loggers import (TensorBoardLogger, CSVLogger, WandbLogger)
from torch import Tensor
from pytorch_lightning.loggers import CSVLogger
import os
import torch
from pytorch_lightning.callbacks import ModelCheckpoint


# ===================== 指标管理器抽象接口 =====================
class BaseMetricsManager(ABC):
    def __init__(self, task_type: str, device: torch.device = None):
        self.task_type = task_type
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_metrics()

    @abstractmethod
    def _init_metrics(self):
        """初始化任务相关指标"""
        pass

    @abstractmethod
    def update_metrics(self, preds, targets, **kwargs):
        """更新指标"""
        pass

    @abstractmethod
    def compute_metrics(self):
        """计算并返回指标字典"""
        pass

    @abstractmethod
    def reset_metrics(self):
        """重置指标"""
        pass




# ===================== Logger管理器抽象接口 =====================

def generate_exp_name(exp_args, add_timestamp: bool = True, only_vals= True) -> str:
    """
    根据 OmegaConf 对象动态生成实验名称
    Args:
        exp_args (OmegaConf.DictConfig): 包含实验参数的 OmegaConf 对象
    Returns:
        str: 生成的实验名称字符串
    """
    if not exp_args:
        return "default"

    if add_timestamp:
        TIMESTAMP_FORMAT = "%Y%m%d%H%M%S"
        timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
        exp_args['v'] =timestamp

    if not only_vals:
        template_parts = [f"{key}-{value}" for key, value in exp_args.items()]
        filled_name = "__".join(template_parts)
    else:
        template_parts = [f"{value}" for key, value in exp_args.items()]
        filled_name = "__".join(template_parts)

    return filled_name


class BaseLoggerManager:
    """
    Logger管理器：集中管理所有日志记录器的初始化、配置、统一记录接口
    支持：TensorBoard/CSV/WandB，可灵活扩展其他Logger
    """

    def __init__(
            self,
            save_dir: str = "./logs",       # 日志根目录
            name: str = "pl_demo",          # 项目名（所有Logger共用）
            version: str = "exp1",          # 实验名（区分不同实验）
            add_prefix: bool = False,        # 是否添加
            use_tensorboard: bool = True,   # 是否启用TensorBoard
            use_csv: bool = True,           # 是否启用CSV
            use_wandb: bool = False,        # 是否启用WandB（默认关闭，避免依赖）
            wandb_entity: str = None,       # WandB的entity（用户名/团队名）
            flush_logs_every_n_steps: int = 100  # 每N步刷新日志（默认10步）
    ):
        self.save_dir = save_dir
        self.name = name
        self.version = version
        self._loggers = []  # 存储所有启用的Logger实例
        self.flush_logs_every_n_steps = flush_logs_every_n_steps  # 刷新日志的频率

        # 1. 创建日志目录（避免报错）
        os.makedirs(self.save_dir, exist_ok=True)
        # 2. 集中初始化所有启用的Logger
        self._init_loggers(use_tensorboard, use_csv, use_wandb, wandb_entity)
        # 3. 日志键名前缀（统一格式，避免混淆）
        self.prefix = f"{self.name}/{self.version}" if add_prefix else None

    def generate_exp_name(self, exp_args) -> str:
        """
        根据 OmegaConf 对象动态生成实验名称
        Args:
            exp_args (OmegaConf.DictConfig): 包含实验参数的 OmegaConf 对象
        Returns:
            str: 生成的实验名称字符串
        """
        if not exp_args:
            return "default"

        TIMESTAMP_FORMAT = "%Y%m%d%H%M%S"
        timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
        exp_args['v'] =timestamp

        template_parts = [f"{value}" for key, value in exp_args.items()]
        filled_name = "__".join(template_parts)

        return filled_name
    @classmethod
    def from_config(cls, config):
        """根据 yaml config 对象初始化。config的示例
          save_dir: "../tb_logs/"
          task_name: "ELFNet"
          exp_name: ""
          use_tensorboard: True
          use_csv: True
          use_wandb: False
          wandb_entity: None
          # 实验参数配置（引用已有字段）
          exp_args:
            version: "v1.0"                 # 版本号
            num_classes: "${model.num_classes}"       # 引用 model.num_classes
            num_neighbor: "${model.num_neighbor}"     # 引用 model.num_neighbor
            num_epochs: "${training.num_epochs}"      # 引用 training.num_epochs
            lr: "${training.lr}"                      # 引用 training.lr
            ncells: "${dataset.ncells}"               # 引用 dataset.ncells
            sample_ratio: "${dataset.sample_ratio}"
            train_val_ratio: "${dataset.train_val_ratio}"
            comb_name: "${dataset.comb_name}"
            rmin: "${model.rmin}"
            rmax: "${model.rmax}"
        """
        instance = cls.__new__(cls)  # 创建未初始化的实例
        save_dir = config.get("save_dir", "./logs")
        name = config.get("task_name", "pl_demo")
        add_prefix = config.get("add_prefix", False)
        use_tensorboard = config.get("use_tensorboard", True)
        use_csv = config.get("use_csv", True)
        use_wandb = config.get("use_wandb", False)
        wandb_entity = config.get("wandb_entity", None)

        exp_args = config.exp_args
        version = instance.generate_exp_name(exp_args)


        return cls(save_dir, name, version, add_prefix=add_prefix,
                   use_tensorboard=use_tensorboard,
                   use_csv=use_csv,
                   use_wandb=use_wandb,
                   wandb_entity=wandb_entity,
                   flush_logs_every_n_steps=config.flush_logs_every_n_steps)


    def _init_loggers(self, use_tensorboard, use_csv, use_wandb, wandb_entity):
        """集中初始化各类Logger"""
        # TensorBoard Logger
        if use_tensorboard:
            tb_logger = TensorBoardLogger(
                save_dir=self.save_dir,
                name=f"{self.name}_tb",
                version=self.version,
                default_hp_metric=False,  # 关闭默认超参记录（避免冗余）
            )
            self._loggers.append(tb_logger)
            print(f"✅ 已启用TensorBoard Logger，日志路径：{tb_logger.log_dir}")

        # CSV Logger
        if use_csv:
            csv_logger = CSVLogger(
                save_dir=self.save_dir,
                name=f"{self.name}_csv",
                version=self.version,
                flush_logs_every_n_steps=self.flush_logs_every_n_steps
            )
            self._loggers.append(csv_logger)
            print(f"✅ 已启用CSV Logger，日志路径：{csv_logger.log_dir}")

        # WandB Logger（需提前安装：pip install wandb）
        if use_wandb:
            if not wandb_entity:
                warnings.warn("⚠️ 启用WandB但未指定entity，将使用默认配置")
            try:
                wandb_logger = WandbLogger(
                    project=self.name,
                    name=self.version,
                    entity=wandb_entity,
                    save_dir=self.save_dir,
                    offline=False  # 离线模式设为True（无网络时）
                )
                self._loggers.append(wandb_logger)
                print(f"✅ 已启用WandB Logger，实验名：{self.version}")
            except ImportError:
                warnings.warn("⚠️ 未安装wandb，自动关闭WandB Logger")

        # 无Logger时的兜底（避免Trainer报错）
        if not self._loggers:
            self._loggers = None
            print("⚠️ 未启用任何Logger，训练过程将不记录日志")

    @property
    def loggers(self):
        """返回所有启用的Logger列表（给Trainer使用）"""
        return self._loggers

    def log_metrics(  # type: ignore[override]
            self, metrics: dict[str, Union[Tensor, float]], step: Optional[int] = None) -> None:
        for logger in self._loggers:
            logger.log_metrics(metrics, step=step)
            # logger.save()  # 刷新日志

    def log_hyperparams(self, params):
        """统一记录超参数（所有Logger都会同步）"""
        for logger in self._loggers or []:
            if hasattr(logger, "log_hyperparams"):
                logger.log_hyperparams(params)

    def close(self):
        """训练结束后关闭所有Logger（释放资源）"""
        for logger in self._loggers or []:
            if hasattr(logger, "finalize"):
                logger.finalize("success")
        print("✅ 所有Logger已关闭，日志保存完成")

    def __repr__(self):
        """打印当前配置信息"""
        return f"{self.__class__.__name__}(save_dir={self.save_dir}, name={self.name}, version={self.version}, loggers={self.loggers})"

class BaseCheckpointManager():
    """
    检查点管理器抽象接口：定义检查点管理的核心规范
    所有检查点管理器都继承此接口，保证跨任务/框架的接口统一
    """
    def __init__(self, ckpt_root: str, exp_name: str):
        """
        初始化基础参数
        Args:
            ckpt_root: 检查点根目录（如 './checkpoints/'）
            exp_name: 实验名称（避免不同实验检查点冲突）
        """
        self.ckpt_root = ckpt_root
        self.exp_name = exp_name
        self.ckpt_dir = os.path.join(ckpt_root, exp_name)
        # 确保目录存在
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def get_ckpt_callback(self) -> ModelCheckpoint:
        """抽象方法：获取PL的ModelCheckpoint回调（封装保存逻辑）"""
        pass

    def load_checkpoint(self, model: torch.nn.Module, ckpt_path: str = None, strict: bool = False) -> tuple[torch.nn.Module, str]:
        """
        抽象方法：加载检查点
        Args:
            model: 待加载的模型实例
            ckpt_path: 指定检查点路径（None则加载最新）
            strict: 是否严格匹配state_dict键名
        Returns:
            model: 加载后的模型
            loaded_path: 实际加载的检查点路径
        """
        pass

    def save_checkpoint(self, model: torch.nn.Module, epoch: int, metrics: dict, filename: str = None) -> str:
        """
        抽象方法：手动保存检查点（补充PL回调的自动保存）
        Args:
            model: 待保存的模型实例
            epoch: 训练轮次
            metrics: 监控指标（如 {'test/loss': 0.12}）
            filename: 自定义文件名（None则自动生成）
        Returns:
            saved_path: 保存的检查点路径
        """
        pass

    def clean_old_checkpoints(self, keep_top_k: int = 1) -> None:
        """抽象方法：清理过期检查点，只保留Top-K"""
        pass

    def list_checkpoints(self) -> list[str]:
        """抽象方法：列出所有检查点文件（按时间排序）"""
        pass

    @property
    def latest_checkpoint_path(self) -> str | None:
        """抽象属性：获取最新检查点路径"""
        pass

# pl_checkpoint_manager.py
class PLCheckpointManager(BaseCheckpointManager):
    """
    适配PyTorch Lightning的检查点管理器：实现BaseCheckpointManager的抽象方法
    封装ModelCheckpoint回调，统一处理加载/保存/清理逻辑
    """
    def __init__(
        self,
        ckpt_root: str = './checkpoints/',
        exp_name: str = 'default_exp',
        monitor_metric: str = 'test/loss',
        monitor_mode: str = 'min',
        save_top_k: int = 1,
        save_last: bool = True
    ):
        """
        初始化PL检查点管理器
        Args:
            ckpt_root: 检查点根目录
            exp_name: 实验名称
            monitor_metric: 监控的指标（如 'test/loss'、'val/acc'）
            monitor_mode: 监控模式（min: 指标越小越好，max: 越大越好）
            save_top_k: 保留最优检查点数量
            save_last: 是否保存最后一个检查点
        """
        super().__init__(ckpt_root, exp_name)
        self.monitor_metric = monitor_metric
        self.monitor_mode = monitor_mode
        self.save_top_k = save_top_k
        self.save_last = save_last
        # 预初始化回调（延迟创建，避免重复实例化）
        self._checkpoint_callback = None

    def get_ckpt_callback(self) -> ModelCheckpoint:
        """实现抽象方法：封装ModelCheckpoint回调，统一保存逻辑"""
        if self._checkpoint_callback is None:
            self._checkpoint_callback = ModelCheckpoint(
                dirpath=self.ckpt_dir,
                filename=f'model-{{epoch:02d}}-{{{self.monitor_metric}:.4f}}',
                save_top_k=self.save_top_k,
                save_last=self.save_last,
                monitor=self.monitor_metric,
                mode=self.monitor_mode,
                auto_insert_metric_name=False  # 避免指标名重复拼接
            )
        return self._checkpoint_callback


    def save_checkpoint(self, model: torch.nn.Module, epoch: int, metrics: dict, filename: str = None) -> str:
        """
        实现抽象方法：手动保存检查点（补充PL自动保存）
        Args:
            model: 待保存模型
            epoch: 训练轮次
            metrics: 监控指标字典（如 {'test/loss': 0.12}）
            filename: 自定义文件名（None则自动生成）
        """
        # 生成文件名
        if filename is None:
            metric_value = metrics.get(self.monitor_metric, 0.0)
            filename = f'model-{epoch:02d}-{self.monitor_metric.replace("/", "_")}-{metric_value:.4f}.ckpt'
        save_path = os.path.join(self.ckpt_dir, filename)

        # 构建PL格式的检查点（兼容PL加载逻辑）
        checkpoint = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'metrics': metrics,
            'hyper_parameters': getattr(model, 'hparams', {})  # 保存超参数
        }

        # 保存文件
        torch.save(checkpoint, save_path)
        # 保存后清理过期检查点
        self.clean_old_checkpoints(self.save_top_k)
        return save_path

    def clean_old_checkpoints(self, keep_top_k: int = 1) -> None:
        """实现抽象方法：清理过期检查点，保留Top-K（按监控指标排序）"""
        ckpt_files = self.list_checkpoints()
        if len(ckpt_files) <= keep_top_k:
            return

        # 解析每个检查点的监控指标值，按指标排序
        ckpt_with_metric = []
        for ckpt in ckpt_files:
            try:
                # 加载检查点的metrics
                checkpoint = torch.load(ckpt, map_location='cpu')
                metric_val = checkpoint['metrics'].get(self.monitor_metric, 0.0)
                ckpt_with_metric.append((ckpt, metric_val))
            except Exception:
                # 解析失败则按修改时间排序
                ckpt_with_metric.append((ckpt, os.path.getmtime(ckpt)))

        # 按监控模式排序（min: 升序，max: 降序）
        sort_key = lambda x: x[1]
        if self.monitor_mode == 'max':
            ckpt_with_metric.sort(key=sort_key, reverse=True)
        else:
            ckpt_with_metric.sort(key=sort_key)

        # 删除非Top-K的检查点
        for ckpt, _ in ckpt_with_metric[keep_top_k:]:
            os.remove(ckpt)
            print(f"🗑️ 清理过期检查点：{ckpt}")

    def list_checkpoints(self) -> list[str]:
        """实现抽象方法：列出所有ckpt文件，按修改时间升序"""
        if not os.path.exists(self.ckpt_dir):
            return []
        ckpt_files = [
            os.path.join(self.ckpt_dir, f)
            for f in os.listdir(self.ckpt_dir)
            if f.endswith('.ckpt') and not f.startswith('last.ckpt')  # last.ckpt单独保留
        ]
        # 按修改时间排序
        ckpt_files.sort(key=os.path.getmtime)
        return ckpt_files

    @property
    def latest_checkpoint_path(self) -> str | None:
        """实现抽象属性：获取最新检查点路径"""
        ckpt_files = self.list_checkpoints()
        if not ckpt_files:
            # 尝试找last.ckpt（PL自动保存的最后一个检查点）
            last_ckpt = os.path.join(self.ckpt_dir, 'last.ckpt')
            return last_ckpt if os.path.exists(last_ckpt) else None
        return ckpt_files[-1]

    def load_checkpoint(self, model: torch.nn.Module, ckpt_path: str = None, strict: bool = False) -> tuple[
        torch.nn.Module, str]:
        """
        实现抽象方法：加载检查点，统一处理键名适配
        Args:
            model: 待加载模型
            ckpt_path: 指定路径（None 则加载最新）
            strict: 是否严格匹配 state_dict
        Returns:
            model: 加载后的模型
            loaded_path: 实际加载的检查点路径
        """
        # 未指定路径则加载最新
        if ckpt_path is None:
            ckpt_path = self.latest_checkpoint_path

        if ckpt_path is None or not os.path.exists(ckpt_path):
            print(f"⚠️ 检查点不存在：{ckpt_path}")
            return model, None

        print(f"\n{'=' * 70}")
        print(f"加载检查点：{os.path.basename(ckpt_path)}")
        print(f"{'=' * 70}")

        # ========== 【方案 A】使用 PL原生的 load_from_checkpoint ==========
        # 但这需要 model 是 LightningModule 且有 hparams
        # 如果你的 model 是普通 nn.Module，用方案 B
        try:
            # 检查是否是 LightningModule
            from pytorch_lightning import LightningModule

            if isinstance(model, LightningModule):
                print("检测到 LightningModule，使用 PL原生加载...")

                # 提取模型的类
                model_class = model.__class__

                # 获取保存时的超参数（如果有）
                checkpoint = torch.load(ckpt_path, map_location='cpu')
                hparams = checkpoint.get('hyper_parameters', {})

                # 合并当前模型的 hparams
                if hasattr(model, 'hparams'):
                    hparams.update(dict(model.hparams))

                # 使用 PL原生方法加载
                loaded_model = model_class.load_from_checkpoint(
                    checkpoint_path=ckpt_path,
                    strict=strict,
                    map_location=next(model.parameters()).device,
                    **hparams
                )

                print(f"✅ 使用 PL原生方式成功加载")
                print(f"   epoch: {checkpoint.get('epoch', 'N/A')}")
                print(f"   metrics: {checkpoint.get('metrics', 'N/A')}")
                print(f"{'=' * 70}\n")

                return loaded_model, ckpt_path

        except Exception as e:
            print(f"⚠️ PL原生加载失败：{str(e)}")
            print(f"   回退到手动加载方式...\n")

        # ========== 【方案 B】手动加载 state_dict ==========
        print("使用手动方式加载 state_dict...")

        # 加载检查点文件
        checkpoint = torch.load(ckpt_path, map_location=next(model.parameters()).device)
        state_dict = checkpoint.get('state_dict', checkpoint)

        print(f"   检查点 epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"   state_dict 键数量：{len(state_dict)}")

        # 统一处理 state_dict 键名适配
        new_state_dict = self._adapt_state_dict_keys(state_dict, model)

        print(f"   适配后键数量：{len(new_state_dict)}")

        # 统计匹配和不匹配的键
        model_keys = set(model.state_dict().keys())
        ckpt_keys = set(new_state_dict.keys())

        missing_keys = model_keys - ckpt_keys
        unexpected_keys = ckpt_keys - model_keys

        if missing_keys:
            print(f"\n⚠️ 缺少的键（模型有但检查点没有）: {len(missing_keys)}")
            for key in list(missing_keys)[:3]:
                print(f"   - {key}")

        if unexpected_keys:
            print(f"\n⚠️ 意外的键（检查点有但模型没有）: {len(unexpected_keys)}")
            for key in list(unexpected_keys)[:3]:
                print(f"   - {key}")

        # 加载权重
        model.load_state_dict(new_state_dict, strict=strict)

        print(f"\n✅ 加载完成")
        print(f"{'=' * 70}\n")

        return model, ckpt_path

    # ... existing code ...

    def _adapt_state_dict_keys(self, state_dict: dict, model: torch.nn.Module) -> dict:
        """
        内部方法：统一适配 state_dict 键名，解决 DataParallel/module 前缀问题

        Args:
            state_dict: 加载的权重字典
            model: 当前模型实例
        """
        new_state_dict = {}
        model_prefix = ''

        # 检查模型是否被 DataParallel/DistributedDataParallel 包装
        if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            model_prefix = 'module.'

        for key, value in state_dict.items():
            # 移除多余的前缀（如 'module.'/'model.'）
            clean_key = key.replace('module.', '').replace('model.', '')

            # 现在：总是添加，让 load_state_dict 处理
            new_key = model_prefix + clean_key if model_prefix else clean_key

            # 如果键存在，直接添加
            if new_key in model.state_dict():
                new_state_dict[new_key] = value
            else:
                # 尝试不带前缀的版本
                clean_key_no_prefix = clean_key.replace('module.', '')
                if clean_key_no_prefix in model.state_dict():
                    new_state_dict[clean_key_no_prefix] = value
                else:
                    # 仍然不匹配，打印警告
                    print(f"⚠️ 跳过不匹配的键：{key} → {new_key}")

        return new_state_dict

    # def load_checkpoint(self, model: torch.nn.Module, ckpt_path: str = None, strict: bool = False) -> tuple[torch.nn.Module, str]:
    #     """
    #     实现抽象方法：加载检查点，统一处理键名适配
    #     Args:
    #         model: 待加载模型
    #         ckpt_path: 指定路径（None则加载最新）
    #         strict: 是否严格匹配state_dict
    #     """
    #     # 未指定路径则加载最新
    #     if ckpt_path is None:
    #         ckpt_path = self.latest_checkpoint_path
    #     if ckpt_path is None or not os.path.exists(ckpt_path):
    #         return model, None
    #
    #     # 加载检查点文件
    #     checkpoint = torch.load(ckpt_path, map_location=next(model.parameters()).device)
    #     state_dict = checkpoint.get('state_dict', checkpoint)  # 兼容PL和原生torch保存格式
    #
    #     # 统一处理state_dict键名适配
    #     new_state_dict = self._adapt_state_dict_keys(state_dict, model)
    #
    #     # 加载权重
    #     model.load_state_dict(new_state_dict, strict=strict)
    #     return model, ckpt_path
    #
    # def _adapt_state_dict_keys(self, state_dict: dict, model: torch.nn.Module) -> dict:
    #     """
    #     内部方法：统一适配 state_dict 键名，解决 DataParallel/module 前缀问题
    #     Args:
    #         state_dict: 加载的权重字典
    #         model: 当前模型实例
    #     """
    #     new_state_dict = {}
    #     model_prefix = ''
    #
    #     # 检查模型是否被 DataParallel/DistributedDataParallel 包装
    #     if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
    #         model_prefix = 'module.'
    #
    #     for key, value in state_dict.items():
    #         # 移除多余的前缀（如 'module.'/'model.'）
    #         clean_key = key.replace('module.', '').replace('model.', '')
    #
    #         # 现在：总是添加，让 load_state_dict 处理
    #         new_key = model_prefix + clean_key if model_prefix else clean_key
    #
    #         # 如果键存在，直接添加
    #         if new_key in model.state_dict():
    #             new_state_dict[new_key] = value
    #         else:
    #             # 尝试不带前缀的版本
    #             clean_key_no_prefix = clean_key.replace('module.', '')
    #             if clean_key_no_prefix in model.state_dict():
    #                 new_state_dict[clean_key_no_prefix] = value
    #             else:
    #                 # 仍然不匹配，打印警告
    #                 print(f"⚠️ 跳过不匹配的键：{key} → {new_key}")
    #
    #     return new_state_dict

