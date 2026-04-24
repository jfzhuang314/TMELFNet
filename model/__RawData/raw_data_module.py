import os
from datetime import datetime
from pathlib import Path

import torch
from hydra import initialize_config_dir, compose
from omegaconf import DictConfig
import torch.nn as nn
import pytorch_lightning as pl
from torch_geometric.data import Data, DataLoader
from common.BaseTrainer import BaseTrainer
from configs.hydra_yaml_config import instantiate_from_yaml_config, print_yaml_config, instantiate_from_config, \
    get_yaml_config, print_yaml_cfg_dict

from datasets import T3DSPlusSegDataset
from datasets import TADPMSegDataset
from datasets.T3DSPlusSegDataset import T3DSPlusSegReader
from evaluation.eval_segnets import save_evaluation_results_to_excel, save_batch_info_to_json, batch_insseg_pred
from evaluation.seg_evaluation import calc_bnd_bg_merged_iou, cal_multi_cls_metric

from metrics.dice_loss import DiceLoss
from model._DilatedToothSegNet.DTSegNet import DTSegNet, BADTSegNet
from thop import profile
from metrics.CrossToothSegMetricsManager import CrossToothSegMetricsManager as SegMetricsManager, \
    DynamicRadiusNNEdgeMaskWithPos
from model._DilatedToothSegNet.cbl import CBLLoss
from external_libs import gen_utils as gu

class RawDataModule(pl.LightningModule):
    """ 牙齿分割网络训练模块 - 采用TripleBinClsModule的逻辑结构重构"""
    def __init__(self):
        super().__init__()
    def predict_step_forward(self, batch):

        return  batch


class RawDataTrainer(BaseTrainer):
    """RawData训练器，继承自BaseTrainer"""
    def __init__(self, config_path="../../configs", config_name="Module_RawData"):
        super().__init__(config_path, config_name)

    # 获得模型实例
    def build_module(self, model_obj: DictConfig, ckpt_path=None):
        """获取RawData模型实例"""
        # 如果指定了检查点路径，则从检查点中加载模型
        if ckpt_path is not None and os.path.exists(ckpt_path):
            try:
                print(f"\n开始加载检查点...")

                module = RawDataModule.load_from_checkpoint(
                    checkpoint_path=ckpt_path,
                    model_cfg=model_obj,  # 传递配置
                    num_classes=15,
                    strict=False,
                    map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                )
                print(f"✅ 成功加载检查点{ckpt_path}")

                return module
            except Exception as e:
                print(f"\n❌ 未能加载检查点{ckpt_path}: {e}")

        # 否则，创建新的模型实例
        name = model_obj["name"]
        if name == "RawDataModule":
            module = RawDataModule().to(device)
        else:
            raise ValueError(f"Invalid model name: {name}")

        return module

    # 准备数据
    def prepare_data(self, cfg_obj: DictConfig):
        if cfg_obj.dataset.name == "T3DSPlusSegDataset":
            _, self.train_loader, self.test_loader \
                = T3DSPlusSegDataset.get_dataloaders_from_cfg_obj(cfg_obj, only_val=True)
        elif cfg_obj.dataset.name == "TADPMSegDataset":
            _, self.train_loader, self.test_loader \
                = TADPMSegDataset.get_dataloaders_from_cfg_obj(cfg_obj, only_val=True)

############################################################################
## Training Testing Prediction
############################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config_path = "../../configs"
config_name = "Module_RawData"

def do_pred():

    # 1. 创建 trainer
    trainer = RawDataTrainer(config_path=config_path, config_name=config_name)
    cfg_obj = trainer.cfg_obj

    # 2. 获取模型实例
    ds_name = cfg_obj.dataset.name
    model_path = cfg_obj.prediction.model_path
    model_name = cfg_obj.prediction.model_name
    ckpt_path = os.path.join(model_path, ds_name, model_name)

    # 更改数据集的采样比例
    cfg_obj.dataset.sample_ratio = 1.0
    # 3. 加载模型
    trainer.prepare_trainer(ckpt_path) if not trainer.trainer else None

    # 4. 获取模型实例
    module = trainer.module.to(device) # 获取模型实例

    # 5. 测试集数据加载
    test_loader = trainer.test_loader

    # 6. 预测并保存结果
    # batch_insseg_pred(ckpt_path, module, test_loader, save_json=True)


def batch_pred(ckpt_path, module, test_loader, save_json=True):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_save_path = Path(ckpt_path).parent
    last_folder_name = Path(base_save_path).name
    # excel 保存目录
    excel_save_path = base_save_path / f"{last_folder_name}_eval_{timestamp}.xlsx"
    # JSON 保存目录
    json_save_dir = base_save_path / f"batch_json_{timestamp}"
    print(f"\n📂 JSON 保存目录：{json_save_dir}")
    # 存储所有样本的结果
    all_results = []
    # 遍历测试集
    for batch_idx, batch in enumerate(test_loader):
        print(f"评估样本 {batch_idx + 1}/{len(test_loader)}: {batch.filename}")

        outputs = module.predict_step_forward(batch)

        # 保存 batch 信息到 JSON
        if save_json:
            try:
                json_path = save_batch_info_to_json(batch, outputs, json_save_dir)
                print(f"✅ JSON 已保存：{os.path.basename(json_path)}")
            except Exception as e:
                print(f"❌ JSON 保存失败：{e}")

def main():

    # 打印配置信息
    print_yaml_config(config_path="../../configs", config_name="Module_RawData")
    cfg_obj = instantiate_from_yaml_config(config_path="../../configs", config_name="Module_RawData")
    print(cfg_obj)


    do_pred()



if __name__ == '__main__':
    main()
