import torch
import torch.nn as nn
import yaml
from omegaconf import DictConfig, OmegaConf
from thop import profile

from configs.hydra_yaml_config import print_yaml_config, instantiate_from_yaml_config
from model._DilatedToothSegNet.layer import BasicPointLayer, EdgeGraphConvBlock, \
    DilatedEdgeGraphConvBlock, ResidualBasicPointLayer, PointFeatureImportance, STNkd


class DTSegNet(nn.Module):
    """基于YAML配置的膨胀牙齿分割网络"""

    def __init__(self, model_cfg: DictConfig):
        """
        初始化网络
        Args:
            model_cfg: 模型配置字典
        """
        super(DTSegNet, self).__init__()

        # 加载配置
        self.cfg = model_cfg

        # 提取基础参数
        self.num_classes = model_cfg.num_classes
        self.feature_dim = model_cfg.feature_dim
        # self.add_cbl = model_cfg.add_cbl

        # 初始化网络组件
        self._build_network(model_cfg)

    def _build_network(self, model_cfg):
        """根据配置构建网络"""
        ###########################################################################
        # DTSegNet.yaml 配置参数说明
        ###########################################################################
        # 基础参数:
        #   num_classes: 15           # 分类数量（牙齿类别数）
        #   feature_dim: 6            # 输入特征维度（xyz + 法向量）
        #   dilation_rates: [200, 900, 2700]  # 膨胀卷积的感受野大小

        # 网络架构参数:
        #   edge_conv_blocks: 3层     # 局部特征提取
        #   dilated_conv_blocks: 3层  # 多尺度上下文特征提取
        #   residual_blocks: 2层      # 特征优化
        #   hidden_layers: 2层        # 特征压缩层

        # 输出配置:
        #   semantic_segmentation: 15类  # 语义分割头
        #   edge_segmentation: 2类       # 边缘检测头
        ###########################################################################

        # 1. 空间变换网络
        self.stnkd = STNkd(k=self.feature_dim)

        # 2. 边缘图卷积块
        self.edge_conv_blocks = nn.ModuleList()
        for block_cfg in model_cfg.edge_conv_blocks:
            block = EdgeGraphConvBlock(
                in_channels=block_cfg.in_channels,
                out_channels=block_cfg.out_channels,
                k=block_cfg.kernel_size,
                hidden_channels=block_cfg.hidden_channels,
                edge_function=block_cfg.edge_function
            )
            self.edge_conv_blocks.append(block)

        # 3. 局部特征压缩层
        local_cfg = model_cfg.hidden_layers.local_feature_compression
        self.local_hidden_layer = BasicPointLayer(
            in_channels=local_cfg.in_channels,
            out_channels=local_cfg.out_channels
        )

        # 4. 膨胀边缘图卷积块
        self.dilated_conv_blocks = nn.ModuleList()
        for block_cfg in model_cfg.dilated_conv_blocks:
            block = DilatedEdgeGraphConvBlock(
                in_channels=block_cfg.in_channels,
                hidden_channels=block_cfg.hidden_channels,
                out_channels=block_cfg.out_channels,
                k=block_cfg.kernel_size,
                dilation_k=block_cfg.dilation_rate,
                edge_function=block_cfg.edge_function
            )
            self.dilated_conv_blocks.append(block)

        # 5. 全局特征压缩层
        global_cfg = model_cfg.hidden_layers.global_feature_compression
        self.global_hidden_layer = BasicPointLayer(
            in_channels=global_cfg.in_channels,
            out_channels=global_cfg.out_channels
        )

        # 6. 特征重要性注意力
        # arch_cfg = self.config.architecture
        self.feature_importance = PointFeatureImportance(
            in_channels=model_cfg.feature_attention.in_channels
        )

        # 7. 残差块
        self.residual_blocks = nn.ModuleList()
        for block_cfg in model_cfg.residual_blocks:
            block = ResidualBasicPointLayer(
                in_channels=block_cfg.in_channels,
                hidden_channels=block_cfg.hidden_channels,
                out_channels=block_cfg.out_channels
            )
            self.residual_blocks.append(block)

        # cbl 特征压缩层
        cbl_cfg = model_cfg.hidden_layers.cbl_feature_compression
        self.cbl_hidden_layer = BasicPointLayer(
            in_channels=cbl_cfg.in_channels,
            out_channels=cbl_cfg.out_channels
        )

        # 8. 输出头
        output_cfg = model_cfg.output_heads
        self.semantic_head = BasicPointLayer(
            in_channels=output_cfg.semantic_segmentation.in_channels,
            out_channels=output_cfg.semantic_segmentation.out_channels,
            is_out=True
        )

        self.edge_head = BasicPointLayer(
            in_channels=output_cfg.edge_segmentation.in_channels,
            out_channels=output_cfg.edge_segmentation.out_channels,
            is_out=True
        )

        self.tte_seg_head = BasicPointLayer(
            in_channels=output_cfg.tte_segmentation.in_channels,
            out_channels=output_cfg.tte_segmentation.out_channels,
            is_out=True
        )

        # self.tg_seg_head = BasicPointLayer(
        #     in_channels=output_cfg.tg_segmentation.in_channels,
        #     out_channels=output_cfg.tg_segmentation.out_channels,
        #     is_out=True
        # )

    def forward(self, data):
        """
        前向传播
        Args:
            data: 输入数据，形状为 (B, C, N)
        Returns:
            tuple: (语义分割结果, 边缘分割结果)
        """
        # 数据预处理
        batch_size = data.shape[0]
        point_size = data.shape[2]

        data = data.permute(0, 2, 1)  # BCN -> BNC
        x = data
        pos = x[:, :, :3]  # 提取坐标信息

        # 计算点对距离（可选）
        cd = torch.cdist(pos, pos)  # 有点夸张，耗费的内存不是一般的多
        x = self.stnkd(x)
        # B, N, _ = pos.shape

        # 1. 局部特征提取
        local_features = []
        for block in self.edge_conv_blocks:
            x, _ = block(x, pos)
            local_features.append(x)

        # 特征融合和压缩
        x = torch.cat(local_features, dim=2)
        x = self.local_hidden_layer(x)

        # 2. 膨胀特征提取
        dilated_features = [x]  # 包含原始特征
        for block in self.dilated_conv_blocks:
            x, _ = block(x, pos, cd=cd)
            dilated_features.append(x)

        # 多尺度特征融合
        x = torch.cat(dilated_features, dim=2)
        x = self.global_hidden_layer(x)

        # 3. 特征优化
        x = self.feature_importance(x)
        for block in self.residual_blocks:
            x = block(x)

        # 4. 输出生成
        semantic_result = self.semantic_head(x)
        semantic_result = semantic_result.permute(0, 2, 1)  # BNC -> BCN

        edge_result = self.edge_head(x)
        edge_result = edge_result.permute(0, 2, 1)  # BNC -> BCN

        tte_seg_result = self.tte_seg_head(x)
        tte_seg_result = tte_seg_result.permute(0, 2, 1)  # BNC -> BCN

        # tg_seg_result = self.tg_seg_head(x)
        # tg_seg_result = tg_seg_result.permute(0, 2, 1)  # BNC -> BCN
        tg_seg_result = semantic_result[:,:1,:]   # B,1,N
        foreground_logits = semantic_result[:, 1:, :]  # [B, C-1, N]
        tooth_conf, _ = torch.max(foreground_logits, dim=1,keepdim= True)  # [B, N], [B, N]
        tg_seg_result = torch.cat([tg_seg_result, tooth_conf], dim=1)  # [B, C, N]

        # cbl_inputs
        offset = torch.tensor([point_size * (i + 1) for i in range(batch_size)], device=device)
        o = offset.int()
        cbl_x = self.cbl_hidden_layer(x)
        cbl_inputs = [pos.reshape(-1, 3), cbl_x.reshape(-1, cbl_x.shape[-1]), o]

        # if not self.add_cbl:
        #     return {
        #          'seg_result': semantic_result,
        #          'edge_seg_result': edge_result,
        #          'tte_seg_result': tte_seg_result
        #          }
        # else:
        return {
            'seg_result': semantic_result,
            'edge_seg_result': edge_result,
            'tte_seg_result': tte_seg_result,
            'tg_seg_result': tg_seg_result,
            'cbl_inputs': cbl_inputs
        }

    def __repr__(self):
        str_repr = ""

        for k, v in self.cfg.items():
            str_repr += f"{k}: {v}\n"

        str_repr += f"total_params: {sum(p.numel() for p in self.parameters()) / 1000:.2f}K\n"
        str_repr += f"trainable_params: {sum(p.numel() for p in self.parameters() if p.requires_grad) / 1000:.2f}K\n"

        return str_repr


class BADTSegNet(DTSegNet):
    def __init__(self, model_cfg: DictConfig):
        """
        初始化网络
        Args:
            model_cfg: 模型配置字典
        """
        super(BADTSegNet, self).__init__()

# -------------------------------测试-----------------------------------#

# 创建全局配置对象
GLOBAL_CONFIG = OmegaConf.create({
    "config_path": "../../configs/nets",
    "config_name": "DTSegNet"
})
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(config_path="../../configs/nets", config_name="DTSegNet"):
    cfg_obj = instantiate_from_yaml_config(config_path=config_path, config_name=config_name)
    model = DTSegNet(cfg_obj.model).to(device)
    return model


# 便捷函数
def test_get_model():
    model = get_model(config_path=GLOBAL_CONFIG.config_path,
                      config_name=GLOBAL_CONFIG.config_name)
    # print(model)

    data = torch.rand(16, 6, 2048).to(device)  # BCN格式

    output = model(data)
    seg_result = output['seg_result']
    edge_seg_result = output['edge_seg_result']
    tte_seg_result = output['tte_seg_result']
    cbl_inputs = output['cbl_inputs']

    print(seg_result.shape)  # torch.Size([1, 15, 24000])
    print(edge_seg_result.shape)  # torch.Size([1, 2, 24000])
    print(tte_seg_result.shape)  # torch.Size([1, 2, 24000])

    p,x,o = cbl_inputs
    print(p.shape) # [1, 16000, 3]
    print(x.shape) # [1, 16000, 256]
    print(o.shape,o) # [16000]

    flops, params = profile(model, (data,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f G, params: %.2f M' % (flops / 1.0e9, params / 1.0e6))


def main():
    print_yaml_config(config_path=GLOBAL_CONFIG.config_path, config_name=GLOBAL_CONFIG.config_name)
    cfg_obj = instantiate_from_yaml_config(config_path=GLOBAL_CONFIG.config_path,
                                           config_name=GLOBAL_CONFIG.config_name)
    print(cfg_obj)
    # 日志管理器初始化
    # test_loggerMgr(cfg_obj.logger)

    test_get_model()
    # do_train()
    # do_pred()
    # test4_cent()
    # save_topo_cluster_results()
    # test_read_json_file()


if __name__ == '__main__':
    main()
