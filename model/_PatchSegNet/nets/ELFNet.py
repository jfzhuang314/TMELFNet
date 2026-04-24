import torch
from torch import nn
from torch.nn import ReLU
from torch_cluster import radius_graph
from torch_geometric.nn import EdgeConv, GATConv
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
import torch.nn.functional as F

class ShallowSharedBackBone(nn.Module):
    """
    浅层特征提取网络。Shallow FA 模块，用于处理输入数据并生成输出特征。
    """
    def __init__(self, input_dim, radii=[0.03, 0.10]):
        super(ShallowSharedBackBone, self).__init__()
        self.input_dim = input_dim
        self.radii = radii

        # 定义 EdgeConv 层
        self.conv1 = EdgeConv(Seq(Lin(self.input_dim * 2, 32), ReLU()))
        self.conv2 = EdgeConv(Seq(Lin(32 * 2, 32), ReLU()))

    def forward(self, batch, edge_index_list=None):
        """
        Args:
            batch: BNC格式
            edge_index_list: 预构建的邻接图列表，如果为None则现场构建
        Returns:
            提取的特征张量
        """
        data = batch
        x0 = data.x.float()

        # 获取或构建图结构
        if edge_index_list is None:
            edge_index_list = self._build_multi_scale_graph(data.pos)
        else:
            # 验证输入的edge_index_list是否符合要求
            self._validate_edge_indices(edge_index_list, len(self.radii))

        # 逐层特征提取
        x1 = F.relu(self.conv1(x0, edge_index_list[0]))
        x2 = F.relu(self.conv2(x1, edge_index_list[1]))

        return x2

    def _build_multi_scale_graph(self, pos):
        """构建多尺度邻域图"""
        edge_indices = []
        for radius in self.radii:
            edge_index = radius_graph(pos, r=radius, max_num_neighbors=32)
            edge_indices.append(edge_index.to(pos.device, dtype=torch.long))
        return edge_indices

    def _validate_edge_indices(self, edge_index_list, expected_length):
        """验证输入的邻接图列表"""
        if len(edge_index_list) != expected_length:
            raise ValueError(
                f"Edge index list length mismatch. Expected {expected_length}, got {len(edge_index_list)}"
            )


class EdgeCNNAttenBlock(nn.Module):
    """ 经典的EdgeCNN注意力块"""
    def __init__(self, input_dim):
        super(EdgeCNNAttenBlock, self).__init__()
        self.input_dim = input_dim

        # 特征提取层
        self.feature_layers = nn.ModuleList([
            EdgeConv(Seq(Lin(input_dim * 2, 32), ReLU())),
            EdgeConv(Seq(Lin(32 * 2, 64), ReLU())),
            EdgeConv(Seq(Lin(64 * 2, 128), ReLU())),
            EdgeConv(Seq(Lin(128 * 2, 128), ReLU()))
        ])

        # 注意力机制层
        self.attention_layers = nn.ModuleList([
            GATConv(32, 8, heads=4, dropout=0.6),
            GATConv(64, 16, heads=4, dropout=0.6),
            GATConv(128, 32, heads=4, dropout=0.6),
            GATConv(128, 32, heads=4, dropout=0.6)
        ])

        # 特征融合MLP
        self.feature_fusion_mlp = nn.Sequential(
            nn.Linear(128 + 128 + 64 + 32, 128),
            nn.ReLU(),
            nn.Linear(128, 256)
        )

        # 最终特征投影
        self.final_projection = nn.Sequential(
            nn.Linear(128 + 128 + 64 + 32 + 256, 128),
            nn.ReLU(),
            nn.Linear(128, 32)
        )

    def forward(self, x, edge_index_list):
        # 多层特征提取与注意力增强
        features = []
        current_x = x

        for i, (conv_layer, attn_layer) in enumerate(zip(self.feature_layers, self.attention_layers)):
            # EdgeConv特征提取
            conv_features = F.relu(conv_layer(current_x, edge_index_list[i]))
            # 注意力增强
            attn_features = attn_layer(conv_features, edge_index_list[i]) + conv_features
            features.append(attn_features)
            current_x = conv_features

        # 特征融合
        concatenated_features = torch.cat(features, dim=-1)
        fused_features = self.feature_fusion_mlp(concatenated_features)

        # 全局池化增强
        pooled_features = self._global_pooling(fused_features, x.size(0))

        # 最终特征组合
        final_input = torch.cat([concatenated_features, pooled_features], dim=-1)
        x_F1 = F.relu(self.final_projection[0](final_input))  # 128维
        x_F2 = self.final_projection[2](x_F1)                 # 32维

        return features[0], x_F1, x_F2, fused_features

    def _global_pooling(self, features, batch_size):
        """全局平均池化并广播到批次维度"""
        pooled = torch.mean(features, dim=0, keepdim=True)
        return pooled.expand(batch_size, -1)

class TGMaskBranch(torch.nn.Module):
    """
    牙齿-背景二分类分支网络
    """
    def __init__(self, num_classes=15, num_feat=35, k=16):
        super().__init__()
        self.input_dim = num_feat
        self.num_classes = num_classes
        self.k = k

        # 特征处理模块
        self.feature_block = EdgeCNNAttenBlock(self.input_dim)
        self.refinement_block = EdgeCNNAttenBlock(self.input_dim + 2)

        # 预测头
        self.prediction_heads = nn.ModuleDict({
            'mask_1': nn.Sequential(
                nn.Linear(32, 2),
                nn.Softmax(dim=1)
            ),
            'weight_1': nn.Linear(32, 1),
            'mask_2': nn.Linear(32, 1)
        })

    def forward(self, batch, edge_index_list):
        points = batch.float()
        batch_size, channels, num_points = points.shape

        # 展平处理
        flat_points = self._flatten_points(points)

        # 第一阶段特征提取
        x1, x1_F1, x1_F2, _ = self.feature_block(flat_points, edge_index_list)

        # 第一阶段预测
        pd_1 = self.prediction_heads['mask_1'](x1_F2)
        weight_1 = self.prediction_heads['weight_1'](x1_F2)

        # 特征增强
        enhanced_points = torch.cat([flat_points, pd_1], dim=1)

        # 第二阶段特征提取与预测
        _, _, x2_F2, _ = self.refinement_block(enhanced_points, edge_index_list)
        pd_2 = self.prediction_heads['mask_2'](x2_F2)

        # 恢复批次维度
        pd_1, weight_1, pd_2 = self._restore_batch_dimensions(
            [pd_1, weight_1, pd_2], batch_size, num_points
        )

        return pd_1, weight_1, pd_2

    def _flatten_points(self, points):
        """将点云数据展平为适合图神经网络的格式"""
        return points.transpose(1, 2).reshape(-1, points.size(1))

    def _restore_batch_dimensions(self, tensors, batch_size, num_points):
        """恢复张量的批次维度"""
        restored = []
        for tensor in tensors:
            restored_tensor = tensor.reshape(batch_size, num_points, -1).transpose(1, 2)
            restored.append(restored_tensor)
        return restored


class EdgeMaskBranches(torch.nn.Module):
    """
    边缘检测双分支网络（TAE和TTE）
    """
    def __init__(self, num_classes=15, num_feat=35, k=16):
        super().__init__()
        self.input_dim = num_feat
        self.num_classes = num_classes
        self.k = k

        # 共享特征提取模块
        self.shared_feature_block = EdgeCNNAttenBlock(self.input_dim)

        # 分支专用模块
        self.tae_refinement = EdgeCNNAttenBlock(self.input_dim + 2)
        self.tte_refinement = EdgeCNNAttenBlock(self.input_dim + 2)

        # 分支预测头
        self.branches = nn.ModuleDict({
            'tae': self._create_prediction_branch(),
            'tte': self._create_prediction_branch()
        })

    def _create_prediction_branch(self):
        """创建单个分支的预测头"""
        return nn.ModuleDict({
            'mask_1': nn.Sequential(
                nn.Linear(32, 2),
                nn.Softmax(dim=1)
            ),
            'weight_1': nn.Linear(32, 1),
            'mask_2': nn.Linear(32, 1)
        })

    def forward(self, batch, edge_index_list):
        points = batch.float()
        batch_size, channels, num_points = points.shape

        # 展平点云数据
        flat_points = self._flatten_points(points)

        # 共享特征提取
        x1, x1_F1, x1_F2, _ = self.shared_feature_block(flat_points, edge_index_list)

        # 分别处理两个分支
        tae_output = self._process_branch(
            self.branches['tae'], self.tae_refinement,
            flat_points, x1_F2, edge_index_list, batch_size, num_points
        )

        tte_output = self._process_branch(
            self.branches['tte'], self.tte_refinement,
            flat_points, x1_F2, edge_index_list, batch_size, num_points
        )

        return tae_output, tte_output

    def _process_branch(self, branch_head, refinement_block,
                       flat_points, shared_features, edge_index_list,
                       batch_size, num_points):
        """处理单个分支的前向传播"""
        # 第一阶段预测
        pd_1 = branch_head['mask_1'](shared_features)
        weight_1 = branch_head['weight_1'](shared_features)

        # 特征增强与第二阶段处理
        enhanced_points = torch.cat([flat_points, pd_1], dim=1)
        _, _, refined_features, _ = refinement_block(enhanced_points, edge_index_list)

        # 第二阶段预测
        pd_2 = branch_head['mask_2'](refined_features)

        # 恢复批次维度
        pd_1, weight_1, pd_2 = self._restore_batch_dimensions(
            [pd_1, weight_1, pd_2], batch_size, num_points
        )

        return pd_1, weight_1, pd_2

    def _flatten_points(self, points):
        return points.transpose(1, 2).reshape(-1, points.size(1))

    def _restore_batch_dimensions(self, tensors, batch_size, num_points):
        restored = []
        for tensor in tensors:
            restored_tensor = tensor.reshape(batch_size, num_points, -1).transpose(1, 2)
            restored.append(restored_tensor)
        return restored


class ELFNet(torch.nn.Module):
    """
    ELFNet主网络：整合牙齿-背景分割和边缘检测
    """

    def __init__(self, num_classes=15, num_feat=6, k=16):
        super().__init__()
        self.input_dim = num_feat
        self.output_dim = num_classes
        self.k = k

        # 网络组件
        self.shared_backbone = ShallowSharedBackBone(input_dim=num_feat)
        self.bg_branch = TGMaskBranch(num_feat=3 + 32, num_classes=num_classes, k=k)
        self.edge_branches = EdgeMaskBranches(num_feat=3 + 32, num_classes=num_classes, k=k)

        # 图构建参数
        self.graph_radii = [0.03, 0.07, 0.15, 0.4]
        self.backbone_radii = [0.03, 0.10]  # ShallowSharedBone专用的半径

    def forward(self, batch):
        # 图结构构建（统一管理）
        edge_index_list = self._build_graph_structure(batch)
        backbone_edge_indices = [edge_index_list[0], edge_index_list[1]]  # 使用前两个尺度

        # 特征提取
        pos = batch.pos.float()
        x = batch.x.float()
        shared_features = F.relu(self.shared_backbone(batch, backbone_edge_indices))

        # 特征融合
        combined_features = self._combine_position_and_features(pos, shared_features)

        # 分支预测
        bg_outputs = self.bg_branch(combined_features, edge_index_list)
        tae_outputs, tte_outputs = self.edge_branches(combined_features, edge_index_list)

        return bg_outputs, tae_outputs, tte_outputs, shared_features

    def _build_graph_structure(self, batch):
        """构建多尺度图结构"""
        if hasattr(batch, 'radii_edge_index') and batch.radii_edge_index:
            return list(batch.radii_edge_index.values())

        # 提取坐标信息
        pos = batch.pos.float()
        batch_size = 1  # 假设batch size为1，可以根据实际调整
        num_points = pos.shape[0]

        # 数据展平（如果是批次数据需要相应调整）
        if pos.dim() == 3:  # [B, N, 3]
            xyz_flat = pos.reshape(-1, 3)
            batch_indices = torch.arange(batch_size, device=pos.device).repeat_interleave(num_points)
        else:  # [N, 3]
            xyz_flat = pos
            batch_indices = None

        # 构建多尺度邻接图
        edge_index_list = []
        for radius in self.graph_radii:
            if batch_indices is not None:
                edge_index = radius_graph(
                    xyz_flat, r=radius, batch=batch_indices, max_num_neighbors=24
                )
            else:
                edge_index = radius_graph(xyz_flat, r=radius, max_num_neighbors=24)
            edge_index_list.append(edge_index.to(pos.device, dtype=torch.long))

        return edge_index_list

    def _combine_position_and_features(self, pos, features):
        """组合位置信息和提取的特征"""
        combined = torch.cat((pos, features), dim=1).unsqueeze(dim=0).transpose(1, 2)
        return combined.float()

