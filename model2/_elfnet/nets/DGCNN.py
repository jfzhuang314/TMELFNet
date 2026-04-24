
import torch
from torch import nn
from torch.nn import ReLU
from torch_cluster import radius_graph
from torch_geometric.nn import EdgeConv, GATConv
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
import torch.nn.functional as F


class ResidualBasicPointLayer(nn.Module):
    """
    Basic point layer consisting of a 1D convolution, batch normalization, leaky ReLU, and dropout
    with a residual connection.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.1):
        """
        Initializes the BasicPointLayer.
        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param dropout: Dropout probability. Default is 0.1.
        """
        super(ResidualBasicPointLayer, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_channels, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout)
        )
        if in_channels != out_channels:
            self.rescale = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels, track_running_stats=False),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Dropout(dropout)
            )
        else:
            self.rescale = nn.Identity()

    def forward(self, x):
        x = x.transpose(2, 1)
        return self.conv(x).transpose(2, 1) + self.rescale(x).transpose(2, 1)


class PointFeatureImportance(nn.Module):
    """
    Point Feature Importance module.
    """

    def __init__(self, in_channels):
        """
        Initializes the PointFeatureImportance module.
        :param in_channels: Number of input channels.
        """
        super(PointFeatureImportance, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(in_channels, track_running_stats=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        weight = self.conv(x.transpose(2, 1))
        return x * weight.transpose(2, 1)


class DGCNN(nn.Module):
    """ 经典的EdgeCNN注意力块"""
    def __init__(self, input_dim):
        super(DGCNN, self).__init__()
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
            nn.Linear(128, 512)
        )

        # 最终特征投影
        self.cbl_feature = nn.Sequential(
            nn.Linear(128 + 128 + 64 + 32 + 512, 128),
            nn.ReLU(),
            nn.Linear(128, 32)
        )

        # branches
        # 8. 输出头
        self.semantic_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_classes)
        )

        self.bg_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

        self.edge_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

        self.tte_seg_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
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