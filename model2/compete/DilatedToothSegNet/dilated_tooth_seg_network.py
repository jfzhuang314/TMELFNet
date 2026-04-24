from torch import nn
import torch
from model2.compete.DilatedToothSegNet.layer import BasicPointLayer, EdgeGraphConvBlock, DilatedEdgeGraphConvBlock, ResidualBasicPointLayer, \
    PointFeatureImportance, STNkd


class DilatedToothSegmentationNetwork(nn.Module):
    def __init__(self, num_classes=17, feature_dim=6):
        """
        :param num_classes: Number of classes to predict
        """
        super(DilatedToothSegmentationNetwork, self).__init__()

        self.num_classes = num_classes

        self.stnkd = STNkd(k=feature_dim)

        self.edge_graph_conv_block1 = EdgeGraphConvBlock(in_channels=feature_dim, out_channels=24, k=32,
                                                         hidden_channels=24,
                                                         edge_function="local_global")
        self.edge_graph_conv_block2 = EdgeGraphConvBlock(in_channels=24, out_channels=24, k=32,
                                                         hidden_channels=24,
                                                         edge_function="local_global")
        self.edge_graph_conv_block3 = EdgeGraphConvBlock(in_channels=24, out_channels=24, k=32,
                                                         hidden_channels=24,
                                                         edge_function="local_global")

        self.local_hidden_layer = BasicPointLayer(in_channels=24 * 3, out_channels=60)

        self.dilated_edge_graph_conv_block1 = DilatedEdgeGraphConvBlock(in_channels=60, hidden_channels=60,
                                                                        out_channels=60, k=32,
                                                                        dilation_k=200, edge_function="local_global")
        self.dilated_edge_graph_conv_block2 = DilatedEdgeGraphConvBlock(in_channels=60, hidden_channels=60,
                                                                        out_channels=60, k=32,
                                                                        dilation_k=900, edge_function="local_global")
        self.dilated_edge_graph_conv_block3 = DilatedEdgeGraphConvBlock(in_channels=60, hidden_channels=60,
                                                                        out_channels=60, k=32,
                                                                        dilation_k=1800, edge_function="local_global")

        self.global_hidden_layer = BasicPointLayer(in_channels=60 * 4, out_channels=1024)

        self.feature_importance = PointFeatureImportance(in_channels=1024)

        self.res_block1 = ResidualBasicPointLayer(in_channels=1024, out_channels=512, hidden_channels=512)
        self.res_block2 = ResidualBasicPointLayer(in_channels=512, out_channels=256, hidden_channels=256)

        self.out = BasicPointLayer(in_channels=256, out_channels=num_classes, is_out=True)

        self.edge_seg_head = BasicPointLayer(in_channels=256, out_channels=2, is_out=True)

    def forward(self, data):
        data = data.permute(0, 2, 1)  # features of the points. Shape: (batch_size, num_points, feature_dim)

        x = data
        pos = x[:, :, :3]  # xyz coordinates of the points. Shape: (batch_size, num_points, 3)

        # precompute pairwise distance of points
        cd = torch.cdist(pos, pos)
        x = self.stnkd(x)

        x1, _ = self.edge_graph_conv_block1(x, pos)
        x2, _ = self.edge_graph_conv_block2(x1)
        x3, _ = self.edge_graph_conv_block3(x2)

        x = torch.cat([x1, x2, x3], dim=2)
        x = self.local_hidden_layer(x)

        x1, _ = self.dilated_edge_graph_conv_block1(x, pos, cd=cd)
        x2, _ = self.dilated_edge_graph_conv_block2(x1, pos, cd=cd)
        x3, _ = self.dilated_edge_graph_conv_block3(x2, pos, cd=cd)

        x = torch.cat([x, x1, x2, x3], dim=2)
        x = self.global_hidden_layer(x)

        x = self.feature_importance(x)

        x = self.res_block1(x)
        x = self.res_block2(x)

        seg_result = self.out(x)
        seg_result = seg_result.permute(0, 2, 1)

        edge_seg_result = self.edge_seg_head(x)
        edge_seg_result = edge_seg_result.permute(0, 2, 1)


        return seg_result, edge_seg_result


if __name__ == '__main__':
    # Create the model
    model = DilatedToothSegmentationNetwork(num_classes=15, feature_dim=6).cuda()
    # dummy input
    data = torch.rand(1, 6, 16000).cuda()

    seg_result, edge_seg_result = model(data)
    print(seg_result.shape)
    print(edge_seg_result.shape)

    # -- coding: utf-8 --
    import torch
    from thop import profile

    flops, params = profile(model, (data,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f G, params: %.2f M' % (flops / 1.0e9, params / 1.0e6))