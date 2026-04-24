import torch.nn as nn
import torch.nn.functional as F
from model2._TSegNet.tsegnet_utils import PointNetSetAbstractionMsg, PointNetFeaturePropagation
import torch


# from external_libs.pointnet2_utils.pointnet2_utils import PointNetSetAbstractionMsg,PointNetFeaturePropagation


def check_nan(tensor, name=""):
    """检查张量是否包含 NaN/Inf"""
    nan_count = torch.isnan(tensor).sum().item()
    inf_count = torch.isinf(tensor).sum().item()

    if nan_count > 0 or inf_count > 0:
        print(f"❌ {name}: NaN={nan_count}, Inf={inf_count}, shape={tensor.shape}")
        return True
    return False

class TSegNet(nn.Module):
    def __init__(self, num_classes=15):
        super(TSegNet, self).__init__()
        self.sa1 = PointNetSetAbstractionMsg(2048, [0.07, 0.15], [16, 32], 6 + 3, [[9, 32, 32], [9, 32, 32]])
        self.sa2 = PointNetSetAbstractionMsg(1024, [0.15, 0.3], [16, 32], 64 + 3, [[64, 64, 128], [64, 64, 128]])
        self.sa3 = PointNetSetAbstractionMsg(256, [0.3, 0.6], [16, 32], 256 + 3, [[256, 196, 256], [256, 196, 256]])
        # self.sa1 = PointNetSetAbstractionMsg(1024, [2.5, 5], [16, 32], 6 + 3, [[9, 32, 32], [9, 32, 32]])
        # self.sa2 = PointNetSetAbstractionMsg(512, [5, 10], [16, 32], 64 + 3, [[64, 64, 128], [64, 64, 128]])
        # self.sa3 = PointNetSetAbstractionMsg(256, [10, 20], [16, 32], 256 + 3, [[256, 196, 256], [256, 196, 256]])


        self.fp3 = PointNetFeaturePropagation(768, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [128, 128])
        self.fp1 = PointNetFeaturePropagation(128 + 6, [64, 64])
        self.conv1 = nn.Conv1d(64, 36, 1)
        self.bn1 = nn.BatchNorm1d(36)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(36, num_classes, 1)

        # 边界分割头
        self.edge_seg_head = nn.Conv1d(36, 2, 1)

        self.displacement = nn.Sequential(
            nn.Conv1d(515, 256, (1,)),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Conv1d(256, 64, (1,)),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Conv1d(64, 4, (1,)),
            nn.BatchNorm1d(4),
            nn.ReLU(True),
        )

        # 最多16颗牙齿
        num_centroids = 16

        # 牙齿中心点预测
        self.centroid = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1792, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(64, num_centroids * 3),
        )

        self.num_centroids = num_centroids

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        l3 = torch.cat((l3_xyz, l3_points), 1)
        displacement = self.displacement(l3)
        # print(displacement.shape)
        distance = displacement[:, 3, :]
        offsets = displacement[:, 0:3, :]

        displacement = torch.cat((l3_xyz, displacement), 1)
        displacement = displacement.view(displacement.size(0), -1)
        centroids = self.centroid(displacement)
        centroids = centroids.view(centroids.size(0), 3, self.num_centroids)


        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)

        check_nan(l2_points, "l2_points")
        check_nan(l1_points, "l1_points")
        check_nan(l0_points, "l0_points")



        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))

        # x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

        seg_result = self.conv2(x)

        edge_seg_result = self.edge_seg_head(x)

        # return seg_result, edge_seg_result, centroids
        return {
            'seg_result': seg_result,
            'edge_seg_result': edge_seg_result,
            'centroids': centroids,
            'l3_xyz': l3_xyz,
            'offsets': offsets,
            'distance': distance
        }


if __name__ == '__main__':
    data = torch.rand((1, 6, 16000))
    model = TSegNet(num_classes=15)
    outs = model(data)
    print(outs['seg_result'].shape)
    print(outs['edge_seg_result'].shape)
    print(outs['centroids'].shape)
    print(outs['l3_xyz'].shape)
    print(outs['offsets'].shape)
    print(outs['distance'].shape)


    # -- coding: utf-8 --
    import torch
    from thop import profile

    flops, params = profile(model, (data,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f G, params: %.2f M' % (flops / 1.0e9, params / 1.0e6))