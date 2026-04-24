import torch.nn as nn
import torch.nn.functional as F
# from model._TSegNet.tsegnet_utils import PointNetSetAbstraction, PointNetSetAbstractionMsg, get_centroids, \
#     PointNetFeaturePropagation, get_proposal
from external_libs.pointnet2_utils.pointnet2_utils import PointNetSetAbstractionMsg,PointNetFeaturePropagation

import torch


# class TSegCentroidNet(nn.Module):
#     def __init__(self, num_classes=15):
#         super(TSegCentroidNet, self).__init__()
#
#         self.num_classes = 15
#         input_feauture_num = 6
#
#         self.sa1 = PointNetSetAbstractionMsg(2048, [0.06, 0.12], [16, 32], 6 + 3, [[9, 32, 32], [9, 32, 32]])
#         self.sa2 = PointNetSetAbstractionMsg(1024, [0.12, 0.20], [16, 32], 64 + 3, [[64, 64, 128], [64, 64, 128]])
#         self.sa3 = PointNetSetAbstractionMsg(256, [0.20, 0.3], [16, 32], 256 + 3, [[256, 196, 256], [256, 196, 256]])
#
#         self.fp3 = PointNetFeaturePropagation(768, [256, 256])
#         self.fp2 = PointNetFeaturePropagation(320, [128, 128])
#         self.fp1 = PointNetFeaturePropagation(128 + 6, [64, 64])
#         self.conv1 = nn.Conv1d(64, 36, 1)
#         self.bn1 = nn.BatchNorm1d(36)
#         self.drop1 = nn.Dropout(0.2)
#         self.conv2 = nn.Conv1d(36, num_classes, 1)
#
#         self.offset_conv_1 = nn.Conv1d(515, 256, 1)
#         self.offset_bn_1 = nn.BatchNorm1d(256)
#         self.dist_conv_1 = nn.Conv1d(515, 256, 1)
#         self.dist_bn_1 = nn.BatchNorm1d(256)
#
#         self.offset_conv_2 = nn.Conv1d(256, 3, 1)
#         self.dist_conv_2 = nn.Conv1d(256, 1, 1)
#
#     def forward(self, xyz):
#         l0_points = xyz
#         l0_xyz = xyz[:, :3, :]
#
#         l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
#         l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
#         l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
#
#
#         l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
#         l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
#         l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)
#
#
#         x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
#         seg_result = self.conv2(x)
#
#         offset_result = F.relu(self.offset_bn_1(self.offset_conv_1(torch.cat([l3_points, l3_xyz], axis=1))))
#         offset_result = self.offset_conv_2(offset_result)
#
#         dist_result = F.relu(self.dist_bn_1(self.dist_conv_1(torch.cat([l3_points, l3_xyz], axis=1))))
#         dist_result = self.dist_conv_2(dist_result)
#
#         return l0_points, l3_points, l0_xyz, l3_xyz, offset_result, dist_result, seg_result


class TSegCentroidNet(nn.Module):
    def __init__(self, num_classes=15):
        super(TSegCentroidNet, self).__init__()

        self.num_classes = num_classes
        input_feauture_num = 6
        self.sa1 = PointNetSetAbstractionMsg(1024, [0.06, 0.12], [32, 64], input_feauture_num, [[32, 32], [32, 32]])
        self.sa2 = PointNetSetAbstractionMsg(512, [0.12, 0.20], [32, 64], 32 + 32, [[64, 128], [64, 128]])
        self.sa3 = PointNetSetAbstractionMsg(256, [0.2, 0.3], [32, 64], 128 + 128, [[196, 256], [196, 256]])

        self.fp3 = PointNetFeaturePropagation(768, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [128, 128])
        self.fp1 = PointNetFeaturePropagation(128 + input_feauture_num, [64, 64])

        self.conv1 = nn.Conv1d(64, 36, 1)
        self.bn1 = nn.BatchNorm1d(36)
        self.drop1 = nn.Dropout(0.2)
        self.conv2 = nn.Conv1d(36, self.num_classes, 1)

        self.offset_conv_1 = nn.Conv1d(515, 256, 1)
        self.offset_bn_1 = nn.BatchNorm1d(256)
        self.dist_conv_1 = nn.Conv1d(515, 256, 1)
        self.dist_bn_1 = nn.BatchNorm1d(256)

        self.offset_conv_1 = nn.Conv1d(515, 256, 1)
        self.offset_bn_1 = nn.BatchNorm1d(256)
        self.dist_conv_1 = nn.Conv1d(515, 256, 1)
        self.dist_bn_1 = nn.BatchNorm1d(256)

        self.offset_conv_2 = nn.Conv1d(256, 3, 1)
        self.dist_conv_2 = nn.Conv1d(256, 1, 1)

        nn.init.zeros_(self.offset_conv_2.weight)
        nn.init.zeros_(self.dist_conv_2.weight)

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        seg_result = self.conv2(x)

        offset_result = F.relu(self.offset_bn_1(self.offset_conv_1(torch.cat([l3_points, l3_xyz], axis=1))))
        offset_result = self.offset_conv_2(offset_result)

        dist_result = F.relu(self.dist_bn_1(self.dist_conv_1(torch.cat([l3_points, l3_xyz], axis=1))))
        dist_result = self.dist_conv_2(dist_result)

        return l0_points, l3_points, l0_xyz, l3_xyz, offset_result, dist_result, seg_result


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    import torch
    model = TSegCentroidNet().to(device)
    xyz = torch.rand(1, 6, 16000).to(device)
    #output is B, C, N order
    for item in model(xyz):
        print(item.shape)
    # input()