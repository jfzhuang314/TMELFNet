import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.to(x.get_device())
        x = x + iden
        x = x.view(-1, 3, 3)
        return x

class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 512, 1)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(128)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.to(x.get_device())
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class MeshSegNet(nn.Module):
    def __init__(self, num_classes=17, num_channels=6, with_dropout=True, dropout_p=0.5):
        super(MeshSegNet, self).__init__()
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.with_dropout = with_dropout
        self.dropout_p = dropout_p

        # MLP-1 [64, 64]
        self.mlp1_conv1 = torch.nn.Conv1d(self.num_channels, 64, 1)
        self.mlp1_conv2 = torch.nn.Conv1d(64, 64, 1)
        self.mlp1_bn1 = nn.BatchNorm1d(64)
        self.mlp1_bn2 = nn.BatchNorm1d(64)
        # FTM (feature-transformer module)
        self.fstn = STNkd(k=64)
        # GLM-1 (graph-contrained learning modulus)
        self.glm1_conv1_1 = torch.nn.Conv1d(64, 32, 1)
        self.glm1_conv1_2 = torch.nn.Conv1d(64, 32, 1)
        self.glm1_bn1_1 = nn.BatchNorm1d(32)
        self.glm1_bn1_2 = nn.BatchNorm1d(32)
        self.glm1_conv2 = torch.nn.Conv1d(32+32, 64, 1)
        self.glm1_bn2 = nn.BatchNorm1d(64)
        # MLP-2
        self.mlp2_conv1 = torch.nn.Conv1d(64, 64, 1)
        self.mlp2_bn1 = nn.BatchNorm1d(64)
        self.mlp2_conv2 = torch.nn.Conv1d(64, 128, 1)
        self.mlp2_bn2 = nn.BatchNorm1d(128)
        self.mlp2_conv3 = torch.nn.Conv1d(128, 512, 1)
        self.mlp2_bn3 = nn.BatchNorm1d(512)
        # GLM-2 (graph-contrained learning modulus)
        self.glm2_conv1_1 = torch.nn.Conv1d(512, 128, 1)
        self.glm2_conv1_2 = torch.nn.Conv1d(512, 128, 1)
        self.glm2_conv1_3 = torch.nn.Conv1d(512, 128, 1)
        self.glm2_bn1_1 = nn.BatchNorm1d(128)
        self.glm2_bn1_2 = nn.BatchNorm1d(128)
        self.glm2_bn1_3 = nn.BatchNorm1d(128)
        self.glm2_conv2 = torch.nn.Conv1d(128*3, 512, 1)
        self.glm2_bn2 = nn.BatchNorm1d(512)
        # MLP-3
        self.mlp3_conv1 = torch.nn.Conv1d(64+512+512+512, 256, 1)
        self.mlp3_conv2 = torch.nn.Conv1d(256, 256, 1)
        self.mlp3_bn1_1 = nn.BatchNorm1d(256)
        self.mlp3_bn1_2 = nn.BatchNorm1d(256)
        self.mlp3_conv3 = torch.nn.Conv1d(256, 128, 1)
        self.mlp3_conv4 = torch.nn.Conv1d(128, 128, 1)
        self.mlp3_bn2_1 = nn.BatchNorm1d(128)
        self.mlp3_bn2_2 = nn.BatchNorm1d(128)
        # output
        self.output_conv = torch.nn.Conv1d(128, self.num_classes, 1)

        # 边界分割头
        self.edge_seg_head = torch.nn.Conv1d(128, 2, 1)
        if self.with_dropout:
            self.dropout = nn.Dropout(p=self.dropout_p)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]

        # 点云坐标
        coords = x[:, :3, :].clone().permute(0, 2, 1)

        # 计算邻接矩阵
        a_s, a_l = [], []
        for i in range(batchsize):
            TX = torch.as_tensor(coords[i], device=coords.device)
            TD = torch.cdist(TX, TX)  # 计算两个tensor中所有向量之间的距离
            D = TD.cpu().numpy()

            S1 = np.zeros([n_pts, n_pts], dtype='float32')
            S2 = np.zeros([n_pts, n_pts], dtype='float32')

            S1[D < 0.1] = 1.0
            # S1 = S1 / (np.dot(np.sum(S1, axis=1, keepdims=True, dtype='float32'), np.ones((1, self.patch_size), dtype='float32')) + 1e-5)
            result = torch.matmul(torch.sum(torch.tensor(S1, device=coords.device), dim=1, keepdim=True),
                                  torch.ones((1, n_pts), device=coords.device))
            S1 = torch.tensor(S1, device=coords.device) / (result + 1e-5)
            a_s.append(S1)

            S2[D < 0.2] = 1.0
            # S2 = S2 / (np.dot(np.sum(S2, axis=1, keepdims=True, dtype='float32'), np.ones((1, self.patch_size), dtype='float32')) + 1e-5)
            result = torch.matmul(
                torch.sum(torch.tensor(S2, device=coords.device), dim=1, keepdim=True),
                torch.ones((1, n_pts), device=coords.device))
            S2 = torch.tensor(S2, device=coords.device) / (result + 1e-5)
            a_l.append(S2)
        a_s = torch.stack(a_s, dim=0)
        a_l = torch.stack(a_l, dim=0)

        # MLP-1
        x = F.relu(self.mlp1_bn1(self.mlp1_conv1(x)))
        x = F.relu(self.mlp1_bn2(self.mlp1_conv2(x)))
        # FTM
        trans_feat = self.fstn(x)
        x = x.transpose(2, 1)
        x_ftm = torch.bmm(x, trans_feat)
        # GLM-1
        sap = torch.bmm(a_s, x_ftm)
        sap = sap.transpose(2, 1)
        x_ftm = x_ftm.transpose(2, 1)
        x = F.relu(self.glm1_bn1_1(self.glm1_conv1_1(x_ftm)))
        glm_1_sap = F.relu(self.glm1_bn1_2(self.glm1_conv1_2(sap)))
        x = torch.cat([x, glm_1_sap], dim=1)
        x = F.relu(self.glm1_bn2(self.glm1_conv2(x)))
        # MLP-2
        x = F.relu(self.mlp2_bn1(self.mlp2_conv1(x)))
        x = F.relu(self.mlp2_bn2(self.mlp2_conv2(x)))
        x_mlp2 = F.relu(self.mlp2_bn3(self.mlp2_conv3(x)))
        if self.with_dropout:
            x_mlp2 = self.dropout(x_mlp2)
        # GLM-2
        x_mlp2 = x_mlp2.transpose(2, 1)
        sap_1 = torch.bmm(a_s, x_mlp2)
        sap_2 = torch.bmm(a_l, x_mlp2)
        x_mlp2 = x_mlp2.transpose(2, 1)
        sap_1 = sap_1.transpose(2, 1)
        sap_2 = sap_2.transpose(2, 1)
        x = F.relu(self.glm2_bn1_1(self.glm2_conv1_1(x_mlp2)))
        glm_2_sap_1 = F.relu(self.glm2_bn1_2(self.glm2_conv1_2(sap_1)))
        glm_2_sap_2 = F.relu(self.glm2_bn1_3(self.glm2_conv1_3(sap_2)))
        x = torch.cat([x, glm_2_sap_1, glm_2_sap_2], dim=1)
        x_glm2 = F.relu(self.glm2_bn2(self.glm2_conv2(x)))
        # GMP
        x = torch.max(x_glm2, 2, keepdim=True)[0]
        # Upsample
        x = torch.nn.Upsample(n_pts)(x)
        # Dense fusion
        x = torch.cat([x, x_ftm, x_mlp2, x_glm2], dim=1)
        # MLP-3
        x = F.relu(self.mlp3_bn1_1(self.mlp3_conv1(x)))
        x = F.relu(self.mlp3_bn1_2(self.mlp3_conv2(x)))
        x = F.relu(self.mlp3_bn2_1(self.mlp3_conv3(x)))
        if self.with_dropout:
            x = self.dropout(x)
        x = F.relu(self.mlp3_bn2_2(self.mlp3_conv4(x)))
        # output
        seg_result = self.output_conv(x)
        edge_seg_result = self.edge_seg_head(x)

        return seg_result, edge_seg_result

if __name__ == '__main__':
    model = MeshSegNet(num_classes=17, num_channels=6)
    data = torch.rand((2, 6, 16000))
    seg_result, edge_seg_result = model(data)
    print(seg_result.shape)
    print(edge_seg_result.shape)