import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class EdgeFeatureLayer(nn.Module):
    def __init__(self, k=16):
        super(EdgeFeatureLayer, self).__init__()
        self.k = k

    def forward(self, X_inputs, nn_idx):
        batch_size, num_points, num_dims = X_inputs.shape
        mesh_central = X_inputs

        idx_ = torch.arange(batch_size, device=X_inputs.device).view(-1, 1, 1) * num_points
        mesh_neighbors = X_inputs.view(-1, num_dims)[nn_idx + idx_]
        mesh_central = mesh_central.unsqueeze(2).expand(-1, -1, self.k, -1)

        edge_feature = torch.cat([mesh_central, mesh_neighbors - mesh_central], dim=-1)
        return edge_feature

class KNNLayer(nn.Module):
    def __init__(self, k=16):
        super(KNNLayer, self).__init__()
        self.k = k

    def forward(self, inputs):
        inner = -2 * torch.matmul(inputs, inputs.transpose(1, 2))
        square = torch.sum(inputs ** 2, dim=-1, keepdim=True)
        adj_matrix = square + inner + square.transpose(1, 2)
        nn_idx = adj_matrix.topk(k=self.k, dim=-1, largest=False)[1]
        return nn_idx

class TeethGNN(nn.Module):
    def __init__(self, num_classes=17, in_channels=6, k=16):
        super(TeethGNN, self).__init__()
        self.k = k

        self.knn = KNNLayer(k=k)
        self.edge_feature1 = EdgeFeatureLayer(k=k)
        self.edge_feature2 = EdgeFeatureLayer(k=k)
        self.edge_feature3 = EdgeFeatureLayer(k=k)

        self.conv1a = nn.Conv2d(in_channels * 2, 64, kernel_size=1, bias=False)
        self.bn1a = nn.BatchNorm2d(64)
        self.conv1b = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.bn1b = nn.BatchNorm2d(64)

        self.conv2a = nn.Conv2d(128, 128, kernel_size=1, bias=False)
        self.bn2a = nn.BatchNorm2d(128)
        self.conv2b = nn.Conv2d(128, 128, kernel_size=1, bias=False)
        self.bn2b = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(256, 256, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv_flatten = nn.Conv1d(448, 1024, kernel_size=1, bias=False)
        self.bn_flatten = nn.BatchNorm1d(1024)

        self.conv_semantic = nn.Conv1d(449, 256, kernel_size=1, bias=False)
        self.bn_semantic = nn.BatchNorm1d(256)

        self.conv_offset1 = nn.Conv1d(449, 256, kernel_size=1, bias=False)
        self.bn_offset1 = nn.BatchNorm1d(256)
        self.conv_offset2 = nn.Conv1d(256, 128, kernel_size=1, bias=False)
        self.bn_offset2 = nn.BatchNorm1d(128)
        self.conv_offset3 = nn.Conv1d(128, 3, kernel_size=1)

        self.knn2 = KNNLayer(k=k)
        self.edge_feature_dyn = EdgeFeatureLayer(k=k)
        self.conv_dyn = nn.Conv1d(512, 512, kernel_size=1, bias=False)
        self.bn_dyn = nn.BatchNorm1d(512)
        self.conv_dense1 = nn.Conv1d(512, 256, kernel_size=1, bias=False)
        self.bn_dense1 = nn.BatchNorm1d(256)
        self.conv_dense2 = nn.Conv1d(256, 128, kernel_size=1, bias=False)
        self.bn_dense2 = nn.BatchNorm1d(128)
        self.conv_output = nn.Conv1d(128, num_classes, kernel_size=1)
        self.edge_seg_head = nn.Conv1d(128, 2, kernel_size=1)

    def forward(self, data):
        X_inputs = data.clone().permute(0, 2, 1).contiguous()
        P_inputs = data.clone().permute(0, 2, 1)[:, :, :3].contiguous()

        nn_idx = self.knn(P_inputs)

        net = self.edge_feature1(X_inputs, nn_idx)
        net = net.permute(0, 3, 1, 2)
        net = F.leaky_relu(self.bn1a(self.conv1a(net)), negative_slope=0.2)
        net = F.leaky_relu(self.bn1b(self.conv1b(net)), negative_slope=0.2)
        net1 = net.max(dim=-1, keepdim=False)[0]
        net1 = net1.permute(0, 2, 1).contiguous()

        net = self.edge_feature2(net1, nn_idx)
        net = net.permute(0, 3, 1, 2)
        net = F.leaky_relu(self.bn2a(self.conv2a(net)), negative_slope=0.2)
        net = F.leaky_relu(self.bn2b(self.conv2b(net)), negative_slope=0.2)
        net2 = net.max(dim=-1, keepdim=False)[0]
        net2 = net2.permute(0, 2, 1).contiguous()

        net = self.edge_feature3(net2, nn_idx)
        net = net.permute(0, 3, 1, 2)
        net = F.leaky_relu(self.bn3(self.conv3(net)), negative_slope=0.2)
        net3 = net.max(dim=-1, keepdim=False)[0]
        net3 = net3.permute(0, 2, 1).contiguous()

        net_448 = torch.cat([net1, net2, net3], dim=-1)
        net_448 = net_448.permute(0, 2, 1).contiguous()

        net = F.leaky_relu(self.bn_flatten(self.conv_flatten(net_448)), negative_slope=0.2)
        net_1024 = net.mean(dim=1, keepdim=True)

        feature_output = torch.cat([net_448, net_1024], dim=1)

        net_s = F.leaky_relu(self.bn_semantic(self.conv_semantic(feature_output)), negative_slope=0.2)
        semantic_branch = net_s

        net = F.leaky_relu(self.bn_offset1(self.conv_offset1(feature_output)), negative_slope=0.2)
        net = F.dropout(net, p=0.5, training=self.training)
        net = F.leaky_relu(self.bn_offset2(self.conv_offset2(net)), negative_slope=0.2)
        net = F.dropout(net, p=0.5, training=self.training)
        offset_branch = self.conv_offset3(net).permute(0, 2, 1).contiguous()

        P_offsets_comb = P_inputs + 6 * offset_branch

        nn_idx2 = self.knn2(P_offsets_comb)

        semantic_branch = semantic_branch.permute(0, 2, 1).contiguous()
        net = self.edge_feature_dyn(semantic_branch, nn_idx2)
        net = net.max(dim=2, keepdim=False)[0]
        net = net.permute(0, 2, 1).contiguous()
        net = F.leaky_relu(self.bn_dyn(self.conv_dyn(net)), negative_slope=0.2)
        net = F.leaky_relu(self.bn_dense1(self.conv_dense1(net)), negative_slope=0.2)
        net = F.dropout(net, p=0.5, training=self.training)
        net = F.leaky_relu(self.bn_dense2(self.conv_dense2(net)), negative_slope=0.2)
        net = F.dropout(net, p=0.5, training=self.training)
        seg_result = self.conv_output(net)

        edge_seg_result = self.edge_seg_head(net)

        return seg_result, edge_seg_result


# Usage example
if __name__ == "__main__":
    model = TeethGNN(num_classes=15, in_channels=6, k=16)
    data = torch.randn(1, 6, 16000)
    seg_result, edge_seg_result = model(data)
    print(seg_result.shape)
    print(edge_seg_result.shape)
