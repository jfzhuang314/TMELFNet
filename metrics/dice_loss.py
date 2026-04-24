import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """适用于[N,C]格式输入和[N]格式targets的Dice损失函数"""

    def __init__(self,
                 smooth: float = 1e-5,
                 reduction: str = 'mean',
                 weight: torch.Tensor = None,
                 ignore_index: int = None):
        """
        初始化Dice损失函数

        参数:
            smooth: 平滑因子，防止分母为零
            reduction: 损失计算方式，可选 'mean', 'sum', 'none'
            weight: 各类别的权重，形状为[C]
            ignore_index: 需要忽略的类别索引
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算Dice损失

        参数:
            inputs: 预测值，形状为 [N, C]
            targets: 真实标签，形状为 [N] （类别索引）
        """
        # 确保输入和目标维度匹配
        assert inputs.dim() == 2 and targets.dim() == 1, "输入必须为[N,C]和[N]格式"
        assert inputs.size(0) == targets.size(0), "batch大小不匹配"

        # 对输入应用softmax获取概率分布
        inputs = F.softmax(inputs, dim=1)

        # 将targets转换为one-hot格式
        num_classes = inputs.size(1)
        targets_onehot = F.one_hot(targets, num_classes=num_classes)  # [N, C]

        # 忽略指定类别（如果有）
        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            inputs = inputs[mask]
            targets_onehot = targets_onehot[mask]

        # 计算交集和并集
        intersection = torch.sum(inputs * targets_onehot, dim=0)  # [C]
        union = torch.sum(inputs + targets_onehot, dim=0)  # [C]

        # 应用类别权重（如果提供）
        if self.weight is not None:
            weight = self.weight.to(inputs.device)
            intersection = weight * intersection
            union = weight * union

        # 计算Dice分数
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)

        # 计算损失（1 - Dice）
        dice_loss = 1 - dice_score

        # 应用reduction
        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        else:
            return dice_loss


# 使用示例
if __name__ == "__main__":
    # 模拟模型输出 [batch_size=4, classes=3]
    logits = torch.randn(4, 3)

    # 模拟真实标签（类别索引）
    targets = torch.tensor([0, 1, 2, 1])  # [N]

    # 创建Dice损失函数实例
    dice_loss = DiceLoss()

    # 计算损失
    loss = dice_loss(logits, targets)
    print(f"Dice Loss: {loss.item():.4f}")