import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.transforms as T


# ===========================================================================
# 1. 自定义数据集类（支持多实例、多类别）
# ===========================================================================
class TSeglabDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "masks"))))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # 读取图片
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        mask_path = os.path.join(self.root, "masks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = np.array(mask)

        # 提取所有物体（去掉背景 0）
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]

        # 拆分每个实例的掩码
        masks = mask == obj_ids[:, None, None]

        # 自动计算 bounding box
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # 转 Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # ====================== 这里修改你的类别！======================
        # 单类别：全部标为 1
        # 多类别：自己写规则映射 labels
        # labels = torch.ones((num_objs,), dtype=torch.int64)

        labels = []
        for obj_id in obj_ids:
            labels.append(obj_id)  # 掩码1=类别1，掩码2=类别2...
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # =================================================================

        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target


# ===========================================================================
# 2. 构建 Mask R-CNN 模型（官方现成）
# ===========================================================================
def get_model(num_classes):
    # 加载预训练模型
    model = maskrcnn_resnet50_fpn(pretrained=True)

    # 替换检测头
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 替换掩码头
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)

    return model


# ===========================================================================
# 3. 训练配置
# ===========================================================================
def main():
    # 1. 路径 & 参数
    data_root = "E:\\mesh_datasets\\3DTeeth\\data\\T3DSPlusSegDataset\\tseglab_processed\\qslim_10000"  # 你的数据集路径
    num_classes = 15  # 1类 + 背景 = 2；如果3类则填 4
    num_epochs = 50
    batch_size = 3
    lr = 0.0005
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # 2. 加载数据集
    dataset = TSeglabDataset(data_root + "/train", transforms=T.ToTensor())
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x))
    )

    # 3. 创建模型
    model = get_model(num_classes)
    model.to(device)

    # 4. 优化器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)

    # 5. 开始训练
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # 前向传播 + 计算损失
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # 反向传播
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()

            # print(f"{targets[0]['image_id']}   Batch Loss: {losses.item():.4f}")

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss:.4f}")

    # 保存模型
    torch.save(model.state_dict(), "mask_rcnn_model.pth")
    print("训练完成，模型已保存为 mask_rcnn_model.pth")


if __name__ == "__main__":
    main()