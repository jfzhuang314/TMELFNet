import torch
import os
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import pandas as pd
from datetime import datetime

from model2._tseglab.train_tseglab import TSeglabDataset

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# 你的数据集类别数
NUM_CLASSES = 15  # 15 类 + 背景
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_model(num_classes):
    model = maskrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    return model


def predict_one_image(img_path, model, threshold=0.5, show_result=False, save_path=None):
    """
    预测单张图像并可选地显示/保存可视化结果

    Args:
        img_path: 图像路径
        model: Mask R-CNN 模型
        threshold: 置信度阈值
        show_result: 是否显示可视化结果
        save_path: 保存可视化结果的路径（可选）

    Returns:
        boxes, masks, labels, scores
    """


    # 加载图像
    img = Image.open(img_path).convert("RGB")
    img_width, img_height = img.size
    # print(img_width, img_height)

    img_np = np.array(img)
    transform = T.ToTensor()
    img_tensor = transform(img).to(DEVICE)

    with torch.no_grad():
        pred = model([img_tensor])

    # 预测结果
    boxes = pred[0]["boxes"].cpu().numpy()
    masks = pred[0]["masks"].cpu().numpy()
    labels = pred[0]["labels"].cpu().numpy()
    scores = pred[0]["scores"].cpu().numpy()

    # 过滤低置信度
    keep = scores > threshold
    boxes = boxes[keep]
    masks = masks[keep]
    labels = labels[keep]
    scores = scores[keep]

    # 过滤背景类（类别 0）
    non_bg_mask = labels > 0
    boxes = boxes[non_bg_mask]
    masks = masks[non_bg_mask]
    labels = labels[non_bg_mask]
    scores = scores[non_bg_mask]

    # 过滤边界框尺寸过大的检测（宽度或高度 > 200）
    box_widths = boxes[:, 2] - boxes[:, 0]
    box_heights = boxes[:, 3] - boxes[:, 1]
    size_mask = (box_widths <= img_width//2) & (box_heights <= img_height//2)
    boxes = boxes[size_mask]
    labels = labels[size_mask]
    scores = scores[size_mask]
    masks = masks[size_mask]

    print("=== 预测结果 ===")
    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        print(f"实例 {i + 1} | 类别：{label} | 置信度：{score:.2f} | 框：{box.astype(int)}")

    # 可视化结果
    if show_result or save_path:
        fig, ax = plt.subplots(1, figsize=(12, 12))

        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        ax.imshow(img_np)


        # 生成随机颜色
        colors = plt.cm.tab10(np.linspace(0, 1, len(boxes)))

        for i, (box, mask, label, score) in enumerate(zip(boxes, masks, labels, scores)):
            # 绘制边界框
            rect = plt.Rectangle(
                (box[0], box[1]),
                box[2] - box[0],
                box[3] - box[1],
                fill=False,
                color=colors[i],
                linewidth=2,
                label=f'Cls:{label} {score:.2f}'
            )
            ax.add_patch(rect)

            # 绘制 mask（半透明）
            mask_img = np.zeros_like(img_np)
            mask_color = colors[i][:3]
            mask_img[mask[0] > 0.5] = mask_color

            # 叠加 mask
            alpha = 0.4
            blended = img_np.astype(float) * (1 - alpha) + mask_img.astype(float) * alpha
            ax.imshow(blended / 255.0, alpha=alpha)

        ax.set_title(f"Prediction Result of Mask R-CNN (Detected Instances: {len(boxes)} instances)")
        ax.axis('off')

        # 显示
        if show_result:
            plt.tight_layout()
            plt.show()

        # 保存
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 可视化结果已保存到：{save_path}")

        plt.close()

    return boxes, masks, labels, scores

def calculate_mask_iou(mask1, mask2):
    """
    计算两个 mask 的 IoU

    Args:
        mask1: [H, W] numpy array
        mask2: [H, W] numpy array

    Returns:
        IoU value
    """
    mask1 = (mask1 > 0.5).astype(np.uint8)
    mask2 = (mask2 > 0.5).astype(np.uint8)

    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 0.0

    return intersection / union


def match_predictions_to_groundtruth(pred_masks, pred_labels, pred_scores,
                                     gt_masks, gt_labels, iou_threshold=0.5, img_filename=None):
    """
    基于 Mask IoU 匹配预测和真实标注

    Args:
        pred_masks: [N, H, W] 预测 masks
        pred_labels: [N] 预测标签
        pred_scores: [N] 预测置信度
        gt_masks: [M, H, W] 真实 masks
        gt_labels: [M] 真实标签
        iou_threshold: IoU 阈值

    Returns:
        tp: 真正例标记
        fp: 假正例标记
        fn: 假负例计数
    """
    num_preds = len(pred_masks)
    num_gts = len(gt_masks)

    tp = np.zeros(num_preds)
    fp = np.zeros(num_preds)

    if num_gts == 0:
        # 没有 GT，所有预测都是 FP
        fp[:] = 1
        return tp, fp, 0

    if num_preds == 0:
        # 没有预测，所有 GT 都是 FN
        return tp, fp, num_gts

    # 计算 IoU 矩阵 [num_preds, num_gts]
    iou_matrix = np.zeros((num_preds, num_gts))
    for i, pred_mask in enumerate(pred_masks):
        for j, gt_mask in enumerate(gt_masks):
            iou_matrix[i, j] = calculate_mask_iou(pred_mask, gt_mask)

    # 按置信度降序排序（贪心匹配）
    sorted_indices = np.argsort(pred_scores)[::-1]
    matched_gts = set()
    fn_count = 0

    for i in sorted_indices:
        best_iou = 0
        best_gt_idx = -1

        # 找到最佳匹配的 GT
        for j in range(num_gts):
            if j not in matched_gts and iou_matrix[i, j] >= iou_threshold:
                if iou_matrix[i, j] > best_iou:
                    best_iou = iou_matrix[i, j]
                    best_gt_idx = j

        if best_gt_idx >= 0:
            # 检查类别是否匹配
            # if pred_labels[i] == gt_labels[best_gt_idx]:
            tp[i] = 1
            matched_gts.add(best_gt_idx)
            # else:
            #     fp[i] = 1
        else:
            fp[i] = 1
            print(f"预测 {i} 未匹配任何 GT. iou: {iou_matrix[i, best_gt_idx]}")

    # 未匹配的 GT 是 FN
    fn_count = num_gts - len(matched_gts)

    if fn_count > 0:
        if img_filename:
            print(f"{img_filename} 未匹配的 GT 个数：{fn_count}")
        # 紧凑输出 IoU 矩阵
        if num_preds > 0 and num_gts > 0:
            print(f"\n📊 IoU Matrix [{num_preds}×{num_gts}]:")
            np.set_printoptions(precision=2, linewidth=120, suppress=True)
            print(iou_matrix)
            np.set_printoptions(precision=8, linewidth=1000, suppress=False)


    return tp, fp, fn_count


def evaluate_model_with_mask_iou(test_loader, model, threshold=0.5,
                                  mask_iou_threshold=0.5, save_csv_path=None):
    """
    使用 Mask IoU 评估模型并计算 TP/FP/FN/Precision/Recall/F1

    Args:
        test_loader: 数据加载器
        model: 模型
        threshold: 置信度阈值
        mask_iou_threshold: Mask IoU 匹配阈值
        save_csv_path: CSV 保存路径

    Returns:
        results: 评估结果字典
    """
    model.eval()
    total = len(test_loader)

    # 存储统计信息
    all_tp = []  # 每个图像的 TP
    all_fp = []  # 每个图像的 FP
    all_fn = []  # 每个图像的 FN

    # 存储每个图像的预测结果
    all_predictions = []

    with torch.no_grad():
        for idx, (images, targets) in enumerate(test_loader):
            print(f"评估 {idx+1}/{total}")

            images = [img.to(DEVICE) for img in images]
            for target in targets:
                for key in target.keys():
                    if isinstance(target[key], torch.Tensor):
                        target[key] = target[key].to(DEVICE)

            preds = model(images)

            # 处理每个图像
            for img_idx, (pred, target) in enumerate(zip(preds, targets)):
                # 提取预测结果
                boxes = pred["boxes"].cpu().numpy()
                labels = pred["labels"].cpu().numpy()
                scores = pred["scores"].cpu().numpy()
                masks = pred["masks"].cpu().numpy()

                # 过滤低置信度
                keep = scores > threshold
                boxes = boxes[keep]
                labels = labels[keep]
                scores = scores[keep]
                masks = masks[keep]

                # 过滤背景类（类别 0）
                non_bg_mask = labels > 0
                boxes = boxes[non_bg_mask]
                labels = labels[non_bg_mask]
                scores = scores[non_bg_mask]
                masks = masks[non_bg_mask]

                # 提取 GT
                gt_boxes = target["boxes"].cpu().numpy()
                gt_labels = target["labels"].cpu().numpy()
                gt_masks = target["masks"].cpu().numpy()

                # 过滤背景类
                gt_non_bg = gt_labels > 0
                gt_boxes = gt_boxes[gt_non_bg]
                gt_labels = gt_labels[gt_non_bg]
                gt_masks = gt_masks[gt_non_bg]

                # 匹配预测和 GT
                tp, fp, fn = match_predictions_to_groundtruth(
                    masks, labels, scores,
                    gt_masks, gt_labels,
                    iou_threshold=mask_iou_threshold
                )

                all_tp.append(tp.sum())
                all_fp.append(fp.sum())
                all_fn.append(fn)

                # 记录预测详情
                for i, (box, label, score, mask) in enumerate(zip(boxes, labels, scores, masks)):
                    all_predictions.append({
                        'image_id': idx * len(images) + img_idx,
                        'category_id': int(label),
                        'confidence': float(score),
                        'x_min': int(box[0]),
                        'y_min': int(box[1]),
                        'x_max': int(box[2]),
                        'y_max': int(box[3]),
                        'width': int(box[2] - box[0]),
                        'height': int(box[3] - box[1]),
                        'is_tp': bool(tp[i] == 1),
                        'is_fp': bool(fp[i] == 1),
                        'is_fn': bool(fn[i] == 1)
                    })

    # 计算总体指标
    total_tp = sum(all_tp)
    total_fp = sum(all_fp)
    total_fn = sum(all_fn)

    # Precision, Recall, F1
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # 打印结果
    print("\n" + "="*70)
    print("📊 实例分割评估结果（基于 Mask IoU）")
    print("="*70)
    print(f"\n【匹配统计】")
    print(f"  True Positives (TP):  {total_tp}")
    print(f"  False Positives (FP): {total_fp}")
    print(f"  False Negatives (FN): {total_fn}")

    print(f"\n【核心指标】")
    print(f"  Precision (查准率):     {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall (召回率):      {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1-Score:             {f1_score:.4f} ({f1_score*100:.2f}%)")
    print(f"  Mask IoU Threshold:   {mask_iou_threshold}")
    print(f"  Confidence Threshold: {threshold}")
    print("="*70)

    # 保存结果
    if save_csv_path and len(all_predictions) > 0:
        os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)

        df_predictions = pd.DataFrame(all_predictions)

        # 汇总信息
        summary_data = {
            'metric': [
                'TP',
                'FP',
                'FN',
                'Precision',
                'Recall',
                'F1_Score',
                'total_images',
                'total_predictions',
                'mask_iou_threshold',
                'confidence_threshold',
                'timestamp'
            ],
            'value': [
                total_tp,
                total_fp,
                total_fn,
                precision,
                recall,
                f1_score,
                total,
                len(df_predictions),
                mask_iou_threshold,
                threshold,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ]
        }
        df_summary = pd.DataFrame(summary_data)

        # 使用 ExcelWriter 保存多个 sheet
        with pd.ExcelWriter(save_csv_path, engine='openpyxl') as writer:
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
            df_predictions.to_excel(writer, sheet_name='Predictions', index=False)

        print(f"\n📊 评估结果已保存到：{save_csv_path}")
        print(f"   - Summary: TP/FP/FN/Precision/Recall/F1")
        print(f"   - Predictions: 所有预测详情 ({len(df_predictions)} 条)")

    return {
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'predictions': all_predictions
    }

def evaluate_model_with_mask_iou_macro(test_loader, model, threshold=0.5,
                                  mask_iou_threshold=0.5, save_csv_path=None):
    """
    使用 Mask IoU 评估模型并计算 TP/FP/FN/Precision/Recall/F1（Macro-Average）

    Args:
        test_loader: 数据加载器
        model: 模型
        threshold: 置信度阈值
        mask_iou_threshold: Mask IoU 匹配阈值
        save_csv_path: CSV 保存路径

    Returns:
        results: 评估结果字典
    """
    # model.eval()
    total = len(test_loader)

    # 存储统计信息（按类别分开）
    class_tp = {}  # {class_id: [每张图片的 TP]}
    class_fp = {}  # {class_id: [每张图片的 FP]}
    class_fn = {}  # {class_id: [每张图片的 FN]}

    # 存储每个图像的预测结果
    all_predictions = []


    for idx, (images, targets) in enumerate(test_loader):
        print(f"评估 {idx+1}/{total}")

        images = [img.to(DEVICE) for img in images]

        img_width, img_height = images[0].shape[1], images[0].shape[2]

        for target in targets:
            for key in target.keys():
                if isinstance(target[key], torch.Tensor):
                    target[key] = target[key].to(DEVICE)

        # img_np= images[0].cpu().numpy().transpose(1, 2, 0)
        # fig, ax = plt.subplots(1, figsize=(12, 12))
        #
        # fig.patch.set_facecolor('white')
        # ax.set_facecolor('white')
        #
        # ax.imshow(img_np)

        with torch.no_grad():
            preds = model(images)

        img_filename = test_loader.dataset.imgs[idx]

        # 处理每个图像
        for img_idx, (pred, target) in enumerate(zip(preds, targets)):
            # 提取预测结果
            boxes = pred["boxes"].cpu().numpy()
            labels = pred["labels"].cpu().numpy()
            scores = pred["scores"].cpu().numpy()
            masks = pred["masks"].cpu().numpy()

            # 过滤低置信度
            keep = scores > threshold
            boxes = boxes[keep]
            labels = labels[keep]
            scores = scores[keep]
            masks = masks[keep]

            # 过滤背景类（类别 0）
            non_bg_mask = labels > 0
            boxes = boxes[non_bg_mask]
            labels = labels[non_bg_mask]
            scores = scores[non_bg_mask]
            masks = masks[non_bg_mask]

            # 过滤边界框尺寸过大的检测（宽度或高度 > 200）
            box_widths = boxes[:, 2] - boxes[:, 0]
            box_heights = boxes[:, 3] - boxes[:, 1]
            size_mask = (box_widths <= img_width//2) & (box_heights <= img_height//2)
            boxes = boxes[size_mask]
            labels = labels[size_mask]
            scores = scores[size_mask]
            masks = masks[size_mask]

            # 过滤边界框尺寸过大的检测（宽度或高度 > 200）


            # 提取 GT
            gt_boxes = target["boxes"].cpu().numpy()
            gt_labels = target["labels"].cpu().numpy()
            gt_masks = target["masks"].cpu().numpy()

            # 过滤背景类
            gt_non_bg = gt_labels > 0
            gt_boxes = gt_boxes[gt_non_bg]
            gt_labels = gt_labels[gt_non_bg]
            gt_masks = gt_masks[gt_non_bg]

            # 匹配预测和 GT
            tp, fp, fn = match_predictions_to_groundtruth(
                masks, labels, scores,
                gt_masks, gt_labels,
                iou_threshold=mask_iou_threshold,
                img_filename=img_filename
            )

            # ========== 按类别统计 TP/FP/FN ==========
            # 获取所有出现的类别
            all_classes = set(labels.tolist()) | set(gt_labels.tolist())

            for cls_id in all_classes:
                # 初始化该类别的统计
                if cls_id not in class_tp:
                    class_tp[cls_id] = []
                    class_fp[cls_id] = []
                    class_fn[cls_id] = []

                # 找出当前图像中该类别的预测
                cls_pred_mask = labels == cls_id
                cls_gt_mask = gt_labels == cls_id

                # 该类别的 TP/FP
                cls_tp = tp[cls_pred_mask].sum() if cls_pred_mask.any() else 0
                cls_fp = fp[cls_pred_mask].sum() if cls_pred_mask.any() else 0


                # 该类别的 FN（未匹配的 GT）
                # 需要计算该类别有多少 GT 被匹配了
                matched_gt_count = 0
                if cls_pred_mask.any():
                    # 找到哪些 GT 被这个类别的预测匹配了
                    pred_indices = np.where(cls_pred_mask)[0]
                    for pred_idx in pred_indices:
                        if tp[pred_idx] == 1:
                            matched_gt_count += 1

                cls_fn_count = max(cls_gt_mask.sum() - matched_gt_count, 0)

                class_tp[cls_id].append(cls_tp)
                class_fp[cls_id].append(cls_fp)
                class_fn[cls_id].append(cls_fn_count)
            # ========================================

            # 记录预测详情
            for i, (box, label, score, mask) in enumerate(zip(boxes, labels, scores, masks)):
                all_predictions.append({
                    'image_id': idx * len(images) + img_idx,
                    'filename': img_filename,  # ← 已包含文件名
                    'category_id': int(label),
                    'confidence': float(score),
                    'x_min': int(box[0]),
                    'y_min': int(box[1]),
                    'x_max': int(box[2]),
                    'y_max': int(box[3]),
                    'width': int(box[2] - box[0]),
                    'height': int(box[3] - box[1]),
                    'is_tp': bool(tp[i] == 1),
                    'is_fp': bool(fp[i] == 1),
                    'num_fn': fn
                })

    # ========== Macro-Average 计算 ==========
    class_metrics = {}

    print("\n" + "="*70)
    print("📊 各类别评估结果（Macro-Average）")
    print("="*70)
    print(f"\n{'类别':<8} {'TP':<6} {'FP':<6} {'FN':<6} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-"*70)

    for cls_id in sorted(class_tp.keys()):
        total_tp = sum(class_tp[cls_id])
        total_fp = sum(class_fp[cls_id])
        total_fn = sum(class_fn[cls_id])

        # 计算该类别的 Precision, Recall, F1
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        class_metrics[cls_id] = {
            'tp': total_tp,
            'fp': total_fp,
            'fn': total_fn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }

        print(f"{cls_id:<8} {total_tp:<6} {total_fp:<6} {total_fn:<6} {precision:<10.4f} {recall:<10.4f} {f1_score:<10.4f}")

    # 计算 Macro-Average（各类别的平均）
    num_classes = len(class_metrics)
    macro_precision = sum(m['precision'] for m in class_metrics.values()) / num_classes
    macro_recall = sum(m['recall'] for m in class_metrics.values()) / num_classes
    macro_f1 = sum(m['f1_score'] for m in class_metrics.values()) / num_classes

    # 计算 Micro-Average（全局汇总后再算）
    total_tp_all = sum(sum(class_tp[cls_id]) for cls_id in class_tp.keys())
    total_fp_all = sum(sum(class_fp[cls_id]) for cls_id in class_fp.keys())
    total_fn_all = sum(sum(class_fn[cls_id]) for cls_id in class_fn.keys())

    micro_precision = total_tp_all / (total_tp_all + total_fp_all) if (total_tp_all + total_fp_all) > 0 else 0.0
    micro_recall = total_tp_all / (total_tp_all + total_fn_all) if (total_tp_all + total_fn_all) > 0 else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

    print("-"*70)
    print(f"\n【Macro-Average 指标】（先分各类，再求平均）")
    print(f"  Precision: {macro_precision:.4f} ({macro_precision*100:.2f}%)")
    print(f"  Recall:    {macro_recall:.4f} ({macro_recall*100:.2f}%)")
    print(f"  F1-Score:  {macro_f1:.4f} ({macro_f1*100:.2f}%)")

    print(f"\n【Micro-Average 指标】（全局汇总后计算）")
    print(f"  Precision: {micro_precision:.4f} ({micro_precision*100:.2f}%)")
    print(f"  Recall:    {micro_recall:.4f} ({micro_recall*100:.2f}%)")
    print(f"  F1-Score:  {micro_f1:.4f} ({micro_f1*100:.2f}%)")

    print(f"\n  Total TP: {total_tp_all}, FP: {total_fp_all}, FN: {total_fn_all}")
    print(f"  Num Classes: {num_classes}")
    print("="*70)
    # ========================================

    # 保存结果
    if save_csv_path and len(all_predictions) > 0:
        os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)

        df_predictions = pd.DataFrame(all_predictions)

        # 汇总信息（包含 Macro 和 Micro）
        summary_data = {
            'metric': [
                'Macro_Precision',
                'Macro_Recall',
                'Macro_F1_Score',
                'Micro_Precision',
                'Micro_Recall',
                'Micro_F1_Score',
                'Total_TP',
                'Total_FP',
                'Total_FN',
                'Num_Classes',
                'total_images',
                'total_predictions',
                'mask_iou_threshold',
                'confidence_threshold',
                'timestamp'
            ],
            'value': [
                macro_precision,
                macro_recall,
                macro_f1,
                micro_precision,
                micro_recall,
                micro_f1,
                total_tp_all,
                total_fp_all,
                total_fn_all,
                num_classes,
                total,
                len(df_predictions),
                mask_iou_threshold,
                threshold,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ]
        }
        df_summary = pd.DataFrame(summary_data)

        # 添加各类别的详细指标
        df_classes = pd.DataFrame([
            {
                'class_id': cls_id,
                'tp': metrics['tp'],
                'fp': metrics['fp'],
                'fn': metrics['fn'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score']
            }
            for cls_id, metrics in sorted(class_metrics.items())
        ])

        # 使用 ExcelWriter 保存多个 sheet
        with pd.ExcelWriter(save_csv_path, engine='openpyxl') as writer:
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
            df_classes.to_excel(writer, sheet_name='Class_Metrics', index=False)
            df_predictions.to_excel(writer, sheet_name='Predictions', index=False)

        print(f"\n📊 评估结果已保存到：{save_csv_path}")
        print(f"   - Summary: Macro/Micro 指标汇总")
        print(f"   - Class_Metrics: 各类别详细指标 ({num_classes} 类)")
        print(f"   - Predictions: 所有预测详情 ({len(df_predictions)} 条)")

    return {
        'class_metrics': class_metrics,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1,
        'predictions': all_predictions
    }


# def batch_predict_images(data_root, save_root, model):
#
#     images_dir = os.path.join(data_root, "test", "images")
#     all_filenames = sorted(os.listdir(images_dir))
#     print(f"📁 找到 {len(all_filenames)} 个文件:")
#     for i, filename in enumerate(all_filenames):
#         print(f"  [{i + 1:3d}] {filename}")
#
#         save_path = os.path.join(save_root, filename)
#         predict_one_image(os.path.join(images_dir, filename), model, threshold=0.8, save_path=save_path)
def batch_predict_images(test_loader, data_root, save_root, model, threshold=0.5):

    total = len(test_loader)
    print(f"📁 找到 {total} 张图像需要预测")

    model.eval()

    with torch.no_grad():
        for idx, (images, targets) in enumerate(test_loader):
            # 获取文件名
            img_filename = test_loader.dataset.imgs[idx]
            print(f"\n[{idx+1}/{total}] 处理：{img_filename}")

            # 构建保存路径
            save_path = os.path.join(save_root, img_filename)

            # 调用 predict_one_image（传入图像路径）
            img_path = os.path.join(data_root, "test", "images", img_filename)
            predict_one_image(
                img_path=img_path,
                model=model,
                threshold=threshold,
                show_result=False,
                save_path=save_path
            )


def main():
    data_root = "E:\\mesh_datasets\\3DTeeth\\data\\T3DSPlusSegDataset\\tseglab_processed\\qslim_10000"  # 你的数据集路径
    save_root = "E:\\mesh_datasets\\3DTeeth\\data\\T3DSPlusSegDataset\\tseglab_processed\\qslim_10000\\pred\\imgs"
    csv_root = "E:\\mesh_datasets\\3DTeeth\\data\\T3DSPlusSegDataset\\tseglab_processed\\qslim_10000\\pred\\csv"
    # 加载训练好的权重
    model = get_model(NUM_CLASSES)
    model.load_state_dict(torch.load("mask_rcnn_model.pth", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()  # 切换到评估模式


    # 使用模型进行预测
    # batch_predict_images(data_root, save_root, model, threshold=0.8)


    # 准备测试集
    test_dataset = TSeglabDataset(data_root + "/test", transforms=T.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             collate_fn=lambda x: tuple(zip(*x)))
    # 评估并保存结果（使用 Mask IoU 方法）
    save_csv_path = os.path.join(csv_root, "eval_" + datetime.now().strftime("%Y%m%d_%H%M%S")
                                 + ".xlsx")

    # batch_predict_images(test_loader, data_root, save_root, model, threshold=0.5)

    evaluate_model_with_mask_iou_macro(test_loader, model, threshold=0.5,
                                       mask_iou_threshold=0.5, save_csv_path=save_csv_path)


if __name__ == "__main__":

    main()
