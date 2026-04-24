import torch


class ConfAwareBinSegLoss:
    @staticmethod
    def first_seg_loss(pred_mask_1, pred_weight_1, gt_bin_label):
        # pred_mask_1: batch_size, 2, cropped
        # pred_weight_1: batch_size, 1, cropped
        # gt_label: batch_size, 1, cropped

        gt_bin_label = gt_bin_label.type(torch.long).view(gt_bin_label.shape[0], -1)

        bce_1 = torch.nn.NLLLoss(reduction='none')(pred_mask_1, gt_bin_label)
        pred_weight_1 = torch.sigmoid(pred_weight_1).view(pred_weight_1.shape[0], -1)
        loss = torch.mean((bce_1 * pred_weight_1) ** 2 + (1 - pred_weight_1) ** 2)

        return loss

    @staticmethod
    def second_seg_loss(pred_mask_2, pred_weight_1, gt_bin_label):
        # pred_mask_2: batch_size, 2, cropped
        # pred_weight_1: batch_size, 1, cropped
        # gt_label: batch_size, 1, cropped

        gt_bin_label = gt_bin_label.type(torch.float32).view(gt_bin_label.shape[0], -1)

        pred_mask_2 = pred_mask_2.view(pred_mask_2.shape[0], -1)
        bce_2 = torch.nn.BCEWithLogitsLoss(reduction='none')(pred_mask_2, gt_bin_label)
        pred_weight_1 = torch.sigmoid(pred_weight_1).view(pred_weight_1.shape[0], -1)
        loss = torch.mean((2.0 - pred_weight_1) * bce_2)

        return loss