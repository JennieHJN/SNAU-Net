import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass


def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)


def tversky_loss(input, target, beta=0.7):
    bs = target.size(0)
    loss = 0.0

    for i in range(bs):
        prob = input[i]
        ref = target[i]

        alpha = 1.0 - beta

        tp = (ref * prob).sum()
        fp = ((1 - ref) * prob).sum()
        fn = (ref * (1 - prob)).sum()
        tversky = tp / (tp + alpha * fp + beta * fn)
        loss = loss + (1 - tversky)
    return loss

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()

    def forward(self, input, target):
        # input = torch.sigmoid(input)
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)

        # tversky = tversky_loss(input, target)

        alpha = 0.25
        gamma = 2
        pt = torch.exp(-bce)
        focal_loss = alpha * (1-pt) ** gamma * bce

        input_1 = input[:, 0, :, :]
        input_2 = input[:, 1, :, :]
        target_1 = target[:, 0, :, :]
        target_2 = target[:, 1, :, :]

        input_1 = input_1.view(num, -1)
        input_2 = input_2.view(num, -1)

        target_1 = target_1.view(num, -1)
        target_2 = target_2.view(num, -1)

        intersection_1 = (input_1 * target_1)
        intersection_2 = (input_2 * target_2)

        dice_1 = (2. * intersection_1.sum(1) + smooth) / (input_1.sum(1) + target_1.sum(1) + smooth)
        dice_2 = (2. * intersection_2.sum(1) + smooth) / (input_2.sum(1) + target_2.sum(1) + smooth)

        dice_1 = 1 - dice_1.sum() / num
        dice_2 = 1 - dice_2.sum() / num

        dice = (dice_1 + dice_2) / 2.0
        # return 2 * focal_loss + 0.5 * dice
        return 2 * focal_loss + 0.5 * dice
        # return 0.5 * focal_loss + 1 * dice
        # return bce

class BCEDiceLoss_liver(nn.Module):
    def __init__(self):
        super(BCEDiceLoss_liver, self).__init__()

    def forward(self, input, target):
        # input = torch.sigmoid(input)
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)

        iou_loss = 1 - iou_score(input, target)

        input_1 = input[:, 0, :, :]

        target_1 = target[:, 0, :, :]

        input_1 = input_1.view(num, -1)

        target_1 = target_1.view(num, -1)

        intersection_1 = (input_1 * target_1)

        dice_1 = (2. * intersection_1.sum(1) + smooth) / (input_1.sum(1) + target_1.sum(1) + smooth)

        dice_1 = 1 - dice_1.sum() / num

        return 0.5 * bce + dice_1


class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super(LovaszHingeLoss, self).__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss
