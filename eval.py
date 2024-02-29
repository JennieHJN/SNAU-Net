import os
import argparse
from collections import OrderedDict
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.svm._libsvm import predict
import warnings
import datetime
import numpy as np
from tqdm import tqdm
import joblib
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import datasets, models, transforms
from net.Swin_run_lcab import lcab_two
from utilities.metrics import dice_coef, batch_iou, mean_iou, iou_score
import utilities.losses as losses
from utilities.utils import str2bool, count_params

from config import get_config

loss_names = list(losses.__dict__.keys())
loss_names.append('BCEWithLogitsLoss')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--deepsupervision', default=False,
                        help='model name: (default: arch+timestamp)')

    parser.add_argument('--arch', '-a', metavar='ARCH', default='SNAU_Net',
                        # choices=arch_names,
                        help='model architecture: ' +
                             ' | '.join(SNAU_Net) +
                             ' (default: NestedUNet)')
    parser.add_argument('--dataset', default="3Diradb",
                        help='dataset name')
    parser.add_argument('--input-channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--image-ext', default='npy',
                        help='image file extension')
    parser.add_argument('--mask-ext', default='npy',
                        help='mask file extension')
    parser.add_argument('--aug', default=False, type=str2bool)
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=loss_names,
                        help='loss: ' +
                             ' | '.join(loss_names) +
                             ' (default: BCEDiceLoss)')
    parser.add_argument('--epochs', default=150, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--early-stop', default=150, type=int,
                        metavar='N', help='early stopping (default: 30)')
    parser.add_argument('-b', '--batch-size', default=8, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                             ' | '.join(['Adam', 'SGD']) +
                             ' (default: Adam)')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    args = parser.parse_args()

    return args


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def validate(args, val_loader, model, criterion):
    losses = AverageMeter()
    ious = AverageMeter()
    dices_1s = AverageMeter()
    dices_2s = AverageMeter()

    model.eval()

    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()

            # compute output
            if args.deepsupervision:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou = iou_score(outputs[-1], target)
            else:
                output = model(input)
                loss = criterion(output, target)
                iou = iou_score(output, target)
                dice_1 = dice_coef(output, target)[0]
                dice_2 = dice_coef(output, target)[1]

            losses.update(loss.item(), input.size(0))
            ious.update(iou, input.size(0))
            dices_1s.update(torch.tensor(dice_1), input.size(0))
            dices_2s.update(torch.tensor(dice_2), input.size(0))

    log = OrderedDict([
        ('loss', losses.avg),
        ('iou', ious.avg),
        ('dice_1', dices_1s.avg),
        ('dice_2', dices_2s.avg)
    ])

    return log


def dice_coef(output, target):
    smooth = 1e-5
    num = output.shape[0]
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    input_1 = output[:, 0, :, :]
    input_2 = output[:, 1, :, :]

    target_1 = target[:, 0, :, :]
    target_2 = target[:, 1, :, :]

    intersection_1 = (input_1 * target_1)
    intersection_2 = (input_2 * target_2)

    dice_1 = (2. * intersection_1.sum() + smooth) / (input_1.sum() + target_1.sum() + smooth)
    dice_2 = (2. * intersection_2.sum() + smooth) / (input_2.sum() + target_2.sum() + smooth)

    return dice_1, dice_2


def predict(model, input, target):
    img = input.cuda()
    img = img.unsqueeze(dim=0)
    target = target.unsqueeze(dim=0)
    output = model(img)
    dice_1 = dice_coef(output, target)[0]
    dice_2 = dice_coef(output, target)[1]

    output = torch.sigmoid(output).data.cpu().numpy()
    probability_map = np.zeros([1, 224, 224], dtype=np.uint8)

    for idx in range(output.shape[2]):
        for idy in range(output.shape[3]):
            if (output[0, 0, idx, idy] > 0.5):
                probability_map[0, idy, idx] = 1
            if (output[0, 1, idx, idy] > 0.5):
                probability_map[0, idy, idx] = 2

    return probability_map, dice_1, dice_2


def plot(orginImage, originMask, pre1, pre2, originImage=None):
    figure, ax = plt.subplots(nrows=1, ncols=4, figsize=(20, 20))
    ax[0].imshow(originImage[:, :, :], cmap='gray')
    ax[1].imshow(originMask, cmap='gray')
    ax[2].imshow(pre1.swapaxes(0, 2), cmap='gray')
    ax[3].imshow(pre2.swapaxes(0, 2), cmap='gray')
    figure.tight_layout()
    figure.show()


ct_path = '/home/ps/image segmentation/sample/3Diradb/Image'  # 需要预测的CT图像
mask_path = '/home/ps/image segmentation/sample/3Diradb/Mask'  # 需要预测的CT图像标签，如果要在线提交codelab，需要先得到预测过的70例肝脏标签
jpg_path = '/home/ps/image segmentation/sample/3Diradb/D_SNAU/'
ct_files = os.listdir(ct_path)
mask_files = os.listdir(mask_path)
import cv2 as cv2


# 定义函数将图像预处理为模型所需的尺寸

def main():
    args = parse_args()
    # args.dataset = "datasets"

    if args.name is None:
        if args.deepsupervision:
            args.name = '%s_%s_lym' % (args.dataset, args.arch)
        else:
            args.name = '%s_%s_lym' % (args.dataset, args.arch)
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if not os.path.exists('models/{}/{}'.format(args.name, timestamp)):
        os.makedirs('models/{}/{}'.format(args.name, timestamp))

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' % (arg, getattr(args, arg)))
    print('------------')

    with open('models/{}/{}/args.txt'.format(args.name, timestamp), 'w') as f:
        for arg in vars(args):
            print('%s: %s' % (arg, getattr(args, arg)), file=f)

    joblib.dump(args, 'models/{}/{}/args.pkl'.format(args.name, timestamp))

    # define loss function (criterion)
    if args.loss == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.BCEDiceLoss().cuda()

    cudnn.benchmark = True

    config = get_config(args)
    model = lcab_two(config, img_size=224, num_classes=2)
    model.load_from(config)
    print(count_params(model))
    model.load_state_dict(
        torch.load(
            '/home/ps/desktop/hjn/image segmentation/models/Lits_tumor_LCAB_three_lym/2024-01-14-21-51-33/epoch35-0.9557-0.7622_model.pth'))

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224),
    ])

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

    for i in range(len(ct_files)):
        print(i, ct_files[i])
        originImage = np.load(os.path.join(ct_path, ct_files[i]))
        originMask = np.load(os.path.join(mask_path, mask_files[i]))

        image = cv2.resize(originImage, (224, 224))
        mask = cv2.resize(originMask, (224, 224))

        dst = Image.fromarray(mask, 'P')
        bin_colormap = [0, 0, 0] + [255, 0, 0] + [0, 255, 0] + [0, 0, 0] * 252  # 二值调色板
        dst.putpalette(bin_colormap)
        dst.save(jpg_path + str(i) + '_Mask.png')
        cv2.imwrite(jpg_path + str(i) + '_Image.png', image * 255)

        liver_label = originMask.copy()
        liver_label[originMask == 2] = 1
        liver_label[originMask == 1] = 1

        tumor_label = originMask.copy()
        tumor_label[originMask == 1] = 0
        tumor_label[originMask == 2] = 1

        nplabel = np.empty((448, 448, 2))
        nplabel[:, :, 0] = liver_label
        nplabel[:, :, 1] = tumor_label

        nplabel = nplabel.astype("float32")
        npimage = originImage.astype("float32")

        npimage = trans(npimage)
        nplabel = trans(nplabel)

        pre, dice_1, dice_2 = predict(model, npimage, nplabel)
        print(f'liver_dice:{dice_1}, tumor_dice:{dice_2}')

        dst = Image.fromarray(pre.swapaxes(0, 2).squeeze(2), 'P')
        bin_colormap = [0, 0, 0] + [255, 0, 0] + [0, 255, 0] + [0, 0, 0] * 252  # 二值调色板
        dst.putpalette(bin_colormap)
        dst.save(jpg_path + str(i) + '_' + '.png')

    print("finish!")


torch.cuda.empty_cache()

if __name__ == '__main__':
    main()