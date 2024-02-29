import importlib
import random

from skimage.io import imread
from torch.nn import DataParallel
import torch
import torch.nn as nn
import time
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms

import densenet
import loss
from dataset.dataset import Dataset
from model.swin_deeplab import SwinDeepLab
from net.SNAU_Net_run import lcab_two
from net.Swin_unet_two import SwinUnet_two
from net.U2Net_TWO import u2net_full

from utilities.metrics import dice_coef, batch_iou, mean_iou, iou_score
import utilities.losses as losses
from utilities.metrics import dice_coef, batch_iou, mean_iou, iou_score, rvd, voe
import utilities.losses as losses
from collections import OrderedDict
from utilities.utils import str2bool, count_params
from utilities.utils import str2bool, count_params
import pandas as pd
from net import Unet,res_unet_plus,R2Unet,sepnet,ResU_Net, Net_dilation, ResU_Net_two
from net.Swin_unet import SwinUnet
from net.swintransformer import swin_tiny_patch4_window7_224 as swintransformer
from config import get_config
from net.swinunet import SwinTransformerSys
#换模型需要修改的地方
from tqdm import tqdm
arch_names = list(Unet.__dict__.keys())
loss_names = list(losses.__dict__.keys())
loss_names.append('BCEWithLogitsLoss')
#%%
import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--deepsupervision', default=False,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--config_file', type=str,
                        default='swin_224_7_1level', help='config file name w/o suffix')
    # 换模型需要修改的地方
    parser.add_argument('--arch', '-a', metavar='ARCH', default='Swin_Net+LCAB',
                        # choices=arch_names,
                        help='model architecture: ' +
                             ' | '.join(arch_names) +
                             ' (default: NestedUNet)')
    # 换数据集需要修改的地方
    parser.add_argument('--dataset', default="Lits_tumor",
                        help='dataset name')
    parser.add_argument('--input-channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--image-ext', default='npy',
                        help='image file extension')
    parser.add_argument('--mask-ext', default='npy',
                        help='mask file extension')
    parser.add_argument('--aug', default=False, type=str2bool)
    parser.add_argument('--loss', default='Loss',
                        choices=loss_names,
                        help='loss: ' +
                             ' | '.join(loss_names) +
                             ' (default: BCEDiceLoss)')
    # 换模型需要修改的地方
    parser.add_argument('--epochs', default=150, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--early-stop', default=80, type=int,
                        metavar='N', help='early stopping (default: 30)')

    # 换模型需要修改的地方
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
        self.li = []

    def update(self, val, n=1):
        self.li.append(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def validate(val_loader, model):
    global liver_score, tumor_score
    voes_1 = AverageMeter()
    voes_2 = AverageMeter()
    dices_1s = AverageMeter()
    dices_2s = AverageMeter()
    rvds_1 = AverageMeter()
    rvds_2 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()

            output = model(input)
            voe_1, voe_2 = voe(output, target)
            dice_1, dice_2 = dice_coef(output, target)
            rvd_1, rvd_2 = rvd(output, target)

            voes_1.update(voe_1, input.size(0))
            voes_2.update(voe_2, input.size(0))
            dices_1s.update(dice_1, input.size(0))
            dices_2s.update(dice_2, input.size(0))
            rvds_1.update(rvd_1, input.size(0))
            rvds_2.update(rvd_2, input.size(0))

    liver_score = dices_1s.li
    tumor_score = dices_2s.li
    log = OrderedDict([
        ('voe_1', voes_1.avg),
        ('voe_2', voes_2.avg),
        ('rvd_1', rvds_1.avg),
        ('rvd_2', rvds_2.avg),
        ('dice_1', dices_1s.avg),
        ('dice_2', dices_2s.avg),
        ('voe_1_var', np.std(voes_1.li)),
        ('voe_2_var', np.std(voes_2.li)),
        ('rvd_1_var', np.std(rvds_1.li)),
        ('rvd_2_var', np.std(rvds_2.li)),
        ('dice_1_var', np.std(dices_1s.li)),
        ('dice_2_var', np.std(dices_2s.li)),
    ])

    return log

import os
import joblib
from glob import glob
import datetime


def main():

    args = parse_args()
    if args.name is None:
        if args.deepsupervision:
            args.name = '%s_%s_lym' %(args.dataset, args.arch)
        else:
            args.name = '%s_%s_lym' %(args.dataset, args.arch)
    timestamp  = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if not os.path.exists('models/{}/{}'.format(args.name,timestamp)):
        os.makedirs('models/{}/{}'.format(args.name,timestamp))

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    with open('models/{}/{}/args.txt'.format(args.name,timestamp), 'w') as f:
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)), file=f)

    joblib.dump(args, 'models/{}/{}/args.pkl'.format(args.name,timestamp))

    # define loss function (criterion)
    if args.loss == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.BCEDiceLoss().cuda()
    # if args.loss == 'BCEWithLogitsLoss':
    #     criterion = nn.BCEWithLogitsLoss().cuda()
    # else:
    #     criterion = loss.CombinedLoss().cuda()

    cudnn.benchmark = True

    config = get_config(args)  # 获取模型配置信息
    # model =SwinUnet_two(config, img_size=224, num_classes=2)
    # model = lcab_two(config, img_size=224, num_classes=2)
    # model.load_from(config)
    # model = u2net_full()
    # model = ResU_Net.U_Net(0)
    model = ResU_Net.NestedUNet(0)
    model = torch.nn.DataParallel(model).cuda()
    # liver_score = None
    # tumor_score = None
    # print(torch.cuda.is_available())
    # # model = torch.nn.DataParallel(model).cuda()

    # model_config = importlib.import_module(f'model.configs.{args.config_file}')
    # model = SwinDeepLab(
    #     model_config.EncoderConfig,
    #     model_config.ASPPConfig,
    #     model_config.DecoderConfig
    # ).cuda()
    #
    # if model_config.EncoderConfig.encoder_name == 'swin' and model_config.EncoderConfig.load_pretrained:
    #     model.encoder.load_from('/home/ps/desktop/hjn/image segmentation/pretrained_ckpt/swin_tiny_patch4_window7_224.pth')
    # if model_config.ASPPConfig.aspp_name == 'swin' and model_config.ASPPConfig.load_pretrained:
    #     model.aspp.load_from('/home/ps/desktop/hjn/image segmentation/pretrained_ckpt/swin_tiny_patch4_window7_224.pth')
    # if model_config.DecoderConfig.decoder_name == 'swin' and model_config.DecoderConfig.load_pretrained and not model_config.DecoderConfig.extended_load:
    #     model.decoder.load_from('/home/ps/desktop/hjn/image segmentation/pretrained_ckpt/swin_tiny_patch4_window7_224.pth')
    # if model_config.DecoderConfig.decoder_name == 'swin' and model_config.DecoderConfig.load_pretrained and model_config.DecoderConfig.extended_load:
    #     model.decoder.load_from_extended('/home/ps/desktop/hjn/image segmentation/pretrained_ckpt/swin_tiny_patch4_window7_224.pth')
    # print(count_params(model))

    if args.optimizer == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                              momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)


    model.load_state_dict(torch.load('/home/ps/desktop/hjn/image segmentation/models/Lits_tumor_U-net++_lym/2024-01-09-17-59-39/epoch26-0.9418-0.6970_model.pth'))

    # model.load_state_dict(torch.load('models/Lits_tumor_LCAB_lym/2024-01-14-11-29-12/epoch21-0.9570-0.7402_model.pth'))#Swin_run_lcab SwinUnetnat_lcab_two model =lcab_two(config, img_size=224, num_classes=2)# encoder:st,decoder:st+SFTB +LCAB

    # model.load_state_dict(torch.load('/home/ps/desktop/hjn/image segmentation/models/Lits_tumor_Swin_unet+NA_lym/2024-01-10-21-25-01/epoch37-0.9532-0.7458_model.pth'))#from net.swin_unet_aspp import SwinUnetnat_two# encoder:SFTB,decoder:st+SFTB
    # model.load_state_dict(torch.load('/home/ps/desktop/hjn/image segmentation/models/Lits_tumor_Swin_Net_LCAB_lym/2024-01-20-15-09-13/epoch24-0.9549-0.7169_model.pth'))#
    # model.load_state_dict(torch.load('/home/ps/desktop/hjn/image segmentation/models/Lits_tumor_DCCN_Net_lym/2024-01-18-18-33-35/epoch237-0.9217-0.7309_model.pth'))
    # model.load_state_dict(torch.load('/home/ps/desktop/hjn/image segmentation/models/Lits_tumor_Swin_unet+NA_lym/2024-01-11-12-01-09/epoch18-0.9537-0.7321_model.pth'))#from net.swin_unet_lcab import SwinUnetnat_lcab # encoder:st,decoder:st +LCAB
    # model.load_state_dict(torch.load('/home/ps/desktop/hjn/image segmentation/models/Lits_tumor_Swin_Net_LCAB_lym/2024-01-20-15-09-13/epoch24-0.9549-0.7169_model.pth'))
    # model.load_state_dict(torch.load(
        # '/home/ps/desktop/hjn/image segmentation/models/Lits_tumor_LCAB_three_lym/2024-01-14-21-51-33/epoch35-0.9557-0.7622_model.pth
    # # /home/ps/desktop/hjn/image segmentation/models/Lits_tumor_Swin_Net_LCAB_lym/2024-01-20-15-09-13/epoch24-0.9549-0.7169_model.pth
    # /home/ps/desktop/hjn/image segmentation/models/Lits_tumor_Swin_unet+NA_lym/2024-01-11-12-01-09/epoch18-0.9537-0.7321_model.pth

    # /home/ps/desktop/hjn/image segmentation/models/Lits_tumor_Swin_unet+NA_lym/2024-01-11-12-01-09/epoch18-0.9537-0.7321_model.pth
    print(count_params(model))
    #
    # val_img_paths = glob('./data/tumor/validImage/*')
    # val_mask_paths = glob('./data/tumor/validMask/*')
    val_img_paths = glob('./data/3Diradb/tumor/Image/*')
    val_mask_paths = glob('./data/3Diradb/tumor/Mask/*')

    val_dataset = Dataset(0, val_img_paths, val_mask_paths, transform=False)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        pin_memory=True,
        drop_last=False)

    log = pd.DataFrame(index=[], columns=[
        'dice_1', 'voe_1', 'rvd_1', 'dice_2', 'voe_2', 'rvd_2'
    ])

    first_time = time.time()

    val_log = validate(val_loader, model)

    # valid = []
    # for i in range(0, 10):
    #     val_log = validate(val_loader, model)
    #     valid.append(np.array([value for value in val_log.values()]))
    #     print(i)
    #
    # mean = np.mean(np.array(valid), axis=0)
    # std = np.std(np.array(valid), axis=0)
    # print("ok")

    # np.array([value for value in val_log.values()])
    print(
        'dice_1: %.4f+%.3f - voe_1: %.4f+%.3f - rvd_1: %.4f+%.3f - dice_2: %.4f+%.3f - voe_2: %.4f+%.3f - rvd_2: %.4f+%.3f'
        % (val_log['dice_1'], val_log['dice_1_var'], val_log['voe_1'], val_log['voe_1_var'], val_log['rvd_1'],
           val_log['rvd_1_var'],
           val_log['dice_2'], val_log['dice_2_var'], val_log['voe_2'], val_log['voe_2_var'], val_log['rvd_2'],
           val_log['rvd_2_var']))

    # print('loss %.4f - iou %.4f - dice %.4f ' %(train_log['loss'], train_log['iou'], train_log['dice']))
    end_time = time.time()
    print("time:", (end_time - first_time) / 60)

    torch.cuda.empty_cache()


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,2'
    main()





