import argparse
from ast import arg
from enum import EnumMeta
import os
from statistics import mode

from numpy import save
from utils import get_model, get_weights, set_seed, get_datasets, AverageMeter, accuracy, test, train_net
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import numpy as np
from visualization import plot, plot_mult

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def main():
    parser = argparse.ArgumentParser(description='a single test to get familiar with the code')
    parser.add_argument('--arch', default='resnet20')
    parser.add_argument('--datasets', default='CIFAR10')
    parser.add_argument('--weight_type', default='weight')
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--randomseed', default=1, type=int)
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--plot_num', default=50, type=int)
    parser.add_argument('--plot_ratio', default=0.5, type=float)
    parser.add_argument('--smalldatasets', default=None, type=float)
    parser.add_argument('--mult_gpu', action='store_true')
    parser.add_argument('--lr', default=0.04, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--optimizer', default='sgd')
    parser.add_argument('--direction_type', default='pca')
    parser.add_argument('--save_dir', default='./../checkpoints/visualization')
    parser.add_argument('--name', default='mul_test')
    parser.add_argument('--load_path', default='')
    parser.add_argument('--plt_path_one', default='')
    parser.add_argument('--plt_path_two', default='')
    parser.add_argument('--direction_path', default='')
    parser.add_argument('--x', default='-1:1:51', help='A string with format xmin:x_max:xnum')
    parser.add_argument('--y', default='-1:1:51', help='A string with format ymin:ymax:ynum')

    args = parser.parse_args()
    print(args)

    set_seed(args.randomseed)
    # --------- dataset---------------------
    train_loader, val_loader = get_datasets(args)
    print(len(train_loader.dataset))
    print(len(val_loader.dataset))

    # -----------model--------------------------
    model = get_model(args)
    # weight = get_weights(model)
    if args.mult_gpu:
        model = torch.nn.DataParallel(model)
    model.cuda()

    # -------------resume------------------------
    print("=> loading checkpoint '{}'".format(args.load_path))
    checkpoint = torch.load(args.load_path)
    model.load_state_dict(checkpoint['state_dict'])
    origin_weight = get_weights(model)

    torch.backends.cudnn.benchmark = True

    # ---------------optimizer-------------------------
    criterion = nn.CrossEntropyLoss().cuda()

    # ---------------------train path -----------------------------
    save_path = os.path.join(args.save_dir, args.name)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    load_plt_path_1 = os.path.join(args.save_dir, args.plt_path_one)
    load_plt_path_2 = os.path.join(args.save_dir, args.plt_path_two)

    weight = get_weights(model)

    fileindices = np.linspace(0, args.epoch, args.epoch + 1)
    filesnames_1 = [load_plt_path_1 + '/save_net_' + args.arch + '_' + str(int(i)).zfill(len(str(args.epoch))) + '.pt'
                    for i in fileindices]
    filesnames_2 = [load_plt_path_2 + '/save_net_' + args.arch + '_' + str(int(i)).zfill(len(str(args.epoch))) + '.pt'
                    for i in fileindices]

    args.xmin, args.xmax, args.xnum = [float(a) for a in args.x.split(':')]
    args.xnum = int(args.xnum)
    args.ymin, args.ymax, args.ynum = [float(a) for a in args.y.split(':')]
    args.ynum = int(args.ynum)

    # xcoordinates = np.linspace(args.xmin, args.xmax, num=args.xnum)
    # ycoordinates = np.linspace(args.ymin, args.ymax, num=args.ynum)
    # ----------------------------
    # get the file names during training process

    print('begin plot')

    plot_mult(model, weight, filesnames_1, filesnames_2, train_loader, args.direction_type, criterion, save_path,
              args.plot_num, args.plot_ratio, args.direction_path)


if __name__ == "__main__":
    main()
