import argparse
from ast import arg
from enum import EnumMeta
from importlib.resources import path
import os
from statistics import mode

from numpy import save
from utils import get_model, get_weights, set_seed, get_datasets, AverageMeter, accuracy, test, train_net
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import numpy as np
from visualization import plot

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"


def main():
    parser = argparse.ArgumentParser(description='a single test to get familiar with the code')
    parser.add_argument('--arch', default='resnet20')
    parser.add_argument('--datasets', default='CIFAR10')
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
    parser.add_argument('--name', default='test_visualization')
    parser.add_argument('--load_path', default='')
    parser.add_argument('--x', default='-1:1:51', help='A string with format xmin:x_max:xnum')
    parser.add_argument('--y', default='-1:1:51', help='A string with format ymin:ymax:ynum')

    args = parser.parse_args()

    set_seed(args.randomseed)
    # --------- dataset---------------------
    train_loader, val_loader = get_datasets(args)
    print(len(train_loader.dataset))
    print(len(val_loader.dataset))

    # -----------model--------------------------
    model_1 = get_model(args)
    model_2 = get_model(args)
    # weight = get_weights(model)
    if args.mult_gpu:
        model_1 = torch.nn.DataParallel(model_1)
        model_2 = torch.nn.DataParallel(model_2)
    model_1.cuda()
    model_2.cuda()

    #-------------resume------------------------

    path_1 = '/home/DiskB/rqding/checkpoints/visualization/train_path_one_sgd/save_net_resnet20_000.pt'
    path_2 = '/home/DiskB/rqding/checkpoints/visualization/train_path_two_adam/save_net_resnet20_001.pt'
    checkpoint1 = torch.load(path_1)
    model_1.load_state_dict(checkpoint1['state_dict'])

    checkpoint2 = torch.load(path_2)
    model_2.load_state_dict(checkpoint2['state_dict'])
    
    torch.backends.cudnn.benchmark = True

    # ---------------optimizer-------------------------
    criterion = nn.CrossEntropyLoss().cuda()

    param1 = [p.data for p in model_1.parameters()]
    param2 = [p.data for p in model_2.parameters()]

    pass










if __name__ == "__main__":
    main()