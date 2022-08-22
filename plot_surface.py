import argparse
import os
import copy
from utils import get_model, get_weight_list, set_seed, get_datasets, AverageMeter, accuracy, test, train_net
import torch
import torch.nn as nn
import numpy as np
from visualization import plot_path

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"


def main():
    parser = argparse.ArgumentParser(description='can plot weight or state')
    parser.add_argument('--arch', default='resnet20')
    parser.add_argument('--datasets', default='CIFAR10')
    # parser.add_argument('--weight_type', default='weight')
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--randomseed', default=1, type=int)
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--plot_num', default=50, type=int)
    parser.add_argument('--plot_ratio', default=0.5, type=float)
    parser.add_argument('--smalldatasets', default=None, type=float)
    parser.add_argument('--mult_gpu', action='store_true')
    parser.add_argument('--lr', default=0.04, type=float)

    parser.add_argument('--save_dir', default='./../checkpoints_0820/visualization')
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
    if args.load_path:
        print("=> loading checkpoint '{}'".format(args.load_path))
        checkpoint = torch.load(args.load_path)
        model.load_state_dict(checkpoint['state_dict'])
    origin_weight_list = get_weight_list(model)
    torch.backends.cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss().cuda()

    save_path = os.path.join(args.save_dir, args.name)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    fileindices = np.linspace(0, args.epoch, args.epoch + 1)

    load_plt_path_1 = os.path.join(args.save_dir, args.plt_path_one)
    filesnames_1 = [load_plt_path_1 + '/save_net_' + args.arch + '_' + str(int(i)).zfill(len(str(args.epoch))) + '.pt'
                    for i in fileindices]

    if args.plt_path_two:
        load_plt_path_2 = os.path.join(args.save_dir, args.plt_path_two)
        filesnames_2 = [
            load_plt_path_2 + '/save_net_' + args.arch + '_' + str(int(i)).zfill(len(str(args.epoch))) + '.pt' for i in
            fileindices]

    print('begin plot')
    coefs_x, coefs_y, path_loss, path_acc, direction = plot_path(model, origin_weight_list, filesnames_1, train_loader,
                                                                 criterion, args)
    np.savez(os.path.join(save_path, 'save_path_val.npz'), losses=path_loss, accuracies=path_acc,
             xcoord_mesh=coefs_x, ycoord_mesh=coefs_y)
    boundaries_x = max(coefs_x[0]) - min(coefs_x[0])
    boundaries_y = max(coefs_y[0]) - min(coefs_y[0])

    xcoordinates = np.linspace(min(coefs_x[0]) - args.plot_ratio * boundaries_x,
                               max(coefs_x[0]) + args.plot_ratio * boundaries_x, args.plot_num)
    ycoordinates = np.linspace(min(coefs_y[0]) - args.plot_ratio * boundaries_y,
                               max(coefs_y[0]) + args.plot_ratio * boundaries_y, args.plot_num)


if __name__ == "__main__":
    main()
