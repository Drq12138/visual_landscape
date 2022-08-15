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
from visualization import plot

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


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

    #-------------resume------------------------
    if args.load_path:
        if os.path.isfile(args.load_path):
            print("=> loading checkpoint '{}'".format(args.load_path))
            checkpoint = torch.load(args.load_path)
            model.load_state_dict(checkpoint['state_dict'])
    
    torch.backends.cudnn.benchmark = True

    # ---------------optimizer-------------------------
    criterion = nn.CrossEntropyLoss().cuda()
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=args.weight_decay)
    if args.datasets == 'CIFAR10':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150])
                                                            
    elif args.datasets == 'CIFAR100':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150])
    
    if args.arch in ['resnet1202', 'resnet110']:
        # for resnet1202 original paper uses lr=0.01 for first 400 minibatches for warm-up
        # then switch back. In this setup it will correspond for first epoch.
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr*0.1
    

    # ---------------------train path -----------------------------
    save_path = os.path.join(args.save_dir, args.name)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    final_checkpoint = os.path.join(save_path, 'save_net_' + args.arch + '_' + str(args.epoch).zfill(len(str(args.epoch))) + '.pt')
    if os.path.isfile(final_checkpoint):
        # checkpoint = torch.load(final_checkpoint)
        # model.load_state_dict(checkpoint['state_dict'])
        print('you have trained before ...')
    else:
        torch.save({'epoch': 0, 'state_dict': model.state_dict()}, os.path.join(save_path, 'save_net_' + args.arch + '_' + str(0).zfill(len(str(args.epoch))) + '.pt'))
        orig_train_loss = []
        orig_test_acc = []
        origin_test_loss = []
        for epoch in range(args.epoch):
            print("Epoch %i"%(epoch))
            
            tloss = train_net(model, train_loader, optimizer, criterion, epoch)
            # lr_scheduler.step()
            orig_train_loss.append(tloss)
            accu, loss = test(model, val_loader, criterion)
            orig_test_acc.append(accu)
            origin_test_loss.append(loss)
            print(accu)
            torch.save({'epoch': epoch+1, 'state_dict': model.state_dict()}, os.path.join(save_path, 'save_net_' + args.arch + '_' + str(epoch+1).zfill(len(str(args.epoch)))+ '.pt'))
            
        np.savez(os.path.join(save_path, 'save_net_' + args.arch + '_orig.npz' ), orig_train_loss = orig_train_loss, origin_test_loss= origin_test_loss, origin_test_acc = orig_test_acc)
        # np.save(os.path.join(save_path, 'save_net_' + args.arch + '_orig_test_acc.npy' ), orig_test_acc)

    weight = get_weights(model)
    num = 0
    for i in weight:
        num+=i.numel()
    fileindices=np.linspace(0,args.epoch,args.epoch + 1)
    filesnames = [save_path + '/save_net_' + args.arch + '_' +str(int(i)).zfill(len(str(args.epoch)))+ '.pt' for i in fileindices]


    args.xmin, args.xmax, args.xnum = [float(a) for a in args.x.split(':')]
    args.xnum = int(args.xnum)
    args.ymin, args.ymax, args.ynum = [float(a) for a in args.y.split(':')]
    args.ynum = int(args.ynum)


    # xcoordinates = np.linspace(args.xmin, args.xmax, num=args.xnum)
    # ycoordinates = np.linspace(args.ymin, args.ymax, num=args.ynum)
    # ----------------------------
    # get the file names during training process

    print('begin plot')


    plot(model,weight,filesnames,train_loader,args.direction_type ,criterion,save_path, args.plot_num, args.plot_ratio, args.direction_path)












if __name__ == "__main__":
    main()