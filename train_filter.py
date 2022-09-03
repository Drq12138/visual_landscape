import argparse
import os

from utils import get_model, set_seed, get_datasets, AverageMeter, accuracy, test, get_model_grad_list, \
    normalize_directions_for_weights, get_weight_list, update_grad
import torch
import torch.nn as nn
import numpy as np
from visualization import get_direction_list

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def train(model, train_loader, optimizer, criterion, epoch, args):
    model.train()
    losses = AverageMeter()

    for batch_id, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        if args.use_filter:
            weight_list = get_weight_list(model)
            grad_list = get_model_grad_list(model)
            normalize_directions_for_weights(grad_list, weight_list)
            update_grad(model, grad_list)

        optimizer.step()

        losses.update(loss.item(), data.size(0))

    return losses.avg


def main():
    parser = argparse.ArgumentParser(description='test if the filter can be used in training process')
    parser.add_argument('--arch', default='resnet20')
    parser.add_argument('--datasets', default='CIFAR10')
    parser.add_argument('--workers', default=12, type=int)
    parser.add_argument('--randomseed', default=1, type=int)
    parser.add_argument('--batch_size', default=2048, type=int)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--smalldatasets', default=None, type=float)
    parser.add_argument('--mult_gpu', action='store_true')
    parser.add_argument('--use_filter', action='store_true')
    parser.add_argument('--lr', default=0.04, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--optimizer', default='sgd')
    parser.add_argument('--save_dir', default='./../checkpoints_0820/visualization')
    parser.add_argument('--name', default='test_visualization')
    parser.add_argument('--load_path', default='')

    args = parser.parse_args()
    print(args)

    set_seed(args.randomseed)
    # --------- dataset---------------------
    train_loader, val_loader = get_datasets(args)
    print(len(train_loader.dataset))
    print(len(val_loader.dataset))

    # -----------model--------------------------
    model = get_model(args)
    if args.mult_gpu:
        model = torch.nn.DataParallel(model)
    model.cuda()

    # -------------resume------------------------
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
            param_group['lr'] = args.lr * 0.1

    # ---------------------train path -----------------------------
    save_path = os.path.join(args.save_dir, args.name)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    torch.save({'epoch': 0, 'state_dict': model.state_dict()},
               os.path.join(save_path, 'save_net_' + args.arch + '_' + str(0).zfill(len(str(args.epoch))) + '.pt'))
    origin_train_loss = []
    origin_test_acc = []
    origin_test_loss = []
    for epoch in range(args.epoch):
        train_loss = train(model, train_loader, optimizer, criterion, epoch, args)
        lr_scheduler.step()
        origin_train_loss.append(train_loss)
        accu, loss = test(model, val_loader, criterion)
        origin_test_acc.append(accu)
        origin_test_loss.append(loss)
        print("train epoch {}/{}: train loss: {}, test loss: {}, test accuracy: {}".format(epoch + 1, args.epoch,
                                                                                           train_loss, loss, accu))
        torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict()},
                   os.path.join(save_path, 'save_net_' + args.arch + '_' +
                                str(epoch + 1).zfill(len(str(args.epoch))) + '.pt'))

    np.savez(os.path.join(save_path, 'save_net_' + args.arch + '_orig.npz'), origin_train_loss=origin_train_loss,
             origin_test_loss=origin_test_loss, origin_test_acc=origin_test_acc)


if __name__ == "__main__":
    main()
