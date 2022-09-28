import argparse
from utils import set_seed, get_datasets, get_model, flat_param, get_weight_list, plot_both_path
from visual_utils import update_grad
import torch
import numpy as np
import os
import torch.nn as nn

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def main():
    parser = argparse.ArgumentParser(description='can plot weight or state')
    parser.add_argument('--arch', default='resnet20')
    parser.add_argument('--datasets', default='CIFAR10')
    parser.add_argument('--load_dir', default='')
    parser.add_argument('--save_dir', default='./../checkpoints_0919/visualization')
    parser.add_argument('--name', default='test')
    parser.add_argument('---origin_direction', default='')
    parser.add_argument('--workers', default=8, type=int)
    parser.add_argument('--randomseed', default=1, type=int)
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--lr', default=0.004, type=float)
    parser.add_argument('--smalldatasets', default=None, type=float)
    parser.add_argument('--mult_gpu', action='store_true')

    args = parser.parse_args()
    print(args)

    set_seed(args.randomseed)
    # --------- dataset---------------------
    train_loader, val_loader = get_datasets(args)
    print(len(train_loader.dataset))
    print(len(val_loader.dataset))
    save_path = os.path.join(args.save_dir, args.name)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    # -----------model--------------------------
    model = get_model(args)
    # weight = get_weights(model)
    if args.mult_gpu:
        model = torch.nn.DataParallel(model)
    model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()

    if args.load_dir:
        origin_data_load = np.load(os.path.join(args.load_dir, 'save_net_resnet20_orig.npz'))
        final_checkpoint_load = torch.load(os.path.join(args.load_dir, 'save_net_resnet20_100.pt'))
        model.load_state_dict(final_checkpoint_load['state_dict'])
        final_weight_list = get_weight_list(model)

        weight_matrix = origin_data_load['origin_weight']
        origin_loss = origin_data_load['origin_train_loss']
        origin_acc = origin_data_load['origin_test_acc']
        pass
    if args.origin_direction:
        direction_load = torch.load(args.origin_direction)
        direction = direction_load["direction"]

    print(weight_matrix.shape)
    temp_dx, temp_dy = update_grad(model, weight_matrix, final_weight_list, origin_loss, direction, train_loader,
                                   criterion, args)
    # np.savez(os.path.join(save_path, 'save_temp_direction.npz'), temp_dx=temp_dx, temp_dy=temp_dy)
    torch.save({'temp_dx': temp_dx, 'temp_dy': temp_dy}, os.path.join(save_path, 'save_temp_direction.pt'))


    # path_coordinate, temp_loss, temp_acc, pro_loss, pro_acc = plot_both_path(model, weight_matrix, final_weight_list,
    #                                                                          direction, train_loader, criterion)
    # path_x = path_coordinate[:, 0][np.newaxis]
    # path_y = path_coordinate[:, 1][np.newaxis]
    # np.savez(os.path.join(save_path, 'save_path_val.npz'), temp_losses=temp_loss, temp_accuracies=temp_acc,
    #          xcoord_mesh=path_x, ycoord_mesh=path_y, pro_loss=pro_loss, pro_acc=pro_acc)

    for epoch in range(args.epoch):
        pass


if __name__ == "__main__":
    main()
