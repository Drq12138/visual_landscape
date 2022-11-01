import argparse
import os
import copy
from utils import get_model, get_weight_list, set_seed, get_datasets, AverageMeter, accuracy, test, train_net, \
    plot_both_path, set_weight, flat_param
import torch
import torch.nn as nn
import numpy as np
from visualization import plot_path, plot_landscape

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def main():
    parser = argparse.ArgumentParser(description='can plot weight or state')
    parser.add_argument('--arch', default='resnet20')
    parser.add_argument('--datasets', default='CIFAR10')
    parser.add_argument('--plot_init', default='save_net_resnet20_100.pt')
    # parser.add_argument('--weight_type', default='weight')
    parser.add_argument('--workers', default=8, type=int)
    parser.add_argument('--randomseed', default=1, type=int)
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--plot_num', default=30, type=int)
    parser.add_argument('--plot_ratio', default=0.5, type=float)
    parser.add_argument('--smalldatasets', default=None, type=float)
    parser.add_argument('--mult_gpu', action='store_true')
    parser.add_argument('--fix_coor', action='store_true')
    parser.add_argument('--back_track_loss', action='store_true')
    parser.add_argument('--forward_search_loss', action='store_true')
    parser.add_argument('--lr', default=0.04, type=float)
    parser.add_argument('--project_point', default='')

    parser.add_argument('--save_dir', default='./../checkpoints_0919/visualization')
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
    print('train loader length: {}'.format(len(train_loader.dataset)))
    print('test loader length: {}'.format(len(val_loader.dataset)))

    # -----------model--------------------------
    model = get_model(args)
    # weight = get_weights(model)
    if args.mult_gpu:
        model = torch.nn.DataParallel(model)
        print('use multi GPU')
    model.cuda()

    # -------------resume------------------------
    if args.load_path:
        origin_data_load = np.load(os.path.join(args.load_path, 'save_net_resnet20_orig.npz'))
        print("=> loading checkpoint '{}'".format(args.load_path))
        checkpoint = torch.load(os.path.join(args.load_path, args.plot_init))
        model.load_state_dict(checkpoint['state_dict'])
        init_point_weight_list = get_weight_list(model)
        print("use checkpoint of '{}' as plot initial point".format(args.plot_init))

        weight_matrix = origin_data_load['origin_weight']
    # origin_weight_list = get_weight_list(model)
    torch.backends.cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss().cuda()

    save_path = os.path.join(args.save_dir, args.name)

    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    print("save all the result in path '{}'".format(save_path))

    file_index = np.linspace(0, args.epoch, args.epoch + 1)

    load_plt_path_1 = os.path.join(args.save_dir, args.plt_path_one)
    filenames_1 = [load_plt_path_1 + '/save_net_' + args.arch + '_' + str(int(i)).zfill(len(str(args.epoch))) + '.pt'
                   for i in file_index]

    if args.plt_path_two:
        load_plt_path_2 = os.path.join(args.save_dir, args.plt_path_two)
        filenames_2 = [
            load_plt_path_2 + '/save_net_' + args.arch + '_' + str(int(i)).zfill(len(str(args.epoch))) + '.pt' for i in
            file_index]

    direction_load = torch.load(args.direction_path)
    direction_list = direction_load["direction"]
    direction_tensors = [flat_param(direction_list[0]), flat_param(direction_list[0])]
    print("use direction file '{}'".format(args.direction_path))

    if args.fix_coor:
        print("use fix coordinate without calculate path")
        x_coordinate = np.linspace(-1, 1, args.plot_num)
        y_coordinate = np.linspace(-1, 1, args.plot_num)
    else:
        print("use coordinate based on calculated path, now begin calculate ...")
        path_coordinate, temp_loss, temp_acc, pro_loss, pro_acc = plot_both_path(model, weight_matrix,
                                                                                 init_point_weight_list,
                                                                                 direction_tensors, train_loader,
                                                                                 criterion)
        # path_x, path_y, path_loss, path_acc, direction = plot_path(model, origin_weight_list, filenames_1, train_loader,
        #                                                            criterion, args)
        path_x = path_coordinate[:, 0][np.newaxis]
        path_y = path_coordinate[:, 1][np.newaxis]
        np.savez(os.path.join(save_path, 'save_path_val.npz'), temp_losses=temp_loss, temp_accuracies=temp_acc,
                 xcoord_mesh=path_x, ycoord_mesh=path_y, pro_loss=pro_loss, pro_acc=pro_acc)
        print("save path file in '{}'".format(os.path.join(save_path, 'save_path_val.npz')))
        # np.savez(os.path.join(save_path, 'save_path_val.npz'), losses=temp_loss, accuracies=temp_acc,
        #          xcoord_mesh=path_x, ycoord_mesh=path_y)
        boundaries_x = max(path_x[0]) - min(path_x[0])
        boundaries_y = max(path_y[0]) - min(path_y[0])

        x_coordinate = np.linspace(min(path_x[0]) - args.plot_ratio * boundaries_x,
                                   max(path_x[0]) + args.plot_ratio * boundaries_x, args.plot_num)
        y_coordinate = np.linspace(min(path_y[0]) - args.plot_ratio * boundaries_y,
                                   max(path_y[0]) + args.plot_ratio * boundaries_y, args.plot_num)
    print("begin calculate loss landscape ...")
    origin_result, back_result, forward_result, [x_coord_grid, y_coord_grid] = plot_landscape(model, weight_matrix,
                                                                                              init_point_weight_list,
                                                                                              train_loader,
                                                                                              criterion,
                                                                                              x_coordinate,
                                                                                              y_coordinate,
                                                                                              direction_list,
                                                                                              args)
    torch.save({'origin_result': origin_result, 'back_result': back_result, 'forward_result': forward_result,
                'x_coord_grid': x_coord_grid, 'y_coord_grid': y_coord_grid},
               os.path.join(save_path, 'save_landscape_val.pt'))
    print("save landscape file in '{}'".format(os.path.join(save_path, 'save_landscape_val.pt')))

    # np.savez(os.path.join(save_path, 'save_landscape_val.npz'), origin_losses=origin_loss, new_loss=new_loss,
    #          accuracies=accuracies, xcoord_mesh=x_coord_grid, ycoord_mesh=y_coord_grid,
    #          search_count=search_count_sum)


if __name__ == "__main__":
    main()
