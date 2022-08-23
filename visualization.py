import enum
from time import time
from mpl_toolkits.mplot3d import Axes3D
import os
from utils import create_random_direction, get_weight_list, cal_path, create_pca_direction, get_delta, \
    decomposition_delta, set_weigth, back_tracking_line_search
import numpy as np
import torch
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib import cm
from tqdm import tqdm
import copy


def get_direction_list(model, direction_type, filesnames, weight_type):
    """
    given the model, return the list of direction based on weight or state, using pca or random method
    :param model:
    :param direction_type:
    :param filesnames:
    :param weight_type:
    :return: list:[dx, dy]
    """
    weight_list = get_weight_list(model)
    if direction_type == 'random':
        xdirection_list = create_random_direction(weight_list, weight_type)
        ydirection_list = create_random_direction(weight_list, weight_type)
    elif direction_type == 'pca':
        xdirection_list, ydirection_list = create_pca_direction(model, weight_list, weight_type, filesnames)

    return [xdirection_list, ydirection_list]


def plot_contour_trajectory(val, xcoord_mesh, ycoord_mesh, name, vmin=0.25, vmax=0.29, vlevel=0.003):
    fig = plt.figure()
    CS = plt.contour(xcoord_mesh, ycoord_mesh, val, cmap='summer', levels=np.arange(vmin, vmax, vlevel))
    plt.clabel(CS, inline=1, fontsize=8)
    fig.savefig(name + '_2dcontour' + '.pdf', dpi=300, bbox_inches='tight', format='pdf')

    fig = plt.figure()
    CS = plt.contourf(xcoord_mesh, ycoord_mesh, val, cmap='summer', levels=np.arange(vmin, vmax, vlevel))
    fig.savefig(name + '_2dcontourf' + '.pdf', dpi=300, bbox_inches='tight', format='pdf')

    fig = plt.figure()
    sns_plot = sns.heatmap(val, cmap='viridis', cbar=True, vmin=vmin, vmax=vmax,
                           xticklabels=False, yticklabels=False)
    sns_plot.invert_yaxis()
    sns_plot.get_figure().savefig(name + '_2dheat.pdf', dpi=300, bbox_inches='tight', format='pdf')

    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_surface(xcoord_mesh, ycoord_mesh, val, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.savefig(name + '_3dsurface.pdf', dpi=300,
                bbox_inches='tight', format='pdf')

    pass


def plot_path(model, origin_weight_list, filesnames, dataloader, criterion, args):
    direction_load = torch.load(args.direction_path)
    direction = direction_load["direction"]
    weight_type = direction_load["weigth_type"]
    coefs, path_loss, path_acc = cal_path(model, origin_weight_list, filesnames, direction, dataloader, criterion)

    coefs_x = coefs[:, 0][np.newaxis]
    coefs_y = coefs[:, 1][np.newaxis]

    return coefs_x, coefs_y, path_loss, path_acc, direction


def plot_landscape(model, origin_weight_list, dataloader, criterion, xcoordinates, ycoordinates, direction, args):
    shape = (len(xcoordinates), len(ycoordinates))
    losses = -np.ones(shape=shape)
    accuracies = -np.ones(shape=shape)
    search_count_sum = []
    inds = np.array(range(losses.size))
    xcoord_mesh, ycoord_mesh = np.meshgrid(xcoordinates, ycoordinates)
    s1 = xcoord_mesh.ravel()
    s2 = ycoord_mesh.ravel()
    coords = np.c_[s1, s2]

    print('begin cal')
    # --------------- get landscape data ----------------------------
    for count, ind in enumerate(tqdm(inds)):
        coord = coords[count]
        dx = direction[0]
        dy = direction[1]
        change_weight_list = [w + d0 * coord[0] + d1 * coord[1] for (w, d0, d1) in zip(origin_weight_list, dx, dy)]
        set_weigth(model, change_weight_list)
        delta_direction, temp_loss = get_delta(model, dataloader, criterion)  # [batch_num, param_num]
        delta_direction_vector = delta_direction[:3, :].mean(0)
        search_direction_vector = decomposition_delta(delta_direction_vector, change_weight_list, origin_weight_list)

        step_t, final_loss, new_weight_list, search_count = back_tracking_line_search(model, dataloader, criterion,
                                                                                      change_weight_list, temp_loss,
                                                                                      search_direction_vector,
                                                                                      delta_direction_vector)
        losses.ravel()[ind] = final_loss
        search_count_sum.append(search_count)
    return losses, accuracies, xcoord_mesh, ycoord_mesh, search_count_sum


'''
def plot(model, filesnames, dataloader, criterion, save_path, args, fix_coordinate=False):
    origin_weight_list = get_weight_list(model)
    origin_state = copy.deepcopy(model.state_dict())
    direction_load = torch.load(args.direction_path)
    direction = direction_load["direction"]
    weight_type = direction_load["weigth_type"]
    if fix_coordinate:
        xcoordinates = np.linspace(-0.5, 0.5, args.plot_num)
        ycoordinates = np.linspace(-0.5, 0.5, args.plot_num)
    else:
        coefs, path_loss, path_acc = plot_path(model,origin_weight_list,filesnames,direction,dataloader,criterion)

        coefs_x = coefs[:, 0][np.newaxis]
        coefs_y = coefs[:, 1][np.newaxis]

        boundaries_x = max(coefs_x[0]) - min(coefs_x[0])
        boundaries_y = max(coefs_y[0]) - min(coefs_y[0])

        xcoordinates = np.linspace(min(coefs_x[0]) - args.plot_ratio * boundaries_x,
                                   max(coefs_x[0]) + args.plot_ratio * boundaries_x, args.plot_num)
        ycoordinates = np.linspace(min(coefs_y[0]) - args.plot_ratio * boundaries_y,
                                   max(coefs_y[0]) + args.plot_ratio * boundaries_y, args.plot_num)

    shape = (len(xcoordinates), len(ycoordinates))
    losses = -np.ones(shape=shape)
    accuracies = -np.ones(shape=shape)
    inds = np.array(range(losses.size))
    xcoord_mesh, ycoord_mesh = np.meshgrid(xcoordinates, ycoordinates)
    s1 = xcoord_mesh.ravel()
    s2 = ycoord_mesh.ravel()
    coords = np.c_[s1, s2]

    print('begin cal')
    # --------------- get landscape data ----------------------------
    for count, ind in enumerate(tqdm(inds)):
        coord = coords[count]
        dx = direction[0]
        dy = direction[1]
        changes = [d0 * coord[0] + d1 * coord[1] for (d0, d1) in zip(dx, dy)]
        if weight_type == 'weight':
            for (p, w, d) in zip(model.parameters(), weight, changes):
                p.data = w + d.type_as(w)
            # p.data = w
        elif weight_type == 'state':
            new_state = copy.deepcopy(state)
            for (k, v), d in zip(new_state.items(), changes):
                d = d.clone().detach()
                v.add_(d.type(v.type()))
            model.load_state_dict(new_state)
        # --------update all the weight or state --------------------
        stime = time()
        delta_direction = get_delta(model, dataloader, criterion)
        decomposition_delta(delta_direction,model, weight)
        acc, loss = test(model, dataloader, criterion)
        # print('cost: ', stime-time())
        losses.ravel()[ind] = loss
        accuracies.ravel()[ind] = acc
        # loss, acc = eval_loss(model, criterion, dataloader)

    # ---------------------get path data ----------------------------
    np.savez(os.path.join(save_path, 'save_coor_val_ori.npz'), losses=losses, accuracies=accuracies,
             xcoord_mesh=xcoord_mesh, ycoord_mesh=ycoord_mesh, coefs_x=coefs_x, coefs_y=coefs_y, path_loss=path_loss,
             path_acc=path_acc)

    # np.save(os.path.join(save_path, 'losses.npy'), losses)
    # np.save(os.path.join(save_path, 'accuracies.npy'), accuracies)
    # np.save(os.path.join(save_path, 'xcoord_mesh.npy'), xcoord_mesh)
    # np.save(os.path.join(save_path, 'ycoord_mesh.npy'), ycoord_mesh)
    # np.save(os.path.join(save_path, 'coefs_x.npy'), coefs_x)
    # np.save(os.path.join(save_path, 'coefs_y.npy'), coefs_y)
    # np.save(os.path.join(save_path, 'coefs_y.npy'), coefs_y)
    # np.save(os.path.join(save_path, 'coefs_y.npy'), coefs_y)
    print('-------------')

    # plot_contour_trajectory(losses, xcoord_mesh, ycoord_mesh, 'loss')
    # plot_contour_trajectory(accuracies, xcoord_mesh, ycoord_mesh,'acc')
    # h5_to_vtp(losses, xcoord_mesh, ycoord_mesh, 'surface',save_path, zmax=-1, interp=-1)
    # h5_to_vtp(path_loss, coefs_x, coefs_y, 'path',save_path, zmax=-1, interp=-1, show_points=True, show_polys=False)


def plot_mult(model, weight, state, filesnames_1, filesnames_2, dataloader, criterion, save_path, N=50, plot_ratio=0.5,
              direction_path=''):
    direction_load = torch.load(direction_path)
    direction = direction_load["direction"]
    weight_type = direction_load["weigth_type"]

    # get path: -----------------

    coefs_1, path_loss_1, path_acc_1 = get_coefs(model, weight, state, filesnames_1, direction, dataloader, criterion,
                                                 weight_type)
    coefs_x_1 = coefs_1[:, 0][np.newaxis]
    coefs_y_1 = coefs_1[:, 1][np.newaxis]

    coefs_2, path_loss_2, path_acc_2 = get_coefs(model, weight, state, filesnames_2, direction, dataloader, criterion,
                                                 weight_type)
    coefs_x_2 = coefs_2[:, 0][np.newaxis]
    coefs_y_2 = coefs_2[:, 1][np.newaxis]

    boundaries_x = max(max(coefs_x_1[0]), max(coefs_x_2[0])) - min(min(coefs_x_1[0]), min(coefs_x_2[0]))
    boundaries_y = max(max(coefs_y_1[0]), max(coefs_y_2[0])) - min(min(coefs_y_1[0]), min(coefs_y_2[0]))

    xcoordinates = np.linspace(min(min(coefs_x_1[0]), min(coefs_x_2[0])) - plot_ratio * boundaries_x,
                               max(max(coefs_x_1[0]), max(coefs_x_2[0])) + plot_ratio * boundaries_x, N)
    ycoordinates = np.linspace(min(min(coefs_y_1[0]), min(coefs_y_2[0])) - plot_ratio * boundaries_y,
                               max(max(coefs_y_1[0]), max(coefs_y_2[0])) + plot_ratio * boundaries_y, N)

    shape = (len(xcoordinates), len(ycoordinates))
    losses = -np.ones(shape=shape)
    accuracies = -np.ones(shape=shape)
    inds = np.array(range(losses.size))
    xcoord_mesh, ycoord_mesh = np.meshgrid(xcoordinates, ycoordinates)
    s1 = xcoord_mesh.ravel()
    s2 = ycoord_mesh.ravel()
    coords = np.c_[s1, s2]

    print('begin cal surface ---------------------')
    # ---------------get landscape data ----------------------------
    for count, ind in enumerate(tqdm(inds)):
        coord = coords[count]
        dx = direction[0]
        dy = direction[1]
        changes = [d0 * coord[0] + d1 * coord[1] for (d0, d1) in zip(dx, dy)]
        if weight_type == 'weight':
            for (p, w, d) in zip(model.parameters(), weight, changes):
                p.data = w + d.type_as(w)
            # p.data = w
        elif weight_type == 'state':
            new_state = copy.deepcopy(state)
            for (k, v), d in zip(new_state.items(), changes):
                d = d.clone().detach()
                v.add_(d.type(v.type()))
            model.load_state_dict(new_state)
        # temp_weight = get_weights(model)

        stime = time()
        acc, loss = test(model, dataloader, criterion)
        # print('cost: ', stime-time())
        losses.ravel()[ind] = loss
        accuracies.ravel()[ind] = acc
        # loss, acc = eval_loss(model, criterion, dataloader)

    # ---------------------get path data ----------------------------
    np.savez(os.path.join(save_path, 'save_coor_val.npz'), losses=losses, accuracies=accuracies,
             xcoord_mesh=xcoord_mesh, ycoord_mesh=ycoord_mesh, coefs_x_1=coefs_x_1, coefs_y_1=coefs_y_1,
             path_loss_1=path_loss_1, path_acc_1=path_acc_1, coefs_x_2=coefs_x_2, coefs_y_2=coefs_y_2,
             path_loss_2=path_loss_2, path_acc_2=path_acc_2)

    print('-------------')

    pass
'''
