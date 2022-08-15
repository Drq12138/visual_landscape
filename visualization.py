import enum
from html.entities import html5
from importlib.resources import path
from time import time
from mpl_toolkits.mplot3d import Axes3D
import os
from operator import mod
from statistics import mode
import xdrlib
from utils import create_random_direction, get_weights, eval_loss,test, h5_to_vtp, get_coefs, create_pca_direction
import numpy as np
import torch
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib import cm
from tqdm import tqdm
import copy




def get_direction(model, direction_type, filesnames, weight_type):
    if direction_type == 'random':
        xdirection = create_random_direction(model, weight_type)
        ydirection = create_random_direction(model, weight_type)
    elif direction_type == 'pca':
        xdirection, ydirection = create_pca_direction(model, weight_type,filesnames)


    # if direction_type == 'random':
    #     xdirection = create_random_direction(model, weight)
    #     ydirection = create_random_direction(model, weight)
    # elif direction_type == 'pca':
    #     xdirection, ydirection = create_pca_direction(model, weight,filesnames)
    # # direction = create_random_direction(model, weight)
    # # xdirection, ydirection = create_pca_direction(model, weight,filesnames)



    return [xdirection, ydirection]


def plot_contour_trajectory(val, xcoord_mesh, ycoord_mesh, name, vmin=0.25, vmax=0.29, vlevel=0.003):
    fig = plt.figure()
    CS = plt.contour(xcoord_mesh, ycoord_mesh, val, cmap='summer', levels=np.arange(vmin, vmax, vlevel))
    plt.clabel(CS, inline=1, fontsize=8)
    fig.savefig(name +'_2dcontour' + '.pdf', dpi=300, bbox_inches='tight', format='pdf')

    fig = plt.figure()
    CS = plt.contourf(xcoord_mesh, ycoord_mesh, val, cmap='summer', levels=np.arange(vmin, vmax, vlevel))
    fig.savefig(name +'_2dcontourf' + '.pdf', dpi=300, bbox_inches='tight', format='pdf')

    fig = plt.figure()
    sns_plot = sns.heatmap(val, cmap='viridis', cbar=True, vmin=vmin, vmax=vmax,
                           xticklabels=False, yticklabels=False)
    sns_plot.invert_yaxis()
    sns_plot.get_figure().savefig(name +'_2dheat.pdf', dpi=300, bbox_inches='tight', format='pdf')



    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_surface(xcoord_mesh, ycoord_mesh, val, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.savefig(name +'_3dsurface.pdf', dpi=300,
                bbox_inches='tight', format='pdf')
    
    # h5_to_vtp(val, xcoord_mesh, ycoord_mesh, name, zmax=100, log=True)
    pass


def plot(model,weight,filesnames,dataloader,direction_type,criterion, save_path, N=50, plot_ratio=0.5, direction_path = ''):
    if direction_path:
        print('use saved direction and weight')
        direction_load = torch.load(direction_path)
        direction = direction_load["direction"]
        weight = direction_load["weight"]
    else:
        direction = get_direction(model,weight, direction_type, filesnames)
        torch.save({"direction":direction, "weight":weight}, os.path.join(save_path, 'direction.pt') )
    coefs, path_loss, path_acc = get_coefs(model, weight, filesnames, direction, dataloader,criterion)
    coefs_x = coefs[:,0][np.newaxis]
    coefs_y = coefs[:,1][np.newaxis]

    boundaries_x = max(coefs_x[0])-min(coefs_x[0])
    boundaries_y = max(coefs_y[0])-min(coefs_y[0])

    xcoordinates = np.linspace(min(coefs_x[0])-plot_ratio*boundaries_x, max(coefs_x[0])+plot_ratio*boundaries_x, N)
    ycoordinates = np.linspace(min(coefs_y[0])-plot_ratio*boundaries_y, max(coefs_y[0])+plot_ratio*boundaries_y, N)
    # xcoordinates = np.linspace(-0.5, 0.5, N)
    # ycoordinates = np.linspace(-0.5, 0.5, N)
    shape = (len(xcoordinates),len(ycoordinates))
    losses = -np.ones(shape=shape)
    accuracies = -np.ones(shape=shape)
    inds = np.array(range(losses.size))
    xcoord_mesh, ycoord_mesh = np.meshgrid(xcoordinates, ycoordinates)
    s1 = xcoord_mesh.ravel()
    s2 = ycoord_mesh.ravel()
    coords = np.c_[s1,s2]

    print('begin cal')
    # ---------------get landscape data ----------------------------
    for count, ind in enumerate(tqdm(inds)):
        coord = coords[count]
        dx = direction[0]
        dy = direction[1]
        changes = [d0*coord[0] + d1*coord[1] for (d0, d1) in zip(dx, dy)]
        for (p, w, d) in zip(model.parameters(), weight, changes):
            p.data = w + d.type_as(w)
            # p.data = w
        # temp_weight = get_weights(model)
        stime = time()
        acc, loss = test(model, dataloader, criterion)
        # print('cost: ', stime-time())
        losses.ravel()[ind] = loss
        accuracies.ravel()[ind] = acc
        # loss, acc = eval_loss(model, criterion, dataloader)
    
    # ---------------------get path data ----------------------------
    np.savez(os.path.join(save_path, 'save_coor_val_ori.npz'), losses = losses, accuracies = accuracies, xcoord_mesh = xcoord_mesh, ycoord_mesh = ycoord_mesh,coefs_x = coefs_x, coefs_y = coefs_y, path_loss = path_loss, path_acc = path_acc)

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




def plot_mult(model, weight, state, filesnames_1, filesnames_2, dataloader, criterion, save_path, N=50, plot_ratio=0.5, direction_path = ''):
    direction_load = torch.load(direction_path)
    direction = direction_load["direction"]
    weight_type = direction_load["weigth_type"]

    # get path: -----------------

    coefs_1, path_loss_1, path_acc_1 = get_coefs(model, weight, state, filesnames_1, direction, dataloader,criterion, weight_type)
    coefs_x_1 = coefs_1[:,0][np.newaxis]
    coefs_y_1 = coefs_1[:,1][np.newaxis]

    coefs_2, path_loss_2, path_acc_2 = get_coefs(model, weight, state, filesnames_2, direction, dataloader,criterion, weight_type)
    coefs_x_2 = coefs_2[:,0][np.newaxis]
    coefs_y_2 = coefs_2[:,1][np.newaxis]

    boundaries_x = max(max(coefs_x_1[0]), max(coefs_x_2[0]))-min(min(coefs_x_1[0]), min(coefs_x_2[0]))
    boundaries_y = max(max(coefs_y_1[0]), max(coefs_y_2[0]))-min(min(coefs_y_1[0]), min(coefs_y_2[0]))

    xcoordinates = np.linspace(min(min(coefs_x_1[0]), min(coefs_x_2[0]))-plot_ratio*boundaries_x, max(max(coefs_x_1[0]), max(coefs_x_2[0]))+plot_ratio*boundaries_x, N)
    ycoordinates = np.linspace(min(min(coefs_y_1[0]), min(coefs_y_2[0]))-plot_ratio*boundaries_y, max(max(coefs_y_1[0]), max(coefs_y_2[0]))+plot_ratio*boundaries_y, N)

    shape = (len(xcoordinates),len(ycoordinates))
    losses = -np.ones(shape=shape)
    accuracies = -np.ones(shape=shape)
    inds = np.array(range(losses.size))
    xcoord_mesh, ycoord_mesh = np.meshgrid(xcoordinates, ycoordinates)
    s1 = xcoord_mesh.ravel()
    s2 = ycoord_mesh.ravel()
    coords = np.c_[s1,s2]

    print('begin cal surface ---------------------')
    # ---------------get landscape data ----------------------------
    for count, ind in enumerate(tqdm(inds)):
        coord = coords[count]
        dx = direction[0]
        dy = direction[1]
        changes = [d0*coord[0] + d1*coord[1] for (d0, d1) in zip(dx, dy)]
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
    np.savez(os.path.join(save_path, 'save_coor_val.npz'), losses = losses, accuracies = accuracies, xcoord_mesh = xcoord_mesh, ycoord_mesh = ycoord_mesh,coefs_x_1 = coefs_x_1, coefs_y_1 = coefs_y_1, path_loss_1 = path_loss_1, path_acc_1 = path_acc_1, coefs_x_2 = coefs_x_2, coefs_y_2 = coefs_y_2, path_loss_2 = path_loss_2, path_acc_2 = path_acc_2)

    print('-------------')






    pass

