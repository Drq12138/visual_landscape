import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib
from utils import h5_to_vtp
from visualization import plot_contour_trajectory
import os
import torch

name_list = ['train_path_two_adam', 'train_path_two_sgd']

''' 
    name = 'train_path_one_sgd'

    outs = np.load('/home/DiskB/rqding/checkpoints/visualization/grad_vis/test_minima_vis_pca.npz', allow_pickle=True)

    flag = outs["b"]
    outs = outs["a"]
    # plot_loss(outs[0][0],outs[0][1],outs[0][2],path_x=outs[1][0],path_y=outs[1][1],path_z=outs[1][2], height=10,degrees=50, filename='test_grad_vis', is_log=True)
    X,Y,Z = outs[0][0],outs[0][1],outs[0][2]
    path_x=outs[1][0]
    path_y=outs[1][1]
    path_z=outs[1][2]
    h5_to_vtp(path_z, path_x, path_y,'test_path_loss', '/home/DiskB/rqding/checkpoints/visualization/grad_vis/', zmax=-1, interp=-1, show_points=True, show_polys=False)
    h5_to_vtp(Z, X, Y, 'test_surface_loss', '/home/DiskB/rqding/checkpoints/visualization/grad_vis/',zmax=-1, interp=-1 )
'''

'''
for name in name_list:
    data = np.load(os.path.join('/home/DiskB/rqding/checkpoints/visualization/', name, 'save_coor_val_ori.npz') )
    # origin_loss = np.load('/home/DiskB/rqding/checkpoints/visualization/test_small/save_net_resnet20_orig_loss.npy')
    # origin_acc = np.load('/home/DiskB/rqding/checkpoints/visualization/test_small/save_net_resnet20_orig_acc.npy')
    losses = data["losses"]
    accuracies = data["accuracies"]
    xcoord_mesh = data["xcoord_mesh"]
    ycoord_mesh = data["ycoord_mesh"]

    coefs_x = data["coefs_x"]
    coefs_y = data["coefs_y"]
    path_loss = data["path_loss"]
    path_acc = data["path_acc"]

    h5_to_vtp(losses, xcoord_mesh, ycoord_mesh, name +'_loss_ori',os.path.join('/home/DiskB/rqding/checkpoints/visualization/', name),log=True,zmax=-1, interp=-1)
    h5_to_vtp(path_loss, coefs_x, coefs_y, name +'_path_ori',os.path.join('/home/DiskB/rqding/checkpoints/visualization/', name),log=True,zmax=-1, interp=-1, show_points=True, show_polys=False)

'''

def plot_result(result, x_coord,y_coord,args, datatype = 'origin'):
    if datatype == 'origin':
        [origin_accuracies, origin_losses] = result
        h5_to_vtp(origin_losses, x_coord, y_coord, args.name + '_origin_losses',
                  os.path.join('/home/DiskB/rqding/checkpoints_0820/visualization/', args.name), log=True, zmax=-1,
                  interp=-1)
    # elif datatype == 'back_result':
    #     [losses, accuracies, num] = result
    #     h5_to_vtp(losses, x_coord, y_coord, args.name + datatype + '_losses',
    #               os.path.join('/home/DiskB/rqding/checkpoints_0820/visualization/', args.name), log=True, zmax=-1,
    #               interp=-1)
    else:
        [accuracies, losses, num] = result
        h5_to_vtp(losses, x_coord, y_coord, args.name + datatype+'_losses',
                  os.path.join('/home/DiskB/rqding/checkpoints_0820/visualization/', args.name), log=True, zmax=-1,
                  interp=-1)


def main():
    parser = argparse.ArgumentParser(description='can plot based on saved data')
    parser.add_argument('--name', default='test_both_path')
    parser.add_argument('--base_dir', default='/home/DiskB/rqding/checkpoints_0820/visualization/')
    args = parser.parse_args()
    print(args)

    data = torch.load(os.path.join(args.base_dir, args.name, 'save_landscape_val.pt'))
    path_data = np.load(os.path.join(args.base_dir, args.name, 'save_path_val.npz'))

    origin_result = data['origin_result']
    back_result = data['back_result']
    forward_result = data['forward_result']
    x_coord_grid = data['x_coord_grid']
    y_coord_grid = data['y_coord_grid']

    plot_result(origin_result, x_coord_grid, y_coord_grid,args, 'origin')
    plot_result(back_result, x_coord_grid, y_coord_grid, args, 'back_result')
    plot_result(forward_result, x_coord_grid, y_coord_grid, args, 'forward_result')

    losses = path_data["temp_losses"]
    accuracies = path_data["temp_accuracies"]
    xcoord_mesh = path_data["xcoord_mesh"]
    ycoord_mesh = path_data["ycoord_mesh"]
    pro_loss = path_data["pro_loss"]
    pro_acc = path_data["pro_acc"]

    h5_to_vtp(losses, xcoord_mesh, ycoord_mesh, args.name + '_path_ori',
              os.path.join('/home/DiskB/rqding/checkpoints_0820/visualization/', args.name), log=True, zmax=-1, interp=-1,
              show_points=True, show_polys=False)
    h5_to_vtp(pro_loss, xcoord_mesh, ycoord_mesh, args.name + '_path_pro',
              os.path.join('/home/DiskB/rqding/checkpoints_0820/visualization/', args.name), log=True, zmax=-1, interp=-1,
              show_points=True, show_polys=False)


    #
    # h5_to_vtp(losses, xcoord_mesh, ycoord_mesh, args.name + '_loss',
    #           os.path.join('/home/DiskB/rqding/checkpoints_0820/visualization/', args.name), log=True, zmax=-1,
    #           interp=-1, show_points=True, show_polys=False)


if __name__ == "__main__":
    main()
