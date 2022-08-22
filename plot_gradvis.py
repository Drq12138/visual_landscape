import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib
from utils import h5_to_vtp
from visualization import plot_contour_trajectory
import os

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


def main():
    parser = argparse.ArgumentParser(description='can plot based on saved data')
    parser.add_argument('--name', default='plot_2')
    parser.add_argument('--base_dir', default='/home/DiskB/rqding/checkpoints_0820/visualization/')
    args = parser.parse_args()
    print(args)

    data = np.load(os.path.join(args.base_dir, args.name, 'save_path_val.npz'))

    losses = data["losses"]
    accuracies = data["accuracies"]
    xcoord_mesh = data["xcoord_mesh"]
    ycoord_mesh = data["ycoord_mesh"]

    h5_to_vtp(losses, xcoord_mesh, ycoord_mesh, args.name + '_loss',
              os.path.join('/home/DiskB/rqding/checkpoints_0820/visualization/', 'save_only_path'), log=True, zmax=-1,
              interp=-1, show_points=True, show_polys=False)


if __name__ == "__main__":
    main()
