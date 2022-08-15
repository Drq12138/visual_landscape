import imp
from json import load
from utils import h5_to_vtp
import numpy as np

base_dir = '/home/DiskB/rqding/checkpoints/visualization/0803/'
losses = np.load(base_dir+'losses.npy')
accuracies = np.load(base_dir + 'accuracies.npy')
xcoord_mesh = np.load(base_dir+ 'xcoord_mesh.npy')
ycoord_mesh = np.load(base_dir+ 'ycoord_mesh.npy')


h5_to_vtp(losses, xcoord_mesh, ycoord_mesh, 'loss', base_dir,zmax=100, log=True)
h5_to_vtp(accuracies, xcoord_mesh, ycoord_mesh, 'accuracy',base_dir, zmax=100, log=True)