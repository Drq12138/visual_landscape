python plot_surface.py \
--batch_size 4096 \
--datasets CIFAR10 \
--name plot_surface_finalpoint \
--epoch 100 \
--mult_gpu \
--plt_path_one train_path_one_sgd_state \
--plt_path_two train_path_two_adam_state \
--load_path /home/DiskB/rqding/checkpoints/visualization/train_path_one_sgd_state/save_net_resnet20_100.pt \
--direction_path /home/DiskB/rqding/checkpoints/visualization/train_path_one_sgd_state/direction.pt