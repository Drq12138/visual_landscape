python plot_surface.py \
--batch_size 4096 \
--datasets CIFAR10 \
--name 000_random_wide1 \
--epoch 100 \
--smalldatasets 0.5 \
--mult_gpu \
--plt_path_one origin_path \
--fix_coor \
--plot_init save_net_resnet20_000.pt \
--load_path /home/DiskB/rqding/checkpoints_0919/visualization/origin_path \
--direction_path /home/DiskB/rqding/checkpoints_0919/visualization/origin_path/random_direction.pt ;

python plot_surface.py \
--batch_size 4096 \
--datasets CIFAR10 \
--name 000_pca_from_other \
--epoch 100 \
--smalldatasets 0.5 \
--mult_gpu \
--plt_path_one origin_path \
--fix_coor \
--plot_init save_net_resnet20_000.pt \
--load_path /home/DiskB/rqding/checkpoints_0919/visualization/origin_path \
--direction_path /home/DiskB/rqding/checkpoints_0919/visualization/new_path/pca_direction.pt ;


python plot_surface.py \
--batch_size 4096 \
--datasets CIFAR10 \
--name 000_pca_from_other_wide1 \
--epoch 100 \
--smalldatasets 0.5 \
--mult_gpu \
--plt_path_one origin_path \
--fix_coor \
--plot_init save_net_resnet20_000.pt \
--load_path /home/DiskB/rqding/checkpoints_0919/visualization/origin_path \
--direction_path /home/DiskB/rqding/checkpoints_0919/visualization/new_path/pca_direction.pt