CUDA_VISIBLE_DEVICES=0,1 python plot_surface.py \
--batch_size 8192 \
--datasets CIFAR10 \
--name pca_0_none_100 \
--epoch 100 \
--smalldatasets 0.5 \
--mult_gpu \
--plt_path_one path_0 \
--plot_landscape \
--fix_coor \
--plot_init save_net_resnet20_000.pt \
--base_checkpoint save_net_resnet20_100.pt \
--load_path ./../checkpoints/1101/path_0 \
--direction_path ./../checkpoints/1101/path_0/pca_direction.pt ;

CUDA_VISIBLE_DEVICES=0,1 python plot_surface.py \
--batch_size 8192 \
--datasets CIFAR10 \
--name pca_20_none_100 \
--epoch 100 \
--smalldatasets 0.5 \
--mult_gpu \
--plt_path_one path_0 \
--plot_landscape \
--fix_coor \
--plot_init save_net_resnet20_020.pt \
--base_checkpoint save_net_resnet20_100.pt \
--load_path ./../checkpoints/1101/path_0 \
--direction_path ./../checkpoints/1101/path_0/pca_direction.pt ;

CUDA_VISIBLE_DEVICES=0,1 python plot_surface.py \
--batch_size 8192 \
--datasets CIFAR10 \
--name pca_50_none_100 \
--epoch 100 \
--smalldatasets 0.5 \
--mult_gpu \
--plt_path_one path_0 \
--plot_landscape \
--fix_coor \
--plot_init save_net_resnet20_050.pt \
--base_checkpoint save_net_resnet20_100.pt \
--load_path ./../checkpoints/1101/path_0 \
--direction_path ./../checkpoints/1101/path_0/pca_direction.pt ;

CUDA_VISIBLE_DEVICES=0,1 python plot_surface.py \
--batch_size 8192 \
--datasets CIFAR10 \
--name pca_100_none_100 \
--epoch 100 \
--smalldatasets 0.5 \
--mult_gpu \
--plt_path_one path_0 \
--plot_landscape \
--fix_coor \
--plot_init save_net_resnet20_100.pt \
--base_checkpoint save_net_resnet20_100.pt \
--load_path ./../checkpoints/1101/path_0 \
--direction_path ./../checkpoints/1101/path_0/pca_direction.pt ;