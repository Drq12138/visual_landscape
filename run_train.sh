python train_path.py \
--batch_size 4096 \
--datasets CIFAR10 \
--name train_path_two_adam_state \
--epoch 100 \
--mult_gpu \
--randomseed 2 \
--optimizer adam \
--direction_type pca \
--load_path /home/DiskB/rqding/checkpoints/visualization/train_path_one_sgd_state/save_net_resnet20_000.pt \
