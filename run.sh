python test_data.py \
--batch_size 4096 \
--datasets CIFAR10 \
--name train_path_two_sgd \
--epoch 100 \
--mult_gpu \
--optimizer adam \
--direction_type pca \
--load_path /home/DiskB/rqding/checkpoints/visualization/train_path_one_sgd/save_net_resnet20_000.pt \
--direction_path /home/DiskB/rqding/checkpoints/visualization/train_path_one_sgd/direction.pt
