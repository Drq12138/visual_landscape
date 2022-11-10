python train_path.py \
--batch_size 4096 \
--datasets CIFAR10 \
--name path_1 \
--epoch 100 \
--mult_gpu \
--randomseed 2 \
--direction_type pca \
--optimizer sgd \
--save_direction_type weight