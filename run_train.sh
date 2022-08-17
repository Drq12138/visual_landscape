python train_path.py \
--batch_size 4096 \
--datasets CIFAR10 \
--name train_path_weight \
--epoch 100 \
--mult_gpu \
--randomseed 3 \
--optimizer sgd \
--direction_type pca \
--save_direction_type weight
