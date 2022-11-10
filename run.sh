python test_python.py \
--name other_pca_0_none_100 \
--direction_path ./../checkpoints/1107/path_1/pca_direction.pt \
--init_point save_net_resnet20_000.pt \
--project_point save_net_resnet20_100.pt ;

python test_python.py \
--name other_pca_20_none_100 \
--direction_path ./../checkpoints/1107/path_1/pca_direction.pt \
--init_point save_net_resnet20_020.pt \
--project_point save_net_resnet20_100.pt ;

python test_python.py \
--name other_pca_50_none_100 \
--direction_path ./../checkpoints/1107/path_1/pca_direction.pt \
--init_point save_net_resnet20_050.pt \
--project_point save_net_resnet20_100.pt ;
