# find all configs in configs/
model=2dtan_128x128_pool_k5l8_tacos
# set your gpu id
gpus=0,3
# number of gpus
gpun=2
# please modify it with different value (e.g., 127.0.0.2, 29502) when you run multi 2dtan task on the same machine
master_addr=127.0.0.1
master_port=29501

# ------------------------ need not change -----------------------------------
config_file=configs/$model\.yaml
output_dir=outputs/$model

CUDA_VISIBLE_DEVICES=$gpus python -m torch.distributed.launch \
--nproc_per_node=$gpun --master_addr $master_addr --master_port $master_port train_net.py --config-file $config_file OUTPUT_DIR $output_dir \

