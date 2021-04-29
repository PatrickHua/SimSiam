#!/bin/bash

for i in 0 50;
	do export CUDA_VISIBLE_DEVICES=1,2 && export WANDB_RUN_GROUP=ucf101_time_jittering_deterministic${i} && python main.py --config_file="configs/simsiam_ucf101.yaml" --data_dir="../ucfimages64x64" --log_dir="../logs/ucf101-contrastive-logs-${WANDB_RUN_GROUP}-${ID}/" --ckpt_dir=".cache/${WANDB_RUN_GROUP}" --linear_monitor --temporal_jitter_range=${i} --preload_dataset --wandb;
done
