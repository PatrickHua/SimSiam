#!/bin/bash

for i in 0;
	do export CUDA_VISIBLE_DEVICES=0,1 && export WANDB_RUN_GROUP=ucf101_time_jittering_deterministic${i} && python main.py --config_file="configs/simsiam_ucf101.yaml" --data_dir="../ucfimages64x64" --log_dir="../logs/ucf101-contrastive-logs-${WANDB_RUN_GROUP}-${ID}/" --ckpt_dir=".cache/${WANDB_RUN_GROUP}" --linear_monitor --temporal_jitter_range=${i} --preload_dataset --small_dataset;
done
