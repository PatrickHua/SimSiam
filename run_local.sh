#!/bin/bash

for i in 0;
	do export WANDB_RUN_GROUP=ucf101_time_jittering_deterministic${i} && python -m torch.utils.bottleneck main.py --config_file="configs/simsiam_ucf101.yaml" --data_dir="../UCF-101/" --log_dir="../logs/ucf101-contrastive-logs-${WANDB_RUN_GROUP}-${ID}/" --ckpt_dir=".cache/${WANDB_RUN_GROUP}" --linear_monitor --temporal_jitter_range=${i} --preload_dataset; 
done
