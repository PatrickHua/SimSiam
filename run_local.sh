#!/bin/bash

for i in 50 0;
	do export CUDA_VISIBLE_DEVICES=0 && export WANDB_RUN_GROUP=flippedevaluationdatasets_time_jittering_deterministic${i} && python main.py --config_file="configs/simsiam_stream51.yaml" --data_dir="../stream_data/" --log_dir="../logs/contrastive-logs-${WANDB_RUN_GROUP}-${ID}/" --ckpt_dir=".cache/${WANDB_RUN_GROUP}" --linear_monitor --temporal_jitter_range=${i} --preload_dataset --wandb; 
done
