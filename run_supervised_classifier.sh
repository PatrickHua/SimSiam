#!/bin/bash

export CUDA_VISIBLE_DEVICES=0 && export WANDB_RUN_GROUP=supervised_trainclassifier && python supervised_classifier.py --config_file="configs/simsiam_stream51.yaml" --data_dir="../stream_data/" --log_dir="../logs/supervised-logs-${WANDB_RUN_GROUP}-${ID}/" --ckpt_dir=".cache/${WANDB_RUN_GROUP}" --preload_dataset
