#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1 export WANDB_RUN_GROUP=supervised_trainclassifier && python supervised_classifier.py --config_file="configs/simsiam_ucf101.yaml" --data_dir="../ucfimages64x64" --log_dir="../logs/ucf101-supervised-logs-${WANDB_RUN_GROUP}-${ID}/" --ckpt_dir=".cache/${WANDB_RUN_GROUP}" --preload_dataset
