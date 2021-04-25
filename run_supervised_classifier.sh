#!/bin/bash

export WANDB_RUN_GROUP=supervised_trainclassifier && python -m torch.utils.bottleneck supervised_classifier.py --config_file="configs/simsiam_ucf101.yaml" --data_dir="../UCF-101/" --log_dir="../logs/ucf101-supervised-logs-${WANDB_RUN_GROUP}-${ID}/" --ckpt_dir=".cache/${WANDB_RUN_GROUP}" --preload_dataset
