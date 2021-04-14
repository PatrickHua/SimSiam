#!/bin/bash

export i=0 && export CUDA_VISIBLE_DEVICES=0 && export WANDB_RUN_GROUP=time_jittering_deterministic${i} && python linear_eval.py --config_file="configs/simsiam_stream51.yaml" --data_dir="../stream_data/" --log_dir="../logs/contrastive-logs-${WANDB_RUN_GROUP}-${ID}/" --ckpt_dir=".cache/${WANDB_RUN_GROUP}" --preload_dataset --small_dataset --eval_from ".cache/time_jittering_deterministic50/simsiam-stream51-experiment-resnet18_cifar_variant1_0402015128.pth"
