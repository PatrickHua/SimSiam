#!/bin/bash
#
# This is a half-day long job
#SBATCH -t 11:00:00
#
# Uses 1 GPU
#SBATCH -p gpu --gres=gpu:1
#
# Uses 64 gb ram
#SBATCH --mem=64G
#
# Uses 1 cpu cores
#SBATCH -c 1
#
# Array
#SBATCH --array=1-4

ID=$(($SLURM_ARRAY_TASK_ID - 1))
exp_type='iid_linearmonitor'

source ~/miniconda3/bin/activate && conda activate simsiam && WANDB_RUN_GROUP=${exp_type} python3 main.py --config_file="configs/simsiam_stream51.yaml" --data_dir="../stream_data/" --log_dir="../logs/contrastive-logs-${exp_type}-${ID}/" --ckpt_dir=".cache/" --wandb --dataset_ordering='iid' --class_awareness --linear_monitor
