#!/bin/bash
python main.py \
--dataset cifar10 \
--image_size 32 \
--model simsiam \
--backbone resnet18 \
--optimizer sgd \
--weight_decay 0.0005 \
--momentum 0.9 \
--warmup_epoch 0 \
--warmup_lr 0 \
--base_lr 0.03 \
--final_lr 0 \
--num_epochs 800 \
--stop_at_epoch 100 \
--batch_size 2048 \
--eval_after_train \
--head_tail_accuracy \
--hide_progress \
--output_dir outputs/cifar10_experiment/ \
# --debug












