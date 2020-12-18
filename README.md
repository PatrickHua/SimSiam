# SimSiam
A PyTorch implementation for the paper [**Exploring Simple Siamese Representation Learning**](https://arxiv.org/abs/2011.10566) by Xinlei Chen & Kaiming He

This repo also provides pytorch implementations for simclr, byol and swav. I wrote the models using the exact set of configurations in their papers. You can open a pull request if mistakes are found.


### Dependencies

If you don't have python 3 environment:
```
conda create -n simsiam python=3.8
conda activate simsiam
```
Then install the required packages:
```
pip install -r requirements.txt
```

### Run this command to test the environment

```
python main.py --debug --dataset cifar10 --data_dir "/Your/data/folder/" --output_dir "/Your/output/folder/"
```
The data folder should look like this:
```
➜  ~ tree /Your/data/folder/
├── cifar-10-batches-py
│   ├── batches.meta
│   ├── data_batch_1
│   ├── ...
└── stl10_binary
    ├── ...
```
```
python main.py --debug --dataset cifar10 --data_dir ~/Data --output_dir ./outputs/
Epoch 0/1: 100%|████████████████████████████████████| 1/1 [00:03<00:00,  3.60s/it, loss=-.0196, loss_avg=-.0196]
Training: 100%|███████████████████████████████████████████████████████████████████| 1/1 [00:03<00:00,  3.83s/it]
Model saved to ./outputs/simsiam-cifar10-epoch1.pth
```
>`export DATA="/path/to/your/datasets/"` and `export OUTPUT="/path/to/your/output/"` will save you the trouble of entering the folder name everytime!

### Run SimSiam
I made an example training script for the cifar10 experiment in Appendix D.

```
sh configs/cifar_experiment.sh
```
```
Training: 100%|#################################| 800/800 [3:27:50<00:00, 15.59s/it, epoch=799, loss_avg=-.895]
Model saved to outputs/cifar10_experiment/simsiam-cifar10-epoch800.pth
Evaluating: 100%|###################################| 100/100 [08:24<00:00,  5.04s/it, epoch=99, accuracy=80.8]

```


### Run SimCLR

```
python main.py \
    --model simclr \
    --optimizer lars \
    --data_dir /path/to/your/datasets/ \
    --output_dir /path/to/your/output/ \
    --backbone resnet50 \
    --dataset imagenet \ 
    --batch_size 4096 \ 
    --num_epochs 800 \
    --optimizer lars_simclr \
    --weight_decay 1e-6 \
    --base_lr 0.3 \
    --warmup_epochs 10
```

### Run BYOL
```
python main.py \
    --model byol \
    --optimizer lars \ 
    --data_dir /path/to/your/datasets/ \
    --output_dir /path/to/your/output/ \
    --backbone resnet50 \
    --dataset imagenet \ 
    --batch_size 256 \ 
    --num_epochs 100 \ 
    --optimizer lars_simclr \ They use simclr version of lars
    --weight_decay 1.5e-6 \
    --base_lr 0.3 \
    --warmup_epochs 10
```

### TODO

- convert from data-parallel (DP) to distributed data-parallel (DDP)
- create PyPI package `pip install simsiam-pytorch`


If you find this repo helpful, please consider star so that I have the motivation to improve it.



