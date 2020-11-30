# SimSiam
A PyTorch implementation for the paper [**Exploring Simple Siamese Representation Learning**](https://arxiv.org/abs/2011.10566) by Xinlei Chen & Kaiming He

This repo also provides pytorch implementations for simclr, byol and swav. I wrote the models using the exact set of configurations in their papers. You can open a pull request if mistakes are found.


### Dependencies

If you don't have python 3.8 environment:
```
conda create -n simsiam python=3.8
conda activate simsiam
```
Then install the required packages:
```
pip install requirement.txt
```

### Run this command to test the environment
```
python main.py --debug --dataset random --output_dir ./outputs/

➜  SimSiam git:(main) python main.py --debug
Epoch 0/1: 100%|█████████████████████████| 1/1 [00:02<00:00,  2.83s/it, loss=0.0273, loss_avg=0.0273]
Training: 100%|████████████████████████████████████████████████████████| 1/1 [00:02<00:00,  3.00s/it]
Model saved to ./outputs/simsiam-debug-epoch1.pth
```

### Choose a dataset
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

### Run SimSiam
The default model is simsiam and all default configurations are set to be the same as the simsiam paper (optimizers, models, datasets, image size ...),
simply run:

```
export DATA="/path/to/your/datasets/"
export OUTPUT="/path/to/your/output/"
python main.py --dataset imagenet
```
OR
```
python main.py --dataset imagenet \
    --data_dir /path/to/your/datasets/ \
    --output_dir /path/to/your/output/ \
```



### Run SimCLR
default hyperparameters are for simsiam, so you'll have to set them manually for simclr ...
maybe I should write a list of configurations for different models ...
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
    --warm_up_epochs 10
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
    --warm_up_epochs 10
```

### TODO
- complete code for byol, simclr and swav
- add code for linear evaluation
- convert from data-parallel (DP) to distributed data-parallel (DDP)
- create PyPI package `pip install simsiam-pytorch`


If you find this repo helpful, please consider star so that I have the motivation to improve it.



