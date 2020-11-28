# SimSiam
A PyTorch implementation for the paper 'Exploring Simple Siamese Representation Learning' 

[paper](https://arxiv.org/abs/2011.10566) 

[code](https://github.com/PatrickHua/SimSiam) 



### Dependencies
`pip install requirement.txt`

### Run the program
```
python main.py --debug
```

### Run SimSiam

All default configurations are the same as the paper,
simply run:

```
export DATA="/path/to/your/datasets/"
export OUTPUT="/path/to/your/output/"
python main.py --dataset imagenet
```
or 
```
python main.py --dataset imagenet \
    --data_dir /path/to/your/datasets/ \
    --output_dir /path/to/your/output/ 
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
```





