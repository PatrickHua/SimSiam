import argparse
import os
import torch
from .byol_cfg import byol_args


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    # training specific args
    parser.add_argument('--dataset', type=str, default='stl10', help='choose from random, stl10, mnist, cifar10, cifar100, imagenet')
    parser.add_argument('--download', action='store_true', help="if can't find dataset, download from web")
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--data_dir', type=str, default=os.getenv('DATA'))
    parser.add_argument('--output_dir', type=str, default=os.getenv('OUTPUT'))
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--eval_from', type=str, default=None)

    parser.add_argument('--use_default_hyperparameters', action='store_true')
    # model related params
    parser.add_argument('--model', type=str, default='simsiam')
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=512)
    # optimization params
    parser.add_argument('--optimizer', type=str, default='sgd', help='sgd, lars(from lars paper), lars_simclr(used in simclr and byol), larc(used in swav)')
    parser.add_argument('--base_lr', type=float, default=0.05)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--warm_up_epochs', type=int, default=0, help='learning rate will be linearly scaled during warm up period')


    args = parser.parse_args()

    if args.use_default_hyperparameters:
        if args.model == 'byol':
            args.__dict__.update(byol_args)
        # elif args.model == ''
        else:
            raise NotImplementedError
    
    return args
