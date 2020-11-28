import argparse
import os
import torch

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='stl10', help='choose from stl10, mnist, cifar10, cifar100, imagenet')
    parser.add_argument('--image_size', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--base_lr', type=float, default=0.05)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--data_dir', type=str, default=os.getenv('DATA'))
    parser.add_argument('--output_dir', type=str, default=os.getenv('OUTPUT'))
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--model', type=str, default='simsiam')
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--download', action='store_true', help="if can't find dataset, download from web")
    return parser.parse_args()





