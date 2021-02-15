import torch
import torchvision
from .random_dataset import RandomDataset
from StreamDataset import StreamDataset


def get_dataset(dataset, data_dir, transform, train=True, download=False, debug_subset_size=None, ordering='iid'):
    if dataset == 'mnist':
        dataset = torchvision.datasets.MNIST(data_dir, train=train, transform=transform, download=download)
    elif dataset == 'stl10':
        dataset = torchvision.datasets.STL10(data_dir, split='train+unlabeled' if train else 'test', transform=transform, download=download)
    elif dataset == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(data_dir, train=train, transform=transform, download=download)
    elif dataset == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(data_dir, train=train, transform=transform, download=download)
    elif dataset == 'imagenet':
        dataset = torchvision.datasets.ImageNet(data_dir, split='train' if train == True else 'val', transform=transform, download=download)
    elif dataset == 'random':
        dataset = RandomDataset()
    elif dataset == 'stream51':
        dataset = StreamDataset(data_dir, train=train, ordering=ordering, transform=transform)
        dataset.classes = ["class_{:d}".format(i) for i in range(51)]
    else:
        raise NotImplementedError

    if debug_subset_size is not None:
        dataset = torch.utils.data.Subset(dataset, range(0, debug_subset_size)) # take only one batch
        dataset.classes = dataset.dataset.classes
        dataset.targets = dataset.dataset.targets
    return dataset
