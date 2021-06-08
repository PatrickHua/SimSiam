import torch
import torchvision
from .random_dataset import RandomDataset


def get_dataset(dataset, data_dir, transform, train=True, download=True, debug_subset_size=None):
    if dataset == 'mnist':
        dataset = torchvision.datasets.MNIST(data_dir, train=train, transform=transform, download=download)
    elif dataset == 'stl10':
        dataset = torchvision.datasets.STL10(data_dir, split='train+unlabeled' if train else 'test', transform=transform, download=download)
    elif dataset == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(data_dir, train=train, transform=transform, download=True)
    elif dataset == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(data_dir, train=train, transform=transform, download=download)
    elif dataset == 'imagenet' and train == True:
        dataset = torchvision.datasets.ImageFolder(data_dir+'train', transform=transform)  
    elif dataset == 'imagenet' and train == False:
        dataset = torchvision.datasets.ImageFolder(data_dir+'val', transform=transform)      
    elif dataset == 'imagenet100' and train == True:
        dataset = torchvision.datasets.ImageFolder(data_dir+'train', transform=transform)  
    elif dataset == 'imagenet100' and train == False:
        dataset = torchvision.datasets.ImageFolder(data_dir+'val', transform=transform)      
    elif dataset == 'random':
        dataset = RandomDataset()
    else:
        raise NotImplementedError

    if debug_subset_size is not None:
        dataset = torch.utils.data.Subset(dataset, range(0, debug_subset_size)) # take only one batch
        dataset.classes = dataset.dataset.classes
        dataset.targets = dataset.dataset.targets
    return dataset