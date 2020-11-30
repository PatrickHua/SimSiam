import torch

class RandomDataset(torch.utils.data.Dataset):
    def __init__(self, root=None, train=True, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

        self.size = 1000
    def __getitem__(self, idx):
        if idx < self.size:
            return [torch.randn((3, 224, 224)), torch.randn((3, 224, 224))], [0,0,0]
        else:
            raise Exception

    def __len__(self):
        return self.size
