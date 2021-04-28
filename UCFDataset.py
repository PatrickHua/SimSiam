import torch
import torchvision
import os
from tqdm import trange
from multiprocessing import Pool
import numpy as np

class UCFDataset(torchvision.datasets.UCF101):
    def __init__(self, root, train, transform, temporal_jitter_range=0):
        super(UCFDataset, self).__init__(root, os.path.join(root, "ucfTrainTestlist"), 1, step_between_clips=1, train=train, transform=None, num_workers=9)

        self.temporal_jitter_range = temporal_jitter_range
        self.img_transform = transform

        # self.img_cache = {}

    def get_item_help(self, index):
        # if index not in self.img_cache:
        img, _, label = super(UCFDataset, self).__getitem__(index)
        img = img[0].permute((2, 0, 1)) / 255.
            # self.img_cache[index] = (img, label)

        return img, label

    def __getitem__(self, index):
        img, label = self.get_item_help(index)
        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.train:
            new_label = None
            new_idx = min(index + self.temporal_jitter_range, len(self)-1) + 1
            while new_label != label:
                new_idx -= 1
                new_img, new_label = self.get_item_help(index)
            if self.img_transform is not None:
                new_img = self.img_transform(new_img)
            return img, new_img, label
        else:
            return img, label

class UCFImageDataset(torchvision.datasets.ImageFolder):
    def __init__(self, root, train, transform, temporal_jitter_range=0):
        if train:
            root += '-test'
        else:
            root += '-train'
        # super(UCFImageDataset, self).__init__(root, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Resize(64)]))
        # super(UCFImageDataset, self).__init__(root, transform=None)
        super(UCFImageDataset, self).__init__(root, transform=None)
        self.img_transform = transform
        self.train = train
        self.temporal_jitter_range = temporal_jitter_range

        # self.cache = {}
        # print("starting loadign", flush=True)
        # if True:
        #     for i in trange(len(self)):
        #         self.cache[i] = self.get_item_helper(i)
        # else:
        #     with Pool(8) as p:
        #         self.cache = p.map(self.get_item_helper, [i for i in range(len(self))])

    def get_item_helper(self, index):
        # if index not in self.cache:
        img, label = super(UCFImageDataset, self).__getitem__(index)

        # img = (img.numpy() * 255.).astype(np.uint8)

        return img, label

        # return self.cache[index]
    
    def __getitem__(self, index):
        img, label = self.get_item_helper(index)
        # img = img.float() / 255.

        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.train:
            new_label = None
            new_idx = min(index + self.temporal_jitter_range, len(self)-1) + 1
            while new_label != label:
                assert new_idx >= index
                new_idx -= 1
                if new_idx == index:
                    new_img = img
                    break
                new_img, new_label = self.get_item_helper(new_idx)
            if self.img_transform is not None:
                new_img = self.img_transform(new_img)
            return img, new_img, label
        else:
            return img, label
