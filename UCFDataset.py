import torch
import torchvision
import os

class UCFDataset(torchvision.datasets.UCF101):
    def __init__(self, root, train, transform, temporal_jitter_range=0):
        super(UCFDataset, self).__init__(root, os.path.join(root, "ucfTrainTestlist"), 1, step_between_clips=1, train=train, transform=None, num_workers=9)

        self.temporal_jitter_range = temporal_jitter_range
        self.img_transform = transform

        self.img_cache = {}

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
            return img, new_img
