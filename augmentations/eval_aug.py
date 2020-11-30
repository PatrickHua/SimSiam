from torchvision import transforms
from PIL import Image

imagenet_norm = [[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]]

class Transform_single():
    def __init__(self, image_size, train, normalize=imagenet_norm):
        if train == True:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0), ratio=(3.0/4.0,4.0/3.0), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(*normalize)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(int(image_size*(8/7)), interpolation=Image.BICUBIC), # 224 -> 256 
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(*normalize)
            ])

    def __call__(self, x):
        return self.transform(x)
