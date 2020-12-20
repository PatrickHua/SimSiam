from torchvision import transforms
from PIL import Image, ImageOps
try:
    from torchvision.transforms import GaussianBlur
except ImportError:
    from .gaussian_blur import GaussianBlur
    torchvision.transforms.GaussianBlur = GaussianBlur

imagenet_norm = [[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]]

class BYOL_transform: # Table 6 
    def __init__(self, image_size, normalize=imagenet_norm):

        self.transform1 = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0), ratio=(3.0/4.0,4.0/3.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.2,0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=image_size//20*2+1, sigma=(0.1, 2.0)), # simclr paper gives the kernel size. Kernel size has to be odd positive number with torchvision
            transforms.ToTensor(),
            transforms.Normalize(*normalize)
        ])
        self.transform2 = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0), ratio=(3.0/4.0,4.0/3.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.2,0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            # transforms.RandomApply([GaussianBlur(kernel_size=int(0.1 * image_size))], p=0.1),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=0.1),
            transforms.RandomApply([Solarization()], p=0.2),
            
            transforms.ToTensor(),
            transforms.Normalize(*normalize)
        ])


    def __call__(self, x):
        x1 = self.transform1(x) 
        x2 = self.transform2(x) 
        return x1, x2


class Transform_single:
    def __init__(self, image_size, train, normalize=imagenet_norm):
        self.denormalize = Denormalize(*imagenet_norm)
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



class Solarization():
    # ImageFilter
    def __init__(self, threshold=128):
        self.threshold = threshold
    def __call__(self, image):
        return ImageOps.solarize(image, self.threshold)


