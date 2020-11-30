from .simsiam_aug import SimSiamTransform
from .byol_aug import BYOL_transform

def get_aug(name, image_size, train):
    if name == 'simsiam':
        if train:
            augmentation = SimSiamTransform(image_size)
        else:
            raise NotImplementedError
    elif name == 'byol':
        if train:
            augmentation = BYOL_transform(image_size)
        else:
            raise NotImplementedError
    elif name == 'simclr':
        if train:
            TODO    
            
    else:
        raise NotImplementedError

    return augmentation








