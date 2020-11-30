from .simsiam_aug import SimSiamTransform
from .eval_aug import Transform_single
from .byol_aug import BYOL_transform

def get_aug(name, image_size, train, train_classifier=True):
    if name == 'simsiam':
        if train:
            augmentation = SimSiamTransform(image_size)
        else:
            if train_classifier:
                augmentation = Transform_single(image_size, train=True)
            else:
                augmentation = Transform_single(image_size, train=False)
            # raise NotImplementedError
    
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








