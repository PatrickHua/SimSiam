from .simsiam_aug import SimSiamTransform
from .eval_aug import Transform_single
from .byol_aug import BYOL_transform
from .simclr_aug import SimCLRTransform
def get_aug(name, image_size, train, train_classifier=True):

    if train:
        if name == 'simsiam':
            augmentation = SimSiamTransform(image_size)
        elif name == 'byol':
            augmentation = BYOL_transform(image_size)
        elif name == 'simclr':
            augmentation = SimCLRTransform(image_size)
        else:
            raise NotImplementedError
    else:
        augmentation = Transform_single(image_size, train=train_classifier)

    return augmentation








