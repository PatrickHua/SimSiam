from .simsiam import SimSiam
from torchvision.models import resnet50, resnet18
def get_model(name, backbone):

    if backbone == 'resnet50':
        backbone = resnet50()
    elif backbone == 'resnet18':
        backbone = resnet18()
    else:
        raise NotImplementedError

    if name == 'simsiam':
        return SimSiam(backbone)
    else:
        raise NotImplementedError






