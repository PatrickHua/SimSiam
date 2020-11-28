from .simsiam import SimSiam
from torchvision.models import resnet50
def get_model(name, backbone):

    if backbone == 'resnet50':
        backbone = resnet50()
    else:
        raise NotImplementedError

    if name == 'simsiam':
        return SimSiam(backbone)
    else:
        raise NotImplementedError






