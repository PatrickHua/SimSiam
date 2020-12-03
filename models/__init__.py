from .simsiam import SimSiam
from .byol import BYOL
from .simclr import SimCLR
from torchvision.models import resnet50, resnet18
import torch

def get_backbone(backbone, castrate=True):
    if backbone == 'resnet50':
        backbone = resnet50()
    elif backbone == 'resnet18':
        backbone = resnet18()
    else:
        raise NotImplementedError
    if castrate:
        backbone.output_dim = backbone.fc.in_features
        backbone.fc = torch.nn.Identity()

    return backbone


def get_model(name, backbone):
    if name == 'simsiam':
        model =  SimSiam(get_backbone(backbone))
    elif name == 'byol':
        model = BYOL(get_backbone(backbone))
    elif name == 'simclr':
        model = SimCLR(get_backbone(backbone))
    else:
        raise NotImplementedError
    return model






