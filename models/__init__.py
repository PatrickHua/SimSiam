from .simsiam import SimSiam
from torchvision.models import resnet50, resnet18
import torch

def get_backbone(backbone, castrate=False):
    if backbone == 'resnet50':
        backbone = resnet50()
    elif backbone == 'resnet18':
        backbone = resnet18()
    else:
        raise NotImplementedError
    if castrate == True:
        backbone.fc = torch.nn.Identity()
        
    return backbone

def get_model(name, backbone):
    if name == 'simsiam':
        return SimSiam(get_backbone(backbone))
    else:
        raise NotImplementedError






