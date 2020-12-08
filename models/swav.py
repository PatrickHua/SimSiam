import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

class SwAV(nn.Module):
    def __init__(self, backbone=resnet50()):
        super().__init__()

        backbone.fc = nn.Identity()
        self.backbone = backbone
    
    def forward(self, x1, x2):
        






