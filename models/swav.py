import torch
import torch.nn as nn
import torch.nn.functional as F










class SwAV(nn.Module):
    def __init__(self, backbone=resnet50()):
        super().__init__()








