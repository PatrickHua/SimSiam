from .lars import LARS
import torch


def get_optimizer(name, parameters, lr, momentum, weight_decay):
    if name == 'lars':
        optimizer = LARS(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise NotImplementedError
    return optimizer



