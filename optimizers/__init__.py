from .lars import LARS
from .lars_simclr import LARS_simclr
from .larc import LARC
import torch


def get_optimizer(name, model, lr, momentum, weight_decay):
    if name == 'lars':
        optimizer = LARS(model.parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name == 'lars_simclr':
        optimizer = LARS_simclr(model.named_modules(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name == 'larc':
        optimizer = LARC(torch.optim.SGD(model.parameters(),lr=lr, momentum=momentum, weight_decay=weight_decay))
    else:
        raise NotImplementedError
    return optimizer



