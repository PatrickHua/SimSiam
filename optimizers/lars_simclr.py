"""The lars optimizer used in simclr is a bit different from the paper where they exclude certain parameters"""
"""I asked the author of byol, they also stick to the simclr lars implementation"""



import torch 
import torchvision
from torch.optim.optimizer import Optimizer 
import torch.nn as nn 
# comments from the lead author of byol
# 2. + 3. We follow the same implementation as the one used in SimCLR for LARS. This is indeed a bit 
# different from the one described in the LARS paper and the implementation you attached to your email. 
# In particular as in SimCLR we first modify the gradient to include the weight decay (with beta corresponding 
# to self.weight_decay in the SimCLR code) and then adapt the learning rate by dividing by the norm of this 
# sum, this is different from the LARS pseudo code where they divide by the sum of the norm (instead of the 
# norm of the sum as SimCLR and us are doing). This is done in the SimCLR code by first adding the weight 
# decay term to the gradient and then using this sum to perform the adaptation. We also use a term (usually 
# referred to as trust_coefficient but referred as eeta in SimCLR code) set to 1e-3 to multiply the updates 
# of linear layers.
# Note that the logic "if w_norm > 0 and g_norm > 0 else 1.0" is there to tackle numerical instabilities.
# In general we closely followed SimCLR implementation of LARS.
class LARS_simclr(Optimizer):
    def __init__(self, 
                 named_modules, 
                 lr,
                 momentum=0.9, # beta? YES
                 trust_coef=1e-3,
                 weight_decay=1.5e-6,
                exclude_bias_from_adaption=True):
        '''byol: As in SimCLR and official implementation of LARS, we exclude bias # and batchnorm weight from the Lars adaptation and weightdecay'''
        defaults = dict(momentum=momentum,
                lr=lr,
                weight_decay=weight_decay,
                 trust_coef=trust_coef)
        parameters = self.exclude_from_model(named_modules, exclude_bias_from_adaption)
        super(LARS_simclr, self).__init__(parameters, defaults)

    @torch.no_grad() 
    def step(self):
        for group in self.param_groups: # only 1 group in most cases 
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            lr = group['lr']

            trust_coef = group['trust_coef']
            # print(group['name'])
            # eps = group['eps']
            for p in group['params']:
                # breakpoint()
                if p.grad is None:
                    continue
                global_lr = lr
                velocity = self.state[p].get('velocity', 0)  
                # if name in self.exclude_from_layer_adaptation:
                if self._use_weight_decay(group):
                    p.grad.data += weight_decay * p.data 

                trust_ratio = 1.0 
                if self._do_layer_adaptation(group):
                    w_norm = torch.norm(p.data, p=2)
                    g_norm = torch.norm(p.grad.data, p=2)
                    trust_ratio = trust_coef * w_norm / g_norm if w_norm > 0 and g_norm > 0 else 1.0 
                scaled_lr = global_lr * trust_ratio # trust_ratio is the local_lr 
                next_v = momentum * velocity + scaled_lr * p.grad.data 
                update = next_v
                p.data = p.data - update 


    def _use_weight_decay(self, group):
        return False if group['name'] == 'exclude' else True
    def _do_layer_adaptation(self, group):
        return False if group['name'] == 'exclude' else True

    def exclude_from_model(self, named_modules, exclude_bias_from_adaption=True):
        base = [] 
        exclude = []
        for name, module in named_modules:
            if type(module) in [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]:
                # if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
                for name2, param in module.named_parameters():
                    exclude.append(param)
            else:
                for name2, param in module.named_parameters():
                    if name2 == 'bias':
                        exclude.append(param)
                    elif name2 == 'weight':
                        base.append(param)
                    else:
                        pass # non leaf modules 
        return [{
            'name': 'base',
            'params': base
            },{
            'name': 'exclude',
            'params': exclude
        }] if exclude_bias_from_adaption == True else [{
            'name': 'base',
            'params': base+exclude 
        }]

if __name__ == "__main__":
    
    resnet = torchvision.models.resnet18(pretrained=False)
    model = resnet

    optimizer = LARS_simclr(model.named_modules(), lr=0.1)
    # print()
    # out = optimizer.exclude_from_model(model.named_modules(),exclude_bias_from_adaption=False) 
    # print(len(out[0]['params']))
    # exit() 

    criterion = torch.nn.CrossEntropyLoss()
    for i in range(100):
        model.zero_grad()
        pred = model(torch.randn((2,3,32,32)))
        loss = pred.mean()
        loss.backward()
        optimizer.step()









