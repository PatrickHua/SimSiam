import copy
import random 
import torch 
from torch import nn 
import torch.nn.functional as F 
from torchvision import transforms 
from math import pi, cos 
from collections import OrderedDict
HPS = dict(
    max_steps=int(1000. * 1281167 / 4096), # 1000 epochs * 1281167 samples / batch size = 100 epochs * N of step/epoch
    # = total_epochs * len(dataloader) 
    mlp_hidden_size=4096,
    projection_size=256,
    base_target_ema=4e-3,
    optimizer_config=dict(
        optimizer_name='lars', 
        beta=0.9, 
        trust_coef=1e-3, 
        weight_decay=1.5e-6,
        exclude_bias_from_adaption=True),
    learning_rate_schedule=dict(
        base_learning_rate=0.2,
        warmup_steps=int(10.0 * 1281167 / 4096), # 10 epochs * N of steps/epoch = 10 epochs * len(dataloader)
        anneal_schedule='cosine'),
    batchnorm_kwargs=dict(
        decay_rate=0.9,
        eps=1e-5), 
    seed=1337,
)

def loss_fn(x, y):
    return 2 - 2 * F.cosine_similarity(x,y, dim=-1)
# loss_fn.py shows this function is the same with the one on the paper (but much faster)
# You can use -2 * cosine_similarity. I add 2 so that the range of loss is align with the paper's description
# def loss_fn(x, y):
#     x = F.normalize(x, dim=-1, p=2)
#     y = F.normalize(y, dim=-1, p=2)
#     return 2 - 2 * (x * y).sum(dim=-1)

class MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, HPS['mlp_hidden_size'])
        self.bn = nn.BatchNorm1d(HPS['mlp_hidden_size'], eps=HPS['batchnorm_kwargs']['eps'], momentum=1-HPS['batchnorm_kwargs']['decay_rate'])
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(HPS['mlp_hidden_size'], HPS['projection_size'])

    def forward(self, x):
        x = self.linear1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

class NetWrapper(nn.Module):
    def __init__(self, net):
        super().__init__() 
        self.encoder = net
        self.z_dim = self.encoder.fc.in_features
        self.projector = MLP(in_dim=self.z_dim)
        self.encoder.fc = nn.Identity() 

    def forward(self, x):
        representation = self.encoder(x) 
        projection = self.projector(representation) 
        return projection, representation 

        
class BYOL(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.online_encoder = NetWrapper(net)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.online_predictor = MLP(HPS['projection_size'])


    def target_ema(self, k, K, base_ema=HPS['base_target_ema']):
        # tau_base = 0.996 
        # base_ema = 1 - tau_base = 0.996 
        return 1 - base_ema * (cos(pi*k/K)+1)/2 
        # return 1 - (1-self.tau_base) * (cos(pi*k/K)+1)/2 

    @torch.no_grad()
    def update_moving_average(self, global_step, max_steps):
        tau = self.target_ema(global_step, max_steps)
        for online, target in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target.data = tau * target.data + (1 - tau) * online.data
            
    def forward(self, image_one, image_two):

        # image_one, image_two = images

        online_proj_one, representation = self.online_encoder(image_one)
        # breakpoint()
        online_proj_two, _ = self.online_encoder(image_two)

        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)

        with torch.no_grad():
            target_proj_one, _ = self.target_encoder(image_one)
            target_proj_two, _ = self.target_encoder(image_two)
        
        loss_one = loss_fn(online_pred_one, target_proj_two.detach())
        loss_two = loss_fn(online_pred_two, target_proj_one.detach())

# the representation of an augmented view of an image should be predictive of the representation of another augmented view of the same image.
# However, predicting directly in representation space can lead to collapsed representations
        # breakpoint()
        loss = (loss_one + loss_two).mean() # batch-wise mean

        return loss#, representation # representation sample for train time cluster evaluation

    
def test_tau():
    class BYOL2(nn.Module):
        def __init__(self):
            super().__init__()
            pass 

        def target_ema(self, k, K, base_ema=HPS['base_target_ema']):
            # tau_base = 0.996 
            # base_ema = 1 - tau_base = 0.996 
            return 1 - base_ema * (cos(pi*k/K)+1)/2 
            # return 1 - (1-self.tau_base) * (cos(pi*k/K)+1)/2 

        @torch.no_grad()
        def update_moving_average(self, global_step, max_steps):
            tau = self.target_ema(global_step, max_steps)
            # for online, target in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            #     target.data = tau * target.data + (1 - tau) * online.data
            return tau 
    model = BYOL2()
    taus = [] #
    import matplotlib.pyplot as plt 
    max_steps = 10000 
    for global_step in range(max_steps):
        taus.append(model.update_moving_average(global_step, max_steps))
    plt.plot(taus) 
    plt.show() 


def test_target_grad():
    import torchvision 
    resnet = torchvision.models.resnet18(pretrained=False)
    model = BYOL(resnet)
    x1 = torch.randn((2,3,128,128)) 
    x2 = torch.randn((2,3,128,128))
    y = model([x1, x2]) 
    y[0].backward()
    for name, param in model.named_parameters():
        print(name) 
        # print(param.shape) 
        if param.grad is None:
            print("grad is None")
if __name__ == "__main__":
    test_target_grad() 