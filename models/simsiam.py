import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchvision.models import resnet50


def D(p, z): # negative cosine similarity
    z = z.detach() # stop gradient
    p = F.normalize(p, dim=1) # l2-normalize 
    z = F.normalize(z, dim=1) # l2-normalize 
    return -(p*z).sum(dim=1).mean()

class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
        super().__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(hidden_dim)
        )

    def forward(self, x):
        x = self.layer1(x.squeeze())
        x = self.layer2(x)
        x = self.layer3(x)
        return x 


class prediction_MLP(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048):
        super().__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, 
        and h’s hidden layer’s dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x 

class SimSiam(nn.Module):
    def __init__(self, backbone=resnet50()):
        super().__init__()

        self.encoder = nn.Sequential( # f encoder
            *list(backbone.children())[:-1],
            projection_MLP(list(backbone.children())[-1].in_features)
        )
        self.predictor = prediction_MLP()
    
    def forward(self, x1, x2):
        z1, z2 = (zz:=self.encoder(torch.cat([x1, x2]))).chunk(2)

        p1, p2 = self.predictor(zz).chunk(2)

        L = D(p1, z2) / 2 + D(p2, z1) / 2

        return L





if __name__ == "__main__":
    model = SimSiam()
    x1 = torch.randn((20, 3, 224, 224))
    x2 = torch.randn_like(x1)

    model.forward(x1, x2).backward()

















