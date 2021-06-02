import sys
sys.path.append("../")

import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

def loss_fn(x, y):
    # x, y : (batch_size, proj_size)
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    loss = 2 - 2 * (x * y).sum(dim=-1) # (batch_size,)
    return loss.mean()
    
class MLP(nn.Module):
    def __init__(self, inplace, hidden_size, place):
        super().__init__()
        self.fc1 = nn.Linear(inplace, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_size, place)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class RepNet(nn.Module):
    def __init__(self, encoder, proj_size, hidden_size):
        super().__init__()
        self.encoder = encoder # view and representation
        self.projection = MLP(proj_size, hidden_size, proj_size)
        self.prediction = MLP(proj_size, hidden_size, proj_size)

    def forward(self, x):
        x = self.project(x)
        x = self.prediction(x)
        return x

    def project(self, x):
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)
        x = self.projection(x)
        return x

    def predict(self, x):
        return self.forward(x)

class ExponentialMovingAverage:
    # the exponential moving average parameter τ starts from τbase = 0.996 and is increased to one during training. 
    def __init__(self, beta=0.996, max_step=None):
        self.base = beta
        self.beta = beta
        self.max_step = max_step
        self.cur_step = -1

    def step(self, online, target):
        self.cur_step += 1
        self.update_beta()
        for on_params, ta_params in zip(online.parameters(), target.parameters()):
            ta_params.data = self.update_param(on_params.data, ta_params.data)

    def update_param(self, online, target):
        return target * self.beta + (1 - self.beta) * online
    
    def update_beta(self):
        if self.max_step is not None:
            beta = 1 - (1 - self.base)*(np.cos(np.pi * self.cur_step / self.max_step) + 1) / 2
            self.beta = min(1., beta)
            

def main():
    model = resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False)
    model = torch.nn.Sequential(*list(model.children())[:-1])

    onlinenet = RepNet(model, 512, 4096)
    # learning rate は, 0.2 * batch_size / 256
    # weight decay は biasとbatchnormのパラメータを除いて, 1.5 * 1e-6に設定
    # LARSとは？ "Scaling SGD Batch Size to 32K for ImageNet Training"
    # LARS optimizer with a cosine decay learning rate schedule, without restarts, over 1000epoch, with warm-up period of 10epoch
    optimizer = torch.optim.Adam(onlinenet.parameters(), lr=1e-3)
    targetnet = deepcopy(onlinenet)
    targetnet.eval()

    ema = ExponentialMovingAverage(max_step=10)

    x1 = torch.rand((4, 3, 32, 32))
    x2 = torch.rand((4, 3, 32, 32))

    print(f"x = {x1.shape}")

    online_y = onlinenet.predict(x1)
    with torch.no_grad():
        target_y = targetnet.project(x2)

    print(f"online_y = {online_y.shape}")
    print(f"target_y = {target_y.shape}")

    loss = loss_fn(online_y, target_y)
    print(f"loss = {loss}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    ema.step(onlinenet, targetnet)

    # for (name, on_params), ta_params in zip(onlinenet.named_parameters(), targetnet.parameters()):
    #     percent = (on_params.data == ta_params.data).sum().item() / on_params.data.view(-1).shape[0]
    #     print(f"{name} = {percent*100:.4f}%")

if __name__ == "__main__":
    main()