import torch
import torch.nn as nn
import torch.nn.functional as F


def representation_loss_fn(x, y):
    # x, y : (batch_size, proj_size)
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    loss = 2 - 2 * (x * y).sum(dim=-1) # (batch_size,)
    return loss.mean()

class RepresentationLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, y):
        return representation_loss_fn(x, y)