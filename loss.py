import torch
import torch.nn as nn
import torch


class TotalVariationLoss(nn.Module):
    def __init__(self):
        super(TotalVariationLoss, self).__init__()
        self.mse_loss = nn.MSELoss() 
        
    def forward(self, input, target):
        reduce_sum = torch.sum(input, target)
        