import torch.nn as nn
from torch import pow
import torch


class TotalLoss(nn.Module):
    def __init__(self):
        super(TotalLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()
        
    def forward(self, input, target, alpha=0.002):
        mse = self.mse_loss(input, target).sum()
        variation_loss = tv_loss(target).sum()
        total_loss = mse + alpha * variation_loss

        return total_loss.mean()


class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()
    
    def forwasrd(self, input):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]

        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])

        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()

        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t) :
        return t.size()[1] * t.size()[2] * t.size()[3]
