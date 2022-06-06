import torch.nn as nn
from torch import pow, divide, abs


def total_variation_loss(img, weight):
     bs_img, c_img, h_img, w_img = img.size()
     tv_h = pow(img[:,:,1:,:]-img[:,:,:-1,:], 2).sum()
     tv_w = pow(img[:,:,:,1:]-img[:,:,:,:-1], 2).sum()
     
     return weight*(tv_h+tv_w)/(bs_img*c_img*h_img*w_img)


class TotalVariationLoss(nn.Module):
    def __init__(self):
        super(TotalVariationLoss, self).__init__()
        self.mse_loss = nn.MSELoss() 
        
    def forward(self, input, target):
        mse = self.mse_loss(input, target).sum()        
        variation_loss = total_variation_loss(target, target.shape[2])
        weight_loss = abs(divide(1, target + 1e-5)).sum()
        total_loss = mse + 0.0002 * variation_loss

        return total_loss.mean(), mse.mean(), variation_loss.mean()

        