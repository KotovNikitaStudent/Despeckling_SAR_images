import torch.nn as nn
import torch
import numpy as np
from scipy.signal import convolve2d
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target).pow(2) + self.eps**2))


def get_sobel_kernel(size=3):
    kernel = np.zeros((size, size))
    kernel[size // 2, :] = np.ones(size)
    kernel -= np.mean(kernel)
    sobel_1 = convolve2d([[0]], kernel, mode="full")
    sobel_2 = convolve2d([[0]], kernel.T, mode="full")
    return torch.tensor(sobel_1, dtype=torch.float32), torch.tensor(
        sobel_2, dtype=torch.float32
    )


class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x, sobel_y = get_sobel_kernel()
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

    def forward(self, pred, target):
        sobel_x = self.sobel_x.type_as(pred)
        sobel_y = self.sobel_y.type_as(pred)

        pred_grad_x = F.conv2d(pred, sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred, sobel_y, padding=1)
        target_grad_x = F.conv2d(target, sobel_x, padding=1)
        target_grad_y = F.conv2d(target, sobel_y, padding=1)

        return F.l1_loss(pred_grad_x, target_grad_x) + F.l1_loss(
            pred_grad_y, target_grad_y
        )


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, channel=1, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.window = self.create_window(window_size, channel)

    def gaussian(self, window_size, sigma):
        x = torch.arange(window_size, dtype=torch.float32)
        x = x - window_size // 2
        gauss = torch.exp(-(x.pow(2)) / (2 * sigma**2))
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t())
        _3D_window = _2D_window.expand(channel, 1, window_size, window_size)
        return _3D_window

    def forward(self, pred, target):
        """
        pred, target: Tensor of shape (B, C, H, W)
        """
        B, C, H, W = pred.shape
        window = self.window.to(pred.device)

        mu1 = F.conv2d(pred, window, padding=self.window_size // 2, groups=C)
        mu2 = F.conv2d(target, window, padding=self.window_size // 2, groups=C)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = (
            F.conv2d(pred * pred, window, padding=self.window_size // 2, groups=C)
            - mu1_sq
        )
        sigma2_sq = (
            F.conv2d(target * target, window, padding=self.window_size // 2, groups=C)
            - mu2_sq
        )
        sigma12 = (
            F.conv2d(pred * target, window, padding=self.window_size // 2, groups=C)
            - mu1_mu2
        )

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )

        if self.size_average:
            return 1 - ssim_map.mean()
        else:
            return 1 - ssim_map.mean([1, 2, 3]).mean()


class CombinedLoss(nn.Module):
    def __init__(self, w_charb=1.0, w_ssim=0.5, w_edge=0.1):
        super().__init__()
        self.charbonnier = CharbonnierLoss()
        self.ssim_loss = SSIMLoss()
        self.edge_loss = EdgeLoss()

        self.w_charb = w_charb
        self.w_ssim = w_ssim
        self.w_edge = w_edge

    def forward(self, pred, target):
        charb = self.charbonnier(pred, target)
        ssim = self.ssim_loss(pred, target)
        edge = self.edge_loss(pred, target)

        total_loss = self.w_charb * charb + self.w_ssim * ssim + self.w_edge * edge
        return total_loss
