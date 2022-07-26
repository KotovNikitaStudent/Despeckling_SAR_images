import torch.nn as nn
from torch import divide, tanh
import collections
from itertools import repeat
import torch.nn.functional as F


class Lambda(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x + 1e-7


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse

_pair = _ntuple(2)


class Conv2dSame(nn.Module):
    def __init__(
            self,
            in_channels, 
            out_channels, 
            kernel_size=3,
            stride=1,
            dilation=1,
            **kwargs):

        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            **kwargs)

        kernel_size_ = _pair(kernel_size)
        dilation_ = _pair(dilation)
        self._reversed_padding_repeated_twice = [0, 0]*len(kernel_size_)

        for d, k, i in zip(dilation_, kernel_size_, 
                                range(len(kernel_size_) - 1, -1, -1)):
            total_padding = d * (k - 1)
            left_pad = total_padding // 2
            self._reversed_padding_repeated_twice[2 * i] = left_pad
            self._reversed_padding_repeated_twice[2 * i + 1] = (
                    total_padding - left_pad)

    def forward(self, imgs):
        padded = F.pad(imgs, self._reversed_padding_repeated_twice)
        return self.conv(padded)


# class DespeckleFilter(nn.Module):
#     def __init__(self, in_c) -> None:
#         super(DespeckleFilter, self).__init__()
        
#         self.conv_0 = Conv2dSame(in_channels=in_c, out_channels=1)
#         self.conv_1 = Conv2dSame(in_channels=in_c, out_channels=64)
#         self.conv_2 = Conv2dSame(in_channels=64, out_channels=in_c)
#         self.leaky_rely = nn.LeakyReLU(negative_slope=0.2)
#         self.b_n = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU()
#         self.conv_dr_1 = Conv2dSame(in_channels=64, out_channels=64, dilation=1)
#         self.conv_dr_2 = Conv2dSame(in_channels=64, out_channels=64, dilation=2)
#         self.conv_dr_3 = Conv2dSame(in_channels=64, out_channels=64, dilation=3)
#         self.conv_dr_4 = Conv2dSame(in_channels=64, out_channels=64, dilation=4)
#         self.lambda_layer = Lambda()
        
#     def forward(self, inputs):
#         input_layer = inputs  
#         x = self.conv_0(inputs)      
#         x = self.conv_1(x)
#         x = self.leaky_rely(x)
        
#         x = self.conv_dr_1(x)
#         x = self.b_n(x)
#         x = self.leaky_rely(x)
        
#         x = self.conv_dr_2(x)
#         x = self.b_n(x)
#         x = self.leaky_rely(x)
        
#         x = self.conv_dr_3(x)
#         x = self.b_n(x)
#         x = self.leaky_rely(x)
        
#         x = self.conv_dr_4(x)
#         x = self.b_n(x)
#         x = self.leaky_rely(x)
        
#         x = self.conv_dr_4(x)
#         x = self.b_n(x)
#         x = self.relu(x)
        
#         x = self.conv_dr_3(x)
#         x = self.b_n(x)
#         x = self.relu(x)
        
#         x = self.conv_dr_2(x)
#         x = self.b_n(x)
#         x = self.relu(x)
        
#         x = self.conv_dr_1(x)
#         x = self.b_n(x)
#         x = self.relu(x)       
        
#         x = self.conv_2(x)
#         x = self.relu(x)
        
#         x = self.lambda_layer(x)
#         x = divide(input_layer, x)
#         x = tanh(x)
        
#         return x


class DespeckleFilter(nn.Module):
    def __init__(self, in_c) -> None:
        super(DespeckleFilter, self).__init__()
        
        self.conv_1 = Conv2dSame(in_channels=in_c, out_channels=64)
        self.conv_2 = Conv2dSame(in_channels=64, out_channels=in_c)
        self.conv_3 = Conv2dSame(in_channels=64, out_channels=64)
        self.b_n = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, inputs):
        input_layer = inputs
        x = self.conv_1(inputs)
        x = self.relu(x)

        x = self.conv_3(x)
        x = self.b_n(x)
        x = self.relu(x)

        x = self.conv_3(x)
        x = self.b_n(x)
        x = self.relu(x)

        x = self.conv_3(x)
        x = self.b_n(x)
        x = self.relu(x)

        x = self.conv_3(x)
        x = self.b_n(x)
        x = self.relu(x)

        x = self.conv_3(x)
        x = self.b_n(x)
        x = self.relu(x)

        x = self.conv_3(x)
        x = self.b_n(x)
        x = self.relu(x)

        x = self.conv_2(x)
        x = self.relu(x)

        x = divide(input_layer, x)
        x = self.tanh(x)
        return x
    

if __name__ == "__main__":
    from torchsummary import summary
    model = DespeckleFilter(1)
    summary(model, input_size=(1, 256, 256))