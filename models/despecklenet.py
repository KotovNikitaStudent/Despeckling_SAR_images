import torch.nn as nn
import collections
from itertools import repeat
import torch.nn.functional as F
import torch


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


_pair = _ntuple(2)


class Conv2dSame(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, bias=True
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,
        )

        kernel_size_ = _pair(kernel_size)
        dilation_ = _pair(dilation)
        self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size_)

        for d, k, i in zip(dilation_, kernel_size_, reversed(range(len(kernel_size_)))):
            total_padding = d * (k - 1)
            left_pad = total_padding // 2
            self._reversed_padding_repeated_twice[2 * i] = left_pad
            self._reversed_padding_repeated_twice[2 * i + 1] = total_padding - left_pad

    def forward(self, x):
        x = F.pad(x, self._reversed_padding_repeated_twice)
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv2dSame(channels, channels, 3)
        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.conv2 = Conv2dSame(channels, channels, 3)
        self.gn2 = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.dropout = nn.Dropout2d(p=0.1)

    def forward(self, x):
        residual = x
        x = self.leaky_relu(self.gn1(self.conv1(x)))
        x = self.gn2(self.conv2(x))
        x = self.dropout(x)
        x += residual
        x = self.leaky_relu(x)
        return x


class DespeckleNet(nn.Module):
    def __init__(self, in_channels=1):
        super(DespeckleNet, self).__init__()

        self.conv1 = Conv2dSame(in_channels, 64, 3)

        self.res_blocks = nn.Sequential(
            ResidualBlock(), ResidualBlock(), ResidualBlock(), ResidualBlock()
        )

        self.conv_out = Conv2dSame(64, in_channels, 3)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.gn = nn.GroupNorm(num_groups=8, num_channels=64)
        self.dropout = nn.Dropout2d(p=0.5)

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.leaky_relu(x)

        x = self.res_blocks(x)

        x = self.gn(x)
        x = self.dropout(x)

        noise_map = self.conv_out(x)
        denoised = identity - noise_map

        return denoised


if __name__ == "__main__":
    from torchsummary import summary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DespeckleNet().to(device)

    summary(model, input_size=(1, 256, 256))
