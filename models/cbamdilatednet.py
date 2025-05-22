import torch.nn as nn
import torch
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2)
        self.gn1 = nn.GroupNorm(8, channels)
        self.gn2 = nn.GroupNorm(8, channels)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.dropout = nn.Dropout2d(p=0.1)

    def forward(self, x):
        identity = x
        x = self.leaky_relu(self.gn1(self.conv1(x)))
        x = self.gn2(self.conv2(x))
        x = self.dropout(x)
        x += identity
        x = self.leaky_relu(x)
        return x


class CBAM(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channel, 1, kernel_size=7, padding=3), nn.Sigmoid()
        )

    def forward(self, x):
        channel_wise = self.channel_gate(x)
        x = x * channel_wise
        spatial_wise = self.spatial_gate(x)
        x = x * spatial_wise
        return x


class CBAMDilatedNet(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)

        self.blocks = nn.Sequential(
            ResidualBlock(),
            CBAM(64),
            ResidualBlock(),
            CBAM(64),
            ResidualBlock(),
            CBAM(64),
        )

        self.conv_out = nn.Conv2d(64, in_channels, kernel_size=3, padding=1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.gn = nn.GroupNorm(8, 64)

    def forward(self, x):
        identity = x

        x = self.leaky_relu(self.conv1(x))
        x = self.blocks(x)
        x = self.gn(x)
        x = self.conv_out(x)

        denoised = identity - x
        return denoised


if __name__ == "__main__":
    from torchsummary import summary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CBAMDilatedNet().to(device)

    summary(model, input_size=(1, 256, 256))
