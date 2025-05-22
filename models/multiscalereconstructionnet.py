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


class MultiScaleReconstructionNet(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.block1 = ResidualBlock(64)
        self.down1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

        self.block2 = ResidualBlock(128)
        self.down2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.block3 = ResidualBlock(256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.recon2 = ResidualBlock(128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.recon1 = ResidualBlock(64)

        self.final_conv = nn.Conv2d(64, in_channels, kernel_size=3, padding=1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        identity = x

        c1 = self.leaky_relu(self.conv1(x))  # [B, 64, H, W]
        c1 = self.block1(c1)
        c2 = self.leaky_relu(self.down1(c1))  # [B, 128, H/2, W/2]
        c2 = self.block2(c2)
        c3 = self.leaky_relu(self.down2(c2))  # [B, 256, H/4, W/4]
        c3 = self.block3(c3)

        u2 = self.up2(c3)  # [B, 128, H/2, W/2]
        u2 = u2 + c2
        u2 = self.recon2(u2)

        u1 = self.up1(u2)  # [B, 64, H, W]
        u1 = u1 + c1
        u1 = self.recon1(u1)

        out = self.final_conv(u1)

        denoised = identity - out
        return denoised


if __name__ == "__main__":
    from torchsummary import summary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiScaleReconstructionNet().to(device)

    summary(model, input_size=(1, 256, 256))
