import torch.nn as nn
from torch import divide, tanh


class conv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)

    def forward(self, inputs):
        x = self.conv(inputs)
        return x
    

class conv_dr(nn.Module):
    def __init__(self, in_c, out_c, dilation_rate=1, padding=1):
        super().__init__()

        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding=padding, dilation=dilation_rate)

    def forward(self, inputs):
        x = self.conv(inputs)
        return x


class Lambda(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x + 1e-7


class DespeckleFilter(nn.Module):
    def __init__(self, in_c) -> None:
        super(DespeckleFilter, self).__init__()
        
        self.conv_1 = conv(in_c=in_c, out_c=64)
        self.conv_2 = conv(in_c=64, out_c=3)
        self.leaky_rely = nn.LeakyReLU(negative_slope=0.2)
        self.b_n = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.conv_dr_1 = conv_dr(64, 64, 1, 1)
        self.conv_dr_2 = conv_dr(64, 64, 2, 2)
        self.conv_dr_3 = conv_dr(64, 64, 3, 3)
        self.conv_dr_4 = conv_dr(64, 64, 4, 4)
        self.lambda_layer = Lambda()
        
    def forward(self, inputs):
        input_layer = inputs
        x = self.conv_1(inputs)
        x = self.leaky_rely(x)
        
        x = self.conv_dr_1(x)
        x = self.b_n(x)
        x = self.leaky_rely(x)
        
        x = self.conv_dr_2(x)
        x = self.b_n(x)
        x = self.leaky_rely(x)
        
        x = self.conv_dr_3(x)
        x = self.b_n(x)
        x = self.leaky_rely(x)
        
        x = self.conv_dr_4(x)
        x = self.b_n(x)
        x = self.leaky_rely(x)
        
        x = self.conv_dr_4(x)
        x = self.b_n(x)
        x = self.relu(x)
        
        x = self.conv_dr_3(x)
        x = self.b_n(x)
        x = self.relu(x)
        
        x = self.conv_dr_2(x)
        x = self.b_n(x)
        x = self.relu(x)
        
        x = self.conv_dr_1(x)
        x = self.b_n(x)
        x = self.relu(x)       
        
        x = self.conv_2(x)
        x = self.relu(x)
        
        x = self.lambda_layer(x)
        x = divide(input_layer, x)
        x = tanh(x)
        
        return x
    

if __name__ == "__main__":
    from torchsummary import summary
    model = DespeckleFilter(3)
    summary(model, input_size=(3, 256, 256))