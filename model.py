from this import d
from tkinter import N
from turtle import forward
import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, dilation_rate=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, dilation=dilation_rate)

class DespeckleFilter(nn.Module):
    def __init__(self, input_shape) -> None:
        super(DespeckleFilter, self).__init__()
        
        self.conv3x3_1 = conv3x3(input_shape)
        self.batch_norm = nn.BatchNorm2d(input_shape)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        
        self.conv3x3_2 = conv3x3(input_shape, dilation_rate=1)
        self.conv3x3_3 = conv3x3(input_shape, dilation_rate=2)
        self.conv3x3_4 = conv3x3(input_shape, dilation_rate=3)
        self.conv3x3_5 = conv3x3(input_shape, dilation_rate=4)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x.size()
        x = self.conv3x3_1(x) # filters = 64
        x = self.leaky_relu(x)
        
        x = self.conv3x3_2(x) # filters = 64
        x = self.batch_norm(x)
        x = self.leaky_relu(x)
        
        x = self.conv3x3_3(x) # filters = 64
        x = self.batch_norm(x)
        x = self.leaky_relu(x)
        
        x = self.conv3x3_4(x) # filters = 64
        x = self.batch_norm(x)
        x = self.leaky_relu(x)
        
        x = self.conv3x3_5(x) # filters = 64
        x = self.batch_norm(x)
        x = self.leaky_relu(x)
        
        x = self.conv3x3_5(x) # filters = 64
        x = self.batch_norm(x)
        x = self.leaky_relu(x)
        
        x = self.conv3x3_4(x) # filters = 64
        x = self.batch_norm(x)
        x = self.leaky_relu(x)
        
        x = self.conv3x3_3(x) # filters = 64
        x = self.batch_norm(x)
        x = self.leaky_relu(x)
        
        x = self.conv3x3_2(x) # filters = 64
        x = self.batch_norm(x)
        x = self.leaky_relu(x)
        
        x = self.conv3x3_5(x) # filter = 1
        x = self.relu(x)
        
        out = x
        return out
        