

'''
    file:   PRnetEncoder.py

    date:   2018_05_22
    author: zhangxiong(1025679612@qq.com)
    mark:   the algorithm is cited from PRNet code
'''

from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Residual(nn.Module):
    def __init__(self, use_bn, input_channels, out_channels, mid_channels, kernel_size = 3, padding = 1, stride = 1):
        super(Residual, self).__init__()
        self.use_bn = use_bn
        self.out_channels   = out_channels
        self.input_channels = input_channels
        self.mid_channels   = mid_channels

        self.down_channel = nn.Conv2d(input_channels, self.mid_channels, kernel_size = 1)
        self.AcFunc       = nn.ReLU()
        if use_bn:
            self.bn_0 = nn.BatchNorm2d(num_features = self.mid_channels)
            self.bn_1 = nn.BatchNorm2d(num_features = self.mid_channels)
            self.bn_2 = nn.BatchNorm2d(num_features = self.out_channels)

        self.conv = nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size = kernel_size, padding = padding, stride = stride)

        self.up_channel = nn.Conv2d(self.mid_channels, out_channels, kernel_size= 1)

        if input_channels != out_channels:
            self.trans = nn.Conv2d(input_channels, out_channels, kernel_size = 1)
    
    def forward(self, inputs):
        x = self.down_channel(inputs)
        if self.use_bn:
            x = self.bn_0(x)
        x = self.AcFunc(x)

        x = self.conv(x)
        if self.use_bn:
            x = self.bn_1(x)
        x = self.AcFunc(x)

        x = self.up_channel(x)

        if self.input_channels != self.out_channels:
            x += self.trans(inputs)
        else:
            x += inputs

        if self.use_bn:
            x = self.bn_2(x)
        
        return self.AcFunc(x)

class PRNetEncoder(nn.Module):
    def __init__(self):
        super(PRNetEncoder, self).__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 8, kernel_size = 3, stride = 1, padding = 1), # to 256 x 256 x 8
            nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3, stride = 1, padding = 1), # to 256 x 256 x 16
            Residual(use_bn = True, input_channels = 16, out_channels = 32, mid_channels = 16, stride = 1, padding = 1), # to 256 x 256 x 32
            nn.MaxPool2d(kernel_size = 2, stride = 2), # to 128 x 128 x 32
            Residual(use_bn = True, input_channels = 32, out_channels = 32, mid_channels = 16, stride = 1, padding = 1), # to 128 x 128 x 32
            Residual(use_bn = True, input_channels = 32, out_channels = 32, mid_channels = 16, stride = 1, padding = 1), # to 128 x 128 x 32
            Residual(use_bn = True, input_channels = 32, out_channels = 64, mid_channels = 32, stride = 1, padding = 1), # to 128 x 128 x 64
            nn.MaxPool2d(kernel_size = 2, stride = 2), # to 64 x 64 x 64
            Residual(use_bn = True, input_channels = 64, out_channels = 64, mid_channels = 32, stride = 1, padding = 1), # to 64 x 64 x 64
            Residual(use_bn = True, input_channels = 64, out_channels = 64, mid_channels = 32, stride = 1, padding = 1), # to 64 x 64 x 64
            Residual(use_bn = True, input_channels = 64, out_channels = 128, mid_channels = 64, stride = 1, padding = 1), # to 64 x 64 x 128
            nn.MaxPool2d(kernel_size = 2, stride = 2), # to 32 x 32 x 128
            Residual(use_bn = True, input_channels = 128, out_channels = 128, mid_channels = 64, stride = 1, padding = 1), # to 32 x 32 x 128
            Residual(use_bn = True, input_channels = 128, out_channels = 128, mid_channels = 64, stride = 1, padding = 1), # to 32 x 32 x 128
            Residual(use_bn = True, input_channels = 128, out_channels = 256, mid_channels = 128, stride = 1, padding = 1), # to 32 x 32 x 256
            nn.MaxPool2d(kernel_size = 2, stride = 2), # to 16 x 16 x 256
            Residual(use_bn = True, input_channels = 256, out_channels = 256, mid_channels = 128, stride = 1, padding = 1), # to 16 x 16 x 256
            Residual(use_bn = True, input_channels = 256, out_channels = 256, mid_channels = 128, stride = 1, padding = 1), # to 16 x 16 x 256
            Residual(use_bn = True, input_channels = 256, out_channels = 512, mid_channels = 256, stride = 1, padding = 1), # to 16 x 16 x 512
            nn.MaxPool2d(kernel_size = 2, stride = 2), # to 8 x 8 x 512
            Residual(use_bn = True, input_channels = 512, out_channels = 512, mid_channels = 256, stride = 1, padding = 1), # to 8 x 8 x 512
            nn.MaxPool2d(kernel_size = 2, stride = 2) , # to 4 x 4 x 512
            Residual(use_bn = True, input_channels = 512, out_channels = 512, mid_channels = 256, stride = 1, padding = 1), # to 4 x 4 x 512
            nn.MaxPool2d(kernel_size = 2, stride = 2), # to 2 x 2 x 512
            Residual(use_bn = True, input_channels = 512, out_channels = 512, mid_channels = 256, stride = 1, padding = 1) # to 2 x 2 x 512
        )
    
    def forward(self, inputs):
        return self.conv_blocks(inputs).view(-1, 2048)


if __name__ == '__main__':
    net = PRNetEncoder()
    inputs = torch.ones(size = (10, 3, 256, 256)).float()
    r = net(inputs)
    print(r.shape)