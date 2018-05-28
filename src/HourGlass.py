
'''
    file:   hourglass.py

    date:   2018_05_12
    author: zhangxiong(1025679612@qq.com)
'''

from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Residual(nn.Module):
    def __init__(self, use_bn, input_channels, out_channels, mid_channels):
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

        self.conv = nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size = 3, padding = 1)

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

class HourGlassBlock(nn.Module):
    def __init__(self, block_count, residual_each_block, input_channels, mid_channels, use_bn, stack_index):
        super(HourGlassBlock, self).__init__()

        self.block_count         = block_count
        self.residual_each_block = residual_each_block
        self.use_bn              = use_bn
        self.stack_index         = stack_index
        self.input_channels      = input_channels
        self.mid_channels        = mid_channels

        if self.block_count == 0: #inner block
            self.process = nn.Sequential()
            for _ in range(residual_each_block * 3):
                self.process.add_module(
                    name = 'inner_{}_{}'.format(self.stack_index, _),
                    module = Residual(input_channels = input_channels, out_channels = input_channels, mid_channels = mid_channels, use_bn = use_bn)
                )
        else:
            #down sampling
            self.down_sampling = nn.Sequential()
            self.down_sampling.add_module(
                name = 'down_sample_{}_{}'.format(self.stack_index, self.block_count), 
                module = nn.MaxPool2d(kernel_size = 2, stride = 2)
            )
            for _ in range(residual_each_block):
                self.down_sampling.add_module(
                    name = 'residual_{}_{}_{}'.format(self.stack_index, self.block_count, _),
                    module = Residual(input_channels = input_channels, out_channels = input_channels, mid_channels = mid_channels, use_bn = use_bn)
                )
            
            #up sampling
            self.up_sampling = nn.Sequential()
            self.up_sampling.add_module(
                name = 'up_sample_{}_{}'.format(self.stack_index, self.block_count),
                module = nn.Upsample(scale_factor=2, mode='bilinear')
            )
            for _ in range(residual_each_block):
                self.up_sampling.add_module(
                    name   = 'residual_{}_{}_{}'.format(self.stack_index, self.block_count, _),
                    module = Residual(input_channels = input_channels, out_channels = input_channels, mid_channels = mid_channels, use_bn = use_bn)
                )

            #sub hour glass
            self.sub_hg = HourGlassBlock(
                block_count         = self.block_count - 1, 
                residual_each_block = self.residual_each_block, 
                input_channels      = self.input_channels,
                mid_channels        = self.mid_channels,
                use_bn              = self.use_bn,
                stack_index         = self.stack_index
            )
            
            # trans
            self.trans = nn.Sequential()
            for _ in range(residual_each_block):
                self.trans.add_module(
                    name = 'trans_{}_{}_{}'.format(self.stack_index, self.block_count, _),
                    module = Residual(input_channels = input_channels, out_channels = input_channels, mid_channels = mid_channels, use_bn = use_bn)
            )


    def forward(self, inputs):
        if self.block_count == 0:
            return self.process(inputs)
        else:
            down_sampled        = self.down_sampling(inputs)
            transed             = self.trans(down_sampled)
            sub_net_output      = self.sub_hg(down_sampled)
            return self.up_sampling(transed + sub_net_output)

'''
    the input is a 256 x 256 x 3 image
'''
class HourGlass(nn.Module):
    def __init__(self, nStack, nBlockCount, nResidualEachBlock, nMidChannels, nChannels, nJointCount, bUseBn):
        super(HourGlass, self).__init__()

        self.nStack             = nStack
        self.nBlockCount        = nBlockCount
        self.nResidualEachBlock = nResidualEachBlock
        self.nChannels          = nChannels
        self.nMidChannels       = nMidChannels
        self.nJointCount        = nJointCount
        self.bUseBn             = bUseBn

        self.pre_process = nn.Sequential(
            nn.Conv2d(3, nChannels, kernel_size = 3, padding = 1),
            Residual(use_bn = bUseBn, input_channels = nChannels, out_channels = nChannels, mid_channels = nMidChannels),
            nn.MaxPool2d(kernel_size = 2, stride = 2), # to 128 x 128 x c
            Residual(use_bn = bUseBn, input_channels = nChannels, out_channels = nChannels, mid_channels = nMidChannels),
            nn.MaxPool2d(kernel_size = 2, stride = 2), # to 64 x 64 x c
            Residual(use_bn = bUseBn, input_channels = nChannels, out_channels = nChannels, mid_channels = nMidChannels)
        )

        self.hg = nn.ModuleList()
        for _ in range(nStack):
            self.hg.append(
                HourGlassBlock(
                    block_count = nBlockCount, 
                    residual_each_block = nResidualEachBlock,
                    input_channels = nChannels, 
                    mid_channels = nMidChannels, 
                    use_bn = bUseBn,
                    stack_index = _
                )
            )

        self.blocks = nn.ModuleList()
        for _ in range(nStack - 1):
            self.blocks.append(
                nn.Sequential(
                    Residual(
                        use_bn = bUseBn, input_channels = nChannels, out_channels = nChannels, mid_channels = nMidChannels 
                    ),
                    Residual(
                        use_bn = bUseBn, input_channels = nChannels, out_channels = nChannels, mid_channels = nMidChannels
                    )
                )
            )
        
        self.intermediate_supervision = nn.ModuleList()
        for _ in range(nStack): # to 64 x 64 x joint_count
            self.intermediate_supervision.append(
                nn.Conv2d(nChannels, nJointCount, kernel_size = 1, stride = 1)
            )

        self.normal_feature_channel = nn.ModuleList()
        for _ in range(nStack - 1):
            self.normal_feature_channel.append(
                Residual(
                    use_bn = bUseBn, input_channels = nJointCount, out_channels = nChannels, mid_channels = nMidChannels
                )
            )

    def forward(self, inputs):
        o = [] #outputs include intermediate supervision result
        x = self.pre_process(inputs)
        for _ in range(self.nStack):
            o1 = self.hg[_](x)
            o2 = self.intermediate_supervision[_](o1)
            o.append(o2.view(-1, 4096))
            if _ == self.nStack - 1:
                break
            o2 = self.normal_feature_channel[_](o2)
            o1 = self.blocks[_](o1)
            x = o1 + o2 + x
        return o

def _create_hourglass_net():
    return HourGlass(
        nStack = 2,
        nBlockCount = 4,
        nResidualEachBlock = 1,
        nMidChannels = 128,
        nChannels = 256,
        nJointCount = 1,
        bUseBn = True,
    )