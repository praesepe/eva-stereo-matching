
# coding: utf-8

# In[1]:


import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import argparse
from tensorboardX import SummaryWriter
import argparse
import os
import os.path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'


# In[2]:


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.elu = nn.ELU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.elu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.elu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.elu(out)

        return out


# In[3]:


class DecoderBlock(nn.Module):

    def __init__(self, inplanes, planes, mid, stride=1):
        super(DecoderBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(mid, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.elu = nn.ELU()

    def forward(self, x, skip, udisp=None):
        x = self.upsample(x)
        x = self.elu(self.bn1(self.conv1(x)))
        #concat
        if udisp is not None:
            x = torch.cat((x, skip, udisp), 1)
        else:
            x = torch.cat((x, skip), 1)
        x = self.elu(self.bn2(self.conv2(x)))
        return x


# In[4]:


class DispBlock(nn.Module):
    
    def __init__(self, inplanes, planes=2, kernel=3, stride=1):
        super(DispBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, 2, kernel_size=kernel, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(2)
        self.sigmoid = nn.Sigmoid()
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x):
        disp = self.sigmoid(self.bn1(self.conv1(x)))
        udisp = self.upsample(disp)
        return disp, udisp


# In[5]:


class MonodepthNet(nn.Module):

    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3], num_classes=1000):
        self.inplanes = 64
        super(MonodepthNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.elu = nn.ELU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.up6 = DecoderBlock(2048, 512, 1536)
        self.up5 = DecoderBlock(512, 256, 768)
        self.up4 = DecoderBlock(256, 128, 384)
        self.up3 = DecoderBlock(128, 64, 130)
        self.up2 = DecoderBlock(64, 32, 98)
        self.up1 = DecoderBlock(32, 16, 18)
        self.get_disp4 = DispBlock(128)
        self.get_disp3 = DispBlock(64)
        self.get_disp2 = DispBlock(32)
        self.get_disp1 = DispBlock(16)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        #encoder
        x = self.conv1(x) #2
        x = self.bn1(x)
        conv1 = self.elu(x)
        pool1 = self.maxpool(conv1) #4
        conv2 = self.layer1(pool1) #8
        conv3 = self.layer2(conv2) #16
        conv4 = self.layer3(conv3) #32
        conv5 = self.layer4(conv4) #64
        
        #skip
        skip1 = conv1
        skip2 = pool1
        skip3 = conv2
        skip4 = conv3
        skip5 = conv4
        
        #decoder
        upconv6 = self.up6(conv5, skip5)
        upconv5 = self.up5(upconv6, skip4)
        upconv4 = self.up4(upconv5, skip3)
        self.disp4, udisp4 = self.get_disp4(upconv4)
        upconv3 = self.up3(upconv4, skip2, udisp4)
        self.disp3, udisp3 = self.get_disp3(upconv3)
        upconv2 = self.up2(upconv3, skip1, udisp3)
        self.disp2, udisp2 = self.get_disp2(upconv2)
        upconv1 = self.up1(upconv2, udisp2)
        self.disp1, udisp1 = self.get_disp1(upconv1)
        
        return [self.disp1, self.disp2, self.disp3, self.disp4]
        

