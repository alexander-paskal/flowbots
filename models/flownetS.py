import torch.nn as nn
import torch
from .decoder import FlowNetDecoder
from .base import Base

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,*args, **kwargs):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, *args, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class FlowNetS(Base):

    title = "flownet-s"

    def __init__(self, in_channels=6):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = ConvBlock(64, 128, kernel_size=5, stride=2, padding=2)
        self.conv3_1 = ConvBlock(128, 256, kernel_size=5, stride=2, padding=2)
        self.conv3_2 = ConvBlock(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = ConvBlock(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv4_2 = ConvBlock(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = ConvBlock(512, 512, kernel_size=3, stride=2, padding=1)
        self.conv5_2 = ConvBlock(512, 512, kernel_size=3, padding=1)
        self.conv6_1 = ConvBlock(512, 1024, kernel_size=3, stride=2, padding=1)
        self.conv6_2 = ConvBlock(1024, 1024, kernel_size=3, padding=1)

        self.decoder = FlowNetDecoder()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3_2(self.conv3_1(x2))
        x4 = self.conv4_2(self.conv4_1(x3))
        x5 = self.conv5_2(self.conv5_1(x4))
        x6 = self.conv6_2(self.conv6_1(x5))

        x = self.decoder(
            [x1, x2, x3, x4, x5, x6]
        )
        return x


