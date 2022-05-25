import torch.nn as nn
import torch
from .decoder import FlowNetDecoder
from utils import HardwareManager


class Attention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.conv_query = nn.Conv2d(in_channels, in_channels, 1)
        self.conv_key = nn.Conv2d(in_channels, in_channels, 1)
        self.conv_value = nn.Conv2d(in_channels, in_channels, 1)
        self.smax = torch.nn.Softmax().to(HardwareManager.get_device())

    def forward(self, x):
        N, C, H, W = x.shape

        xwq = self.conv_query(x)
        xwq = xwq.view((N, C, H * W))

        xwk = self.conv_key(x)
        xwk = xwk.view((N, C, H * W))

        xwv = self.conv_key(x)
        xwv = xwv.view((N, C, H * W))

        print(x.get_device())
        print(xwv.get_device())
        x = x.view((N, C, -1))
        x = torch.transpose(x, 1, 2)

        top_left = xwq
        top_right = xwk
        top_left = torch.transpose(top_left, 1, 2)

        top = torch.matmul(top_left, top_right)
        bottom = torch.sqrt(torch.tensor(C))

        middle = self.smax(top / bottom)
        right = torch.transpose(xwv, 1, 2)
        right = torch.matmul(middle, right)
        attention = x + right
        attention = attention.reshape(N, C, H, W)
        return attention


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, attention=False, *args, **kwargs):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, *args, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        ]

        if attention:
            attention_layer = Attention(out_channels)
            layers.append(attention_layer)


        self.block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.block(x)
        return x


class FlowNetT(nn.Module):

    title = "flownet-s"

    def __init__(self, in_channels=6):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = ConvBlock(64, 128, kernel_size=5, stride=2, padding=2)
        self.conv3_1 = ConvBlock(128, 256, kernel_size=5, stride=2, padding=2)
        self.conv3_2 = ConvBlock(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = ConvBlock(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv4_2 = ConvBlock(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = ConvBlock(512, 512, kernel_size=3, stride=2, padding=1, attention=True)
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


