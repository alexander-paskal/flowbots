import torch.nn as nn
import torch
from .decoder import FlowNetDecoder
#import decoder as d
#from utils import HardwareManager

# Initialize the attention module as a nn.Module subclass
import math
class Attention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        # TODO: Implement the Key, Query and Value linear transforms as 1x1 convolutional layers
        
        # Hint: channel size remains constant throughout
        self.conv_query = nn.Conv2d(in_channels, in_channels, kernel_size = 1)
        self.conv_key = nn.Conv2d(in_channels, in_channels, kernel_size = 1)
        self.conv_value = nn.Conv2d(in_channels, in_channels, kernel_size = 1)

    def forward(self, x):
        N, C, H, W = x.shape
        #x = x.view(H*W,C)
        #print('x shape')
        #print(x.shape)
        # TODO: Pass the input through conv_query, reshape the output volume to (N, C, H*W)
        q = self.conv_query(x)
        #print(q.shape)
        q = q.view(N, C, H*W)
        #print('Q shape')
        #print(q.shape)
        q = q.permute(0,2,1)
        #print(q.shape)
        # TODO: Pass the input through conv_key, reshape the output volume to (N, C, H*W)
        k = self.conv_key(x)
        k = k.view(N,C,H*W)
        #print('K shape')
        #print(k.shape)
        k = k.permute(0,2,1)
        
        # TODO: Pass the input through conv_value, reshape the output volume to (N, C, H*W)
        v = self.conv_value(x)
        v = v.view(N,C,H*W)
        #print('V shape')
        #print(v.shape)
        v = v.permute(0,2,1)
        # TODO: Implement the above formula for attention using q, k, v, C
        # NOTE: The X in the formula is already added for you in the return line
        test_tensor = torch.matmul(q,k.permute(0,2,1))
        #print(test_tensor.shape)
        tensor_1 = torch.matmul(q,k.permute(0,2,1))/math.sqrt(C)
        tensor_2 = torch.nn.functional.softmax(tensor_1)
        attention = torch.matmul(tensor_2,v)
        #print('attention shape')
        #print(attention.shape)
        # Reshape the output to (N, C, H, W) before adding to the input volume
        attention = attention.reshape(N, C, H, W)
        h=x+attention
        #print('h shape')
        #print(h.shape)
        return x + attention

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
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.1, inplace=True))


        self.block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.block(x)
        return x


class FlowNetT(nn.Module):

    title = "flownet-t"

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
#atest = torch.rand(16, 6, 384, 512)
#model = FlowNetT()
#bleh = model.forward(atest)
