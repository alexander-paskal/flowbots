# Add official website of pytorch
from .decoder import FlowNetDecoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T
import torch.nn.functional as F


import numpy as np
from spatial_correlation_sampler import spatial_correlation_sample


def conv_block(in_planes, out_planes, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True)
    )



def i_conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, bias = True):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=bias),
            nn.BatchNorm2d(out_planes),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=bias),
        )

def predict_flow(in_planes):
    return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=True)

def deconv_block(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
        nn.ReLU(inplace=True)
    )

class tofp16(nn.Module):
    def __init__(self):
        super(tofp16, self).__init__()

    def forward(self, input):
        return input.half()


class tofp32(nn.Module):
    def __init__(self):
        super(tofp32, self).__init__()

    def forward(self, input):
        return input.float()


def init_deconv_bilinear(weight):
    f_shape = weight.size()
    heigh, width = f_shape[-2], f_shape[-1]
    f = np.ceil(width/2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([heigh, width])
    for x in range(width):
        for y in range(heigh):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weight.data.fill_(0.)
    for i in range(f_shape[0]):
        for j in range(f_shape[1]):
            weight.data[i,j,:,:] = torch.from_numpy(bilinear)


def save_grad(grads, name):
    def hook(grad):
        grads[name] = grad
    return hook



def correlate(input1, input2):
    out_corr = spatial_correlation_sample(input1, input2, kernel_size=1, patch_size=21, stride=1, padding=0, dilation_patch=2)
    # collate dimensions 1 and 2 in order to be treated as a
    # regular 4D tensor
    b, ph, pw, h, w = out_corr.size()
    out_corr = out_corr.view(b, ph * pw, h, w)/input1.size(1)
    return F.leaky_relu_(out_corr, 0.1)

class FlowNetC(nn.Module):

    title = "flownet-c"

    def __init__(self, in_channels = 12):
        super().__init__()

        
        #x1 block
        self.conv1 = conv_block(3, 64, kernel_size = 7, stride = 2) #(N,3,384,512) -> (N,64,192,256)
        self.conv2 = conv_block(64, 128, kernel_size = 5, stride = 2) #(N,64,192,256)-> (N,128,96,128)
        self.conv3 = conv_block(128, 256, kernel_size = 5, stride = 2) #(N,128,96,128)-> (N,256,48,64)
        #this block is for concatinating after correlation layer
        self.conv_redir = conv_block(256, 32, kernel_size=1, stride=1)
        #Correlation Layer Occurs Here
        #self.corr_layer = get_correlation_layer(pad_size=20, kernel_size=1, max_displacement=20, stride1=1, stride2=2, corr_multiply=1)
        self.conv3_1 = conv_block(473, 256, kernel_size = 3) #(N,256,48,64)-> (N,256,48,64)
        self.conv4 = conv_block(256, 512, kernel_size = 3, stride = 2) #(N,256,48,64)-> (N,512,24,32)
        self.conv4_1 = conv_block(512, 512, kernel_size = 3) #(N,512,24,32)-> (N,512,24,32)
        self.conv5 = conv_block(512, 512, kernel_size = 3, stride = 2) #(N,512,24,32)-> (N,512,12,16)
        self.conv5_1 = conv_block(512, 512, kernel_size = 3) #(N,512,12,16)-> (N,512,12,16)
        self.conv6 = conv_block(512, 1024, kernel_size = 3, stride = 2) #(N,512,12,16)-> (N,1024,6,8)
        self.conv6_1 = conv_block(1024, 1024) #(N,1024,6,8)-> (N,1024,6,8)
        
        self.decode = FlowNetDecoder()
        
    def forward(self, x):
        #x1 = input_im[0:3]
        #x2 = input_im[3:]
        #input1 as (N,3,384,512)
        #input2 as (N,3,384,512)
        x1 = x[:,0:3,:,:]
        x2 = x[:,3::,:,:]
        
        #Two branches, one for each input image
        #before reaching the correlation layer
        #in where they will be combined
        conv1_output_im1 = self.conv1(x1) #(N,3,384,512) -> (N,64,192,256)
        conv2_output_im1 = self.conv2(conv1_output_im1) #(N,64,192,256)-> (N,128,96,128)
        conv3_output_im1 = self.conv3(conv2_output_im1) #(N,128,96,128)-> (N,256,48,64)
        
        conv1_output_im2 = self.conv1(x2) #(N,3,384,512) -> (N,64,192,256)
        conv2_output_im2 = self.conv2(conv1_output_im2) #(N,64,192,256)-> (N,128,96,128)
        conv3_output_im2 = self.conv3(conv2_output_im2) #(N,128,96,128)-> (N,256,48,64)
        
        #Redirect layer, save here and use later
        conv3_redirect = self.conv_redir(conv3_output_im1) #(N,256,48,64)-> (N,32,48,64) 
        
        #Correlation layer, merge two branches
        correlate_layer = correlate(conv3_output_im1, conv3_output_im2) #(N,256,48,64)-> (N,441,48,64)
        
        #Concatenate correlation layer and redirect block
        correlation_output = torch.cat((correlate_layer, conv3_redirect), dim=1) 
        #(N,441,48,64)+(N,32,48,64) -> (N,473,48,64)
        
        #Restart convolution layers as usual
        conv3_1_output = self.conv3_1(correlation_output) #(N,473,48,64)-> (N,256,48,64)
        conv4_output = self.conv4(conv3_1_output) #(N,256,48,64)-> (N,512,24,32)
        conv4_1_output = self.conv4_1(conv4_output) #(N,512,24,32)-> (N,512,24,32)
        conv5_output = self.conv5(conv4_1_output) #(N,512,24,32)-> (N,512,24,32)
        conv5_1_output = self.conv5_1(conv5_output) #(N,512,24,32)-> (N,512,12,16)
        conv6_output = self.conv6(conv5_1_output) #(N,512,12,16)-> (N,1024,6,8)
        conv6_1_output = self.conv6_1(conv6_output) #(N,1024,6,8)-> (N,1024,6,8)
        
        #pass in a tuple
        output_tuple = (conv1_output_im1, conv2_output_im1, conv3_output_im1, conv4_1_output, conv5_1_output, conv6_1_output)
        decoder_output = self.decode(output_tuple)
        
        
        return decoder_output











    
    