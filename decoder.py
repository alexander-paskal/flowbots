import torch.nn as nn
import torch



class FlowNetDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.deconv5_1 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=True)
        self.deconv5_2 = nn.LeakyReLU(0.1,inplace=True)
        self.deconv4_1 = nn.ConvTranspose2d(1026, 256, kernel_size=4, stride=2, padding=1, bias=True)
        self.deconv4_2 = nn.LeakyReLU(0.1, inplace=True)
        self.deconv3_1 = nn.ConvTranspose2d(770, 128, kernel_size=4, stride=2, padding=1, bias=True)
        self.deconv3_2 = nn.LeakyReLU(0.1, inplace=True)
        self.deconv2_1 = nn.ConvTranspose2d(386, 64, kernel_size=4, stride=2, padding=1, bias=True)
        self.deconv2_2 = nn.LeakyReLU(0.1, inplace=True)

        self.im_to_flow6 = nn.Conv2d(1024,2,kernel_size=3,stride=1,padding=1,bias=True)
        self.im_to_flow5 = nn.Conv2d(1026,2,kernel_size=3,stride=1,padding=1,bias=True)
        self.im_to_flow4 = nn.Conv2d(770,2,kernel_size=3,stride=1,padding=1,bias=True)
        self.im_to_flow3 = nn.Conv2d(386,2,kernel_size=3,stride=1,padding=1,bias=True)
        self.im_to_flow2 = nn.Conv2d(194,2,kernel_size=3,stride=1,padding=1,bias=True)

        self.upsample_6 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsample_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsample_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsample_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, x):
        """
        Takes in tuple of out convolutions
        :param x: 
        :return: 
        """
        out1, out2, out3, out4, out5, out6 = x

        flow6 = self.im_to_flow6(out6)
        flow6_up = self.upsample_6(flow6)
        out_deconv5 = self.deconv5_2(self.deconv5_1(out6))

        concat5 = torch.cat((out5, out_deconv5, flow6_up), 1)
        flow5 = self.im_to_flow5(concat5)
        flow5_up = self.upsample_5(flow5)
        out_deconv4 = self.deconv4_2(self.deconv4_1(concat5))

        concat4 = torch.cat((out4, out_deconv4, flow5_up), 1)
        flow4 = self.im_to_flow4(concat4)
        flow4_up = self.upsample_4(flow4)
        out_deconv3 = self.deconv3_2(self.deconv3_1(concat4))

        concat3 = torch.cat((out3, out_deconv3, flow4_up), 1)
        flow3 = self.im_to_flow3(concat3)
        flow3_up = self.upsample_3(flow3)
        out_deconv2 = self.deconv2_2(self.deconv2_1(concat3))

        concat2 = torch.cat((out2, out_deconv2, flow3_up), 1)
        flow2 = self.im_to_flow2(concat2)

        return self.upsample1(flow2)

if __name__ == '__main__':
    expected_sizes = [
        (1, 64, 192, 256),
        (1, 128, 96, 128),
        (1, 256, 48, 64),
        (1, 512, 24, 32),
        (1, 512, 12, 16),
        (1, 1024, 6, 8),
    ]

    x = [
        torch.zeros(s) for s in expected_sizes
    ]
    d = FlowNetDecoder()
    print(d.forward(x).size())