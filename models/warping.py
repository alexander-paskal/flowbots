from .base import Base
import torch.nn as nn
import torch
import torch.nn.functional as F

USE_GPU = True
if torch.cuda.is_available() and USE_GPU:
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")



class WarpingLayer:
    def forward(self, x):
        """
        Warps image 1
        :param x:
        :return:
        """

        im1 = x[:, :3, :, :]
        im2 = x[:, 3:6, :, :]
        flow = x[:, 6:, :, :]

        im2_warped = self.warping(im1, flow)
        error = torch.norm(im2_warped - im2, dim=1)[:, None, :, :]

        output = torch.concat([im1, im2_warped, im2, flow, error], dim=1)
        return output

    @staticmethod
    def warping(image, flow, interp=False):
        """
        Performs warping on an image
        """
        with torch.no_grad():
            B, C, H, W = image.size()

            image_inds = torch.arange(H * W, dtype=torch.int64).to(DEVICE)
            image_inds = image_inds.repeat(B, 1)

            has_flow = torch.sum(torch.abs(flow), dim=1, dtype=torch.bool).view((B, H * W)).to(DEVICE)
            warped_inds = torch.zeros((B, H * W), dtype=torch.int64).to(DEVICE)

            warped_inds[has_flow] += image_inds[has_flow]

            flow_ind_shift = torch.tensor((flow[:, 0, :, :] * H + flow[:, 1, :, :]),
                                          dtype=torch.int64).view(B, H * W).to(DEVICE)
            warped_inds[has_flow] += flow_ind_shift[has_flow]
            warped_inds = torch.clamp(warped_inds, 0, H*W-1)
            # batch offset
            offset = torch.cat([torch.ones(H * W, dtype=torch.int64).to(DEVICE) * H * W * i for i in range(B)]).view((B, H * W))
            warped_inds = torch.flatten(warped_inds + offset)
            image_inds = torch.flatten(image_inds + offset)

            warped_im = torch.zeros((B, C, H, W)).to(DEVICE)
            for channel in (0, 1, 2):
                im_channel = torch.flatten(image[:, channel, :, :])
                warped_channel = torch.flatten(warped_im[:, channel, :, :])
                warped_channel[warped_inds] = im_channel[image_inds]
                warped_im[:, channel, :, :] = warped_channel.view((B, H, W))

        return warped_im
