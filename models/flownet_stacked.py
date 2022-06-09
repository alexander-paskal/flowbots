from .base import Base
import torch.nn as nn
import torch
import torch.nn.functional as F

USE_GPU = False
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

            image_inds = torch.arange(H * W, dtype=torch.int64)
            image_inds = image_inds.repeat(B, 1)

            has_flow = torch.sum(torch.abs(flow), dim=1, dtype=torch.bool).view((B, H * W))
            warped_inds = torch.zeros((B, H * W), dtype=torch.int64)

            warped_inds[has_flow] += image_inds[has_flow]

            flow_ind_shift = torch.tensor((flow[:, 0, :, :] * H + flow[:, 1, :, :]),
                                          dtype=torch.int64).view(B, H * W)
            warped_inds[has_flow] += flow_ind_shift[has_flow]
            warped_inds = torch.clamp(warped_inds, 0, H*W-1)
            # batch offset
            offset = torch.cat([torch.ones(H * W, dtype=torch.int64) * H * W * i for i in range(B)]).view((B, H * W))
            warped_inds = torch.flatten(warped_inds + offset)
            image_inds = torch.flatten(image_inds + offset)

            warped_im = torch.zeros((B, C, H, W))
            for channel in (0, 1, 2):
                im_channel = torch.flatten(image[:, channel, :, :])
                warped_channel = torch.flatten(warped_im[:, channel, :, :])
                warped_channel[warped_inds] = im_channel[image_inds]
                warped_im[:, channel, :, :] = warped_channel.view((B, H, W))

        return warped_im


class FlownetStacked(Base):
    def __init__(self, *args, warping=False, frozen=None):
        super().__init__()

        if frozen is None:
            frozen = [False for _ in args]

        self.warping = warping
        if warping:
            dims = 12
        else:
            dims = 8

        if len(args) == 0:
            raise RuntimeError("No base models specified for stacked network")

        first_model = args[0]

        if isinstance(first_model, type):  # passed a CLASS and not an INSTANCE
            first_model = first_model()
        if frozen[0]:
            for parameter in first_model.parameters():
                parameter.requires_grad = False

        models = nn.ModuleList()
        models.append(first_model)

        for i, arg in enumerate(args[1:]):  # for all other models passed
            if isinstance(arg, type):  # passed a model class and not an instance
                arg = arg(dims)

            models.append(arg)

            if frozen[i+1]:
                for parameter in arg.parameters:
                    parameter.requires_grad = False

        # self.sequential = nn.Sequential(*models)
        self.models = models
        self.warping_layer = WarpingLayer()

    def forward(self, x):
        """
        Takes in the original input, which is a N x 6 x H x W tensor with im1 and im2 concatenated
        :param x:
        :return:
        """

        im1 = x[:, :3, :, :]
        im2 = x[:, 3:, :, :]

        for i, model in enumerate(self.models):
            flow = model(x)
            if i == len(self.models) - 1:
                break

            x = torch.concat([im1, im2, flow], dim=1)  # N x 8 x H x W
            if self.warping:
                x = self.warping_layer.forward(x)  # -> N x 12 x H x W

        return flow

