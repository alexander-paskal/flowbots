from .base import Base
import torch.nn as nn
import torch
import torch.nn.functional as F


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


class WarpingLayer(nn.Module):
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
    def warping(img, flow, interp=False):
        """
        Performs warping on an image
        """

        new_img = torch.zeros(img.size()).to(DEVICE)
        *_, height, width = img.size()

        for i in range(height):
            for j in range(width):
                flow_ij = flow[:, :, i, j]  # -> N x 2
                FLOWY = flow_ij[0]  # N x 1
                FLOWX = flow_ij[1]  # N x 1

                for n, (flowx, flowy) in enumerate(zip(FLOWX.tolist(), FLOWY.tolist())):
                    x = round(flowx + i)
                    y = round(flowy + j)
                    if 0 <= x < height and 0 <= y < width:
                        rgb = img[n, :, i, j]
                        new_img[n, :, x, y] = rgb
                    else:
                        pass

        # interpolation
        if interp:
            new_img = F.interpolate(new_img, size=(384, 512), mode="bilinear")

        return new_img


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

            if frozen[i + 1]:
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

