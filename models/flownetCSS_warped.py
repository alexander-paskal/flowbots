from .flownetS import FlowNetS
from .flownetC import FlowNetC
from .flownetSS import FlownetSS
from .base import Base
from .warping import WarpingLayer
import torch
import os

"""
For training two whole flownets stacked together
"""

NET1 = "FlowNetC_FlyingChairs_scheduler_long"

PARAMETERS_DIR = "weights"
def load_net1(name):
    """

    :param name:
    :return:
    """


    model_cls = FlowNetC
    model = model_cls()

    model.load_state_dict(torch.load(os.path.join(PARAMETERS_DIR, name + ".pth"), map_location=torch.device("cpu")))

    return model


class FlownetCSS(Base):

    title = "flownet-css"

    def __init__(self):
        super().__init__()

        self.net1 = load_net1(NET1)

        for parameter in self.net1.parameters():
            parameter.requires_grad = False

        self.net2 = FlowNetS(in_channels=12)

        # # Uncomment when training net3 final training
        # for parameter in self.net2.parameters():
        #     parameter.requires_grad = False

        self.net3 = FlowNetS(in_channels=12)
        self.warping = WarpingLayer()

    def forward(self, x):

        flow = self.net1(x)

        x1 = torch.cat([x, flow], dim=1)
        x1_warped = self.warping.forward(x1)
        flow2 = self.net2(x1_warped)

        # Comment out when training net 3
        # return flow2

        x2 = torch.cat([x, flow2], dim=1)
        x2_warped = self.warping.forward(x2)
        prediction = self.net3(x2_warped)
        return prediction




