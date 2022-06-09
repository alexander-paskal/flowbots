from .flownetS import FlowNetS
from .base import Base
from .warping import WarpingLayer
import torch
import os

"""
For training two whole flownets stacked together
"""

NET1 = "simple_s"
PARAMETERS_DIR = "weights"
def load_net1(name):
    """

    :param name:
    :return:
    """


    model_cls = FlowNetS
    model = model_cls()

    model.load_state_dict(torch.load(os.path.join(PARAMETERS_DIR, name + ".pth"), map_location=torch.device("cpu")))

    return model


class FlownetSSWarped(Base):

    title = "flownet-ss-warped"

    def __init__(self):
        super().__init__()

        self.net1 = load_net1(NET1)

        for parameter in self.net1.parameters():
            parameter.requires_grad = False

        self.net2 = FlowNetS(in_channels=12)
        self.warping = WarpingLayer()

    def forward(self, x):

        flow = self.net1(x)

        x1 = torch.cat([x, flow], dim=1)
        x_warped = self.warping.forward(x1)

        prediction = self.net2(x_warped)

        return prediction


if __name__ == '__main__':
    m = FlownetSSWarped()
    t = torch.zeros((10, 6, 384, 512))
    output = m.forward(t)
    print(output.size())

