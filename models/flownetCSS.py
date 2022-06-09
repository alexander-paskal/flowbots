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
NET2 = "flownet-ss-first-0607"

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

def load_net2(name):
    """

    :param name:
    :return:
    """


    model_cls = FlownetSS
    model = model_cls()

    model.load_state_dict(torch.load(os.path.join(PARAMETERS_DIR, name + ".pth"), map_location=torch.device("cpu")))

    return model.net2

class FlownetCSS(Base):

    title = "flownet-css"

    def __init__(self):
        super().__init__()

        self.net1 = load_net1(NET1)

        for parameter in self.net1.parameters():
            parameter.requires_grad = False

        self.net2 = load_net2(NET2)
        self.net3 = FlowNetS(in_channels=8)

    def forward(self, x):

        flow = self.net1(x)

        x1 = torch.cat([x, flow], dim=1)
        flow2 = self.net2(x1)
        x2 = torch.cat([x, flow2], dim=1)
        prediction = self.net3(x2)
        return prediction


if __name__ == '__main__':
    m = FlownetSSWarped()
    t = torch.zeros((10, 6, 384, 512))
    output = m.forward(t)
    print(output.size())

