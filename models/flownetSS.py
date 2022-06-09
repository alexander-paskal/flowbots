from .flownetS import FlowNetS
from .base import Base
import torch.nn as nn
import torch
"""
For training two whole flownets stacked together
"""


class FlownetSS(Base):

    title = "flownet-SS"

    def __init__(self):
        super().__init__()

        self.net1 = FlowNetS(in_channels=6)
        self.net2 = FlowNetS(in_channels=8)

    def forward(self, x):

        flow = self.net1(x)

        x1 = torch.cat([x, flow], dim=1)

        prediction = self.net2(x1)

        return prediction


if __name__ == '__main__':
    m = FlownetSS()
    t = torch.zeros((10, 6, 384, 512))
    output = m.forward(t)
    print(output.size())


