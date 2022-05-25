from models.flownetT2 import FlowNetT
import torch
from utils import HardwareManager


if __name__ == '__main__':
    t = torch.zeros((100, 6, 384, 512))
    t = t.to(HardwareManager.get_device())
    m = FlowNetT().to(HardwareManager.get_device())

    o = m.forward(t)
    print(o.size())