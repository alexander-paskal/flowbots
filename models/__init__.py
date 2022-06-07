from .flownetC import FlowNetC
from .flownetS import FlowNetS
from .flownetT3 import FlowNetT


lookup = {
    FlowNetS.title: FlowNetS,
    FlowNetC.title: FlowNetC,
    FlowNetT.title: FlowNetT
}