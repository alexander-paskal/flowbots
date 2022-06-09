from .flownetC import FlowNetC
from .flownetS import FlowNetS
from .flownetT3 import FlowNetT
from .flownet_stacked import FlownetStacked
from .flownetSS import FlownetSS
from .flownetSS_warped import FlownetSSWarped
from .flownetCSS_warped import FlownetCSSWarped

lookup = {
    FlowNetS.title: FlowNetS,
    FlowNetC.title: FlowNetC,
    FlowNetT.title: FlowNetT,
    FlownetStacked.title: FlownetStacked,
    FlownetSS.title: FlownetSS,
    FlownetSSWarped.title: FlownetSSWarped,
    FlownetCSSWarped.title: FlownetCSSWarped
}