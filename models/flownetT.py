"""
Flownet built on top of DPT backbone
"""
from models.dpt.models import DPT
from models.dpt.blocks import Interpolate
import torch
import torch.nn as nn
from timm.models.layers import PatchEmbed


class FlowNetT(DPT):
    def __init__(self, path=None, **kwargs):

        features = kwargs["features"] if "features" in kwargs else 256

        kwargs["use_bn"] = True

        head = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            nn.Dropout(0.1, False),
            nn.Conv2d(features, 2, kernel_size=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        )

        super().__init__(head, **kwargs)


        self.pretrained.model.patch_embed = PatchEmbed(
            img_size=(384, 512),
            patch_size=16,
            in_chans=6,
        embed_dim=768)

        if path is not None:
            self.load(path)

    def load(self, path):
        """
        Modified load method - loads parameters from file path except for those
        in the head layer, to enable transfer learning

        Args:
            path (str): file path
        """
        parameters = torch.load(path, map_location=torch.device("cpu"))

        if "optimizer" in parameters:
            parameters = parameters["model"]

        pop_keys = [k for k in parameters if "output_conv" in k]
        for k in pop_keys:
            parameters.pop(k)

        self.load_state_dict(parameters, strict=False)


