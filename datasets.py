from torchvision.datasets import FlyingChairs
import torchvision.transforms as T
import torch
import json

CONFIG = "config.json"


def flying_chairs(root=None, split="train"):
    """
    Gets a flying chairs dataset object
    :return:
    """
    if root is None:
        try:
            with open(CONFIG) as f:
                cfg = json.load(f)
        except FileNotFoundError:
            print("Could not load flying chairs data - config.json not found")

        root = cfg["flying_chairs"]

    dset = FlyingChairsModified(root,split=split)

    return dset


class FlyingChairsModified(FlyingChairs):
    def __getitem__(self, index):
        im1, im2, flow = super().__getitem__(index)

        im_transform = T.Compose([
            T.functional.pil_to_tensor,
            T.ConvertImageDtype(torch.float32)
        ])

        im1 = im_transform(im1)
        im2 = im_transform(im2)

        flow = T.ToTensor()(flow)
        flow = torch.transpose(flow, 0, 1)
        flow = torch.transpose(flow, 1, 2)
        flow[0, :, :] /= im1.size(1)
        flow[1, :, :] /= im1.size(2)

        return im1, im2, flow

