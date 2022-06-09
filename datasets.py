from torchvision.datasets import FlyingChairs, Sintel, HD1K
import torchvision.transforms as T
import torch.nn.functional as F
import torch
import json

CONFIG = "config.json"
INPUT_SIZE = (384, 512)
LABEL_SIZE = (384, 512)


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
            return
        root = cfg["flying_chairs"]

    dset = FlyingChairsModified(root,split=split)

    return dset


def sintel(root=None, split="train", pass_name="clean", interpolate=True):
    """
    Gets a sintel dataset object
    :param root:
    :param split:
    :param pass_name:
    :return:
    """
    if root is None:
        try:
            with open(CONFIG) as f:
                cfg = json.load(f)
        except FileNotFoundError:
            print("Could not load flying chairs data - config.json not found")
            return
        root = cfg["flying_chairs"]

    dset = SintelModified(root, split=split, pass_name=pass_name, interpolate=interpolate)
    return dset


def hd1k(root=None, split="train", interpolate=True):
    """
    Gets a sintel dataset object
    :param root:
    :param split:
    :param pass_name:
    :return:
    """
    if root is None:
        try:
            with open(CONFIG) as f:
                cfg = json.load(f)
        except FileNotFoundError:
            print("Could not load flying chairs data - config.json not found")
            return
        root = cfg["flying_chairs"]

    dset = HD1KModified(root, split=split, interpolate=interpolate)
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

        im_concat = torch.cat([im1, im2], dim=0)

        return im_concat, flow


class FlyingThings3D:
    pass


class SintelModified(Sintel):
    def __init__(self, *args, interpolate=False, **kwargs):
        self.interpolate = interpolate
        super().__init__(*args, **kwargs)

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

        # for sintel
        flow = F.interpolate(flow[None, :, :, :], size=LABEL_SIZE).squeeze()

        # flow[0, :, :] /= im1.size(1)
        # flow[1, :, :] /= im1.size(2)

        im_concat = torch.cat([im1, im2], dim=0)

        # for sintel
        im_concat = F.interpolate(im_concat[None, :, :, :], size=INPUT_SIZE).squeeze()
        return im_concat, flow


class HD1KModified(HD1K):
    def __init__(self, *args, interpolate=False, **kwargs):
        self.interpolate = interpolate
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        im1, im2, flow, *_ = super().__getitem__(index)

        im_transform = T.Compose([
            T.functional.pil_to_tensor,
            T.ConvertImageDtype(torch.float32)
        ])

        im1 = im_transform(im1)
        im2 = im_transform(im2)

        flow = T.ToTensor()(flow)
        flow = torch.transpose(flow, 0, 1)
        flow = torch.transpose(flow, 1, 2)

        flow = F.interpolate(flow[None, :, :, :], size=LABEL_SIZE).squeeze()

        # flow[0, :, :] /= im1.size(1)
        # flow[1, :, :] /= im1.size(2)

        im_concat = torch.cat([im1, im2], dim=0)

        im_concat = F.interpolate(im_concat[None, :, :, :], size=INPUT_SIZE).squeeze()
        return im_concat, flow
