import numpy as np
import torch



def warping(image, flow):

    B, C, H, W = image.size()


    image_inds = torch.arange(H * W, dtype=torch.int64)
    image_inds = image_inds.repeat(B, 1)

    has_flow = torch.sum(torch.abs(flow), dim=1, dtype=torch.bool).view((B, H*W))
    warped_inds = torch.zeros((B, H*W), dtype=torch.int64)

    warped_inds[has_flow] += image_inds[has_flow]

    flow_ind_shift = torch.tensor((flow[:, 0, :, :] * H + flow[:, 1, :, :]),
                                             dtype=torch.int64).view(B, H*W)
    warped_inds[has_flow] += flow_ind_shift[has_flow]


    # batch offset
    offset = torch.cat([torch.ones(H*W, dtype=torch.int64) * H*W*i for i in range(B)]).view((B, H*W))
    warped_inds = torch.flatten(warped_inds + offset)
    image_inds = torch.flatten(image_inds + offset)

    warped_im = torch.zeros((B, C, H, W))
    for channel in (0, 1, 2):
        im_channel = torch.flatten(image[:, channel, :, :])
        warped_channel = torch.flatten(warped_im[:, channel, :, :])
        warped_channel[warped_inds] = im_channel[image_inds]
        warped_im[:, channel, :, :] = warped_channel.view((B, H, W))

    return warped_im





if __name__ == '__main__':
    flow = torch.zeros((5, 2, 10, 10))
    images = torch.zeros((5, 3, 10, 10))

    images[:, :, 0, 0] = 1
    flow[:, 0, 0, 0] = 4
    flow[:, 1, 0, 0] = 1

    images[:, :, 1, 1] = 3
    flow[:, 0, 1, 1] = 1
    flow[:, 1, 1, 1] = 1

    warped_images = warping(images, flow)
    print(warped_images[0, 0, :, :].detach().numpy())
    print(warped_images.max())