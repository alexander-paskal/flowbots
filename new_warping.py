import numpy as np
import torch



def warping(image, flow):

    B, C, H, W = image.size()


    image_inds = torch.arange(H * W, dtype=torch.uint8)
    image_inds = image_inds.repeat(B, 1)

    has_flow = torch.sum(torch.abs(flow), dim=1, dtype=torch.bool).view((B, H*W))
    warped_inds = torch.zeros((B, H*W), dtype=torch.uint8)

    warped_inds[has_flow] += image_inds[has_flow]

    flow_ind_shift = torch.tensor((flow[:, 0, :, :] * H + flow[:, 1, :, :]),
                                             dtype=torch.uint8).view(B, H*W)
    warped_inds[has_flow] += flow_ind_shift[has_flow]

    image = image.view((B, C, H * W))
    warped_image = torch.zeros((B, C, H*W))




im1 = np.zeros((10,10))
H, W = im1.shape

# im1_inds = np.arange(0, H*W)
#
# im1[5, 0] = 10
# im1[2, 0] = 6
# im1[6,0] = 5
#
# flow = np.zeros((2,10,10))
# flow[:, 5, 0] = [3, 4]
# flow[:, 2, 0] = [1,1]
# flow[:, 6, 0] = [-3, 0]
#
#
# has_flow = np.sum(np.abs(flow), axis=0).astype(bool).flatten()
# im2_inds = np.zeros((H*W))
# im2_inds[has_flow] += im1_inds[has_flow]
# flow_ind_shift = (flow[0, :, :].flatten() * H + flow[1, :, :].flatten()).astype(int)
#
# im2_inds[has_flow] += flow_ind_shift[has_flow]
# im2_inds = im2_inds.astype(int)


im2 = np.zeros((10, 10))
im2_flat = im2.flatten()
im2_flat[im2_inds] = im1.flatten()[im1_inds]
im2 = im2_flat.reshape((10,10))

print(im1)
print()

print(im2.reshape((10,10)))
print(im1_inds)
print(im2_inds)