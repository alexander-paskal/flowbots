import numpy as np


def warping_them(img, flow):


    new_img = img.copy()
    *_, height, width = img.shape

    for i in range(height):
        for j in range(width):

            flowx, flowy = flow[:, i, j]

            #
            x = flowx + i
            y = flowy + j

            # if int(round(x)) > width:

            # bilinear coefficients
            theta_x = x - np.floor(x)
            theta_y = y - np.floor(y)
            theta_x_bar = 1 - theta_x
            theta_y_bar = 1 - theta_y

            new_img[int(round(x)), int(round(y))] = sum([
                theta_x_bar*theta_y_bar * img[int(np.floor(x)), int(np.floor(y))],
                theta_x * theta_y_bar * img[int(np.ceil(x)), int(np.floor(y))],
                theta_x_bar * theta_y * img[int(np.floor(x)), int(np.ceil(y))],
                theta_x * theta_y * img[int(np.ceil(x)), int(np.ceil(y))],
            ])


            return new_img


def warping_me(img, flow):


    new_img = np.zeros(img.shape)
    *_, height, width = img.shape

    for i in range(height):
        for j in range(width):
            flowy, flowx = flow[:, i, j]

            #
            x = round(flowx + i)
            y = round(flowy + j)

            if 0 <= x < height and 0 <= y < width:
                rgb = img[:, i, j]
                new_img[:, x, y] = rgb
                # print((i, j), rgb, (x, y))
            else:
                pass

    return new_img
