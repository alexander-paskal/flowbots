import numpy as np


def warping(img, flow):


    new_img = img.copy()
    height, width = img.shape



    for i in range(height):
        for j in range(width):

            flowx, flowy = flow[:, i, j]

            #
            x = flowx + i
            y = flowy + j

            # bilinear coefficients
            theta_x = x - np.floor(x)
            theta_y = y - np.floor(y)
            theta_x_bar = 1 - theta_x
            theta_y_bar = 1 - theta_y

            new_img[int(round(x)), int(round(y))] = sum([
                theta_x_bar*theta_y_bar * img[np.floor(x), np.floor(y)],
                theta_x * theta_y_bar * img[np.ceil(x), np.floor(y)],
                theta_x_bar * theta_y * img[np.floor(x), np.ceil(y)],
                theta_x * theta_y * img[np.ceil(x), np.ceil(y)],
            ])


            return new_img