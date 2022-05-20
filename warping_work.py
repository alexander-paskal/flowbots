import cv2
import numpy as np
from imageio import imread


import torch
import torchvision as tv
import matplotlib.pyplot as plt
import numpy as np

import json

from warping import warping
import os
os.listdir()
#os.chdir("C:\\\\Users\\kevin\\OneDrive\\Documents\\ECE 285\\project_work\\datasets")
os.chdir("D:\\\\datasets")
d = {"flying_chairs": os.getcwd()}
with open("config.json", "w") as f:
    json.dump(d, f)
    
with open("config.json") as f:
    cfg = json.load(f)
print(cfg)
fc = tv.datasets.FlyingChairs(cfg["flying_chairs"])

a=fc[0][0]
b=fc[0][1]
flow=fc[0][2]

im1 = np.array(a.getdata())
im2 = np.array(b.getdata())

im1 = im1.reshape([a.size[1], a.size[0], 3])
im2 = im2.reshape([b.size[1], b.size[0], 3])


im1 = np.float64(im1)
im2 = np.float64(im2)



#flow = cv2.calcOpticalFlowFarneback(im1, im2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def warp_flow(img, flow):
    #h, w = flow.shape[:2]
    #flow = -flow
    #flow[:,:,0] += np.arange(w)
    #flow[:,:,1] += np.arange(h)[:,np.newaxis]
    #res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    res = warping(img, flow)
    return res

flow_im = flow.reshape(384,512,2)
print(flow_im.shape)


#hsv = draw_hsv(flow_im)
im1_test = np.transpose(im1, (2,0,1))
im2w = warp_flow(im1_test, flow)
cv2.imshow("image1", im2w/255)
#cv2.waitKey(0)
#cv2.imwrite("project_workflow.jpg",hsv/255)
#cv2.imwrite("im1.jpg", im1/255)
#cv2.imwrite("im2.jpg", im2/255)
#cv2.imwrite("im2w.jpg", im2w/255)