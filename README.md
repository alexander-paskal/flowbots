# Introduction

This repo is all about optical flow. We are reimplementing FlowNet and FlowNet2 from scratch in Pytorch, as well as a Transformer-based version of FlowNet. 

# Installation

All code was run in a conda environment. To replicate the exact environment, run the following commands in the Anaconda prompt:

    conda env create -f environment.yml

    conda activate flowbots

Alternatively, the minimum essential packages are included in a requirements.txt file. This can be installed using pip

    pip install -r requirements.txt


# Datasets

We use the following datasets:

- Flying Chairs
- Flying Things 3D

## Flying Chairs

The FlyingChairs dataset must be downloaded and organized in the following manner:

    root
        FlyingChairs
            data
                00001_flow.flo
                00001_img1.ppm
                00001_img2.ppm
                ...
            FlyingChairs_train_val.txt

The root file must be designated in the accompanying config.json file by setting the "flying_chairs" field

For more information see https://pytorch.org/vision/stable/generated/torchvision.datasets.FlyingChairs.html
