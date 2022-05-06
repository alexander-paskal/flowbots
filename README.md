# Introduction
This repo is all about optical flow. We are reimplementing FlowNet and FlowNet2 from scratch in Pytorch, as well as a Transformer-based version of FlowNet.
# Installation
All code was run in a conda environment. To replicate the exact environment, run the following commands in the Anaconda prompt:

    conda env create -f environment.yml

    conda activate flowbots

For more information on installing conda environments, see https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file. Alternatively, the minimum essential packages are included in a requirements.txt file. This can be installed using pip

    pip install -r requirements.txt

# Datasets

We use the following optical flow datasets:

- Flying Chairs
- Flying Things 3D
- Sintel

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


## Flying Things 3D

The FlyingThings3D dataset must be downloaded and organized in the following format:

    root
        FlyingThings3D
            frames_cleanpass
                TEST
                TRAIN
            frames_finalpass
                TEST
                TRAIN
            optical_flow
                TEST
                TRAIN

The torrent files can be obtained from https://academictorrents.com/userdetails.php?id=9551. We recommend only downloading the necessary files, as the downloads are quite large. The root file The root file must be designated in the accompanying config.json file by setting the "flying_things_3d" field. 


## Sintel

The Sintel dataset must be downloaded and organized in the following format:

    root
        Sintel
            testing
                clean
                    scene_1
                    scene_2
                    ...
                final
                    scene_1
                    scene_2
                    ...
            training
                clean
                    scene_1
                    scene_2
                    ...
                final
                    scene_1
                    scene_2
                    ...
                flow
                    scene_1
                    scene_2
                    ...


The root file must be designated in the accompanying config.json file by setting the "sintel" field

For more information see https://pytorch.org/vision/stable/generated/torchvision.datasets.Sintel.html