#!/bin/bash

# try this for fix s
source $(conda info --base)/etc/profile.d/conda.sh


# Activate your nnU-Net environment first
conda activate nn-unet


# Set CUDA_VISIBLE_DEVICES to specify GPU ID 2
export CUDA_VISIBLE_DEVICES=3

# -num_gpus NUM_GPUS

# nnUNetv2_train 4 3d_fullres 0 -tr nnUNetTrainer 
nnUNetv2_train 005 3d_fullres 0 -tr nnUNetTrainerNoDADS 



# prerequisites for hippocampus dataset(Task04):
# 1. nnUNetv2_convert_MSD_dataset -i dataset/MSD/Task05_Prostate/
# 2. nnUNetv2_plan_and_preprocess -d 5 --verify_dataset_integrity