#!/bin/bash
# This script runs inference on multiple tasks using different models.

# ========================
# SCALE 0 ~ Hippocampus
# ========================
python -m inference.predict_our_models \
    --model Backbone \
    --imagests-dir datasets/Task04_Hippocampus_test1/fullres/upsampled/scale0/imagesTs \
    --labelsts-dir datasets/Task04_Hippocampus_test1/lowres/downsampled/scale0/labelsTs \
    --output-dir inference/predictions/hippocampus/fullres/ \
    --model_path /home/si-hj/Desktop/medsegnet/trained_models/unet3d/Task04_Hippocampus/2025-07-01_02:56:37_Hippo-Baseline-UNet \
    --device cuda:2 
    
python -m inference.predict_our_models \
    --model Backbone+Aug \
    --imagests-dir datasets/Task04_Hippocampus_test1/fullres/upsampled/scale0/imagesTs \
    --labelsts-dir datasets/Task04_Hippocampus_test1/lowres/downsampled/scale0/labelsTs \
    --output-dir inference/predictions/hippocampus/fullres/ \
    --model_path /home/si-hj/Desktop/medsegnet/trained_models/unet-aug3d/Task04_Hippocampus/2025-07-01_02:56:37_Hippo-Baseline-UNet-Aug \
    --device cuda:2 
    
python -m inference.predict_our_models \
    --model Multiscale \
    --imagests-dir datasets/Task04_Hippocampus_test1/fullres/upsampled/scale0/imagesTs \
    --labelsts-dir datasets/Task04_Hippocampus_test1/lowres/downsampled/scale0/labelsTs \
    --output-dir inference/predictions/hippocampus/fullres/ \
    --model_path /home/si-hj/Desktop/medsegnet/trained_models/ms-unet3d/Task04_Hippocampus/2025-07-01_02:56:37_Hippo-RAREUNet-Final \
    --device cuda:2 

# =======================
# SCALE 1 ~ Hippocampus
# =======================
python -m inference.predict_our_models \
    --model Backbone \
    --imagests-dir datasets/Task04_Hippocampus_test1/fullres/upsampled/scale1/imagesTs \
    --labelsts-dir datasets/Task04_Hippocampus_test1/lowres/downsampled/scale1/labelsTs \
    --output-dir inference/predictions/hippocampus/lowres/downsampled/scale1/ \
    --model_path /home/si-hj/Desktop/medsegnet/trained_models/unet3d/Task04_Hippocampus/2025-07-01_02:56:37_Hippo-Baseline-UNet \
    --device cuda:2 
    
python -m inference.predict_our_models \
    --model Backbone+Aug \
    --imagests-dir datasets/Task04_Hippocampus_test1/fullres/upsampled/scale1/imagesTs \
    --labelsts-dir datasets/Task04_Hippocampus_test1/lowres/downsampled/scale1/labelsTs \
    --output-dir inference/predictions/hippocampus/lowres/downsampled/scale1/ \
    --model_path /home/si-hj/Desktop/medsegnet/trained_models/unet-aug3d/Task04_Hippocampus/2025-07-01_02:56:37_Hippo-Baseline-UNet-Aug \
    --device cuda:2 
    
python -m inference.predict_our_models \
    --model Multiscale \
    --imagests-dir datasets/Task04_Hippocampus_test1/lowres/downsampled/scale1/imagesTs \
    --labelsts-dir datasets/Task04_Hippocampus_test1/lowres/downsampled/scale1/labelsTs \
    --output-dir inference/predictions/hippocampus/lowres/downsampled/scale1/ \
    --model_path /home/si-hj/Desktop/medsegnet/trained_models/ms-unet3d/Task04_Hippocampus/2025-07-01_02:56:37_Hippo-RAREUNet-Final \
    --device cuda:2 


# =======================
# SCALE 2 ~ Hippocampus
# =======================
python -m inference.predict_our_models \
    --model Backbone \
    --imagests-dir datasets/Task04_Hippocampus_test1/fullres/upsampled/scale2/imagesTs \
    --labelsts-dir datasets/Task04_Hippocampus_test1/lowres/downsampled/scale2/labelsTs \
    --output-dir inference/predictions/hippocampus/lowres/downsampled/scale2/ \
    --model_path /home/si-hj/Desktop/medsegnet/trained_models/unet3d/Task04_Hippocampus/2025-07-01_02:56:37_Hippo-Baseline-UNet \
    --device cuda:2 
    
python -m inference.predict_our_models \
    --model Backbone+Aug \
    --imagests-dir datasets/Task04_Hippocampus_test1/fullres/upsampled/scale2/imagesTs \
    --labelsts-dir datasets/Task04_Hippocampus_test1/lowres/downsampled/scale2/labelsTs \
    --output-dir inference/predictions/hippocampus/lowres/downsampled/scale2/ \
    --model_path /home/si-hj/Desktop/medsegnet/trained_models/unet-aug3d/Task04_Hippocampus/2025-07-01_02:56:37_Hippo-Baseline-UNet-Aug \
    --device cuda:2 
    
python -m inference.predict_our_models \
    --model Multiscale \
    --imagests-dir datasets/Task04_Hippocampus_test1/lowres/downsampled/scale2/imagesTs \
    --labelsts-dir datasets/Task04_Hippocampus_test1/lowres/downsampled/scale2/labelsTs \
    --output-dir inference/predictions/hippocampus/lowres/downsampled/scale2/ \
    --model_path /home/si-hj/Desktop/medsegnet/trained_models/ms-unet3d/Task04_Hippocampus/2025-07-01_02:56:37_Hippo-RAREUNet-Final \
    --device cuda:2 

# =======================
# SCALE 3 ~ Hippocampus
# =======================
python -m inference.predict_our_models \
    --model Backbone \
    --imagests-dir datasets/Task04_Hippocampus_test1/fullres/upsampled/scale3/imagesTs \
    --labelsts-dir datasets/Task04_Hippocampus_test1/lowres/downsampled/scale3/labelsTs \
    --output-dir inference/predictions/hippocampus/lowres/downsampled/scale3/ \
    --model_path /home/si-hj/Desktop/medsegnet/trained_models/unet3d/Task04_Hippocampus/2025-07-01_02:56:37_Hippo-Baseline-UNet \
    --device cuda:2 
    
python -m inference.predict_our_models \
    --model Backbone+Aug \
    --imagests-dir datasets/Task04_Hippocampus_test1/fullres/upsampled/scale3/imagesTs \
    --labelsts-dir datasets/Task04_Hippocampus_test1/lowres/downsampled/scale3/labelsTs \
    --output-dir inference/predictions/hippocampus/lowres/downsampled/scale3/ \
    --model_path /home/si-hj/Desktop/medsegnet/trained_models/unet-aug3d/Task04_Hippocampus/2025-07-01_02:56:37_Hippo-Baseline-UNet-Aug \
    --device cuda:2 
    
python -m inference.predict_our_models \
    --model Multiscale \
    --imagests-dir datasets/Task04_Hippocampus_test1/lowres/downsampled/scale3/imagesTs \
    --labelsts-dir datasets/Task04_Hippocampus_test1/lowres/downsampled/scale3/labelsTs \
    --output-dir inference/predictions/hippocampus/lowres/downsampled/scale3/ \
    --model_path /home/si-hj/Desktop/medsegnet/trained_models/ms-unet3d/Task04_Hippocampus/2025-07-01_02:56:37_Hippo-RAREUNet-Final \
    --device cuda:2 


# ========================
# SCALE 0 ~ Brain Tumour
# ========================
python -m inference.predict_our_models \
    --model Backbone \
    --imagests-dir datasets/Task01_BrainTumour_test1/fullres/upsampled/scale0/imagesTs \
    --labelsts-dir datasets/Task01_BrainTumour_test1/lowres/downsampled/scale0/labelsTs \
    --output-dir inference/predictions/braintumour/fullres/ \
    --model_path /home/si-hj/Desktop/medsegnet/trained_models/unet3d/Task01_BrainTumour/2025-07-01_02:56:40_Brats-Baseline-UNet \
    --device cuda:2 
    
python -m inference.predict_our_models \
    --model Backbone+Aug \
    --imagests-dir datasets/Task01_BrainTumour_test1/fullres/upsampled/scale0/imagesTs \
    --labelsts-dir datasets/Task01_BrainTumour_test1/lowres/downsampled/scale0/labelsTs \
    --output-dir inference/predictions/braintumour/fullres/ \
    --model_path /home/si-hj/Desktop/medsegnet/trained_models/unet-aug3d/Task01_BrainTumour/2025-07-01_02:56:40_Brats-Baseline-UNet-Aug \
    --device cuda:2 
    
python -m inference.predict_our_models \
    --model Multiscale \
    --imagests-dir datasets/Task01_BrainTumour_test1/fullres/upsampled/scale0/imagesTs \
    --labelsts-dir datasets/Task01_BrainTumour_test1/lowres/downsampled/scale0/labelsTs \
    --output-dir inference/predictions/braintumour/fullres/ \
    --model_path /home/si-hj/Desktop/medsegnet/trained_models/ms-unet3d/Task01_BrainTumour/2025-07-01_02:56:40_Brats-RAREUNet-Final \
    --device cuda:2 


# =======================
# SCALE 1 ~ Brain Tumour
# =======================
python -m inference.predict_our_models \
    --model Backbone \
    --imagests-dir datasets/Task01_BrainTumour_test1/fullres/upsampled/scale1/imagesTs \
    --labelsts-dir datasets/Task01_BrainTumour_test1/lowres/downsampled/scale1/labelsTs \
    --output-dir inference/predictions/braintumour/lowres/downsampled/scale1/ \
    --model_path /home/si-hj/Desktop/medsegnet/trained_models/unet3d/Task01_BrainTumour/2025-07-01_02:56:40_Brats-Baseline-UNet \
    --device cuda:2 
    
python -m inference.predict_our_models \
    --model Backbone+Aug \
    --imagests-dir datasets/Task01_BrainTumour_test1/fullres/upsampled/scale1/imagesTs \
    --labelsts-dir datasets/Task01_BrainTumour_test1/lowres/downsampled/scale1/labelsTs \
    --output-dir inference/predictions/braintumour/lowres/downsampled/scale1/ \
    --model_path /home/si-hj/Desktop/medsegnet/trained_models/unet-aug3d/Task01_BrainTumour/2025-07-01_02:56:40_Brats-Baseline-UNet-Aug \
    --device cuda:2 
    
python -m inference.predict_our_models \
    --model Multiscale \
    --imagests-dir datasets/Task01_BrainTumour_test1/lowres/downsampled/scale1/imagesTs \
    --labelsts-dir datasets/Task01_BrainTumour_test1/lowres/downsampled/scale1/labelsTs \
    --output-dir inference/predictions/braintumour/lowres/downsampled/scale1/ \
    --model_path /home/si-hj/Desktop/medsegnet/trained_models/ms-unet3d/Task01_BrainTumour/2025-07-01_02:56:40_Brats-RAREUNet-Final \
    --device cuda:2 


# =======================
# SCALE 2 ~ Brain Tumour
# =======================
python -m inference.predict_our_models \
    --model Backbone \
    --imagests-dir datasets/Task01_BrainTumour_test1/fullres/upsampled/scale2/imagesTs \
    --labelsts-dir datasets/Task01_BrainTumour_test1/lowres/downsampled/scale2/labelsTs \
    --output-dir inference/predictions/braintumour/lowres/downsampled/scale2/ \
    --model_path /home/si-hj/Desktop/medsegnet/trained_models/unet3d/Task01_BrainTumour/2025-07-01_02:56:40_Brats-Baseline-UNet \
    --device cuda:2 
    
python -m inference.predict_our_models \
    --model Backbone+Aug \
    --imagests-dir datasets/Task01_BrainTumour_test1/fullres/upsampled/scale2/imagesTs \
    --labelsts-dir datasets/Task01_BrainTumour_test1/lowres/downsampled/scale2/labelsTs \
    --output-dir inference/predictions/braintumour/lowres/downsampled/scale2/ \
    --model_path /home/si-hj/Desktop/medsegnet/trained_models/unet-aug3d/Task01_BrainTumour/2025-07-01_02:56:40_Brats-Baseline-UNet-Aug \
    --device cuda:2 
    
python -m inference.predict_our_models \
    --model Multiscale \
    --imagests-dir datasets/Task01_BrainTumour_test1/lowres/downsampled/scale2/imagesTs \
    --labelsts-dir datasets/Task01_BrainTumour_test1/lowres/downsampled/scale2/labelsTs \
    --output-dir inference/predictions/braintumour/lowres/downsampled/scale2/ \
    --model_path /home/si-hj/Desktop/medsegnet/trained_models/ms-unet3d/Task01_BrainTumour/2025-07-01_02:56:40_Brats-RAREUNet-Final \
    --device cuda:2 

# =======================
# SCALE 3 ~ Brain Tumour
# =======================
python -m inference.predict_our_models \
    --model Backbone \
    --imagests-dir datasets/Task01_BrainTumour_test1/fullres/upsampled/scale3/imagesTs \
    --labelsts-dir datasets/Task01_BrainTumour_test1/lowres/downsampled/scale3/labelsTs \
    --output-dir inference/predictions/braintumour/lowres/downsampled/scale3/ \
    --model_path /home/si-hj/Desktop/medsegnet/trained_models/unet3d/Task01_BrainTumour/2025-07-01_02:56:40_Brats-Baseline-UNet \
    --device cuda:2 
    
python -m inference.predict_our_models \
    --model Backbone+Aug \
    --imagests-dir datasets/Task01_BrainTumour_test1/fullres/upsampled/scale3/imagesTs \
    --labelsts-dir datasets/Task01_BrainTumour_test1/lowres/downsampled/scale3/labelsTs \
    --output-dir inference/predictions/braintumour/lowres/downsampled/scale3/ \
    --model_path /home/si-hj/Desktop/medsegnet/trained_models/unet-aug3d/Task01_BrainTumour/2025-07-01_02:56:40_Brats-Baseline-UNet-Aug \
    --device cuda:2 
    
python -m inference.predict_our_models \
    --model Multiscale \
    --imagests-dir datasets/Task01_BrainTumour_test1/lowres/downsampled/scale3/imagesTs \
    --labelsts-dir datasets/Task01_BrainTumour_test1/lowres/downsampled/scale3/labelsTs \
    --output-dir inference/predictions/braintumour/lowres/downsampled/scale3/ \
    --model_path /home/si-hj/Desktop/medsegnet/trained_models/ms-unet3d/Task01_BrainTumour/2025-07-01_02:56:40_Brats-RAREUNet-Final \
    --device cuda:2 

