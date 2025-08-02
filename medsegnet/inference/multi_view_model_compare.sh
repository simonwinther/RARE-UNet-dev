#!/bin/bash
# =========================
# SCALE 0 ~ Brain Tumour
# =========================
echo "Plotting segmentation results for Brain Tumour at scale 0..."
python multi_view_model_compare.py \
    --images ~/Desktop/datasets/Task01_BrainTumour_test1/lowres/downsampled/scale0/imagesTs/BRATS_380.nii.gz \
            ~/Desktop/datasets/Task01_BrainTumour_test1/lowres/downsampled/scale0/imagesTs/BRATS_326.nii.gz \
            ~/Desktop/datasets/Task01_BrainTumour_test1/lowres/downsampled/scale0/imagesTs/BRATS_396.nii.gz \
            ~/Desktop/datasets/Task01_BrainTumour_test1/lowres/downsampled/scale0/imagesTs/BRATS_389.nii.gz \
            ~/Desktop/datasets/Task01_BrainTumour_test1/lowres/downsampled/scale0/imagesTs/BRATS_463.nii.gz \
    --ours-seg ~/Desktop/inference/predictions/braintumour/fullres/Multiscale \
    --bb-seg ~/Desktop/inference/predictions/braintumour/fullres/Backbone \
    --bb-aug-seg ~/Desktop/inference/predictions/braintumour/fullres/Backbone+Aug \
    --nnunet-seg ~/Desktop/nn-unet/predictions_simon/braintumour/fullres/ \
    --gt-seg ~/Desktop/datasets/Task01_BrainTumour_test1/lowres/downsampled/scale0/labelsTs/ \
    --out-dir ~/Desktop/inference/figures/multi_view_model_comparison/fullres \
    --name LOWEST AVERAGE HIGHEST \
    --font-path /home/si-hj/Desktop/inference/fonts/times.ttf \
    --flips 1 1 1 1 1

# # ========================
# # SCALE 1 ~ Brain Tumour
# # Mean Dice: 0.7165
# # ========================
# echo "Plotting segmentation results for Brain Tumour at scale 1..."
# python -m inference.plot_segmentation_results \
#     --images ~/Desktop/datasets/Task01_BrainTumour_test1/lowres/downsampled/scale1/imagesTs/BRATS_380.nii.gz \
#             ~/Desktop/datasets/Task01_BrainTumour_test1/lowres/downsampled/scale1/imagesTs/BRATS_326.nii.gz \
#             ~/Desktop/datasets/Task01_BrainTumour_test1/lowres/downsampled/scale1/imagesTs/BRATS_396.nii.gz \
#     --ours-seg ~/Desktop/inference/predictions/braintumour/lowres/downsampled/scale1/Multiscale \
#     --bb-seg ~/Desktop/inference/predictions/braintumour/lowres/downsampled/scale1/Backbone \
#     --bb-aug-seg ~/Desktop/inference/predictions/braintumour/lowres/downsampled/scale1/Backbone+Aug \
#     --nnunet-seg nn-unet/predictions_simon/braintumour/lowres/downsampled/scale1 \
#     --gt-seg datasets/Task01_BrainTumour_test1/lowres/downsampled/scale1/labelsTs/ \
#     --out-dir inference/figures/braintumourtesting69/scale1 \
#     --name LOWEST AVERAGE HIGHEST \
#     --flips 1 1 0


# # ========================
# # SCALE 2 ~ Brain Tumour
# # Mean Dice: 0.7165
# # ========================
# echo "Plotting segmentation results for Brain Tumour at scale 2..."
# python -m inference.plot_segmentation_results \
#     --images ~/Desktop/datasets/Task01_BrainTumour_test1/lowres/downsampled/scale2/imagesTs/BRATS_380.nii.gz \
#             ~/Desktop/datasets/Task01_BrainTumour_test1/lowres/downsampled/scale2/imagesTs/BRATS_326.nii.gz \
#             ~/Desktop/datasets/Task01_BrainTumour_test1/lowres/downsampled/scale2/imagesTs/BRATS_396.nii.gz \
#     --ours-seg ~/Desktop/inference/predictions/braintumour/lowres/downsampled/scale2/Multiscale \
#     --bb-seg ~/Desktop/inference/predictions/braintumour/lowres/downsampled/scale2/Backbone \
#     --bb-aug-seg ~/Desktop/inference/predictions/braintumour/lowres/downsampled/scale2/Backbone+Aug \
#     --nnunet-seg nn-unet/predictions_simon/braintumour/lowres/downsampled/scale2 \
#     --gt-seg datasets/Task01_BrainTumour_test1/lowres/downsampled/scale2/labelsTs/ \
#     --out-dir inference/figures/braintumourtesting69/scale2 \
#     --name LOWEST AVERAGE HIGHEST \
#     --flips 1 1 0



# =========================
# SCALE 0 ~ Hippocampus OLD
# =========================
echo "Plotting segmentation results for Hippocampus at scale 0..."
python multi_view_model_compare.py \
    --images ~/Desktop/datasets/Task04_Hippocampus_test1/lowres/downsampled/scale0/imagesTs/hippocampus_173.nii.gz \
            ~/Desktop/datasets/Task04_Hippocampus_test1/lowres/downsampled/scale0/imagesTs/hippocampus_234.nii.gz \
            ~/Desktop/datasets/Task04_Hippocampus_test1/lowres/downsampled/scale0/imagesTs/hippocampus_065.nii.gz \
    --ours-seg ~/Desktop/inference/predictions/hippocampus/fullres/Multiscale \
    --bb-seg ~/Desktop/inference/predictions/hippocampus/fullres/Backbone \
    --bb-aug-seg ~/Desktop/inference/predictions/hippocampus/fullres/Backbone+Aug \
    --nnunet-seg ~/Desktop/nn-unet/predictions_simon/hippocampus/fullres/ \
    --gt-seg ~/Desktop/datasets/Task04_Hippocampus_test1/lowres/downsampled/scale0/labelsTs/ \
    --out-dir ~/Desktop/inference/figures/multi_view_model_comparison/fullres \
    --name LOWEST AVERAGE HIGHEST \
    --font-path /home/si-hj/Desktop/inference/fonts/times.ttf \
    --flips 1 1 1
    