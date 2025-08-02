
python scale_model_compare.py \
    --main-preds-dir ~/Desktop/inference/predictions \
    --nnunet-preds-dir ~/Desktop/nn-unet/predictions_simon \
    --dataset-dir ~/Desktop/datasets/Task04_Hippocampus_test1 \
    --dataset hippocampus \
    --output-dir ~/Desktop/inference/figures/scale_view_comparison \
    --case-ids hippocampus_173 hippocampus_234 hippocampus_065  \
    --font-path /home/si-hj/Desktop/inference/fonts/times.ttf \

python scale_model_compare.py \
    --main-preds-dir ~/Desktop/inference/predictions \
    --nnunet-preds-dir ~/Desktop/nn-unet/predictions_simon \
    --dataset-dir ~/Desktop/datasets/Task01_BrainTumour_test1 \
    --dataset braintumour \
    --output-dir ~/Desktop/inference/figures/scale_view_comparison \
    --case-ids BRATS_380 BRATS_326 BRATS_396 BRATS_389 BRATS_463 \
    --font-path /home/si-hj/Desktop/inference/fonts/times.ttf
