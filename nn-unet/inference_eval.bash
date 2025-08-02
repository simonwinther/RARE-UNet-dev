# Assuming you are in ~/Desktop/nn-unet/
# And your (nn-unet) conda environment is active

NNUNET_PROJECT_BASE="$HOME/Desktop/nn-unet/predictions/hp/pad/scale3"
OUR_DATASET_BASE="$HOME/Desktop/datasets"


# ---------------------------- CHANGES THIS PART Start ----------------------------

GT_FOLDER="$OUR_DATASET_BASE/Task04_Hippocampus_test1/lowres/downsampled/scale3/labelsTs"
PRED_FOLDER="$NNUNET_PROJECT_BASE"

# ----------------------------- CHANGES THIS PART END -----------------------------

OUTPUT_JSON_FILE="$PRED_FOLDER/evaluation_summary_simple.json"


nnUNetv2_evaluate_simple \
    "$GT_FOLDER" \
    "$PRED_FOLDER" \
    -l 1 2 \
    -o "$OUTPUT_JSON_FILE"