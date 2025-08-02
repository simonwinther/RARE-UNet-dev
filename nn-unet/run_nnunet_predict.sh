#!/bin/bash

# 1 Setup
n_scales=3 #0, 1, ..., 3
echo "Running Setup..."



# 2 Hippocampus
run_hippocampus_processing() {
    echo "# 2 Hippocampus Processing"

    echo "## 2.0 Hippocampus Full Resolution Images"

    python ./run_nnunet_predict.py \
        -i ~/Desktop/datasets/Task04_Hippocampus_test1/fullres/upsampled/scale0/imagesTs \
        -o ./predictions_simon/hippocampus/fullres \
        -d Dataset004_Hippocampus \
        --scale 0

    echo "## 2.1 Hippocampus Interpolated Images"
    for ((i=1; i <= n_scales; i++)); do
        python ./run_nnunet_predict.py \
            -i ~/Desktop/datasets/Task04_Hippocampus_test1/fullres/upsampled/scale$i/imagesTs \
            -o ./predictions_simon/hippocampus/lowres/downsampled/scale$i \
            -d Dataset004_Hippocampus \
            --scale "$i"
    done
    echo "done ## 2.1"

    echo "## 2.2 Hippocampus Padded Images"
    for ((i=1; i <= n_scales; i++)); do
        python ./run_nnunet_predict.py \
            -i ~/Desktop/datasets/Task04_Hippocampus_test1/fullres/pad/scale$i/imagesTs \
            -o ./predictions_simon/hippocampus/lowres/crop/scale$i \
            -d Dataset004_Hippocampus \
            --scale "$i"
    done
    echo "done ## 2.2"
}

# 3 Brain Tumour 
run_braintumour_processing(){
	echo "# 3 Brain Tumour Processing"
	
	echo "## 3.0 Brain Tumour Full Resolution Images"
	python ~/Desktop/nn-unet/run_nnunet_predict.py \
			-i ~/Desktop/datasets/Task01_BrainTumour_test1/fullres/upsampled/scale0/imagesTs \
			-o ./predictions_simon/braintumour/fullres \
			-d Dataset001_BrainTumour \
			--scale 0

	echo "## 3.1 Brain Tumour Interpolated Images"
	for ((i=1; i <= n_scales; i++)); do  
		python ~/Desktop/nn-unet/run_nnunet_predict.py \
			-i ~/Desktop/datasets/Task01_BrainTumour_test1/fullres/upsampled/scale$i/imagesTs \
			-o ./predictions_simon/braintumour/lowres/downsampled/scale$i \
			-d Dataset001_BrainTumour \
			--scale "$i"
	done
    	echo "done ## 3.1"

    	echo "## 3.2 Brain Tumour Padded Images"
	for ((i=1; i <= n_scales; i++)); do  
		python ~/Desktop/nn-unet/run_nnunet_predict.py \
			-i ~/Desktop/datasets/Task01_BrainTumour_test1/fullres/pad/scale$i/imagesTs \
			-o ./predictions_simon/braintumour/lowres/crop/scale$i \
			-d Dataset001_BrainTumour \
			--scale "$i"
	done
	echo "done ## 3.2"
}

# 4. Run
## 4.1 Which dataset to run?

usage() {
    echo "Usage: $0 <section_name_or_number>"
    echo "Options:"
    echo "  2 or hippocampus  : Run Hippocampus processing (Section #2)"
    echo "  3 or braintumour  : Run Brain Tumour processing (Section #3)"
    exit 1
}

[ -z "$1" ] && usage

case "${1,,}" in  # `${1,,}` lowercases the input (bash >= 4)
    2 | hippocampus)
        run_hippocampus_processing
        ;;
    3 | braintumour)
        run_braintumour_processing
	;;
    *)
        echo "Error: Invalid choice '$1'."
        usage
        ;;
esac

echo "Selected processing finished."
