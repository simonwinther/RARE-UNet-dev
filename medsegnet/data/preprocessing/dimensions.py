import os
import numpy as np
import nibabel as nib
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import torch
import re
from tqdm import tqdm
import json
import argparse
import argcomplete


def resize_nd_image(img, target_shape, is_mask=False):
    """
    Resizes an n-dimensional image or mask by either cropping (centered) or padding (symmetrical)
    to match the target shape.
    
    Args:
        img (np.ndarray): Input image array (any number of dimensions)
        target_shape (tuple): Target shape to resize to (must match number of dimensions)
        is_mask (bool): Whether the input is a mask (affects padding strategy)
        
    Returns:
        np.ndarray: Resized image/mask with exactly the target shape
    """
    if len(target_shape) != np.array(img).ndim:
        raise ValueError(f"Target shape {target_shape} must have same dimensions as input image ({img.ndim})")
   

    # Initialize lists to store crop/pad parameters
    slices = []         # Crop slices for each dimension
    pad_widths = []     # Padding amounts for each dimension

    # Calculate crop/pad for each dimension
    for dim, (current_size, target_size) in enumerate(zip(img.shape, target_shape)):
        # Calculate difference between current and target size
        size_diff = target_size - current_size

        # Handle cropping (current size > target size)
        if size_diff < 0:
            crop_start = (current_size - target_size) // 2
            crop_end = crop_start + target_size
            slices.append(slice(crop_start, crop_end))
            pad_widths.append((0, 0))  # No padding needed
            
        # Handle padding (current size < target size)
        elif size_diff > 0:
            slices.append(slice(None))  # Take entire dimension
            pad_before = size_diff // 2
            pad_after = size_diff - pad_before
            pad_widths.append((pad_before, pad_after))
            
        # No action needed
        else:
            slices.append(slice(None))
            pad_widths.append((0, 0))

    # Apply cropping first
    cropped = img[tuple(slices)]

    # Then apply padding with appropriate strategy
    if is_mask:
        # For masks, pad with 0s (background)
        padded = np.pad(cropped, pad_widths, mode='constant', constant_values=0)
    else:
        # For images, pad with edge values (avoids black borders)
        # padded = np.pad(cropped, pad_widths, mode='edge')
        padded = np.pad(cropped, pad_widths, mode='constant', constant_values=0)
    return padded

def save_precomputed_dimensions(dim_dict, filename="./data/precomputed_dimensions.json"):
    existing_data = load_precomputed_dimensions(filename)
    existing_data.update(dim_dict)
    with open(filename, "w") as f:
        json.dump(existing_data, f, indent=4)

def load_precomputed_dimensions(filename="./data/precomputed_dimensions.json"):
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {} 

def nearest_power_of_two(value):
    return int(2 ** np.round(np.log2(value))) if value > 0 else 0

def get_save_dir(task):
    return os.path.join("preprocessing", "images", task if task else "other")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Precompute dimensions for dataset images."
    )
    parser.add_argument("-dp", "--dataset", type=str, help="Path to the dataset directory", required=True)
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    
    precompute_dimensions(args.dataset)