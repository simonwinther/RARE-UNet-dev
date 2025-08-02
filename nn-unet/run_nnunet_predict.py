#!/usr/bin/env python3
from datetime import date
import textwrap
from nnunetv2.dataset_conversion.convert_MSD_dataset import split_4d_nifti
import argparse
import shutil
import subprocess
import tempfile
import os
import sys
import re
from pathlib import Path
import nibabel as nib
import torch
import torch.nn.functional as F
import torchio as tio
import numpy as np
from pathlib import Path
today = date.today().strftime("%Y%m%d")

target_shapes = {
    "Dataset001_BrainTumour": (256, 256, 128),
    "Dataset004_Hippocampus": (32, 64, 32), 
}

def get_encodings(str: str) -> str:
    """
    Extract all encodings from a string.
    """
    if "pad" in str and "upsampled" in str:
        raise ValueError(
            f"Both 'pad' and 'upsample' found in string: {str}."
            "Please specify only one encoding."
        )
    if "pad" in str:
        return "pad"
    elif "upsampled" in str:
        return "upsampled"
    else:
        raise ValueError(f"Unknown encoding in string: {str}")
    

def parse_args():
    p = argparse.ArgumentParser(
        description="Prepare inputs for nnUNetv2 and run prediction"
    )
    p.add_argument(
        "--project-base",
        "-b",
        default="/home/si-hj/Desktop/nn-unet",
        help="NNUNET_PROJECT_BASE",
    )
    p.add_argument(
        "--input-folder",
        "-i",
        required=True,
        help="Original folder containing .nii.gz files (no _0000 suffix)",
    )
    p.add_argument(
        "--output-folder",
        "-o",
        required=True,
        help="Folder where predictions will be written",
    )
    p.add_argument(
        "--scale",
        type=int,
        required=True,
        help="Scale factor to downsample image",
    )
    p.add_argument(
        "--dataset",
        "-d",
        required=True,
        help="Dataset name in nnUNet_raw (e.g. Dataset004_Hippocampus)",
    )
    p.add_argument(
        "--trainer", "-t", default="nnUNetTrainerNoDADS", help="nnUNet Trainer class"
    )
    p.add_argument(
        "--plans", "-p", default="nnUNetPlans", help="nnUNet Plans identifier"
    )
    p.add_argument(
        "--config",
        "-c",
        default="3d_fullres",
        help="Configuration (e.g. 3d_fullres, 2d)",
    )
    p.add_argument("--fold", "-f", type=int, default=0, help="Fold number")
    p.add_argument(
        "--ckpt", "-k", default="checkpoint_best.pth", help="Checkpoint filename"
    )
    p.add_argument(
        "--use-symlink",
        "-s",
        action="store_true",
        help="Symlink instead of copying (faster, but leaves originals untouched)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    orig_dir = Path(args.input_folder)
    if not orig_dir.is_dir():
        print(f"Error: input folder not found: {orig_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    # find all .nii.gz files
    all_imgs = sorted(orig_dir.glob("*.nii.gz"))
    if not all_imgs:
        print("Error: no .nii.gz files found in input folder.", file=sys.stderr)
        sys.exit(1)

    with tempfile.TemporaryDirectory(
        prefix="temp_nnunet_input_", dir=args.project_base
    ) as tmp:
        tmp_path = Path(tmp)
        # print(f"Temporary input directory: {tmp_path}\n")

        for img in sorted(Path(args.input_folder).glob("*.nii.gz")):
            split_4d_nifti(str(img), str(tmp_path))

            base = img.name[:-7]
            for out in sorted(tmp_path.glob(f"{base}_*.nii.gz")):
                # print(f"Copied: {img.name} → {out.name}")
                break

        # build command
        cmd = [
            "nnUNetv2_predict",
            "-i",
            str(tmp_path),
            "-o",
            str(out_dir),
            "-d",
            args.dataset,
            "-tr",
            args.trainer,
            "-p",
            args.plans,
            "-c",
            args.config,
            "-f",
            str(args.fold),
            "-chk",
            args.ckpt,
        ]

        try:
            subprocess.run(cmd, check=True)
            # print("Prediction finished successfully.")
            # Loop through the output directory and downsample or crop files
            scale = args.scale
            assert scale >= 0, "Scale must be in [0, inf]"
            
            encoding = get_encodings(args.input_folder)
                
                        
            shape = target_shapes.get(args.dataset)
            if shape is None:
                print(f"Unknown dataset {args.dataset}, please set its target shape.", file=sys.stderr)
                sys.exit(1)

            factor = 1 / 2 ** scale
            target_shape = tuple(int(s * factor) for s in shape)  # (W, H, D)

            for img_path in Path(out_dir).glob("*.nii.gz"):
                data = nib.load(str(img_path)).get_fdata()  # (W, H, D)
                t_img = torch.tensor(data).unsqueeze(0)  # (1, W, H, D)

                if encoding == "pad":
                    transform = tio.CropOrPad(target_shape=target_shape, padding_mode="constant")
                    transformed = transform(t_img)
                    final_data = transformed.numpy()  # still (W, H, D)
                elif encoding == "upsampled":
                    tensor = t_img.unsqueeze(0).float()  # (1, 1, W, H, D)
                    downsampled = F.interpolate(tensor, size=target_shape, mode="nearest")  # still (W, H, D)
                    final_data = downsampled.squeeze().numpy()

                else:
                    raise ValueError("Unknown encoding type.")
                print(f"Processed {img_path.name} with shape {data.shape} → {final_data.shape}")
                nib.save(nib.Nifti1Image(final_data.astype(np.uint8), np.eye(4)), str(img_path))

        except subprocess.CalledProcessError as e:
            print(
                f"nnUNetv2_predict failed with exit code {e.returncode}",
                file=sys.stderr,
            )
            sys.exit(e.returncode)


if __name__ == "__main__":
    main()




