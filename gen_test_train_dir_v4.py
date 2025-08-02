import argparse
from cProfile import label
import os
import shutil
import random
from tqdm import tqdm
from medsegnet.utils.utils import setup_seed
import torchio as tio
import torch.nn.functional as F
import torch
import nibabel as nib
import numpy as np


ToCanonical = tio.ToCanonical()


def load_nifti(input_path, is_label=False):
    """
    Loads a NIfTI file using TorchIO and converts it to canonical (RAS+) orientation.
    (C, W, H, D)
    """
    if is_label:
        tio_img = tio.LabelMap(input_path)
    else:
        tio_img = tio.ScalarImage(input_path)
    return ToCanonical(tio_img)


# def save_nifti(tio_img, output_path):
#     """
#     Saves a TorchIO image object to a NIfTI file.
#     """
#     if tio_img.data.dim() == 4:
#         tensor = tio_img.data.permute(1, 2, 3, 0)
#     elif tio_img.data.dim() == 3:
#         tensor = tio_img.data.permute(1, 2, 0)
#     else:
#         raise ValueError(f"Unsupported tensor shape: {tio_img.data.shape}")

#     nib.save(
#         nib.Nifti1Image(tensor.numpy(), tio_img.affine),
#         output_path
#     )


def save_nifti(tio_img, output_path):
    """
    Saves a TorchIO image object to a NIfTI file.
    """
    # tio_img.save(path=output_path, squeeze=True)

    img = tio_img.data.numpy()
    img = np.moveaxis(img, 0, -1)

    if isinstance(tio_img, tio.LabelMap):
        out_dtype = np.int16
        img = img.squeeze(-1)  # Remove singleton dimension if present
    else:
        out_dtype = np.float32

    img = img.astype(out_dtype)

    nib_image = nib.Nifti1Image(img, tio_img.affine)


    nib.save(nib_image, output_path)


def down_or_up_sample_nifti(image, factor, interpolation):
    """
    Resample a TorchIO image by the given factor.

    Args:
        image (tio.ScalarImage or tio.LabelMap): input TorchIO image.
        factor (float): scaling factor for spacing (new_spacing = old_spacing * factor).
        interpolation (str): interpolation mode for image resampling, e.g. 'linear' or 'nearest'.

    Returns:
        tio.ScalarImage or tio.LabelMap: resampled image.
    """
    target_shape = tuple(round(shape * (factor)) for shape in image.shape[1:])
    x = image.data.unsqueeze(0).float()
    if interpolation == "nearest":
        x_ms = F.interpolate(x, size=target_shape, mode=interpolation)
    else:
        x_ms = F.interpolate(
            x, size=target_shape, mode=interpolation, align_corners=False
        )
    x_ms = x_ms.squeeze(0)

    image_affine = image.affine.copy()
    image_affine[:3, :3] *= 1 / factor

    if isinstance(image, tio.LabelMap):
        x_ms = torch.round(x_ms).long()
        return tio.LabelMap(tensor=x_ms, affine=image_affine)
    else:
        return tio.ScalarImage(tensor=x_ms, affine=image_affine)


def process_split(
    train_set,
    test_set,
    in_images_dir,
    in_labels_dir,
    out_base_dir,
    dataset_json_src,
    transform_img_fn,
    transform_lab_fn=lambda x: x,
):
    # Create output directories
    dirs = {
        "train_img": os.path.join(out_base_dir, "imagesTr"),
        "train_lab": os.path.join(out_base_dir, "labelsTr"),
        "test_img": os.path.join(out_base_dir, "imagesTs"),
        "test_lab": os.path.join(out_base_dir, "labelsTs"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # Copy dataset.json
    shutil.copy(dataset_json_src, os.path.join(out_base_dir, "dataset.json"))

    # Helper to process a set
    def _run(items, desc, out_img_dir, out_lab_dir):
        for img_file, lab_file in tqdm(items, desc=desc, unit="pair", leave=False):
            img = load_nifti(os.path.join(in_images_dir, img_file))
            lab = load_nifti(os.path.join(in_labels_dir, lab_file), is_label=True)

            img = transform_img_fn(img)
            lab = transform_lab_fn(lab)

            save_nifti(img, os.path.join(out_img_dir, img_file))
            save_nifti(lab, os.path.join(out_lab_dir, lab_file))

    _run(train_set, f"Train Set @ {out_base_dir}", dirs["train_img"], dirs["train_lab"])
    _run(test_set, f"Test Set @ {out_base_dir}", dirs["test_img"], dirs["test_lab"])


def split_and_organize_data(input_dir, output_dir, split_ratio, scales, target_shape):
    # Prepare input paths and split
    in_img_dir = os.path.join(input_dir, "imagesTr")
    in_lab_dir = os.path.join(input_dir, "labelsTr")
    dataset_json = os.path.join(input_dir, "dataset.json")

    files = sorted(os.listdir(in_img_dir))
    labels = sorted(os.listdir(in_lab_dir))
    combined = list(zip(files, labels))
    random.shuffle(combined)
    idx = int(len(combined) * split_ratio)
    train_set, test_set = combined[:idx], combined[idx:]

    # Raw full resolution
    raw_dir = os.path.join(output_dir, "fullres", "raw")
    process_split(
        train_set,
        test_set,
        in_img_dir,
        in_lab_dir,
        raw_dir,
        dataset_json,
        transform_img_fn=lambda x: x,
        transform_lab_fn=lambda x: x,
    )
    cp = tio.CropOrPad(target_shape, padding_mode="constant")

    # Padding and scaling
    for mode in ["pad", "upsampled"]:
        mode_dir = os.path.join(output_dir, "fullres", mode)
        for scale in scales:
            base = os.path.join(mode_dir, f"scale{scale}")

            if mode == "pad":

                def img_fn(img, scale=scale):
                    img = cp(img)
                    img = down_or_up_sample_nifti(img, 1 / (2**scale), "trilinear")
                    return cp(img)

            else:  # upsampled

                def img_fn(img, scale=scale):
                    img = cp(img)
                    img = down_or_up_sample_nifti(img, 1 / (2**scale), "trilinear")
                    return down_or_up_sample_nifti(img, 2**scale, "trilinear")

            lab_fn = lambda lab, scale=None: tio.CropOrPad(
                target_shape, padding_mode="constant"
            )(lab)

            process_split(
                train_set,
                test_set,
                in_img_dir,
                in_lab_dir,
                base,
                dataset_json,
                transform_img_fn=img_fn,
                transform_lab_fn=lab_fn,
            )

    # Low resolution downsampling
    lowres_dir = os.path.join(output_dir, "lowres", "downsampled")
    for scale in scales:
        base = os.path.join(lowres_dir, f"scale{scale}")

        def img_fn(img, scale=scale):
            img = cp(img)
            return down_or_up_sample_nifti(img, 1 / (2**scale), "trilinear")

        def lab_fn(lab, scale=scale):
            lab = cp(lab)
            return down_or_up_sample_nifti(lab, 1 / (2**scale), "nearest")

        process_split(
            train_set,
            test_set,
            in_img_dir,
            in_lab_dir,
            base,
            dataset_json,
            transform_img_fn=img_fn,
            transform_lab_fn=lab_fn,
        )


# run the script
# python gen_test_train_dir_v4.py --input_dir=datasets/Task04_Hippocampus --target_shape 32 64 32
# nohup python gen_test_train_dir_v4.py --input_dir=datasets/Task04_Hippocampus --target_shape 32 64 32 > gen_HP.log 2>&1 &

# python gen_test_train_dir_v4.py --input_dir=datasets/Task01_BrainTumour --target_shape 256 256 128
# nohup python gen_test_train_dir_v4.py --input_dir=datasets/Task01_BrainTumour --target_shape 256 256 128 > gen_BP.log 2>&1 &

# python gen_test_train_dir_v4.py --input_dir=datasets/Task05_Prostate --target_shape 256 256 16
# nohup python gen_test_train_dir_v4.py --input_dir=datasets/Task05_Prostate --target_shape 256 256 16 > gen_BP.log 2>&1 &
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split and organize dataset into training and testing sets with scales."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the input directory containing imagesTr and labelsTr.",
    )
    parser.add_argument(
        "--target_shape",
        type=int,
        nargs=3,
        required=True,
        help="Target shape for downsampling.",
    )
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.8,
        help="Ratio of training to testing split (default: 0.8).",
    )
    parser.add_argument(
        "--scales",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
        help="List of scales to generate (default: [0, 1, 2, 3]).",
    )

    args = parser.parse_args()

    input_dir = args.input_dir.rstrip(os.sep)
    parent, name = os.path.split(input_dir)
    nameIdx = name.find("_OG")
    if nameIdx != -1:
        name = name[:nameIdx]
    name += "_test1"
    output_dir = os.path.join(parent, name)

    setup_seed(42)
    split_and_organize_data(
        input_dir=args.input_dir,
        output_dir=f"{output_dir}",
        split_ratio=args.split_ratio,
        scales=args.scales,
        target_shape=args.target_shape,
    )
