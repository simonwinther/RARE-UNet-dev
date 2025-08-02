import json
import os
import argparse
from textwrap import dedent
import time
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from glob import glob
from pathlib import Path
import numpy as np
from medsegnet.data.datasets import MedicalDecathlonDataset, ModalitiesDataset
from medsegnet.utils.metrics import dice_coefficient
from omegaconf import OmegaConf
from tqdm import tqdm
import torchio as tio
from torch.utils.data import DataLoader
import re

# --------------------------
# Model Definitions (Imports)
# --------------------------
from medsegnet.models.unet import UNet3D
from medsegnet.models.unet_aug import UNetAugmented
from medsegnet.models.rare_unet import MSUNet3D

# -----------------------
# Utility Functions
# -----------------------
def check_prediction_shape_validity(pred, gt):
    """
    Return True if any dimension of pred is smaller than gt.
    This will hopefully never occur, but just a nice sanity check.
    """
    return any(p < g for p, g in zip(pred.shape, gt.shape))


# -----------------------
# Factory Methods
# -----------------------
def get_encoding(path: str) -> str:
    p = path.lower()
    if "pad" in p and "upsampled" in p:
        raise ValueError(
            "Path contains both 'pad' and 'upsampled'. "
            "Please ensure the path is correctly specified."
        )
    if "pad" in p:
        return "pad"
    if "upsampled" in p:
        return "upsampled"
    if "downsampled" in p:
        return "downsampled"
    return "unknown"

def create_trained_model(
    name: str, best_model_path: str, device, cfg, **kwargs
) -> nn.Module:
    kwargs["mode"] = "inference"
    model_dict = {
        "Backbone": UNet3D,
        "Backbone+Aug": UNetAugmented,
        "Multiscale": MSUNet3D,
    }
    model = model_dict.get(name, None)
    if model is None: 
        valid = ", ".join(model_dict.keys())
        raise ValueError(f"Model {name!r} isn’t supported (choose from: {valid})")
    model = model(cfg, **kwargs)
    best_model = torch.load(best_model_path, map_location=device)
    model.load_state_dict(best_model["model_state_dict"])
    return model


def create_test_dataset(
    cfg, image_files, mask_files, images_dir, labels_dir
) -> MedicalDecathlonDataset:
    _name = cfg.dataset.name
    cls = (
        ModalitiesDataset if _name == "Task01_BrainTumour" else MedicalDecathlonDataset
    )
    return cls(
        cfg,
        "test",
        image_files=image_files,
        mask_files=mask_files,
        images_path=images_dir,
        masks_path=labels_dir,
    )
    
# -----------------------
# Main Execution
# -----------------------
def main():
    parser = argparse.ArgumentParser(
        description="Segmentation inference script",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["Backbone", "Backbone+Aug", "Multiscale"],
    )
    parser.add_argument(
        "--imagests-dir",
        type=str,
        required=True,
        help="Directory containing image files",
    )
    parser.add_argument(
        "--labelsts-dir",
        type=str,
        required=True,
        help="Directory containing label files",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Output directory for predictions"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path containing config.yaml and model.pth",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Specify the device to use (e.g., 'cuda', 'cuda:0', 'cpu')",
    )   
    parser.add_argument(
        "--ignore_index",
        type=int,
        default=0,
        help="Index to ignore in the labels (default: 0, e.g., background class)",
    )
    
    args = parser.parse_args()
    encoding = get_encoding(args.imagests_dir)

    inference_times = []
    dice_scores = []
    
    # Load config
    config = os.path.join(args.model_path, "config.yaml")
    if not os.path.exists(config):
        raise FileNotFoundError(
            f"Config file not found at {config}. Please ensure it exists."
        )

    cfg = OmegaConf.load(config)
    
    # Set device
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    best_model_path = os.path.join(args.model_path, "best_model.pth")
    model = create_trained_model(
        args.model, best_model_path=best_model_path, device=device, cfg=cfg
    )
    model.to(device)
    model.eval()

    image_filenames = [
        os.path.basename(p)
        for p in sorted(glob(os.path.join(args.imagests_dir, "*.nii*")))
    ]
    label_filenames = [
        os.path.basename(p)
        for p in sorted(glob(os.path.join(args.labelsts_dir, "*.nii*")))
    ]
    dataset = create_test_dataset(
        cfg,
        image_files=image_filenames,
        mask_files=label_filenames,
        images_dir=str(args.imagests_dir),
        labels_dir=str(args.labelsts_dir),
    )
    batch_size = 1
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print(f"\U0001f50d Found {len(image_filenames)} image(s) in {args.imagests_dir}")

    output_path = Path(args.output_dir) / args.model
    output_path.mkdir(parents=True, exist_ok=True)

    for i, (image, label) in enumerate(tqdm(dataloader, desc="Running inference")):
        image = image.to(device)

        with torch.no_grad():
            start_time = time.time()
            logits = model(image)
            elapsed_time = time.time() - start_time
            inference_times.append({
                "case": image_filenames[i],
                "inference_time": elapsed_time,
            })
            
            pred_maps = torch.argmax(logits, dim=1).squeeze(0) # (W, H, D)
            gt_maps = label.squeeze(0)  # (W, H, D)
            
        pred_maps = pred_maps.cpu().long()  # shape (W,H,D), integer class labels
        gt_maps   = gt_maps.cpu().long()
        
        # Get corresponding filename
        filename = image_filenames[i]
        gt_shape = tuple(gt_maps.shape) 
    
        
        pred_shape = pred_maps.shape
        if pred_shape != gt_shape:
            if check_prediction_shape_validity(pred_maps, gt_maps):
                raise ValueError(
                    f"Prediction shape {pred_shape} is larger than ground truth shape {gt_shape}. "
                    "This should not happen. Check your model and input data."
                )
            if encoding == "pad": # Crop using TORCHIO
                t = pred_maps.unsqueeze(0) #(1, W, H, D)
                transform = tio.CropOrPad(target_shape=gt_shape, padding_mode="constant")
                t2 = transform(t)
                pred_maps = t2.squeeze().long() # (W, H, D)
                print(f"Cropped/Padded via TorchIO: {pred_shape} → {pred_maps.shape}")
            elif encoding == "upsampled": 
                t = pred_maps.unsqueeze(0).unsqueeze(0).float() #(1, 1, W, H, D)
                t2 = F.interpolate(
                    t,
                    size=gt_shape,
                    mode="nearest"
                )
                pred_maps = t2.squeeze().long()
                print(f"Interpolated via F.interpolate: {pred_shape} → {gt_shape}")
            else:
                # Downsampled should never get in here anyways, since predshape=gtshape
                raise ValueError(
                    f"Unsupported encoding '{encoding}' for resizing predictions. "
                    "Use 'pad' or 'upsampled'."
                )
                
        print("Ignore index:", args.ignore_index, " num_classes:", cfg.dataset.num_classes)
        dsc = dice_coefficient(
            pred_maps, 
            gt_maps, 
            ignore_index=args.ignore_index, 
            num_classes=cfg.dataset.num_classes
        )
        dice_scores.append({"case": image_filenames[i], "dice_score": dsc.item()})
        
        
        img_nii = nib.load(os.path.join(args.imagests_dir, filename))
        pred_nii = nib.Nifti1Image(pred_maps.numpy(), affine=img_nii.affine, header=img_nii.header)
        tqdm.write(f"Storing image with shape {pred_maps.shape} at {filename}")
        nib.save(pred_nii, output_path / filename)
        tqdm.write(f"Saved: {output_path / filename}")
        
    # Save timing stats to JSON
    print("average_dice_score", np.mean([case["dice_score"] for case in dice_scores]))
    data_stats = {
        args.model: {
            "average_inference_time": float(np.mean([case["inference_time"] for case in inference_times])),
            "num_images": len(inference_times),
            "inference_times": inference_times, 
            "dice_scores": dice_scores,
            "average_dice_score": float(np.mean([case["dice_score"] for case in dice_scores])),
        }
    }

    stats_path = output_path / "data_stats.json"
    with open(stats_path, "w") as f:
        json.dump(data_stats, f, indent=4)
    tqdm.write(f"\nSaved timing stats to: {stats_path}")



if __name__ == "__main__":
    main()




