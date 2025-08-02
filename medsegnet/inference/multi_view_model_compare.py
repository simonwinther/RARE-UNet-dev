#!/usr/bin/env python3
import os
import numpy as np
from utils import setup_font
import nibabel as nib
import matplotlib.pyplot as plt
import argparse
from textwrap import dedent
from typing import Tuple, List
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------
# Dataset configuration
# ----------------------
_dataset_configs = {
    "Task01_BrainTumour": {
        "colors": [
            "black",  
            "#e41a1c",    
            "#1fe41c",    
            "#ffd700",  
        ],
        "label_names": [
            "Background (0)",
            "Edema (1)",
            "Non-enhancing Tumor (2)",
            "Enhancing Tumor (3)",
        ],
    },
    "Task04_Hippocampus": {
        "colors": [
            "black",   # 0 = background
            "#ff0000", 
            "#00ff15", 
        ],
        "label_names": ["Background (0)", "Left Hippocampus (1)", "Right Hippocampus (2)"],
    },
}

# ---------------------
# Utility to load NIfTI
# ---------------------
def load_nifti(path: str) -> np.ndarray:
    return nib.load(path).get_fdata()

# ----------------
# Pad image to a target shape
# ----------------
def pad_to_shape(img: np.ndarray, target_shape: Tuple[int,int]) -> np.ndarray:
    pad_y = (target_shape[0] - img.shape[0]) // 2
    pad_x = (target_shape[1] - img.shape[1]) // 2
    return np.pad(
        img,
        (
            (pad_y, target_shape[0] - img.shape[0] - pad_y),
            (pad_x, target_shape[1] - img.shape[1] - pad_x),
        ),
        mode="constant",
    )

# ---------------------------------------------------
# Slicing function: choose slice balanced across classes
# ---------------------------------------------------
def get_mid_slices(volume: np.ndarray):
    # Collapse modalities if present
    if volume.ndim == 4:
        volume = np.mean(volume, axis=-1)
    W, H, D = volume.shape
    axial = np.rot90(volume[:, :, D // 2])
    coronal = np.rot90(volume[:, H // 2, :])
    sagittal = np.rot90(volume[W // 2, :, :])
    return axial, coronal, sagittal


def get_slices_balanced_by_min_count(
    volume: np.ndarray,
    gt_mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if volume.ndim == 4:
        volume = np.mean(volume, axis=-1)
    if gt_mask.ndim == 4:
        gt_mask = np.mean(gt_mask, axis=-1).astype(int)
    else:
        gt_mask = gt_mask.astype(int)

    W, H, D = volume.shape
    classes = [int(c) for c in np.unique(gt_mask) if c != 0]
    if not classes:
        # no labels → exact centers
        return (
            np.rot90(volume[:, :, D // 2]),
            np.rot90(volume[:, H // 2, :]),
            np.rot90(volume[W // 2, :, :]),
        )

    def best_idx(axis_len: int, slice_fn) -> int:
        stats = []
        for i in range(axis_len):
            sl = slice_fn(i)
            counts = [int(np.count_nonzero(sl == c)) for c in classes]
            stats.append((min(counts), sum(counts), i))
        best_min, best_tot, best_i = max(stats, key=lambda x: (x[0], x[1]))
        if best_min == 0:
            return axis_len // 2
        return best_i

    z = best_idx(D, lambda i: gt_mask[:, :, i])
    y = best_idx(H, lambda i: gt_mask[:, i, :])
    x = best_idx(W, lambda i: gt_mask[i, :, :])

    axial    = np.rot90(volume[:, :, z])
    coronal  = np.rot90(volume[:, y, :])
    sagittal = np.rot90(volume[x, :, :])
    return axial, coronal, sagittal

# ----------------
# Plotting routine
# ----------------
def plot_segmentation(
    image_path: str,
    seg_paths:    dict,
    out_path:     str,
    use_contour:  bool,
    cid:          str,
    flip:         bool = False,
):
    logger.info(f"[{cid}] Starting case")
    # 1. Load volumes
    img     = load_nifti(image_path)
    gt_mask = load_nifti(seg_paths["gt"])
    logger.info(f"[{cid}] Loaded image {img.shape}, GT {gt_mask.shape}")

    # Extract the three “best” slices from image and GT
    #slices         = get_slices_balanced_by_min_count(img,      gt_mask)
    #mask_gt_slices = get_slices_balanced_by_min_count(gt_mask, gt_mask)
    
    _fn_slices = get_mid_slices
    slices         = _fn_slices(img)
    mask_gt_slices = _fn_slices(gt_mask)
    logger.info(f"[{cid}] Slice shapes  (mid): { [s.shape for s in slices] }")
    logger.info(f"[{cid}] Slice shapes (GT):  { [s.shape for s in mask_gt_slices] }")

    # 2. Setup config & colormap
    lower = image_path.lower()
    for key,cfg in _dataset_configs.items():
        if key.lower() in lower:
            break
    else:
        raise ValueError(f"Cannot detect dataset from '{image_path}'")
    label_colors = cfg["colors"]
    class_cmap   = ListedColormap(label_colors)
    plane_names  = ["Axial", "Coronal", "Sagittal"]

    # Pad to uniform shape
    target_shape    = tuple(np.max([s.shape for s in slices], axis=0))
    logger.info(f"[{cid}] Target 2D shape for all slices: {target_shape}")
    
    slices          = tuple(pad_to_shape(s, target_shape) for s in slices)
    mask_gt_slices  = tuple(pad_to_shape(s, target_shape) for s in mask_gt_slices)

    # Models to plot (GT now uses use_contour flag)
    models = [
        ("GT",             seg_paths["gt"],    "white",   1.0 if use_contour else 0.4, use_contour),
        ("RARE-UNet",      seg_paths["ours"],  "red",     0.4,                   False),
        ("U-Net",          seg_paths["bb"],    "green",   0.4,                   False),
        ("U-Net+Aug",      seg_paths["bb_aug"],"blue",    0.4,                   False),
        ("nnU-Net",        seg_paths["nnunet"],"yellow",  0.4,                   False),
    ]

    # 3a. Non-flip layout: rows=models, cols=planes
    if not flip:
        n_rows, n_cols = len(models), 3
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(n_cols*3, n_rows*2.5),
                                 constrained_layout=False)

        for col, title in enumerate(plane_names):
            axes[0, col].set_title(title, fontsize=12, pad=8)

        fig.tight_layout(h_pad=-1, w_pad=-12.2)
    

        # plot each model
        for r, (label, mpath, color, alpha, is_contour) in enumerate(models):
            if label == "GT":
                pred_slices = mask_gt_slices
            else:
                pred = load_nifti(mpath)
                logger.info(f"[{cid}] {label} prediction volume shape: {pred.shape}")
                pred_slices = _fn_slices(pred)
            logger.info(f"[{cid}] {label} prediction slice shapes: { [s.shape for s in pred_slices] }")
            pred_slices = tuple(pad_to_shape(s, target_shape) for s in pred_slices)

            for c in range(n_cols):
                ax = axes[r, c]
                img_slice  = slices[c]
                mask_slice = pred_slices[c]
                vmin, vmax = np.percentile(img_slice, [5, 99])

                # background
                ax.imshow(img_slice, cmap="gray", vmin=vmin, vmax=vmax)

                # overlay
                if is_contour:
                    ax.contour(mask_slice, colors=color, linewidths=1)
                else:
                    mp = np.ma.masked_where(mask_slice == 0, mask_slice)
                    ax.imshow(mp, cmap=class_cmap, alpha=alpha,
                              interpolation="none", vmin=0, vmax=len(label_colors)-1)

                if c == 0:
                            # only first column keeps its spine so we can draw the row label
                            ax.set_xticks([])
                            ax.set_yticks([])
                            ax.set_ylabel(
                                label,              # the model name: "GT", "Proposed Model", etc.
                                rotation=90,
                                fontsize=12,
                                labelpad=8
                            )
                else:
                    ax.axis("off") 

                # zoom-in inset with rectangle and visible border
                coords = np.argwhere(mask_slice > 0)
                if coords.size:
                    y0, x0 = coords.min(axis=0) - 5
                    y1, x1 = coords.max(axis=0) + 5
                    y0, x0 = max(0, y0), max(0, x0)
                    y1 = min(mask_slice.shape[0], y1)
                    x1 = min(mask_slice.shape[1], x1)

                    # draw rectangle on main
                    rect = patches.Rectangle((x0, y0), x1-x0, y1-y0,
                                             linewidth=1, edgecolor="yellow",
                                             facecolor="none", linestyle="--")
                    ax.add_patch(rect)

                    # inset
                    axins = inset_axes(ax, width="30%", height="30%",
                                       loc="upper right",
                                       bbox_to_anchor=(0,0,1,1),
                                       bbox_transform=ax.transAxes)
                    patch_img  = img_slice[y0:y1, x0:x1]
                    patch_mask = mask_slice[y0:y1, x0:x1]
                    axins.imshow(patch_img, cmap="gray", vmin=vmin, vmax=vmax)
                    if is_contour:
                        axins.contour(patch_mask, colors=color, linewidths=1)
                    else:
                        mp2 = np.ma.masked_where(patch_mask==0, patch_mask)
                        axins.imshow(mp2, cmap=class_cmap, alpha=alpha,
                                     interpolation="none",
                                     vmin=0, vmax=len(label_colors)-1)
                    # visible inset border
                    for spine in axins.spines.values():
                        spine.set_edgecolor("white")
                        spine.set_linewidth(1)
                    axins.set_xticks([]); axins.set_yticks([])

        # legend
        legend_el = [
            mpatches.Patch(color=label_colors[i],
                          label=f"{cfg['label_names'][i]}",
                          alpha=0.4)
            for i in range(1, len(label_colors))
        ]
        fig.legend(handles=legend_el,
                   loc="lower center", ncol=len(legend_el),
                   fontsize=10, frameon=False, bbox_to_anchor=(0.5,0))

    # 3b. Flip layout: rows=planes, cols=models
    else:
        n_rows, n_cols = 3, len(models)
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(n_cols*3, n_rows*2.5),
                                 constrained_layout=False)
        fig.tight_layout(h_pad=-1.7, w_pad=-20)

        for r, plane in enumerate(plane_names):
            axes[r, 0].axis("on")
            axes[r, 0].set_ylabel(
                plane,
                rotation=90,
                fontsize=12,
                labelpad=8
            )
            axes[r, 0].set_xticks([])
            axes[r, 0].set_yticks([])

        # column titles
        for col, (label, *_rest) in enumerate(models):
            axes[0, col].set_title(label, fontsize=10, pad=8)

        for r in range(n_rows):
            img_slice = slices[r]
            vmin, vmax = np.percentile(img_slice, [5, 99])
            for c, (label, mpath, color, alpha, is_contour) in enumerate(models):
                ax = axes[r, c]
                ax.imshow(img_slice, cmap="gray", vmin=vmin, vmax=vmax)
                
                # load & slice this model
                
                if label == "GT":
                    pred_slices = mask_gt_slices
                else:
                    pred = load_nifti(mpath)
                    pred_slices = _fn_slices(pred)
                m_slice = pad_to_shape(pred_slices[r], target_shape)

                # overlay
                if is_contour:
                    ax.contour(m_slice, colors=color, linewidths=1)
                else:
                    mp = np.ma.masked_where(m_slice == 0, m_slice)
                    ax.imshow(mp, cmap=class_cmap, alpha=alpha,
                              interpolation="none",
                              vmin=0, vmax=len(label_colors)-1)

                if c == 0:
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_ylabel(
                        plane_names[r],
                        rotation=90,
                        fontsize=12,
                        labelpad=8
                    )
                else:
                    ax.axis("off")


                # inset & rectangle
                coords = np.argwhere(m_slice > 0)
                if coords.size:
                    y0, x0 = coords.min(axis=0) - 5
                    y1, x1 = coords.max(axis=0) + 5
                    y0, x0 = max(0, y0), max(0, x0)
                    y1 = min(m_slice.shape[0], y1)
                    x1 = min(m_slice.shape[1], x1)

                    rect = patches.Rectangle((x0, y0), x1-x0, y1-y0,
                                             linewidth=1, edgecolor="yellow",
                                             facecolor="none", linestyle="--")
                    ax.add_patch(rect)

                    axins = inset_axes(ax, width="30%", height="30%",
                                       loc="upper right",
                                       bbox_to_anchor=(0,0,1,1),
                                       bbox_transform=ax.transAxes)
                    patch_img  = img_slice[y0:y1, x0:x1]
                    patch_mask = m_slice[y0:y1, x0:x1]
                    axins.imshow(patch_img, cmap="gray", vmin=vmin, vmax=vmax)
                    if is_contour:
                        axins.contour(patch_mask, colors=color, linewidths=1)
                    else:
                        mp2 = np.ma.masked_where(patch_mask==0, patch_mask)
                        axins.imshow(mp2, cmap=class_cmap, alpha=alpha,
                                     interpolation="none",
                                     vmin=0, vmax=len(label_colors)-1)
                    for spine in axins.spines.values():
                        spine.set_edgecolor("white")
                        spine.set_linewidth(1)
                    axins.set_xticks([]); axins.set_yticks([])

        # legend
        legend_el = [
            mpatches.Patch(color=label_colors[i],
                          label=f"{cfg['label_names'][i]}",
                          alpha=0.4)
            for i in range(1, len(label_colors))
        ]
        fig.legend(handles=legend_el,
                   loc="lower center", ncol=len(legend_el),
                   fontsize=10, frameon=False, bbox_to_anchor=(0.5,0))

    # 4. Save and close
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

# ----------------
# CLI entry-point
# ----------------
def main():
    parser = argparse.ArgumentParser(
        description="Plot segmentations for multiple image cases."
    )
    parser.add_argument("--images", nargs="+",  required=True, help="Paths to image files")
    parser.add_argument("--ours-seg",         required=True, help="Folder of our model segs")
    parser.add_argument("--bb-seg",           required=True, help="Folder of Backbone segs")
    parser.add_argument("--bb-aug-seg",       required=True, help="Folder of Backbone+Aug segs")
    parser.add_argument("--nnunet-seg",       required=True, help="Folder of nnU-Net segs")
    parser.add_argument("--gt-seg",           required=True, help="Folder of GT masks")
    parser.add_argument("--out-dir",          required=True, help="Output directory")
    parser.add_argument("--contour", action="store_true",
                        help="Use contour for GT instead of overlay")
    parser.add_argument("--flips", type=int, nargs="*",
                        help="0/1 flags to flip each image", default=None)
    parser.add_argument("--names", type=str, nargs="*",
                        help="Suffix per case", default=[])
    parser.add_argument("--font-path", type=str, default=None, help="Optional: Path to a .ttf font file to use for all plot text.")

    args = parser.parse_args()
    setup_font(args.font_path)
    
    def case_id(path: str) -> str:
        return os.path.basename(path).replace(".nii.gz", "")

    for idx, image_path in enumerate(args.images):
        cid = case_id(image_path)
        seg_paths = {
            "gt":     os.path.join(args.gt_seg,     f"{cid}.nii.gz"),
            "ours":   os.path.join(args.ours_seg,   f"{cid}.nii.gz"),
            "bb":     os.path.join(args.bb_seg,     f"{cid}.nii.gz"),
            "bb_aug": os.path.join(args.bb_aug_seg, f"{cid}.nii.gz"),
            "nnunet": os.path.join(args.nnunet_seg, f"{cid}.nii.gz"),
        }
        suffix = f"_{args.names[idx]}" if idx < len(args.names) else ""
        out_path = os.path.join(args.out_dir, f"case_{cid}{suffix}.pdf")
        flip     = bool(args.flips[idx]) if args.flips and idx < len(args.flips) else False

        plot_segmentation(image_path, seg_paths, out_path,
                          use_contour=args.contour, cid=cid, flip=flip)

if __name__ == "__main__":
    main()
