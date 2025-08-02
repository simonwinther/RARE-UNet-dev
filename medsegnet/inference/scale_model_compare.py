#!/usr/bin/env python3
import os
import argparse
import logging
from typing import Dict, Optional

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from utils import setup_font
import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
_dataset_configs = {
    "braintumour": {
        "colors": ["black", "#e41a1c", "#1fe41c", "#ffd700"],
        "label_names": ["Background (0)",  "Edema (1)", "Non-enhancing Tumor (2)", "Enhancing Tumor (3)"],
    },
    "hippocampus": { "colors": ["black", "#ff0000", "#00ff15"], "label_names": ["Background (0)", "Left Hippocampus (1)", "Right Hippocampus (2)"], }
}
MODEL_NAME_MAP = {"RARE-UNet": "Multiscale", "U-Net": "Backbone", "U-Net+Aug": "Backbone+Aug"}
MODEL_ORDER = ["GT", "RARE-UNet", "U-Net", "U-Net+Aug", "nnU-Net"]

MODELS_CONFIG = {
    "GT":        {"color": "white", "alpha": 0.4, "is_contour": False},
    "RARE-UNet": {"color": "#e41a1c", "alpha": 0.4, "is_contour": False},
    "U-Net":     {"color": "#377eb8", "alpha": 0.4, "is_contour": False},
    "U-Net+Aug": {"color": "#4daf4a", "alpha": 0.4, "is_contour": False},
    "nnU-Net":   {"color": "#984ea3", "alpha": 0.4, "is_contour": False},
}
SCALE_LABELS = {"fullres": "Scale 1", "scale1": "Scale 1/2", "scale2": "Scale 1/4", "scale3": "Scale 1/8"}

# --- YOUR PERFECT LAYOUT CONFIGURATION ---
LAYOUT_CONFIG = {
    "braintumour": {"figsize_w_mult": 2.5, "figsize_h_mult": 2.5},
    # --- MODIFIED: Create a square canvas to draw our stretched image onto. ---
    "hippocampus": {"figsize_w_mult": 2.5, "figsize_h_mult": 2.5} 
}


OFFSET = 5
LEGEND_FONTSIZE = 12 + OFFSET
MODEL_FONTSIZE = 14 + OFFSET
SCALE_FONTSIZE = 14 + OFFSET
NOT_ANNOUNCED_FONTSIZE = 12 + OFFSET

def load_nifti(path: str) -> np.ndarray: return nib.load(path).get_fdata()

def get_best_axial_slice_idx(gt_mask: np.ndarray) -> int:
    if gt_mask.ndim == 4: gt_mask = np.mean(gt_mask, axis=-1).astype(int)
    else: gt_mask = gt_mask.astype(int)
    _W, _H, D = gt_mask.shape
    classes = [int(c) for c in np.unique(gt_mask) if c != 0]
    if not classes: return D // 2
    stats = max([(min(np.count_nonzero(gt_mask[:, :, i] == c) for c in classes), sum(np.count_nonzero(gt_mask[:, :, i] == c) for c in classes), i) for i in range(D)], key=lambda x: (x[0], x[1]))
    if stats[0] == 0: return D // 2
    return stats[2]

def plot_scales_vs_models(scales_data, model_order, models_config, out_path, dataset_name, cid):
    logger.info(f"[{cid}] Plotting case with native resolutions per row.")

    cfg = _dataset_configs.get(dataset_name)
    class_cmap = ListedColormap(cfg["colors"])

    scale_names = sorted([s for s in scales_data if scales_data[s]], key=lambda x: ('0' if x == 'fullres' else x))
    n_rows, n_cols = len(scale_names), len(model_order)

    layout_cfg = LAYOUT_CONFIG[dataset_name]
    logger.info(f"Using layout strategy for '{dataset_name}'")
    figsize = (n_cols * layout_cfg['figsize_w_mult'], n_rows * layout_cfg['figsize_h_mult'])
    
    constrained_layout_pads = {'w_pad': 0.05, 'h_pad': 0.05, 'wspace': 0.02, 'hspace': 0.01}
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, 
                             constrained_layout=True)
    fig.set_constrained_layout_pads(**constrained_layout_pads)


    if n_rows == 1: axes = np.array([axes])

    for r, scale_name in enumerate(scale_names):
        scale_info = scales_data[scale_name]
        if 'image_path' not in scale_info or 'GT' not in scale_info:
            for c_ax in axes[r, :]: c_ax.axis('off')
            continue
        img_vol, gt_vol = load_nifti(scale_info['image_path']), load_nifti(scale_info['GT'])
        axial_idx = get_best_axial_slice_idx(gt_vol)
        if img_vol.ndim == 4: img_slice = np.rot90(np.mean(img_vol[:, :, axial_idx, :], axis=-1))
        else: img_slice = np.rot90(img_vol[:, :, axial_idx])
        axes[r, 0].set_ylabel(SCALE_LABELS.get(scale_name, scale_name), fontsize=SCALE_FONTSIZE, labelpad=15)
        for c, model_name in enumerate(model_order):
            ax = axes[r, c]
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0: ax.set_title(model_name, fontsize=MODEL_FONTSIZE, pad=10)
            if model_name not in scale_info:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=NOT_ANNOUNCED_FONTSIZE, color='gray')
                ax.set_facecolor('0.95'); [s.set_visible(False) for s in ax.spines.values()]
                continue
            pred_vol = load_nifti(scale_info[model_name])
            mask_slice = np.rot90(pred_vol[..., axial_idx])
            vmin, vmax = np.percentile(img_slice, [5, 99])
            ax.imshow(img_slice, cmap="gray", vmin=vmin, vmax=vmax)
            m_cfg = models_config[model_name]
            masked_overlay = np.ma.masked_where(mask_slice == 0, mask_slice)
            ax.imshow(masked_overlay, cmap=class_cmap, alpha=m_cfg["alpha"], interpolation="none", vmin=0, vmax=len(cfg["colors"]) - 1)
            
            # This calculates the image's width/height ratio and applies it as an
            # aspect stretch, forcing the final rendered image to appear square.
            ax.set_aspect(img_slice.shape[1] / img_slice.shape[0])

            h_img, w_img = img_slice.shape
            ax.set_xlim(-0.5, w_img - 0.5); ax.set_ylim(h_img - 0.5, -0.5)
            coords = np.argwhere(mask_slice > 0)
            if coords.size > 0:
                y_min, x_min = coords.min(axis=0); y_max, x_max = coords.max(axis=0)
                padding_factor = 0.2
                pad_y = max(1, int((y_max - y_min) * padding_factor))
                pad_x = max(1, int((x_max - x_min) * padding_factor))
                y0 = max(0, y_min - pad_y); x0 = max(0, x_min - pad_x)
                y1 = min(mask_slice.shape[0], y_max + pad_y + 1)
                x1 = min(mask_slice.shape[1], x_max + pad_x + 1)
                rect = patches.Rectangle((x0, y0), (x1-x0-1), (y1-y0-1), linewidth=1.2, edgecolor="yellow", facecolor="none", linestyle="--")
                ax.add_patch(rect)
                
                center_y = (y_min + y_max) / 2
                inset_loc = 'lower right' if center_y < h_img / 2 else 'upper right'

                axins = inset_axes(ax, width="40%", height="40%", loc=inset_loc)
                patch_img = img_slice[y0:y1, x0:x1]; patch_mask = mask_slice[y0:y1, x0:x1]
                axins.imshow(patch_img, cmap="gray", vmin=vmin, vmax=vmax, aspect='equal')
                masked_patch = np.ma.masked_where(patch_mask == 0, patch_mask)
                axins.imshow(masked_patch, cmap=class_cmap, alpha=m_cfg["alpha"], interpolation="none", vmin=0, vmax=len(cfg["colors"]) - 1, aspect='equal')
                h_patch, w_patch = patch_img.shape
                axins.set_xlim(-0.5, w_patch - 0.5); axins.set_ylim(h_patch - 0.5, -0.5)
                axins.set_xticks([]); axins.set_yticks([]);
                [s.set_edgecolor("yellow") for s in axins.spines.values()]

    overlay_alpha = MODELS_CONFIG["GT"]["alpha"]
    legend_el = [mpatches.Patch(color=c, label=n, alpha=overlay_alpha) for i, (n, c) in enumerate(zip(cfg["label_names"], cfg["colors"])) if i != 0]
    
    fig.legend(handles=legend_el, loc="upper center", bbox_to_anchor=(0.5, 0.01), ncol=len(legend_el), fontsize=LEGEND_FONTSIZE, frameon=False)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    logger.info(f"[{cid}] Figure saved to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate figures comparing models at their native scales.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--main-preds-dir", required=True, help="Root directory for your main predictions")
    parser.add_argument("--nnunet-preds-dir", required=True, help="Root directory for nnU-Net predictions")
    parser.add_argument("--dataset-dir", required=True, help="Root directory of the dataset")
    parser.add_argument("--dataset", required=True, choices=_dataset_configs.keys(), help="Name of the dataset")
    parser.add_argument("--output-dir", required=True, help="Directory to save the output figures.")
    parser.add_argument("--case-ids", nargs="+", required=True, help="Specific case IDs to process")
    parser.add_argument("--font-path", type=str, default=None, help="Optional: Path to a .ttf font file to use for all plot text.")
    args = parser.parse_args()
    setup_font(args.font_path)

    for cid in args.case_ids:
        logger.info(f"--- Processing Case: {cid} ---")
        scales_data = {}
        scales_data['fullres'] = {}
        img_path = os.path.join(args.dataset_dir, 'lowres', 'downsampled', 'scale0', 'imagesTs', f"{cid}.nii.gz")
        gt_path = os.path.join(args.dataset_dir, 'lowres', 'downsampled', 'scale0', 'labelsTs', f"{cid}.nii.gz")
        if os.path.exists(img_path): scales_data['fullres']['image_path'] = img_path
        if os.path.exists(gt_path): scales_data['fullres']['GT'] = gt_path
        for model_name, model_folder in MODEL_NAME_MAP.items():
            path = os.path.join(args.main_preds_dir, args.dataset, 'fullres', model_folder, f"{cid}.nii.gz")
            if os.path.exists(path): scales_data['fullres'][model_name] = path
        nn_path = os.path.join(args.nnunet_preds_dir, args.dataset, 'fullres', f"{cid}.nii.gz")
        if os.path.exists(nn_path): scales_data['fullres']['nnU-Net'] = nn_path
        if not scales_data.get('fullres', {}).get('image_path') or not scales_data.get('fullres', {}).get('GT'):
            if 'fullres' in scales_data: del scales_data['fullres']
        for scale_num in [1, 2, 3]:
            scale_name = f'scale{scale_num}'
            scale_path = os.path.join('lowres', 'downsampled', scale_name)
            scales_data[scale_name] = {}
            img_path = os.path.join(args.dataset_dir, scale_path, 'imagesTs', f"{cid}.nii.gz")
            gt_path = os.path.join(args.dataset_dir, scale_path, 'labelsTs', f"{cid}.nii.gz")
            if os.path.exists(img_path): scales_data[scale_name]['image_path'] = img_path
            if os.path.exists(gt_path): scales_data[scale_name]['GT'] = gt_path
            for model_name, model_folder in MODEL_NAME_MAP.items():
                path = os.path.join(args.main_preds_dir, args.dataset, scale_path, model_folder, f"{cid}.nii.gz")
                if os.path.exists(path): scales_data[scale_name][model_name] = path
            nn_path = os.path.join(args.nnunet_preds_dir, args.dataset, scale_path, f"{cid}.nii.gz")
            if os.path.exists(nn_path): scales_data[scale_name]['nnU-Net'] = nn_path
            if not scales_data.get(scale_name, {}).get('image_path') or not scales_data.get(scale_name, {}).get('GT'):
                 if scale_name in scales_data: del scales_data[scale_name]

        out_path = os.path.join(args.output_dir, f"comparison_native_{args.dataset}_{cid}.pdf")
        try:
            plot_scales_vs_models(scales_data, MODEL_ORDER, MODELS_CONFIG, out_path, args.dataset, cid)
        except Exception as e:
            logger.error(f"Failed to generate plot for case '{cid}': {e}", exc_info=True)

if __name__ == "__main__":
    main()