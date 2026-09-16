"""
generate_consolidated_barcodes.py
==================================
Consolidated Barcode Generator across full 11.9M dataset.
Balanced Height Version (240px single, 160px per stacked row).
"""

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

BASE_DIR   = Path(__file__).resolve().parent.parent.parent.parent
OUT_DIR    = Path(__file__).resolve().parent.parent
GRAPHS_DIR = OUT_DIR / "graphs"
PRED_PATH  = BASE_DIR / "sticker" / "variations" / "sticker_e56_n56_bs128_default" / "predictions.parquet"

COLOR_UP   = np.array([0, 230, 0], dtype=np.uint8)    # Green (+1)
COLOR_DOWN = np.array([230, 0, 0], dtype=np.uint8)    # Red (-1)
COLOR_FLAT = np.array([128, 128, 128], dtype=np.uint8)# Gray (0)


def sequence_to_rgb_stripe(sequence: np.ndarray, height: int = 240, stripe_width: int = 8) -> np.ndarray:
    n = len(sequence)
    width = n * stripe_width
    img_arr = np.zeros((height, width, 3), dtype=np.uint8)

    for i, val in enumerate(sequence):
        if val > 0:
            c = COLOR_UP
        elif val < 0:
            c = COLOR_DOWN
        else:
            c = COLOR_FLAT
        
        start_x = i * stripe_width
        end_x = (i + 1) * stripe_width
        img_arr[:, start_x:end_x] = c

    return img_arr


def build_binned_sequence(preds_valid: np.ndarray, true_valid: np.ndarray, n_bins: int):
    n_valid = len(preds_valid)
    bin_size = n_valid // n_bins

    binned_preds = []
    binned_trues = []

    for b in range(n_bins):
        start_i = b * bin_size
        end_i   = (b + 1) * bin_size if b < n_bins - 1 else n_valid

        sub_p = preds_valid[start_i:end_i]
        sub_t = true_valid[start_i:end_i]

        sum_p = sub_p.sum()
        sum_t = sub_t.sum()

        sign_p = 1 if sum_p > 0 else (-1 if sum_p < 0 else 0)
        sign_t = 1 if sum_t > 0 else (-1 if sum_t < 0 else 0)

        binned_preds.append(sign_p)
        binned_trues.append(sign_t)

    return np.array(binned_preds), np.array(binned_trues)


def save_stacked_barcode(binned_preds, binned_trues, out_path, h_single=160, stripe_width=8, gap=12):
    w = len(binned_preds) * stripe_width
    stacked_arr = np.ones((h_single * 2 + gap, w, 3), dtype=np.uint8) * 255
    stacked_arr[:h_single] = sequence_to_rgb_stripe(binned_preds, height=h_single, stripe_width=stripe_width)
    stacked_arr[h_single + gap:] = sequence_to_rgb_stripe(binned_trues, height=h_single, stripe_width=stripe_width)

    img_stacked = Image.fromarray(stacked_arr)
    img_stacked.save(out_path)


def main():
    df = pd.read_parquet(PRED_PATH).sort_values("tick_idx").reset_index(drop=True)
    df["mid"]  = df["mid"].astype(float)
    df["pred"] = df["pred"].astype(int)

    mid_target = df["mid"].shift(-1)
    diff = mid_target - df["mid"]
    true_labels = np.where(diff > 1e-8, 1, np.where(diff < -1e-8, -1, 0))
    valid_mask = mid_target.notna().values

    preds_valid = df["pred"].values[valid_mask]
    true_valid  = true_labels[valid_mask]

    # Intermediate Consolidation (300 Bins ~ 40,000 ticks/bin)
    b_preds_300, b_trues_300 = build_binned_sequence(preds_valid, true_valid, n_bins=300)

    # 1. Single Sticker Barcode (240px Height)
    sticker_rgb = sequence_to_rgb_stripe(b_preds_300, height=240, stripe_width=8)
    Image.fromarray(sticker_rgb).save(GRAPHS_DIR / "sticker_pred_full_11.9M_consolidated_barcode.png")

    # 2. Single True Label Barcode (240px Height)
    true_rgb = sequence_to_rgb_stripe(b_trues_300, height=240, stripe_width=8)
    Image.fromarray(true_rgb).save(GRAPHS_DIR / "true_label_full_11.9M_consolidated_barcode.png")

    # 3. Stacked Sticker vs True Barcode (160px per row + 12px gap)
    save_stacked_barcode(b_preds_300, b_trues_300, GRAPHS_DIR / "sticker_vs_true_full_11.9M_stacked_barcode.png", h_single=160, stripe_width=8, gap=12)

    print("Balanced height (240px / 160px row) barcodes generated successfully!")


if __name__ == "__main__":
    main()
