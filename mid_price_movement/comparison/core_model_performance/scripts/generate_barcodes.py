"""
generate_barcodes.py
====================
Balanced Height (240px) Barcode Generator for First 200 Ticks.
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


def sequence_to_rgb_stripe(sequence: np.ndarray, height: int = 240, stripe_width: int = 10) -> np.ndarray:
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


def main():
    df = pd.read_parquet(PRED_PATH).sort_values("tick_idx").reset_index(drop=True)

    n_ticks = 200
    df_sub = df.iloc[:n_ticks + 1].copy()

    sticker_preds = df_sub["pred"].iloc[:n_ticks].values

    mids = df_sub["mid"].astype(float).values
    diffs = mids[1:n_ticks + 1] - mids[:n_ticks]
    true_labels = np.where(diffs > 1e-8, 1, np.where(diffs < -1e-8, -1, 0))

    # 1. Sticker Prediction Barcode (240px Height)
    sticker_rgb = sequence_to_rgb_stripe(sticker_preds, height=240, stripe_width=10)
    Image.fromarray(sticker_rgb).save(GRAPHS_DIR / "sticker_pred_first200_barcode.png")

    # 2. True Label Barcode (240px Height)
    true_rgb = sequence_to_rgb_stripe(true_labels, height=240, stripe_width=10)
    Image.fromarray(true_rgb).save(GRAPHS_DIR / "true_label_first200_barcode.png")

    # 3. Stacked Sticker vs True Label Barcode (160px per row + 12px gap)
    h_single = 160
    w = 200 * 10
    gap = 12
    stacked_arr = np.ones((h_single * 2 + gap, w, 3), dtype=np.uint8) * 255
    stacked_arr[:h_single] = sequence_to_rgb_stripe(sticker_preds, height=h_single, stripe_width=10)
    stacked_arr[h_single + gap:] = sequence_to_rgb_stripe(true_labels, height=h_single, stripe_width=10)

    Image.fromarray(stacked_arr).save(GRAPHS_DIR / "sticker_vs_true_first200_stacked_barcode.png")
    print("First 200 ticks balanced height (240px) barcodes generated successfully!")


if __name__ == "__main__":
    main()
