import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

EVAL_ROOT = Path(__file__).resolve().parent.parent.parent

HORIZONS  = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500]

CELL_TAGS = [["TP", "FP"], ["FN", "TN"]]


def compute_horizon_cm(preds, mids, h):
    n      = len(mids) - h
    diff   = mids[h:] - mids[:n]
    y_true = np.where(diff > 1e-8, 1, -1)
    y_pred = preds[:n]
    tp = int(((y_true == 1)  & (y_pred == 1)).sum())
    fp = int(((y_true == -1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1)  & (y_pred == -1)).sum())
    tn = int(((y_true == -1) & (y_pred == -1)).sum())
    return np.array([[tp, fp], [fn, tn]])


def fmt_count(n):
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def draw_single_cm(cm, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    total    = cm.sum()
    norm_cm  = cm.astype(float) / (total + 1e-8)
    max_val  = cm.max()

    cmap     = mcolors.LinearSegmentedColormap.from_list("navy", ["#eaf0fb", "#0d2b6e"])
    sz       = 2.2

    fig = plt.figure(figsize=(sz, sz), facecolor="#ffffff")
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("#ffffff")

    for r in range(2):
        for c in range(2):
            val      = cm[r, c]
            pct      = norm_cm[r, c]
            rel      = val / (max_val + 1e-8)
            tag      = CELL_TAGS[r][c]
            color    = cmap(0.10 + rel * 0.90)
            lum      = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
            txt_col  = "#ffffff" if lum < 0.50 else "#1a1a2e"

            x0 = c * 1.0
            y0 = (1 - r) * 1.0

            rect = plt.Rectangle((x0, y0), 1.0, 1.0,
                                  facecolor=color, edgecolor="#ffffff", linewidth=2.5)
            ax.add_patch(rect)

            cx = x0 + 0.5
            cy = y0 + 0.5

            ax.text(cx, cy + 0.175, tag,
                    ha="center", va="center", color=txt_col,
                    fontsize=9, fontweight="bold", fontfamily="Times New Roman")

            ax.text(cx, cy - 0.02, fmt_count(val),
                    ha="center", va="center", color=txt_col,
                    fontsize=10, fontweight="bold", fontfamily="Times New Roman")

            ax.text(cx, cy - 0.21, f"{pct * 100:.1f}%",
                    ha="center", va="center", color=txt_col,
                    fontsize=8, fontfamily="Times New Roman")

    fig.savefig(out_path, dpi=250, bbox_inches=None, pad_inches=0)
    plt.close()


def process_model(model_dir):
    pred_path = model_dir / "predictions.parquet"
    if not pred_path.exists():
        return

    graphs_dir = model_dir / "graphs"
    graphs_dir.mkdir(exist_ok=True)

    df    = pd.read_parquet(pred_path).sort_values("tick_idx").reset_index(drop=True)
    n     = len(df)
    cut   = int(n * 0.25)
    df    = df.iloc[cut:].reset_index(drop=True)
    mids  = df["mid"].astype(float).values
    preds = df["pred"].values

    for h in HORIZONS:
        cm       = compute_horizon_cm(preds, mids, h)
        out_path = graphs_dir / f"confusion_matrix_h{h}.png"
        draw_single_cm(cm, out_path)


def main():
    model_dirs = [
        p.parent
        for p in EVAL_ROOT.rglob("trading_metrics.json")
        if "comparison" not in str(p) and "sticker_best_metrics" not in str(p)
    ]

    total = len(model_dirs)
    print(f"[CM] Generating per-horizon confusion matrices for {total} models ...", flush=True)

    for i, model_dir in enumerate(sorted(model_dirs), 1):
        try:
            process_model(model_dir)
            print(f"  [{i}/{total}] {model_dir.name}", flush=True)
        except Exception as e:
            print(f"  [{i}/{total}] SKIP {model_dir.name}: {e}", flush=True)

    print("[CM COMPLETE]", flush=True)


if __name__ == "__main__":
    main()
