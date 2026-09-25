import os
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from PIL import Image

os.environ["MPLCONFIGDIR"] = "/tmp"
warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR   = SCRIPT_DIR.parent
EVAL_DIR   = ROOT_DIR.parent
PRED_PATH  = EVAL_DIR / "sticker" / "variations" / "sticker_e56_n56_bs128_default" / "predictions.parquet"
OUT_DIR    = ROOT_DIR / "analysis"
GRAPHS_DIR = OUT_DIR / "graphs"
MANUSCRIPT_FIGS_DIR = EVAL_DIR / "manuscript" / "figures"
PAPER_FIGS_DIR      = EVAL_DIR.parent / "paper" / "figures"

OUT_DIR.mkdir(parents=True, exist_ok=True)
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
MANUSCRIPT_FIGS_DIR.mkdir(parents=True, exist_ok=True)

RC = {
    "font.family":       "Times New Roman",
    "font.serif":        ["Times New Roman"],
    "mathtext.fontset":  "stix",
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "legend.fancybox":   False,
    "figure.dpi":        300,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linestyle":    "--",
}


def main():
    df = pd.read_parquet(PRED_PATH).sort_values("tick_idx").reset_index(drop=True)
    df["mid"]      = df["mid"].astype(float)
    df["sz"]       = df["sz"].astype(float)
    df["side_num"] = np.where(df["side"] == "B", 1, -1)

    step      = 200
    sub_df    = df.iloc[::step].copy().reset_index(drop=True)

    vol_series = pd.Series(sub_df["mid"]).pct_change().rolling(20).std().fillna(0.0).values * 10000.0
    signed_vol = sub_df["side_num"] * sub_df["sz"]
    ofi_series = pd.Series(signed_vol).rolling(20).mean().fillna(0.0).values

    X_regime   = np.column_stack([vol_series, ofi_series])
    valid_mask = ~np.isnan(X_regime).any(axis=1)
    X_clean    = X_regime[valid_mask]

    gmm        = GaussianMixture(n_components=3, covariance_type="full", random_state=42)
    raw_states = gmm.fit_predict(X_clean)

    means_vol = [X_clean[raw_states == k, 0].mean() for k in range(3)]
    order     = np.argsort(means_vol)
    remap     = {old: new for new, old in enumerate(order)}
    states    = np.array([remap[s] for s in raw_states])

    regime_names = {
        0: "Low-Vol Consolidation",
        1: "Moderate Trend",
        2: "High-Vol Extreme Shock",
    }

    trans_matrix = np.zeros((3, 3))
    for t in range(len(states) - 1):
        trans_matrix[states[t], states[t + 1]] += 1
    trans_matrix = trans_matrix / (trans_matrix.sum(axis=1, keepdims=True) + 1e-8)

    regime_stats = []
    for k in range(3):
        mask_k = states == k
        n_k    = int(mask_k.sum())
        regime_stats.append({
            "state":                k,
            "regime_name":          regime_names[k],
            "sample_count_binned":  n_k,
            "pct_time_spent":       round(n_k / len(states) * 100.0, 2),
            "mean_volatility_bps":  round(float(X_clean[mask_k, 0].mean()), 4),
            "mean_ofi_signed_vol":  round(float(X_clean[mask_k, 1].mean()), 4),
            "transition_self_prob": round(float(trans_matrix[k, k]), 4),
        })

    pd.DataFrame(regime_stats).to_csv(OUT_DIR / "market_regimes_summary.csv", index=False)
    with open(OUT_DIR / "market_regimes_summary.json", "w") as f:
        json.dump({"regimes": regime_stats, "transition_matrix": trans_matrix.tolist(),
                   "state_labels": regime_names}, f, indent=2)

    generate_plots(sub_df["mid"].values[valid_mask], states, trans_matrix)


def generate_plots(mids, states, trans_matrix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    rcParams.update(RC)

    print("[GRAPH] Figure 5: Market Regime Classification ...", flush=True)
    fig, ax  = plt.subplots(figsize=(14, 5.5))
    ticks_m  = np.arange(len(mids)) * 200 / 1e6
    colors   = {0: "#2ca02c", 1: "#1f77b4", 2: "#d62728"}
    labels   = {0: "Low-Vol Consolidation", 1: "Moderate Trend", 2: "High-Vol Extreme Shock"}

    for k in range(3):
        mask = states == k
        ax.scatter(ticks_m[mask], mids[mask], color=colors[k], s=0.4, alpha=0.7,
                   label=labels[k], rasterized=False)

    ax.set_xlabel("Ticks (millions)")
    ax.set_ylabel("Mid Price (USD)")
    ax.set_title("Hyperliquid BTC Perpetual Futures - Microstructural Market Regime Classification (11.9M Ticks)")
    ax.legend(loc="upper left", markerscale=12, frameon=True, facecolor="white",
              edgecolor="#cccccc", framealpha=0.9)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.91, bottom=0.12)
    out_uncropped = GRAPHS_DIR / "market_regime_classification.png"
    fig.savefig(out_uncropped, dpi=300)
    plt.close()
    print(f"  saved uncropped -> {out_uncropped}", flush=True)

    # Tightest crop version for manuscript/figures/
    crop_tight_regime_plot(out_uncropped, MANUSCRIPT_FIGS_DIR / "market_regime_classification.png")


def crop_tight_regime_plot(src_path: Path, dst_path: Path):
    im = Image.open(src_path).convert("RGB")
    arr = np.array(im)
    mask = ~np.all(arr >= 250, axis=2)

    # Find row where title ends and gap before legend starts
    row_has_ink = mask.any(axis=1)
    r_title_start = np.where(row_has_ink)[0][0]
    r_blank = None
    for r in range(r_title_start + 10, len(row_has_ink)):
        if not row_has_ink[r]:
            r_blank = r
            break

    # Content starts after blank row below title
    content_mask = mask.copy()
    if r_blank is not None:
        content_mask[:r_blank] = False

    rows = np.any(content_mask, axis=1)
    cols = np.any(content_mask, axis=0)
    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        cropped = im.crop((cmin, rmin, cmax + 1, rmax + 1))
    else:
        cropped = im

    cropped.save(dst_path, dpi=(300, 300))
    print(f"  saved tight crop -> {dst_path}", flush=True)

    if PAPER_FIGS_DIR.exists():
        cropped.save(PAPER_FIGS_DIR / "market_regime_classification.png", dpi=(300, 300))
        print(f"  saved tight crop -> {PAPER_FIGS_DIR / 'market_regime_classification.png'}", flush=True)


if __name__ == "__main__":
    main()
