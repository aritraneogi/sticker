import sys
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

SCRIPT_DIR  = Path(__file__).resolve().parent
ROOT_DIR    = SCRIPT_DIR.parent
EVAL_DIR    = ROOT_DIR.parent
MASTER_PATH = ROOT_DIR / "extracted" / "btc_trades_master.parquet"
PRED_PATH   = EVAL_DIR / "sticker" / "variations" / "sticker_e56_n56_bs128_default" / "predictions.parquet"
OUT_DIR     = ROOT_DIR / "analysis"
GRAPHS_DIR  = OUT_DIR / "graphs"

OUT_DIR.mkdir(parents=True, exist_ok=True)
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500]

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
    "figure.dpi":        150,
    "savefig.dpi":       250,
    "savefig.bbox":      "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.28,
    "grid.linestyle":    "--",
}


def main():
    print(f"[INGEST] Reading predictions parquet (mid prices) ...", flush=True)
    pred_df = pd.read_parquet(PRED_PATH).sort_values("tick_idx").reset_index(drop=True)
    mids_arr = pred_df["mid"].astype(float).values
    ts_ns    = pd.to_datetime(pred_df["ts_ns"]).values.astype(np.int64)

    print(f"[INGEST] Reading master parquet (trade metadata) ...", flush=True)
    raw = pd.read_parquet(MASTER_PATH).sort_values("tick_idx").reset_index(drop=True)

    print(f"[INGEST COMPLETE] {len(mids_arr):,} ticks", flush=True)

    horizon_stats = []
    df_horizons   = {}

    print("[ANALYSIS] Computing horizon statistics ...", flush=True)
    for h in HORIZONS:
        n       = len(mids_arr) - h
        diff    = mids_arr[h:] - mids_arr[:n]
        ret_bps = (diff / (mids_arr[:n] + 1e-8)) * 10000.0

        n_up    = int((diff > 1e-8).sum())
        n_down  = int((diff < -1e-8).sum())
        n_flat  = int((np.abs(diff) <= 1e-8).sum())
        n_total = n

        df_horizons[h] = ret_bps
        horizon_stats.append({
            "horizon":         h,
            "total_samples":   n_total,
            "n_up":            n_up,
            "pct_up":          round(n_up   / n_total * 100.0, 2),
            "n_down":          n_down,
            "pct_down":        round(n_down  / n_total * 100.0, 2),
            "n_flat":          n_flat,
            "pct_flat":        round(n_flat  / n_total * 100.0, 2),
            "mean_return_bps": round(float(np.mean(ret_bps)),        4),
            "std_return_bps":  round(float(np.std(ret_bps)),         4),
            "skewness":        round(float(stats.skew(ret_bps)),      4),
            "kurtosis":        round(float(stats.kurtosis(ret_bps)), 4),
        })
        print(f"  h={h} done", flush=True)

    df_stats = pd.DataFrame(horizon_stats)
    df_stats.to_csv(OUT_DIR / "dataset_summary_statistics.csv", index=False)
    with open(OUT_DIR / "dataset_summary_statistics.json", "w") as f:
        json.dump(horizon_stats, f, indent=2)
    print("[OUTPUT] Saved dataset statistics", flush=True)

    raw_ts_ns = pd.to_datetime(raw["time"]).values.astype(np.int64)

    generate_figures(mids_arr, raw_ts_ns, df_horizons, df_stats)


def generate_figures(mids, timestamps, df_horizons, df_stats):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    rcParams.update(RC)

    print("[GRAPH] Figure 1: Full-Month Mid Price Path ...", flush=True)
    ticks_m = np.arange(len(mids)) / 1e6
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(ticks_m[::100], mids[::100], color="#1a1a6e", lw=0.6,
             label="Hyperliquid BTC Mid Price ($M_t$)")
    ax1.set_ylabel("Mid Price (USD)")
    ax1.set_title("Hyperliquid BTC Perpetual Futures - Full Month Trade-by-Trade Mid Price Trajectory")
    ax1.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#cccccc")

    step      = 50000
    sub_mids  = mids[::step]
    vol       = pd.Series(sub_mids).pct_change().rolling(20).std() * 10000.0
    sub_ticks = ticks_m[::step]
    ax2.plot(sub_ticks, vol, color="#8b0000", lw=1.0, label="20k-Tick Volatility (bps)")
    ax2.set_xlabel("Trade Ticks (Millions)")
    ax2.set_ylabel("Volatility (bps)")
    ax2.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#cccccc")
    fig.subplots_adjust(left=0.09, right=0.97, top=0.91, bottom=0.11, hspace=0.08)
    fig.savefig(GRAPHS_DIR / "full_month_mid_price_path.png")
    plt.close()
    print("  saved full_month_mid_price_path.png", flush=True)

    print("[GRAPH] Figure 2: Empirical Return Distributions ...", flush=True)
    fig, axes = plt.subplots(2, 5, figsize=(15, 6.2))
    axes = axes.flatten()
    for idx, h in enumerate(HORIZONS):
        ax          = axes[idx]
        rets        = df_horizons[h]
        clip_low, clip_high = np.percentile(rets, [0.1, 99.9])
        trimmed     = rets[(rets >= clip_low) & (rets <= clip_high)]
        ax.hist(trimmed, bins=60, density=True, color="#2b5c8f", alpha=0.75, edgecolor="none")
        mu, s       = np.mean(trimmed), np.std(trimmed)
        x_grid      = np.linspace(clip_low, clip_high, 100)
        ax.plot(x_grid, stats.norm.pdf(x_grid, mu, s), color="#c0392b", ls="--", lw=1.2, label="Normal")
        ax.set_yscale("log")
        ax.set_title(f"$h = {h}$")
        ax.set_xlabel("Return (bps)")
        ax.set_ylabel("Log Density")
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle(r"Empirical Mid-Price Return Distributions $P(\Delta M_h)$ Across All 10 Horizons",
                 fontsize=13, y=1.01)
    fig.subplots_adjust(left=0.06, right=0.97, top=0.92, bottom=0.10, hspace=0.52, wspace=0.40)
    fig.savefig(GRAPHS_DIR / "empirical_return_distributions.png")
    plt.close()
    print("  saved empirical_return_distributions.png", flush=True)

    print("[GRAPH] Figure 3: Directional Class Balance ...", flush=True)
    x = np.arange(len(HORIZONS))
    w = 0.26
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(x - w, df_stats["pct_up"],   w, label="Up ($+1$)",   color="#2ca02c", zorder=3)
    ax.bar(x,     df_stats["pct_down"], w, label="Down ($-1$)", color="#d62728", zorder=3)
    ax.bar(x + w, df_stats["pct_flat"], w, label="Flat ($0$)",  color="#7f7f7f", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([str(h) for h in HORIZONS])
    ax.set_xlabel("Prediction Horizon (trades ahead)")
    ax.set_ylabel("Proportion (%)")
    ax.set_title("Directional Movement Class Proportions Across Prediction Horizons")
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc")
    fig.subplots_adjust(left=0.09, right=0.97, top=0.91, bottom=0.13)
    fig.savefig(GRAPHS_DIR / "directional_class_balance.png")
    plt.close()
    print("  saved directional_class_balance.png", flush=True)

    print("[GRAPH] Figure 4: Inter-Trade Arrival Time ...", flush=True)
    dt_ms   = np.diff(timestamps) / 1e6
    dt_ms   = dt_ms[dt_ms > 0]
    clip_dt = dt_ms[dt_ms < np.percentile(dt_ms, 99.5)]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.hist(clip_dt, bins=80, density=True, color="#4b0082", alpha=0.72, edgecolor="white", lw=0.3)
    ax.set_yscale("log")
    ax.set_xlabel(r"Inter-Trade Arrival Time $\Delta t$ (ms)")
    ax.set_ylabel("Log Probability Density")
    ax.set_title(r"Marked Point Process: Inter-Trade Arrival Time Distribution $P(\Delta t)$")
    fig.subplots_adjust(left=0.10, right=0.97, top=0.91, bottom=0.13)
    fig.savefig(GRAPHS_DIR / "inter_trade_arrival_times.png")
    plt.close()
    print("  saved inter_trade_arrival_times.png", flush=True)

    print(f"[GRAPH COMPLETE] All figures written to {GRAPHS_DIR}", flush=True)


if __name__ == "__main__":
    main()
