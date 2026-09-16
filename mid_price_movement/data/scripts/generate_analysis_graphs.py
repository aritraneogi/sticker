import json
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = SCRIPT_DIR.parent / "analysis"
GRAPHS_DIR = ANALYSIS_DIR / "graphs"
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


def load_dataset_stats():
    path = ANALYSIS_DIR / "dataset_summary_statistics.json"
    with open(path) as f:
        return json.load(f)


def load_acf_mi():
    path = ANALYSIS_DIR / "raw_data_acf_mi_summary.json"
    with open(path) as f:
        return json.load(f)


def load_regimes():
    path = ANALYSIS_DIR / "market_regimes_summary.json"
    with open(path) as f:
        return json.load(f)


def figure_class_balance(stats):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    rcParams.update(RC)

    horizons = [s["horizon"] for s in stats]
    pct_up   = [s["pct_up"]   for s in stats]
    pct_down = [s["pct_down"] for s in stats]
    pct_flat = [s["pct_flat"] for s in stats]

    x = np.arange(len(horizons))
    w = 0.26

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(x - w, pct_up,   w, label="Up ($+1$)",   color="#2ca02c", zorder=3)
    ax.bar(x,     pct_down, w, label="Down ($-1$)", color="#d62728", zorder=3)
    ax.bar(x + w, pct_flat, w, label="Flat ($0$)",  color="#7f7f7f", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([str(h) for h in horizons])
    ax.set_xlabel("Prediction Horizon (trades ahead)")
    ax.set_ylabel("Proportion (%)")
    ax.set_title("Directional Movement Class Proportions Across Prediction Horizons")
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc")
    fig.subplots_adjust(left=0.09, right=0.97, top=0.91, bottom=0.13)
    fig.savefig(GRAPHS_DIR / "03_directional_class_balance.png")
    plt.close()
    print(f"  saved 03_directional_class_balance.png", flush=True)


def figure_acf(acf_data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    rcParams.update(RC)

    df = pd.DataFrame(acf_data)
    lags = df["lag"].values

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(lags, df["acf_order_flow_imbalance"], "o-", color="#1f77b4", lw=1.8, ms=4, label="Order Flow Imbalance (OFI)")
    ax.plot(lags, df["acf_trade_direction"],       "s-", color="#2ca02c", lw=1.8, ms=4, label="Trade Direction ($S_t$)")
    ax.plot(lags, df["acf_returns"],               "^-", color="#d62728", lw=1.8, ms=4, label="Mid-Price Returns ($r_t$)")
    ax.axhline(0.0, color="#999999", ls="--", lw=0.8)
    ax.set_xticks(lags)
    ax.set_xlabel(r"Lag $\tau$ (trades)")
    ax.set_ylabel(r"Autocorrelation $\rho(\tau)$")
    ax.set_title("Hyperliquid BTC Perpetual Futures - Raw Data Autocorrelation Function (ACF)")
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc")
    fig.subplots_adjust(left=0.09, right=0.97, top=0.91, bottom=0.14)
    fig.savefig(GRAPHS_DIR / "07_autocorrelation_acf.png")
    plt.close()
    print(f"  saved 07_autocorrelation_acf.png", flush=True)


def figure_mi(mi_data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    rcParams.update(RC)

    df = pd.DataFrame(mi_data)
    hzs = df["horizon"].values

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(hzs, df["mi_ofi_bits"],    "o-", color="#8c564b", lw=2.0, ms=5, label=r"$I(\mathrm{OFI}_t;\, Y_{t+h})$")
    ax.plot(hzs, df["mi_return_bits"], "s-", color="#e377c2", lw=2.0, ms=5, label=r"$I(r_t;\, Y_{t+h})$")
    ax.set_xscale("log")
    ax.set_xticks(hzs)
    ax.set_xticklabels([str(h) for h in hzs])
    ax.set_xlabel(r"Prediction Horizon $h$ (trades ahead, log scale)")
    ax.set_ylabel(r"Mutual Information $I(X;\, Y)$ (bits)")
    ax.set_title(r"Non-linear Mutual Information Across Prediction Horizons $h$")
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc")
    fig.subplots_adjust(left=0.09, right=0.97, top=0.91, bottom=0.14)
    fig.savefig(GRAPHS_DIR / "08_mutual_information_horizon.png")
    plt.close()
    print(f"  saved 08_mutual_information_horizon.png", flush=True)


def figure_transition_matrix(regimes_data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    rcParams.update(RC)

    tm = np.array(regimes_data["transition_matrix"])
    state_names = ["Low-Vol", "Mod-Trend", "High-Vol"]

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(tm, cmap="Blues", vmin=0, vmax=1, aspect="equal")

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(state_names)
    ax.set_yticklabels(state_names)
    ax.set_xlabel(r"Next State $S_{t+1}$")
    ax.set_ylabel(r"Current State $S_t$")
    ax.set_title(r"Market Regime Markov Transition Matrix $P(S_{t+1}\,|\,S_t)$")
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)

    for i in range(3):
        for j in range(3):
            val = tm[i, j]
            color = "white" if val >= 0.6 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=color, fontsize=10, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    fig.subplots_adjust(left=0.14, right=0.92, top=0.90, bottom=0.13)
    fig.savefig(GRAPHS_DIR / "06_regime_transition_matrix.png")
    plt.close()
    print(f"  saved 06_regime_transition_matrix.png", flush=True)


def main():
    print("Loading cached analysis data ...", flush=True)
    stats   = load_dataset_stats()
    acf_mi  = load_acf_mi()
    regimes = load_regimes()

    print("Generating figures ...", flush=True)
    figure_class_balance(stats)
    figure_acf(acf_mi["acf_lags_1_to_20"])
    figure_mi(acf_mi["mutual_information_horizons"])
    figure_transition_matrix(regimes)

    print(f"\nDone. Graphs written to: {GRAPHS_DIR}", flush=True)


if __name__ == "__main__":
    main()
