import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR   = SCRIPT_DIR.parent
PRED_PATH  = ROOT_DIR.parent / "results" / "predictions.parquet"
OUT_DIR    = ROOT_DIR / "analysis"
GRAPHS_DIR = OUT_DIR / "graphs"

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
    "figure.dpi":        200,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.28,
    "grid.linestyle":    "--",
}


def compute_acf(series, max_lags=20):
    mean = np.mean(series)
    var = np.var(series)
    if var == 0:
        return [1.0] + [0.0] * max_lags
    sub = series - mean
    n = len(series)
    acf_vals = [1.0]
    for lag in range(1, max_lags + 1):
        cov = np.sum(sub[:n - lag] * sub[lag:]) / n
        acf_vals.append(float(cov / var))
    return acf_vals


def compute_binned_mi(x, y, n_bins=30):
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_c, y_c = x[mask], y[mask]
    if len(x_c) == 0:
        return 0.0
    c_xy, _, _ = np.histogram2d(x_c, y_c, bins=n_bins)
    p_xy = c_xy / np.sum(c_xy)
    p_x = np.sum(p_xy, axis=1)
    p_y = np.sum(p_xy, axis=0)
    nz = p_xy > 0
    mi = np.sum(p_xy[nz] * np.log2(p_xy[nz] / (p_x[:, None] * p_y[None, :])[nz]))
    return float(np.max([0.0, mi]))


def main():
    df = pd.read_parquet(PRED_PATH).sort_values("tick_idx").reset_index(drop=True)
    df["mid"]      = df["mid"].astype(float)
    df["sz"]       = df["sz"].astype(float)
    df["side_num"] = np.where(df["side"] == "B", 1, -1)
    df["ret"]      = df["mid"].diff().fillna(0.0)
    df["ofi"]      = df["side_num"] * df["sz"]

    sub_sample = df.iloc[:1000000].copy()
    acf_ret   = compute_acf(sub_sample["ret"].values,      max_lags=20)
    acf_ofi   = compute_acf(sub_sample["ofi"].values,      max_lags=20)
    acf_trade = compute_acf(sub_sample["side_num"].values, max_lags=20)

    acf_summary = []
    for lag in range(1, 21):
        acf_summary.append({
            "lag": lag,
            "acf_returns":               round(acf_ret[lag],   6),
            "acf_order_flow_imbalance":  round(acf_ofi[lag],   6),
            "acf_trade_direction":       round(acf_trade[lag],  6),
        })

    mi_sample = df.iloc[:500000].copy()
    mid_vals  = mi_sample["mid"].values

    mi_summary = []
    for h in HORIZONS:
        target_diff = pd.Series(mid_vals).shift(-h) - pd.Series(mid_vals)
        y_label = np.where(target_diff > 1e-8, 1, np.where(target_diff < -1e-8, -1, 0)).astype(float)
        valid_m = ~np.isnan(y_label)
        mi_ofi = compute_binned_mi(mi_sample["ofi"].values[valid_m], y_label[valid_m], n_bins=30)
        mi_ret = compute_binned_mi(mi_sample["ret"].values[valid_m], y_label[valid_m], n_bins=30)
        mi_summary.append({"horizon": h, "mi_ofi_bits": round(mi_ofi, 6), "mi_return_bits": round(mi_ret, 6)})

    summary_data = {"acf_lags_1_to_20": acf_summary, "mutual_information_horizons": mi_summary}
    with open(OUT_DIR / "raw_data_acf_mi_summary.json", "w") as f:
        json.dump(summary_data, f, indent=2)

    pd.DataFrame(acf_summary).to_csv(OUT_DIR / "raw_data_acf_summary.csv", index=False)
    pd.DataFrame(mi_summary).to_csv(OUT_DIR / "raw_data_mi_summary.csv", index=False)

    generate_figures(pd.DataFrame(acf_summary), pd.DataFrame(mi_summary))


def generate_figures(df_acf, df_mi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    rcParams.update(RC)

    lags = df_acf["lag"].values
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(lags, df_acf["acf_order_flow_imbalance"], "o-", color="#1f77b4", lw=1.8, ms=4, label="Order Flow Imbalance (OFI)")
    ax.plot(lags, df_acf["acf_trade_direction"],       "s-", color="#2ca02c", lw=1.8, ms=4, label="Trade Direction ($S_t$)")
    ax.plot(lags, df_acf["acf_returns"],               "^-", color="#d62728", lw=1.8, ms=4, label="Mid-Price Returns ($r_t$)")
    ax.axhline(0.0, color="#999999", ls="--", lw=0.8)
    ax.set_xticks(lags)
    ax.set_xlabel(r"Lag $\tau$ (trades)")
    ax.set_ylabel(r"Autocorrelation $\rho(\tau)$")
    ax.set_title("Hyperliquid BTC Perpetual Futures - Raw Data Autocorrelation Function (ACF)")
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc")
    fig.subplots_adjust(left=0.09, right=0.97, top=0.91, bottom=0.14)
    fig.savefig(GRAPHS_DIR / "autocorrelation_acf.png")
    plt.close()

    hzs = df_mi["horizon"].values
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(hzs, df_mi["mi_ofi_bits"],    "o-", color="#8c564b", lw=2.0, ms=5, label=r"$I(\mathrm{OFI}_t;\, Y_{t+h})$")
    ax.plot(hzs, df_mi["mi_return_bits"], "s-", color="#e377c2", lw=2.0, ms=5, label=r"$I(r_t;\, Y_{t+h})$")
    ax.set_xscale("log")
    ax.set_xticks(hzs)
    ax.set_xticklabels([str(h) for h in hzs])
    ax.set_xlabel(r"Prediction Horizon $h$ (trades ahead, log scale)")
    ax.set_ylabel(r"Mutual Information $I(X;\, Y)$ (bits)")
    ax.set_title(r"Non-linear Mutual Information Across Prediction Horizons $h$")
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc")
    fig.subplots_adjust(left=0.09, right=0.97, top=0.91, bottom=0.14)
    fig.savefig(GRAPHS_DIR / "mutual_information_horizon.png")
    plt.close()

    print(f"[GRAPH COMPLETE] Saved figures to {GRAPHS_DIR}", flush=True)


if __name__ == "__main__":
    main()
