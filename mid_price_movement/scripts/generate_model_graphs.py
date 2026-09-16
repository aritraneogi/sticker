import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import rcParams
from PIL import Image

HORIZONS    = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500]
TOTAL_TICKS = 11_918_880

MARGIN_L = 0.13
MARGIN_R = 0.04
MARGIN_T = 0.11
MARGIN_B = 0.13

rcParams.update({
    "font.family":       "Times New Roman",
    "font.serif":        ["Times New Roman"],
    "mathtext.fontset":  "stix",
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   9.5,
    "legend.framealpha": 0.92,
    "legend.edgecolor":  "#cccccc",
    "legend.fancybox":   False,
    "figure.dpi":        150,
    "savefig.dpi":       200,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linestyle":    "--",
    "grid.linewidth":    0.6,
    "axes.linewidth":    0.8,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
})


def format_model_label(summary):
    name = summary.get("model", summary.get("model_name", "Unknown"))
    name = name.replace("_", " ")
    parts = []
    for key in ("n_experts", "expert_dim", "batch_size"):
        if key in summary:
            short = {"n_experts": "E", "expert_dim": "N", "batch_size": "BS"}[key]
            parts.append(f"{short}={summary[key]}")
    if "plasticity" in summary:
        parts.append(f"Plasticity={summary['plasticity']}")
    if "lr" in summary:
        parts.append(f"LR={summary['lr']}")
    if "l2_lambda" in summary:
        parts.append(f"L2={summary['l2_lambda']}")
    if "init" in summary:
        parts.append(f"Init={summary['init'].capitalize()}")
    param_str = "  |  ".join(parts) if parts else ""
    return name.strip(), param_str


def place_title_and_params(fig, model_name, param_str, y_title=0.955, y_param=0.910):
    fig.text(0.5, y_title, model_name, ha="center", va="top",
             fontsize=13, fontfamily="Times New Roman", fontweight="normal")
    if param_str:
        fig.text(0.5, y_param, param_str, ha="center", va="top",
                 fontsize=9, fontfamily="Times New Roman", color="#555555",
                 fontstyle="italic")


def graph_cumulative_running_avg_full_da_h1(m, summary, out_path):
    model_name, param_str = format_model_label(summary)
    total_ticks  = summary.get("total_ticks", TOTAL_TICKS)
    last75_start = total_ticks // 4
    full_da      = summary.get("horizons", {}).get("1") or summary.get("horizons", {}).get("h1") or {}
    da_full_val  = (
        full_da.get("da_nz_full") or
        full_da.get("da_nz_online") or
        full_da.get("da_pa_nz_full") or 0.0
    )
    da_last30 = full_da.get("da_nz_last30") or full_da.get("da_pa_nz_last30")

    col = "da_nz_h1" if "da_nz_h1" in m.columns else "da_h1"
    y   = m[col].values * 100.0
    x   = m["tick_idx"].values / 1e6

    ymin = max(40.0, np.floor(np.nanmin(y) * 4) / 4 - 0.5)
    ymax = min(100.0, np.ceil(np.nanmax(y) * 4) / 4 + 0.5)
    if ymax - ymin < 1.0:
        ymin -= 0.5
        ymax += 0.5

    fig, ax = plt.subplots(figsize=(10.0, 4.2))

    ax.plot(x, y, color="#1a7a4a", lw=1.4, label="Cumulative NZ DA h=1")

    last75_x = last75_start / 1e6
    ax.axvline(last75_x, color="#888888", ls="--", lw=0.9,
               label=f"Last 75% Start ({last75_x:.1f}M ticks)")

    label_full = f"Full Stream DA: {da_full_val*100:.2f}%"
    if da_last30 is not None:
        label_full += f"  |  Last 30%: {da_last30*100:.2f}%"
    ax.axhline(da_full_val * 100.0, color="#3a3ab0", ls=":", lw=0.9, label=label_full)

    ax.set_xlabel("Ticks (millions)", labelpad=6)
    ax.set_ylabel("Cumulative NZ DA h=1 (%)", labelpad=6)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(ymin, ymax)

    legend = ax.legend(loc="lower right", handlelength=2.0, handletextpad=0.6,
                       borderpad=0.6, labelspacing=0.4, frameon=True)
    legend.get_frame().set_linewidth(0.6)

    fig.subplots_adjust(left=MARGIN_L, right=1-MARGIN_R-0.02, top=1-MARGIN_T-0.04, bottom=MARGIN_B)
    place_title_and_params(fig, model_name, param_str, y_title=0.96, y_param=0.913)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def graph_cumulative_multi_horizon_full_da_curves(m, summary, out_path):
    model_name, param_str = format_model_label(summary)
    total_ticks  = summary.get("total_ticks", TOTAL_TICKS)
    last75_start = total_ticks // 4

    cols_ordered = [f"da_nz_h{h}" for h in HORIZONS if f"da_nz_h{h}" in m.columns]
    if not cols_ordered:
        cols_ordered = [f"da_h{h}" for h in HORIZONS if f"da_h{h}" in m.columns]
    horizons_present = [int(c.split("h")[-1]) for c in cols_ordered]

    x      = m["tick_idx"].values / 1e6
    all_y  = [m[c].values * 100.0 for c in cols_ordered]
    ymin   = max(40.0, np.floor(min(np.nanmin(y) for y in all_y)) - 1.0)
    ymax   = min(100.0, np.ceil(max(np.nanmax(y) for y in all_y)) + 1.0)

    n_curves = len(cols_ordered)
    cmap     = LinearSegmentedColormap.from_list("hz_grad",
               ["#3b0f70", "#6f1f7b", "#b05462", "#d98f3a", "#5fbf6a"], N=n_curves)
    colors   = [cmap(i / max(n_curves - 1, 1)) for i in range(n_curves)]

    fig, ax = plt.subplots(figsize=(10.5, 4.8))

    for col, h, c in zip(cols_ordered, horizons_present, colors):
        ax.plot(x, m[col].values * 100.0, color=c, lw=1.1, label=f"h={h}")

    last75_x = last75_start / 1e6
    ax.axvline(last75_x, color="#888888", ls="--", lw=0.9, label="Last 75% Start")

    ax.set_xlabel("Ticks (millions)", labelpad=6)
    ax.set_ylabel("Cumulative NZ DA (%)", labelpad=6)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(ymin, ymax)

    legend = ax.legend(loc="lower right", ncol=2,
                       handlelength=1.6, handletextpad=0.5,
                       borderpad=0.55, labelspacing=0.35, columnspacing=1.0, frameon=True)
    legend.get_frame().set_linewidth(0.6)

    fig.subplots_adjust(left=MARGIN_L, right=1-MARGIN_R-0.02, top=1-MARGIN_T-0.04, bottom=MARGIN_B)
    place_title_and_params(fig, model_name, param_str, y_title=0.96, y_param=0.913)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def graph_da_vs_horizon_bar(summary, out_path):
    model_name, param_str = format_model_label(summary)

    horizons_data = summary.get("horizons", {})
    da_vals = []
    for h in HORIZONS:
        hdata = horizons_data.get(str(h)) or horizons_data.get(f"h{h}") or {}
        val = (
            hdata.get("da_nz_last30") or
            hdata.get("da_pa_nz_last30") or
            hdata.get("da_nz_full") or
            hdata.get("da_pa_nz_full") or
            hdata.get("da_nz_online") or 0.5
        )
        da_vals.append(val * 100.0)

    ymin = max(40.0, np.floor(min(da_vals)) - 2.0)
    ymax = np.ceil(max(da_vals)) + 3.0

    cmap = LinearSegmentedColormap.from_list("bar_grad",
           ["#3b0f70", "#6f1f7b", "#b05462", "#d98f3a", "#5fbf6a"], N=len(HORIZONS))
    bar_colors = [cmap(i / (len(HORIZONS) - 1)) for i in range(len(HORIZONS))]

    data_range = ymax - ymin
    fig_h = max(4.0, min(5.5, 3.0 + data_range * 0.06))
    fig, ax = plt.subplots(figsize=(9.0, fig_h))

    x_pos = np.arange(len(HORIZONS))
    bars  = ax.bar(x_pos, da_vals, color=bar_colors, width=0.62, edgecolor="none")

    ax.axhline(50.0, color="#cc3333", ls="--", lw=0.9, label="Random Baseline (50%)")

    label_gap = (ymax - ymin) * 0.012
    for bar, val in zip(bars, da_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + label_gap,
            f"{val:.1f}%",
            ha="center", va="bottom",
            fontsize=8.5, fontfamily="Times New Roman",
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(h) for h in HORIZONS])
    ax.set_xlabel("Prediction Horizon (ticks ahead)", labelpad=6)
    ax.set_ylabel("NZ Directional Accuracy (%)", labelpad=6)
    ax.set_ylim(ymin, ymax + (ymax - ymin) * 0.06)

    legend = ax.legend(loc="upper right", handlelength=1.8, borderpad=0.6, frameon=True)
    legend.get_frame().set_linewidth(0.6)

    fig.subplots_adjust(left=MARGIN_L, right=1-MARGIN_R, top=1-MARGIN_T-0.04, bottom=MARGIN_B)
    place_title_and_params(fig, model_name, param_str, y_title=0.96, y_param=0.913)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def graph_cumulative_running_avg_loss_vs_da_trajectory(m, summary, model_dir, out_path):
    model_name, param_str = format_model_label(summary)

    pred_path = model_dir / "predictions.parquet"
    if pred_path.exists():
        import pyarrow.parquet as pq
        schema_names = pq.read_schema(pred_path).names
        cols = ["mid", "pred", "tick_idx"]
        if "logit" in schema_names:
            cols.append("logit")

        df_pred = pd.read_parquet(pred_path, columns=cols)
        mids = df_pred["mid"].values
        logits = df_pred["logit"].values if "logit" in df_pred.columns else df_pred["pred"].values
        preds = df_pred["pred"].values
        ticks = df_pred["tick_idx"].values

        diff = np.diff(mids, append=mids[-1])
        labels = np.where(diff > 1e-8, 1, np.where(diff < -1e-8, -1, 0))
        mask_nz = labels != 0

        nz_labels = labels[mask_nz]
        nz_logits = logits[mask_nz]
        nz_preds = preds[mask_nz]
        nz_ticks = ticks[mask_nz]

        raw_losses = np.maximum(0.0, 1.0 - nz_labels * nz_logits)
        full_cum_loss = pd.Series(raw_losses).expanding().mean().values
        full_cum_da = pd.Series((nz_preds == nz_labels).astype(float)).expanding().mean().values

        n_pts = len(full_cum_loss)
        idx_sampled = np.linspace(100, n_pts - 1, min(2000, n_pts - 100), dtype=int)
        loss_vals = full_cum_loss[idx_sampled]
        da_vals = full_cum_da[idx_sampled]
        ticks_M = nz_ticks[idx_sampled] / 1e6
    elif "loss" in m.columns and (m["loss"].fillna(0.0) > 0).sum() > 2:
        da_col = "da_nz_h1" if "da_nz_h1" in m.columns else "da_h1"
        raw_losses = m["loss"].ffill().bfill().fillna(0.01).values
        loss_vals = pd.Series(raw_losses).expanding().mean().values
        da_vals = m[da_col].values
        ticks_M = m["tick_idx"].values / 1e6
    else:
        return

    valid = (loss_vals > 0) & (pd.Series(da_vals).notna())
    loss_vals = loss_vals[valid]
    da_vals = da_vals[valid]
    ticks_M = ticks_M[valid]

    if len(loss_vals) < 5:
        return

    ymin = max(0.0, np.floor(da_vals.min() * 20) / 20 - 0.02)
    ymax = min(1.0, np.ceil(da_vals.max() * 20) / 20 + 0.02)

    norm = mcolors.Normalize(vmin=ticks_M.min(), vmax=ticks_M.max())
    cmap = plt.cm.plasma

    fig, ax = plt.subplots(figsize=(8.5, 5.4))

    scatter = ax.scatter(
        loss_vals, da_vals,
        c=ticks_M, cmap=cmap, norm=norm,
        s=14, alpha=0.82, linewidths=0,
    )

    ax.set_xscale("log")
    ax.set_xlabel("Cumulative Running Avg Loss (log scale)", labelpad=6)
    ax.set_ylabel("Cumulative NZ DA h=1", labelpad=6)
    ax.set_ylim(ymin, ymax)

    cbar_ax = fig.add_axes([1 - MARGIN_R - 0.025, MARGIN_B, 0.022, 1 - MARGIN_T - MARGIN_B - 0.04])
    cbar = fig.colorbar(scatter, cax=cbar_ax)
    cbar.set_label("Ticks (millions)", fontsize=10, labelpad=8)
    cbar.ax.tick_params(labelsize=9)
    cbar.outline.set_linewidth(0.5)

    fig.subplots_adjust(left=MARGIN_L, right=1-MARGIN_R-0.068, top=1-MARGIN_T-0.04, bottom=MARGIN_B)
    place_title_and_params(fig, model_name, param_str, y_title=0.965, y_param=0.918)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def generate_last75_barcode(model_dir: Path, out_path: Path, n_bins: int = 300, height: int = 240, stripe_width: int = 8):
    pred_path = model_dir / "predictions.parquet"
    if not pred_path.exists():
        return
    df = pd.read_parquet(pred_path, columns=["pred"])
    n_total = len(df)
    preds_75 = df["pred"].iloc[n_total // 4:].values

    n_valid = len(preds_75)
    bin_size = n_valid // n_bins
    binned_preds = []
    for b in range(n_bins):
        start_i = b * bin_size
        end_i   = (b + 1) * bin_size if b < n_bins - 1 else n_valid
        sub_p = preds_75[start_i:end_i]
        sum_p = sub_p.sum()
        sign_p = 1 if sum_p > 0 else (-1 if sum_p < 0 else 0)
        binned_preds.append(sign_p)
    b_seq = np.array(binned_preds)

    color_up   = np.array([0, 230, 0], dtype=np.uint8)
    color_down = np.array([230, 0, 0], dtype=np.uint8)
    color_flat = np.array([128, 128, 128], dtype=np.uint8)

    width = n_bins * stripe_width
    img_arr = np.zeros((height, width, 3), dtype=np.uint8)
    for i, val in enumerate(b_seq):
        c = color_up if val > 0 else (color_down if val < 0 else color_flat)
        img_arr[:, i * stripe_width:(i + 1) * stripe_width] = c

    img = Image.fromarray(img_arr)
    img.save(out_path)


def generate_for_model(model_dir: Path, eval_root: Path):
    runtime_path = model_dir / "runtime_metrics.parquet"
    summary_path = model_dir / "run_summary.json"
    graphs_dir   = model_dir / "graphs"

    if not runtime_path.exists() or not summary_path.exists():
        print(f"  SKIP (missing data): {model_dir.relative_to(eval_root)}", flush=True)
        return

    graphs_dir.mkdir(exist_ok=True)

    m = pd.read_parquet(runtime_path).sort_values("tick_idx").reset_index(drop=True)
    m["tick_idx"] = m["tick_idx"].astype(float)
    for c in m.columns:
        if c.startswith("da_"):
            m[c] = pd.to_numeric(m[c], errors="coerce")
    if "loss" in m.columns:
        m["loss"] = pd.to_numeric(m["loss"], errors="coerce")

    with open(summary_path) as f:
        summary = json.load(f)

    graph_cumulative_running_avg_full_da_h1(
        m, summary, graphs_dir / "cumulative_running_avg_full_da_h1.png")
    graph_cumulative_multi_horizon_full_da_curves(
        m, summary, graphs_dir / "cumulative_multi_horizon_full_da_curves.png")
    graph_da_vs_horizon_bar(
        summary, graphs_dir / "da_vs_horizon_bar.png")
    graph_cumulative_running_avg_loss_vs_da_trajectory(
        m, summary, model_dir, graphs_dir / "cumulative_running_avg_loss_vs_da_trajectory.png")
    generate_last75_barcode(
        model_dir, graphs_dir / "prediction_sequence_barcode.png")

    print(f"  OK  {model_dir.relative_to(eval_root)}", flush=True)


def main():
    eval_root = Path(__file__).resolve().parent.parent

    tm_files = [
        f for f in eval_root.rglob("trading_metrics.json")
        if "comparison" not in str(f) and "sticker_best_metrics" not in str(f)
    ]

    print(f"Generating graphs for {len(tm_files)} models ...", flush=True)
    for tm in sorted(tm_files):
        generate_for_model(tm.parent, eval_root)

    print(f"\nDone. {len(tm_files)} models processed.", flush=True)


if __name__ == "__main__":
    main()
