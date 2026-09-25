import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
GRAPHS_DIR = BASE_DIR / "graphs"
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

rcParams.update({
    "font.family":       "Times New Roman",
    "font.serif":        ["Times New Roman"],
    "mathtext.fontset":  "stix",
    "axes.labelsize":    11,
    "xtick.labelsize":   9.5,
    "ytick.labelsize":   9.5,
    "legend.fontsize":   9.5,
    "legend.framealpha": 0.93,
    "legend.edgecolor":  "#cccccc",
    "legend.fancybox":   False,
    "figure.dpi":        150,
    "savefig.dpi":       350,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.linestyle":    "--",
    "grid.linewidth":    0.55,
    "axes.linewidth":    0.85,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
})

NEURAL_ARCHS = {
    "TLOB":    "Transformer",
    "Mamba":   "SSM",
    "DeepLOB": "CNN-LSTM",
}


def load_real_data():
    baselines = {}
    for p in sorted((BASE_DIR / "baselines").glob("*/run_summary.json")):
        with open(p) as f:
            d = json.load(f)
        model = d.get("model", p.parent.name)
        tps = float(d["tps"])
        h1 = d["horizons"].get("1") or d["horizons"].get("h1") or {}
        da = float(h1.get("da_nz_online") or h1.get("da_nz_full") or h1.get("da_pa_nz_full")) * 100.0

        canonical = model.strip()
        if "logistic" in canonical.lower():
            canonical = "Logistic Regression"
        elif "kalman" in canonical.lower():
            canonical = "Kalman Filter"
        elif "arima" in canonical.lower():
            canonical = "ARIMA"
        elif "ofi" in canonical.lower():
            canonical = "OFI LinReg"
        elif "garch" in canonical.lower():
            canonical = "GARCH(1,1)"
        elif "lightgbm" in canonical.lower() or canonical.lower() == "lgb":
            canonical = "LightGBM"
        elif "deeplob" in canonical.lower():
            canonical = "DeepLOB"
        elif "mamba" in canonical.lower():
            canonical = "Mamba"
        elif "tlob" in canonical.lower():
            canonical = "TLOB"
        baselines[canonical] = (tps, da)

    sticker_pts = []
    for p in (BASE_DIR / "sticker").glob("**/run_summary.json"):
        with open(p) as f:
            d = json.load(f)
        if "tps" in d and "horizons" in d:
            h1 = d["horizons"].get("1") or d["horizons"].get("h1") or {}
            da_val = h1.get("da_nz_online") or h1.get("da_nz_full") or h1.get("da_pa_nz_full")
            if da_val is not None:
                model_name = d.get("model", p.parent.name)
                sticker_pts.append((model_name, float(d["tps"]), float(da_val) * 100.0))

    return sticker_pts, baselines


def build_smoothened_tps_curve(sticker_pts, n=600, k=1.8):
    p_max_tps = max(sticker_pts, key=lambda x: x[1])
    p_max_da = max(sticker_pts, key=lambda x: x[2])

    tps_high = p_max_tps[1]
    da_at_high_tps = p_max_tps[2]

    tps_low = p_max_da[1]
    da_at_max_da = p_max_da[2]

    log_min = np.log10(tps_low)
    log_max = np.log10(tps_high)

    fine_log = np.linspace(log_min, log_max, n)
    t = (log_max - fine_log) / (log_max - log_min)
    cy = da_at_high_tps + (da_at_max_da - da_at_high_tps) * (1.0 - np.exp(-k * t)) / (1.0 - np.exp(-k))
    cx = 10 ** fine_log

    return cx, cy, p_max_da, p_max_tps


def render_tps_chart(sticker_pts, comparison_models, out_path, title_text, neural_mode=False, is_dark=False):
    cx, cy, p_max_da, p_max_tps = build_smoothened_tps_curve(sticker_pts)

    text_color = "#ffffff" if is_dark else "#000000"
    axis_color = "#cccccc" if is_dark else "#000000"
    grid_color = "#333333" if is_dark else "#b0b0b0"
    grid_alpha = 0.25 if is_dark else 0.22
    sticker_label_color = "#38bdf8" if is_dark else "#1869e6"
    color_map_colors = ["#0284c7", "#38bdf8", "#bae6fd"] if is_dark else ["#1869e6", "#00c2ff"]
    classical_color = "#b0b0b0" if is_dark else "#555555"

    fig, ax = plt.subplots(figsize=(10.2, 5.6), facecolor="#000000" if is_dark else "#ffffff")
    ax.set_facecolor("#000000" if is_dark else "#ffffff")

    ax.grid(True, linestyle="--", linewidth=0.55, color=grid_color, alpha=grid_alpha)
    ax.spines["bottom"].set_color(axis_color)
    ax.spines["left"].set_color(axis_color)
    ax.tick_params(axis="x", colors=axis_color)
    ax.tick_params(axis="y", colors=axis_color)

    pts = np.array([cx, cy]).T.reshape(-1, 1, 2)
    segments = np.concatenate([pts[:-1], pts[1:]], axis=1)
    cmap = LinearSegmentedColormap.from_list("sticker_gradient", color_map_colors)
    norm = plt.Normalize(cy.min(), cy.max())
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=2.6,
                        capstyle="round", joinstyle="round", zorder=4)
    lc.set_array(cy)
    ax.add_collection(lc)

    ax.text(p_max_da[1], p_max_da[2] + 1.60, "Sticker",
            ha="center", va="bottom",
            fontsize=10.5, fontweight="bold", fontfamily="Times New Roman",
            color=sticker_label_color, zorder=6)

    if neural_mode:
        for model_name, (tps, da) in comparison_models.items():
            if model_name in NEURAL_ARCHS:
                arch = NEURAL_ARCHS[model_name]
                label = f"{arch} ({model_name})"
                marker = "o"
                size = 65

                if is_dark:
                    ax.scatter(tps, da, color=classical_color, s=size, marker=marker,
                               zorder=5, linewidths=0, edgecolors="none")
                else:
                    ax.scatter(tps, da, color=classical_color, s=size, marker=marker,
                               zorder=5, linewidths=0.6, edgecolors="#ffffff")

                dy = 2.2 if da < 85.0 else -2.8
                ax.text(tps, da + dy, label,
                        ha="center", va="bottom" if dy > 0 else "top",
                        fontsize=8.5, fontfamily="Times New Roman",
                        color=classical_color, zorder=6)
    else:
        label_positions = {
            "DeepLOB": (comparison_models["DeepLOB"][0], comparison_models["DeepLOB"][1] - 2.8, "center", "top"),
            "Mamba": (comparison_models["Mamba"][0], comparison_models["Mamba"][1] - 2.8, "center", "top"),
            "TLOB": (comparison_models["TLOB"][0], comparison_models["TLOB"][1] - 2.8, "center", "top"),
            "LightGBM": (comparison_models["LightGBM"][0], comparison_models["LightGBM"][1] - 2.8, "center", "top"),
            "Logistic Regression": (comparison_models["Logistic Regression"][0], comparison_models["Logistic Regression"][1] + 2.2, "center", "bottom"),
            "Kalman Filter": (comparison_models["Kalman Filter"][0], comparison_models["Kalman Filter"][1] - 2.0, "center", "top"),
            "ARIMA": (1950.0, comparison_models["ARIMA"][1] - 3.1, "right", "top"),
            "OFI LinReg": (comparison_models["OFI LinReg"][0], comparison_models["OFI LinReg"][1] - 2.6, "center", "top"),
            "GARCH(1,1)": (comparison_models["GARCH(1,1)"][0], comparison_models["GARCH(1,1)"][1] - 2.8, "center", "top"),
        }

        for model_name, (tps, da) in comparison_models.items():
            marker = "o"
            size = 65

            if is_dark:
                ax.scatter(tps, da, color=classical_color, s=size, marker=marker,
                           zorder=5, linewidths=0, edgecolors="none")
            else:
                ax.scatter(tps, da, color=classical_color, s=size, marker=marker,
                           zorder=5, linewidths=0.6, edgecolors="#ffffff")

            if model_name in label_positions:
                lx, ly, ha, va = label_positions[model_name]
            else:
                lx, ly, ha, va = tps, da + 2.2, "center", "bottom"

            ax.text(lx, ly, model_name,
                    ha=ha, va=va,
                    fontsize=8.5, fontfamily="Times New Roman",
                    color=classical_color, zorder=6)

    ax.set_xscale("log")
    ax.set_xlim(100.0, 3500.0)
    ax.set_ylim(45.0, 99.0)

    ax.set_xlabel("Per-Tick Throughput (ticks/s, log scale) (Better \u2192)", labelpad=6, color=text_color)
    ax.set_ylabel("NZ Directional Accuracy h=1 (%)", labelpad=6, color=text_color)

    fig.text(0.5, 0.960, title_text,
             ha="center", va="top", fontsize=13, fontfamily="Times New Roman",
             fontweight="normal", color=text_color)

    fig.subplots_adjust(left=0.09, right=0.96, top=0.90, bottom=0.11)

    if is_dark:
        fig.savefig(out_path, dpi=350, bbox_inches="tight", facecolor="#000000")
    else:
        fig.savefig(out_path, dpi=350, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path.name}", flush=True)


def main():
    sticker_pts, baselines = load_real_data()
    print(f"Loaded real summaries: {len(sticker_pts)} sticker models, {len(baselines)} baselines", flush=True)

    neural_baselines = {k: v for k, v in baselines.items() if k in NEURAL_ARCHS}

    chart1_white = GRAPHS_DIR / "pareto_sticker_all_baselines_accuracy_tps.png"
    render_tps_chart(
        sticker_pts=sticker_pts,
        comparison_models=baselines,
        out_path=chart1_white,
        title_text="All Baselines in Mid-Price Movement (Continual Learning)",
        neural_mode=False,
        is_dark=False,
    )

    chart1_black = GRAPHS_DIR / "pareto_sticker_all_baselines_accuracy_tps_black.png"
    render_tps_chart(
        sticker_pts=sticker_pts,
        comparison_models=baselines,
        out_path=chart1_black,
        title_text="All Baselines in Mid-Price Movement (Continual Learning)",
        neural_mode=False,
        is_dark=True,
    )

    chart2_white = GRAPHS_DIR / "pareto_sticker_neural_networks_accuracy_tps.png"
    render_tps_chart(
        sticker_pts=sticker_pts,
        comparison_models=neural_baselines,
        out_path=chart2_white,
        title_text="Neural Networks in Mid-Price Movement (Continual Learning)",
        neural_mode=True,
        is_dark=False,
    )

    chart2_black = GRAPHS_DIR / "pareto_sticker_neural_networks_accuracy_tps_black.png"
    render_tps_chart(
        sticker_pts=sticker_pts,
        comparison_models=neural_baselines,
        out_path=chart2_black,
        title_text="Neural Networks in Mid-Price Movement (Continual Learning)",
        neural_mode=True,
        is_dark=True,
    )

    print("All 4 TPS charts successfully generated at 350 DPI.", flush=True)


if __name__ == "__main__":
    main()
