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
GRAPHS_DIR = SCRIPT_DIR.parent / "graphs"
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_PATH = SCRIPT_DIR.parent / "comparison" / "core_model_performance" / "test_metrics.json"

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


def load_data():
    with open(METRICS_PATH) as f:
        entries = json.load(f)

    sticker_pts = []
    baselines   = {}

    for e in entries:
        name = e.get("model_name", "")
        lat  = e.get("latency_p50_ms")
        da   = e.get("da_nz_h1")
        if lat is None or da is None:
            continue
        lat_ms = float(lat)
        da_pct = float(da) * 100.0
        if name.lower().startswith("sticker"):
            sticker_pts.append((lat_ms, da_pct, name))
        else:
            canonical = name.strip()
            if "deeplob" in canonical.lower():
                canonical = "DeepLOB"
            elif "mamba" in canonical.lower():
                canonical = "Mamba"
            elif "tlob" in canonical.lower():
                canonical = "TLOB"
            if canonical in NEURAL_ARCHS:
                baselines[canonical] = (lat_ms, da_pct)

    return sticker_pts, baselines


def is_batch_size_1(name):
    lower = name.lower()
    if lower == "sticker":
        return False
    return "_bs" not in lower


def extract_sticker_edge_to_peak(sticker_pts):
    sorted_pts = sorted(sticker_pts, key=lambda p: p[0])
    peak_pt = max(sorted_pts, key=lambda p: p[1])

    frontier = []
    best_da = -np.inf
    for p in sorted_pts:
        if p[0] <= peak_pt[0]:
            if p[1] > best_da:
                best_da = p[1]
                frontier.append(p)

    return frontier


def smooth_edge_curve(frontier_pts, n=600, k=2.6):
    if len(frontier_pts) < 2:
        return None, None
    lats = np.array([frontier_pts[0][0], frontier_pts[-1][0]])
    das  = np.array([frontier_pts[0][1], frontier_pts[-1][1]])
    log_lats = np.log10(lats)
    fine_log = np.linspace(log_lats[0], log_lats[-1], n)
    dx = log_lats[-1] - log_lats[0]
    dy = das[-1] - das[0]
    t = (fine_log - log_lats[0]) / dx
    cy = das[0] + dy * (1.0 - np.exp(-k * t)) / (1.0 - np.exp(-k))
    return 10 ** fine_log, cy


def render_chart(sticker_pts, comparison_models, out_path, is_dark=False):
    edge_pts = extract_sticker_edge_to_peak(sticker_pts)
    curve_x, curve_y = smooth_edge_curve(edge_pts)

    text_color = "#ffffff" if is_dark else "#000000"
    axis_color = "#cccccc" if is_dark else "#000000"
    grid_color = "#333333" if is_dark else "#b0b0b0"
    grid_alpha = 0.25 if is_dark else 0.22
    sticker_label_color = "#38bdf8" if is_dark else "#1869e6"
    color_map_colors = ["#0284c7", "#38bdf8", "#bae6fd"] if is_dark else ["#1869e6", "#00c2ff"]
    baseline_color = "#b0b0b0" if is_dark else "#555555"

    fig, ax = plt.subplots(figsize=(10.2, 5.6), facecolor="#000000" if is_dark else "#ffffff")
    ax.set_facecolor("#000000" if is_dark else "#ffffff")

    ax.grid(True, linestyle="--", linewidth=0.55, color=grid_color, alpha=grid_alpha)
    ax.spines["bottom"].set_color(axis_color)
    ax.spines["left"].set_color(axis_color)
    ax.tick_params(axis="x", colors=axis_color)
    ax.tick_params(axis="y", colors=axis_color)

    if curve_x is not None:
        pts = np.array([curve_x, curve_y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([pts[:-1], pts[1:]], axis=1)
        cmap = LinearSegmentedColormap.from_list("sticker_gradient", color_map_colors)
        norm = plt.Normalize(curve_y.min(), curve_y.max())
        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=2.6,
                            capstyle="round", joinstyle="round", zorder=4)
        lc.set_array(curve_y)
        ax.add_collection(lc)

    for model_name, (lat_ms, da_pct) in comparison_models.items():
        arch   = NEURAL_ARCHS[model_name]
        label  = f"{arch} ({model_name})"
        marker = "o"
        size   = 65

        if is_dark:
            ax.scatter(lat_ms, da_pct, color=baseline_color, s=size, marker=marker,
                       zorder=5, linewidths=0, edgecolors="none")
        else:
            ax.scatter(lat_ms, da_pct, color=baseline_color, s=size, marker=marker,
                       zorder=5, linewidths=0.6, edgecolors="#ffffff")

        dy = 2.2 if da_pct < 65 else -2.8
        ax.text(lat_ms, da_pct + dy, label,
                ha="center", va="bottom" if dy > 0 else "top",
                fontsize=8.5, fontfamily="Times New Roman",
                color=baseline_color, zorder=6)

    ax.set_xscale("log")
    ax.set_xlim(10 ** -2.3, 10 ** 3.4)
    ax.set_ylim(45.0, 98.0)

    if curve_x is not None:
        mid_idx = len(curve_x) // 2
        label_x = float(curve_x[-1])
        label_y = float(curve_y[mid_idx])
        ax.text(label_x, label_y + 2.20, "Sticker",
                ha="center", va="bottom",
                fontsize=10.5, fontweight="bold", fontfamily="Times New Roman",
                color=sticker_label_color, zorder=6)

    ax.set_xlabel("Per-Tick Inference Latency (ms, log scale) (\u2190 Better)", labelpad=6, color=text_color)
    ax.set_ylabel("NZ Directional Accuracy h=1 (%)", labelpad=6, color=text_color)

    fig.text(0.5, 0.960, "Neural Networks in Mid-Price Movement (Continual Learning)",
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
    sticker_pts, neural_baselines = load_data()
    sticker_pts_filtered = [p for p in sticker_pts if not is_batch_size_1(p[2])]
    print(f"Loaded {len(sticker_pts_filtered)} sticker configs, {len(neural_baselines)} neural baselines", flush=True)

    white_path = GRAPHS_DIR / "pareto_sticker_neural_networks_accuracy_latency.png"
    render_chart(
        sticker_pts=sticker_pts_filtered,
        comparison_models=neural_baselines,
        out_path=white_path,
        is_dark=False,
    )

    black_path = GRAPHS_DIR / "pareto_sticker_neural_networks_accuracy_latency_black.png"
    render_chart(
        sticker_pts=sticker_pts_filtered,
        comparison_models=neural_baselines,
        out_path=black_path,
        is_dark=True,
    )

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
