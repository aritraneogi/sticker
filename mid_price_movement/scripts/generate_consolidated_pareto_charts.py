import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.interpolate import PchipInterpolator

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
    "savefig.dpi":       250,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.22,
    "grid.linestyle":    "--",
    "grid.linewidth":    0.55,
    "axes.linewidth":    0.85,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
})

NEURAL_BASELINES = {
    "TLOB":    "Transformer",
    "Mamba":   "SSM",
    "DeepLOB": "CNN-LSTM",
}

NEURAL_COLORS = {
    "TLOB":    "#ea7500",
    "Mamba":   "#e51c23",
    "DeepLOB": "#7b1fa2",
}

CLASSICAL_COLOR  = "#555555"
STICKER_FRONTIER = "#1869e6"


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
            baselines[canonical] = (lat_ms, da_pct)

    return sticker_pts, baselines


def extract_sticker_edge(sticker_pts):
    sorted_pts = sorted(sticker_pts, key=lambda p: p[0])
    peak_pt = max(sorted_pts, key=lambda p: p[1])

    frontier = []
    best_da = -np.inf
    for p in sorted_pts:
        if p[0] <= peak_pt[0]:
            if p[1] > best_da:
                best_da = p[1]
                frontier.append(p)

    upper_rest = [p for p in sorted_pts if p[0] > peak_pt[0]]
    if upper_rest:
        max_lat = max(p[0] for p in upper_rest)
        bins = np.logspace(np.log10(peak_pt[0]), np.log10(max_lat), 6)
        tail_pts = []
        for i in range(len(bins) - 1):
            sub = [p for p in upper_rest if bins[i] < p[0] <= bins[i + 1]]
            if sub:
                tail_pts.append(max(sub, key=lambda p: p[1]))
        all_edge = frontier + tail_pts
    else:
        all_edge = frontier

    unique_edge = []
    last_lat = -1
    for p in all_edge:
        if p[0] > last_lat:
            unique_edge.append(p)
            last_lat = p[0]

    return unique_edge


def extract_sticker_edge_neural(sticker_pts):
    sorted_pts = sorted(sticker_pts, key=lambda p: p[0])
    peak_pt = max(sorted_pts, key=lambda p: p[1])

    frontier = []
    best_da = -np.inf
    for p in sorted_pts:
        if p[0] <= peak_pt[0]:
            if p[1] > best_da:
                best_da = p[1]
                frontier.append(p)

    upper_rest = [p for p in sorted_pts if p[0] > peak_pt[0]]
    if upper_rest:
        rev_tail = []
        max_da = -np.inf
        for p in reversed(sorted(upper_rest, key=lambda x: x[0])):
            if p[1] >= max_da:
                rev_tail.append(p)
                max_da = p[1]
        tail_pts = [p for p in reversed(rev_tail) if p[0] > frontier[-1][0]]
        all_edge = frontier + tail_pts
    else:
        all_edge = frontier

    unique_edge = []
    last_lat = -1
    for p in all_edge:
        if p[0] > last_lat:
            unique_edge.append(p)
            last_lat = p[0]

    return unique_edge


def smooth_edge_curve(edge_pts, n=500):
    if len(edge_pts) < 2:
        return None, None
    lats = np.array([p[0] for p in edge_pts])
    das  = np.array([p[1] for p in edge_pts])
    log_lats = np.log10(lats)
    interp   = PchipInterpolator(log_lats, das)
    fine_log = np.linspace(log_lats[0], log_lats[-1], n)
    return 10 ** fine_log, interp(fine_log)


def place_title(fig, title_text, y=0.960):
    fig.text(0.5, y, title_text, ha="center", va="top",
             fontsize=13, fontfamily="Times New Roman", fontweight="normal")


def make_chart(
    sticker_pts,
    comparison_models,
    out_path,
    title_text,
    neural_mode=False,
):
    fig, ax = plt.subplots(figsize=(10.2, 5.6))

    if neural_mode:
        edge_pts = extract_sticker_edge_neural(sticker_pts)
    else:
        edge_pts = extract_sticker_edge(sticker_pts)
    curve_x, curve_y = smooth_edge_curve(edge_pts)

    if curve_x is not None:
        ax.plot(curve_x, curve_y, color=STICKER_FRONTIER, lw=2.4, zorder=4,
                solid_capstyle="round")

    placed_labels = []

    for model_name, (lat_ms, da_pct) in comparison_models.items():
        if neural_mode and model_name in NEURAL_BASELINES:
            color  = NEURAL_COLORS[model_name]
            arch   = NEURAL_BASELINES[model_name]
            label  = f"{model_name} ({arch})"
            marker = "s"
            size   = 90
        else:
            color  = CLASSICAL_COLOR
            label  = model_name
            marker = "o"
            size   = 65

        ax.scatter(lat_ms, da_pct, color=color, s=size, marker=marker,
                   zorder=5, linewidths=0.6, edgecolors="#ffffff")

        placed_labels.append((lat_ms, da_pct, label, color))

    ax.set_xscale("log")

    all_lats = [p[0] for p in edge_pts] + [v[0] for v in comparison_models.values()]
    all_das  = [p[1] for p in edge_pts] + [v[1] for v in comparison_models.values()]
    xmin = 10 ** (np.floor(np.log10(min(all_lats))) - 0.3)
    xmax = 10 ** (np.ceil(np.log10(max(all_lats)))  + 0.4)
    ymin = max(40.0, np.floor(min(all_das)) - 4.0)
    ymax = min(100.0, np.ceil(max(all_das)) + 3.0)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    for lat_ms, da_pct, label, color in placed_labels:
        log_x   = np.log10(lat_ms)
        log_xmn = np.log10(xmin)
        log_xmx = np.log10(xmax)
        frac_x  = (log_x - log_xmn) / (log_xmx - log_xmn)
        frac_y  = (da_pct - ymin) / (ymax - ymin)
        if "ARIMA" in label:
            dy = -4.40
        elif "Kalman" in label:
            dy = -3.25
        else:
            dy = 2.2 if frac_y < 0.65 else -2.8
        ha = "center"
        ax.text(lat_ms, da_pct + dy, label,
                ha=ha, va="bottom" if dy > 0 else "top",
                fontsize=8.5, fontfamily="Times New Roman",
                color=color, zorder=6)

    if curve_x is not None:
        peak_idx = int(np.argmax(curve_y))
        peak_x = float(curve_x[peak_idx])
        peak_y = float(curve_y[peak_idx])
        ax.text(peak_x, peak_y + 0.85, "Sticker",
                ha="center", va="bottom",
                fontsize=10.5, fontweight="bold", fontfamily="Times New Roman",
                color=STICKER_FRONTIER, zorder=6)

    ax.set_xlabel("Per-Tick Inference Latency (ms, log scale)", labelpad=6)
    ax.set_ylabel("NZ Directional Accuracy h=1 (%)", labelpad=6)

    fig.subplots_adjust(left=0.09, right=0.96, top=0.90, bottom=0.11)
    place_title(fig, title_text, y=0.960)

    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path.name}", flush=True)


def is_batch_size_1(name):
    lower = name.lower()
    if lower == "sticker":
        return False
    return "_bs" not in lower


def main():
    sticker_pts, baselines = load_data()
    print(f"Loaded {len(sticker_pts)} sticker configs, {len(baselines)} baselines", flush=True)

    make_chart(
        sticker_pts=sticker_pts,
        comparison_models=baselines,
        out_path=GRAPHS_DIR / "pareto_sticker_vs_all_baselines.png",
        title_text="Pareto Frontier: Sticker vs All Baselines",
        neural_mode=False,
    )

    neural_names = set(NEURAL_BASELINES.keys())
    neural_baselines = {k: v for k, v in baselines.items() if k in neural_names}
    sticker_pts_neural = [p for p in sticker_pts if not is_batch_size_1(p[2])]

    make_chart(
        sticker_pts=sticker_pts_neural,
        comparison_models=neural_baselines,
        out_path=GRAPHS_DIR / "pareto_sticker_vs_neural_network_baselines.png",
        title_text="Pareto Frontier: Sticker vs Neural Network Baselines",
        neural_mode=True,
    )

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
