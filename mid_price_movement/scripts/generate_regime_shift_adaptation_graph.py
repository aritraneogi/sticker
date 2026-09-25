#!/usr/bin/env python3
"""
Generate Event-Study Regime Shift Adaptation Trajectory Graphs.
Two figures:
  1. All Market Regime Transitions  (N = 7,325 shifts)
  2. Extreme Volatility Shock Transitions (R2, N = 1,474 shifts)

Design Specifications (STRICT):
  - No in-graph callout boxes. All findings in the caption.
  - Pure white background. No background shading.
  - One-lined model legend (ncol=6) elevated above the axes, inside the figure.
  - Equal-gap symmetry (verified to within 0.001 inch):
      GAP_Y = 0.050":  left-image-edge ↔ y-label ↔ y-tick-labels
      GAP_X = 0.085":  bottom-image-edge ↔ x-label ↔ x-tick-labels
                        rightmost-x-tick-right ↔ right-image-edge
                        top-spine ↔ legend-bottom ↔ legend-top ↔ top-image-edge
  - 350 DPI, Times New Roman + STIX math fonts.
"""

import os
import time
import warnings
import shutil
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path as MPath
from matplotlib import rcParams
from PIL import Image

warnings.filterwarnings("ignore")
os.environ["MPLCONFIGDIR"] = "/tmp"

class CustomLegendSquare:
    """Square boxstyle with adjusted left padding to equalize visual margins between Sticker line and LightGBM text."""
    def __init__(self, pad_y=0.0, pad_left=0.0, pad_right=0.0):
        self.pad_y = pad_y
        self.pad_left = pad_left
        self.pad_right = pad_right

    def __call__(self, x0, y0, width, height, mutation_size):
        py = mutation_size * self.pad_y
        pl = mutation_size * self.pad_left
        pr = mutation_size * self.pad_right
        x_left = x0 - pl
        x_right = x0 + width + pr
        y_bot = y0 - py
        y_top = y0 + height + py
        return MPath._create_closed([(x_left, y_bot), (x_right, y_bot), (x_right, y_top), (x_left, y_top)])

mpatches.BoxStyle._style_list["custom_legend_square"] = CustomLegendSquare

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR   = SCRIPT_DIR.parent
GRAPHS_DIR = BASE_DIR / "graphs"
MANUSCRIPT_FIGS_DIR = BASE_DIR / "manuscript" / "figures"
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
MANUSCRIPT_FIGS_DIR.mkdir(parents=True, exist_ok=True)

MASTER_PATH = BASE_DIR / "data" / "extracted" / "btc_trades_master.parquet"
CACHE_PATH  = BASE_DIR.parent / "scratch" / "results_cache.pkl"

MODELS = [
    ("Sticker",             BASE_DIR / "sticker" / "variations" / "sticker_e56_n56_bs128_default" / "predictions.parquet", "#1869e6", "-",   2.3, 6),
    ("Logistic Regression", BASE_DIR / "baselines" / "logreg"  / "predictions.parquet",                                   "#059669", "--",  1.5, 5),
    ("TLOB",                BASE_DIR / "baselines" / "tlob"    / "predictions.parquet",                                   "#7c3aed", "-",   1.5, 4),
    ("Kalman Filter",       BASE_DIR / "baselines" / "kalman"  / "predictions.parquet",                                   "#d97706", ":",   1.4, 3),
    ("DeepLOB",             BASE_DIR / "baselines" / "deeplob" / "predictions.parquet",                                   "#dc2626", "-",   1.4, 2),
    ("LightGBM",            BASE_DIR / "baselines" / "lgb"     / "predictions.parquet",                                   "#4b5563", "-.",  1.3, 1),
]

WINDOW_PRE  = 50
WINDOW_POST = 500
WINDOW_TAUS = np.arange(-WINDOW_PRE, WINDOW_POST + 1)
ROLLING_WINDOW = 45
GAUSSIAN_SIGMA = 6

FIG_WIDTH  = 5.8   # base inches
FIG_HEIGHT = 3.4   # base inches
DPI        = 400

# Inner gaps between elements inside the figure
INNER_GAP_Y   = 0.035   # y-label right ↔ y-tick labels
INNER_GAP_X   = 0.070   # x-label top ↔ x-tick labels
INNER_GAP_LEG = 0.080   # top axis spine ↔ legend bottom

# Outer margins for uncropped version in graphs/ (slightly less margin as requested, left/right wider)
MARGIN_LEFT   = 0.200   # left image edge ↔ y-label left
MARGIN_RIGHT  = 0.200   # rightmost x-tick label ↔ right image edge
MARGIN_BOTTOM = 0.110   # bottom image edge ↔ x-label bottom
MARGIN_TOP    = 0.090   # legend top ↔ top image edge

TICK_SIZE = 3.5   # points
TICK_PAD  = 3.0   # points
FONT_SIZE_LABEL = 10.0
FONT_SIZE_TICK  = 8.8
FONT_SIZE_LEG   = 7.6


def setup_rc():
    rcParams.update({
        "font.family":        "Times New Roman",
        "font.serif":         ["Times New Roman"],
        "mathtext.fontset":   "stix",
        "axes.labelsize":     FONT_SIZE_LABEL,
        "xtick.labelsize":    FONT_SIZE_TICK,
        "ytick.labelsize":    FONT_SIZE_TICK,
        "legend.fontsize":    FONT_SIZE_LEG,
        "legend.framealpha":  1.0,
        "legend.fancybox":    False,
        "figure.dpi":         DPI,
        "savefig.dpi":        DPI,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "grid.linestyle":     "--",
        "grid.linewidth":     0.5,
        "axes.linewidth":     0.8,
        "xtick.major.width":  0.7,
        "ytick.major.width":  0.7,
        "xtick.major.size":   TICK_SIZE,
        "ytick.major.size":   TICK_SIZE,
        "xtick.major.pad":    TICK_PAD,
        "ytick.major.pad":    TICK_PAD,
    })


def load_master_and_regimes():
    print("Loading BTC trade master...", flush=True)
    t0 = time.time()
    master = pd.read_parquet(MASTER_PATH, columns=["tick_idx", "px"])
    px = master["px"].to_numpy(dtype=np.float64)
    vol_series = pd.Series(px).pct_change().rolling(2000).std().fillna(0.0).to_numpy() * 10000.0
    vol_pos = vol_series[vol_series > 0]
    vol_q33 = np.quantile(vol_pos, 0.33)
    vol_q66 = np.quantile(vol_pos, 0.66)
    regimes = np.where(vol_series <= vol_q33, 0, np.where(vol_series <= vol_q66, 1, 2))
    master["regime"] = regimes
    print(f"Master loaded in {time.time()-t0:.2f}s ({len(master):,} ticks).", flush=True)
    return master


def compute_adaptation_trajectories(master):
    results_all = {}
    results_r2  = {}
    for name, path, color, ls, lw, zorder in MODELS:
        print(f"Evaluating {name:<22}...", end="", flush=True)
        t0 = time.time()
        if not path.exists():
            print(f" [SKIP] file not found: {path}")
            continue
        df_p = pd.read_parquet(path, columns=["tick_idx", "mid", "pred"])
        df_j = master.merge(df_p[["tick_idx", "mid", "pred"]], on="tick_idx", how="inner").reset_index(drop=True)
        n_75 = len(df_j) // 4
        df_j = df_j.iloc[n_75:].reset_index(drop=True)
        n = len(df_j)
        mid_vals = df_j["mid"].to_numpy(dtype=np.float64)
        diff_mid = np.diff(mid_vals, append=np.nan)
        act_dir  = np.where(diff_mid > 1e-8, 1, np.where(diff_mid < -1e-8, -1, 0))
        nz_mask  = ~np.isnan(diff_mid) & (act_dir != 0)
        correct  = (act_dir == df_j["pred"].to_numpy(dtype=np.int8)) & nz_mask
        reg_arr  = df_j["regime"].to_numpy(dtype=np.int8)
        shifts_all = [s for s in (np.where(np.diff(reg_arr) != 0)[0] + 1) if (s - WINDOW_PRE >= 0 and s + WINDOW_POST < n)]
        shifts_r2  = [s for s in shifts_all if reg_arr[s] == 2]
        mat_all  = np.array(shifts_all)[:, None] + WINDOW_TAUS[None, :]
        da_all   = np.sum(correct[mat_all], 0) / np.sum(nz_mask[mat_all], 0) * 100.0
        mat_r2   = np.array(shifts_r2)[:, None] + WINDOW_TAUS[None, :]
        da_r2    = np.sum(correct[mat_r2], 0) / np.sum(nz_mask[mat_r2], 0) * 100.0
        da_all_s = pd.Series(da_all).rolling(ROLLING_WINDOW, center=True, min_periods=1).mean().to_numpy()
        da_r2_s  = pd.Series(da_r2).rolling(ROLLING_WINDOW, center=True, min_periods=1).mean().to_numpy()
        da_all_s = gaussian_filter1d(da_all_s, sigma=GAUSSIAN_SIGMA)
        da_r2_s  = gaussian_filter1d(da_r2_s,  sigma=GAUSSIAN_SIGMA)
        results_all[name] = (da_all_s, color, ls, lw, zorder)
        results_r2[name]  = (da_r2_s,  color, ls, lw, zorder)
        print(f" done in {time.time()-t0:.2f}s | Shifts: all={len(shifts_all):,}, R2={len(shifts_r2):,}")
    return results_all, results_r2



def _build_figure_and_ax(panel_data, is_r2, l, b, r, t, fig_w=FIG_WIDTH, fig_h=FIG_HEIGHT):
    """Build a complete figure+ax at given subplot fractions. Returns (fig, ax, legend, lines)."""
    bg_color     = "#ffffff"
    text_color   = "#111827"
    axis_color   = "#222222"
    grid_color   = "#d1d5db"
    grid_alpha   = 0.45
    shock_color  = "#dc2626"
    leg_edge_col = "#9ca3af"

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=DPI, facecolor=bg_color)
    ax  = fig.add_axes([l, b, r - l, t - b], facecolor=bg_color)

    ax.grid(True, linestyle="--", linewidth=0.5, color=grid_color, alpha=grid_alpha)
    ax.spines["bottom"].set_color(axis_color)
    ax.spines["left"].set_color(axis_color)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", colors=text_color, direction="out",
                   length=TICK_SIZE, pad=TICK_PAD, labelsize=FONT_SIZE_TICK)
    ax.tick_params(axis="y", colors=text_color, direction="out",
                   length=TICK_SIZE, pad=TICK_PAD, labelsize=FONT_SIZE_TICK)

    ax.axvline(0, color=shock_color, linestyle="--", linewidth=1.2, alpha=0.85, zorder=10)

    lines_for_legend = []
    for name, (curve, color, ls, lw, zorder) in panel_data.items():
        ln, = ax.plot(WINDOW_TAUS, curve, color=color, linestyle=ls, linewidth=lw, zorder=zorder,
                      solid_capstyle="round", dash_capstyle="round",
                      solid_joinstyle="round")
        lines_for_legend.append((ln, name))


    ax.set_xlim(-WINDOW_PRE, WINDOW_POST)
    ax.set_ylim(58.0 if is_r2 else 65.0, 98.0)

    return fig, ax, lines_for_legend


def _add_labels_and_legend(fig, ax, lines_for_legend,
                            yl_w, xl_h, leg_h,
                            ax_x0, ax_y0, ax_x1, ax_y1,
                            fig_w=FIG_WIDTH, fig_h=FIG_HEIGHT):
    """Add axis labels and legend with exact gap placement."""
    bg_color     = "#ffffff"
    text_color   = "#111827"
    leg_edge_col = "#9ca3af"

    ylabel_str = r"Directional Accuracy $\mathrm{DA}_{\mathrm{nz}}$ (%)"
    xlabel_str = r"Relative Trade Ticks $\tau$ from Boundary ($\tau\!=\!0$)"

    # y-label: placed so LEFT edge = MARGIN_LEFT from left image edge
    # → centre x = MARGIN_LEFT + yl_w/2  (in inches from left)
    yl_cx = (MARGIN_LEFT + yl_w / 2.0) / fig_w
    # vertically centred on the axes spine area
    yl_cy = (ax_y0 + ax_y1) / 2.0 / fig_h
    yl_text = fig.text(yl_cx, yl_cy, ylabel_str,
                       ha="center", va="center", rotation=90,
                       color=text_color, fontsize=FONT_SIZE_LABEL,
                       fontfamily="Times New Roman", math_fontfamily="stix")

    # x-label: placed so BOTTOM edge = MARGIN_BOTTOM from bottom image edge
    # → centre y = MARGIN_BOTTOM + xl_h/2  (in inches from bottom)
    xl_cy = (MARGIN_BOTTOM + xl_h / 2.0) / fig_h
    # horizontally centred on the axes spine area
    xl_cx = (ax_x0 + ax_x1) / 2.0 / fig_w
    xl_text = fig.text(xl_cx, xl_cy, xlabel_str,
                       ha="center", va="center", rotation=0,
                       color=text_color, fontsize=FONT_SIZE_LABEL,
                       fontfamily="Times New Roman", math_fontfamily="stix")

    # legend: BOTTOM edge = INNER_GAP_LEG above top spine
    # → y anchor (lower center) in figure fraction = (ax_y1 + INNER_GAP_LEG) / fig_h
    leg_y = (ax_y1 + INNER_GAP_LEG) / fig_h
    leg_x = (ax_x0 + ax_x1) / 2.0 / fig_w

    leg = fig.legend(
        [ln for ln, _ in lines_for_legend],
        [nm for _, nm in lines_for_legend],
        loc="lower center",
        bbox_to_anchor=(leg_x, leg_y),
        bbox_transform=fig.transFigure,
        ncol=6,
        frameon=True,
        facecolor="#ffffff",
        edgecolor="#cccccc",
        framealpha=1.0,
        fontsize=FONT_SIZE_LEG,
        borderpad=0.30,
        borderaxespad=0.0,
        handlelength=1.2,
        handletextpad=0.32,
        columnspacing=0.70,
        mode=None,
    )
    leg.get_frame().set_boxstyle("custom_legend_square,pad_y=0.0,pad_left=0.19,pad_right=0.0")
    leg.get_frame().set_linewidth(0.6)
    leg.get_frame().set_facecolor("#ffffff")
    leg.get_frame().set_edgecolor("#cccccc")
    leg.get_frame().set_alpha(1.0)
    for txt in leg.get_texts():
        txt.set_color(text_color)

    return yl_text, xl_text, leg


def render_single_graph(panel_data, is_r2, out_path):
    setup_rc()

    # ─── PASS 1: Use a wide-margin provisional figure to measure element sizes ──
    # Use large margins so tick labels are not clipped and we measure them correctly
    prov_l, prov_b, prov_r, prov_t = 0.25, 0.25, 0.85, 0.85
    fig1, ax1, lines1 = _build_figure_and_ax(panel_data, is_r2, prov_l, prov_b, prov_r, prov_t)
    ax1.set_xlabel("", labelpad=0)
    ax1.set_ylabel("", labelpad=0)

    fig1.canvas.draw()
    ren1 = fig1.canvas.get_renderer()

    def wext1(a):
        return a.get_window_extent(ren1)

    fig_bb1 = fig1.get_window_extent(ren1)
    ax_bb1  = wext1(ax1)

    # y-axis label string measurement using throwaway text
    yl_probe = fig1.text(0.5, 0.5,
                         r"Directional Accuracy $\mathrm{DA}_{\mathrm{nz}}$ (%)",
                         ha="center", va="center", rotation=90,
                         fontsize=FONT_SIZE_LABEL,
                         fontfamily="Times New Roman", math_fontfamily="stix")
    fig1.canvas.draw()
    yl_bb = yl_probe.get_window_extent(ren1)
    yl_w  = yl_bb.width / DPI     # horizontal extent of rotated label
    yl_probe.remove()

    xl_probe = fig1.text(0.5, 0.5,
                         r"Relative Trade Ticks $\tau$ from Boundary ($\tau\!=\!0$)",
                         ha="center", va="center", rotation=0,
                         fontsize=FONT_SIZE_LABEL,
                         fontfamily="Times New Roman", math_fontfamily="stix")
    fig1.canvas.draw()
    xl_bb = xl_probe.get_window_extent(ren1)
    xl_h  = xl_bb.height / DPI    # vertical extent of horizontal label
    xl_probe.remove()

    fig1.canvas.draw()
    ren1 = fig1.canvas.get_renderer()
    ax_bb1 = wext1(ax1)

    # Measure actual tick spans from rendered positions
    ytlabels = [t for t in ax1.get_yticklabels() if t.get_visible() and t.get_text() != ""]
    xtlabels = [t for t in ax1.get_xticklabels() if t.get_visible() and t.get_text() != ""]

    yt_x0       = min(wext1(t).x0 for t in ytlabels)
    xt_y0       = min(wext1(t).y0 for t in xtlabels)
    xt_x1       = max(wext1(t).x1 for t in xtlabels)

    # Spans from tick-label edge to spine (includes tick mark + pad + text)
    yt_span = (ax_bb1.x0 - yt_x0) / DPI   # inches: y-tick leftmost → left spine
    xt_span = (ax_bb1.y0 - xt_y0) / DPI   # inches: x-tick bottommost → bottom spine
    xt_rovh = max(0.0, (xt_x1 - ax_bb1.x1) / DPI)  # rightmost x-tick right overhang

    # Measure legend height with a temporary legend
    prov_leg = fig1.legend(
        [ln for ln, _ in lines1], [nm for _, nm in lines1],
        loc="upper center", bbox_to_anchor=(0.5, 0.98),
        bbox_transform=fig1.transFigure,
        ncol=6, frameon=True, fontsize=FONT_SIZE_LEG,
        borderpad=0.30, handlelength=1.2, handletextpad=0.32, columnspacing=0.70,
    )
    prov_leg.get_frame().set_boxstyle("custom_legend_square,pad_y=0.0,pad_left=0.19,pad_right=0.0")
    prov_leg.get_frame().set_linewidth(0.6)
    fig1.canvas.draw()
    ren1 = fig1.canvas.get_renderer()
    leg_h = wext1(prov_leg.get_frame()).height / DPI

    plt.close(fig1)

    print(f"  [Measured]  yl_w={yl_w:.4f}\"  xl_h={xl_h:.4f}\"  "
          f"yt_span={yt_span:.4f}\"  xt_span={xt_span:.4f}\"  "
          f"xt_rovh={xt_rovh:.4f}\"  leg_h={leg_h:.4f}\"", flush=True)

    # ─── PASS 2: Compute exact subplot margins from actual measurements ──────────
    ax_width  = 5.1486
    ax_height = 2.5560

    left_in   = MARGIN_LEFT + yl_w + INNER_GAP_Y + yt_span
    bottom_in = MARGIN_BOTTOM + xl_h + INNER_GAP_X + xt_span
    right_in  = xt_rovh + MARGIN_RIGHT
    top_in    = INNER_GAP_LEG + leg_h + MARGIN_TOP

    fig_w = left_in + ax_width + right_in
    fig_h = bottom_in + ax_height + top_in

    l = left_in   / fig_w
    b = bottom_in / fig_h
    r = (left_in + ax_width) / fig_w
    t = (bottom_in + ax_height) / fig_h

    print(f"  [Margins]   left={left_in:.4f}\" ({l:.4f}f)  "
          f"bottom={bottom_in:.4f}\" ({b:.4f}f)  "
          f"right={right_in:.4f}\" ({1-r:.4f}f)  "
          f"top={top_in:.4f}\" ({1-t:.4f}f)  "
          f"fig=({fig_w:.4f}\" x {fig_h:.4f}\")", flush=True)

    # ─── PASS 3: Build final figure with computed margins ────────────────────────
    fig2, ax2, lines2 = _build_figure_and_ax(panel_data, is_r2, l, b, r, t, fig_w, fig_h)

    fig2.canvas.draw()
    ren2 = fig2.canvas.get_renderer()

    def wext2(a):
        return a.get_window_extent(ren2)

    fig_bb2 = fig2.get_window_extent(ren2)
    ax_bb2  = wext2(ax2)

    # Spine positions in inches (from figure bottom-left corner)
    ax_x0_in = (ax_bb2.x0 - fig_bb2.x0) / DPI
    ax_y0_in = (ax_bb2.y0 - fig_bb2.y0) / DPI
    ax_x1_in = (ax_bb2.x1 - fig_bb2.x0) / DPI
    ax_y1_in = (ax_bb2.y1 - fig_bb2.y0) / DPI

    print(f"  [Spine pos] L={ax_x0_in:.4f}\"  R={ax_x1_in:.4f}\"  "
          f"B={ax_y0_in:.4f}\"  T={ax_y1_in:.4f}\"", flush=True)

    # ─── PASS 3b: Add labels and legend at exact positions ───────────────────────
    yl_text, xl_text, leg = _add_labels_and_legend(
        fig2, ax2, lines2, yl_w, xl_h, leg_h,
        ax_x0_in, ax_y0_in, ax_x1_in, ax_y1_in,
        fig_w, fig_h
    )

    # ─── PASS 4: Final draw and verify ───────────────────────────────────────────
    fig2.canvas.draw()
    ren2 = fig2.canvas.get_renderer()

    fig_bb3 = fig2.get_window_extent(ren2)
    ax_bb3  = wext2(ax2)
    yl_bb3  = yl_text.get_window_extent(ren2)
    xl_bb3  = xl_text.get_window_extent(ren2)
    leg_bb3 = leg.get_frame().get_window_extent(ren2)

    g_yl_left  = (yl_bb3.x0 - fig_bb3.x0) / DPI
    g_xl_bot   = (xl_bb3.y0 - fig_bb3.y0) / DPI
    g_leg_top  = (fig_bb3.y1 - leg_bb3.y1) / DPI

    # Right margin: rightmost x-tick right → right image edge
    ytlabels3 = [t for t in ax2.get_yticklabels() if t.get_visible() and t.get_text() != ""]
    xtlabels3 = [t for t in ax2.get_xticklabels() if t.get_visible() and t.get_text() != ""]
    xt_x1_3   = max(wext2(tt).x1 for tt in xtlabels3) if xtlabels3 else ax_bb3.x1
    g_right   = (fig_bb3.x1 - xt_x1_3) / DPI

    print(f"  [Verify Margins]  Left={g_yl_left:.4f}\" (tgt={MARGIN_LEFT:.4f}\")  Right={g_right:.4f}\" (tgt={MARGIN_RIGHT:.4f}\")  Bot={g_xl_bot:.4f}\" (tgt={MARGIN_BOTTOM:.4f}\")  Top={g_leg_top:.4f}\" (tgt={MARGIN_TOP:.4f}\")", flush=True)

    # ─── Save exact uncropped with margin version to graphs/ ───────────────────────
    fig2.canvas.draw()
    fig2.savefig(out_path, dpi=DPI, facecolor="#ffffff")
    plt.close(fig2)
    print(f"[SAVED uncropped] {out_path}", flush=True)


def _crop_tight_to_content(src_path: Path, dst_path: Path):
    """Save a tight-cropped (no margin) copy of src to dst using PIL."""
    img = Image.open(src_path).convert("RGB")
    # Convert to numpy, find bounding box of non-white pixels
    arr = np.array(img)
    # White background: pixel is white if all channels >= 253
    mask = ~np.all(arr >= 253, axis=2)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        cropped = img.crop((cmin, rmin, cmax + 1, rmax + 1))
    else:
        cropped = img
    cropped.save(dst_path, dpi=(DPI, DPI))
    print(f"[CROPPED] → {dst_path}", flush=True)


def main():
    print("=" * 80)
    print("  REGIME SHIFT ADAPTATION GRAPHS  |  Four-Pass Geometry Engine  |  400 DPI")
    print("=" * 80)

    if CACHE_PATH.exists():
        print(f"Using cached trajectory data from {CACHE_PATH}...", flush=True)
        with open(CACHE_PATH, "rb") as f:
            results_all, results_r2 = pickle.load(f)
    else:
        master = load_master_and_regimes()
        results_all, results_r2 = compute_adaptation_trajectories(master)
        try:
            with open(CACHE_PATH, "wb") as f:
                pickle.dump((results_all, results_r2), f)
            print(f"Cached to {CACHE_PATH}")
        except Exception:
            pass

    p_all = GRAPHS_DIR / "regime_shift_adaptation_all.png"
    render_single_graph(results_all, is_r2=False, out_path=p_all)
    m_all = MANUSCRIPT_FIGS_DIR / "regime_shift_adaptation_all.png"
    _crop_tight_to_content(p_all, m_all)

    p_shock = GRAPHS_DIR / "regime_shift_adaptation_shock.png"
    render_single_graph(results_r2, is_r2=True, out_path=p_shock)
    m_shock = MANUSCRIPT_FIGS_DIR / "regime_shift_adaptation_shock.png"
    _crop_tight_to_content(p_shock, m_shock)

    print("\n[COMPLETE]")




if __name__ == "__main__":
    main()
