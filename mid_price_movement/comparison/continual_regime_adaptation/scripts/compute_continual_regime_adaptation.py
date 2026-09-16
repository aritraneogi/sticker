import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

warnings.filterwarnings("ignore")

BASE_DIR    = Path(__file__).resolve().parent.parent.parent.parent
OUT_DIR     = Path(__file__).resolve().parent.parent
MASTER_PATH = BASE_DIR / "data" / "extracted" / "btc_trades_master.parquet"

NOTIONAL_USD     = 100_000
AVG_TICK_TIME_MS = 22.0
POST_SHIFT_TICKS = 500
HORIZONS         = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500]

import sys
sys.path.insert(0, str(BASE_DIR / "comparison" / "core_model_performance" / "scripts"))
from compute_trading_metrics import MODEL_CONFIGS


def evaluate_continual_adaptation(name: str, pred_path: Path, master_df: pd.DataFrame):
    if not pred_path.exists():
        print(f"[SKIP] {name}: file not found", flush=True)
        return None

    t0 = time.time()

    schema_names = pq.read_schema(pred_path).names
    cols = ["tick_idx", "mid", "pred"] + (["lat_ms"] if "lat_ms" in schema_names else [])
    df_p = pd.read_parquet(pred_path, columns=cols)

    p50_ms = float(df_p["lat_ms"].median()) if "lat_ms" in df_p.columns and df_p["lat_ms"].notna().any() else 0.0
    p99_ms = float(df_p["lat_ms"].quantile(0.99)) if "lat_ms" in df_p.columns and df_p["lat_ms"].notna().any() else 0.0

    df_j = master_df.merge(df_p[["tick_idx", "mid", "pred"]], on="tick_idx", how="inner").reset_index(drop=True)
    n_total = len(df_j)
    if n_total < 100:
        return None

    n_75 = n_total // 4
    df_j = df_j.iloc[n_75:].reset_index(drop=True)
    n = len(df_j)

    raw_pred  = df_j["pred"].to_numpy(dtype=np.int8)
    side_sign = df_j["side_sign"].to_numpy(dtype=np.int8)
    mid_vals  = df_j["mid"].to_numpy(dtype=np.float64)
    reg_arr   = df_j["regime"].to_numpy(dtype=np.int8)
    dm5_j     = df_j["dm5_bps"].to_numpy(dtype=np.float64)

    diff_mid = np.diff(mid_vals, append=np.nan)
    act_dir  = np.where(diff_mid > 1e-8, 1, np.where(diff_mid < -1e-8, -1, 0))
    nz_mask  = ~np.isnan(diff_mid) & (act_dir != 0)

    correct_nz = (act_dir == raw_pred) & nz_mask
    df_j["is_nz"] = nz_mask
    df_j["correct_nz"] = correct_nz

    da_overall = float(np.mean(act_dir[nz_mask] == raw_pred[nz_mask])) * 100.0 if nz_mask.sum() > 0 else 50.0

    r0_mask = (reg_arr == 0) & nz_mask
    r1_mask = (reg_arr == 1) & nz_mask
    r2_mask = (reg_arr == 2) & nz_mask

    da_r0 = float(np.mean(correct_nz[r0_mask])) * 100.0 if r0_mask.sum() > 0 else None
    da_r1 = float(np.mean(correct_nz[r1_mask])) * 100.0 if r1_mask.sum() > 0 else None
    da_r2 = float(np.mean(correct_nz[r2_mask])) * 100.0 if r2_mask.sum() > 0 else None

    shift_indices    = np.where(np.diff(reg_arr) != 0)[0] + 1
    post_shift_mask  = np.zeros(n, dtype=bool)
    for idx in shift_indices:
        post_shift_mask[idx:min(idx + POST_SHIFT_TICKS, n)] = True

    ps_nz_mask     = post_shift_mask & nz_mask
    steady_nz_mask = (~post_shift_mask) & nz_mask

    da_post_shift = float(np.mean(correct_nz[ps_nz_mask])) * 100.0 if ps_nz_mask.sum() > 0 else None
    da_steady     = float(np.mean(correct_nz[steady_nz_mask])) * 100.0 if steady_nz_mask.sum() > 0 else None
    adaptation_speed_delta = (da_post_shift - da_steady) if (da_post_shift is not None and da_steady is not None) else None

    r0_indices = np.where(reg_arr == 0)[0]
    n_r0 = len(r0_indices)
    if n_r0 >= 6:
        r0_early_idx = r0_indices[:n_r0 // 3]
        r0_late_idx  = r0_indices[-(n_r0 // 3):]
        early_nz_mask = nz_mask[r0_early_idx]
        late_nz_mask  = nz_mask[r0_late_idx]
        da_r0_early = float(np.mean(correct_nz[r0_early_idx][early_nz_mask])) * 100.0 if early_nz_mask.sum() > 0 else None
        da_r0_late  = float(np.mean(correct_nz[r0_late_idx][late_nz_mask])) * 100.0 if late_nz_mask.sum() > 0 else None
        bwt = (da_r0_late - da_r0_early) if (da_r0_late is not None and da_r0_early is not None) else None
    else:
        da_r0_early = None
        da_r0_late  = None
        bwt         = None

    p_up   = np.mean(raw_pred == 1)
    p_down = np.mean(raw_pred == -1)
    p_neut = np.mean(raw_pred == 0)
    probs  = np.array([p_up, p_down, p_neut])
    probs  = probs[probs > 0]
    entropy = float(-np.sum(probs * np.log2(probs))) if len(probs) > 0 else 0.0

    days = pd.qcut(df_j["tick_idx"], q=31, labels=range(1, 32), duplicates="drop")
    df_j["day"] = days
    daily_stats = df_j.groupby("day").agg(
        total_nz=("is_nz", "sum"),
        correct_nz_sum=("correct_nz", "sum")
    )
    daily_stats["da_nz"] = daily_stats["correct_nz_sum"] / daily_stats["total_nz"].replace(0, np.nan) * 100.0
    daily_stats = daily_stats.dropna(subset=["da_nz"])
    x = np.arange(len(daily_stats))
    y = daily_stats["da_nz"].to_numpy()
    drift_slope = float(np.polyfit(x, y, 1)[0]) if len(x) > 1 else 0.0

    raw_bps = np.where(side_sign == -1, 1.45 + dm5_j, 1.45 - dm5_j)
    pred_correct = (((raw_pred == 1) & (dm5_j > 0)) | ((raw_pred == -1) & (dm5_j < 0)) | (raw_pred == 0))
    p_adverse = np.where(pred_correct, 0.08, 0.50)
    eff_bps = raw_bps * (1.0 - p_adverse)

    pnl_r0_usd = float(np.sum(eff_bps[reg_arr == 0])) / 10_000.0 * (NOTIONAL_USD / 10.0) if r0_mask.sum() > 0 else 0.0
    pnl_r1_usd = float(np.sum(eff_bps[reg_arr == 1])) / 10_000.0 * (NOTIONAL_USD / 10.0) if r1_mask.sum() > 0 else 0.0
    pnl_r2_usd = float(np.sum(eff_bps[reg_arr == 2])) / 10_000.0 * (NOTIONAL_USD / 10.0) if r2_mask.sum() > 0 else 0.0
    total_pnl_usd = pnl_r0_usd + pnl_r1_usd + pnl_r2_usd

    h_lag = int(np.round(p50_ms / AVG_TICK_TIME_MS))
    lat_penalty = 1.0 if h_lag == 0 else (1.0 + h_lag * 0.25)

    da_shock_frac  = (da_r2 / 100.0) if da_r2 is not None else 0.5
    da_post_frac   = (da_post_shift / 100.0) if da_post_shift is not None else 0.5
    da_steady_frac = max((da_steady / 100.0) if da_steady is not None else 0.5, 1e-6)
    bwt_val        = bwt if bwt is not None else 0.0

    score_fractional = float(
        (da_shock_frac * (1.0 + bwt_val / 100.0) * (da_post_frac / da_steady_frac) * min(entropy, 1.0)) / lat_penalty
    )

    ms_mid = pd.Series(mid_vals)
    horizon_da_nz = {}
    for h in HORIZONS:
        diff_h = ms_mid.shift(-h).values - mid_vals
        act_h  = np.where(diff_h > 1e-8, 1, np.where(diff_h < -1e-8, -1, 0))
        nz_h   = ~np.isnan(diff_h) & (act_h != 0)
        horizon_da_nz[f"h{h}"] = float(np.mean(act_h[nz_h] == raw_pred[nz_h])) if nz_h.sum() > 0 else None

    elapsed = time.time() - t0
    print(f"[DONE] {name:<45} | R2(Worst)={da_r2 or 'N/A'} | PostShift={da_post_shift or 'N/A'} | BWT={bwt} | S_Continual={score_fractional:.4f} | {elapsed:.1f}s", flush=True)

    return {
        "model_name": name,
        "eval_ticks": n,
        "overall_da_non_zero_pct": round(da_overall, 2),
        "da_nz_h1": round(da_overall / 100.0, 4),
        "horizon_da_nz": horizon_da_nz,
        "da_low_vol_consolidation_pct": round(da_r0, 2) if da_r0 is not None else None,
        "da_moderate_trend_pct": round(da_r1, 2) if da_r1 is not None else None,
        "da_high_vol_shock_worst_pct": round(da_r2, 2) if da_r2 is not None else None,
        "immediate_post_shift_da_pct": round(da_post_shift, 2) if da_post_shift is not None else None,
        "steady_state_da_pct": round(da_steady, 2) if da_steady is not None else None,
        "adaptation_speed_delta_pct": round(adaptation_speed_delta, 2) if adaptation_speed_delta is not None else None,
        "early_regime0_da_pct": round(da_r0_early, 2) if da_r0_early is not None else None,
        "late_regime0_da_pct": round(da_r0_late, 2) if da_r0_late is not None else None,
        "backward_transfer_bwt_pct": round(bwt, 2) if bwt is not None else None,
        "shannon_entropy_bits": round(entropy, 4),
        "daily_drift_slope_pct_day": round(drift_slope, 4),
        "p50_latency_ms": round(p50_ms, 4),
        "stale_lag_ticks": h_lag,
        "pnl_usd_low_vol": round(pnl_r0_usd, 2),
        "pnl_usd_moderate_trend": round(pnl_r1_usd, 2),
        "pnl_usd_high_vol_shock": round(pnl_r2_usd, 2),
        "total_realized_pnl_usd": round(total_pnl_usd, 2),
        "continual_adaptation_score": round(score_fractional, 4),
    }


def main():
    print("=" * 95, flush=True)
    print("  CONTINUAL REGIME ADAPTATION - LAST 75% TICKS EVALUATION")
    print("=" * 95, flush=True)

    print("\nLoading BTC trade master...", flush=True)
    master = pd.read_parquet(MASTER_PATH)
    px = master["px"].to_numpy(dtype=np.float64)
    sz = master["sz"].to_numpy(dtype=np.float64)
    ss = master["side_sign"].to_numpy(dtype=np.int8)

    mid5 = np.roll(px, -5); mid5[-5:] = px[-5:]
    dm5_bps = (mid5 - px) / px * 10_000.0

    vol_series = pd.Series(px).pct_change().rolling(2000).std().fillna(0.0).to_numpy() * 10000.0
    vol_q33 = np.quantile(vol_series[vol_series > 0], 0.33)
    vol_q66 = np.quantile(vol_series[vol_series > 0], 0.66)
    regimes = np.where(vol_series <= vol_q33, 0, np.where(vol_series <= vol_q66, 1, 2))

    master = master.copy()
    master["dm5_bps"] = dm5_bps
    master["regime"]  = regimes

    print(f"Loaded {len(master):,} trades. Regimes segmented. Evaluating...\n", flush=True)

    results = []
    for name, path in MODEL_CONFIGS:
        res = evaluate_continual_adaptation(name, path, master)
        if res is not None:
            results.append(res)

    results_sorted = sorted(results, key=lambda x: -x["continual_adaptation_score"])

    json_path = OUT_DIR / "test_metrics.json"
    with open(json_path, "w") as f:
        json.dump(results_sorted, f, indent=2)
    print(f"\n[SAVED] {json_path}")

    csv_path = OUT_DIR / "test_rank.csv"
    pd.DataFrame(results_sorted).sort_values("da_nz_h1", ascending=False).to_csv(csv_path, index=False)
    print(f"[SAVED] {csv_path}")

    md = [
        "# Continual Learning & Regime Adaptation Leaderboard",
        "",
        "**Dataset**: 11,918,929 Hyperliquid BTC Perpetual Futures Ticks (Dec 2025)  ",
        "**Evaluation Basis**: Last 75% of ticks after merge with trade master  ",
        "",
        "---",
        "",
        "## Master Leaderboard (Ranked by Continual Adaptation Score)",
        "",
        "| Rank | Model Architecture | Overall DA | Low-Vol DA | Mod-Trend DA | Worst Shock DA | Post-Shift DA (500tk) | BWT | Entropy | p50 Latency | Continual Score |",
        "|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for i, r in enumerate(results_sorted, 1):
        lat_str = f"{r['p50_latency_ms']*1000:.1f} us" if r['p50_latency_ms'] < 0.1 else f"{r['p50_latency_ms']:.2f} ms"
        bwt_val = r['backward_transfer_bwt_pct']
        bwt_str = f"+{bwt_val:.2f}%" if bwt_val is not None and bwt_val >= 0 else (f"{bwt_val:.2f}%" if bwt_val is not None else "N/A")
        da_r2   = r['da_high_vol_shock_worst_pct']
        da_ps   = r['immediate_post_shift_da_pct']
        da_r0   = r['da_low_vol_consolidation_pct']
        da_r1   = r['da_moderate_trend_pct']
        md.append(
            f"| {i} | **{r['model_name']}** | {r['overall_da_non_zero_pct']:.1f}% "
            f"| {da_r0:.1f}% " if da_r0 is not None else "| N/A "
            f"| {da_r1:.1f}% " if da_r1 is not None else "| N/A "
            f"| {da_r2:.1f}% " if da_r2 is not None else "| N/A "
            f"| {da_ps:.1f}% " if da_ps is not None else "| N/A "
            f"| **{bwt_str}** | {r['shannon_entropy_bits']:.3f} | {lat_str} | **{r['continual_adaptation_score']:.4f}** |"
        )

    md += [
        "",
        "---",
        "",
        "## Continual Learning Metric Definitions",
        "",
        "| Metric | Formula / Meaning |",
        "|:---|:---|",
        "| **Worst Shock DA** | Directional accuracy during High-Vol Extreme Shock regimes |",
        "| **Post-Shift DA (500tk)** | Accuracy in the first 500 ticks following a regime transition |",
        "| **Backward Transfer (BWT)** | DA(Late Regime 0) - DA(Early Regime 0); + = retained/improved, - = forgetting |",
        "| **Shannon Entropy** | -sum p_i log2(p_i) (max 1.0); mode-collapse diagnostic |",
        "| **Continual Score** | (Worst DA x (1 + BWT/100) x (Post-Shift DA / Steady DA) x H) / Latency Penalty |",
    ]

    md_path = OUT_DIR / "README.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"[SAVED] {md_path}")
    print("\n" + "=" * 95)
    print("  CONTINUAL REGIME ADAPTATION EVALUATION COMPLETE!")
    print("=" * 95)


if __name__ == "__main__":
    main()
