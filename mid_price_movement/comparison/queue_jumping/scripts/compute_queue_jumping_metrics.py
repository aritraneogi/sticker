import json
import time
import warnings
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

warnings.filterwarnings("ignore")

BASE_DIR    = Path(__file__).resolve().parent.parent.parent.parent
OUT_DIR     = Path(__file__).resolve().parent.parent
MASTER_PATH = BASE_DIR / "data" / "extracted" / "btc_trades_master.parquet"

NOTIONAL_USD      = 100_000
MAX_INV_BTC       = 5.0
AVG_TICK_TIME_MS  = 22.0
DATA_DAYS         = 31.0
HORIZONS          = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500]

Q_JUMP_COST_BPS   = 0.05
PRICE_IMPROVE_JUMP= 0.70
BASE_HS_BPS       = 1.50
Q_WINDOW_TICKS    = 20
ARRIVAL_DECAY     = 0.18
SWEEP_FACTOR      = 2.5
BASE_ADV_SEL_RATE = 0.08

sys.path.insert(0, str(BASE_DIR / "comparison" / "core_model_performance" / "scripts"))
from compute_trading_metrics import MODEL_CONFIGS


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def evaluate_model(name, pred_path, master_df, dm5_bps_col, px_col):
    if not pred_path.exists():
        print(f"[SKIP] {name}: not found", flush=True)
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
    sz        = df_j["sz"].to_numpy(dtype=np.float64)
    mid_vals  = df_j["mid"].to_numpy(dtype=np.float64)
    dm5_j     = df_j["dm5_bps"].to_numpy(dtype=np.float64)
    px_j      = df_j["px"].to_numpy(dtype=np.float64)

    diff_mid = np.diff(mid_vals, append=np.nan)
    act_dir  = np.where(diff_mid > 1e-8, 1, np.where(diff_mid < -1e-8, -1, 0))
    nz_mask  = ~np.isnan(diff_mid) & (act_dir != 0)
    da_nz    = float(np.mean(act_dir[nz_mask] == raw_pred[nz_mask])) * 100.0 if nz_mask.sum() > 0 else 50.0
    da_trade = float(np.mean(raw_pred == side_sign)) * 100.0

    h_lag = int(np.round(p50_ms / AVG_TICK_TIME_MS))
    if h_lag > 0:
        pred = np.roll(raw_pred, h_lag); pred[:h_lag] = 0
        px_stale = np.roll(px_j, h_lag); px_stale[:h_lag] = px_j[:h_lag]
        stale_drift = np.abs(px_j - px_stale) / px_j * 10_000.0
        lat_decay   = float(np.exp(-ARRIVAL_DECAY * h_lag))
    else:
        pred = raw_pred
        stale_drift = np.zeros(n, dtype=np.float32)
        lat_decay   = 1.0

    qd_our    = np.where(side_sign == -1, df_j["qd_buy"].to_numpy(), df_j["qd_sell"].to_numpy())
    pi_fac    = np.where(pred != 0, PRICE_IMPROVE_JUMP, 0.0)
    queue_pos = qd_our * (1.0 - pi_fac) + 1e-9
    raw_pof   = np.minimum(1.0, sz / queue_pos)
    pof_lat   = raw_pof * lat_decay

    pred_correct = (((pred == 1) & (dm5_j > 0)) | ((pred == -1) & (dm5_j < 0)) | (pred == 0))
    p_adv_correct   = np.full(n, BASE_ADV_SEL_RATE, dtype=np.float64)
    p_adv_incorrect = sigmoid(pof_lat * SWEEP_FACTOR) * 0.85
    p_adverse = np.where(pred_correct, p_adv_correct, p_adv_incorrect)

    raw_bps = np.where(
        side_sign == -1,
        (BASE_HS_BPS - Q_JUMP_COST_BPS) + dm5_j,
        (BASE_HS_BPS - Q_JUMP_COST_BPS) - dm5_j,
    )
    eff_bps = raw_bps * pof_lat * (1.0 - p_adverse) - stale_drift

    inv = 0.0
    fills_taken = 0
    actual_eff_fills = 0.0
    pnl_arr = np.zeros(n, dtype=np.float32)
    inv_arr = np.zeros(n, dtype=np.float32)

    for i in range(n):
        ss = side_sign[i]; size = sz[i]; pr = pred[i]
        wb = (pr == 0) or (pr == 1); ws = (pr == 0) or (pr == -1)
        if ss == -1 and wb and inv + size <= MAX_INV_BTC:
            inv += size * pof_lat[i]
            pnl_arr[i] = float(eff_bps[i])
            actual_eff_fills += pof_lat[i]
            fills_taken += 1
        elif ss == 1 and ws and inv - size >= -MAX_INV_BTC:
            inv -= size * pof_lat[i]
            pnl_arr[i] = float(eff_bps[i])
            actual_eff_fills += pof_lat[i]
            fills_taken += 1
        inv_arr[i] = inv

    net_bps = float(np.sum(pnl_arr))
    net_usd = net_bps / 10_000.0 * NOTIONAL_USD
    tpy     = n / DATA_DAYS * 365.0
    mu      = float(np.mean(pnl_arr))
    sd      = float(np.std(pnl_arr))
    sharpe  = float((mu / sd) * np.sqrt(tpy)) if sd > 0 else 0.0
    neg     = pnl_arr[pnl_arr < 0]
    ds      = float(np.std(neg)) if len(neg) > 0 else 1e-10
    sortino = float((mu / ds) * np.sqrt(tpy))

    mean_pof_raw = float(np.mean(raw_pof))
    mean_pof_lat = float(np.mean(pof_lat))
    mean_adv_sel = float(np.mean(p_adverse)) * 100.0
    flow_purity  = 100.0 - mean_adv_sel

    realized_bps_per_fill = net_bps / max(actual_eff_fills, 1.0)

    inv_std = float(np.std(inv_arr))
    inv_max = float(np.max(np.abs(inv_arr)))

    return_on_inv_risk = realized_bps_per_fill / max(inv_std, 0.1)

    qd_mean = float(np.mean(qd_our))
    qd_p95  = float(np.percentile(qd_our, 95))
    wc_pct  = float(np.sum((pof_lat > 0.9) & (~pred_correct))) / n * 100.0

    lat_penalty = 1.0 if h_lag == 0 else (1.0 + h_lag * 0.25)

    risk_adj_quality_score = float(
        ((da_nz / 100.0) * (flow_purity / 100.0) * realized_bps_per_fill)
        / (max(inv_std, 0.5) * lat_penalty) * 100.0
    )

    ms_mid = pd.Series(mid_vals)
    horizon_da_nz = {}
    for h in HORIZONS:
        diff_h = ms_mid.shift(-h).values - mid_vals
        act_h  = np.where(diff_h > 1e-8, 1, np.where(diff_h < -1e-8, -1, 0))
        nz_h   = ~np.isnan(diff_h) & (act_h != 0)
        horizon_da_nz[f"h{h}"] = float(np.mean(act_h[nz_h] == raw_pred[nz_h])) if nz_h.sum() > 0 else None

    elapsed = time.time() - t0
    print(
        f"[DONE] {name:<45} | DA={da_nz:.1f}% | Edge={realized_bps_per_fill:.3f} bps | "
        f"InvStd={inv_std:.2f} | Net=${net_usd/1e6:.2f}M | Q_RiskAdj={risk_adj_quality_score:.4f} | {elapsed:.1f}s",
        flush=True
    )
    return {
        "model_name"                         : name,
        "eval_ticks"                         : n,
        "true_da_non_zero_pct"               : round(da_nz, 2),
        "da_nz_h1"                           : round(da_nz / 100.0, 4),
        "horizon_da_nz"                      : horizon_da_nz,
        "trade_level_da_pct"                 : round(da_trade, 2),
        "p50_latency_ms"                     : round(p50_ms, 4),
        "p99_latency_ms"                     : round(p99_ms, 4),
        "stale_lag_ticks"                    : h_lag,
        "queue_depth_mean_btc"               : round(qd_mean, 4),
        "queue_depth_p95_btc"                : round(qd_p95, 4),
        "mean_pof_raw"                       : round(mean_pof_raw, 4),
        "mean_pof_latency_adjusted"          : round(mean_pof_lat, 4),
        "latency_decay_factor"               : round(lat_decay, 4),
        "mean_adverse_selection_pct"         : round(mean_adv_sel, 2),
        "flow_purity_pct"                    : round(flow_purity, 2),
        "winners_curse_events_pct"           : round(wc_pct, 4),
        "fills_executed_count"               : fills_taken,
        "effective_fills_executed"           : round(actual_eff_fills, 2),
        "net_bps_per_executed_fill"          : round(realized_bps_per_fill, 4),
        "net_pnl_bps"                        : round(net_bps, 2),
        "net_pnl_usd"                        : round(net_usd, 2),
        "inventory_std_btc"                  : round(inv_std, 4),
        "inventory_max_btc"                  : round(inv_max, 4),
        "return_on_inventory_risk"           : round(return_on_inv_risk, 4),
        "pof_adjusted_sharpe"                : round(sharpe, 2),
        "pof_adjusted_sortino"               : round(sortino, 2),
        "risk_adjusted_institutional_quality": round(risk_adj_quality_score, 4),
    }


def main():
    print("=" * 95, flush=True)
    print("  PoF-WEIGHTED QUEUE-JUMPING - LAST 75% TICKS EVALUATION")
    print("  Layers: Queue Depth + Latency PoF Decay + Winner's Curse Adverse Selection")
    print("=" * 95, flush=True)

    print("\nLoading BTC trade master...", flush=True)
    master  = pd.read_parquet(MASTER_PATH)
    px      = master["px"].to_numpy(dtype=np.float64)
    sz_arr  = master["sz"].to_numpy(dtype=np.float64)
    ss_arr  = master["side_sign"].to_numpy(dtype=np.int8)

    mid5    = np.roll(px, -5); mid5[-5:] = px[-5:]
    dm5_bps = (mid5 - px) / px * 10_000.0

    buy_sz  = np.where(ss_arr == 1,  sz_arr, 0.0)
    sell_sz = np.where(ss_arr == -1, sz_arr, 0.0)
    qd_buy  = pd.Series(buy_sz).rolling(Q_WINDOW_TICKS,  min_periods=1).sum().to_numpy()
    qd_sell = pd.Series(sell_sz).rolling(Q_WINDOW_TICKS, min_periods=1).sum().to_numpy()

    master = master.copy()
    master["dm5_bps"] = dm5_bps
    master["qd_buy"]  = qd_buy
    master["qd_sell"] = qd_sell

    print(f"Loaded {len(master):,} trades | Q_window={Q_WINDOW_TICKS} ticks | Evaluating...\n", flush=True)

    results = []
    for name, path in MODEL_CONFIGS:
        res = evaluate_model(name, path, master, dm5_bps, px)
        if res is not None:
            results.append(res)

    results_sorted = sorted(results, key=lambda x: -x["risk_adjusted_institutional_quality"])

    json_path = OUT_DIR / "test_metrics.json"
    with open(json_path, "w") as f:
        json.dump(results_sorted, f, indent=2)
    print(f"\n[SAVED] {json_path}")

    csv_path = OUT_DIR / "test_rank.csv"
    pd.DataFrame(results_sorted).sort_values("da_nz_h1", ascending=False).to_csv(csv_path, index=False)
    print(f"[SAVED] {csv_path}")

    md = [
        "# Risk & Capital-Constrained Queue-Jumping MM Leaderboard",
        "",
        "**Dataset**: 11,918,929 Hyperliquid BTC Perpetual Futures Ticks (Dec 2025)  ",
        "**Evaluation Basis**: Last 75% of ticks after merge with trade master  ",
        f"**Inventory Cap**: +/-5.0 BTC | **Q Window**: {Q_WINDOW_TICKS} ticks",
        "",
        "---",
        "",
        "## Overall Leaderboard (Ranked by Risk-Adjusted Quality Score)",
        "",
        "| Rank | Model Architecture | True DA (NZ) | Edge (bps/fill) | Flow Purity (% Safe) | Inv. Std | Return on Inv. Risk | p50 Latency | Realized Net ($) | Risk-Adj. Quality Score |",
        "|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for i, r in enumerate(results_sorted, 1):
        lat_str = f"{r['p50_latency_ms']*1000:.1f} us" if r['p50_latency_ms'] < 0.1 else f"{r['p50_latency_ms']:.2f} ms"
        md.append(
            f"| {i} | **{r['model_name']}** | {r['true_da_non_zero_pct']:.1f}% "
            f"| {r['net_bps_per_executed_fill']:.3f} | {r['flow_purity_pct']:.1f}% "
            f"| {r['inventory_std_btc']:.2f} BTC | {r['return_on_inventory_risk']:.3f} "
            f"| {lat_str} | ${r['net_pnl_usd']:+,.0f} "
            f"| **{r['risk_adjusted_institutional_quality']:.2f}** |"
        )

    md += [
        "", "---", "",
        "## Metric Definitions", "",
        "| Metric | Formula / Meaning |", "|:---|:---|",
        "| **Edge (bps/fill)** | Net PnL (bps) / Executed Fills |",
        "| **Flow Purity (% Safe)** | 100% - Adverse Selection % |",
        "| **Return on Inv. Risk** | Edge (bps/fill) / sigma_Inv |",
        "| **Risk-Adj. Quality Score** | (DA% x Flow Purity% x Edge/fill) / (sigma_Inv x Latency Penalty) |",
    ]

    md_path = OUT_DIR / "README.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"[SAVED] {md_path}")

    print("\n" + "=" * 95)
    print("  TOP 15 by Risk-Adjusted Institutional Quality Score")
    print("=" * 95)
    print(f"{'Rank':<5} {'Model':<48} {'DA%':>6} {'Edge(bps)':>10} {'Purity%':>8} {'InvStd':>8} {'Net$M':>8} {'Q_Risk':>10}")
    print("-" * 95)
    for i, r in enumerate(results_sorted[:15], 1):
        print(f"{i:<5} {r['model_name']:<48} {r['true_da_non_zero_pct']:>6.1f} "
              f"{r['net_bps_per_executed_fill']:>10.4f} {r['flow_purity_pct']:>8.1f} {r['inventory_std_btc']:>8.2f} "
              f"{r['net_pnl_usd']/1e6:>8.2f} {r['risk_adjusted_institutional_quality']:>10.2f}")
    print("=" * 95)
    print("  COMPLETE!")
    print("=" * 95)


if __name__ == "__main__":
    main()
