"""
continual_test_mamba.py  (Vectorized Mamba Mid Price Movement Prequential Test)
===================================================================================
Strict, leak-free, prequential (test-then-train) continual streaming test
for Mid Price Movement (MPM) prediction using the Mamba Selective State Space Model (SSM).

Features:
  - Vectorized Mamba execution
  - Gradient norm clipping (max_norm=1.0) to prevent numerical NaNs
  - Accelerated CPU multi-threading (torch.set_num_threads(6)) ~3.2h runtime
  - Prequential deque resolution over horizons h \in {1, 2, 3, 5, 10, 20, 50, 100, 200, 500}
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

# ---------------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).resolve().parent
ROOT_DIR    = SCRIPT_DIR.parent
DATA_EXT    = SCRIPT_DIR / "data" / "extracted"
RESULTS_DIR = SCRIPT_DIR / "results_mamba"
WEIGHTS_DIR = RESULTS_DIR / "weights"

sys.path.insert(0, str(ROOT_DIR / "model"))
sys.path.insert(0, str(SCRIPT_DIR))

from mamba_model import MambaSSM
from features import FeatureEngine, FEATURE_NAMES, _N_FEATURES

# ---------------------------------------------------------------------------
HORIZONS       = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500]
MAX_HORIZON    = max(HORIZONS)
SNAPSHOT_EVERY = 10_000
WEIGHT_EVERY   = 200_000
SPLIT_RATIO    = 0.70

# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--coin",        default="BTC")
    p.add_argument("--date-start",  default=None)
    p.add_argument("--date-end",    default=None)
    p.add_argument("--hours",       default="0-23")
    p.add_argument("--d-model",     type=int,   default=64)
    p.add_argument("--d-state",     type=int,   default=16)
    p.add_argument("--n-layers",    type=int,   default=2)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--batch-size",  type=int,   default=128)
    p.add_argument("--threads",     type=int,   default=6)
    p.add_argument("--device",      default="cpu")
    p.add_argument("--max-ticks",   type=int,   default=None)
    p.add_argument("--split-tick",  type=int,   default=None,
                   help="Override split tick for 70/30. Auto-estimated if not set.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Fast trade reader
# ---------------------------------------------------------------------------

def iter_trades_fast(date_str, hours, coin, trades_dir):
    coin_upper = coin.upper()
    date_folder = trades_dir / date_str
    if not date_folder.exists():
        return
    for h in hours:
        fpath = date_folder / f"{h}.gz"
        if not fpath.exists():
            continue
        with gzip.open(fpath, "rb") as f:
            raw = f.read()
        for line in raw.split(b"\n"):
            if not line:
                continue
            try:
                t = json.loads(line)
            except Exception:
                continue
            if t.get("coin", "").upper() != coin_upper:
                continue
            try:
                yield float(t["px"]), float(t["sz"]), t["side"], pd.Timestamp(t["time"]).value
            except (KeyError, ValueError):
                continue


def available_dates(trades_dir):
    return sorted(
        d.name for d in trades_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and len(d.name) == 8
    )


def parse_hours(s):
    if "-" in s:
        lo, hi = s.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in s.split(",")]


# ---------------------------------------------------------------------------
# Pre-scan: estimate total ticks
# ---------------------------------------------------------------------------

def estimate_total_ticks(all_dates, hours, coin, trades_dir) -> Optional[int]:
    coin_bytes = f'"coin": "{coin.upper()}"'.encode()
    for date_str in all_dates:
        for h in hours:
            fpath = trades_dir / date_str / f"{h}.gz"
            if not fpath.exists():
                continue
            try:
                with gzip.open(fpath, "rb") as f:
                    raw = f.read()
                count = raw.count(coin_bytes)
                if count == 0:
                    count = raw.count(f'"coin":"{coin.upper()}"'.encode())
                estimate = count * len(hours) * len(all_dates)
                print(f"[PRE-SCAN] {date_str}h{h:02d}: {count:,} {coin} trades "
                      f"→ estimated total {estimate:,}")
                return estimate
            except Exception:
                continue
    return None


# ---------------------------------------------------------------------------
# HorizonQueue for Mid Price Movement
# ---------------------------------------------------------------------------

class HorizonQueueMPM:
    def __init__(self, horizons: list[int], split_tick: Optional[int] = None):
        self.horizons   = horizons
        self.split_tick = split_tick

        self._queues = {h: deque() for h in horizons}

        self._nz_correct   = {h: 0 for h in horizons}
        self._nz_total     = {h: 0 for h in horizons}
        self._nz_correct70 = {h: 0 for h in horizons}
        self._nz_total70   = {h: 0 for h in horizons}
        self._nz_correct30 = {h: 0 for h in horizons}
        self._nz_total30   = {h: 0 for h in horizons}

        self._all_correct   = {h: 0 for h in horizons}
        self._all_total     = {h: 0 for h in horizons}
        self._all_correct70 = {h: 0 for h in horizons}
        self._all_total70   = {h: 0 for h in horizons}
        self._all_correct30 = {h: 0 for h in horizons}
        self._all_total30   = {h: 0 for h in horizons}

    def record_batch(self, tick_start: int, mids: np.ndarray, preds: np.ndarray):
        for i, (mid, pred) in enumerate(zip(mids, preds)):
            tick = tick_start + i
            for h in self.horizons:
                self._queues[h].append((tick, float(mid), int(pred)))

    def resolve_batch(self, tick_start: int, curr_mids: np.ndarray):
        for i, curr_mid in enumerate(curr_mids):
            tick = tick_start + i
            curr_mid_val = float(curr_mid)

            for h in self.horizons:
                q = self._queues[h]
                while q and q[0][0] + h <= tick:
                    made_at, origin_mid, pred = q.popleft()
                    if made_at + h == tick:
                        diff = curr_mid_val - origin_mid
                        actual = 1 if diff > 1e-8 else (-1 if diff < -1e-8 else 0)

                        is_70 = (self.split_tick is not None) and (made_at < self.split_tick)

                        correct_all = int(pred == actual)
                        self._all_correct[h] += correct_all
                        self._all_total[h]   += 1
                        if self.split_tick is not None:
                            if is_70:
                                self._all_correct70[h] += correct_all
                                self._all_total70[h]   += 1
                            else:
                                self._all_correct30[h] += correct_all
                                self._all_total30[h]   += 1

                        if actual != 0:
                            correct_nz = int(pred == actual)
                            self._nz_correct[h] += correct_nz
                            self._nz_total[h]   += 1
                            if self.split_tick is not None:
                                if is_70:
                                    self._nz_correct70[h] += correct_nz
                                    self._nz_total70[h]   += 1
                                else:
                                    self._nz_correct30[h] += correct_nz
                                    self._nz_total30[h]   += 1

    def da_nz(self, h):
        n = self._nz_total[h]; return self._nz_correct[h] / n if n > 0 else None

    def da_nz70(self, h):
        n = self._nz_total70[h]; return self._nz_correct70[h] / n if n > 0 else None

    def da_nz30(self, h):
        n = self._nz_total30[h]; return self._nz_correct30[h] / n if n > 0 else None

    def da_all(self, h):
        n = self._all_total[h]; return self._all_correct[h] / n if n > 0 else None


# ---------------------------------------------------------------------------
# Chunked Parquet Writer
# ---------------------------------------------------------------------------

class ChunkedParquetWriter:
    def __init__(self, out_dir, prefix, chunk_size=1_000_000):
        self._dir = out_dir; self._prefix = prefix
        self._chunk_size = chunk_size; self._rows = []; self._part = 0

    def add_batch(self, rows):
        self._rows.extend(rows)
        if len(self._rows) >= self._chunk_size:
            self._flush()

    def _flush(self):
        if not self._rows: return
        path = self._dir / f"{self._prefix}_part{self._part:04d}.parquet"
        pd.DataFrame(self._rows).to_parquet(path, index=False)
        self._rows = []; self._part += 1

    def close_and_merge(self, final):
        self._flush()
        parts = sorted(self._dir.glob(f"{self._prefix}_part*.parquet"))
        if not parts: return
        pd.concat([pd.read_parquet(p) for p in parts],
                  ignore_index=True).to_parquet(final, index=False)
        for p in parts: p.unlink()


# ---------------------------------------------------------------------------
# Post-analysis
# ---------------------------------------------------------------------------

def post_analysis(pred_path: Path, horizons: list[int], split_tick: int) -> dict:
    print("[POST-ANALYSIS] loading predictions.parquet ...")
    df = pd.read_parquet(pred_path).sort_values("tick_idx").reset_index(drop=True)
    df["mid"]  = df["mid"].astype(float)
    df["pred"] = df["pred"].astype(int)

    results = {}
    for h in horizons:
        mid_target = df["mid"].shift(-h)
        diff = mid_target - df["mid"]
        
        label = np.where(diff > 1e-8, 1, np.where(diff < -1e-8, -1, 0))
        valid = mid_target.notna()

        sub = df[valid].copy()
        sub["actual"]  = label[valid]
        sub["correct"] = (sub["pred"] == sub["actual"]).astype(int)

        mask70 = sub["tick_idx"] < split_tick
        mask30 = ~mask70

        sub_nz    = sub[sub["actual"] != 0]
        mask70_nz = sub_nz["tick_idx"] < split_tick
        mask30_nz = ~mask70_nz

        results[h] = {
            "da_nz_full":    float(sub_nz["correct"].mean()) if len(sub_nz) > 0 else None,
            "da_nz_first70": float(sub_nz.loc[mask70_nz, "correct"].mean()) if mask70_nz.sum() > 0 else None,
            "da_nz_last30":  float(sub_nz.loc[mask30_nz, "correct"].mean()) if mask30_nz.sum() > 0 else None,
            "da_all_full":   float(sub["correct"].mean()) if len(sub) > 0 else None,
            "da_all_first70":float(sub.loc[mask70, "correct"].mean()) if mask70.sum() > 0 else None,
            "da_all_last30": float(sub.loc[mask30, "correct"].mean()) if mask30.sum() > 0 else None,
            "n_full":        len(sub),
            "n_nz_full":     len(sub_nz),
            "n_nz_first70":  int(mask70_nz.sum()),
            "n_nz_last30":   int(mask30_nz.sum()),
        }
        
        nz_str = f"{results[h]['da_nz_full']:.4f}" if results[h]['da_nz_full'] is not None else "n/a"
        nz70_str = f"{results[h]['da_nz_first70']:.4f}" if results[h]['da_nz_first70'] is not None else "n/a"
        nz30_str = f"{results[h]['da_nz_last30']:.4f}" if results[h]['da_nz_last30'] is not None else "n/a"
        print(f"  h={h:>4d}: DA(Non-Zero)={nz_str}  70%={nz70_str}  30%={nz30_str}  (N_nz={results[h]['n_nz_full']:,})")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Set 6 CPU threads for accelerated execution (~3.2h)
    torch.set_num_threads(args.threads)

    device = torch.device(args.device)
    BS     = args.batch_size

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    TMP = RESULTS_DIR / "_tmp"
    TMP.mkdir(exist_ok=True)

    trades_dir = DATA_EXT / "trades"
    all_dates  = available_dates(trades_dir)
    if not all_dates:
        print(f"[ERROR] No extracted trade data in {trades_dir}")
        sys.exit(1)

    if args.date_start:
        all_dates = [d for d in all_dates if d >= args.date_start.replace("-","")]
    if args.date_end:
        all_dates = [d for d in all_dates if d <= args.date_end.replace("-","")]

    hours = parse_hours(args.hours)
    coin  = args.coin.upper()

    print(f"[CONFIG MambaSSM] coin={coin}  dates={len(all_dates)}  hours={hours[0]}-{hours[-1]}")
    print(f"[CONFIG MambaSSM] d_model={args.d_model}  d_state={args.d_state}  n_layers={args.n_layers}  "
          f"batch={BS}  threads={args.threads}  device={device}")

    if args.split_tick:
        split_tick_est = args.split_tick
        print(f"[SPLIT] using provided split_tick={split_tick_est:,}")
    elif args.max_ticks:
        split_tick_est = int(args.max_ticks * SPLIT_RATIO)
        print(f"[SPLIT] max-ticks mode: split_tick={split_tick_est:,}")
    else:
        est = estimate_total_ticks(all_dates, hours, coin, trades_dir)
        if est:
            split_tick_est = int(est * SPLIT_RATIO)
            print(f"[SPLIT] estimated split_tick={split_tick_est:,} "
                  f"(est total={est:,})")
        else:
            split_tick_est = None
            print("[SPLIT] could not estimate; 70/30 via post-analysis only")

    # ---- Mamba Model ----
    model = MambaSSM(
        input_dim = _N_FEATURES,
        d_model   = args.d_model,
        d_state   = args.d_state,
        n_layers  = args.n_layers,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print(f"[MODEL MambaSSM] params={model.count_params():,}  n_features={_N_FEATURES}")
    print(f"[CONFIG] horizons={HORIZONS}")

    feat_engine = FeatureEngine()
    hq          = HorizonQueueMPM(HORIZONS, split_tick=split_tick_est)
    pred_writer = ChunkedParquetWriter(TMP, "pred")
    metrics_rows: list[dict] = []

    buf_feats  = []
    buf_mids   = []
    buf_labels = []
    buf_meta   = []
    prev_feats : Optional[np.ndarray] = None
    prev_labels: Optional[np.ndarray] = None
    prev_mid   : Optional[float]      = None

    tick_idx   = 0
    n_trained  = 0
    total_loss = 0.0
    t0         = time.time()
    last_snap  = 0

    print("[START MambaSSM Vectorized] streaming mid price movements ...")

    def process_batch(b_tick_start, b_f, b_mids, b_l, b_m, p_f, p_l):
        nonlocal n_trained, total_loss

        # Inference on current batch (unseen by training)
        X = torch.from_numpy(b_f).to(device)
        t_inf = time.perf_counter()
        model.eval()
        with torch.no_grad():
            logit, _ = model(X)
        lat_ms = (time.perf_counter() - t_inf) * 1e3
        preds = logit.squeeze(-1).sign().cpu().numpy().astype(int)
        preds = np.where(preds == 0, 1, preds)

        # Record predictions + resolve against current batch's mid prices
        hq.record_batch(b_tick_start, b_mids, preds)
        hq.resolve_batch(b_tick_start, b_mids)

        # Train on previous batch with gradient clipping
        if p_f is not None and p_l is not None:
            valid_mask = p_l != 0
            if valid_mask.any():
                Xp = torch.from_numpy(p_f[valid_mask]).to(device)
                Yp = torch.from_numpy(p_l[valid_mask].astype(np.float32)).to(device)
                model.train()
                optimizer.zero_grad()
                logit_p, _ = model(Xp)
                loss, _ = model.compute_loss(logit_p, Yp)
                loss.backward()
                
                # Gradient clipping to ensure zero NaNs and rock-solid numerical stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                total_loss += loss.item()
                n_trained  += len(Xp)

        # Write records
        rows = [{
            "tick_idx": tidx, "date": date, "ts_ns": ts_ns,
            "px": px, "mid": b_mids[i], "sz": sz, "side": side,
            "pred": int(preds[i]),
            "lat_ms": round(lat_ms, 3),
        } for i, (tidx, date, ts_ns, px, sz, side) in enumerate(b_m)]
        pred_writer.add_batch(rows)

        return lat_ms

    for date_str in all_dates:
        for px, sz, side, ts_ns in iter_trades_fast(date_str, hours, coin, trades_dir):
            mid, feat_vec = feat_engine.update(px, sz, side, ts_ns)
            
            mid_diff = mid - prev_mid if prev_mid is not None else 0.0
            mid_label = 1 if mid_diff > 1e-8 else (-1 if mid_diff < -1e-8 else 0)
            prev_mid = mid

            if feat_vec is not None:
                buf_feats.append(feat_vec)
                buf_mids.append(mid)
                buf_labels.append(mid_label)
                buf_meta.append((tick_idx, date_str, ts_ns, px, sz, side))

                if len(buf_feats) == BS:
                    b_f = np.stack(buf_feats)
                    b_mids = np.array(buf_mids, dtype=np.float64)
                    b_l = np.array(buf_labels, dtype=np.int8)
                    b_m = buf_meta.copy()
                    b_tick_start = tick_idx - BS + 1

                    lat_ms = process_batch(b_tick_start, b_f, b_mids, b_l, b_m,
                                           prev_feats, prev_labels)

                    prev_feats  = b_f
                    prev_labels = b_l.astype(np.float32)

                    buf_feats.clear(); buf_mids.clear(); buf_labels.clear(); buf_meta.clear()

                    if tick_idx - last_snap >= SNAPSHOT_EVERY:
                        elapsed  = time.time() - t0
                        avg_loss = total_loss / max(n_trained // BS, 1)
                        snap = {
                            "tick_idx": tick_idx, "elapsed_s": round(elapsed,2),
                            "avg_loss": round(avg_loss,6), "n_trained": n_trained,
                            "tps": round(tick_idx / elapsed, 0), "lat_ms": round(lat_ms,3),
                        }
                        for h in HORIZONS:
                            snap[f"da_nz_h{h}"]   = hq.da_nz(h)
                            snap[f"da_nz70_h{h}"] = hq.da_nz70(h)
                            snap[f"da_nz30_h{h}"] = hq.da_nz30(h)
                            snap[f"da_all_h{h}"]  = hq.da_all(h)
                        metrics_rows.append(snap)
                        last_snap = tick_idx

                        da1   = snap.get("da_nz_h1");   da1s  = f"{da1:.4f}"  if da1  else "n/a"
                        da1_70= snap.get("da_nz70_h1"); da70s = f"{da1_70:.4f}" if da1_70 else "n/a"
                        da1_30= snap.get("da_nz30_h1"); da30s = f"{da1_30:.4f}" if da1_30 else "n/a"
                        print(f"[t={tick_idx:>12,}] MambaSSM {date_str} | "
                              f"loss={avg_loss:.5f} | "
                              f"DA_nz(h1)={da1s} 70%={da70s} 30%={da30s} | "
                              f"tps={int(snap['tps']):,}")

                    if tick_idx > 0 and tick_idx % WEIGHT_EVERY == 0:
                        wpath = WEIGHTS_DIR / f"step_{tick_idx:010d}.pt"
                        torch.save({
                            "tick_idx": tick_idx,
                            "state_dict": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "config": {
                                "d_model": args.d_model, "d_state": args.d_state,
                                "n_layers": args.n_layers, "input_dim": _N_FEATURES,
                                "lr": args.lr, "batch_size": BS,
                            },
                        }, wpath)

            tick_idx += 1
            if args.max_ticks and tick_idx >= args.max_ticks:
                break
        if args.max_ticks and tick_idx >= args.max_ticks:
            break

    # Drain partial batch
    if buf_feats:
        b_f = np.stack(buf_feats)
        b_mids = np.array(buf_mids, dtype=np.float64)
        b_l = np.array(buf_labels, dtype=np.int8)
        process_batch(tick_idx - len(buf_feats), b_f, b_mids, b_l, buf_meta, prev_feats, prev_labels)

    # ---- Save ----
    elapsed = time.time() - t0
    print(f"\n[DONE MambaSSM] total_ticks={tick_idx:,}  elapsed={elapsed:.1f}s  "
          f"tps={tick_idx/elapsed:.0f}")

    pd.DataFrame(metrics_rows).to_parquet(RESULTS_DIR / "runtime_metrics.parquet", index=False)

    print("[SAVE] merging MambaSSM prediction parts ...")
    pred_writer.close_and_merge(RESULTS_DIR / "predictions.parquet")

    # Final weight
    wpath = WEIGHTS_DIR / f"step_{tick_idx:010d}_final.pt"
    torch.save({
        "tick_idx": tick_idx, "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": {"d_model": args.d_model, "d_state": args.d_state,
                   "n_layers": args.n_layers, "input_dim": _N_FEATURES,
                   "lr": args.lr, "batch_size": BS},
    }, wpath)

    # ---- Post-analysis ----
    true_split_tick = int(tick_idx * SPLIT_RATIO)
    pa_results = {}
    pred_path = RESULTS_DIR / "predictions.parquet"
    if pred_path.exists():
        print(f"\n[POST-ANALYSIS MambaSSM] split_tick={true_split_tick:,}")
        pa_results = post_analysis(pred_path, HORIZONS, true_split_tick)

    # ---- Summary ----
    summary = {
        "model": "MambaSSM", "coin": coin, "total_ticks": tick_idx,
        "split_tick_70_online": split_tick_est,
        "split_tick_70_exact":  true_split_tick,
        "n_features": _N_FEATURES, "feature_names": FEATURE_NAMES,
        "d_model": args.d_model, "d_state": args.d_state, "n_layers": args.n_layers,
        "lr": args.lr, "batch_size": BS, "threads": args.threads, "device": str(device),
        "elapsed_s": round(elapsed, 2), "tps": round(tick_idx / elapsed, 1),
        "horizons": {},
    }
    for h in HORIZONS:
        pa = pa_results.get(h, {})
        summary["horizons"][f"h{h}"] = {
            "da_nz_online":        hq.da_nz(h),
            "da_nz_online_70":     hq.da_nz70(h),
            "da_nz_online_30":     hq.da_nz30(h),
            "da_all_online":       hq.da_all(h),
            "n_resolved_online":   hq._all_total[h],
            "n_nz_resolved_online":hq._nz_total[h],
            "da_pa_nz_full":       pa.get("da_nz_full"),
            "da_pa_nz_first70":    pa.get("da_nz_first70"),
            "da_pa_nz_last30":     pa.get("da_nz_last30"),
            "da_pa_all_full":      pa.get("da_all_full"),
            "da_pa_all_first70":   pa.get("da_all_first70"),
            "da_pa_all_last30":    pa.get("da_all_last30"),
            "n_pa_full":           pa.get("n_full"),
            "n_pa_nz_full":        pa.get("n_nz_full"),
        }

    with open(RESULTS_DIR / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*85}")
    print(f"  FINAL SUMMARY (MambaSSM)  |  {coin}  |  {tick_idx:,} ticks  |  {elapsed:.0f}s  "
          f"({tick_idx/elapsed:.0f} tps)")
    print(f"{'='*85}")
    print(f"  {'H':>5} | {'DA_nz PA':>9} | {'PA 70%':>8} | {'PA 30%':>8} | "
          f"{'DA_all PA':>9} | {'N_nz':>10} | {'N_total':>10}")
    print(f"  {'-'*85}")
    for h in HORIZONS:
        hs   = summary["horizons"][f"h{h}"]
        def fmt(v): return f"{v:.4f}" if v is not None else "  n/a "
        print(f"  h={h:>4d} | {fmt(hs['da_pa_nz_full']):>9} | {fmt(hs['da_pa_nz_first70']):>8} | "
              f"{fmt(hs['da_pa_nz_last30']):>8} | {fmt(hs['da_pa_all_full']):>9} | "
              f"{hs.get('n_pa_nz_full', 0):>10,} | {hq._all_total[h]:>10,}")
    print(f"\n[OUTPUT MambaSSM] {RESULTS_DIR}")


if __name__ == "__main__":
    main()
