"""
run_persistence.py
==================
Evaluates naive persistence baseline for Mid Price Movements (MPM):
  Predicts that the mid price movement over the next h trades will match the sign of the mid price movement over the previous h trades.

Saves persistence results to results/persistence_summary.json and results/persistence_metrics.parquet.
"""

import gzip
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, matthews_corrcoef
import time

SCRIPT_DIR = Path(__file__).resolve().parent
TRADES_DIR = SCRIPT_DIR / 'data' / 'extracted' / 'trades'
RESULTS_DIR = SCRIPT_DIR / 'results'

def get_mids_for_coin(coin):
    dates = sorted(d.name for d in TRADES_DIR.iterdir() if d.is_dir() and d.name.isdigit())
    
    mids = []
    coin_upper = coin.upper()
    
    c1 = f'"coin":"{coin_upper}"'.encode()
    c2 = f'"coin": "{coin_upper}"'.encode()
    
    curr_bid = None
    curr_ask = None
    
    t0 = time.time()
    for date_str in dates:
        for h in range(24):
            fpath = TRADES_DIR / date_str / f"{h}.gz"
            if not fpath.exists():
                continue
            with gzip.open(fpath, "rb") as f:
                for line in f:
                    if not line or (c1 not in line and c2 not in line):
                        continue
                    try:
                        t = json.loads(line)
                        if t.get("coin", "").upper() == coin_upper:
                            px = float(t["px"])
                            side = t["side"]
                            if side == 'B': curr_ask = px
                            else: curr_bid = px
                            
                            if curr_bid is not None and curr_ask is not None:
                                bid_eff = min(curr_bid, curr_ask) if curr_ask < curr_bid else curr_bid
                                ask_eff = max(curr_bid, curr_ask) if curr_ask < curr_bid else curr_ask
                                mid = (bid_eff + ask_eff) / 2.0
                            elif curr_bid is not None: mid = curr_bid
                            elif curr_ask is not None: mid = curr_ask
                            else: mid = px
                            
                            mids.append(mid)
                    except Exception:
                        pass
    t1 = time.time()
    print(f"Total {coin} trades: {len(mids):,} (parsed in {t1-t0:.1f}s)")
    return np.array(mids, dtype=np.float64)

horizons = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500]

def evaluate_persistence(coin="BTC"):
    print(f"\n--- {coin} Mid Price Movement Persistence Baseline ---")
    mids = get_mids_for_coin(coin)
    n = len(mids)
    
    results = {}
    rows = []
    
    print(f"{'h':>4} | {'Non-Zero DA':>13} | {'Overall DA':>13} | {'MCC':>10}")
    print('-' * 50)
    for h in horizons:
        if 2 * h >= n:
            continue
        
        past_delta   = mids[h : n-h] - mids[0 : n-2*h]
        future_delta = mids[2*h : n] - mids[h : n-h]
        
        actual = np.where(future_delta > 1e-8, 1, np.where(future_delta < -1e-8, -1, 0))
        pred   = np.where(past_delta > 1e-8, 1, np.where(past_delta < -1e-8, -1, -1))
        
        nz_mask = actual != 0
        if nz_mask.sum() > 0:
            da_nz = float(accuracy_score(actual[nz_mask], pred[nz_mask]))
            mcc_nz = float(matthews_corrcoef(actual[nz_mask], pred[nz_mask]))
        else:
            da_nz, mcc_nz = 0.0, 0.0
            
        da_all = float(accuracy_score(actual, pred))
        print(f"{h:>4} | {da_nz:>13.6f} | {da_all:>13.6f} | {mcc_nz:>10.6f}")
        
        res = {
            "h": h,
            "da_nz_persistence": da_nz,
            "da_all_persistence": da_all,
            "mcc_persistence": mcc_nz,
            "n_nz": int(nz_mask.sum()),
            "n_total": len(actual)
        }
        results[f"h{h}"] = res
        rows.append(res)
        
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "persistence_summary.json", "w") as f:
        json.dump({"coin": coin, "total_trades": n, "horizons": results}, f, indent=2)
        
    pd.DataFrame(rows).to_parquet(RESULTS_DIR / "persistence_metrics.parquet", index=False)
    print(f"\n[SAVE] Persistence results saved to {RESULTS_DIR / 'persistence_summary.json'} and {RESULTS_DIR / 'persistence_metrics.parquet'}")

if __name__ == "__main__":
    evaluate_persistence("BTC")
