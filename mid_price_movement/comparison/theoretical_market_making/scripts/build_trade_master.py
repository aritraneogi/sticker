"""
build_trade_master.py
=========================
One-time extraction: load all BTC trades from December 2025 raw gz files,
assign sequential tick_idx (matching model predictions.parquet tick_idx),
and save as a fast-loading Parquet file.

Output: data/btc_trades_master.parquet
Columns:
  tick_idx   : int64   - sequential BTC trade index (matches predictions.parquet)
  time       : str     - ISO 8601 nanosecond timestamp
  px         : float64 - execution price (USD)
  sz         : float64 - execution size (BTC)
  side       : str     - 'B' = buy aggressor (price up), 'A' = sell aggressor (price down)
  side_sign  : int8    - +1 if B (we sold as MM), -1 if A (we bought as MM)
  notional   : float64 - px * sz (USD value of trade)
"""

import gzip
import json
import time
from pathlib import Path

import pandas as pd

RAW_TRADES_DIR = Path(__file__).resolve().parent.parent / "data" / "extracted" / "trades"
OUT_PATH       = Path(__file__).resolve().parent / "data" / "btc_trades_master.parquet"

def main():
    t0 = time.time()
    print("Building BTC trade master from raw gz files...", flush=True)

    records = []
    tick_idx = 0
    day_dirs = sorted(RAW_TRADES_DIR.iterdir())

    for day_dir in day_dirs:
        day = day_dir.name
        hour_files = sorted(day_dir.glob("*.gz"), key=lambda p: int(p.stem))
        print(f"  Day {day}: {len(hour_files)} hours", flush=True)

        for hf in hour_files:
            with gzip.open(hf, "rb") as f:
                for line in f:
                    rec = json.loads(line)
                    if rec.get("coin") != "BTC":
                        continue
                    side = rec["side"]
                    px   = float(rec["px"])
                    sz   = float(rec["sz"])
                    records.append((
                        tick_idx,
                        rec["time"],
                        px,
                        sz,
                        side,
                        1 if side == "B" else -1,   # side_sign
                        px * sz,                     # notional USD
                    ))
                    tick_idx += 1

    print(f"\nTotal BTC trades indexed: {tick_idx:,}", flush=True)

    df = pd.DataFrame(records, columns=[
        "tick_idx", "time", "px", "sz", "side", "side_sign", "notional"
    ])
    df["tick_idx"]  = df["tick_idx"].astype("int64")
    df["px"]        = df["px"].astype("float32")
    df["sz"]        = df["sz"].astype("float32")
    df["side_sign"] = df["side_sign"].astype("int8")
    df["notional"]  = df["notional"].astype("float32")

    df.to_parquet(OUT_PATH, index=False, compression="snappy")

    elapsed = time.time() - t0
    print(f"Saved: {OUT_PATH}  ({df.memory_usage(deep=True).sum()/1e6:.1f} MB)")
    print(f"Done in {elapsed:.1f}s")

if __name__ == "__main__":
    main()
