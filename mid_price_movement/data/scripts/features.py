"""
features.py
===========
Dense feature set construction from trade-event stream for Mid Price Movement (MPM) Prediction.

Each 'tick' is one trade. Features are computed from rolling windows over the trade and mid-price history.
All features are standardized incrementally via running mean/variance (Welford).
Output is float32, shape (56,).

Feature groups (56 total):
1.  Mid price momentum     : log-returns of Mid Price at windows [1,2,3,5,10,20,50] (7)
2.  Trade price momentum   : log-returns of Execution Price at windows [1,2,3,5,10,20,50] (7)
3.  Trade direction        : aggressor-side rolling rate at [1,2,5,10,20,50] (6)
4.  Mid Volatility         : rolling std of mid returns at [5,10,20,50] (4)
5.  Volume profile         : log-size (1)
6.  VWAP spread            : mid price distance from rolling VWAP [5,10,20,50] (4)
7.  Spread proxies         : half-spread, price position relative to mid (2)
8.  Inter-trade time       : log-dt, rolling mean dt, dt z-score (3)
9.  Mid velocity & accel   : first/second difference of log mid-price (2)
10. Volume imbalance       : cumulative volume imbalance (1)
11. Trade size ratios      : size vs rolling mean at [5,10,20] (3)
12. Signed volume          : buy_vol - sell_vol at [5,10,20,50] (4)
13. Order flow imbalance   : OFI = (buy_vol - sell_vol) / total_vol at [5,10,20,50] (4)
14. Autocorrelation proxy  : lagged direction product at lags [1,2,3,5] (4)
15. Price level features   : mid price rel to rolling high/low [20,50] (4)
"""

from __future__ import annotations

import numpy as np
from collections import deque
from typing import Optional, Tuple


class RollingBuffer:
    """Fixed-length circular buffer with O(1) append."""
    __slots__ = ("_buf", "_maxlen")

    def __init__(self, maxlen: int):
        self._buf = deque(maxlen=maxlen)
        self._maxlen = maxlen

    def append(self, v: float) -> None:
        self._buf.append(v)

    def to_array(self) -> np.ndarray:
        return np.array(self._buf, dtype=np.float64)

    def __len__(self) -> int:
        return len(self._buf)


class WelfordOnline:
    """Incremental mean and variance (Welford's algorithm)."""
    __slots__ = ("n", "mean", "M2")

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def var(self) -> float:
        return self.M2 / self.n if self.n > 1 else 1.0

    @property
    def std(self) -> float:
        return max(self.var ** 0.5, 1e-9)

    def normalize(self, x: float) -> float:
        return (x - self.mean) / self.std


# Windows used per feature group
_MID_MOM_WINDOWS   = [1, 2, 3, 5, 10, 20, 50]    # 7
_TRADE_MOM_WINDOWS = [1, 2, 3, 5, 10, 20, 50]    # 7
_DIR_WINDOWS       = [1, 2, 5, 10, 20, 50]        # 6
_VOL_STD_WINDOWS   = [5, 10, 20, 50]              # 4
_VWAP_WINDOWS      = [5, 10, 20, 50]              # 4
_SIZE_RATIO_WINS   = [5, 10, 20]                  # 3
_SIGNED_VOL_WINS   = [5, 10, 20, 50]              # 4
_OFI_WINDOWS       = [5, 10, 20, 50]              # 4
_LAG_AC_WINDOWS    = [1, 2, 3, 5]                 # 4
_HL_WINDOWS        = [20, 50]                      # 4
_MAX_WIN           = 50

_N_FEATURES = (
    len(_MID_MOM_WINDOWS)     # 7
    + len(_TRADE_MOM_WINDOWS)  # 7
    + len(_DIR_WINDOWS)        # 6
    + len(_VOL_STD_WINDOWS)    # 4
    + 1                        # log_size
    + len(_VWAP_WINDOWS)       # 4
    + 2                        # half_spread, px_to_mid_spread
    + 3                        # log_dt, mean_dt_ratio, dt_zscore
    + 2                        # mid_vel, mid_accel
    + 1                        # cum_imbalance
    + len(_SIZE_RATIO_WINS)    # 3
    + len(_SIGNED_VOL_WINS)    # 4
    + len(_OFI_WINDOWS)        # 4
    + len(_LAG_AC_WINDOWS)     # 4
    + 2 * len(_HL_WINDOWS)     # 4
)
# Total: 56


class FeatureEngine:
    """
    Stateful, tick-by-tick feature engine for Mid Price Movement prediction.

    Call `.update(px, sz, side, ts_ns)` for each trade.
    Returns (mid_price, feature_vector_or_None).
    """

    def __init__(self, min_trades: int = _MAX_WIN):
        self._min_trades = min_trades
        self._n_trades   = 0

        self._curr_bid = None
        self._curr_ask = None

        self._log_mid = RollingBuffer(_MAX_WIN + 2)
        self._log_px  = RollingBuffer(_MAX_WIN + 2)
        self._sz      = RollingBuffer(_MAX_WIN + 1)
        self._dir     = RollingBuffer(_MAX_WIN + 1)
        self._ts      = RollingBuffer(_MAX_WIN + 1)

        self._normalizers: list[WelfordOnline] = [WelfordOnline() for _ in range(_N_FEATURES)]

    @property
    def n_features(self) -> int:
        return _N_FEATURES

    @property
    def ready(self) -> bool:
        return self._n_trades >= self._min_trades

    def update(self, px: float, sz: float, side: str, ts_ns: int) -> Tuple[float, Optional[np.ndarray]]:
        """
        Ingest one trade tick.

        Returns (current_mid_price, float32_feature_vector_or_None).
        """
        if side == 'B':
            self._curr_ask = px
        else:
            self._curr_bid = px

        if self._curr_bid is not None and self._curr_ask is not None:
            bid_eff = min(self._curr_bid, self._curr_ask) if self._curr_ask < self._curr_bid else self._curr_bid
            ask_eff = max(self._curr_bid, self._curr_ask) if self._curr_ask < self._curr_bid else self._curr_ask
            mid = (bid_eff + ask_eff) / 2.0
        elif self._curr_bid is not None:
            mid = self._curr_bid
        elif self._curr_ask is not None:
            mid = self._curr_ask
        else:
            mid = px

        log_mid = np.log(mid) if mid > 0 else 0.0
        log_px  = np.log(px) if px > 0 else 0.0
        d       = 1.0 if side == 'B' else -1.0

        self._log_mid.append(log_mid)
        self._log_px.append(log_px)
        self._sz.append(sz)
        self._dir.append(d)
        self._ts.append(ts_ns)
        self._n_trades += 1

        if not self.ready:
            return mid, None

        raw = self._compute_raw()
        out = np.empty(_N_FEATURES, dtype=np.float32)
        for i, (v, norm) in enumerate(zip(raw, self._normalizers)):
            norm.update(v)
            out[i] = np.float32(np.clip(norm.normalize(v), -10.0, 10.0))

        return mid, out

    def features(self) -> Optional[np.ndarray]:
        """Return normalized float32 feature vector, or None if not ready."""
        if not self.ready:
            return None

        raw = self._compute_raw()
        out = np.empty(_N_FEATURES, dtype=np.float32)
        for i, (v, norm) in enumerate(zip(raw, self._normalizers)):
            norm.update(v)
            out[i] = np.float32(np.clip(norm.normalize(v), -10.0, 10.0))

        return out

    def _compute_raw(self) -> list[float]:
        raw = []
        log_mid_arr = self._log_mid.to_array()
        log_px_arr  = self._log_px.to_array()
        sz_arr      = self._sz.to_array()
        dir_arr     = self._dir.to_array()
        ts_arr      = self._ts.to_array()

        # ---- 1. Mid price momentum ----
        for w in _MID_MOM_WINDOWS:
            if len(log_mid_arr) > w:
                ret = log_mid_arr[-1] - log_mid_arr[-(w + 1)]
            else:
                ret = 0.0
            raw.append(ret)

        # ---- 2. Trade price momentum ----
        for w in _TRADE_MOM_WINDOWS:
            if len(log_px_arr) > w:
                ret = log_px_arr[-1] - log_px_arr[-(w + 1)]
            else:
                ret = 0.0
            raw.append(ret)

        # ---- 3. Direction rate (buy fraction) ----
        for w in _DIR_WINDOWS:
            slice_ = dir_arr[-w:] if len(dir_arr) >= w else dir_arr
            raw.append(float(np.mean(slice_ == 1.0) if len(slice_) else 0.0))

        # ---- 4. Volatility (rolling std of mid returns) ----
        for w in _VOL_STD_WINDOWS:
            if len(log_mid_arr) > w:
                rets = np.diff(log_mid_arr[-w - 1:])
                raw.append(float(np.std(rets)) if len(rets) > 1 else 0.0)
            else:
                raw.append(0.0)

        # ---- 5. Log size ----
        raw.append(float(np.log(sz_arr[-1] + 1e-12)))

        # ---- 6. VWAP spread relative to mid ----
        for w in _VWAP_WINDOWS:
            s  = sz_arr[-w:] if len(sz_arr) >= w else sz_arr
            lm = log_mid_arr[-w:] if len(log_mid_arr) >= w else log_mid_arr
            s  = s[-len(lm):]
            vwap_log = float(np.average(lm, weights=s + 1e-12)) if len(lm) else log_mid_arr[-1]
            raw.append(log_mid_arr[-1] - vwap_log)

        # ---- 7. Half-spread and trade distance to mid ----
        bid_eff = self._curr_bid if self._curr_bid is not None else np.exp(log_px_arr[-1])
        ask_eff = self._curr_ask if self._curr_ask is not None else np.exp(log_px_arr[-1])
        mid_eff = (bid_eff + ask_eff) / 2.0
        half_spread = (ask_eff - bid_eff) / (2.0 * mid_eff) if mid_eff > 0 else 0.0
        px_to_mid   = (np.exp(log_px_arr[-1]) - mid_eff) / mid_eff if mid_eff > 0 else 0.0
        raw.append(half_spread)
        raw.append(px_to_mid)

        # ---- 8. Inter-trade time features ----
        if len(ts_arr) >= 2:
            dt_ns   = max(float(ts_arr[-1] - ts_arr[-2]), 1.0)
            log_dt  = np.log(dt_ns)
        else:
            dt_ns, log_dt = 1e6, 0.0
        raw.append(log_dt)

        if len(ts_arr) >= 2:
            dts = np.diff(ts_arr[-min(len(ts_arr), _MAX_WIN):]).astype(np.float64)
            mean_dt = float(np.mean(dts)) if len(dts) else 1.0
            mean_dt = max(mean_dt, 1.0)
        else:
            mean_dt = 1.0
        raw.append(dt_ns / mean_dt)
        raw.append((dt_ns - mean_dt) / (np.std(dts) + 1.0) if len(ts_arr) >= 3 else 0.0)

        # ---- 9. Mid velocity & acceleration ----
        if len(log_mid_arr) >= 2:
            vel = log_mid_arr[-1] - log_mid_arr[-2]
        else:
            vel = 0.0
        if len(log_mid_arr) >= 3:
            vel_prev = log_mid_arr[-2] - log_mid_arr[-3]
            accel = vel - vel_prev
        else:
            accel = 0.0
        raw.append(vel)
        raw.append(accel)

        # ---- 10. Cumulative signed volume imbalance ----
        signed_vol = dir_arr * sz_arr[-len(dir_arr):]
        total_vol  = sz_arr[-len(dir_arr):].sum() + 1e-12
        raw.append(float(signed_vol.sum() / total_vol))

        # ---- 11. Size ratio vs rolling mean ----
        cur_sz = float(sz_arr[-1])
        for w in _SIZE_RATIO_WINS:
            s = sz_arr[-w:] if len(sz_arr) >= w else sz_arr
            raw.append(cur_sz / (float(np.mean(s)) + 1e-12))

        # ---- 12. Signed volume at windows ----
        for w in _SIGNED_VOL_WINS:
            d  = dir_arr[-w:] if len(dir_arr) >= w else dir_arr
            s  = sz_arr[-w:] if len(sz_arr) >= w else sz_arr
            s  = s[-len(d):]
            raw.append(float((d * s).sum()) / (s.sum() + 1e-12))

        # ---- 13. OFI at windows ----
        for w in _OFI_WINDOWS:
            d  = dir_arr[-w:] if len(dir_arr) >= w else dir_arr
            s  = sz_arr[-w:] if len(sz_arr) >= w else sz_arr
            s  = s[-len(d):]
            buy_vol  = float(((d > 0) * s).sum())
            sell_vol = float(((d < 0) * s).sum())
            total    = buy_vol + sell_vol + 1e-12
            raw.append((buy_vol - sell_vol) / total)

        # ---- 14. Lagged autocorrelation proxy ----
        for lag in _LAG_AC_WINDOWS:
            if len(dir_arr) > lag:
                raw.append(float(dir_arr[-1] * dir_arr[-(lag + 1)]))
            else:
                raw.append(0.0)

        # ---- 15. Mid Price rel to rolling high/low ----
        cur_mid = float(log_mid_arr[-1])
        for w in _HL_WINDOWS:
            lm = log_mid_arr[-w:] if len(log_mid_arr) >= w else log_mid_arr
            high = float(np.max(lm))
            low  = float(np.min(lm))
            rng  = high - low + 1e-12
            raw.append((cur_mid - low) / rng)   # position in [0,1]
            raw.append((cur_mid - high) / rng)  # negative: distance below high

        assert len(raw) == _N_FEATURES, f"Feature count mismatch: {len(raw)} vs {_N_FEATURES}"
        return raw


FEATURE_NAMES = (
    [f"mid_mom_{w}"      for w in _MID_MOM_WINDOWS]
    + [f"trade_mom_{w}"  for w in _TRADE_MOM_WINDOWS]
    + [f"dir_rate_{w}"   for w in _DIR_WINDOWS]
    + [f"vol_std_{w}"    for w in _VOL_STD_WINDOWS]
    + ["log_size"]
    + [f"vwap_spread_{w}" for w in _VWAP_WINDOWS]
    + ["half_spread", "px_to_mid_spread"]
    + ["log_dt", "dt_mean_ratio", "dt_zscore"]
    + ["mid_vel", "mid_accel"]
    + ["cum_imbalance"]
    + [f"size_ratio_{w}" for w in _SIZE_RATIO_WINS]
    + [f"signed_vol_{w}" for w in _SIGNED_VOL_WINS]
    + [f"ofi_{w}"        for w in _OFI_WINDOWS]
    + [f"lag_ac_{w}"     for w in _LAG_AC_WINDOWS]
    + [f"pos_in_range_{w}" for w in _HL_WINDOWS]
    + [f"dist_from_high_{w}" for w in _HL_WINDOWS]
)

assert len(FEATURE_NAMES) == _N_FEATURES
