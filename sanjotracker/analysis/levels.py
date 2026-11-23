"""Support / resistance zones from confirmed pivots, plus reference levels.

Pipeline (all "as of" the last bar given, so no future data is ever used):
1. Pivots: bar i is a swing high if its high is the highest of the `left` bars
   before it and strictly higher than the `right` bars after it. That can only be
   known once those `right` bars have closed -> a pivot at i is *confirmed* at
   bar i + right. The last `right` bars therefore never contain pivots.
2. Zones: pivot prices (highs and lows together — old resistance often becomes
   support) are sorted and grouped greedily so no zone is wider than
   zone_width_atr x ATR. (1-D DBSCAN would chain neighbours in a trend into one
   giant zone; the width cap prevents that.)
3. Strength = touches x recency weight x relative volume at the touches,
   then scaled 0–100 against the strongest zone of that symbol/timeframe.
4. Reference levels: previous day/week/month high-low, 52-week high-low,
   round numbers, optional volume-profile POC.
"""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pandas as pd

from sanjotracker.analysis.indicators import atr
from sanjotracker.config import LevelsConfig

LEVEL_COLUMNS = [
    "symbol",
    "timeframe",
    "zone_low",
    "zone_high",
    "type",
    "touches",
    "strength",
    "last_touch_ts",
    "source",
]

# Fixed 0–100 strengths for reference levels (pivot zones are scored from data).
REFERENCE_STRENGTH = {
    "prev_day": 40,
    "prev_week": 55,
    "prev_month": 65,
    "52w": 80,
    "round": 25,
    "poc": 50,
}

_PERIODS = [("D", "prev_day", "1d"), ("W", "prev_week", "1w"), ("M", "prev_month", None)]
_TF_ORDER = ["5m", "15m", "1h", "1d", "1w"]


def empty_levels() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in LEVEL_COLUMNS})


# --------------------------------------------------------------------- pivots


def find_pivots(df: pd.DataFrame, left: int = 5, right: int = 5) -> pd.DataFrame:
    """Confirmed swing highs/lows.

    Returns columns: idx, pivot_ts, confirm_idx, confirm_ts, price, kind ("high"/"low"), volume.
    `confirm_idx = idx + right` is the first bar at which the pivot is known.
    """
    cols = ["idx", "pivot_ts", "confirm_idx", "confirm_ts", "price", "kind", "volume"]
    if len(df) < left + right + 1:
        return pd.DataFrame(columns=cols)
    d = df.reset_index(drop=True)

    def _future(s: pd.Series, how: str) -> pd.Series:
        # value at i = max/min of s[i+1 .. i+right]; NaN when those bars don't exist yet
        rev = s[::-1].rolling(right, min_periods=right)
        agg = rev.max() if how == "max" else rev.min()
        return agg[::-1].shift(-1)

    h, lo = d["high"], d["low"]
    past_max = h.rolling(left + 1, min_periods=left + 1).max()
    past_min = lo.rolling(left + 1, min_periods=left + 1).min()
    # ">=" on the left and ">" on the right: of equal highs, only the last one counts
    is_high = (h >= past_max) & (h > _future(h, "max"))
    is_low = (lo <= past_min) & (lo < _future(lo, "min"))

    frames = []
    for mask, kind, price in ((is_high, "high", h), (is_low, "low", lo)):
        idx = np.flatnonzero(mask.to_numpy())
        if len(idx):
            frames.append(
                pd.DataFrame(
                    {
                        "idx": idx,
                        "pivot_ts": d["timestamp"].to_numpy()[idx],
                        "confirm_idx": idx + right,
                        "confirm_ts": d["timestamp"].to_numpy()[idx + right],
                        "price": price.to_numpy()[idx],
                        "kind": kind,
                        "volume": d["volume"].to_numpy()[idx],
                    }
                )
            )
    if not frames:
        return pd.DataFrame(columns=cols)
    out = pd.concat(frames).sort_values(["idx", "kind"]).reset_index(drop=True)
    out["pivot_ts"] = pd.to_datetime(out["pivot_ts"], utc=True)
    out["confirm_ts"] = pd.to_datetime(out["confirm_ts"], utc=True)
    return out[cols]


# --------------------------------------------------------------------- clustering


def cluster_prices(prices: np.ndarray, max_width: float) -> list[np.ndarray]:
    """Group prices so each group spans at most `max_width`. Returns index arrays."""
    if len(prices) == 0:
        return []
    order = np.argsort(prices, kind="stable")
    groups: list[list[int]] = [[order[0]]]
    start = prices[order[0]]
    for i in order[1:]:
        if prices[i] - start <= max_width:
            groups[-1].append(i)
        else:
            groups.append([i])
            start = prices[i]
    return [np.array(g) for g in groups]


def zone_arrays(
    prices: np.ndarray,
    idx: np.ndarray,
    volumes: np.ndarray,
    last_i: int,
    avg_vol: float,
    atr_now: float,
    cfg: LevelsConfig,
) -> dict[str, np.ndarray]:
    """Numpy core of the zone builder (fast enough to call on every backtest bar).

    Inputs are the confirmed pivots' price, bar index and volume. Returns arrays
    low, high, touches, score (raw), last_idx (newest pivot in the zone).
    """
    width = cfg.zone_width_atr * atr_now
    min_width = 0.2 * atr_now  # a single-pivot zone still needs some thickness
    groups = cluster_prices(prices, width)
    k = len(groups)
    out = {
        "low": np.empty(k),
        "high": np.empty(k),
        "touches": np.empty(k, dtype=int),
        "score": np.empty(k),
        "last_idx": np.empty(k, dtype=int),
    }
    for j, g in enumerate(groups):
        lo, hi = float(prices[g].min()), float(prices[g].max())
        if hi - lo < min_width:
            mid = (lo + hi) / 2
            lo, hi = mid - min_width / 2, mid + min_width / 2
        recency = float(np.mean(0.5 ** ((last_i - idx[g]) / cfg.recency_half_life_bars)))
        rel_vol = float(volumes[g].mean()) / avg_vol if avg_vol > 0 else 1.0
        out["low"][j], out["high"][j] = lo, hi
        out["touches"][j] = len(g)
        out["score"][j] = len(g) * recency * rel_vol
        out["last_idx"][j] = int(idx[g].max())
    return out


def _pivot_zones(
    df: pd.DataFrame, pivots: pd.DataFrame, cfg: LevelsConfig, atr_now: float
) -> pd.DataFrame:
    if pivots.empty or not np.isfinite(atr_now) or atr_now <= 0:
        return empty_levels()
    close = float(df["close"].iloc[-1])
    z = zone_arrays(
        pivots["price"].to_numpy(dtype=float),
        pivots["idx"].to_numpy(dtype=int),
        pivots["volume"].to_numpy(dtype=float),
        len(df) - 1,
        float(df["volume"].mean()),
        atr_now,
        cfg,
    )
    ts_by_idx = dict(zip(pivots["idx"].to_numpy(dtype=int), pivots["pivot_ts"], strict=True))
    zones = pd.DataFrame(
        {
            "zone_low": z["low"],
            "zone_high": z["high"],
            "type": np.where(close >= (z["low"] + z["high"]) / 2, "S", "R"),
            "touches": z["touches"],
            "score": z["score"],
            "last_touch_ts": [ts_by_idx[i] for i in z["last_idx"]],
            "source": "pivot",
        }
    )
    zones["strength"] = (100 * zones["score"] / zones["score"].max()).round(1)
    return zones


# --------------------------------------------------------------------- reference levels


def nice_step(x: float) -> float:
    """Smallest 1/2/5 x 10^k that is >= x (e.g. 27 -> 50, 13 -> 20)."""
    if x <= 0:
        return 1.0
    k = 10 ** math.floor(math.log10(x))
    for m in (1, 2, 5, 10):
        if m * k >= x:
            return m * k
    return 10 * k


def _ref(name: str, price: float, close: float, half: float, ts: pd.Timestamp | None) -> dict:
    return {
        "zone_low": price - half,
        "zone_high": price + half,
        "type": "S" if close >= price else "R",
        "touches": 0,
        "score": np.nan,
        "strength": float(REFERENCE_STRENGTH[name.removesuffix("_high").removesuffix("_low")]),
        "last_touch_ts": ts,
        "source": name,
    }


def reference_levels(
    df: pd.DataFrame,
    timeframe: str,
    atr_now: float,
    session_tz: str = "UTC",
    round_step_pct: float = 1.0,
    include_poc: bool = False,
    poc_bars: int = 200,
) -> pd.DataFrame:
    """Previous day/week/month high-low, 52-week high-low, round numbers, optional POC."""
    if df.empty or not np.isfinite(atr_now):
        return empty_levels()
    close = float(df["close"].iloc[-1])
    half = 0.1 * atr_now
    rows: list[dict] = []
    local = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(session_tz).dt.tz_localize(None)
    tf_rank = _TF_ORDER.index(timeframe) if timeframe in _TF_ORDER else 0

    for freq, name, min_tf in _PERIODS:
        # "previous day" makes no sense on weekly bars, etc.
        if min_tf is not None and tf_rank > _TF_ORDER.index(min_tf):
            continue
        periods = local.dt.to_period(freq)
        current = periods.iloc[-1]
        prev = periods[periods < current]
        if prev.empty:
            continue
        mask = periods == prev.max()
        seg = df.loc[mask.to_numpy()]
        rows.append(_ref(f"{name}_high", float(seg["high"].max()), close, half, None))
        rows.append(_ref(f"{name}_low", float(seg["low"].min()), close, half, None))

    last_ts = df["timestamp"].iloc[-1]
    if df["timestamp"].iloc[0] <= last_ts - pd.Timedelta(days=300):  # enough history to call it 52w
        yr = df[df["timestamp"] > last_ts - pd.Timedelta(days=365)]
        hi_i, lo_i = yr["high"].idxmax(), yr["low"].idxmin()
        rows.append(
            _ref("52w_high", float(yr.loc[hi_i, "high"]), close, half, yr.loc[hi_i, "timestamp"])
        )
        rows.append(
            _ref("52w_low", float(yr.loc[lo_i, "low"]), close, half, yr.loc[lo_i, "timestamp"])
        )

    step = nice_step(close * round_step_pct / 100)
    base = math.floor(close / step) * step
    for k in (-1, 0, 1, 2):
        lvl = base + k * step
        if lvl > 0:
            rows.append(_ref("round", lvl, close, half, None))

    if include_poc:
        tail = df.tail(poc_bars)
        if tail["volume"].sum() > 0:
            typical = (tail["high"] + tail["low"] + tail["close"]) / 3
            hist, edges = np.histogram(typical, bins=50, weights=tail["volume"])
            i = int(np.argmax(hist))
            rows.append(_ref("poc", float((edges[i] + edges[i + 1]) / 2), close, half, None))

    return pd.DataFrame(rows) if rows else empty_levels()


# --------------------------------------------------------------------- main entry


def compute_levels(
    df: pd.DataFrame,
    symbol: str = "",
    timeframe: str = "1d",
    cfg: LevelsConfig | None = None,
    *,
    asof: datetime | pd.Timestamp | None = None,
    session_tz: str = "UTC",
    atr_n: int = 14,
    include_reference: bool = True,
    include_poc: bool = False,
) -> pd.DataFrame:
    """S/R zones known at the close of bar `asof` (default: last bar in df).

    Bars after `asof` are dropped before anything is computed, so the result can
    never depend on future data. Returns LEVEL_COLUMNS (+ `score`), strongest first.
    """
    cfg = cfg or LevelsConfig()
    if asof is not None:
        df = df[df["timestamp"] <= pd.Timestamp(asof)]
    df = df.reset_index(drop=True)
    if len(df) < max(atr_n + 1, cfg.pivot_left + cfg.pivot_right + 1):
        return empty_levels()

    atr_now = float(atr(df, atr_n).iloc[-1])
    parts = [_pivot_zones(df, find_pivots(df, cfg.pivot_left, cfg.pivot_right), cfg, atr_now)]
    if include_reference:
        parts.append(
            reference_levels(
                df, timeframe, atr_now, session_tz, cfg.round_number_step_pct, include_poc
            )
        )
    parts = [p for p in parts if not p.empty]
    if not parts:
        return empty_levels()
    out = pd.concat(parts, ignore_index=True)
    out.insert(0, "symbol", symbol)
    out.insert(1, "timeframe", timeframe)
    out["last_touch_ts"] = pd.to_datetime(out["last_touch_ts"], utc=True)
    cols = [*LEVEL_COLUMNS, "score"]
    return out[cols].sort_values("strength", ascending=False, ignore_index=True)


def nearest_levels(levels: pd.DataFrame, price: float, n: int = 3) -> pd.DataFrame:
    """The n closest zones below (S) and above (R) a price."""
    if levels.empty:
        return levels
    below = levels[levels["zone_high"] <= price].sort_values("zone_high", ascending=False).head(n)
    above = levels[levels["zone_low"] >= price].sort_values("zone_low").head(n)
    return pd.concat([below, above], ignore_index=True)
