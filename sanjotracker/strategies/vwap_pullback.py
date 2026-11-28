"""VWAP pullback (intraday).

In an intraday uptrend (EMA fast > EMA slow, previous close above VWAP), a bar
that dips to/through VWAP but closes back above it, bullish, is a long entry.
Short mirror in a downtrend. Stop = below the dip (min(low, VWAP)) - stop_atr x ATR;
target = target_r x risk. The first `skip_bars` of each session are ignored
(VWAP is unstable right after the reset). NSE: no entries after `last_entry` IST.
VWAP resets at the start of each IST trading day.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel

from sanjotracker.analysis.indicators import atr, ema, vwap
from sanjotracker.data.providers.base import TIMEFRAME_DELTAS
from sanjotracker.strategies.base import (
    Strategy,
    StrategyContext,
    bars_into_session,
    empty_signals,
    local_times,
    minutes_of_day,
)


class VwapParams(BaseModel):
    ema_fast: int = 20
    ema_slow: int = 50
    stop_atr: float = 0.5
    target_r: float = 1.5
    skip_bars: int = 2
    last_entry: str = "14:45"  # IST, by bar close
    max_hold_bars: int | None = 24


class VwapPullback(Strategy):
    name = "vwap_pullback"
    description = "Trend + pullback to session VWAP that closes back on the trend side"
    timeframes = {"5m", "15m"}
    Params = VwapParams
    param_grid = {"stop_atr": [0.5, 1.0], "target_r": [1.0, 1.5, 2.0]}

    def generate_signals(self, df: pd.DataFrame, ctx: StrategyContext) -> pd.DataFrame:
        p = self.params
        sig = empty_signals(df)
        if df.empty:
            return sig
        c, o, h, lo = df["close"], df["open"], df["high"], df["low"]
        vw = vwap(df, ctx.session_tz)
        f, s = ema(c, p.ema_fast), ema(c, p.ema_slow)
        a = atr(df)
        local = local_times(df, ctx.session_tz)
        warm = bars_into_session(local) >= p.skip_bars
        # no new entries late in the NSE session
        tf_min = int(TIMEFRAME_DELTAS[ctx.timeframe].total_seconds() // 60)
        hh, mm = (int(x) for x in p.last_entry.split(":"))
        warm &= minutes_of_day(local) + tf_min <= hh * 60 + mm
        prev_above = c.shift(1) > vw.shift(1)
        prev_below = c.shift(1) < vw.shift(1)
        same_day = local.dt.date == local.dt.date.shift(1)  # yesterday's VWAP doesn't count
        long_in = warm & same_day & (f > s) & prev_above & (lo <= vw) & (c > vw) & (c > o)
        short_in = warm & same_day & (f < s) & prev_below & (h >= vw) & (c < vw) & (c < o)
        sig.loc[long_in, "entry"] = 1
        sig.loc[short_in, "entry"] = -1
        sig["stop"] = np.where(sig["entry"] == 1, np.minimum(lo, vw) - p.stop_atr * a, np.nan)
        sig["stop"] = np.where(sig["entry"] == -1, np.maximum(h, vw) + p.stop_atr * a, sig["stop"])
        sig.loc[sig["entry"] != 0, "target_r"] = p.target_r
        return sig
