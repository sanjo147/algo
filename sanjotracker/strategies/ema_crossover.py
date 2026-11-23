"""EMA crossover strategy.

Long when EMA(fast) crosses above EMA(slow), short (where allowed) when it crosses
below; the opposite cross exits (and reverses). Initial stop = close -/+ atr_mult x ATR.
Optional trend filter: only longs above EMA(trend_ema), only shorts below it.
A classic trend-follower: expect a low win rate with a few big winners.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel

from sanjotracker.analysis.indicators import atr, ema
from sanjotracker.strategies.base import Strategy, StrategyContext, empty_signals


class EmaCrossoverParams(BaseModel):
    fast: int = 20
    slow: int = 50
    atr_mult: float = 2.0
    trend_ema: int = 0  # 0 = no filter
    max_hold_bars: int | None = None


class EmaCrossover(Strategy):
    name = "ema_crossover"
    description = "EMA fast/slow cross, ATR stop, exit on opposite cross"
    Params = EmaCrossoverParams
    param_grid = {"fast": [10, 20], "slow": [50, 100], "atr_mult": [2.0, 3.0]}

    def generate_signals(self, df: pd.DataFrame, ctx: StrategyContext) -> pd.DataFrame:
        p = self.params
        sig = empty_signals(df)
        c = df["close"]
        f, s = ema(c, p.fast), ema(c, p.slow)
        valid = f.notna() & s.notna() & f.shift(1).notna() & s.shift(1).notna()
        above = f > s
        cross_up = valid & above & ~above.shift(1, fill_value=False)
        cross_dn = valid & ~above & above.shift(1, fill_value=False)
        if p.trend_ema:
            t = ema(c, p.trend_ema)
            long_ok, short_ok = c > t, c < t
        else:
            long_ok = short_ok = pd.Series(True, index=df.index)
        a = atr(df)
        sig.loc[cross_up & long_ok, "entry"] = 1
        sig.loc[cross_dn & short_ok, "entry"] = -1
        sig["exit"] = (cross_up | cross_dn).to_numpy()
        sig["stop"] = np.where(sig["entry"] == 1, c - p.atr_mult * a, np.nan)
        sig["stop"] = np.where(sig["entry"] == -1, c + p.atr_mult * a, sig["stop"])
        return sig
