"""RSI mean-reversion strategy.

Buy the dip in an uptrend: RSI < lower while close > EMA(trend_ema); exit when RSI
recovers above exit_level or after max_hold_bars. Short mirror: RSI > upper while
close < EMA(trend_ema). Stop = close -/+ atr_mult x ATR.
The trend filter matters: without it, mean reversion keeps buying a falling knife.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel

from sanjotracker.analysis.indicators import atr, ema, rsi
from sanjotracker.strategies.base import Strategy, StrategyContext, empty_signals


class RsiParams(BaseModel):
    rsi_n: int = 14
    lower: float = 30
    upper: float = 70
    exit_level: float = 50  # longs exit above this, shorts below (100 - exit_level)
    trend_ema: int = 200
    atr_mult: float = 2.0
    max_hold_bars: int | None = 10


class RsiMeanReversion(Strategy):
    name = "rsi_mean_reversion"
    description = "RSI oversold in uptrend (overbought in downtrend), exit on RSI recovery"
    Params = RsiParams
    param_grid = {"lower": [25, 30], "exit_level": [50, 60], "atr_mult": [2.0, 3.0]}

    def generate_signals(self, df: pd.DataFrame, ctx: StrategyContext) -> pd.DataFrame:
        p = self.params
        sig = empty_signals(df)
        c = df["close"]
        r = rsi(c, p.rsi_n)
        t = ema(c, p.trend_ema)
        a = atr(df)
        long_in = (r < p.lower) & (c > t)
        short_in = (r > p.upper) & (c < t)
        sig.loc[long_in, "entry"] = 1
        sig.loc[short_in, "entry"] = -1
        # exit when RSI is back to "normal" from either side
        sig["exit"] = ((r > p.exit_level) & (r.shift(1) <= p.exit_level)) | (
            (r < 100 - p.exit_level) & (r.shift(1) >= 100 - p.exit_level)
        )
        sig["stop"] = np.where(sig["entry"] == 1, c - p.atr_mult * a, np.nan)
        sig["stop"] = np.where(sig["entry"] == -1, c + p.atr_mult * a, sig["stop"])
        return sig
