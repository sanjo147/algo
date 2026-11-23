"""Support/resistance breakout strategy.

Same rules as the live scanner's BREAKOUT/BREAKDOWN (pivot zones only):
- zones built from pivots confirmed BEFORE bar t, using at most `lookback_bars`
  of history (the scanner also loads ~1500 bars);
- close crosses through a resistance (support) zone with >= min_touches and
  strength >= min_strength, volume > volume_mult x avg of the prior 20 bars,
  candle body > min_body_atr x ATR in the breakout direction.
Stop = other side of the zone -/+ stop_atr x ATR; target = target_r x risk.

Speed: the volume/body filters are vectorised first; zones are only rebuilt on
the (few) bars that pass them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel

from sanjotracker.analysis.indicators import atr
from sanjotracker.analysis.levels import find_pivots, zone_arrays
from sanjotracker.strategies.base import Strategy, StrategyContext, empty_signals


class SrBreakoutParams(BaseModel):
    min_touches: int = 2
    min_strength: float = 30
    volume_mult: float | None = None  # None -> config breakout.volume_mult
    min_body_atr: float | None = None  # None -> config breakout.min_body_atr
    stop_atr: float = 0.5
    target_r: float = 2.0
    lookback_bars: int = 1500
    max_hold_bars: int | None = 50


class SrBreakout(Strategy):
    name = "sr_breakout"
    description = "Close through a strong pivot S/R zone on volume, stop beyond the zone"
    timeframes = {"15m", "1h", "1d"}
    Params = SrBreakoutParams
    param_grid = {"min_touches": [2, 3], "stop_atr": [0.5, 1.0], "target_r": [1.5, 2.0, 3.0]}

    def generate_signals(self, df: pd.DataFrame, ctx: StrategyContext) -> pd.DataFrame:
        p = self.params
        bcfg, lcfg = ctx.breakout, ctx.levels
        vol_mult = p.volume_mult if p.volume_mult is not None else bcfg.volume_mult
        body_min = p.min_body_atr if p.min_body_atr is not None else bcfg.min_body_atr
        d = df.reset_index(drop=True)
        sig = empty_signals(d)
        n = len(d)
        if n < lcfg.pivot_left + lcfg.pivot_right + 20:
            sig.index = df.index
            return sig

        o, h, lo, c, v = (
            d[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close", "volume")
        )
        a = atr(d).to_numpy()
        m = bcfg.volume_ma_bars
        vma = d["volume"].rolling(m, min_periods=m).mean().shift(1).to_numpy()
        has_vol = np.isfinite(vma) & (vma > 0)
        vol_ok = ~has_vol | (v > vol_mult * np.where(has_vol, vma, 1.0))
        body = (c - o) / a
        cand_up = vol_ok & (body > body_min)
        cand_dn = vol_ok & (-body > body_min)
        cands = np.flatnonzero((cand_up | cand_dn) & np.isfinite(a))

        piv = find_pivots(d, lcfg.pivot_left, lcfg.pivot_right)
        if piv.empty:
            sig.index = df.index
            return sig
        conf = piv["confirm_idx"].to_numpy(dtype=int)
        pidx = piv["idx"].to_numpy(dtype=int)
        pprice = piv["price"].to_numpy(dtype=float)
        pvol = piv["volume"].to_numpy(dtype=float)
        cumvol = np.concatenate([[0.0], np.cumsum(v)])

        for i in cands:
            if i < 2 or not np.isfinite(a[i - 1]) or a[i - 1] <= 0:
                continue
            start = max(0, i - p.lookback_bars)
            mask = (conf <= i - 1) & (pidx >= start)
            if not mask.any():
                continue
            avg_vol = (cumvol[i] - cumvol[start]) / (i - start)
            z = zone_arrays(pprice[mask], pidx[mask], pvol[mask], i - 1, avg_vol, a[i - 1], lcfg)
            strength = 100 * z["score"] / z["score"].max()
            strong = (z["touches"] >= p.min_touches) & (strength >= p.min_strength)
            mid = (z["low"] + z["high"]) / 2
            pc = c[i - 1]
            is_r = mid > pc  # zone above the previous close = resistance
            up = strong & is_r & (pc <= z["high"]) & (z["high"] < c[i])
            dn = strong & ~is_r & (pc >= z["low"]) & (z["low"] > c[i])
            if cand_up[i] and up.any():
                j = np.flatnonzero(up)[np.argmax(strength[up])]
                sig.at[i, "entry"] = 1
                sig.at[i, "stop"] = z["low"][j] - p.stop_atr * a[i]
                sig.at[i, "target_r"] = p.target_r
            elif cand_dn[i] and dn.any():
                j = np.flatnonzero(dn)[np.argmax(strength[dn])]
                sig.at[i, "entry"] = -1
                sig.at[i, "stop"] = z["high"][j] + p.stop_atr * a[i]
                sig.at[i, "target_r"] = p.target_r
        sig.index = df.index
        return sig
