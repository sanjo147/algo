"""Opening range breakout (NSE intraday).

Opening range (OR) = high/low of the first `or_minutes` after 09:15 IST. After the
OR's last bar has CLOSED, the first close above OR high goes long (below OR low
goes short); one trade per day, no new entries after `last_entry`. Stop = the
other side of the OR ("opposite") or its middle ("mid"); target = target_r x risk.
The engine squares off at the session close (MIS-style), so nothing is held overnight.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel

from sanjotracker.data.providers.base import TIMEFRAME_DELTAS
from sanjotracker.strategies.base import (
    Strategy,
    StrategyContext,
    empty_signals,
    local_times,
    minutes_of_day,
)

SESSION_OPEN_MIN = 9 * 60 + 15


class OrbParams(BaseModel):
    or_minutes: int = 15
    target_r: float = 2.0
    stop_at: Literal["opposite", "mid"] = "opposite"
    last_entry: str = "14:30"  # IST, by bar close


class OrbIntraday(Strategy):
    name = "orb_intraday"
    description = "First close outside the opening range, stop at the other side, EOD exit"
    timeframes = {"5m", "15m"}
    Params = OrbParams
    param_grid = {
        "or_minutes": [15, 30],
        "target_r": [1.5, 2.0, 3.0],
        "stop_at": ["opposite", "mid"],
    }

    def generate_signals(self, df: pd.DataFrame, ctx: StrategyContext) -> pd.DataFrame:
        p = self.params
        sig = empty_signals(df)
        if df.empty:
            return sig
        tf_min = int(TIMEFRAME_DELTAS[ctx.timeframe].total_seconds() // 60)
        local = local_times(df, "Asia/Kolkata")
        mins = minutes_of_day(local)
        day = local.dt.date
        or_end = SESSION_OPEN_MIN + p.or_minutes
        in_or = (mins >= SESSION_OPEN_MIN) & (mins + tf_min <= or_end)
        after_or = mins >= or_end

        # OR stats only from OR bars; a day needs ALL its OR bars to trade
        or_high = df["high"].where(in_or).groupby(day).transform("max")
        or_low = df["low"].where(in_or).groupby(day).transform("min")
        or_count = in_or.groupby(day).transform("sum")
        complete = or_count == max(1, p.or_minutes // tf_min)

        hh, mm = (int(x) for x in p.last_entry.split(":"))
        before_cutoff = mins + tf_min <= hh * 60 + mm
        c = df["close"]
        up = complete & after_or & before_cutoff & (c > or_high)
        dn = complete & after_or & before_cutoff & (c < or_low)
        trig = up | dn
        first = trig & (trig.astype(int).groupby(day).cumsum() == 1)  # one trade per day

        mid = (or_high + or_low) / 2
        long_stop = or_low if p.stop_at == "opposite" else mid
        short_stop = or_high if p.stop_at == "opposite" else mid
        sig.loc[first & up, "entry"] = 1
        sig.loc[first & dn, "entry"] = -1
        sig["stop"] = np.where(sig["entry"] == 1, long_stop, np.nan)
        sig["stop"] = np.where(sig["entry"] == -1, short_stop, sig["stop"])
        sig.loc[sig["entry"] != 0, "target_r"] = p.target_r
        return sig
