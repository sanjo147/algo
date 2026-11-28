"""Strategy interface: generate_signals(df).

A strategy looks at closed bars and, for every bar t, says what to do at the
NEXT bar's open (the engine does the filling, never the strategy):

    entry     +1 = go long, -1 = go short, 0 = nothing      (only acted on when flat)
    exit      True = close the open position                 (only acted on when in a trade)
    stop      protective stop price for an entry (required with an entry)
    target_r  take-profit as a multiple of the initial risk (NaN = none)

Rule for every strategy: row t may only use rows 0..t. tests/test_strategies.py
checks this for every registered strategy by recomputing on truncated data.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from itertools import product
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from pydantic import BaseModel

from sanjotracker.config import BreakoutConfig, LevelsConfig

SIGNAL_COLUMNS = ["entry", "exit", "stop", "target_r"]


@dataclass(frozen=True)
class StrategyContext:
    symbol: str = ""
    timeframe: str = "1d"
    session_tz: str = "Asia/Kolkata"
    levels: LevelsConfig = field(default_factory=LevelsConfig)
    breakout: BreakoutConfig = field(default_factory=BreakoutConfig)


def empty_signals(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    return pd.DataFrame(
        {
            "entry": np.zeros(n, dtype=int),
            "exit": np.zeros(n, dtype=bool),
            "stop": np.full(n, np.nan),
            "target_r": np.full(n, np.nan),
        },
        index=df.index,
    )


class Strategy(ABC):
    """Subclass, set `name`, `Params`, `param_grid`, and implement generate_signals."""

    name: ClassVar[str] = "base"
    description: ClassVar[str] = ""
    timeframes: ClassVar[set[str]] = {"5m", "15m", "1h", "1d", "1w"}
    Params: ClassVar[type[BaseModel]]
    param_grid: ClassVar[dict[str, list[Any]]] = {}  # for walk-forward

    def __init__(self, **params: Any) -> None:
        self.params = self.Params(**params)

    def max_hold(self) -> int | None:
        """Bars after which the engine closes the trade (param `max_hold_bars`, if any)."""
        return getattr(self.params, "max_hold_bars", None)

    def supports(self, timeframe: str) -> bool:
        return timeframe in self.timeframes

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame, ctx: StrategyContext) -> pd.DataFrame:
        """Return SIGNAL_COLUMNS aligned with df (one row per bar)."""

    @classmethod
    def grid(cls) -> list[dict[str, Any]]:
        """Every parameter combination in param_grid (other params keep defaults)."""
        if not cls.param_grid:
            return [{}]
        keys = list(cls.param_grid)
        return [dict(zip(keys, vals, strict=True)) for vals in product(*cls.param_grid.values())]

    def label(self) -> str:
        changed = {
            k: v
            for k, v in self.params.model_dump().items()
            if v != self.Params.model_fields[k].default
        }
        return self.name + (
            f"({', '.join(f'{k}={v}' for k, v in changed.items())})" if changed else ""
        )


# ------------------------------------------------------------------ helpers for strategies


def local_times(df: pd.DataFrame, tz: str) -> pd.Series:
    """Bar open times in the session timezone (naive), aligned with df."""
    return pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(tz).dt.tz_localize(None)


def bars_into_session(local: pd.Series) -> pd.Series:
    """0 for the first bar of each local calendar day, 1 for the next, ..."""
    return local.groupby(local.dt.date).cumcount()


def minutes_of_day(local: pd.Series) -> pd.Series:
    return local.dt.hour * 60 + local.dt.minute
