"""Trend vs range, volatility regime, and benchmark market regime.

- Trend: ADX(14) > 25 = trending, < 20 = ranging, in between = transition.
- Volatility: where today's ATR% sits among the last year's ATR% values
  (percentile rank) -> low / normal / high. ATR% (ATR / close) is used so a
  stock that doubled in price doesn't look "more volatile".
- Market: benchmark (NIFTY / BTC) daily close above its 200 EMA = bull, else bear.

Every function only uses bars that had closed by the time asked about.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel

from sanjotracker.analysis.indicators import adx, atr, ema
from sanjotracker.config import RegimeConfig
from sanjotracker.data.providers.base import TIMEFRAME_DELTAS

TrendRegime = Literal["trending", "ranging", "transition", "unknown"]
VolRegime = Literal["low", "normal", "high", "unknown"]
MarketRegime = Literal["bull", "bear", "unknown"]
Direction = Literal["up", "down", "flat", "unknown"]


class Regime(BaseModel):
    """Snapshot of the conditions at one bar (stored with every event)."""

    trend: TrendRegime = "unknown"
    adx: float | None = None
    volatility: VolRegime = "unknown"
    atr_pct_rank: float | None = None
    market: MarketRegime = "unknown"

    def label(self) -> str:
        return f"{self.trend}/{self.volatility} vol/{self.market} market"


# ------------------------------------------------------------------ per-bar series


def classify_trend(adx_value: float, cfg: RegimeConfig) -> TrendRegime:
    if not np.isfinite(adx_value):
        return "unknown"
    if adx_value > cfg.adx_trending:
        return "trending"
    if adx_value < cfg.adx_ranging:
        return "ranging"
    return "transition"


def atr_pct_rank(
    df: pd.DataFrame, cfg: RegimeConfig, atr_n: int = 14, min_bars: int = 50
) -> pd.Series:
    """Percentile (0–1) of each bar's ATR% within the trailing ~1 year (time-based window).

    A time window (not a bar count) works for both NSE (6.25h sessions) and 24/7 crypto.
    """
    ap = (atr(df, atr_n) / df["close"]).to_numpy()
    s = pd.Series(ap, index=pd.DatetimeIndex(pd.to_datetime(df["timestamp"], utc=True)))
    window = pd.Timedelta(days=round(cfg.vol_lookback_days * 365 / 252))

    def _rank(x: np.ndarray) -> float:
        if np.isnan(x[-1]):
            return np.nan
        v = x[~np.isnan(x)]
        return float((v <= x[-1]).mean())

    ranks = s.rolling(window, min_periods=min_bars).apply(_rank, raw=True)
    return pd.Series(ranks.to_numpy(), index=df.index)


def classify_volatility(rank: float, cfg: RegimeConfig) -> VolRegime:
    if not np.isfinite(rank):
        return "unknown"
    if rank <= cfg.vol_low_pct:
        return "low"
    if rank >= cfg.vol_high_pct:
        return "high"
    return "normal"


def regime_frame(df: pd.DataFrame, cfg: RegimeConfig | None = None) -> pd.DataFrame:
    """Columns adx, trend, atr_pct_rank, volatility for every bar (causal)."""
    cfg = cfg or RegimeConfig()
    a = adx(df, cfg.adx_period)["adx"]
    r = atr_pct_rank(df, cfg)
    return pd.DataFrame(
        {
            "adx": a,
            "trend": [classify_trend(v, cfg) for v in a],
            "atr_pct_rank": r,
            "volatility": [classify_volatility(v, cfg) for v in r],
        },
        index=df.index,
    )


# ------------------------------------------------------------------ "as of a time" helpers


def bars_closed_by(df: pd.DataFrame, timeframe: str, at: pd.Timestamp) -> pd.DataFrame:
    """Rows whose bar had fully closed at time `at` (open + bar length <= at).

    This is how a higher timeframe is used without lookahead: a 15m bar closing at
    11:00 may use the 1h bar 10:00–11:00, never the 11:00–12:00 one.
    """
    close_time = pd.to_datetime(df["timestamp"], utc=True) + TIMEFRAME_DELTAS[timeframe]
    return df[close_time <= pd.Timestamp(at)]


def trend_direction(
    df: pd.DataFrame, timeframe: str, at: pd.Timestamp, fast: int = 20, slow: int = 50
) -> Direction:
    """'up' if EMA(fast) > EMA(slow) on bars closed by `at`, 'down' if below."""
    d = bars_closed_by(df, timeframe, at)
    if len(d) < slow:
        return "unknown"
    f, s = float(ema(d["close"], fast).iloc[-1]), float(ema(d["close"], slow).iloc[-1])
    if f > s:
        return "up"
    if f < s:
        return "down"
    return "flat"


def market_regime(
    bench: pd.DataFrame | None, at: pd.Timestamp, ema_n: int = 200, timeframe: str = "1d"
) -> MarketRegime:
    """'bull' if the benchmark's last closed bar is above its EMA(ema_n)."""
    if bench is None or bench.empty:
        return "unknown"
    d = bars_closed_by(bench, timeframe, at)
    if len(d) < ema_n:
        return "unknown"
    return "bull" if d["close"].iloc[-1] > ema(d["close"], ema_n).iloc[-1] else "bear"


def regime_at(
    frame: pd.DataFrame,
    i: int,
    at: pd.Timestamp,
    bench: pd.DataFrame | None,
    cfg: RegimeConfig,
) -> Regime:
    """Regime for bar position i of a regime_frame, plus the benchmark at time `at`."""
    row = frame.iloc[i]
    return Regime(
        trend=row["trend"],
        adx=None if pd.isna(row["adx"]) else round(float(row["adx"]), 1),
        volatility=row["volatility"],
        atr_pct_rank=None if pd.isna(row["atr_pct_rank"]) else round(float(row["atr_pct_rank"]), 2),
        market=market_regime(bench, at, cfg.benchmark_ema),
    )


def current_regime(
    df: pd.DataFrame,
    timeframe: str,
    bench: pd.DataFrame | None = None,
    cfg: RegimeConfig | None = None,
) -> Regime:
    """Regime at the last bar of df."""
    cfg = cfg or RegimeConfig()
    if df.empty:
        return Regime()
    at = pd.Timestamp(df["timestamp"].iloc[-1]) + TIMEFRAME_DELTAS[timeframe]
    return regime_at(regime_frame(df, cfg), len(df) - 1, at, bench, cfg)
