"""Abstract DataProvider interface shared by every market data source.

Contract for get_ohlcv():
- returns a DataFrame with columns OHLCV_COLUMNS, sorted, no duplicate timestamps
- `timestamp` is the bar OPEN time, tz-aware UTC
- may include the still-forming last bar; callers drop it with `drop_incomplete`
  (a bar is usable only once `bar_close_time(ts) <= now`, which is what keeps the
  rest of the system free of lookahead)
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import TypeVar

import pandas as pd
from loguru import logger

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

TIMEFRAME_DELTAS: dict[str, timedelta] = {
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "1h": timedelta(hours=1),
    "1d": timedelta(days=1),
    "1w": timedelta(weeks=1),
}

# How far back a first-time backfill goes (before provider limits are applied).
DEFAULT_BACKFILL: dict[str, timedelta] = {
    "5m": timedelta(days=59),
    "15m": timedelta(days=59),
    "1h": timedelta(days=729),
    "1d": timedelta(days=365 * 10),
    "1w": timedelta(days=365 * 15),
}

T = TypeVar("T")


class ProviderError(RuntimeError):
    """A provider failed to return data after retries."""


class UnsupportedTimeframeError(ValueError):
    """The provider cannot serve the requested timeframe."""


def utcnow() -> datetime:
    """Current time, tz-aware UTC (single place to monkeypatch in tests)."""
    return datetime.now(UTC)


def empty_ohlcv() -> pd.DataFrame:
    """An empty frame with the standard OHLCV schema."""
    df = pd.DataFrame({c: pd.Series(dtype="float64") for c in OHLCV_COLUMNS})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a raw frame to the standard schema: UTC timestamps, floats, sorted, deduped."""
    if df.empty:
        return empty_ohlcv()
    missing = set(OHLCV_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"OHLCV frame missing columns: {sorted(missing)}")
    out = df[OHLCV_COLUMNS].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    for c in OHLCV_COLUMNS[1:]:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    out = out.dropna(subset=["open", "high", "low", "close"])
    out["volume"] = out["volume"].fillna(0.0)
    out = out.drop_duplicates("timestamp", keep="last").sort_values("timestamp")
    return out.reset_index(drop=True)


def with_retries(
    fn: Callable[[], T], *, attempts: int = 4, base_delay: float = 2.0, what: str = "request"
) -> T:
    """Call fn with exponential backoff (2s, 4s, 8s...). Raises ProviderError at the end."""
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # providers raise many unrelated exception types
            last_exc = exc
            if i < attempts - 1:
                delay = base_delay * 2**i
                logger.warning(
                    "{} failed ({}); retry {}/{} in {:.0f}s", what, exc, i + 1, attempts - 1, delay
                )
                time.sleep(delay)
    raise ProviderError(f"{what} failed after {attempts} attempts: {last_exc}") from last_exc


class DataProvider(ABC):
    """Interface every data source implements. Nothing outside data/providers/
    should know which concrete provider (yfinance, ccxt, broker API) is in use."""

    name: str = "base"
    #: seconds to pause between consecutive symbol requests (free-API politeness)
    request_pause: float = 0.0

    @property
    @abstractmethod
    def timeframes(self) -> set[str]:
        """Timeframes this provider supports."""

    def max_history(self, timeframe: str) -> timedelta | None:
        """Oldest data available for a timeframe, as a lookback from now (None = unlimited)."""
        return None

    def check_timeframe(self, timeframe: str) -> None:
        if timeframe not in self.timeframes:
            raise UnsupportedTimeframeError(f"{self.name} does not support {timeframe!r}")

    def bar_close_time(self, ts: pd.Timestamp, timeframe: str) -> pd.Timestamp:
        """When the bar opening at `ts` is final. Default: open + timeframe length."""
        return ts + TIMEFRAME_DELTAS[timeframe]

    def drop_incomplete(
        self, df: pd.DataFrame, timeframe: str, now: datetime | None = None
    ) -> pd.DataFrame:
        """Remove bars that have not closed yet (the live, still-changing candle)."""
        if df.empty:
            return df
        now_ts = pd.Timestamp(now or utcnow())
        closes = df["timestamp"].map(lambda t: self.bar_close_time(t, timeframe))
        return df.loc[closes <= now_ts].reset_index(drop=True)

    @abstractmethod
    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """Candles with open time in [start, end). Standard schema (see module docstring)."""

    @abstractmethod
    def get_latest(self, symbol: str) -> float:
        """Last traded price (may be delayed for free sources)."""
