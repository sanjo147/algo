"""Shared fixtures: synthetic candles and a fake in-memory provider."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from sanjotracker.data.providers.base import (
    TIMEFRAME_DELTAS,
    DataProvider,
    empty_ohlcv,
)
from sanjotracker.data.store import CandleStore


def make_candles(
    start: datetime, n: int, timeframe: str = "1h", seed: int = 0, base: float = 100.0
) -> pd.DataFrame:
    """Random-walk OHLCV with valid high/low, `n` bars from `start`."""
    rng = np.random.default_rng(seed)
    step = TIMEFRAME_DELTAS[timeframe]
    close = base + rng.normal(0, 1, n).cumsum()
    open_ = np.r_[base, close[:-1]]
    high = np.maximum(open_, close) + rng.uniform(0, 1, n)
    low = np.minimum(open_, close) - rng.uniform(0, 1, n)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(start, periods=n, freq=step, tz="UTC"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.integers(1_000, 10_000, n).astype(float),
        }
    )


class FakeProvider(DataProvider):
    """Serves a fixed frame, honouring start/end; records every call."""

    name = "fake"

    def __init__(self, data: pd.DataFrame, history: timedelta | None = None) -> None:
        self.data = data
        self.history = history
        self.calls: list[tuple[str, str, datetime | None, datetime | None]] = []

    @property
    def timeframes(self) -> set[str]:
        return set(TIMEFRAME_DELTAS)

    def max_history(self, timeframe: str) -> timedelta | None:
        return self.history

    def get_ohlcv(self, symbol, timeframe, start=None, end=None):  # noqa: ANN001
        self.calls.append((symbol, timeframe, start, end))
        df = self.data
        if start is not None:
            df = df[df["timestamp"] >= pd.Timestamp(start)]
        if end is not None:
            df = df[df["timestamp"] < pd.Timestamp(end)]
        return df.reset_index(drop=True) if not df.empty else empty_ohlcv()

    def get_latest(self, symbol: str) -> float:
        return float(self.data["close"].iloc[-1])


T0 = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture
def store(tmp_path) -> CandleStore:  # noqa: ANN001
    return CandleStore(tmp_path / "test.db")
