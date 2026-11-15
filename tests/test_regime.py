"""Trend / volatility / market regime."""

import numpy as np
import pandas as pd
import pytest

from sanjotracker.analysis.regime import (
    atr_pct_rank,
    bars_closed_by,
    classify_trend,
    classify_volatility,
    current_regime,
    market_regime,
    regime_frame,
    trend_direction,
)
from sanjotracker.config import RegimeConfig
from tests.conftest import T0, make_candles

CFG = RegimeConfig()


@pytest.mark.parametrize(
    ("adx", "label"),
    [
        (30, "trending"),
        (25.1, "trending"),
        (22, "transition"),
        (15, "ranging"),
        (np.nan, "unknown"),
    ],
)
def test_classify_trend(adx: float, label: str) -> None:
    assert classify_trend(adx, CFG) == label


@pytest.mark.parametrize(("rank", "label"), [(0.1, "low"), (0.5, "normal"), (0.9, "high")])
def test_classify_volatility(rank: float, label: str) -> None:
    assert classify_volatility(rank, CFG) == label


def test_atr_rank_high_after_volatility_spike() -> None:
    df = make_candles(T0, 300, "1d", seed=1)
    df.loc[280:, "high"] += 10  # last bars suddenly very wide
    r = atr_pct_rank(df, CFG)
    assert r.iloc[-1] == pytest.approx(1.0)
    assert regime_frame(df, CFG)["volatility"].iloc[-1] == "high"


def test_atr_rank_needs_history() -> None:
    assert atr_pct_rank(make_candles(T0, 40, "1d"), CFG).isna().all()


def test_regime_frame_is_causal() -> None:
    df = make_candles(T0, 300, "1d", seed=2)
    full = regime_frame(df, CFG)
    for t in (60, 150, 299):
        part = regime_frame(df.iloc[: t + 1], CFG)
        pd.testing.assert_frame_equal(part, full.iloc[: t + 1])


def test_bars_closed_by_excludes_forming_bar() -> None:
    h = make_candles(T0, 5, "1h")
    at = T0 + pd.Timedelta(hours=3)  # 1h bars opening 00,01,02 have closed; 03 has not
    assert len(bars_closed_by(h, "1h", at)) == 3


def test_trend_direction_ignores_unclosed_htf_bar() -> None:
    d = make_candles(T0, 80, "1d", seed=4)
    d[["open", "high", "low", "close"]] += np.linspace(0, 40, 80)[:, None]  # uptrend
    at = d["timestamp"].iloc[-2] + pd.Timedelta(days=1)  # last bar still forming
    before = trend_direction(d, "1d", at)
    crash = d.copy()
    crash.loc[crash.index[-1], ["open", "high", "low", "close"]] = 1.0
    assert before == "up"
    assert trend_direction(crash, "1d", at) == before


def test_trend_direction_unknown_with_little_data() -> None:
    d = make_candles(T0, 10, "1d")
    assert trend_direction(d, "1d", T0 + pd.Timedelta(days=30)) == "unknown"


def test_market_regime() -> None:
    bull = make_candles(T0, 260, "1d", seed=1)
    bull[["open", "high", "low", "close"]] += np.linspace(0, 100, 260)[:, None]
    bear = bull.copy()
    ohlc = ["open", "high", "low", "close"]
    bear[ohlc] = bear[ohlc].iloc[::-1].to_numpy()
    end = bull["timestamp"].iloc[-1] + pd.Timedelta(days=1)
    assert market_regime(bull, end) == "bull"
    assert market_regime(bear, end) == "bear"
    assert market_regime(bull.head(50), end) == "unknown"
    assert market_regime(None, end) == "unknown"


def test_current_regime_snapshot() -> None:
    df = make_candles(T0, 300, "1d", seed=6)
    r = current_regime(df, "1d", df, CFG)
    assert r.trend in ("trending", "ranging", "transition")
    assert r.volatility in ("low", "normal", "high")
    assert r.market in ("bull", "bear")
    assert "market" in r.label()
