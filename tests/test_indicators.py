"""Indicator correctness on tiny synthetic series + a causality check for all of them."""

import numpy as np
import pandas as pd
import pytest

from sanjotracker.analysis import indicators as ind
from tests.conftest import T0, make_candles


def _bars(close, spread: float = 1.0, start=T0, freq="1h") -> pd.DataFrame:  # noqa: ANN001
    close = np.asarray(close, dtype=float)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(start, periods=len(close), freq=freq, tz="UTC"),
            "open": close,
            "high": close + spread,
            "low": close - spread,
            "close": close,
            "volume": np.full(len(close), 100.0),
        }
    )


def test_ema_known_values() -> None:
    s = pd.Series([1.0, 2, 3, 4])
    e = ind.ema(s, 3)  # alpha = 0.5
    assert np.isnan(e.iloc[1])
    assert e.iloc[2] == pytest.approx(2.25)
    assert e.iloc[3] == pytest.approx(3.125)


def test_rsi_extremes_and_bounds() -> None:
    up = ind.rsi(pd.Series(np.arange(1.0, 40)), 14)
    down = ind.rsi(pd.Series(np.arange(40.0, 1, -1)), 14)
    assert up.iloc[-1] == pytest.approx(100)
    assert down.iloc[-1] == pytest.approx(0)
    r = ind.rsi(make_candles(T0, 300)["close"]).dropna()
    assert r.between(0, 100).all()


def test_atr_constant_range() -> None:
    df = _bars(np.full(50, 100.0), spread=1.0)  # every bar high-low = 2, no gaps
    assert ind.atr(df, 14).iloc[-1] == pytest.approx(2.0)


def test_adx_higher_in_trend_than_in_range() -> None:
    trend = _bars(np.linspace(100, 200, 120), spread=0.5)
    chop = _bars(100 + 3 * np.sin(np.arange(120) / 2), spread=0.5)
    a_trend = ind.adx(trend)["adx"].iloc[-1]
    a_chop = ind.adx(chop)["adx"].iloc[-1]
    assert a_trend > 25
    assert a_trend > a_chop


def test_vwap_resets_each_session() -> None:
    df = make_candles(pd.Timestamp("2026-11-10 03:45", tz="UTC"), 100, "15m")  # 25h: 2 UTC days
    v = ind.vwap(df, "UTC")
    typical = (df["high"] + df["low"] + df["close"]) / 3
    day = df["timestamp"].dt.date
    first_of_day = day.ne(day.shift())
    assert first_of_day.sum() == 2
    assert np.allclose(v[first_of_day], typical[first_of_day])


def test_vwap_zero_volume_falls_back_to_typical() -> None:
    df = make_candles(T0, 10)
    df["volume"] = 0.0
    assert ind.vwap(df).notna().all()


def test_bollinger_flat_series_has_zero_width() -> None:
    bb = ind.bollinger(pd.Series(np.full(30, 50.0)))
    assert bb["bb_width"].iloc[-1] == pytest.approx(0)


def test_add_indicators_columns() -> None:
    out = ind.add_indicators(make_candles(T0, 250), intraday=True)
    for c in ("ema_20", "ema_200", "rsi", "atr", "atr_pct", "adx", "vol_ratio", "bb_upper", "vwap"):
        assert c in out.columns


def test_indicators_are_causal() -> None:
    """Changing future bars must not change any indicator value up to bar t."""
    df = make_candles(T0, 300, seed=3)
    t = 200
    future = df.copy()
    future.loc[t + 1 :, ["open", "high", "low", "close"]] *= 3
    future.loc[t + 1 :, "volume"] *= 10
    a = ind.add_indicators(df, intraday=True).iloc[: t + 1]
    b = ind.add_indicators(future, intraday=True).iloc[: t + 1]
    pd.testing.assert_frame_equal(a, b)
