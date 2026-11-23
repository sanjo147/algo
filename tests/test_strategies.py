"""Each strategy's rules, plus THE strategy lookahead proof for every registered one."""

import numpy as np
import pandas as pd
import pytest

from sanjotracker.backtest.costs import ZeroCostModel
from sanjotracker.backtest.engine import run_backtest
from sanjotracker.strategies import STRATEGIES, StrategyContext, get_strategy
from tests.conftest import T0, make_candles
from tests.test_breakout import BREAKOUT_BAR, _append, _eventful_series, _range


def nse_15m(days: int = 12, seed: int = 0) -> pd.DataFrame:
    """25 x 15m bars per weekday session (09:15–15:15 IST opens), random walk."""
    stamps = []
    d = pd.Timestamp("2026-11-02")  # Monday
    while len(stamps) < days * 25:
        if d.weekday() < 5:
            stamps += list(
                pd.date_range(d + pd.Timedelta("03:45:00"), periods=25, freq="15min", tz="UTC")
            )
        d += pd.Timedelta(days=1)
    df = make_candles(T0, len(stamps), "15m", seed=seed, base=1000)
    df["timestamp"] = pd.DatetimeIndex(stamps)
    return df


CRYPTO_1H = StrategyContext("X/USDT", "1h", "UTC", "crypto")
NSE_15M = StrategyContext("X.NS", "15m", "Asia/Kolkata", "nse")


def _data_for(name: str) -> tuple[pd.DataFrame, StrategyContext]:
    if name in ("orb_intraday", "vwap_pullback"):
        return nse_15m(16, seed=3), NSE_15M
    df = make_candles(T0, 600, "1h", seed=5)
    df[["open", "high", "low", "close"]] += 15 * np.sin(np.arange(600) / 40)[:, None]
    if name == "sr_breakout":
        df = _eventful_series()  # ranges, breakouts, breakdowns, a retest
    return df, CRYPTO_1H


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_no_lookahead_signals(name: str) -> None:
    """THE strategy lookahead proof: signals at bar t are identical whether the
    strategy sees bars 0..t or the whole history.

    Checked bars: every bar where the full-history run signals, the bars just
    before/after it (a leak usually shifts a signal by a bar), plus a random sample.
    """
    df, ctx = _data_for(name)
    strat = get_strategy(name)
    full = strat.generate_signals(df, ctx)
    assert (full["entry"] != 0).any(), f"{name}: test data should produce signals"
    hits = np.flatnonzero((full["entry"] != 0).to_numpy() | full["exit"].to_numpy())
    near = {int(i + d) for i in hits for d in (-2, -1, 0, 1)}
    sample = np.random.default_rng(0).choice(len(df), size=40, replace=False)
    for t in sorted(t for t in near | set(sample.tolist()) if 30 <= t < len(df)):
        part = strat.generate_signals(df.iloc[: t + 1], ctx)
        pd.testing.assert_series_equal(
            part.iloc[t], full.iloc[t], check_names=False, obj=f"{name} bar {t}"
        )


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_signals_shape_and_stop_side(name: str) -> None:
    df, ctx = _data_for(name)
    sig = get_strategy(name).generate_signals(df, ctx)
    assert list(sig.columns) == ["entry", "exit", "stop", "target_r"] and len(sig) == len(df)
    e = sig["entry"]
    c = df["close"].to_numpy()
    assert (sig.loc[e == 1, "stop"].to_numpy() < c[(e == 1).to_numpy()]).all()
    assert (sig.loc[e == -1, "stop"].to_numpy() > c[(e == -1).to_numpy()]).all()


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_backtests_run_end_to_end(name: str) -> None:
    df, ctx = _data_for(name)
    res = run_backtest(df, get_strategy(name), ctx, ZeroCostModel())
    assert len(res.equity) == len(df)
    assert res.equity.iloc[-1] == pytest.approx(100_000 + res.trades["net_pnl"].sum())


def test_ema_crossover_enters_on_the_cross_bar() -> None:
    close = np.r_[np.linspace(100, 80, 80), np.linspace(80, 120, 80)]
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range(T0, periods=160, freq="1h", tz="UTC"),
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 1.0,
        }
    )
    sig = get_strategy("ema_crossover", fast=5, slow=20).generate_signals(df, CRYPTO_1H)
    longs = np.flatnonzero(sig["entry"] == 1)
    assert len(longs) == 1 and longs[0] > 80
    from sanjotracker.analysis.indicators import ema

    f, s = ema(df["close"], 5), ema(df["close"], 20)
    i = longs[0]
    assert f.iloc[i] > s.iloc[i] and f.iloc[i - 1] <= s.iloc[i - 1]


def test_rsi_needs_uptrend_for_longs() -> None:
    df = make_candles(T0, 500, "1h", seed=11)
    df[["open", "high", "low", "close"]] -= np.linspace(0, 60, 500)[:, None]  # downtrend
    sig = get_strategy("rsi_mean_reversion").generate_signals(df, CRYPTO_1H)
    assert not (sig["entry"] == 1).any()


def test_sr_breakout_enters_on_breakout_bar() -> None:
    df = _append(_range(), [BREAKOUT_BAR])
    sig = get_strategy("sr_breakout").generate_signals(df, CRYPTO_1H)
    assert sig["entry"].iloc[-1] == 1
    assert sig["stop"].iloc[-1] < 110.5 and sig["target_r"].iloc[-1] == 2.0


def test_orb_one_trade_per_day_after_the_range() -> None:
    df = nse_15m(16, seed=3)
    sig = get_strategy("orb_intraday").generate_signals(df, NSE_15M)
    local = df["timestamp"].dt.tz_convert("Asia/Kolkata")
    entries = sig[sig["entry"] != 0]
    assert entries.groupby(local.dt.date[entries.index]).size().max() == 1
    assert (local[entries.index].dt.strftime("%H:%M") >= "09:30").all()
    for i in entries.index:  # stop = other side of the 09:15 bar's range
        day_first = df[local.dt.date == local[i].date()].iloc[0]
        want = day_first["low"] if sig.at[i, "entry"] == 1 else day_first["high"]
        assert sig.at[i, "stop"] == pytest.approx(want)


def test_vwap_skips_session_open() -> None:
    df = nse_15m(16, seed=3)
    sig = get_strategy("vwap_pullback").generate_signals(df, NSE_15M)
    local = df["timestamp"].dt.tz_convert("Asia/Kolkata")
    times = local[sig["entry"] != 0].dt.strftime("%H:%M")
    assert (times >= "09:45").all() and (times <= "14:30").all()


def test_grid_and_label() -> None:
    cls = STRATEGIES["ema_crossover"]
    assert len(cls.grid()) == 8
    assert get_strategy("ema_crossover", fast=10).label() == "ema_crossover(fast=10)"
    with pytest.raises(KeyError):
        get_strategy("nope")


def test_no_lookahead_sr_breakout_sensitive() -> None:
    """sr_breakout with every filter relaxed, so nearly any zone crossing is a signal
    and a zone built from not-yet-confirmed pivots would change some bar's output."""
    df = make_candles(T0, 180, "1h", seed=21)
    strat = get_strategy(
        "sr_breakout", min_touches=1, min_strength=0, volume_mult=0, min_body_atr=-99
    )
    full = strat.generate_signals(df, CRYPTO_1H)
    assert (full["entry"] != 0).sum() > 10
    for t in range(30, len(df)):
        part = strat.generate_signals(df.iloc[: t + 1], CRYPTO_1H)
        pd.testing.assert_series_equal(part.iloc[t], full.iloc[t], check_names=False)
