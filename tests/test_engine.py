"""Backtest engine: fills, stops, targets, sizing, square-off, costs — and no lookahead."""

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel

from sanjotracker.backtest.costs import NseCostModel, ZeroCostModel
from sanjotracker.backtest.engine import run_backtest
from sanjotracker.config import PROJECT_ROOT, load_config
from sanjotracker.strategies.base import Strategy, StrategyContext, empty_signals
from tests.conftest import T0, make_candles

DAILY = StrategyContext("X.NS", "1d")  # delivery: long only, whole shares, no EOD exit
ZERO = ZeroCostModel()


class _P(BaseModel):
    max_hold_bars: int | None = None


class Scripted(Strategy):
    """Emits exactly the signals it is given (by bar position)."""

    name = "scripted"
    Params = _P

    def __init__(self, rows: dict[int, dict], **params) -> None:  # noqa: ANN003
        super().__init__(**params)
        self.rows = rows

    def generate_signals(self, df: pd.DataFrame, ctx: StrategyContext) -> pd.DataFrame:
        sig = empty_signals(df)
        for i, vals in self.rows.items():
            if i < len(df):
                for k, v in vals.items():
                    sig.iat[i, sig.columns.get_loc(k)] = v
        return sig


def flat(n: int = 20, price: float = 100.0, freq: str = "1h") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(T0, periods=n, freq=freq, tz="UTC"),
            "open": price,
            "high": price + 1,
            "low": price - 1,
            "close": price,
            "volume": 1000.0,
        }
    )


def test_signal_fills_at_next_open() -> None:
    df = flat()
    df.loc[6, "open"] = 101.0
    res = run_backtest(df, Scripted({5: {"entry": 1, "stop": 90.0}}), DAILY, ZERO)
    t = res.trades.iloc[0]
    assert t["entry_ts"] == df["timestamp"].iloc[6]
    assert t["entry_price"] == 101.0
    assert t["signal_ts"] == df["timestamp"].iloc[5]


def test_position_size_risks_one_percent() -> None:
    res = run_backtest(flat(), Scripted({5: {"entry": 1, "stop": 95.0}}), DAILY, ZERO, risk_pct=1.0)
    t = res.trades.iloc[0]
    assert t["qty"] * (t["entry_price"] - t["stop"]) == pytest.approx(1000, rel=1e-4)


def test_no_leverage_cap() -> None:
    res = run_backtest(flat(), Scripted({5: {"entry": 1, "stop": 99.9}}), DAILY, ZERO)
    assert res.trades.iloc[0]["qty"] * 100 <= 100_000


def test_stop_and_gap_through_stop() -> None:
    df = flat()
    df.loc[8, "low"] = 94.0  # touches the 95 stop
    res = run_backtest(df, Scripted({5: {"entry": 1, "stop": 95.0}}), DAILY, ZERO)
    t = res.trades.iloc[0]
    assert (t["exit_reason"], t["exit_price"]) == ("stop", 95.0)
    assert t["r_multiple"] == pytest.approx(-1.0)

    df = flat()
    df.loc[8, ["open", "low"]] = [90.0, 89.0]  # gaps BELOW the stop: fill at the open
    t = run_backtest(df, Scripted({5: {"entry": 1, "stop": 95.0}}), DAILY, ZERO).trades.iloc[0]
    assert t["exit_price"] == 90.0 and t["r_multiple"] == pytest.approx(-2.0)


def test_target_and_stop_same_bar_assumes_stop() -> None:
    df = flat()
    df.loc[8, ["high", "low"]] = [120.0, 90.0]
    rows = {5: {"entry": 1, "stop": 95.0, "target_r": 2.0}}
    t = run_backtest(df, Scripted(rows), DAILY, ZERO).trades.iloc[0]
    assert t["exit_reason"] == "stop"

    df = flat()
    df.loc[8, "high"] = 112.0
    t = run_backtest(df, Scripted(rows), DAILY, ZERO).trades.iloc[0]
    assert (t["exit_reason"], t["exit_price"]) == ("target", 110.0)


def test_exit_signal_and_time_exit_fill_next_open() -> None:
    df = flat()
    df.loc[10, "open"] = 103.0
    t = run_backtest(
        df, Scripted({5: {"entry": 1, "stop": 90.0}, 9: {"exit": True}}), DAILY, ZERO
    ).trades.iloc[0]
    assert (t["exit_reason"], t["exit_ts"], t["exit_price"]) == (
        "signal",
        df["timestamp"].iloc[10],
        103.0,
    )
    t = run_backtest(
        df, Scripted({5: {"entry": 1, "stop": 90.0}}, max_hold_bars=3), DAILY, ZERO
    ).trades.iloc[0]
    assert t["exit_reason"] == "time" and t["bars_held"] == 3


def test_shorts_only_where_allowed() -> None:
    rows = {5: {"entry": -1, "stop": 105.0}}
    assert run_backtest(flat(), Scripted(rows), DAILY, ZERO).trades.empty  # delivery: no shorts
    nse15 = StrategyContext("X.NS", "15m")
    df = flat(20, freq="15min")
    df["timestamp"] = pd.date_range("2026-11-16 03:45", periods=20, freq="15min", tz="UTC")
    t = run_backtest(df, Scripted(rows), nse15, ZERO).trades.iloc[0]
    assert t["direction"] == "short"


def test_nse_intraday_square_off_at_session_close() -> None:
    day = pd.date_range("2026-11-16 03:45", periods=25, freq="15min", tz="UTC")  # 09:15..15:15
    nxt = pd.date_range("2026-11-17 03:45", periods=25, freq="15min", tz="UTC")
    df = flat(50, freq="15min")
    df["timestamp"] = day.append(nxt)
    ctx = StrategyContext("X.NS", "15m")
    rows = {10: {"entry": 1, "stop": 90.0}, 24: {"entry": 1, "stop": 90.0}}
    res = run_backtest(df, Scripted(rows), ctx, ZERO)
    assert len(res.trades) == 1  # the 15:15 (last-bar) signal is ignored
    t = res.trades.iloc[0]
    assert t["exit_reason"] == "eod" and t["exit_ts"] == day[-1]


def test_costs_reduce_pnl_exactly() -> None:
    df = flat()
    df.loc[10, "open"] = 110.0
    rows = {5: {"entry": 1, "stop": 90.0}, 9: {"exit": True}}
    costs = load_config(PROJECT_ROOT / "config.example.yaml").costs.nse_delivery
    cm = NseCostModel(costs, "nse_delivery")
    t = run_backtest(df, Scripted(rows), DAILY, cm).trades.iloc[0]
    buy, sell = 100 * 1.0005, 110 * 0.9995
    assert t["entry_price"] == pytest.approx(buy) and t["exit_price"] == pytest.approx(sell)
    assert t["qty"] == int(t["qty"])  # whole shares
    fees = cm.charges("buy", buy, t["qty"]) + cm.charges("sell", sell, t["qty"])
    assert t["fees"] == pytest.approx(fees)
    assert t["net_pnl"] == pytest.approx((sell - buy) * t["qty"] - fees)


def test_equity_matches_trades() -> None:
    df = make_candles(T0, 300, seed=4)
    rows = {i: {"entry": 1, "stop": float(df["close"].iloc[i] - 3)} for i in range(20, 280, 15)}
    rows.update({i + 7: {"exit": True} for i in range(20, 280, 15)})
    res = run_backtest(df, Scripted(rows), DAILY, ZERO)
    assert res.equity.iloc[-1] == pytest.approx(100_000 + res.trades["net_pnl"].sum())


def test_no_lookahead_engine() -> None:
    """Scrambling every bar after t changes no trade closed by t and no equity up to t."""
    df = make_candles(T0, 300, seed=9)
    rows = {
        i: {"entry": 1, "stop": float(df["low"].iloc[i] - 1), "target_r": 2.0}
        for i in range(20, 290, 9)
    }
    base = run_backtest(df, Scripted(rows), DAILY, ZERO)
    rng = np.random.default_rng(3)
    for t in (60, 150, 240):
        s = df.copy()
        fut = s.index > t
        mid = rng.uniform(50, 150, int(fut.sum()))
        s.loc[fut, ["open", "close"]] = np.c_[mid, mid + rng.normal(0, 5, len(mid))]
        s.loc[fut, "high"] = s.loc[fut, ["open", "close"]].max(axis=1) + 2
        s.loc[fut, "low"] = s.loc[fut, ["open", "close"]].min(axis=1) - 2
        other = run_backtest(s, Scripted(rows), DAILY, ZERO)
        cut = df["timestamp"].iloc[t]
        a = base.trades[base.trades["exit_ts"] <= cut].reset_index(drop=True)
        b = other.trades[other.trades["exit_ts"] <= cut].reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)
        pd.testing.assert_series_equal(base.equity.loc[:cut], other.equity.loc[:cut])


def test_end_cuts_data_before_the_strategy_sees_it() -> None:
    df = make_candles(T0, 200, seed=2)
    seen: list[int] = []

    class Spy(Scripted):
        def generate_signals(self, d: pd.DataFrame, ctx: StrategyContext) -> pd.DataFrame:
            seen.append(len(d))
            return super().generate_signals(d, ctx)

    run_backtest(df, Spy({}), DAILY, ZERO, start=50, end=120)
    assert seen == [120]
