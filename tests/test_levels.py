"""Pivots, zones and reference levels — including the no-lookahead proof."""

import numpy as np
import pandas as pd
import pytest

from sanjotracker.analysis.levels import (
    LEVEL_COLUMNS,
    cluster_prices,
    compute_levels,
    find_pivots,
    nearest_levels,
    nice_step,
)
from sanjotracker.config import LevelsConfig
from tests.conftest import T0, make_candles


def _from_highs_lows(highs, lows) -> pd.DataFrame:  # noqa: ANN001
    highs, lows = np.asarray(highs, float), np.asarray(lows, float)
    mid = (highs + lows) / 2
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(T0, periods=len(highs), freq="1h", tz="UTC"),
            "open": mid,
            "high": highs,
            "low": lows,
            "close": mid,
            "volume": np.full(len(highs), 100.0),
        }
    )


def _oscillating(n: int = 200, period: int = 20) -> pd.DataFrame:
    """Price swinging between ~99.5 and ~110.5: clear support and resistance."""
    close = 105 + 5 * np.sin(2 * np.pi * np.arange(n) / period)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(T0, periods=n, freq="1h", tz="UTC"),
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(n, 100.0),
        }
    )


# ------------------------------------------------------------------ pivots


def test_pivot_high_found_and_confirmed_after_right_bars() -> None:
    highs = [1, 2, 3, 10, 3, 2, 1, 1]
    df = _from_highs_lows(highs, np.array(highs) - 0.5)
    piv = find_pivots(df, left=2, right=2)
    ph = piv[piv["kind"] == "high"]
    assert list(ph["idx"]) == [3]
    assert list(ph["confirm_idx"]) == [5]


def test_pivot_not_visible_until_right_bars_close() -> None:
    highs = [1, 2, 3, 10, 3, 2, 1, 1]
    df = _from_highs_lows(highs, np.array(highs) - 0.5)
    # bars 0..4: only one bar after the peak -> not confirmed yet
    assert find_pivots(df.iloc[:5], 2, 2).query("kind == 'high'").empty
    # bars 0..5: two bars after the peak -> confirmed
    assert list(find_pivots(df.iloc[:6], 2, 2).query("kind == 'high'")["idx"]) == [3]


def test_equal_highs_give_one_pivot() -> None:
    highs = [1, 2, 10, 10, 2, 1, 1]
    df = _from_highs_lows(highs, np.array(highs) - 0.5)
    ph = find_pivots(df, 2, 2).query("kind == 'high'")
    assert list(ph["idx"]) == [3]


def test_no_lookahead_pivots_and_levels() -> None:
    """THE lookahead proof.

    For many cut-off bars t:
    1. pivots found on bars[0..t] == pivots from the full history whose
       confirm_idx <= t (the full-history detector never "leaks" a pivot early);
    2. levels as of t are identical whether computed on bars[0..t], on the full
       history with asof=t, or on a history whose future bars were scrambled.
    """
    cfg = LevelsConfig()
    df = make_candles(T0, 400, seed=7)
    full_pivots = find_pivots(df, cfg.pivot_left, cfg.pivot_right)

    rng = np.random.default_rng(99)

    for t in range(40, 400, 37):
        cut = df.iloc[: t + 1]

        p_cut = find_pivots(cut, cfg.pivot_left, cfg.pivot_right)
        p_full = full_pivots[full_pivots["confirm_idx"] <= t].reset_index(drop=True)
        pd.testing.assert_frame_equal(p_cut, p_full, check_dtype=False)

        scrambled = df.copy()  # same past, completely different future
        future = scrambled.index > t
        scrambled.loc[future, ["open", "high", "low", "close"]] = rng.uniform(
            50, 150, (int(future.sum()), 4)
        )
        asof = df["timestamp"].iloc[t]
        a = compute_levels(cut, "X", "1h", cfg)
        b = compute_levels(df, "X", "1h", cfg, asof=asof)
        c = compute_levels(scrambled, "X", "1h", cfg, asof=asof)
        pd.testing.assert_frame_equal(a, b)
        pd.testing.assert_frame_equal(a, c)


# ------------------------------------------------------------------ zones


def test_cluster_prices_respects_width() -> None:
    groups = cluster_prices(np.array([100.0, 110.0, 100.2, 99.9, 110.3]), max_width=0.5)
    sizes = sorted(len(g) for g in groups)
    assert sizes == [2, 3]


def test_zones_never_wider_than_cap_in_a_trend() -> None:
    cfg = LevelsConfig()
    df = make_candles(T0, 500, seed=11)
    df[["open", "high", "low", "close"]] += np.linspace(0, 80, 500)[:, None]  # steady uptrend
    lv = compute_levels(df, "X", "1h", cfg, include_reference=False)
    from sanjotracker.analysis.indicators import atr

    cap = cfg.zone_width_atr * atr(df).iloc[-1]
    assert ((lv["zone_high"] - lv["zone_low"]) <= cap + 1e-9).all()


def test_oscillating_market_finds_support_and_resistance() -> None:
    df = _oscillating()
    lv = compute_levels(df, "X", "1h", include_reference=False)
    assert set(LEVEL_COLUMNS) <= set(lv.columns)
    top2 = lv.head(2)
    r = top2[top2["type"] == "R"].iloc[0]
    s = top2[top2["type"] == "S"].iloc[0]
    assert r["zone_low"] <= 110.5 <= r["zone_high"]
    assert s["zone_low"] <= 99.5 <= s["zone_high"]
    assert r["touches"] >= 8 and s["touches"] >= 8
    assert lv["strength"].max() == 100


def test_more_touches_means_stronger() -> None:
    df = _oscillating()
    df.loc[150:, "high"] = np.minimum(df.loc[150:, "high"], 108)  # later peaks stop short of 110.5
    lv = compute_levels(df, "X", "1h", include_reference=False)
    z110 = lv[(lv["zone_low"] <= 110.5) & (lv["zone_high"] >= 110.5)].iloc[0]
    z108 = lv[(lv["zone_low"] <= 108) & (lv["zone_high"] >= 108)].iloc[0]
    assert z110["touches"] > z108["touches"]


def test_too_little_data_returns_empty() -> None:
    assert compute_levels(make_candles(T0, 8)).empty


# ------------------------------------------------------------------ reference levels


def test_previous_day_uses_session_timezone() -> None:
    # 15m bars from 2026-11-10 03:45 UTC (09:15 IST) over two NSE sessions
    day1 = pd.date_range("2026-11-10 03:45", periods=25, freq="15min", tz="UTC")
    day2 = pd.date_range("2026-11-11 03:45", periods=25, freq="15min", tz="UTC")
    ts = day1.append(day2)
    rng = np.random.default_rng(1)
    close = 100 + rng.normal(0, 1, len(ts)).cumsum()
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 100.0,
        }
    )
    lv = compute_levels(df, "X.NS", "15m", session_tz="Asia/Kolkata")
    pdh = lv.loc[lv["source"] == "prev_day_high"]
    pdl = lv.loc[lv["source"] == "prev_day_low"]
    mid_h = (pdh["zone_low"] + pdh["zone_high"]).iloc[0] / 2
    mid_l = (pdl["zone_low"] + pdl["zone_high"]).iloc[0] / 2
    assert mid_h == pytest.approx(df["high"].iloc[:25].max())
    assert mid_l == pytest.approx(df["low"].iloc[:25].min())


def test_52w_levels_need_enough_history() -> None:
    short = make_candles(T0, 100, "1d")
    long = make_candles(T0, 400, "1d")
    assert "52w_high" not in set(compute_levels(short, "X", "1d")["source"])
    lv = compute_levels(long, "X", "1d")
    hi = lv.loc[lv["source"] == "52w_high"]
    expected = long[long["timestamp"] > long["timestamp"].iloc[-1] - pd.Timedelta(days=365)]
    assert (hi["zone_low"] + hi["zone_high"]).iloc[0] / 2 == pytest.approx(expected["high"].max())


def test_round_numbers_bracket_price() -> None:
    df = make_candles(T0, 100, base=2850)
    lv = compute_levels(df, "X", "1h")
    rounds = (
        lv.loc[lv["source"] == "round", ["zone_low", "zone_high"]].sum(axis=1) / 2
    ).sort_values()
    price = df["close"].iloc[-1]
    assert (rounds < price).any() and (rounds > price).any()


@pytest.mark.parametrize(("x", "step"), [(27, 50), (13, 20), (0.8, 1), (100, 100), (4.2, 5)])
def test_nice_step(x: float, step: float) -> None:
    assert nice_step(x) == step


def test_nearest_levels_splits_above_and_below() -> None:
    lv = compute_levels(_oscillating(), "X", "1h", include_reference=False)
    near = nearest_levels(lv, 105.0, n=1)
    assert len(near) == 2
    assert near["zone_high"].iloc[0] <= 105 <= near["zone_low"].iloc[1]
