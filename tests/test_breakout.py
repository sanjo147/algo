"""Breakout / breakdown / retest / failure / approaching — and the no-lookahead proof."""

import numpy as np
import pandas as pd

from sanjotracker.analysis.breakout import (
    EventType,
    detect_events,
    last_bar_events,
    strong_zones,
)
from sanjotracker.analysis.levels import compute_levels
from sanjotracker.config import BreakoutConfig
from tests.conftest import T0, make_candles

N_RANGE = 200  # bars of sideways range before the action


def _range(n: int = N_RANGE, freq: str = "1h") -> pd.DataFrame:
    """Price swinging 99.5–110.5 (period 20): strong support ~99.5, resistance ~110.5."""
    close = 105 + 5 * np.sin(2 * np.pi * np.arange(n) / 20)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(T0, periods=n, freq=freq, tz="UTC"),
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(n, 100.0),
        }
    )


def _append(df: pd.DataFrame, bars: list[tuple[float, float, float, float, float]]) -> pd.DataFrame:
    """Append (open, high, low, close, volume) bars at the same frequency."""
    step = df["timestamp"].iloc[1] - df["timestamp"].iloc[0]
    ts = [df["timestamp"].iloc[-1] + step * (k + 1) for k in range(len(bars))]
    new = pd.DataFrame(bars, columns=["open", "high", "low", "close", "volume"])
    new.insert(0, "timestamp", ts)
    return pd.concat([df, new], ignore_index=True)


def _kinds(events) -> list[str]:  # noqa: ANN001
    return [e.event.value for e in events]


BREAKOUT_BAR = (104.0, 116.5, 103.5, 116.0, 300.0)
BREAKDOWN_BAR = (104.0, 104.5, 91.5, 92.0, 300.0)


def _last(df: pd.DataFrame, **kw) -> list:  # noqa: ANN003
    return last_bar_events(detect_events(df, "X", "1h", **kw), df)


# ------------------------------------------------------------------ breakouts


def test_breakout_through_resistance() -> None:
    df = _append(_range(), [BREAKOUT_BAR])
    ev = [e for e in _last(df) if e.event == EventType.BREAKOUT]
    assert len(ev) == 1
    e = ev[0]
    assert e.direction == "up"
    assert e.zone_low <= 110.5 <= e.zone_high + 1e-9
    assert e.volume_ratio == 3.0
    assert e.body_atr and e.body_atr > 0.5
    assert 0 <= e.confidence <= 100 and e.reasons


def test_breakdown_through_support() -> None:
    df = _append(_range(), [BREAKDOWN_BAR])
    ev = [e for e in _last(df) if e.event == EventType.BREAKDOWN]
    assert len(ev) == 1 and ev[0].direction == "down"
    assert ev[0].zone_low - 1e-9 <= 99.5 <= ev[0].zone_high


def test_no_breakout_without_volume() -> None:
    o, h, lo, c, _ = BREAKOUT_BAR
    df = _append(_range(), [(o, h, lo, c, 100.0)])  # 1.0x average volume
    assert "breakout" not in _kinds(_last(df))


def test_no_breakout_with_small_body() -> None:
    df = _append(_range(), [(115.8, 116.5, 115.5, 116.0, 300.0)])  # gap up, tiny body
    assert "breakout" not in _kinds(_last(df))


def test_thresholds_are_configurable() -> None:
    o, h, lo, c, _ = BREAKOUT_BAR
    df = _append(_range(), [(o, h, lo, c, 130.0)])  # 1.3x volume
    assert "breakout" not in _kinds(_last(df))
    assert "breakout" in _kinds(_last(df, cfg=BreakoutConfig(volume_mult=1.2)))


def test_zone_must_exist_before_the_bar() -> None:
    """The breakout bar's own high can't create the zone it breaks."""
    df = _append(_range(), [BREAKOUT_BAR])
    for e in _last(df):
        assert e.zone_high < BREAKOUT_BAR[1]


# ------------------------------------------------------------------ follow-ups


def test_failed_breakout_within_k_bars() -> None:
    df = _append(_range(), [BREAKOUT_BAR, (115.0, 115.5, 106.5, 107.0, 200.0)])
    ev = detect_events(df, "X", "1h")
    kinds = _kinds(ev)
    assert kinds.index("breakout") < kinds.index("failed_breakout")
    failed = next(e for e in ev if e.event == EventType.FAILED_BREAKOUT)
    assert failed.direction == "down"
    assert failed.bar_ts == df["timestamp"].iloc[-1]


def test_failed_breakdown() -> None:
    df = _append(_range(), [BREAKDOWN_BAR, (93.0, 103.5, 92.5, 103.0, 200.0)])
    assert "failed_breakdown" in _kinds(_last(df))


def test_retest_after_breakout() -> None:
    df = _append(_range(), [BREAKOUT_BAR])
    bo = next(e for e in _last(df) if e.event == EventType.BREAKOUT)
    mid = (bo.zone_low + bo.zone_high) / 2
    df = _append(
        df,
        [
            (116.0, 117.0, 115.0, 116.5, 150.0),  # holds above
            (116.0, 116.2, mid, 114.0, 120.0),  # dips into the zone, closes above it
        ],
    )
    last = _last(df)
    rt = [e for e in last if e.event == EventType.RETEST]
    assert len(rt) == 1 and rt[0].direction == "up"
    assert rt[0].zone_low == bo.zone_low  # same (frozen) zone as the breakout


def test_no_retest_after_window_expires() -> None:
    cfg = BreakoutConfig(retest_within_bars=3)
    df = _append(_range(), [BREAKOUT_BAR])
    bo = next(e for e in _last(df, cfg=cfg) if e.event == EventType.BREAKOUT)
    hold = [(118.0, 119.0, 117.0, 118.0, 100.0)] * 4
    df = _append(df, [*hold, (118.0, 118.2, bo.zone_high - 0.1, 117.0, 100.0)])
    assert "retest" not in _kinds(_last(df, cfg=cfg))


# ------------------------------------------------------------------ approaching / MTF


def test_approaching_resistance() -> None:
    base = _range()
    zones = strong_zones(compute_levels(base, "X", "1h"), BreakoutConfig())
    r = min((z for z in zones if z.type == "R"), key=lambda z: z.low)
    df = _append(base, [(r.low - 0.6, r.low - 0.1, r.low - 0.7, r.low - 0.2, 100.0)])
    ap = [e for e in _last(df) if e.event == EventType.APPROACHING and e.direction == "up"]
    assert ap and ap[0].zone_low == r.low


def _daily_trend(slope: float) -> pd.DataFrame:
    d = make_candles(T0 - pd.Timedelta(days=90), 100, "1d", seed=3)
    d[["open", "high", "low", "close"]] += np.linspace(0, slope, 100)[:, None]
    return d


def test_mtf_agreement_raises_confidence() -> None:
    df = _append(_range(), [BREAKOUT_BAR])
    up = _last(df, trend_df=_daily_trend(+60), trend_tf="1d")
    down = _last(df, trend_df=_daily_trend(-60), trend_tf="1d")
    bu = next(e for e in up if e.event == EventType.BREAKOUT)
    bd = next(e for e in down if e.event == EventType.BREAKOUT)
    assert bu.trend_htf == "up" and bu.mtf_ok is True
    assert bd.trend_htf == "down" and bd.mtf_ok is False
    assert bu.confidence > bd.confidence


def test_zero_volume_series_still_works() -> None:
    df = _append(_range(), [BREAKOUT_BAR])
    df["volume"] = 0.0  # like ^NSEI
    bo = [e for e in _last(df) if e.event == EventType.BREAKOUT]
    assert bo and bo[0].volume_ratio is None and "no volume data" in bo[0].reasons


# ------------------------------------------------------------------ THE lookahead proof


def _eventful_series() -> pd.DataFrame:
    """Range, breakout, new range higher, breakdown back... plenty of events."""
    df = _range(160)
    df = _append(df, [BREAKOUT_BAR, (116.0, 117.0, 115.0, 116.5, 150.0)])
    up = _range(60)
    up[["open", "high", "low", "close"]] += 12
    df = _append(df, list(up[["open", "high", "low", "close", "volume"]].itertuples(False)))
    df = _append(df, [(116.0, 116.5, 96.5, 97.0, 400.0), (97.0, 108.0, 96.0, 107.5, 300.0)])
    rest = make_candles(T0, 40, seed=5, base=107)
    return _append(df, list(rest[["open", "high", "low", "close", "volume"]].itertuples(False)))


def _dump(events) -> list[dict]:  # noqa: ANN001
    return [e.model_dump() for e in events]


def test_no_lookahead_events() -> None:
    """THE breakout lookahead proof.

    1. Walking the history in one pass gives, for every bar t, exactly the events
       a live scan would have produced at t seeing only bars 0..t.
    2. Replacing every bar after t with random prices changes no event up to t.
    (The walk starts at bar 130; anything earlier can't affect bars >= 150 because
    a breakout is only followed for retest_within_bars.)
    """
    df = _eventful_series()
    ts = df["timestamp"]
    s0 = 130
    full = detect_events(df, "X", "1h", start=s0)
    assert {"breakout", "breakdown"} <= set(_kinds(full)), "test series should be eventful"

    for t in range(150, len(df), 13):
        cut = df.iloc[: t + 1]
        live = last_bar_events(detect_events(cut, "X", "1h"), cut)
        from_full = [e for e in full if e.bar_ts == ts.iloc[t]]
        assert _dump(live) == _dump(from_full), f"bar {t}"

    rng = np.random.default_rng(42)
    for t in (160, 190, 222, 254):  # breakout bar, mid-range, breakdown bar, late breakdown
        scrambled = df.iloc[: t + 6].copy()
        future = scrambled.index > t
        k = int(future.sum())
        mid = rng.uniform(80, 130, k)
        scrambled.loc[future, "open"] = mid
        scrambled.loc[future, "close"] = mid + rng.normal(0, 3, k)
        scrambled.loc[future, "high"] = scrambled.loc[future, ["open", "close"]].max(axis=1) + 1
        scrambled.loc[future, "low"] = scrambled.loc[future, ["open", "close"]].min(axis=1) - 1
        scrambled.loc[future, "volume"] = rng.uniform(0, 1000, k)
        got = detect_events(scrambled, "X", "1h", start=s0)
        assert _dump([e for e in got if e.bar_ts <= ts.iloc[t]]) == _dump(
            [e for e in full if e.bar_ts <= ts.iloc[t]]
        ), f"bar {t}"
