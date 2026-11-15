"""Breakout / breakdown / retest / failed-breakout / approaching events.

How a bar t is judged (no lookahead):
- Zones come from compute_levels(bars 0..t-1): the zones that existed *before*
  bar t. (Using bar t's own data to draw the zone it then "breaks" is a classic
  backtest leak.)
- Bar t's close, body, volume and ATR are known at its close, so they may be used.
- The higher-timeframe trend only uses HTF bars that had closed by bar t's close.
- Any trade would happen at bar t+1's open (Phase 5); nothing here trades.

Events:
- BREAKOUT:  previous close at/below a resistance zone, close above it, volume >
  volume_mult x avg of the previous 20 bars, bullish body > min_body_atr x ATR.
  BREAKDOWN is the mirror image through support.
- FAILED_BREAKOUT / FAILED_BREAKDOWN: within failed_within_bars, a close back
  inside (or through) the broken zone.
- RETEST: after a breakout that has not failed, a bar dips back into the zone
  and closes on the breakout side (within retest_within_bars).
- APPROACHING: last close within approach_atr x ATR of a strong zone (or inside it).

Each event gets a 0–100 confidence built from listed reasons (additive points),
so you can always see *why* a score is high or low.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from sanjotracker.analysis.indicators import atr
from sanjotracker.analysis.levels import compute_levels
from sanjotracker.analysis.regime import (
    Direction,
    Regime,
    regime_at,
    regime_frame,
    trend_direction,
)
from sanjotracker.config import BreakoutConfig, LevelsConfig, RegimeConfig
from sanjotracker.data.providers.base import TIMEFRAME_DELTAS

ATR_N = 14


def fmt_price(x: float) -> str:
    """Enough decimals for both 64,150 (BTC) and 0.9123 (XRP)."""
    ax = abs(x)
    decimals = 2 if ax >= 10 else 3 if ax >= 1 else 4 if ax >= 0.01 else 6
    return f"{x:,.{decimals}f}"


class EventType(StrEnum):
    APPROACHING = "approaching"
    BREAKOUT = "breakout"
    BREAKDOWN = "breakdown"
    RETEST = "retest"
    FAILED_BREAKOUT = "failed_breakout"
    FAILED_BREAKDOWN = "failed_breakdown"


class Event(BaseModel):
    """One detected event on one closed bar. Shared by scanner, alerts and dashboard."""

    symbol: str
    timeframe: str
    event: EventType
    direction: Literal["up", "down"]  # which way price is expected to go next
    bar_ts: datetime  # OPEN time (UTC) of the bar the event happened on
    price: float  # that bar's close
    zone_low: float
    zone_high: float
    zone_type: Literal["S", "R"]
    zone_source: str
    zone_touches: int
    zone_strength: float
    volume_ratio: float | None = None
    body_atr: float | None = None
    atr: float
    trend_tf: str | None = None
    trend_htf: Direction = "unknown"
    mtf_ok: bool | None = None
    regime: Regime = Field(default_factory=Regime)
    confidence: int
    reasons: list[str]
    style: str | None = None

    @property
    def key(self) -> str:
        """Identity for cooldown/dedupe: same symbol + timeframe + event + zone."""
        mid = (self.zone_low + self.zone_high) / 2
        return f"{self.symbol}|{self.timeframe}|{self.event.value}|{mid:.6g}"

    @property
    def zone_text(self) -> str:
        return f"{fmt_price(self.zone_low)}–{fmt_price(self.zone_high)}"


@dataclass(frozen=True)
class Zone:
    low: float
    high: float
    type: str
    source: str
    touches: int
    strength: float

    @property
    def describe(self) -> str:
        what = f"{self.touches} touches" if self.source == "pivot" else self.source
        return f"zone {self.low:,.2f}–{self.high:,.2f} ({what}, strength {self.strength:.0f})"


@dataclass
class _Active:
    """A breakout/breakdown being watched for a retest or failure."""

    zone: Zone
    bar: int
    direction: str  # "up" for breakout, "down" for breakdown


@dataclass(frozen=True)
class _Ctx:
    """Everything the scorer may know about the context of bar i."""

    trend_tf: str | None
    trend: Direction
    regime: Regime


# ------------------------------------------------------------------ zone filter


def strong_zones(levels: pd.DataFrame, cfg: BreakoutConfig) -> list[Zone]:
    """Zones worth alerting on: multi-touch pivot zones and the major reference levels."""
    if levels.empty:
        return []
    is_pivot = levels["source"] == "pivot"
    keep = (
        is_pivot
        & (levels["touches"] >= cfg.min_zone_touches)
        & (levels["strength"] >= cfg.min_zone_strength)
    ) | (~is_pivot & (levels["strength"] >= cfg.reference_min_strength))
    return [
        Zone(
            float(r.zone_low),
            float(r.zone_high),
            str(r.type),
            str(r.source),
            int(r.touches),
            float(r.strength),
        )
        for r in levels[keep].itertuples()
    ]


# ------------------------------------------------------------------ scoring


def _clamp(points: float) -> int:
    return int(round(min(100.0, max(0.0, points))))


def _score(
    kind: EventType,
    direction: str,
    zone: Zone,
    ctx: _Ctx,
    cfg: BreakoutConfig,
    vol_ratio: float | None,
    body_atr: float | None,
    dist_atr: float | None = None,
) -> tuple[int, list[str], bool | None]:
    """Additive confidence: base + zone + volume + candle + MTF + market + regime."""
    base = {
        EventType.BREAKOUT: 40,
        EventType.BREAKDOWN: 40,
        EventType.RETEST: 45,
        EventType.FAILED_BREAKOUT: 40,
        EventType.FAILED_BREAKDOWN: 40,
        EventType.APPROACHING: 25,
    }[kind]
    pts = float(base)
    reasons = [zone.describe]
    pts += min(25.0, 0.25 * zone.strength)

    is_break = kind in (EventType.BREAKOUT, EventType.BREAKDOWN)
    if kind == EventType.APPROACHING:
        pass  # volume says little before the zone is actually tested
    elif vol_ratio is None:
        reasons.append("no volume data")
    else:
        reasons.append(f"volume {vol_ratio:.1f}x 20-bar avg")
        if is_break:
            pts += min(15.0, 5 + 10 * (vol_ratio - cfg.volume_mult))
        elif kind in (EventType.FAILED_BREAKOUT, EventType.FAILED_BREAKDOWN) and vol_ratio > 1:
            pts += 5
    if is_break and body_atr is not None:
        reasons.append(f"body {body_atr:.1f}x ATR")
        pts += min(10.0, 10 * (body_atr - cfg.min_body_atr))
    if dist_atr is not None:
        if dist_atr == 0:
            reasons.append("price inside the zone")
        else:
            reasons.append(f"{dist_atr:.2f}x ATR from the zone")
        pts += 10 * (1 - dist_atr / cfg.approach_atr) if cfg.approach_atr > 0 else 0

    mtf_ok: bool | None = None
    if kind != EventType.APPROACHING and ctx.trend_tf:
        if ctx.trend in ("up", "down"):
            mtf_ok = ctx.trend == direction
            if mtf_ok:
                pts += 15
                reasons.append(f"{ctx.trend_tf} trend {ctx.trend} agrees (EMA fast vs slow)")
            else:
                pts -= 20
                reasons.append(f"against {ctx.trend_tf} trend ({ctx.trend})")
        else:
            reasons.append(f"{ctx.trend_tf} trend {ctx.trend}")
    elif kind == EventType.APPROACHING and ctx.trend_tf and ctx.trend in ("up", "down"):
        reasons.append(f"{ctx.trend_tf} trend {ctx.trend}")

    m = ctx.regime.market
    if m != "unknown":
        agrees = (m == "bull") == (direction == "up")
        pts += 5 if agrees else -5
        reasons.append(f"market {m} (benchmark vs 200 EMA)")

    r = ctx.regime
    if is_break and r.volatility == "low":
        pts += 5
        reasons.append("breaking out of low volatility")
    if kind in (EventType.FAILED_BREAKOUT, EventType.FAILED_BREAKDOWN) and r.trend == "ranging":
        pts += 5
        reasons.append("ranging market (ADX low): false breaks are common")
    if kind == EventType.RETEST and r.trend == "trending":
        pts += 5
        reasons.append("trending market (ADX high)")
    return _clamp(pts), reasons, mtf_ok


# ------------------------------------------------------------------ detection


def _bar_close_time(ts: pd.Timestamp, timeframe: str) -> pd.Timestamp:
    return pd.Timestamp(ts) + TIMEFRAME_DELTAS[timeframe]


def detect_events(
    df: pd.DataFrame,
    symbol: str = "",
    timeframe: str = "1d",
    *,
    cfg: BreakoutConfig | None = None,
    levels_cfg: LevelsConfig | None = None,
    regime_cfg: RegimeConfig | None = None,
    session_tz: str = "UTC",
    trend_df: pd.DataFrame | None = None,
    trend_tf: str | None = None,
    benchmark_df: pd.DataFrame | None = None,
    start: int | None = None,
    style: str | None = None,
) -> list[Event]:
    """Events on bars `start`..last of df (df = closed bars only, oldest first).

    By default only the last (retest_within_bars + 1) bars are walked, which is
    exactly enough to know every retest/failure that can fire on the last bar.
    Pass start=0 to get the full event history (used by tests, charts, backtests).
    """
    cfg = cfg or BreakoutConfig()
    levels_cfg = levels_cfg or LevelsConfig()
    regime_cfg = regime_cfg or RegimeConfig()
    df = df.reset_index(drop=True)
    n = len(df)
    min_bars = max(levels_cfg.pivot_left + levels_cfg.pivot_right + 1, ATR_N + 1) + 1
    window = max(cfg.retest_within_bars, cfg.failed_within_bars) + 1
    first = max(min_bars, n - window if start is None else start)
    if first >= n:
        return []

    atr_s = atr(df, ATR_N).to_numpy()
    m = cfg.volume_ma_bars
    vol_ma = df["volume"].rolling(m, min_periods=m).mean().shift(1).to_numpy()  # prior m bars
    reg = regime_frame(df, regime_cfg)
    cols = ("open", "high", "low", "close", "volume")
    o, h, lo, c, v = (df[k].to_numpy(dtype=float) for k in cols)
    ts = pd.to_datetime(df["timestamp"], utc=True)

    def ctx_at(i: int) -> _Ctx:
        at = _bar_close_time(ts.iloc[i], timeframe)
        trend: Direction = "unknown"
        if trend_df is not None and trend_tf:
            trend = trend_direction(trend_df, trend_tf, at, cfg.trend_ema_fast, cfg.trend_ema_slow)
        regime = regime_at(reg, i, at, benchmark_df, regime_cfg)
        return _Ctx(trend_tf, trend, regime)

    def vol_ratio_at(i: int) -> float | None:
        if not np.isfinite(vol_ma[i]) or vol_ma[i] <= 0:
            return None  # indices like ^NSEI have no volume
        return round(float(v[i] / vol_ma[i]), 2)

    def make(kind: EventType, direction: str, zone: Zone, i: int, **extra) -> Event:  # noqa: ANN003
        ctx = ctx_at(i)
        vr = vol_ratio_at(i)
        body = extra.pop("body_atr", None)
        conf, reasons, mtf_ok = _score(kind, direction, zone, ctx, cfg, vr, body, **extra)
        return Event(
            symbol=symbol,
            timeframe=timeframe,
            event=kind,
            direction=direction,
            bar_ts=ts.iloc[i].to_pydatetime(),
            price=float(c[i]),
            zone_low=zone.low,
            zone_high=zone.high,
            zone_type=zone.type,
            zone_source=zone.source,
            zone_touches=zone.touches,
            zone_strength=zone.strength,
            volume_ratio=vr,
            body_atr=body,
            atr=float(atr_s[i]),
            trend_tf=trend_tf,
            trend_htf=ctx.trend,
            mtf_ok=mtf_ok,
            regime=ctx.regime,
            confidence=conf,
            reasons=reasons,
            style=style,
        )

    events: list[Event] = []
    active: list[_Active] = []
    zones: list[Zone] = []
    broken_now: set[Zone] = set()
    for i in range(first, n):
        levels = compute_levels(
            df.iloc[:i], symbol, timeframe, levels_cfg, session_tz=session_tz, atr_n=ATR_N
        )
        zones = strong_zones(levels, cfg)

        # 1) follow-ups on earlier breakouts (never on the breakout bar itself)
        still: list[_Active] = []
        for a in active:
            age = i - a.bar
            z = a.zone
            if a.direction == "up":
                back_in, held_after_dip = c[i] <= z.high, lo[i] <= z.high and c[i] > z.high
            else:
                back_in, held_after_dip = c[i] >= z.low, h[i] >= z.low and c[i] < z.low
            if back_in:
                if age <= cfg.failed_within_bars:
                    kind = (
                        EventType.FAILED_BREAKOUT
                        if a.direction == "up"
                        else EventType.FAILED_BREAKDOWN
                    )
                    events.append(make(kind, "down" if a.direction == "up" else "up", z, i))
                continue  # failed (or faded after K bars): stop watching
            if held_after_dip:
                events.append(make(EventType.RETEST, a.direction, z, i))
                continue
            if age < cfg.retest_within_bars:
                still.append(a)
        active = still

        # 2) new breakouts / breakdowns on bar i through zones that existed before it
        a_i = atr_s[i]
        broken_now = set()
        if not np.isfinite(a_i) or a_i <= 0:
            continue
        pc = c[i - 1]
        vr = vol_ratio_at(i)
        vol_ok = vr is None or vr > cfg.volume_mult
        body = (c[i] - o[i]) / a_i  # signed: + bullish, - bearish
        ups = [z for z in zones if z.type == "R" and pc <= z.high < c[i]]
        downs = [z for z in zones if z.type == "S" and pc >= z.low > c[i]]
        for cands, direction, kind, body_ok in (
            (ups, "up", EventType.BREAKOUT, body > cfg.min_body_atr),
            (downs, "down", EventType.BREAKDOWN, -body > cfg.min_body_atr),
        ):
            if not cands:
                continue
            broken_now.update(cands)
            if not (vol_ok and body_ok):
                continue
            z = max(cands, key=lambda q: q.strength)
            events.append(make(kind, direction, z, i, body_atr=round(abs(float(body)), 2)))
            active.append(_Active(z, i, direction))

    # 3) approaching: last bar only, nearest qualifying zone on each side
    i = n - 1
    a_i = atr_s[i]
    if np.isfinite(a_i) and a_i > 0:
        best: dict[str, tuple[float, Zone]] = {}
        for z in zones:
            if z in broken_now:
                continue
            if z.type == "R" and c[i] <= z.high:
                side, dist = "up", max(0.0, z.low - c[i])
            elif z.type == "S" and c[i] >= z.low:
                side, dist = "down", max(0.0, c[i] - z.high)
            else:
                continue
            d_atr = dist / a_i
            if d_atr <= cfg.approach_atr and (side not in best or d_atr < best[side][0]):
                best[side] = (d_atr, z)
        for side, (d_atr, z) in best.items():
            events.append(make(EventType.APPROACHING, side, z, i, dist_atr=round(d_atr, 2)))

    return events


def events_frame(events: list[Event]) -> pd.DataFrame:
    """Flat table of events (regime expanded), handy for printing and the dashboard."""
    if not events:
        return pd.DataFrame()
    rows = []
    for e in events:
        d = e.model_dump(exclude={"regime", "reasons"})
        d["event"] = e.event.value
        d.update({f"regime_{k}": val for k, val in e.regime.model_dump().items()})
        d["reasons"] = "; ".join(e.reasons)
        rows.append(d)
    out = pd.DataFrame(rows)
    out["bar_ts"] = pd.to_datetime(out["bar_ts"], utc=True)
    return out


def last_bar_events(events: list[Event], df: pd.DataFrame, lookback_bars: int = 1) -> list[Event]:
    """Keep events from the last `lookback_bars` bars of df."""
    if df.empty or not events:
        return []
    cutoff = pd.Timestamp(df["timestamp"].iloc[-min(lookback_bars, len(df))])
    return [e for e in events if pd.Timestamp(e.bar_ts) >= cutoff]
