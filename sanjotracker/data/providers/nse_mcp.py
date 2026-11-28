"""NSE's official MCP servers as a data provider (free, no login).

- Bhavcopy server (end of day, ~5 years): daily OHLCV via `get_stock_history`
  (max 3 months per call, paged backwards with `next_end_date`) and split/bonus
  factors via `get_corporate_actions`.
- CM Market Live server (~1-3 min delayed): `cm_get_stock_quote` for the latest price.

Prices from the Bhavcopy are RAW. Old prices are back-adjusted here for splits and
bonuses only (not dividends), the same convention the Yahoo provider uses, so the
two can back each other up. NSE has no intraday candles over MCP: 5m/15m/1h stay
with another provider. Equity symbols only (indices like ^NSEI are not in the
Bhavcopy; the routed provider falls back to Yahoo for them).

NSE's terms: informational/educational use, no commercial use, no AI training.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
from loguru import logger

from sanjotracker.data.providers.base import (
    DataProvider,
    ProviderError,
    empty_ohlcv,
    normalize_ohlcv,
    utcnow,
)
from sanjotracker.data.providers.mcp_client import McpClient, McpToolError
from sanjotracker.data.providers.yfinance_nse import SESSION_CLOSE

BHAVCOPY_URL = "https://mcp.nseindia.in/bhavcopy/cm/mcp"
LIVE_URL = "https://mcp.nseindia.in/cmmkt/mcp"
IST = "Asia/Kolkata"
MONTHS_PER_CALL = 3
HISTORY = timedelta(days=5 * 365 - 15)  # NSE keeps about 5 years
ADJUSTING_ACTIONS = ("split", "bonus", "sub-division", "subdivision", "face value")


class UnsupportedSymbolError(ProviderError):
    """The provider has no data for this kind of symbol (e.g. an index)."""


def nse_symbol(symbol: str) -> str:
    """'RELIANCE.NS' -> 'RELIANCE'. Indices (^NSEI) are not in the Bhavcopy."""
    if symbol.startswith("^"):
        raise UnsupportedSymbolError(f"{symbol}: indices are not in the NSE Bhavcopy")
    return symbol.upper().removesuffix(".NS")


def _parse_dates(values: pd.Series) -> pd.Series:
    """ISO dates first; Indian day-first formats (05-10-2026) otherwise."""
    iso = pd.to_datetime(values, format="%Y-%m-%d", errors="coerce")
    if iso.notna().all():
        return iso
    return pd.to_datetime(values, dayfirst=True, errors="coerce")


def price_multiplier(factor: float) -> float:
    """Multiplier for prices BEFORE the ex-date. A 1:1 bonus halves old prices, whether
    NSE states the factor as 2 or as 0.5."""
    if not factor or not np.isfinite(factor) or factor <= 0:
        return 1.0
    return factor if factor < 1 else 1.0 / factor


def adjust_for_actions(df: pd.DataFrame, actions: list[dict]) -> pd.DataFrame:
    """Back-adjust OHLC (and volume, inversely) for splits/bonuses before each ex-date."""
    if df.empty or not actions:
        return df
    out = df.copy()
    day = pd.to_datetime(out["timestamp"], utc=True).dt.tz_convert(IST).dt.normalize()
    for a in actions:
        kind = str(a.get("actionType", "") or a.get("purpose", "")).lower()
        if not any(k in kind for k in ADJUSTING_ACTIONS):
            continue
        ex = _parse_dates(pd.Series([a.get("exDate")])).iloc[0]
        if pd.isna(ex):
            continue
        m = price_multiplier(float(a.get("adjustmentFactor") or 0))
        if m == 1.0:
            continue
        before = (day < pd.Timestamp(ex).tz_localize(IST)).to_numpy()
        out.loc[before, ["open", "high", "low", "close"]] *= m
        out.loc[before, "volume"] /= m
    return out


def rows_to_frame(rows: list[dict]) -> pd.DataFrame:
    """Bhavcopy rows -> OHLCV with bar time 00:00 IST (same as Yahoo's daily bars)."""
    if not rows:
        return empty_ohlcv()
    raw = pd.DataFrame(rows)
    d = _parse_dates(raw["date"].astype(str))
    ts = d.dt.tz_localize(IST).dt.tz_convert("UTC")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": pd.to_numeric(raw["open"], errors="coerce"),
            "high": pd.to_numeric(raw["high"], errors="coerce"),
            "low": pd.to_numeric(raw["low"], errors="coerce"),
            "close": pd.to_numeric(raw["close"], errors="coerce"),
            "volume": pd.to_numeric(raw.get("volume", 0), errors="coerce").fillna(0),
        }
    ).dropna(subset=["timestamp", "close"])
    return normalize_ohlcv(df)


def to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """Weekly bars from daily ones; each bar stamped Monday 00:00 IST (like Yahoo)."""
    if daily.empty:
        return daily
    local = pd.to_datetime(daily["timestamp"], utc=True).dt.tz_convert(IST)
    monday = local.dt.normalize() - pd.to_timedelta(local.dt.day_of_week, unit="D")
    week = monday.dt.tz_convert("UTC")
    g = daily.assign(week=week.to_numpy()).groupby("week", sort=True)
    out = pd.DataFrame(
        {
            "open": g["open"].first(),
            "high": g["high"].max(),
            "low": g["low"].min(),
            "close": g["close"].last(),
            "volume": g["volume"].sum(),
        }
    ).reset_index(names="timestamp")
    return normalize_ohlcv(out)


class NseMcpProvider(DataProvider):
    name = "nse_mcp"
    request_pause = 0.5  # be gentle: bursts get tarpitted

    def __init__(
        self,
        data_delay_minutes: int = 15,
        bhavcopy: McpClient | None = None,
        live: McpClient | None = None,
        call_pause: float = 0.3,
    ) -> None:
        self.data_delay = timedelta(minutes=data_delay_minutes)
        self.bhav = bhavcopy or McpClient(BHAVCOPY_URL)
        self.live = live or McpClient(LIVE_URL)
        self.call_pause = call_pause

    @property
    def timeframes(self) -> set[str]:
        return {"1d", "1w"}

    def max_history(self, timeframe: str) -> timedelta | None:
        return HISTORY

    def bar_close_time(self, ts: pd.Timestamp, timeframe: str) -> pd.Timestamp:
        """Daily bar final 15 min after the 15:30 close; weekly after Friday's."""
        local = ts.tz_convert(IST).normalize()
        if timeframe == "1w":
            local = local + pd.Timedelta(days=4 - local.weekday())
        close = local + pd.Timedelta(hours=SESSION_CLOSE.hour, minutes=SESSION_CLOSE.minute)
        return (close + self.data_delay).tz_convert("UTC")

    # ------------------------------------------------------------------ data

    def _history(self, sym: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Page backwards 3 months at a time from `end` until `start` is covered."""
        frames: list[pd.DataFrame] = []
        end_arg = "today" if end.date() >= datetime.now(UTC).date() else end.date().isoformat()
        for _ in range(40):  # 40 x 3 months = 10 years, a hard stop
            data = self.bhav.call_tool(
                "get_stock_history",
                {"symbol": sym, "months": MONTHS_PER_CALL, "endDate": end_arg},
            )
            rows = (data or {}).get("data", []) if isinstance(data, dict) else []
            if not rows:
                break
            frame = rows_to_frame(rows)
            frames.append(frame)
            nxt = data.get("next_end_date")
            if not nxt or frame["timestamp"].min() <= pd.Timestamp(start):
                break
            end_arg = str(nxt)
            time.sleep(self.call_pause)
        if not frames:
            return empty_ohlcv()
        return normalize_ohlcv(pd.concat(frames, ignore_index=True))

    def _actions(self, sym: str) -> list[dict]:
        try:
            data = self.bhav.call_tool(
                "get_corporate_actions", {"symbol": sym, "fromDate": "", "toDate": ""}
            )
        except McpToolError as exc:
            logger.debug("{}: no corporate actions ({})", sym, exc)
            return []
        return (data or {}).get("actions", []) if isinstance(data, dict) else []

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        self.check_timeframe(timeframe)
        sym = nse_symbol(symbol)
        now = utcnow()
        start = max(start or now - HISTORY, now - HISTORY)
        end = end or now
        try:
            daily = self._history(sym, start, end)
        except McpToolError as exc:
            raise UnsupportedSymbolError(f"{symbol}: {exc}") from exc
        if daily.empty:
            return daily
        daily = adjust_for_actions(daily, self._actions(sym))
        df = to_weekly(daily) if timeframe == "1w" else daily
        lo = pd.Timestamp(start) - (pd.Timedelta(days=7) if timeframe == "1w" else pd.Timedelta(0))
        return df[(df["timestamp"] >= lo) & (df["timestamp"] < pd.Timestamp(end))].reset_index(
            drop=True
        )

    def get_latest(self, symbol: str) -> float:
        q = self.live.call_tool("cm_get_stock_quote", {"symbol": nse_symbol(symbol)})
        price = (q or {}).get("stock", {}).get("lastTradedPrice") if isinstance(q, dict) else None
        if price is None:
            raise ProviderError(f"no live quote for {symbol}")
        return float(price)
