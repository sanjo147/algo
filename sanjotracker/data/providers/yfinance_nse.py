"""NSE equity provider backed by yfinance (free, ~15 min delayed).

Free-data limits (Yahoo): 5m/15m only for the last ~60 days, 1h for the last ~730
days. Requests older than that are clamped with a warning instead of failing.
Prices are split/dividend adjusted (auto_adjust=True).
"""

from __future__ import annotations

from datetime import datetime, time, timedelta
from typing import Any

import pandas as pd
import yfinance as yf
from loguru import logger

from sanjotracker.data.providers.base import (
    DEFAULT_BACKFILL,
    DataProvider,
    ProviderError,
    empty_ohlcv,
    normalize_ohlcv,
    utcnow,
    with_retries,
)

IST = "Asia/Kolkata"
SESSION_CLOSE = time(15, 30)

_YF_INTERVAL = {"5m": "5m", "15m": "15m", "1h": "1h", "1d": "1d", "1w": "1wk"}
# A day inside Yahoo's hard limit so a request never lands right on the edge.
_MAX_HISTORY = {"5m": timedelta(days=59), "15m": timedelta(days=59), "1h": timedelta(days=729)}


class YFinanceNSEProvider(DataProvider):
    """yfinance source for `.NS` symbols (and indices such as ^NSEI)."""

    name = "yfinance_nse"
    request_pause = 1.0

    def __init__(self, data_delay_minutes: int = 15, ticker_factory: Any = None) -> None:
        """
        Args:
            data_delay_minutes: how long after the 15:30 IST close a daily/weekly bar is
                treated as final (Yahoo's NSE feed lags the exchange).
            ticker_factory: injectable replacement for `yf.Ticker` (tests).
        """
        self.data_delay = timedelta(minutes=data_delay_minutes)
        self._ticker = ticker_factory or yf.Ticker

    @property
    def timeframes(self) -> set[str]:
        return set(_YF_INTERVAL)

    def max_history(self, timeframe: str) -> timedelta | None:
        return _MAX_HISTORY.get(timeframe)

    def bar_close_time(self, ts: pd.Timestamp, timeframe: str) -> pd.Timestamp:
        """NSE bars end at the session close, not at open + timeframe for 1d/1w."""
        local = ts.tz_convert(IST)
        if timeframe in ("1d", "1w"):
            day = local.normalize()
            if timeframe == "1w":  # weekly bars open Monday; final after Friday's close
                day = day + pd.Timedelta(days=4 - day.weekday())
            close = day + pd.Timedelta(hours=SESSION_CLOSE.hour, minutes=SESSION_CLOSE.minute)
            return (close + self.data_delay).tz_convert("UTC")
        session_end = local.normalize() + pd.Timedelta(
            hours=SESSION_CLOSE.hour, minutes=SESSION_CLOSE.minute
        )
        # the 15:15 1h bar only runs 15 minutes, so cap at the session end
        return min(super().bar_close_time(local, timeframe), session_end).tz_convert("UTC")

    def _clamp_start(self, timeframe: str, start: datetime, now: datetime) -> datetime:
        limit = self.max_history(timeframe)
        if limit is not None and start < now - limit:
            clamped = now - limit
            if clamped - start < timedelta(days=1):  # rounding, not a real request for old data
                return clamped
            logger.warning(
                "yfinance keeps only {} days of {} data; start moved {} -> {}",
                limit.days,
                timeframe,
                start.date(),
                clamped.date(),
            )
            return clamped
        return start

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        self.check_timeframe(timeframe)
        now = utcnow()
        start = self._clamp_start(timeframe, start or now - DEFAULT_BACKFILL[timeframe], now)
        end = end or now + timedelta(days=1)  # yfinance `end` is exclusive

        def _fetch() -> pd.DataFrame:
            return self._ticker(symbol).history(
                interval=_YF_INTERVAL[timeframe],
                start=start,
                end=end,
                auto_adjust=True,
                actions=False,
                prepost=False,
            )

        raw = with_retries(_fetch, what=f"yfinance {symbol} {timeframe}")
        if raw is None or raw.empty:
            logger.warning("yfinance returned no data for {} {}", symbol, timeframe)
            return empty_ohlcv()
        df = raw.reset_index()
        df = df.rename(columns={df.columns[0]: "timestamp"})
        df = df.rename(columns=str.lower)
        out = normalize_ohlcv(df)
        return out[
            (out["timestamp"] >= pd.Timestamp(start)) & (out["timestamp"] < pd.Timestamp(end))
        ]

    def get_latest(self, symbol: str) -> float:
        def _fetch() -> float:
            price = self._ticker(symbol).fast_info["last_price"]
            if price is None or pd.isna(price):
                raise ProviderError(f"no last price for {symbol}")
            return float(price)

        return with_retries(_fetch, what=f"yfinance latest {symbol}")
