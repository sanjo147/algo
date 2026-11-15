"""Crypto spot provider via ccxt public endpoints (no API key needed).

Defaults to Binance via its official market-data-only host
(data-api.binance.vision). That host serves the same public klines/tickers as
api.binance.com but is reachable on networks (e.g. some Indian ISPs) where the
main API domain is blocked. `exchange_id` can point at any other ccxt exchange.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import ccxt
import pandas as pd
from loguru import logger

from sanjotracker.data.providers.base import (
    DEFAULT_BACKFILL,
    TIMEFRAME_DELTAS,
    DataProvider,
    empty_ohlcv,
    normalize_ohlcv,
    utcnow,
    with_retries,
)

BINANCE_DATA_API = "https://data-api.binance.vision/api/v3"
_PAGE_LIMIT = 1000
_MAX_PAGES = 500  # safety stop (~500k candles)


class BinanceCCXTProvider(DataProvider):
    """Crypto OHLCV from a ccxt exchange; bars are UTC and run 24/7."""

    name = "binance_ccxt"
    request_pause = 0.2  # ccxt's enableRateLimit already throttles each call

    def __init__(
        self,
        exchange_id: str = "binance",
        public_api_url: str | None = BINANCE_DATA_API,
        exchange: Any = None,
    ) -> None:
        """
        Args:
            exchange_id: any ccxt exchange id.
            public_api_url: override for Binance's public REST base URL (None = ccxt default).
            exchange: inject a fake ccxt exchange object (tests).
        """
        if exchange is None:
            opts: dict[str, Any] = {"enableRateLimit": True}
            if exchange_id == "binance":
                opts["options"] = {"fetchMarkets": {"types": ["spot"]}}  # skip futures endpoints
            exchange = getattr(ccxt, exchange_id)(opts)
            if exchange_id == "binance" and public_api_url:
                exchange.urls["api"]["public"] = public_api_url
        self.exchange = exchange

    @property
    def timeframes(self) -> set[str]:
        return set(TIMEFRAME_DELTAS)

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        self.check_timeframe(timeframe)
        now = utcnow()
        start = start or now - DEFAULT_BACKFILL[timeframe]
        end = end or now
        since = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        step_ms = int(TIMEFRAME_DELTAS[timeframe].total_seconds() * 1000)

        rows: list[list[float]] = []
        for _ in range(_MAX_PAGES):
            batch = with_retries(
                lambda s=since: self.exchange.fetch_ohlcv(
                    symbol, timeframe, since=s, limit=_PAGE_LIMIT
                ),
                what=f"ccxt {symbol} {timeframe}",
            )
            if not batch:
                break
            rows.extend(batch)
            last = int(batch[-1][0])
            if last + step_ms >= end_ms or last < since:
                break
            since = last + step_ms
        else:
            logger.warning("ccxt {} {}: hit page limit, data may be truncated", symbol, timeframe)

        if not rows:
            return empty_ohlcv()
        df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        out = normalize_ohlcv(df)
        return out[
            (out["timestamp"] >= pd.Timestamp(start)) & (out["timestamp"] < pd.Timestamp(end))
        ]

    def get_latest(self, symbol: str) -> float:
        ticker = with_retries(
            lambda: self.exchange.fetch_ticker(symbol), what=f"ccxt ticker {symbol}"
        )
        return float(ticker["last"])

    def max_history(self, timeframe: str) -> timedelta | None:
        return None
