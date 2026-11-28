"""Incremental updater: fetch only candles the store does not have yet.

For each (symbol, timeframe):
1. Find the newest stored candle. None -> first-time backfill.
2. Re-fetch from that candle onward (one bar of overlap).
3. If the overlap candle's close changed by more than `adjust_tolerance`, the
   source re-adjusted history (split/bonus/dividend) -> wipe and re-download.
4. Drop the still-forming last bar, then upsert.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from datetime import datetime

import pandas as pd
from loguru import logger
from pydantic import BaseModel

from sanjotracker.config import Providers
from sanjotracker.data.providers.base import DEFAULT_BACKFILL, DataProvider, utcnow
from sanjotracker.data.store import CandleStore


class UpdateResult(BaseModel):
    """Outcome of updating one symbol+timeframe."""

    symbol: str
    timeframe: str
    new_rows: int = 0
    total_rows: int = 0
    last_candle: datetime | None = None
    full_refetch: bool = False
    error: str | None = None


SESSION_TZ = "Asia/Kolkata"


def market_of(symbol: str) -> str:
    """Every symbol is on NSE (.NS stocks, ^ indices). Pairs like BTC/USDT are refused."""
    if "/" in symbol:
        raise ValueError(f"{symbol!r} looks like a crypto pair; SanjoTracker is NSE-only")
    return "nse"


def session_tz_for(symbol: str) -> str:
    """Timezone whose calendar day defines a session (VWAP reset, previous-day levels)."""
    return SESSION_TZ


def normalize_symbol(symbol: str) -> str:
    """'tcs' -> 'TCS.NS'; indices (^NSEI) and '.NS' symbols are kept as they are."""
    s = symbol.strip().upper()
    if not s or s.startswith("^") or s.endswith(".NS"):
        return s
    return f"{s}.NS"


class Updater:
    """Keeps the CandleStore current using one DataProvider per market."""

    def __init__(
        self,
        store: CandleStore,
        providers: dict[str, DataProvider],
        adjust_tolerance: float = 0.005,
        sleep: bool = True,
    ) -> None:
        self.store = store
        self.providers = providers
        self.adjust_tolerance = adjust_tolerance
        self.sleep = sleep

    def provider_for(self, symbol: str) -> DataProvider:
        market = market_of(symbol)
        if market not in self.providers:
            raise KeyError(f"no provider configured for market {market!r}")
        return self.providers[market]

    def _first_start(self, provider: DataProvider, timeframe: str, now: datetime) -> datetime:
        start = now - DEFAULT_BACKFILL[timeframe]
        limit = provider.max_history(timeframe)
        return max(start, now - limit) if limit is not None else start

    def _history_changed(
        self, symbol: str, timeframe: str, fresh: pd.DataFrame, last: pd.Timestamp
    ) -> bool:
        """True if the re-fetched overlap candle disagrees with what is stored."""
        new = fresh.loc[fresh["timestamp"] == last, "close"]
        if new.empty:
            return False
        old = self.store.load(symbol, timeframe, start=last, limit=1)["close"]
        if old.empty or old.iloc[0] == 0:
            return False
        change = abs(new.iloc[0] / old.iloc[0] - 1)
        return change > self.adjust_tolerance

    def update(self, symbol: str, timeframe: str, now: datetime | None = None) -> UpdateResult:
        """Bring one symbol+timeframe up to date. Never raises; errors go in the result."""
        now = now or utcnow()
        try:
            provider = self.provider_for(symbol)
            provider.check_timeframe(timeframe)
            last = self.store.last_timestamp(symbol, timeframe)
            full = False

            if last is None:
                start = self._first_start(provider, timeframe, now)
            else:
                start = last.to_pydatetime()
                limit = provider.max_history(timeframe)
                if limit is not None and start < now - limit:
                    logger.warning(
                        "{} {}: last candle {} is older than the provider keeps; "
                        "there will be a gap",
                        symbol,
                        timeframe,
                        last,
                    )
                    start = now - limit

            df = provider.get_ohlcv(symbol, timeframe, start=start, end=now)

            if last is not None and self._history_changed(symbol, timeframe, df, last):
                logger.warning(
                    "{} {}: history was re-adjusted at source; re-downloading", symbol, timeframe
                )
                self.store.delete(symbol, timeframe)
                full = True
                df = provider.get_ohlcv(
                    symbol, timeframe, start=self._first_start(provider, timeframe, now), end=now
                )

            df = provider.drop_incomplete(df, timeframe, now)
            before = 0 if full else self.store.count(symbol, timeframe)
            self.store.upsert(symbol, timeframe, df, provider.name)
            total = self.store.count(symbol, timeframe)
            res = UpdateResult(
                symbol=symbol,
                timeframe=timeframe,
                new_rows=total - before,
                total_rows=total,
                last_candle=self.store.last_timestamp(symbol, timeframe),
                full_refetch=full,
            )
            logger.info("{} {}: +{} candles (total {})", symbol, timeframe, res.new_rows, total)
            return res
        except Exception as exc:  # one bad symbol must not stop the whole run
            logger.error("{} {}: update failed: {}", symbol, timeframe, exc)
            return UpdateResult(symbol=symbol, timeframe=timeframe, error=str(exc))

    def update_many(
        self, symbols: Iterable[str], timeframes: Iterable[str], now: datetime | None = None
    ) -> list[UpdateResult]:
        """Update every symbol x timeframe, pausing between requests per provider."""
        tfs = list(timeframes)
        results: list[UpdateResult] = []
        for symbol in symbols:
            for tf in tfs:
                results.append(self.update(symbol, tf, now=now))
                if self.sleep:
                    try:
                        time.sleep(self.provider_for(symbol).request_pause)
                    except KeyError:
                        pass
        return results


def build_updater(store: CandleStore, providers: Providers) -> Updater:
    """Create an Updater from the config's provider chains (`cfg.providers`)."""
    from sanjotracker.data.providers import build_nse_provider

    return Updater(store, {"nse": build_nse_provider(providers)})
