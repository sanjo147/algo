"""A provider that routes each request to an ordered chain of providers.

Daily/weekly: e.g. [nse_mcp, yfinance_nse] -> NSE's official data, Yahoo if NSE is
unreachable (or for indices, which the NSE Bhavcopy does not carry).
Intraday: e.g. [yfinance_nse] (NSE's MCP has no intraday candles).

`name` is set to the provider that served the last request, so the candle store
records where every bar came from.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
from loguru import logger

from sanjotracker.data.providers.base import DataProvider, ProviderError

DAILY_TFS = {"1d", "1w"}


class RoutedProvider(DataProvider):
    def __init__(self, daily: list[DataProvider], intraday: list[DataProvider]) -> None:
        if not daily or not intraday:
            raise ValueError("need at least one daily and one intraday provider")
        self.daily = daily
        self.intraday = intraday
        self.name = daily[0].name
        self.request_pause = max(p.request_pause for p in [*daily, *intraday])
        self._down: set[str] = set()  # providers that failed to connect this run

    def chain(self, timeframe: str) -> list[DataProvider]:
        group = self.daily if timeframe in DAILY_TFS else self.intraday
        return [p for p in group if timeframe in p.timeframes]

    @property
    def timeframes(self) -> set[str]:
        return {tf for p in [*self.daily, *self.intraday] for tf in p.timeframes}

    def max_history(self, timeframe: str) -> timedelta | None:
        """The longest history any provider in the chain has (None = unlimited). Each
        provider clamps requests to its own limit, so a fallback still gets its full depth."""
        limits = [p.max_history(timeframe) for p in self.chain(timeframe)]
        if not limits or any(x is None for x in limits):
            return None
        return max(limits)

    def bar_close_time(self, ts: pd.Timestamp, timeframe: str) -> pd.Timestamp:
        chain = self.chain(timeframe)
        return (
            chain[0].bar_close_time(ts, timeframe)
            if chain
            else super().bar_close_time(ts, timeframe)
        )

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        self.check_timeframe(timeframe)
        errors = []
        for p in self.chain(timeframe):
            if p.name in self._down:
                continue
            try:
                df = p.get_ohlcv(symbol, timeframe, start, end)
            except ProviderError as exc:
                errors.append(f"{p.name}: {exc}")
                if "certificate" in str(exc).lower() or "connect" in str(exc).lower():
                    self._down.add(p.name)  # don't hammer an unreachable server
                logger.warning(
                    "{} {} via {} failed, trying next: {}", symbol, timeframe, p.name, exc
                )
                continue
            self.name = p.name
            if timeframe in DAILY_TFS:
                df = self._top_up(p, symbol, timeframe, df, end)
            return df
        raise ProviderError(f"{symbol} {timeframe}: every provider failed: " + " | ".join(errors))

    def _top_up(
        self,
        used: DataProvider,
        symbol: str,
        timeframe: str,
        df: pd.DataFrame,
        end: datetime | None,
    ) -> pd.DataFrame:
        """NSE publishes the Bhavcopy after the close: fill bars newer than the primary's
        last one from the next provider. The next update overwrites them with NSE's own
        (same price convention, so no 'history changed' re-download)."""
        rest = [q for q in self.chain(timeframe) if q is not used and q.name not in self._down]
        if not rest or df.empty:
            return df
        last = pd.Timestamp(df["timestamp"].iloc[-1])
        if end is not None and last + pd.Timedelta(days=1) > pd.Timestamp(end):
            return df
        try:
            extra = rest[0].get_ohlcv(
                symbol, timeframe, (last + pd.Timedelta(seconds=1)).to_pydatetime(), end
            )
        except ProviderError as exc:
            logger.debug("{} {} top-up via {} failed: {}", symbol, timeframe, rest[0].name, exc)
            return df
        extra = extra[extra["timestamp"] > last]
        if extra.empty:
            return df
        logger.info("{} {}: +{} newest bar(s) from {}", symbol, timeframe, len(extra), rest[0].name)
        return pd.concat([df, extra], ignore_index=True)

    def get_latest(self, symbol: str) -> float:
        errors = []
        for p in [*self.intraday, *self.daily]:
            try:
                return p.get_latest(symbol)
            except (ProviderError, NotImplementedError) as exc:
                errors.append(f"{p.name}: {exc}")
        raise ProviderError(f"{symbol}: no latest price: " + " | ".join(errors))
