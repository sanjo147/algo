"""Runs breakout analysis over the watchlist for one style and returns events.

Per symbol (style = swing: 1d primary / 1w trend; intraday: 15m primary / 1h trend):
1. optionally fetch new candles (primary, trend, and the market benchmark daily);
2. detect_events() on the primary timeframe with the trend timeframe as the
   multi-timeframe filter and the benchmark for the market regime;
3. keep events from the last `lookback_bars` closed bars.

Breakouts/breakdowns that go AGAINST the higher-timeframe trend are dropped by
default (the spec's MTF filter); retests and failures are kept but scored lower.
One bad symbol never stops the scan: its error is logged and kept in `errors`.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd
from loguru import logger

from sanjotracker.analysis.breakout import Event, EventType, detect_events, last_bar_events
from sanjotracker.analysis.regime import Regime, current_regime
from sanjotracker.config import Config, get_config
from sanjotracker.data.store import CandleStore
from sanjotracker.data.updater import Updater, build_updater, market_of, session_tz_for

Style = Literal["swing", "intraday"]
Market = Literal["nse", "crypto", "all"]
BENCHMARK_TF = "1d"


class Scanner:
    """Holds config + store; `scan()` returns events and records regimes/errors."""

    def __init__(
        self,
        cfg: Config | None = None,
        store: CandleStore | None = None,
        updater: Updater | None = None,
    ) -> None:
        self.cfg = cfg or get_config()
        self.store = store or CandleStore(self.cfg.db_file)
        self._updater = updater
        self.errors: dict[str, str] = {}
        self.regimes: dict[str, Regime] = {}

    # ------------------------------------------------------------------ helpers

    @property
    def updater(self) -> Updater:
        if self._updater is None:
            names = {"nse": self.cfg.providers.nse, "crypto": self.cfg.providers.crypto}
            self._updater = build_updater(self.store, names)
        return self._updater

    def symbols_for(self, market: Market) -> list[str]:
        out: list[str] = []
        if market in ("nse", "all"):
            out += self.cfg.watchlist.nse
        if market in ("crypto", "all"):
            out += self.cfg.watchlist.crypto
        return list(dict.fromkeys(out))

    def benchmark_for(self, symbol: str) -> str:
        return getattr(self.cfg.benchmarks, market_of(symbol))

    def _load(self, symbol: str, timeframe: str) -> pd.DataFrame:
        return self.store.load(symbol, timeframe, limit=self.cfg.breakout.scan_history_bars)

    def _refresh(self, symbols: list[str], timeframes: list[str]) -> None:
        benches = list(dict.fromkeys(self.benchmark_for(s) for s in symbols))
        for res in self.updater.update_many(symbols, timeframes):
            if res.error:
                self.errors[f"{res.symbol} {res.timeframe}"] = f"update: {res.error}"
        for res in self.updater.update_many(benches, [BENCHMARK_TF]):
            if res.error:
                self.errors[f"{res.symbol} {res.timeframe}"] = f"update: {res.error}"

    # ------------------------------------------------------------------ main

    def scan_symbol(self, symbol: str, style: Style, lookback_bars: int = 1) -> list[Event]:
        """Events for one symbol on its style's primary timeframe."""
        st = getattr(self.cfg.styles, style)
        df = self._load(symbol, st.primary)
        if df.empty:
            raise ValueError(f"no {st.primary} candles stored; run scripts/backfill.py")
        trend_df = self._load(symbol, st.trend)
        bench = self.store.load(self.benchmark_for(symbol), BENCHMARK_TF)
        self.regimes[symbol] = current_regime(df, st.primary, bench, self.cfg.regime)
        window = max(self.cfg.breakout.retest_within_bars, self.cfg.breakout.failed_within_bars)
        events = detect_events(
            df,
            symbol,
            st.primary,
            cfg=self.cfg.breakout,
            levels_cfg=self.cfg.levels,
            regime_cfg=self.cfg.regime,
            session_tz=session_tz_for(symbol),
            trend_df=None if trend_df.empty else trend_df,
            trend_tf=st.trend,
            benchmark_df=None if bench.empty else bench,
            start=max(0, len(df) - window - lookback_bars),
            style=style,
        )
        return last_bar_events(events, df, lookback_bars)

    def scan(
        self,
        style: Style,
        *,
        market: Market = "all",
        symbols: list[str] | None = None,
        update: bool = True,
        lookback_bars: int = 1,
        include_counter_trend: bool = False,
        min_confidence: int = 0,
    ) -> list[Event]:
        """Scan the watchlist (or `symbols`). Returns events, highest confidence first."""
        self.errors, self.regimes = {}, {}
        syms = list(dict.fromkeys(symbols)) if symbols else self.symbols_for(market)
        st = getattr(self.cfg.styles, style)
        if update:
            self._refresh(syms, list(dict.fromkeys([st.primary, st.trend])))

        events: list[Event] = []
        for sym in syms:
            try:
                events += self.scan_symbol(sym, style, lookback_bars)
            except Exception as exc:  # keep scanning the rest
                logger.error("{} scan failed: {}", sym, exc)
                self.errors[sym] = str(exc)

        if not include_counter_trend:
            breaks = (EventType.BREAKOUT, EventType.BREAKDOWN)
            events = [e for e in events if not (e.event in breaks and e.mtf_ok is False)]
        events = [e for e in events if e.confidence >= min_confidence]
        events.sort(key=lambda e: (-e.confidence, e.symbol))
        logger.info("{} scan: {} symbols, {} events", style, len(syms), len(events))
        return events

    def benchmark_regimes(self, market: Market = "all") -> dict[str, Regime]:
        """Regime of each benchmark on its daily chart (for the header line)."""
        out: dict[str, Regime] = {}
        for m in ("nse", "crypto"):
            if market not in (m, "all"):
                continue
            sym = getattr(self.cfg.benchmarks, m)
            df = self.store.load(sym, BENCHMARK_TF)
            out[sym] = current_regime(df, BENCHMARK_TF, df, self.cfg.regime)
        return out


def scan(style: Style, **kwargs) -> list[Event]:  # noqa: ANN003
    """Convenience: Scanner().scan(style, ...)."""
    return Scanner().scan(style, **kwargs)
