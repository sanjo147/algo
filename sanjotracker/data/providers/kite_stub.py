"""Placeholder for a paid broker API (Zerodha Kite / Upstox / Dhan).

Replaced in Phase 9. It exists now so config can name it and the rest of the
code is written against the DataProvider interface from day one.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from sanjotracker.data.providers.base import TIMEFRAME_DELTAS, DataProvider

_MSG = "Broker API provider not implemented yet (Phase 9). Use 'yfinance_nse' for NSE."


class KiteProvider(DataProvider):
    """Stub: every data call raises NotImplementedError."""

    name = "kite"

    def __init__(self, **_: object) -> None:
        pass

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
        raise NotImplementedError(_MSG)

    def get_latest(self, symbol: str) -> float:
        raise NotImplementedError(_MSG)
