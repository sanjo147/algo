"""Scanner: watchlist loop, MTF filter, error isolation (no network: update=False)."""

import numpy as np
import pandas as pd

from sanjotracker.analysis.breakout import Event, EventType
from sanjotracker.config import PROJECT_ROOT, load_config
from sanjotracker.scanner import Scanner
from tests.conftest import T0, make_candles
from tests.test_breakout import BREAKOUT_BAR, _append, _range

SYM = "AAA.NS"


def _cfg():  # noqa: ANN202
    cfg = load_config(PROJECT_ROOT / "config.example.yaml")
    cfg.watchlist.nse = [SYM, "MISSING.NS"]
    return cfg


def _hourly(slope: float) -> pd.DataFrame:
    """1h trend data that ends before the 15m breakout bar closes."""
    h = make_candles(T0 - pd.Timedelta(hours=100), 150, "1h", seed=8)
    h[["open", "high", "low", "close"]] += np.linspace(0, slope, 150)[:, None]
    return h


def _fill(store, slope: float) -> None:  # noqa: ANN001
    store.upsert(SYM, "15m", _append(_range(freq="15min"), [BREAKOUT_BAR]), "test")
    store.upsert(SYM, "1h", _hourly(slope), "test")


def test_scan_finds_breakout_with_trend(store) -> None:  # noqa: ANN001
    _fill(store, +60)
    sc = Scanner(_cfg(), store)
    events = sc.scan("intraday", update=False)
    assert all(isinstance(e, Event) for e in events)
    bo = [e for e in events if e.event == EventType.BREAKOUT]
    assert len(bo) == 1
    assert bo[0].symbol == SYM and bo[0].timeframe == "15m" and bo[0].style == "intraday"
    assert bo[0].trend_tf == "1h" and bo[0].mtf_ok is True
    assert "MISSING.NS" in sc.errors  # one bad symbol doesn't stop the scan
    assert SYM in sc.regimes


def test_counter_trend_breakout_filtered_by_default(store) -> None:  # noqa: ANN001
    _fill(store, -60)
    sc = Scanner(_cfg(), store)
    kinds = [e.event for e in sc.scan("intraday", update=False)]
    assert EventType.BREAKOUT not in kinds
    kinds_all = [e.event for e in sc.scan("intraday", update=False, include_counter_trend=True)]
    assert EventType.BREAKOUT in kinds_all


def test_min_confidence_and_sorting(store) -> None:  # noqa: ANN001
    _fill(store, +60)
    sc = Scanner(_cfg(), store)
    events = sc.scan("intraday", update=False, lookback_bars=5)
    conf = [e.confidence for e in events]
    assert conf == sorted(conf, reverse=True)
    assert all(e.confidence >= 101 for e in sc.scan("intraday", update=False, min_confidence=101))


def test_event_key_is_stable(store) -> None:  # noqa: ANN001
    _fill(store, +60)
    e = next(
        e for e in Scanner(_cfg(), store).scan("intraday", update=False) if e.event == "breakout"
    )
    assert e.key.startswith(f"{SYM}|15m|breakout|")
    assert Event.model_validate_json(e.model_dump_json()) == e  # serialisable for Phase 4
