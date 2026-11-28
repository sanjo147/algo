"""Updater: incremental fetch, incomplete-bar drop, re-adjust detection, error isolation."""

from datetime import timedelta

import pandas as pd

from sanjotracker.data.updater import Updater, market_of
from tests.conftest import T0, FakeProvider, make_candles


def _updater(store, provider) -> Updater:  # noqa: ANN001
    return Updater(store, {"nse": provider}, sleep=False)


def test_market_of() -> None:
    assert market_of("RELIANCE.NS") == "nse"
    assert market_of("^NSEI") == "nse"


def test_first_run_backfills_then_only_fetches_new(store) -> None:
    data = make_candles(T0, 100)
    prov = FakeProvider(data)
    up = _updater(store, prov)

    now = T0 + timedelta(hours=50)  # bars 0..49 have closed
    r1 = up.update("RELIANCE.NS", "1h", now=now)
    assert r1.error is None
    assert r1.new_rows == 50
    assert store.last_timestamp("RELIANCE.NS", "1h") == data["timestamp"].iloc[49]

    now2 = T0 + timedelta(hours=60)
    r2 = up.update("RELIANCE.NS", "1h", now=now2)
    assert r2.new_rows == 10
    assert r2.total_rows == 60
    # second call started at the last stored candle, not from scratch
    assert pd.Timestamp(prov.calls[-1][2]) == data["timestamp"].iloc[49]


def test_rerun_with_no_new_bars_adds_nothing(store) -> None:
    prov = FakeProvider(make_candles(T0, 30))
    up = _updater(store, prov)
    now = T0 + timedelta(hours=30)
    up.update("X.NS", "1h", now=now)
    r = up.update("X.NS", "1h", now=now)
    assert r.new_rows == 0
    assert store.count("X.NS", "1h") == 30


def test_forming_bar_is_not_stored(store) -> None:
    prov = FakeProvider(make_candles(T0, 10))
    up = _updater(store, prov)
    # 30 minutes into bar 5: bars 0..4 closed, bar 5 still forming
    up.update("X.NS", "1h", now=T0 + timedelta(hours=5, minutes=30))
    assert store.count("X.NS", "1h") == 5


def test_history_readjustment_triggers_full_refetch(store) -> None:
    data = make_candles(T0, 40)
    prov = FakeProvider(data)
    up = _updater(store, prov)
    up.update("RELIANCE.NS", "1h", now=T0 + timedelta(hours=20))

    adjusted = data.copy()  # e.g. a 1:1 bonus halves all past prices
    adjusted[["open", "high", "low", "close"]] /= 2
    prov.data = adjusted
    r = up.update("RELIANCE.NS", "1h", now=T0 + timedelta(hours=30))

    assert r.full_refetch
    stored = store.load("RELIANCE.NS", "1h")
    assert len(stored) == 30
    assert stored["close"].iloc[0] == adjusted["close"].iloc[0]


def test_backfill_respects_provider_history_limit(store) -> None:
    data = make_candles(T0, 24 * 10)  # 10 days of 1h bars
    prov = FakeProvider(data, history=timedelta(days=3))
    up = _updater(store, prov)
    now = T0 + timedelta(days=10)
    up.update("X.NS", "1h", now=now)
    assert pd.Timestamp(prov.calls[0][2]) == pd.Timestamp(now - timedelta(days=3))
    assert store.count("X.NS", "1h") == 72


def test_one_failing_symbol_does_not_stop_the_rest(store) -> None:
    class Boom(FakeProvider):
        def get_ohlcv(self, symbol, timeframe, start=None, end=None):  # noqa: ANN001
            if symbol == "BAD.NS":
                raise RuntimeError("exchange down")
            return super().get_ohlcv(symbol, timeframe, start, end)

    up = _updater(store, Boom(make_candles(T0, 10)))
    res = up.update_many(["BAD.NS", "GOOD.NS"], ["1h"], now=T0 + timedelta(hours=10))
    assert res[0].error and "exchange down" in res[0].error
    assert res[1].error is None and res[1].new_rows == 10


def test_unsupported_timeframe_is_reported(store) -> None:
    up = _updater(store, FakeProvider(make_candles(T0, 5)))
    r = up.update("X.NS", "3m", now=T0 + timedelta(hours=5))
    assert r.error is not None
