"""CandleStore: upsert without duplicates, ranged loads, UTC round-trip."""

from datetime import timedelta

import pandas as pd

from tests.conftest import T0, make_candles


def test_upsert_twice_does_not_duplicate(store) -> None:
    df = make_candles(T0, 50)
    store.upsert("BTC/USDT", "1h", df, "fake")
    store.upsert("BTC/USDT", "1h", df, "fake")
    assert store.count("BTC/USDT", "1h") == 50


def test_upsert_overwrites_changed_candle(store) -> None:
    df = make_candles(T0, 5)
    store.upsert("X", "1h", df, "fake")
    df.loc[4, "close"] = 999.0
    store.upsert("X", "1h", df.tail(1), "fake")
    out = store.load("X", "1h")
    assert len(out) == 5
    assert out["close"].iloc[-1] == 999.0


def test_symbols_and_timeframes_are_separate(store) -> None:
    store.upsert("A", "1h", make_candles(T0, 10), "fake")
    store.upsert("A", "1d", make_candles(T0, 3, "1d"), "fake")
    store.upsert("B", "1h", make_candles(T0, 7), "fake")
    assert store.count("A", "1h") == 10
    assert store.count("A", "1d") == 3
    assert store.count("B", "1h") == 7
    inv = store.inventory()
    assert set(zip(inv["symbol"], inv["timeframe"], strict=True)) == {
        ("A", "1h"),
        ("A", "1d"),
        ("B", "1h"),
    }


def test_load_roundtrip_and_range(store) -> None:
    df = make_candles(T0, 24)
    store.upsert("X", "1h", df, "fake")
    out = store.load("X", "1h")
    assert str(out["timestamp"].dt.tz) == "UTC"
    pd.testing.assert_frame_equal(out, df, check_dtype=False)

    part = store.load("X", "1h", start=T0 + timedelta(hours=5), end=T0 + timedelta(hours=10))
    assert len(part) == 5
    assert part["timestamp"].iloc[0] == pd.Timestamp(T0 + timedelta(hours=5))

    tail = store.load("X", "1h", limit=3)
    assert list(tail["timestamp"]) == list(df["timestamp"].tail(3))


def test_last_timestamp_and_delete(store) -> None:
    assert store.last_timestamp("X", "1h") is None
    df = make_candles(T0, 10)
    store.upsert("X", "1h", df, "fake")
    assert store.last_timestamp("X", "1h") == df["timestamp"].iloc[-1]
    assert store.delete("X", "1h") == 10
    assert store.load("X", "1h").empty
