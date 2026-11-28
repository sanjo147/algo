"""Provider tests with mocked yfinance objects (no network). NSE MCP: test_nse_mcp.py."""

import ast
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from sanjotracker.config import PROJECT_ROOT
from sanjotracker.data.providers import available_providers, get_provider
from sanjotracker.data.providers.base import (
    OHLCV_COLUMNS,
    ProviderError,
    UnsupportedTimeframeError,
    normalize_ohlcv,
    with_retries,
)
from sanjotracker.data.providers.kite_stub import KiteProvider
from sanjotracker.data.providers.yfinance_nse import YFinanceNSEProvider

# ------------------------------------------------------------------ helpers


class FakeTicker:
    """Mimics yf.Ticker(...).history(): IST-indexed, capitalised columns."""

    calls: list[dict] = []

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self.fast_info = {"last_price": 2850.5}

    def history(self, **kwargs) -> pd.DataFrame:  # noqa: ANN003
        FakeTicker.calls.append(kwargs)
        idx = pd.date_range(
            "2026-11-10 09:15", periods=4, freq="15min", tz="Asia/Kolkata", name="Datetime"
        )
        return pd.DataFrame(
            {
                "Open": [1.0, 2, 3, 4],
                "High": [2.0, 3, 4, 5],
                "Low": [0.5, 1, 2, 3],
                "Close": [1.5, 2.5, 3.5, 4.5],
                "Volume": [100, 200, 300, 400],
            },
            index=idx,
        )


# ------------------------------------------------------------------ base


def test_normalize_sorts_dedupes_and_sets_utc() -> None:
    raw = pd.DataFrame(
        {
            "timestamp": ["2026-01-01 02:00", "2026-01-01 01:00", "2026-01-01 01:00"],
            "open": [1, 2, 3],
            "high": [1, 2, 3],
            "low": [1, 2, 3],
            "close": [1, 2, 3],
            "volume": [None, 1, 1],
        }
    )
    out = normalize_ohlcv(raw)
    assert list(out.columns) == OHLCV_COLUMNS
    assert len(out) == 2
    assert out["timestamp"].is_monotonic_increasing
    assert str(out["timestamp"].dt.tz) == "UTC"
    assert out["volume"].iloc[1] == 0.0


def test_with_retries_gives_up(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("time.sleep", lambda s: None)
    calls = {"n": 0}

    def boom() -> None:
        calls["n"] += 1
        raise ConnectionError("nope")

    with pytest.raises(ProviderError):
        with_retries(boom, attempts=3)
    assert calls["n"] == 3


# ------------------------------------------------------------------ yfinance


def test_yfinance_normalises_to_utc() -> None:
    p = YFinanceNSEProvider(ticker_factory=FakeTicker)
    start = datetime(2026, 11, 1, tzinfo=UTC)
    end = datetime(2026, 11, 11, tzinfo=UTC)
    df = p.get_ohlcv("RELIANCE.NS", "15m", start=start, end=end)
    assert list(df.columns) == OHLCV_COLUMNS
    assert len(df) == 4
    # 09:15 IST == 03:45 UTC
    assert df["timestamp"].iloc[0] == pd.Timestamp("2026-11-10 03:45", tz="UTC")
    assert FakeTicker.calls[-1]["interval"] == "15m"
    assert p.get_latest("RELIANCE.NS") == 2850.5


def test_yfinance_clamps_old_intraday_start(monkeypatch: pytest.MonkeyPatch) -> None:
    now = datetime(2026, 11, 15, tzinfo=UTC)
    monkeypatch.setattr("sanjotracker.data.providers.yfinance_nse.utcnow", lambda: now)
    p = YFinanceNSEProvider(ticker_factory=FakeTicker)
    p.get_ohlcv("TCS.NS", "5m", start=now - timedelta(days=365), end=now)
    assert FakeTicker.calls[-1]["start"] == now - timedelta(days=59)


def test_yfinance_weekly_interval_name() -> None:
    p = YFinanceNSEProvider(ticker_factory=FakeTicker)
    p.get_ohlcv("TCS.NS", "1w", start=datetime(2020, 1, 1, tzinfo=UTC))
    assert FakeTicker.calls[-1]["interval"] == "1wk"


@pytest.mark.parametrize(
    ("bar_open_ist", "tf", "final_ist"),
    [
        ("2026-11-10 09:15", "15m", "2026-11-10 09:30"),
        ("2026-11-10 15:15", "1h", "2026-11-10 15:30"),  # short last hourly bar
        ("2026-11-10 00:00", "1d", "2026-11-10 15:45"),  # close + 15 min feed delay
        ("2026-11-09 00:00", "1w", "2026-11-13 15:45"),  # Monday bar -> Friday close
    ],
)
def test_nse_bar_close_times(bar_open_ist: str, tf: str, final_ist: str) -> None:
    p = YFinanceNSEProvider(ticker_factory=FakeTicker)
    ts = pd.Timestamp(bar_open_ist, tz="Asia/Kolkata").tz_convert("UTC")
    assert p.bar_close_time(ts, tf) == pd.Timestamp(final_ist, tz="Asia/Kolkata")


def test_nse_daily_bar_dropped_until_after_close() -> None:
    p = YFinanceNSEProvider(ticker_factory=FakeTicker)
    df = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-11-10", tz="Asia/Kolkata").tz_convert("UTC")],
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "volume": [1.0],
        }
    )
    during = pd.Timestamp("2026-11-10 14:00", tz="Asia/Kolkata")
    after = pd.Timestamp("2026-11-10 15:45", tz="Asia/Kolkata")
    assert p.drop_incomplete(df, "1d", during).empty
    assert len(p.drop_incomplete(df, "1d", after)) == 1


def test_unsupported_timeframe() -> None:
    p = YFinanceNSEProvider(ticker_factory=FakeTicker)
    with pytest.raises(UnsupportedTimeframeError):
        p.get_ohlcv("RELIANCE.NS", "3m")


# ------------------------------------------------------------------ stub + registry


def test_kite_stub_raises() -> None:
    with pytest.raises(NotImplementedError):
        KiteProvider().get_ohlcv("RELIANCE", "1d")


def test_registry() -> None:
    assert {"nse_mcp", "yfinance_nse", "kite"} <= set(available_providers())
    assert isinstance(get_provider("kite"), KiteProvider)
    with pytest.raises(ValueError):
        get_provider("nope")


# ------------------------------------------------------------------ architecture rule


def test_only_providers_import_vendor_libraries() -> None:
    """Rule 4: vendor data libraries may only be imported inside data/providers/."""
    banned = {"yfinance", "ccxt", "kiteconnect", "mcp"}
    allowed_dir = PROJECT_ROOT / "sanjotracker" / "data" / "providers"
    offenders = []
    for folder in ("sanjotracker", "app", "scripts"):
        for path in (PROJECT_ROOT / folder).rglob("*.py"):
            if allowed_dir in path.parents:
                continue
            tree = ast.parse(Path(path).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                if any(n.split(".")[0] in banned for n in names):
                    offenders.append(str(path.relative_to(PROJECT_ROOT)))
    assert not offenders, f"vendor imports outside providers: {offenders}"
