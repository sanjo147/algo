"""Tests for config loading."""

from datetime import date
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from sanjotracker.config import PROJECT_ROOT, Secrets, load_config

EXAMPLE = PROJECT_ROOT / "config.example.yaml"


def test_example_config_loads() -> None:
    cfg = load_config(EXAMPLE)
    assert len(cfg.watchlist.nse) == 10
    assert len(cfg.watchlist.crypto) == 5
    assert all(s.endswith(".NS") for s in cfg.watchlist.nse)
    assert cfg.styles.intraday.primary == "15m"
    assert cfg.styles.swing.trend == "1w"
    assert cfg.costs.binance_spot.taker_pct == pytest.approx(0.001)


def test_trading_day_skips_weekends_and_holidays() -> None:
    cfg = load_config(EXAMPLE)
    m = cfg.nse_market
    assert m.is_trading_day(date(2026, 11, 16))  # Monday
    assert not m.is_trading_day(date(2026, 11, 15))  # Sunday
    assert not m.is_trading_day(date(2026, 12, 25))  # holiday in example config


def test_bad_timeframe_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    raw["timeframes"] = ["1d", "3m"]
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(bad)


def test_secrets_are_masked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")
    s = Secrets(_env_file=None)
    assert s.telegram_bot_token is not None
    assert "123:abc" not in repr(s)
    assert s.telegram_bot_token.get_secret_value() == "123:abc"
