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
    assert cfg.providers.daily == ["nse_mcp", "yfinance_nse"]
    assert all(s.endswith(".NS") for s in cfg.watchlist.nse)
    assert cfg.styles.intraday.primary == "15m"
    assert cfg.styles.swing.trend == "1w"
    assert cfg.costs.nse_delivery.stt_sell_pct == pytest.approx(0.001)


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


def test_update_config_values_keeps_comments_and_validates(tmp_path) -> None:  # noqa: ANN001
    from datetime import time

    from sanjotracker.config import update_config_values

    p = tmp_path / "config.yaml"
    p.write_text(EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    update_config_values(
        {
            "breakout.volume_mult": 2.0,
            "alerts.quiet_hours.enabled": True,
            "alerts.quiet_hours.start": time(22, 30),
            "alerts.events": ["breakout", "retest"],
            "backtest.risk_per_trade_pct": 0.5,
        },
        p,
    )
    cfg = load_config(p)
    assert cfg.breakout.volume_mult == 2.0
    assert cfg.alerts.quiet_hours.enabled and cfg.alerts.quiet_hours.start == time(22, 30)
    assert cfg.alerts.events == ["breakout", "retest"]
    text = p.read_text(encoding="utf-8")
    assert "risk_per_trade_pct: 0.5        # % of equity risked" in text  # comment kept
    old, new = EXAMPLE.read_text(encoding="utf-8").splitlines(), text.splitlines()
    assert sum(a != b for a, b in zip(old, new, strict=True)) == 5  # only those lines changed

    with pytest.raises(ValidationError):
        update_config_values({"alerts.min_confidence": 500}, p)  # > 100 is refused
    assert load_config(p).alerts.min_confidence == 60  # file untouched


def test_env_overrides_config_and_db(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    from sanjotracker.config import CONFIG_ENV, DB_ENV

    p = tmp_path / "c.yaml"
    p.write_text(EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setenv(CONFIG_ENV, str(p))
    monkeypatch.setenv(DB_ENV, str(tmp_path / "x.db"))
    cfg = load_config()
    assert cfg.db_file == tmp_path / "x.db"


def test_update_config_values_writes_null(tmp_path) -> None:  # noqa: ANN001
    from sanjotracker.config import update_config_values

    p = tmp_path / "config.yaml"
    p.write_text(EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    update_config_values({"paper.stop_atr": None}, p)
    assert load_config(p).paper.stop_atr is None
    assert "  stop_atr: null" in p.read_text(encoding="utf-8")
