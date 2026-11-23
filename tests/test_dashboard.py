"""Streamlit dashboard: every page renders on a synthetic DB (AppTest, no browser),
and the Settings page writes only to a temporary config file."""

from pathlib import Path

import numpy as np
import pytest
import yaml
from streamlit.testing.v1 import AppTest

from sanjotracker.alerts.dashboard_feed import AlertFeed
from sanjotracker.config import CONFIG_ENV, DB_ENV, PROJECT_ROOT, load_config
from sanjotracker.data.store import CandleStore
from sanjotracker.paper.book import PaperBook
from tests.conftest import T0, make_candles
from tests.test_alerts import make_event

APP = str(PROJECT_ROOT / "app" / "Home.py")
SYM = "AAA/USDT"


@pytest.fixture
def app_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    raw = yaml.safe_load((PROJECT_ROOT / "config.example.yaml").read_text(encoding="utf-8"))
    raw["watchlist"] = {"nse": [], "crypto": [SYM]}
    raw["benchmarks"] = {"nse": "^NSEI", "crypto": SYM}
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    db = tmp_path / "test.db"
    monkeypatch.setenv(CONFIG_ENV, str(cfg_path))
    monkeypatch.setenv(DB_ENV, str(db))

    store = CandleStore(db)
    for tf, n in (("1d", 500), ("1w", 120), ("1h", 400), ("15m", 400)):
        df = make_candles(T0, n, tf, seed=3, base=100)
        df[["open", "high", "low", "close"]] += 10 * np.sin(np.arange(n) / 25)[:, None]
        store.upsert(SYM, tf, df, "test")
    ev = make_event(
        symbol=SYM, timeframe="1d", bar_ts=make_candles(T0, 500, "1d")["timestamp"].iloc[-5]
    )
    AlertFeed(engine=store.engine).record(ev, ev.bar_ts)

    book = PaperBook(store.engine)
    d1 = make_candles(T0, 500, "1d")["timestamp"]
    base = {
        "setup_id": "s1",
        "label": "ema_crossover [2ATR stop, 2R target]",
        "symbol": SYM,
        "timeframe": "1d",
        "direction": "long",
        "entry_price": 100.0,
        "stop": 95.0,
        "target": 110.0,
        "qty": 10.0,
        "risk_amount": 50.0,
    }
    closed_id = book.insert(
        {
            **base,
            "status": "closed",
            "signal_ts": int(d1.iloc[300].timestamp()),
            "entry_ts": int(d1.iloc[301].timestamp()),
            "exit_ts": int(d1.iloc[310].timestamp()),
            "exit_price": 110.0,
            "exit_reason": "target",
            "fees": 1.0,
            "net_pnl": 99.0,
            "r_multiple": 1.98,
        }
    )
    book.insert(
        {
            **base,
            "status": "open",
            "signal_ts": int(d1.iloc[480].timestamp()),
            "entry_ts": int(d1.iloc[481].timestamp()),
        }
    )
    book.save_journal(closed_id, "clean breakout", ["textbook"], 5)
    return cfg_path


def _run(page: str | None = None) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120).run()
    if page:
        at.switch_page(f"app_pages/{page}.py").run()
    return at


@pytest.mark.parametrize(
    "page", [None, "charts", "scanner", "backtest", "paper", "journal", "settings"]
)
def test_page_renders_without_errors(app_env: Path, page: str | None) -> None:
    at = _run(page)
    assert not at.exception, [e.value for e in at.exception]


def test_alerts_page_shows_the_stored_alert(app_env: Path) -> None:
    at = _run()
    labels = {m.label: str(m.value) for m in at.metric}
    assert labels["Alerts, last 7 days"] == "1"
    assert len(at.dataframe) >= 2  # alert feed + watchlist


def test_backtest_page_reports_metrics(app_env: Path) -> None:
    at = _run("backtest")
    labels = {m.label for m in at.metric}
    assert {"Total return", "Sharpe", "Max drawdown", "Trades"} <= labels
    at.segmented_control(key="bt_mode").set_value("Leaderboard").run()
    assert not at.exception


def test_settings_saves_thresholds_to_the_temp_config(app_env: Path) -> None:
    before = app_env.read_text(encoding="utf-8")
    at = _run("settings")
    at.number_input(key="s_vm").set_value(2.5)
    at.slider(key="s_mc").set_value(75)
    at.button(key="FormSubmitter:thr_form-Save thresholds").click().run()
    assert not at.exception
    cfg = load_config(app_env)
    assert cfg.breakout.volume_mult == 2.5 and cfg.alerts.min_confidence == 75
    assert before != app_env.read_text(encoding="utf-8")
    assert (app_env.parent / "config.yaml.bak").exists()


def test_settings_rejects_bad_ticker(app_env: Path) -> None:
    at = _run("settings")
    at.multiselect(key="set_nse").set_value(["NOTATICKER"])
    at.button(key="FormSubmitter:wl_form-Save watchlist").click().run()
    assert at.error and "Not valid" in at.error[0].value
    assert load_config(app_env).watchlist.nse == []


def test_paper_page_shows_account(app_env: Path) -> None:
    at = _run("paper")
    m = {x.label: str(x.value) for x in at.metric}
    assert m["Open positions"] == "1" and m["Closed trades"] == "1"
    assert m["Realized P&L"] == "+99"


def test_journal_shows_tags(app_env: Path) -> None:
    at = _run("journal")
    assert not at.exception
    assert any("textbook" in str(df.value.to_dict()) for df in at.dataframe)


def test_settings_paper_tab_saves(app_env: Path) -> None:
    at = _run("settings")
    at.number_input(key="s_psa").set_value(3.0)  # (AppTest can't clear a number_input)
    at.number_input(key="s_pc").set_value(50_000.0)
    at.button(key="FormSubmitter:thr_form-Save thresholds").click().run()
    assert not at.exception
    cfg = load_config(app_env)
    assert cfg.paper.stop_atr == 3.0 and cfg.paper.capital == 50_000
