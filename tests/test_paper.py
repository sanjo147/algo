"""Paper trading: live stepping == one replay, no back-dated trades, ATR stop/target,
notifications once, idempotent reruns, data revisions, journal, scheduler hook."""

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from sanjotracker.analysis.indicators import atr
from sanjotracker.config import PROJECT_ROOT, PaperSetup, load_config
from sanjotracker.data.store import CandleStore
from sanjotracker.paper.book import PaperBook
from sanjotracker.paper.trader import (
    AtrRiskOverride,
    PaperTrader,
    build_strategy,
    expand_symbols,
    paper_summary,
)
from sanjotracker.strategies import StrategyContext, get_strategy
from tests.conftest import T0, make_candles

SYM = "AAA.NS"
H = timedelta(days=1)  # one daily bar


def _cfg(**paper):  # noqa: ANN003, ANN202
    cfg = load_config(PROJECT_ROOT / "config.example.yaml")
    cfg.watchlist.nse = [SYM]
    cfg.paper.setups = [
        PaperSetup(strategy="ema_crossover", timeframe="1d", params={"fast": 5, "slow": 20})
    ]
    for k, v in paper.items():
        setattr(cfg.paper, k, v)
    return cfg


def _series(n: int = 600, seed: int = 4) -> pd.DataFrame:
    df = make_candles(T0, n, "1d", seed=seed)
    df[["open", "high", "low", "close"]] += 8 * np.sin(np.arange(n) / 30)[:, None]
    return df


class FakeNotifier:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def send_text(self, text: str) -> None:
        self.texts.append(text)

    def send_photo(self, png: bytes, caption: str) -> None:
        self.texts.append(caption)


COLS = [
    "direction",
    "entry_ts",
    "entry_price",
    "stop",
    "exit_ts",
    "exit_price",
    "exit_reason",
    "net_pnl",
    "status",
]


def _trades(store: CandleStore) -> pd.DataFrame:
    return PaperBook(store.engine).trades()[COLS].sort_values("entry_ts").reset_index(drop=True)


def test_live_stepping_equals_one_replay(tmp_path) -> None:  # noqa: ANN001
    """THE paper proof: stepping after every few new bars gives exactly the trades of a
    single replay over the same data (so paper = backtest rules, no lookahead)."""
    cfg, full = _cfg(), _series()
    start = full["timestamp"].iloc[200]

    live = CandleStore(tmp_path / "live.db")
    live.upsert(SYM, "1d", full.iloc[:200], "t")
    PaperTrader(cfg, live).run(now=start)
    for k in range(205, 601, 5):
        live.upsert(SYM, "1d", full.iloc[:k], "t")
        PaperTrader(cfg, live).run(now=full["timestamp"].iloc[k - 1] + H)

    once = CandleStore(tmp_path / "once.db")
    once.upsert(SYM, "1d", full.iloc[:200], "t")
    PaperTrader(cfg, once).run(now=start)
    once.upsert(SYM, "1d", full, "t")
    PaperTrader(cfg, once).run(now=full["timestamp"].iloc[-1] + H)

    a, b = _trades(live), _trades(once)
    assert len(a) >= 3
    pd.testing.assert_frame_equal(a, b)


def test_no_back_dated_trades(store) -> None:  # noqa: ANN001
    full = _series()
    store.upsert(SYM, "1d", full, "t")
    now = full["timestamp"].iloc[-1] + H
    PaperTrader(_cfg(), store).run(now=now)
    assert PaperBook(store.engine).trades().empty  # history before the setup never trades
    # later runs keep the original start time
    PaperTrader(_cfg(), store).run(now=now + 10 * H)
    assert PaperBook(store.engine).setups()["start_ts"].iloc[0] == now


def test_atr_override_sets_stop_and_target() -> None:
    df = _series(300)
    inner = get_strategy("ema_crossover", fast=5, slow=20)
    ctx = StrategyContext(SYM, "1d")
    sig = AtrRiskOverride(inner, 2.0, 3.0).generate_signals(df, ctx)
    e = sig["entry"] != 0
    want = df["close"] - np.sign(sig["entry"]) * 2.0 * atr(df)
    assert e.any()
    np.testing.assert_allclose(sig.loc[e, "stop"], want[e])
    assert (sig.loc[e, "target_r"] == 3.0).all()
    assert "2ATR stop" in AtrRiskOverride(inner, 2.0, 3.0).label()
    cfg = _cfg(stop_atr=None, target_r=None)
    assert not isinstance(build_strategy(cfg.paper.setups[0], cfg), AtrRiskOverride)


def test_notifications_once_and_reruns_idempotent(store) -> None:  # noqa: ANN001
    cfg, full = _cfg(), _series()
    store.upsert(SYM, "1d", full.iloc[:200], "t")
    n = FakeNotifier()
    PaperTrader(cfg, store, n).run(now=full["timestamp"].iloc[200])
    for k in (300, 450, 600):
        store.upsert(SYM, "1d", full.iloc[:k], "t")
        PaperTrader(cfg, store, n).run(now=full["timestamp"].iloc[k - 1] + H)
    t = PaperBook(store.engine).trades()
    assert len(t) >= 3
    entries = sum("PAPER 🟢" in x or "PAPER 🔴" in x for x in n.texts)
    exits = sum("PAPER EXIT" in x for x in n.texts)
    assert entries == len(t) and exits == (t["status"] == "closed").sum()
    assert t["notified_entry"].all()

    before, msgs = _trades(store), len(n.texts)
    PaperTrader(cfg, store, n).run(now=full["timestamp"].iloc[-1] + H)  # nothing new
    pd.testing.assert_frame_equal(before, _trades(store))
    assert len(n.texts) == msgs


def test_open_trade_closes_on_a_later_run(store) -> None:  # noqa: ANN001
    cfg, full = _cfg(), _series()
    store.upsert(SYM, "1d", full.iloc[:200], "t")
    PaperTrader(cfg, store).run(now=full["timestamp"].iloc[200])
    for k in range(201, 601):
        store.upsert(SYM, "1d", full.iloc[:k], "t")
        PaperTrader(cfg, store).run(now=full["timestamp"].iloc[k - 1] + H)
        if (PaperBook(store.engine).trades()["status"] == "open").any():
            break
    else:
        pytest.skip("series never left a trade open at a step")
    tid = int(PaperBook(store.engine).trades(status="open")["id"].iloc[0])
    store.upsert(SYM, "1d", full, "t")
    PaperTrader(cfg, store).run(now=full["timestamp"].iloc[-1] + H)
    t = PaperBook(store.engine).trades().set_index("id").loc[tid]
    assert t["status"] == "closed" and t["exit_reason"] in ("stop", "target", "signal")


def test_revised_history_closes_orphan_open_trade(store) -> None:  # noqa: ANN001
    cfg = _cfg()
    book = PaperBook(store.engine)
    store.upsert(SYM, "1d", _series(), "t")
    trader = PaperTrader(cfg, store)
    trader.run(now=T0)  # registers the setup from the start of the data
    sid = book.setups()["setup_id"].iloc[0]
    fake_entry = int(pd.Timestamp("2026-02-05 18:30", tz="UTC").timestamp())
    book.insert(
        {
            "setup_id": sid,
            "label": "x",
            "symbol": SYM,
            "timeframe": "1d",
            "direction": "long",
            "status": "open",
            "signal_ts": fake_entry - 3600,
            "entry_ts": fake_entry,
            "entry_price": 100.0,
            "stop": 95.0,
            "qty": 1.0,
            "risk_amount": 5.0,
        }
    )
    trader.run(now=T0 + 700 * H)
    orphan = book.trades().set_index("entry_ts").loc[pd.Timestamp(fake_entry, unit="s", tz="UTC")]
    assert orphan["status"] == "closed" and orphan["exit_reason"] == "data_revised"


def test_journal_upsert_and_tags(store) -> None:  # noqa: ANN001
    book = PaperBook(store.engine)
    tid = book.insert(
        {
            "setup_id": "s",
            "label": "x",
            "symbol": SYM,
            "timeframe": "1d",
            "direction": "long",
            "status": "closed",
            "signal_ts": 1,
            "entry_ts": 2,
            "entry_price": 10.0,
            "stop": 9.0,
            "qty": 1.0,
            "risk_amount": 1.0,
            "exit_ts": 3,
            "exit_price": 12.0,
            "net_pnl": 2.0,
            "r_multiple": 2.0,
        }
    )
    book.save_journal(tid, "first", [" Textbook ", "gap", "gap"], 4)
    book.save_journal(tid, "edited", ["textbook"], None)
    row = book.trades().iloc[0]
    assert row["notes"] == "edited" and row["tags"] == "textbook" and pd.isna(row["rating"])
    book.save_journal(tid, "x", ["news", "Gap"], 5)
    assert book.all_tags() == ["gap", "news"]


def test_summary_counts_open_pnl(store) -> None:  # noqa: ANN001
    book = PaperBook(store.engine)
    store.upsert(
        SYM,
        "1d",
        pd.DataFrame(
            {
                "timestamp": [pd.Timestamp(T0)],
                "open": [110.0],
                "high": [111.0],
                "low": [109.0],
                "close": [110.0],
                "volume": [1.0],
            }
        ),
        "t",
    )
    book.insert(
        {
            "setup_id": "s",
            "label": "x",
            "symbol": SYM,
            "timeframe": "1d",
            "direction": "long",
            "status": "open",
            "signal_ts": 1,
            "entry_ts": 2,
            "entry_price": 100.0,
            "stop": 95.0,
            "qty": 2.0,
            "risk_amount": 10.0,
        }
    )
    s = paper_summary(book, store, 1000)
    assert s["unrealized"] == 20 and s["equity"] == 1020
    assert s["open"]["unrealized_r"].iloc[0] == 2.0


def test_expand_symbols() -> None:
    cfg = _cfg()
    cfg.watchlist.nse = ["TCS.NS"]
    s = PaperSetup(strategy="ema_crossover", symbols=["watchlist", "infy", "TCS.NS"])
    assert expand_symbols(s, cfg) == ["TCS.NS", "INFY.NS"]
    assert expand_symbols(PaperSetup(strategy="x"), cfg) == ["TCS.NS"]


def test_scheduler_job_steps_paper(store) -> None:  # noqa: ANN001
    from sanjotracker.scheduler import AlertService

    cfg = _cfg()
    cfg.paper.setups = [
        PaperSetup(strategy="ema_crossover", timeframe="1d", params={"fast": 5, "slow": 20})
    ]
    full = make_candles(T0, 400, "1d", seed=2)
    full[["open", "high", "low", "close"]] += 15 * np.sin(np.arange(400) / 20)[:, None]
    store.upsert(SYM, "1d", full.iloc[:200], "t")

    class NoScan:
        errors: dict = {}

        def scan(self, style):  # noqa: ANN001, ANN201
            return []

        def benchmark_regimes(self):  # noqa: ANN201
            return {}

    n = FakeNotifier()
    svc = AlertService(
        n,
        load_cfg=lambda: cfg,
        store=store,
        scanner_factory=lambda c, s: NoScan(),
        chart=lambda e: b"",
    )
    svc.run_job("nse_swing", now=full["timestamp"].iloc[199] + timedelta(days=1), force=True)
    store.upsert(SYM, "1d", full, "t")
    info = svc.run_job("nse_swing", now=full["timestamp"].iloc[-1] + timedelta(days=1), force=True)
    assert info.paper_opened >= 1 and not info.errors
    assert any("PAPER" in t for t in n.texts)
