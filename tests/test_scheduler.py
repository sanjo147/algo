"""Scheduler timing rules and the scan -> rules -> feed -> Telegram pipeline (all offline)."""

from datetime import UTC, date, datetime

import pytest

from sanjotracker.alerts.dashboard_feed import AlertFeed
from sanjotracker.config import PROJECT_ROOT, load_config, save_watchlist
from sanjotracker.scheduler import IST, AlertService, build_scheduler, nse_intraday_due
from tests.test_alerts import make_event

CFG = load_config(PROJECT_ROOT / "config.example.yaml")
CFG.paper.enabled = False  # paper trading has its own tests (test_paper.py)


def ist(y: int, mo: int, d: int, h: int, mi: int) -> datetime:
    return datetime(y, mo, d, h, mi, tzinfo=IST)


@pytest.mark.parametrize(
    ("now", "due"),
    [
        (ist(2026, 11, 16, 9, 32), True),  # Monday, first 15m bar closed
        (ist(2026, 11, 16, 15, 32), True),  # last bar
        (ist(2026, 11, 16, 9, 17), False),  # before the first bar closes
        (ist(2026, 11, 16, 15, 47), False),  # after close
        (ist(2026, 11, 15, 11, 2), False),  # Sunday
        (ist(2026, 12, 25, 11, 2), False),  # holiday in config
    ],
)
def test_nse_intraday_due(now: datetime, due: bool) -> None:
    assert (nse_intraday_due(now, CFG) is None) is due


class FakeScanner:
    def __init__(self, events, fail: bool = False) -> None:  # noqa: ANN001
        self.events, self.fail, self.errors = events, fail, {}

    def scan(self, style, market):  # noqa: ANN001, ANN201
        if self.fail:
            raise RuntimeError("provider down")
        self.errors = {"BAD.NS": "no data"}
        return list(self.events)

    def benchmark_regimes(self, market):  # noqa: ANN001, ANN201
        return {}


class FakeNotifier:
    def __init__(self) -> None:
        self.texts: list[str] = []
        self.photos: list[str] = []

    def send_text(self, text: str) -> None:
        self.texts.append(text)

    def send_photo(self, png: bytes, caption: str) -> None:
        self.photos.append(caption)


def _service(store, events, notifier=None, fail=False):  # noqa: ANN001, ANN202
    notifier = notifier or FakeNotifier()
    svc = AlertService(
        notifier,
        load_cfg=lambda: CFG,
        store=store,
        scanner_factory=lambda c, s: FakeScanner(events, fail),
        chart=lambda e: b"png",
    )
    return svc, notifier


MONDAY_NOON = ist(2026, 11, 16, 12, 2).astimezone(UTC)


def test_job_sends_once_and_records(store) -> None:  # noqa: ANN001
    svc, n = _service(store, [make_event(), make_event(confidence=30)])
    info = svc.run_job("crypto_swing", now=MONDAY_NOON)
    assert (info.events, info.stored, info.sent) == (2, 1, 1)
    assert len(n.photos) == 1 and "RELIANCE.NS" in n.photos[0]
    assert info.errors == {"BAD.NS": "no data"}
    feed = AlertFeed(engine=store.engine)
    assert bool(feed.recent()["telegram_sent"].iloc[0])
    again = svc.run_job("crypto_swing", now=MONDAY_NOON)  # same bar again -> nothing new
    assert again.sent == 0 and len(n.photos) == 1


def test_nse_intraday_skipped_outside_session(store) -> None:  # noqa: ANN001
    svc, n = _service(store, [make_event()])
    info = svc.run_job("nse_intraday", now=ist(2026, 11, 15, 12, 2))  # Sunday
    assert info.skipped == "not an NSE trading day" and n.photos == []
    forced = svc.run_job("nse_intraday", now=ist(2026, 11, 15, 12, 2), force=True)
    assert forced.sent == 1


def test_scan_crash_does_not_raise(store) -> None:  # noqa: ANN001
    svc, _ = _service(store, [], fail=True)
    info = svc.run_job("crypto_intraday", now=MONDAY_NOON)
    assert "scan" in info.errors


def test_telegram_failure_is_recorded(store) -> None:  # noqa: ANN001
    class Broken(FakeNotifier):
        def send_photo(self, png: bytes, caption: str) -> None:
            raise RuntimeError("telegram down")

    svc, _ = _service(store, [make_event()], notifier=Broken())
    info = svc.run_job("crypto_swing", now=MONDAY_NOON)
    assert info.sent == 0 and info.stored == 1
    row = AlertFeed(engine=store.engine).recent().iloc[0]
    assert not row["telegram_sent"] and "telegram error" in row["note"]


def test_nse_swing_sends_daily_summary(store) -> None:  # noqa: ANN001
    svc, n = _service(store, [make_event()])
    svc.run_job("nse_swing", now=ist(2026, 11, 16, 15, 46).astimezone(UTC))
    assert n.texts and "Daily summary" in n.texts[-1] and "Alerts today: 1" in n.texts[-1]


def test_scheduler_has_four_jobs_at_the_right_times(store) -> None:  # noqa: ANN001
    svc, _ = _service(store, [])
    sched = build_scheduler(svc, CFG)
    jobs = {j.id: j for j in sched.get_jobs()}
    assert set(jobs) == {"nse_intraday", "nse_swing", "crypto_intraday", "crypto_swing"}
    start = ist(2026, 11, 15, 12, 0)  # Sunday noon
    nxt = {k: j.trigger.get_next_fire_time(None, start) for k, j in jobs.items()}
    assert nxt["nse_intraday"] == ist(2026, 11, 16, 9, 2)  # Monday (skips weekend)
    assert nxt["nse_swing"] == ist(2026, 11, 16, 15, 46)
    assert nxt["crypto_intraday"] == ist(2026, 11, 15, 12, 0).replace(second=30)
    assert nxt["crypto_swing"] == ist(2026, 11, 16, 5, 35)


def test_save_watchlist_keeps_other_settings(tmp_path) -> None:  # noqa: ANN001
    p = tmp_path / "config.yaml"
    p.write_text((PROJECT_ROOT / "config.example.yaml").read_text(encoding="utf-8"), "utf-8")
    save_watchlist(["TCS.NS"], [], p)
    cfg = load_config(p)
    assert cfg.watchlist.nse == ["TCS.NS"] and cfg.watchlist.crypto == []
    assert cfg.nse_market.holidays == [date(2026, 12, 25), date(2027, 1, 26)]
    assert "# " in p.read_text(encoding="utf-8")  # comments survive
    assert (tmp_path / "config.yaml.bak").exists()
