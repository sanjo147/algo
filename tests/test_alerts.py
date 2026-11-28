"""Alert feed, rules (cooldown / dedupe / quiet hours), Telegram sender and bot actions."""

import json
from datetime import UTC, datetime, time, timedelta

import httpx
import pandas as pd
import pytest
from pydantic import SecretStr

from sanjotracker.alerts.dashboard_feed import AlertFeed
from sanjotracker.alerts.rules import AlertRules, in_quiet_hours
from sanjotracker.alerts.telegram_bot import (
    BotActions,
    TelegramError,
    TelegramNotifier,
    format_event,
    send_event,
)
from sanjotracker.analysis.breakout import Event, EventType
from sanjotracker.config import PROJECT_ROOT, AlertsConfig, QuietHours, load_config
from tests.conftest import T0, make_candles

NOW = datetime(2026, 11, 16, 6, 0, tzinfo=UTC)  # 11:30 IST, Monday
TOKEN = "123456:SECRET-TOKEN-abc"


def make_event(**kw) -> Event:  # noqa: ANN003
    base = dict(
        symbol="RELIANCE.NS",
        timeframe="1d",
        event=EventType.BREAKOUT,
        direction="up",
        bar_ts=datetime(2026, 11, 13, 18, 30, tzinfo=UTC),
        price=1350.0,
        zone_low=1334.4,
        zone_high=1345.9,
        zone_type="R",
        zone_source="pivot",
        zone_touches=5,
        zone_strength=92,
        volume_ratio=2.1,
        body_atr=1.2,
        atr=20.0,
        confidence=80,
        reasons=["zone 1,334.40–1,345.90 (5 touches, strength 92)", "volume 2.1x <avg>"],
    )
    base.update(kw)
    return Event(**base)


@pytest.fixture
def feed(store) -> AlertFeed:  # noqa: ANN001
    return AlertFeed(engine=store.engine)


# ------------------------------------------------------------------ feed


def test_feed_record_and_read_back(feed: AlertFeed) -> None:
    e = make_event()
    aid = feed.record(e, NOW)
    feed.mark_sent(aid, True)
    df = feed.recent()
    assert len(df) == 1 and bool(df["telegram_sent"].iloc[0])
    assert df["reasons"].iloc[0] == e.reasons
    assert feed.events()[0] == e
    assert feed.seen(e)
    assert feed.last_alert_at(e.key) == NOW


# ------------------------------------------------------------------ rules


def _rules(feed: AlertFeed, **cfg) -> AlertRules:  # noqa: ANN003
    return AlertRules(AlertsConfig(**cfg), feed)


def test_low_confidence_dropped(feed: AlertFeed) -> None:
    assert _rules(feed).decide(make_event(confidence=40), NOW).action == "drop"


def test_disabled_event_type_dropped(feed: AlertFeed) -> None:
    r = _rules(feed, events=["breakout"])
    assert r.decide(make_event(event=EventType.APPROACHING), NOW).action == "drop"
    assert r.decide(make_event(), NOW).action == "send"


def test_same_bar_is_not_alerted_twice(feed: AlertFeed) -> None:
    r = _rules(feed, cooldown_minutes=0)
    assert len(r.process([make_event()], NOW)) == 1
    assert r.process([make_event()], NOW + timedelta(hours=5)) == []


def test_cooldown_blocks_then_expires(feed: AlertFeed) -> None:
    r = _rules(feed, cooldown_minutes=240)
    e1 = make_event()
    e2 = make_event(bar_ts=e1.bar_ts + timedelta(days=1))  # new bar, same zone/event
    r.process([e1], NOW)
    assert r.decide(e2, NOW + timedelta(hours=3)).reason.startswith("cooldown")
    assert r.decide(e2, NOW + timedelta(hours=5)).action == "send"


def test_cooldown_is_per_zone_and_event(feed: AlertFeed) -> None:
    r = _rules(feed)
    r.process([make_event()], NOW)
    other_zone = make_event(zone_low=1400, zone_high=1410)
    other_event = make_event(event=EventType.RETEST)
    assert r.decide(other_zone, NOW).action == "send"
    assert r.decide(other_event, NOW).action == "send"


def test_strongest_wins_within_batch(feed: AlertFeed) -> None:
    weak = make_event(confidence=65, bar_ts=datetime(2026, 11, 12, 18, 30, tzinfo=UTC))
    strong = make_event(confidence=90)
    out = _rules(feed).process([weak, strong], NOW)
    assert [e.confidence for e, _, _ in out] == [90]


@pytest.mark.parametrize(
    ("ist", "expected"),
    [(time(23, 30), True), (time(2, 0), True), (time(7, 0), False), (time(12, 0), False)],
)
def test_quiet_hours_across_midnight(ist: time, expected: bool) -> None:
    qh = QuietHours(enabled=True, start=time(23, 0), end=time(7, 0))
    now = pd.Timestamp.combine(pd.Timestamp("2026-11-16").date(), ist).tz_localize("Asia/Kolkata")
    assert in_quiet_hours(now.to_pydatetime(), qh) is expected


def test_quiet_hours_store_only(feed: AlertFeed) -> None:
    r = _rules(feed, quiet_hours=QuietHours(enabled=True, start=time(0, 0), end=time(23, 59)))
    (_, d, _), *_ = r.process([make_event()], NOW)
    assert d.action == "store_only" and d.reason == "quiet hours"
    assert feed.recent()["note"].iloc[0] == "quiet hours"


# ------------------------------------------------------------------ telegram


def test_format_event_has_required_fields_and_escapes_html() -> None:
    text = format_event(make_event())
    for part in ("RELIANCE.NS", "1d", "BREAKOUT", "1,350.00", "1,334.40", "2.1x", "80", "🚀"):
        assert part in text
    assert "&lt;avg&gt;" in text and "<avg>" not in text


class Recorder:
    def __init__(self, responses: list[httpx.Response] | None = None) -> None:
        self.requests: list[httpx.Request] = []
        self.responses = responses or []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.responses:
            return self.responses.pop(0)
        return httpx.Response(200, json={"ok": True, "result": {}})


def _notifier(rec: Recorder) -> TelegramNotifier:
    client = httpx.Client(transport=httpx.MockTransport(rec))
    return TelegramNotifier(SecretStr(TOKEN), "42", client=client, pause=0)


def test_send_text_and_photo() -> None:
    rec = Recorder()
    n = _notifier(rec)
    n.send_text("hello")
    n.send_photo(b"\x89PNG", "caption")
    assert [r.url.path.rsplit("/", 1)[-1] for r in rec.requests] == ["sendMessage", "sendPhoto"]
    assert b"chat_id=42" in rec.requests[0].content


def test_long_caption_split_into_photo_plus_text() -> None:
    rec = Recorder()
    _notifier(rec).send_photo(b"png", "short head\n\n" + "x" * 2000)
    assert [r.url.path.rsplit("/", 1)[-1] for r in rec.requests] == ["sendPhoto", "sendMessage"]


def test_rate_limit_retried_once(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr("sanjotracker.alerts.telegram_bot.time.sleep", lambda s: None)
    rec = Recorder([httpx.Response(429, json={"ok": False, "parameters": {"retry_after": 1}})])
    _notifier(rec).send_text("hi")
    assert len(rec.requests) == 2


def test_errors_never_contain_the_token() -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"cannot reach {request.url}")

    n = TelegramNotifier(
        SecretStr(TOKEN), "42", client=httpx.Client(transport=httpx.MockTransport(boom)), pause=0
    )
    with pytest.raises(TelegramError) as ei:
        n.send_text("hi")
    assert TOKEN not in str(ei.value) and "***" in str(ei.value)

    rec = Recorder([httpx.Response(401, json={"ok": False, "description": "Unauthorized"})])
    with pytest.raises(TelegramError) as ei:
        _notifier(rec).send_text("hi")
    assert TOKEN not in str(ei.value)


def test_send_event_falls_back_to_text_when_chart_fails() -> None:
    rec = Recorder()

    def broken_chart(e: Event) -> bytes:
        raise RuntimeError("no chrome")

    send_event(_notifier(rec), make_event(), "Asia/Kolkata", broken_chart)
    assert rec.requests[0].url.path.endswith("sendMessage")


# ------------------------------------------------------------------ bot actions


def _actions(store, saved: list, valid: bool = True) -> BotActions:  # noqa: ANN001
    cfg = load_config(PROJECT_ROOT / "config.example.yaml")

    def save(nse: list[str]) -> None:
        saved.append(nse)
        cfg.watchlist.nse = nse

    return BotActions(
        load_cfg=lambda: cfg,
        store=store,
        save_watchlist=save,
        validate_symbol=lambda s: None if valid else "no data",
        status_text=lambda: "ok",
    )


def test_bot_add_remove_watchlist(store) -> None:  # noqa: ANN001
    saved: list = []
    a = _actions(store, saved)
    assert "Added WIPRO.NS" in a.add("wipro.ns")
    assert "Added TATAPOWER.NS" in a.add("tatapower")  # .NS added for you
    assert "TATAPOWER.NS" in saved[-1]
    assert "NSE-only" in a.add("DOGE/USDT")
    assert "already" in a.add("RELIANCE.NS")
    assert "Removed" in a.remove("WIPRO.NS") and "WIPRO.NS" not in saved[-1]
    assert "not on the watchlist" in a.remove("NOPE.NS")
    assert "RELIANCE.NS" in a.watchlist()


def test_bot_add_rejects_invalid_symbol(store) -> None:  # noqa: ANN001
    saved: list = []
    assert "Could not add" in _actions(store, saved, valid=False).add("XXXX.NS")
    assert saved == []


def test_bot_levels(store, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr("sanjotracker.alerts.telegram_bot.render_png", lambda fig: b"png")
    store.upsert("RELIANCE.NS", "1d", make_candles(T0, 300, "1d", base=1300), "test")
    text, png = _actions(store, []).levels("RELIANCE.NS")
    assert "last close" in text and png == b"png"
    text, png = _actions(store, []).levels("NODATA.NS")
    assert "No 1d data" in text and png is None


def test_event_json_roundtrip_in_feed(feed: AlertFeed) -> None:
    feed.record(make_event(), NOW)
    payload = feed.recent()
    assert json.dumps(payload["reasons"].iloc[0])
