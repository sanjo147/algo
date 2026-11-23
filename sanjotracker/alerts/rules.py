"""Which events trigger alerts; dedupe, cooldown and quiet hours.

Order of checks for each event:
1. event type enabled?            no  -> dropped
2. confidence >= min_confidence?  no  -> dropped
3. same event on the same bar already stored? -> dropped (scheduler re-runs)
4. same symbol+timeframe+event+zone alerted within cooldown_minutes? -> dropped
5. otherwise -> stored for the dashboard; sent to Telegram unless it is quiet
   hours or Telegram is off.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

from loguru import logger
from pydantic import BaseModel

from sanjotracker.alerts.dashboard_feed import AlertFeed
from sanjotracker.analysis.breakout import Event
from sanjotracker.config import AlertsConfig, QuietHours

Action = Literal["send", "store_only", "drop"]


class Decision(BaseModel):
    action: Action
    reason: str


def in_quiet_hours(now: datetime, qh: QuietHours, tz: str = "Asia/Kolkata") -> bool:
    """True if local time is inside [start, end); handles windows across midnight."""
    if not qh.enabled:
        return False
    t: time = now.astimezone(ZoneInfo(tz)).time()
    if qh.start <= qh.end:
        return qh.start <= t < qh.end
    return t >= qh.start or t < qh.end


class AlertRules:
    def __init__(self, cfg: AlertsConfig, feed: AlertFeed, tz: str = "Asia/Kolkata") -> None:
        self.cfg = cfg
        self.feed = feed
        self.tz = tz

    def decide(self, event: Event, now: datetime) -> Decision:
        if event.event.value not in self.cfg.events:
            return Decision(action="drop", reason=f"{event.event.value} alerts are off")
        if event.confidence < self.cfg.min_confidence:
            return Decision(
                action="drop", reason=f"confidence {event.confidence} < {self.cfg.min_confidence}"
            )
        if self.feed.seen(event):
            return Decision(action="drop", reason="already alerted for this bar")
        last = self.feed.last_alert_at(event.key)
        if last is not None and now - last < timedelta(minutes=self.cfg.cooldown_minutes):
            return Decision(action="drop", reason=f"cooldown (last alert {last:%H:%M} UTC)")
        if not self.cfg.telegram_enabled:
            return Decision(action="store_only", reason="telegram disabled")
        if in_quiet_hours(now, self.cfg.quiet_hours, self.tz):
            return Decision(action="store_only", reason="quiet hours")
        return Decision(action="send", reason="ok")

    def process(self, events: list[Event], now: datetime) -> list[tuple[Event, Decision, int]]:
        """Decide + store. Returns (event, decision, alert_id) for stored events only.

        Events are processed strongest first, so if two share a key in one batch
        the stronger one wins and the other hits the cooldown.
        """
        out: list[tuple[Event, Decision, int]] = []
        for ev in sorted(events, key=lambda e: -e.confidence):
            d = self.decide(ev, now)
            if d.action == "drop":
                logger.debug("skip {} {} {}: {}", ev.symbol, ev.timeframe, ev.event.value, d.reason)
                continue
            note = None if d.action == "send" else d.reason
            alert_id = self.feed.record(ev, now, telegram_sent=False, note=note)
            out.append((ev, d, alert_id))
        return out
