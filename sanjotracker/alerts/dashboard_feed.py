"""Writes alerts to the `alerts` table (same SQLite DB) for the dashboard.

Every event that passes the alert rules is stored once here, whether or not it
also went to Telegram (quiet hours, Telegram off). The table doubles as the
memory for cooldown and duplicate checks, so they survive a restart.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import (
    Boolean,
    Column,
    Engine,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    func,
    select,
    update,
)

from sanjotracker.analysis.breakout import Event
from sanjotracker.data.store import make_engine

metadata = MetaData()

alerts = Table(
    "alerts",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", Integer, nullable=False, index=True),  # epoch s, UTC
    Column("key", String, nullable=False, index=True),
    Column("symbol", String, nullable=False),
    Column("timeframe", String, nullable=False),
    Column("style", String),
    Column("event", String, nullable=False),
    Column("direction", String, nullable=False),
    Column("bar_ts", Integer, nullable=False),
    Column("price", Float, nullable=False),
    Column("zone_low", Float, nullable=False),
    Column("zone_high", Float, nullable=False),
    Column("confidence", Integer, nullable=False),
    Column("volume_ratio", Float),
    Column("reasons", Text, nullable=False),  # JSON list
    Column("payload", Text, nullable=False),  # full Event JSON
    Column("telegram_sent", Boolean, nullable=False, default=False),
    Column("note", String),  # e.g. "quiet hours", "telegram error"
)

FEED_COLUMNS = [c.name for c in alerts.columns if c.name != "payload"]


def _epoch(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp())


class AlertFeed:
    """Read/write access to the alerts table."""

    def __init__(self, db_path: str | Path | None = None, engine: Engine | None = None) -> None:
        if engine is None:
            if db_path is None:
                raise ValueError("give db_path or engine")
            engine = make_engine(db_path)
        self.engine = engine
        metadata.create_all(engine)

    def record(
        self, event: Event, now: datetime, telegram_sent: bool = False, note: str | None = None
    ) -> int:
        """Store one alert; returns its id."""
        row = {
            "created_at": _epoch(now),
            "key": event.key,
            "symbol": event.symbol,
            "timeframe": event.timeframe,
            "style": event.style,
            "event": event.event.value,
            "direction": event.direction,
            "bar_ts": _epoch(event.bar_ts),
            "price": event.price,
            "zone_low": event.zone_low,
            "zone_high": event.zone_high,
            "confidence": event.confidence,
            "volume_ratio": event.volume_ratio,
            "reasons": json.dumps(event.reasons),
            "payload": event.model_dump_json(),
            "telegram_sent": telegram_sent,
            "note": note,
        }
        with self.engine.begin() as conn:
            return int(conn.execute(alerts.insert().values(**row)).inserted_primary_key[0])

    def mark_sent(self, alert_id: int, sent: bool, note: str | None = None) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                update(alerts).where(alerts.c.id == alert_id).values(telegram_sent=sent, note=note)
            )

    def last_alert_at(self, key: str) -> datetime | None:
        """When this symbol+timeframe+event+zone was last alerted (None = never)."""
        q = select(func.max(alerts.c.created_at)).where(alerts.c.key == key)
        with self.engine.connect() as conn:
            v = conn.execute(q).scalar()
        return None if v is None else datetime.fromtimestamp(v, UTC)

    def seen(self, event: Event) -> bool:
        """True if this exact event (same key, same bar) is already stored."""
        q = select(func.count()).where(
            alerts.c.key == event.key, alerts.c.bar_ts == _epoch(event.bar_ts)
        )
        with self.engine.connect() as conn:
            return bool(conn.execute(q).scalar())

    def recent(self, since: datetime | None = None, limit: int = 200) -> pd.DataFrame:
        """Newest first. created_at / bar_ts come back as UTC timestamps."""
        q = select(*[alerts.c[c] for c in FEED_COLUMNS]).order_by(alerts.c.id.desc()).limit(limit)
        if since is not None:
            q = q.where(alerts.c.created_at >= _epoch(since))
        with self.engine.connect() as conn:
            rows = conn.execute(q).all()
        df = pd.DataFrame(rows, columns=FEED_COLUMNS)
        if not df.empty:
            for c in ("created_at", "bar_ts"):
                df[c] = pd.to_datetime(df[c], unit="s", utc=True)
            df["reasons"] = df["reasons"].map(json.loads)
        return df

    def events(self, since: datetime | None = None, limit: int = 200) -> list[Event]:
        """Stored alerts as Event objects (newest first)."""
        q = select(alerts.c.payload).order_by(alerts.c.id.desc()).limit(limit)
        if since is not None:
            q = q.where(alerts.c.created_at >= _epoch(since))
        with self.engine.connect() as conn:
            return [Event.model_validate_json(p) for (p,) in conn.execute(q).all()]
