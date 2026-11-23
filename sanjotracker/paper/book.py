"""Paper-trading tables: setups, trades and the trade journal (same SQLite DB).

paper_setups   one row per (strategy, params, risk overrides, symbol, timeframe), with
               the time it was first seen: trades only start after that moment.
paper_trades   one row per virtual trade (status open/closed). Closed trades never change.
paper_journal  your notes, tags and a 1-5 rating per trade (kept separate so a data
               refresh can never overwrite what you wrote).
"""

from __future__ import annotations

from datetime import UTC, datetime

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
    select,
    update,
)
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

metadata = MetaData()

setups = Table(
    "paper_setups",
    metadata,
    Column("setup_id", String, primary_key=True),
    Column("label", String, nullable=False),
    Column("strategy", String, nullable=False),
    Column("symbol", String, nullable=False),
    Column("timeframe", String, nullable=False),
    Column("params", Text, nullable=False),
    Column(
        "start_ts", Integer, nullable=False
    ),  # epoch s: first bar that may trade closes after this
    Column("last_run_ts", Integer),
)

trades = Table(
    "paper_trades",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("setup_id", String, nullable=False, index=True),
    Column("label", String, nullable=False),
    Column("symbol", String, nullable=False),
    Column("timeframe", String, nullable=False),
    Column("direction", String, nullable=False),
    Column("status", String, nullable=False),  # open | closed
    Column("signal_ts", Integer, nullable=False),
    Column("entry_ts", Integer, nullable=False),
    Column("entry_price", Float, nullable=False),
    Column("stop", Float, nullable=False),
    Column("target", Float),
    Column("qty", Float, nullable=False),
    Column("risk_amount", Float, nullable=False),
    Column("exit_ts", Integer),
    Column("exit_price", Float),
    Column("exit_reason", String),
    Column("fees", Float),
    Column("net_pnl", Float),
    Column("r_multiple", Float),
    Column("regime_trend", String),
    Column("regime_volatility", String),
    Column("regime_market", String),
    Column("notified_entry", Boolean, nullable=False, default=False),
    Column("notified_exit", Boolean, nullable=False, default=False),
    Column("created_at", Integer, nullable=False),
    Column("updated_at", Integer, nullable=False),
)

journal = Table(
    "paper_journal",
    metadata,
    Column("trade_id", Integer, primary_key=True),
    Column("notes", Text, nullable=False, default=""),
    Column("tags", String, nullable=False, default=""),  # comma separated
    Column("rating", Integer),  # 1..5, optional
    Column("updated_at", Integer, nullable=False),
)

TS_COLUMNS = ("signal_ts", "entry_ts", "exit_ts", "created_at", "updated_at")


def epoch(ts: object) -> int:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return int(t.timestamp())


def now_epoch() -> int:
    return int(datetime.now(UTC).timestamp())


class PaperBook:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        metadata.create_all(engine)

    # ------------------------------------------------------------------ setups

    def ensure_setup(self, row: dict, start_ts: int) -> int:
        """Register a setup the first time it is seen; returns its (unchanged) start_ts."""
        with self.engine.begin() as conn:
            conn.execute(
                sqlite_insert(setups)
                .values(**row, start_ts=start_ts)
                .on_conflict_do_nothing(index_elements=["setup_id"])
            )
            return int(
                conn.execute(
                    select(setups.c.start_ts).where(setups.c.setup_id == row["setup_id"])
                ).scalar_one()
            )

    def touch_setup(self, setup_id: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                update(setups).where(setups.c.setup_id == setup_id).values(last_run_ts=now_epoch())
            )

    def setups(self) -> pd.DataFrame:
        with self.engine.connect() as conn:
            df = pd.DataFrame(conn.execute(select(setups)).mappings().all())
        for c in ("start_ts", "last_run_ts"):
            if c in df:
                df[c] = pd.to_datetime(df[c], unit="s", utc=True)
        return df

    # ------------------------------------------------------------------ trades

    def open_trades(self, setup_id: str) -> dict[int, dict]:
        """{entry_ts epoch: row} of a setup's open trades."""
        q = select(trades).where(trades.c.setup_id == setup_id, trades.c.status == "open")
        with self.engine.connect() as conn:
            return {r["entry_ts"]: dict(r) for r in conn.execute(q).mappings()}

    def known_entries(self, setup_id: str) -> set[int]:
        q = select(trades.c.entry_ts).where(trades.c.setup_id == setup_id)
        with self.engine.connect() as conn:
            return {r for (r,) in conn.execute(q)}

    def insert(self, row: dict) -> int:
        t = now_epoch()
        with self.engine.begin() as conn:
            res = conn.execute(trades.insert().values(**row, created_at=t, updated_at=t))
            return int(res.inserted_primary_key[0])

    def close(self, trade_id: int, values: dict) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                update(trades)
                .where(trades.c.id == trade_id)
                .values(**values, status="closed", updated_at=now_epoch())
            )

    def mark_notified(self, trade_id: int, which: str) -> None:
        col = {"entry": "notified_entry", "exit": "notified_exit"}[which]
        with self.engine.begin() as conn:
            conn.execute(update(trades).where(trades.c.id == trade_id).values(**{col: True}))

    def trades(self, status: str | None = None) -> pd.DataFrame:
        """All paper trades joined with their journal entry, newest first."""
        q = (
            select(trades, journal.c.notes, journal.c.tags, journal.c.rating)
            .select_from(trades.outerjoin(journal, journal.c.trade_id == trades.c.id))
            .order_by(trades.c.entry_ts.desc())
        )
        if status:
            q = q.where(trades.c.status == status)
        with self.engine.connect() as conn:
            df = pd.DataFrame(conn.execute(q).mappings().all())
        if df.empty:
            return pd.DataFrame(
                columns=[c.name for c in trades.columns] + ["notes", "tags", "rating"]
            )
        for c in TS_COLUMNS:
            df[c] = pd.to_datetime(df[c], unit="s", utc=True)
        df["notes"] = df["notes"].fillna("")
        df["tags"] = df["tags"].fillna("")
        return df

    # ------------------------------------------------------------------ journal

    def save_journal(
        self, trade_id: int, notes: str, tags: list[str] | str, rating: int | None
    ) -> None:
        if isinstance(tags, list):
            tags = ",".join(dict.fromkeys(t.strip().lower() for t in tags if t.strip()))
        row = {
            "trade_id": int(trade_id),
            "notes": notes or "",
            "tags": tags or "",
            "rating": rating,
            "updated_at": now_epoch(),
        }
        stmt = sqlite_insert(journal).values(**row)
        stmt = stmt.on_conflict_do_update(
            index_elements=["trade_id"], set_={k: stmt.excluded[k] for k in row if k != "trade_id"}
        )
        with self.engine.begin() as conn:
            conn.execute(stmt)

    def all_tags(self) -> list[str]:
        with self.engine.connect() as conn:
            rows = conn.execute(select(journal.c.tags)).all()
        tags = {t.strip() for (s,) in rows for t in (s or "").split(",") if t.strip()}
        return sorted(tags)
