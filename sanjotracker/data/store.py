"""SQLite candle store.

One row per (symbol, timeframe, ts) — that triple is the primary key, so writing
the same candle twice updates it instead of duplicating it (upsert).
`ts` is the bar open time as integer epoch seconds, UTC.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from loguru import logger
from sqlalchemy import (
    Column,
    Engine,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    delete,
    event,
    func,
    select,
)
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from sanjotracker.data.providers.base import OHLCV_COLUMNS, empty_ohlcv

metadata = MetaData()

candles = Table(
    "candles",
    metadata,
    Column("symbol", String, primary_key=True),
    Column("timeframe", String, primary_key=True),
    Column("ts", Integer, primary_key=True),
    Column("open", Float, nullable=False),
    Column("high", Float, nullable=False),
    Column("low", Float, nullable=False),
    Column("close", Float, nullable=False),
    Column("volume", Float, nullable=False),
    Column("provider", String, nullable=False),
    Column("fetched_at", Integer, nullable=False),
)

_CHUNK = 500


def make_engine(db_path: str | Path) -> Engine:
    """SQLite engine with WAL mode so the dashboard can read while the scheduler writes."""
    if str(db_path) != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}")

    @event.listens_for(engine, "connect")
    def _pragmas(dbapi_conn, _record) -> None:  # noqa: ANN001
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.close()

    return engine


def _to_epoch(ts: datetime | pd.Timestamp) -> int:
    """Epoch seconds; naive datetimes are treated as UTC."""
    t = pd.Timestamp(ts)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return int(t.timestamp())


class CandleStore:
    """Read/write OHLCV candles in SQLite."""

    def __init__(self, db_path: str | Path | None = None, engine: Engine | None = None) -> None:
        if engine is None:
            if db_path is None:
                raise ValueError("pass db_path or engine")
            engine = make_engine(db_path)
        self.engine = engine
        metadata.create_all(self.engine)

    def upsert(self, symbol: str, timeframe: str, df: pd.DataFrame, provider: str) -> int:
        """Insert or update candles. Returns the number of rows written."""
        if df.empty:
            return 0
        now = int(datetime.now(UTC).timestamp())
        records = [
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "ts": _to_epoch(r.timestamp),
                "open": float(r.open),
                "high": float(r.high),
                "low": float(r.low),
                "close": float(r.close),
                "volume": float(r.volume),
                "provider": provider,
                "fetched_at": now,
            }
            for r in df[OHLCV_COLUMNS].itertuples(index=False)
        ]
        with self.engine.begin() as conn:
            for i in range(0, len(records), _CHUNK):
                stmt = sqlite_insert(candles).values(records[i : i + _CHUNK])
                stmt = stmt.on_conflict_do_update(
                    index_elements=["symbol", "timeframe", "ts"],
                    set_={
                        c: stmt.excluded[c]
                        for c in (
                            "open",
                            "high",
                            "low",
                            "close",
                            "volume",
                            "provider",
                            "fetched_at",
                        )
                    },
                )
                conn.execute(stmt)
        logger.debug("upserted {} {} {} rows", len(records), symbol, timeframe)
        return len(records)

    def load(
        self,
        symbol: str,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Candles in [start, end), oldest first. `limit` keeps only the most recent N."""
        q = select(
            candles.c.ts,
            candles.c.open,
            candles.c.high,
            candles.c.low,
            candles.c.close,
            candles.c.volume,
        ).where(candles.c.symbol == symbol, candles.c.timeframe == timeframe)
        if start is not None:
            q = q.where(candles.c.ts >= _to_epoch(start))
        if end is not None:
            q = q.where(candles.c.ts < _to_epoch(end))
        q = q.order_by(candles.c.ts.desc() if limit else candles.c.ts)
        if limit:
            q = q.limit(limit)
        with self.engine.connect() as conn:
            rows = conn.execute(q).all()
        if not rows:
            return empty_ohlcv()
        df = pd.DataFrame(rows, columns=["ts", *OHLCV_COLUMNS[1:]])
        df.insert(0, "timestamp", pd.to_datetime(df.pop("ts"), unit="s", utc=True))
        return df.sort_values("timestamp").reset_index(drop=True)

    def last_timestamp(self, symbol: str, timeframe: str) -> pd.Timestamp | None:
        """Open time of the newest stored candle, or None if there is none."""
        q = select(func.max(candles.c.ts)).where(
            candles.c.symbol == symbol, candles.c.timeframe == timeframe
        )
        with self.engine.connect() as conn:
            v = conn.execute(q).scalar()
        return None if v is None else pd.Timestamp(v, unit="s", tz="UTC")

    def count(self, symbol: str, timeframe: str) -> int:
        q = (
            select(func.count())
            .select_from(candles)
            .where(candles.c.symbol == symbol, candles.c.timeframe == timeframe)
        )
        with self.engine.connect() as conn:
            return int(conn.execute(q).scalar() or 0)

    def delete(self, symbol: str, timeframe: str) -> int:
        """Remove all candles for a symbol+timeframe (used before a full re-download)."""
        with self.engine.begin() as conn:
            res = conn.execute(
                delete(candles).where(candles.c.symbol == symbol, candles.c.timeframe == timeframe)
            )
        return int(res.rowcount or 0)

    def inventory(self) -> pd.DataFrame:
        """Summary of what is stored: symbol, timeframe, rows, first/last candle."""
        q = select(
            candles.c.symbol,
            candles.c.timeframe,
            func.count().label("rows"),
            func.min(candles.c.ts).label("first"),
            func.max(candles.c.ts).label("last"),
        ).group_by(candles.c.symbol, candles.c.timeframe)
        with self.engine.connect() as conn:
            df = pd.DataFrame(
                conn.execute(q).all(), columns=["symbol", "timeframe", "rows", "first", "last"]
            )
        for c in ("first", "last"):
            df[c] = pd.to_datetime(df[c], unit="s", utc=True)
        return df
