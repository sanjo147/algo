"""One-off cleanup: delete crypto rows (symbols like BTC/USDT) left in the database
from before SanjoTracker became NSE-only.

Dry run by default (only counts). Add --yes to actually delete. A copy of the
database is saved next to it first (sanjotracker.db.before-crypto-cleanup).

    python scripts/remove_crypto_data.py          # show what would be deleted
    python scripts/remove_crypto_data.py --yes    # back up, then delete
"""

from __future__ import annotations

import argparse
import shutil

from sqlalchemy import inspect, text

from sanjotracker.config import get_config
from sanjotracker.data.store import CandleStore
from sanjotracker.log import setup_logging

TABLES = {  # table -> column holding the symbol
    "candles": "symbol",
    "alerts": "symbol",
    "paper_trades": "symbol",
    "paper_setups": "symbol",
    "backtest_runs": "symbol",
    "backtest_trades": "symbol",
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--yes", action="store_true", help="really delete (after a backup)")
    args = p.parse_args()
    setup_logging()
    cfg = get_config()
    engine = CandleStore(cfg.db_file).engine
    present = set(inspect(engine).get_table_names())
    counts = {}
    with engine.connect() as conn:
        for table, col in TABLES.items():
            if table in present:
                n = conn.execute(text(f"SELECT COUNT(*) FROM {table} WHERE {col} LIKE '%/%'"))
                counts[table] = int(n.scalar() or 0)
    total = sum(counts.values())
    for table, n in counts.items():
        print(f"  {table:<16} {n:>8} crypto rows")
    if not total:
        print("Nothing to delete.")
        return 0
    if not args.yes:
        print(f"\nDry run: {total} rows would be deleted. Re-run with --yes to delete them.")
        return 0

    backup = cfg.db_file.with_name(cfg.db_file.name + ".before-crypto-cleanup")
    shutil.copyfile(cfg.db_file, backup)
    print(f"\nBackup saved: {backup}")
    with engine.begin() as conn:
        # journal entries of crypto paper trades go too
        if {"paper_journal", "paper_trades"} <= present:
            conn.execute(
                text(
                    "DELETE FROM paper_journal WHERE trade_id IN "
                    "(SELECT id FROM paper_trades WHERE symbol LIKE '%/%')"
                )
            )
        for table, col in TABLES.items():
            if table in present:
                conn.execute(text(f"DELETE FROM {table} WHERE {col} LIKE '%/%'"))
    with engine.connect() as conn:
        conn.execute(text("VACUUM"))
    print(f"Deleted {total} crypto rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
