"""Download history for the watchlist into SQLite (safe to re-run: only new candles).

Examples:
    python scripts/backfill.py                          # everything in config
    python scripts/backfill.py --timeframes 1d 1w
    python scripts/backfill.py --symbols RELIANCE TCS --timeframes 15m
    python scripts/backfill.py --full                   # wipe and re-download everything

Daily/weekly come from NSE's official MCP (falling back to Yahoo), intraday from Yahoo.
"""

from __future__ import annotations

import argparse

import pandas as pd

from sanjotracker.config import get_config
from sanjotracker.data.store import CandleStore
from sanjotracker.data.updater import build_updater, normalize_symbol
from sanjotracker.log import setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--symbols", nargs="*", help="override the watchlist")
    p.add_argument("--timeframes", nargs="*", help="default: all timeframes in config")
    p.add_argument("--no-benchmarks", action="store_true", help="skip the ^NSEI benchmark")
    p.add_argument(
        "--full", action="store_true", help="delete stored candles first and re-download"
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging()
    cfg = get_config()

    if args.symbols:
        symbols = [normalize_symbol(s) for s in args.symbols]
    else:
        symbols = list(cfg.watchlist.nse)
        if not args.no_benchmarks:
            symbols.append(cfg.benchmarks.nse)
    symbols = list(dict.fromkeys(symbols))
    timeframes = args.timeframes or cfg.timeframes

    store = CandleStore(cfg.db_file)
    if args.full:
        for s in symbols:
            for tf in timeframes:
                store.delete(s, tf)
    updater = build_updater(store, cfg.providers)

    results = updater.update_many(symbols, timeframes)

    table = pd.DataFrame([r.model_dump() for r in results])
    table["last_candle"] = (
        pd.to_datetime(table["last_candle"], utc=True)
        .dt.tz_convert(cfg.app.timezone_display)
        .dt.strftime("%Y-%m-%d %H:%M")
    )
    cols = ["symbol", "timeframe", "new_rows", "total_rows", "last_candle", "error"]
    print(table[cols].fillna("").to_string(index=False))
    failed = int(table["error"].fillna("").ne("").sum())
    print(f"\nDB: {cfg.db_file}   ok: {len(table) - failed}   failed: {failed}")
    return 1 if failed == len(table) else 0


if __name__ == "__main__":
    raise SystemExit(main())
