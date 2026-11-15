"""Download history for the watchlist into SQLite (safe to re-run: only new candles).

Examples:
    python scripts/backfill.py                          # everything in config
    python scripts/backfill.py --market crypto --timeframes 1d 1w
    python scripts/backfill.py --symbols RELIANCE.NS BTC/USDT --timeframes 15m
"""

from __future__ import annotations

import argparse

import pandas as pd

from sanjotracker.config import get_config
from sanjotracker.data.store import CandleStore
from sanjotracker.data.updater import build_updater, market_of
from sanjotracker.log import setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--market", choices=["nse", "crypto", "all"], default="all")
    p.add_argument("--symbols", nargs="*", help="override the watchlist")
    p.add_argument("--timeframes", nargs="*", help="default: all timeframes in config")
    p.add_argument("--no-benchmarks", action="store_true", help="skip ^NSEI / BTC benchmark")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging()
    cfg = get_config()

    if args.symbols:
        symbols = list(args.symbols)
    else:
        symbols = []
        if args.market in ("nse", "all"):
            symbols += cfg.watchlist.nse
            if not args.no_benchmarks:
                symbols.append(cfg.benchmarks.nse)
        if args.market in ("crypto", "all"):
            symbols += cfg.watchlist.crypto
    symbols = list(dict.fromkeys(symbols))  # dedupe, keep order (BTC is also the benchmark)

    markets = {market_of(s) for s in symbols}
    names = {m: getattr(cfg.providers, m) for m in markets}
    updater = build_updater(CandleStore(cfg.db_file), names)

    results = updater.update_many(symbols, args.timeframes or cfg.timeframes)

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
