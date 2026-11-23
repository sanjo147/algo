"""Advance paper trading now and print the paper account.

The scheduler already does this after each scan; use this to check by hand.

Examples:
    python scripts/run_paper.py                 # fetch new candles, step, print
    python scripts/run_paper.py --no-update     # use stored candles only
    python scripts/run_paper.py --show          # just print, don't step
    python scripts/run_paper.py --no-telegram   # step without notifying
"""

from __future__ import annotations

import argparse

import pandas as pd

from sanjotracker.alerts.telegram_bot import NullNotifier, build_notifier
from sanjotracker.config import get_config
from sanjotracker.data.store import CandleStore
from sanjotracker.data.updater import build_updater
from sanjotracker.log import setup_logging
from sanjotracker.paper.book import PaperBook
from sanjotracker.paper.trader import PaperTrader, paper_summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--no-update", action="store_true", help="don't fetch new candles first")
    p.add_argument("--no-telegram", action="store_true")
    p.add_argument("--show", action="store_true", help="only print the account")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging()
    cfg = get_config()
    store = CandleStore(cfg.db_file)
    if not args.show:
        notifier = NullNotifier() if args.no_telegram else build_notifier(cfg)
        updater = (
            None
            if args.no_update
            else build_updater(store, {"nse": cfg.providers.nse, "crypto": cfg.providers.crypto})
        )
        res = PaperTrader(cfg, store, notifier, updater).run()
        print(f"\nStepped {res.setups} setups: {len(res.opened)} opened, {len(res.closed)} closed")
        for k, v in res.errors.items():
            print(f"  problem: {k}: {v}")

    s = paper_summary(PaperBook(store.engine), store, cfg.paper.capital)
    print(
        f"\nPaper account · capital {s['capital']:,.0f} · equity {s['equity']:,.0f} "
        f"(realized {s['realized']:+,.0f}, open {s['unrealized']:+,.0f})"
    )
    tz = cfg.app.timezone_display
    if len(s["open"]):
        o = s["open"].copy()
        o["entry"] = o["entry_ts"].dt.tz_convert(tz).dt.strftime("%d %b %H:%M")
        print("\nOpen:")
        print(
            o[
                [
                    "symbol",
                    "timeframe",
                    "label",
                    "direction",
                    "entry",
                    "entry_price",
                    "stop",
                    "last",
                    "unrealized_pnl",
                    "unrealized_r",
                ]
            ]
            .round(2)
            .to_string(index=False)
        )
    if len(s["closed"]):
        cl = s["closed"].head(15).copy()
        cl["exit"] = cl["exit_ts"].dt.tz_convert(tz).dt.strftime("%d %b %H:%M")
        print(f"\nClosed (latest {len(cl)}; win rate {s['win_rate']:.0%}, avg {s['avg_r']:+.2f}R):")
        print(
            cl[
                [
                    "symbol",
                    "timeframe",
                    "label",
                    "direction",
                    "exit",
                    "exit_reason",
                    "net_pnl",
                    "r_multiple",
                ]
            ]
            .round(2)
            .to_string(index=False)
        )
    if not len(s["open"]) and not len(s["closed"]):
        print("No paper trades yet. They start from the first bar that closes after setup.")
    pd.set_option("display.width", 200)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
