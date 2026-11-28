"""Scan the watchlist now and print breakout / retest / approaching events.

Examples:
    python scripts/scan_now.py swing                     # 1d bars with the 1w trend
    python scripts/scan_now.py intraday                  # 15m bars with the 1h trend
    python scripts/scan_now.py swing --lookback 5        # events from the last 5 bars
    python scripts/scan_now.py swing --all --no-update   # include counter-trend breakouts
"""

from __future__ import annotations

import argparse

import pandas as pd

from sanjotracker.analysis.breakout import events_frame, fmt_price
from sanjotracker.config import get_config
from sanjotracker.log import setup_logging
from sanjotracker.scanner import Scanner

EMOJI = {
    "breakout": "🚀",
    "breakdown": "🔻",
    "retest": "🔁",
    "failed_breakout": "⚠️",
    "failed_breakdown": "⚠️",
    "approaching": "👀",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("style", choices=["swing", "intraday"])
    p.add_argument("--symbols", nargs="*", help="override the watchlist")
    p.add_argument("--lookback", type=int, default=1, help="report events from the last N bars")
    p.add_argument("--min-confidence", type=int, default=0)
    p.add_argument("--all", action="store_true", help="keep breakouts against the HTF trend")
    p.add_argument("--no-update", action="store_true", help="don't fetch new candles first")
    p.add_argument("--reasons", action="store_true", help="print every reason under the table")
    return p.parse_args()


def _regime_line(name: str, r) -> str:  # noqa: ANN001
    return (
        f"  {name:<10} market {r.market:<7} trend {r.trend:<10} "
        f"(ADX {r.adx if r.adx is not None else '-'})  volatility {r.volatility}"
    )


def main() -> int:
    args = parse_args()
    setup_logging()
    cfg = get_config()
    scanner = Scanner(cfg)
    events = scanner.scan(
        args.style,
        symbols=args.symbols,
        update=not args.no_update,
        lookback_bars=args.lookback,
        include_counter_trend=args.all,
        min_confidence=args.min_confidence,
    )
    st = getattr(cfg.styles, args.style)
    print(f"\n{args.style.upper()} scan · {st.primary} bars · {st.trend} trend filter\n")
    print("Benchmarks (daily):")
    for name, r in scanner.benchmark_regimes().items():
        print(_regime_line(name, r))

    if not events:
        print(f"\nNo events on the last {args.lookback} closed bar(s).")
    else:
        t = events_frame(events)
        fmt = "%d-%b" if st.primary in ("1d", "1w") else "%d-%b %H:%M"
        t["when"] = t["bar_ts"].dt.tz_convert(cfg.app.timezone_display).dt.strftime(fmt)
        t["event"] = t["event"].map(lambda e: f"{EMOJI.get(e, '')} {e}")
        t["zone"] = [e.zone_text for e in events]
        t["vol"] = t["volume_ratio"].map(lambda x: "-" if pd.isna(x) else f"{x:.1f}x")
        t["htf"] = t["trend_htf"]
        t["price"] = [fmt_price(e.price) for e in events]
        cols = ["when", "symbol", "event", "price", "zone", "vol", "htf", "confidence"]
        print(f"\n{len(events)} event(s):\n")
        print(t[cols].to_string(index=False))
        if args.reasons:
            print()
            for e in events:
                print(f"{e.symbol} {e.event.value} ({e.confidence}):")
                for r in e.reasons:
                    print(f"    - {r}")

    if scanner.errors:
        print("\nProblems:")
        for k, v in scanner.errors.items():
            print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
