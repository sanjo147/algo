"""Save an HTML candlestick chart with S/R zones so you can eyeball the levels.

Examples:
    python scripts/plot_levels.py RELIANCE.NS 1d
    python scripts/plot_levels.py TCS 15m --bars 400 --open
    python scripts/plot_levels.py ^NSEI 1h --no-update
"""

from __future__ import annotations

import argparse
import re
import webbrowser
from pathlib import Path

from sanjotracker.analysis.levels import compute_levels, find_pivots, nearest_levels
from sanjotracker.charts import levels_chart
from sanjotracker.config import PROJECT_ROOT, get_config
from sanjotracker.data.store import CandleStore
from sanjotracker.data.updater import build_updater, normalize_symbol, session_tz_for
from sanjotracker.log import setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("symbol")
    p.add_argument("timeframe", choices=["5m", "15m", "1h", "1d", "1w"])
    p.add_argument("--bars", type=int, default=300, help="bars to plot (levels use all history)")
    p.add_argument("--no-update", action="store_true", help="don't fetch new candles first")
    p.add_argument("--poc", action="store_true", help="add volume-profile POC level")
    p.add_argument("--open", action="store_true", help="open the chart in your browser")
    p.add_argument("--out", help="output HTML path (default reports/levels_<sym>_<tf>.html)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging()
    cfg = get_config()
    store = CandleStore(cfg.db_file)
    args.symbol = normalize_symbol(args.symbol)

    if not args.no_update:
        build_updater(store, cfg.providers).update(args.symbol, args.timeframe)

    df = store.load(args.symbol, args.timeframe)
    if df.empty:
        print(f"No {args.timeframe} data for {args.symbol}. Run scripts/backfill.py first.")
        return 1

    tz = session_tz_for(args.symbol)
    levels = compute_levels(
        df, args.symbol, args.timeframe, cfg.levels, session_tz=tz, include_poc=args.poc
    )
    pivots = find_pivots(df, cfg.levels.pivot_left, cfg.levels.pivot_right)
    shown = df.tail(args.bars).reset_index(drop=True)

    fig = levels_chart(
        shown,
        levels,
        pivots=pivots,
        title=f"{args.symbol} · {args.timeframe} · S/R zones (as of last closed bar)",
        display_tz=cfg.app.timezone_display,
        is_nse=True,
        intraday=args.timeframe in ("5m", "15m", "1h"),
    )
    safe = re.sub(r"[^A-Za-z0-9]+", "-", args.symbol).strip("-")
    out = Path(args.out) if args.out else Path("reports") / f"levels_{safe}_{args.timeframe}.html"
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out, include_plotlyjs="cdn")

    price = float(df["close"].iloc[-1])
    near = nearest_levels(levels, price, n=3)
    cols = ["type", "zone_low", "zone_high", "touches", "strength", "source"]
    print(
        f"\n{args.symbol} {args.timeframe}  last close {price:,.2f}  "
        f"({len(levels)} levels, {len(pivots)} pivots)\n"
    )
    print(near[cols].round(2).to_string(index=False))
    print(f"\nChart: {out}")
    if args.open:
        webbrowser.open(out.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
