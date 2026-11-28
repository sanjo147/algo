"""Check that NSE's official MCP servers work from this network and agree with Yahoo.

    python scripts/check_nse_mcp.py               # RELIANCE
    python scripts/check_nse_mcp.py --symbol TCS

Steps: list both servers' tools -> check the ones SanjoTracker uses exist ->
live quote -> ~6 months of daily history adjusted for splits/bonuses -> compare
closes with Yahoo on the same dates. Ends with PASS or FAIL and what to do.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta

import pandas as pd

from sanjotracker.data.providers import get_provider
from sanjotracker.data.providers.mcp_client import McpClient, McpError
from sanjotracker.data.providers.nse_mcp import BHAVCOPY_URL, LIVE_URL
from sanjotracker.data.updater import normalize_symbol
from sanjotracker.log import setup_logging

NEEDED = {
    BHAVCOPY_URL: {"get_stock_history", "get_corporate_actions"},
    LIVE_URL: {"cm_get_stock_quote"},
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--symbol", default="RELIANCE")
    args = p.parse_args()
    setup_logging(level="WARNING")
    sym = normalize_symbol(args.symbol)
    ok = True

    for url, need in NEEDED.items():
        print(f"\n{url}")
        try:
            tools = {t["name"] for t in McpClient(url, retries=2).list_tools()}
        except McpError as exc:
            print(f"  FAIL  cannot connect: {exc}")
            print(
                "\nFAIL: NSE's MCP is not reachable from this network. SanjoTracker will keep "
                "using Yahoo. Try a different network (e.g. a mobile hotspot) and run this again."
            )
            return 1
        print(f"  OK    {len(tools)} tools")
        missing = need - tools
        if missing:
            ok = False
            print(f"  FAIL  missing tools SanjoTracker needs: {sorted(missing)}")

    nse = get_provider("nse_mcp")
    try:
        print(f"\nLive quote {sym}: {nse.get_latest(sym):,.2f} (1-3 min delayed)")
    except Exception as exc:
        ok = False
        print(f"  FAIL  live quote: {exc}")

    end = datetime.now(UTC)
    start = end - timedelta(days=180)
    try:
        ours = nse.get_ohlcv(sym, "1d", start, end)
    except Exception as exc:
        print(f"  FAIL  history: {exc}")
        return 1
    print(
        f"NSE daily bars: {len(ours)} ({ours['timestamp'].iloc[0].date()} .. "
        f"{ours['timestamp'].iloc[-1].date()})"
        if len(ours)
        else "NSE daily bars: 0"
    )
    yahoo = get_provider("yfinance_nse").get_ohlcv(sym, "1d", start, end)
    both = pd.merge(ours, yahoo, on="timestamp", suffixes=("_nse", "_yahoo"))
    if both.empty:
        print("  FAIL  no overlapping dates with Yahoo (date format or symbol mismatch?)")
        return 1
    diff = (both["close_nse"] / both["close_yahoo"] - 1).abs()
    print(
        f"Compared {len(both)} days with Yahoo: median close difference {diff.median():.3%}, "
        f"worst {diff.max():.3%}"
    )
    if diff.median() > 0.005:
        ok = False
        print("  FAIL  NSE and Yahoo disagree by more than 0.5% -> adjustment or dates are off.")
        print(
            both.loc[diff.nlargest(5).index, ["timestamp", "close_nse", "close_yahoo"]].to_string(
                index=False
            )
        )

    print(
        "\nPASS: NSE MCP works here; daily data will come from NSE." if ok else "\nFAIL (see above)"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
