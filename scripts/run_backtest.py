"""Backtest strategies on stored candles (with real costs) and build a leaderboard.

Examples:
    python scripts/run_backtest.py --list
    python scripts/run_backtest.py sr_breakout RELIANCE.NS 1d
    python scripts/run_backtest.py ema_crossover TCS 1h --param fast=10 --param slow=50
    python scripts/run_backtest.py sr_breakout RELIANCE.NS 1d --walkforward --plot
    python scripts/run_backtest.py --leaderboard
    python scripts/run_backtest.py --leaderboard --strategies ema_crossover --timeframes 1d

Data comes from the local DB: run scripts/backfill.py first.
"""

from __future__ import annotations

import argparse
import re
import webbrowser
from pathlib import Path

import pandas as pd

from sanjotracker.backtest.costs import cost_model_for
from sanjotracker.backtest.metrics import compute_metrics, regime_breakdown
from sanjotracker.backtest.montecarlo import monte_carlo
from sanjotracker.backtest.runner import ResultStore, make_context, run_leaderboard, run_one
from sanjotracker.backtest.walkforward import walk_forward
from sanjotracker.charts import equity_chart
from sanjotracker.config import PROJECT_ROOT, get_config
from sanjotracker.data.store import CandleStore
from sanjotracker.data.updater import normalize_symbol
from sanjotracker.log import setup_logging
from sanjotracker.strategies import STRATEGIES, get_strategy

PCT = {
    "total_return",
    "cagr",
    "max_drawdown",
    "win_rate",
    "buy_hold_return",
    "excess_vs_buy_hold",
    "fees_pct_of_gross",
    "mc_dd_p5",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("strategy", nargs="?", choices=sorted(STRATEGIES))
    p.add_argument("symbol", nargs="?")
    p.add_argument("timeframe", nargs="?", choices=["5m", "15m", "1h", "1d", "1w"])
    p.add_argument("--param", action="append", default=[], help="override a parameter: name=value")
    p.add_argument("--walkforward", action="store_true", help="also run walk-forward optimisation")
    p.add_argument("--plot", action="store_true", help="save + open an equity/drawdown chart")
    p.add_argument("--no-save", action="store_true", help="don't store results in the DB/CSV")
    p.add_argument("--trades", type=int, default=10, help="how many recent trades to print")
    p.add_argument("--list", action="store_true", help="list strategies and their parameters")
    p.add_argument("--leaderboard", action="store_true")
    p.add_argument("--strategies", nargs="*")
    p.add_argument("--symbols", nargs="*")
    p.add_argument("--timeframes", nargs="*")
    return p.parse_args()


def _parse_params(pairs: list[str]) -> dict:
    out = {}
    for pair in pairs:
        k, _, v = pair.partition("=")
        if not _:
            raise SystemExit(f"--param needs name=value, got {pair!r}")
        try:
            out[k] = int(v)
        except ValueError:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = None if v.lower() == "none" else v
    return out


def _fmt(metrics: dict) -> pd.DataFrame:
    rows = []
    for k, v in metrics.items():
        if v is None or (isinstance(v, float) and pd.isna(v)):
            s = "-"
        elif (
            k in PCT or k.startswith(("dd_", "return_")) or k in ("actual_return", "actual_max_dd")
        ):
            s = f"{v:+.2%}"
        else:
            s = f"{v:,.2f}"
        rows.append((k, s))
    return pd.DataFrame(rows, columns=["metric", "value"])


def _onoff(flag: bool) -> str:
    return "on" if flag else "off"


def _list() -> None:
    for name, cls in STRATEGIES.items():
        print(f"\n{name}: {cls.description}")
        print(f"  timeframes {sorted(cls.timeframes)}")
        for k, f in cls.Params.model_fields.items():
            grid = cls.param_grid.get(k)
            print(f"  {k} = {f.default!r}" + (f"   (walk-forward tries {grid})" if grid else ""))


def _single(args: argparse.Namespace) -> int:
    cfg = get_config()
    store = CandleStore(cfg.db_file)
    strat = get_strategy(args.strategy, **_parse_params(args.param))
    args.symbol = normalize_symbol(args.symbol)
    if not strat.supports(args.timeframe):
        print(f"{strat.name} supports {sorted(strat.timeframes)} only.")
        return 1
    res = run_one(cfg, store, strat, args.symbol, args.timeframe)
    m = compute_metrics(res, cfg.backtest.risk_free_pct / 100)
    mc = monte_carlo(res.trades, cfg.backtest.monte_carlo_runs)
    cost = cost_model_for(args.symbol, args.timeframe, cfg.costs)

    print(f"\n{res.strategy} · {args.symbol} {args.timeframe} · {len(res.equity)} bars")
    print(
        f"costs: {cost.name} (~{cost.round_trip_pct():.2%} per round trip incl. slippage) · "
        f"risk {cfg.backtest.risk_per_trade_pct}%/trade · "
        f"shorts {_onoff(res.meta['allow_short'])} · EOD exit {_onoff(res.meta['eod_exit'])}\n"
    )
    print(_fmt(m).to_string(index=False))
    rb = regime_breakdown(res.trades)
    if not rb.empty:
        print("\nBy regime (at the signal bar):")
        print(rb.to_string(index=False))
    if len(mc) > 1:
        print(f"\nMonte Carlo ({int(mc['runs'])} runs on {int(mc['trades'])} trades):")
        print(
            f"  max drawdown  bad 5%: {mc['dd_p5']:+.1%}  median: {mc['dd_p50']:+.1%}  "
            f"(actual {mc['actual_max_dd']:+.1%})"
        )
        print(
            f"  final return  5%: {mc['return_p5']:+.1%}  median: {mc['return_p50']:+.1%}  "
            f"95%: {mc['return_p95']:+.1%}  · chance of a loss {mc['prob_loss']:.0%}"
        )
    if not res.trades.empty and args.trades:
        cols = ["direction", "entry_ts", "entry_price", "exit_ts", "exit_price", "exit_reason"]
        t = res.trades.tail(args.trades)[[*cols, "net_pnl", "r_multiple"]].copy()
        for c in ("entry_ts", "exit_ts"):
            t[c] = t[c].dt.tz_convert(cfg.app.timezone_display).dt.strftime("%Y-%m-%d %H:%M")
        print(f"\nLast {len(t)} trades:")
        print(t.round(2).to_string(index=False))

    if not args.no_save:
        run_id = ResultStore(store.engine).save(res, m, "single", mc=mc)
        print(f"\nSaved as run {run_id} (tables backtest_runs / backtest_trades).")

    if args.walkforward:
        bt = cfg.backtest
        print(
            f"\nWalk-forward: {bt.walkforward_windows} windows, "
            f"objective {bt.walkforward_objective}…"
        )
        bench = store.load(cfg.benchmarks.nse, "1d")
        wf = walk_forward(
            store.load(args.symbol, args.timeframe),
            type(strat),
            make_context(cfg, args.symbol, args.timeframe),
            cost,
            n_windows=bt.walkforward_windows,
            train_ratio=bt.walkforward_train_ratio,
            objective=bt.walkforward_objective,
            initial_capital=bt.initial_capital,
            risk_pct=bt.risk_per_trade_pct,
            benchmark_df=None if bench.empty else bench,
        )
        w = wf.windows.copy()
        w["test_from"] = w["test_from"].dt.strftime("%Y-%m-%d")
        w["test_to"] = w["test_to"].dt.strftime("%Y-%m-%d")
        print(w.drop(columns=["train_from"]).round(3).to_string(index=False))
        o = wf.oos_metrics
        print(
            f"\nOut-of-sample total: {o.get('trades', 0):.0f} trades, return "
            f"{o.get('total_return', 0):+.2%}, Sharpe {o.get('sharpe', float('nan')):.2f}, "
            f"expectancy {o.get('expectancy_r', float('nan')):+.2f}R "
            f"(full-history default params: {m.get('expectancy_r', float('nan')):+.2f}R)"
        )
        if not args.no_save:
            ResultStore(store.engine).save(wf.oos, wf.oos_metrics, "walkforward")

    if args.plot:
        safe = re.sub(r"[^A-Za-z0-9]+", "-", f"{args.strategy}_{args.symbol}_{args.timeframe}")
        out = PROJECT_ROOT / "reports" / f"backtest_{safe.strip('-')}.html"
        out.parent.mkdir(parents=True, exist_ok=True)
        equity_chart(
            res.equity,
            res.close,
            title=f"{res.strategy} · {args.symbol} {args.timeframe}",
            display_tz=cfg.app.timezone_display,
        ).write_html(out, include_plotlyjs="cdn")
        print(f"\nChart: {out}")
        webbrowser.open(Path(out).resolve().as_uri())
    return 0


def _leaderboard(args: argparse.Namespace) -> int:
    cfg = get_config()
    store = CandleStore(cfg.db_file)
    lb = run_leaderboard(
        cfg,
        store,
        strategies=args.strategies,
        symbols=args.symbols,
        timeframes=args.timeframes,
        save=not args.no_save,
        progress=lambda s: print(f"  {s}", end="\r", flush=True),
    )
    t = lb.table.copy()
    if t.empty:
        print("\nNothing to rank. Run scripts/backfill.py first.")
        return 1
    for c in PCT & set(t.columns):
        t[c] = t[c].map(lambda v: "-" if pd.isna(v) else f"{v:+.1%}")
    for c in ("sharpe", "sortino", "calmar", "profit_factor", "expectancy_r", "years"):
        t[c] = t[c].map(lambda v: "-" if pd.isna(v) else f"{v:.2f}")
    t["rank"] = t["rank"].map(lambda v: "" if pd.isna(v) else int(v))
    t["trades"] = t["trades"].astype(int)
    cols = [
        "rank",
        "strategy",
        "symbol",
        "timeframe",
        "trades",
        "years",
        "total_return",
        "sharpe",
        "max_drawdown",
        "expectancy_r",
        "buy_hold_return",
        "note",
    ]
    print(f"\nLeaderboard (default params, costs included) · batch {lb.batch_id}\n")
    print(t[cols].head(40).to_string(index=False))
    if len(t) > 40:
        print(f"... {len(t) - 40} more rows in the CSV")
    if lb.csv_path:
        print(f"\nCSV: {lb.csv_path}")
    if lb.skipped:
        print(f"\nSkipped {len(lb.skipped)}: " + "; ".join(lb.skipped[:8]))
    return 0


def main() -> int:
    args = parse_args()
    setup_logging()
    if args.list:
        _list()
        return 0
    if args.leaderboard:
        return _leaderboard(args)
    if not (args.strategy and args.symbol and args.timeframe):
        print("Give STRATEGY SYMBOL TIMEFRAME, or --leaderboard, or --list. See --help.")
        return 1
    return _single(args)


if __name__ == "__main__":
    raise SystemExit(main())
