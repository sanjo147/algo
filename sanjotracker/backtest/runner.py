"""Glue: run a strategy on stored candles, run the leaderboard, save results.

Used by scripts/run_backtest.py now and by the dashboard (Phase 6).
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import Column, Engine, Float, Integer, MetaData, String, Table, Text, select

from sanjotracker.backtest.costs import cost_model_for
from sanjotracker.backtest.engine import BacktestResult, run_backtest
from sanjotracker.backtest.metrics import compute_metrics, regime_breakdown
from sanjotracker.backtest.montecarlo import monte_carlo
from sanjotracker.config import PROJECT_ROOT, Config
from sanjotracker.data.store import CandleStore
from sanjotracker.data.updater import market_of, session_tz_for
from sanjotracker.strategies import STRATEGIES, Strategy, StrategyContext

LEADERBOARD_COLUMNS = [
    "rank",
    "strategy",
    "symbol",
    "timeframe",
    "trades",
    "years",
    "total_return",
    "cagr",
    "sharpe",
    "sortino",
    "calmar",
    "max_drawdown",
    "win_rate",
    "profit_factor",
    "expectancy_r",
    "exposure_pct",
    "buy_hold_return",
    "excess_vs_buy_hold",
    "mc_dd_p5",
    "fees_pct_of_gross",
    "note",
]

metadata = MetaData()
runs_table = Table(
    "backtest_runs",
    metadata,
    Column("run_id", String, primary_key=True),
    Column("batch_id", String, index=True),
    Column("kind", String, nullable=False),  # single | leaderboard | walkforward
    Column("created_at", Integer, nullable=False),
    Column("strategy", String, nullable=False),
    Column("symbol", String, nullable=False),
    Column("timeframe", String, nullable=False),
    Column("params", Text, nullable=False),
    Column("cost_model", String),
    Column("start_ts", Integer),
    Column("end_ts", Integer),
    Column("trades", Integer),
    Column("total_return", Float),
    Column("sharpe", Float),
    Column("max_drawdown", Float),
    Column("expectancy_r", Float),
    Column("metrics", Text, nullable=False),  # full metrics JSON
    Column("regimes", Text),  # regime breakdown JSON
    Column("monte_carlo", Text),
)
TRADES_TABLE = "backtest_trades"


# ------------------------------------------------------------------ single runs


def make_context(cfg: Config, symbol: str, timeframe: str) -> StrategyContext:
    return StrategyContext(
        symbol=symbol,
        timeframe=timeframe,
        session_tz=session_tz_for(symbol),
        market=market_of(symbol),
        levels=cfg.levels,
        breakout=cfg.breakout,
    )


def run_one(
    cfg: Config,
    store: CandleStore,
    strategy: Strategy,
    symbol: str,
    timeframe: str,
    df: pd.DataFrame | None = None,
) -> BacktestResult:
    """Backtest one strategy on stored candles with the configured costs."""
    df = store.load(symbol, timeframe) if df is None else df
    if df.empty:
        raise ValueError(f"no {timeframe} candles for {symbol}; run scripts/backfill.py")
    bench = store.load(getattr(cfg.benchmarks, market_of(symbol)), "1d")
    return run_backtest(
        df,
        strategy,
        make_context(cfg, symbol, timeframe),
        cost_model_for(symbol, timeframe, cfg.costs),
        initial_capital=cfg.backtest.initial_capital,
        risk_pct=cfg.backtest.risk_per_trade_pct,
        benchmark_df=None if bench.empty else bench,
    )


# ------------------------------------------------------------------ storage


def _jsonable(obj: object) -> object:
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    return obj


class ResultStore:
    """backtest_runs (one row per run) + backtest_trades (every trade, by run_id)."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        metadata.create_all(engine)

    def save(
        self,
        res: BacktestResult,
        metrics: dict,
        kind: str = "single",
        batch_id: str | None = None,
        mc: dict | None = None,
    ) -> str:
        run_id = uuid.uuid4().hex[:12]
        eq = res.equity
        row = {
            "run_id": run_id,
            "batch_id": batch_id,
            "kind": kind,
            "created_at": int(datetime.now(UTC).timestamp()),
            "strategy": res.strategy,
            "symbol": res.symbol,
            "timeframe": res.timeframe,
            "params": json.dumps(_jsonable(res.params)),
            "cost_model": res.cost_model,
            "start_ts": int(eq.index[0].timestamp()) if len(eq) else None,
            "end_ts": int(eq.index[-1].timestamp()) if len(eq) else None,
            "trades": int(metrics.get("trades", 0) or 0),
            "total_return": metrics.get("total_return"),
            "sharpe": metrics.get("sharpe"),
            "max_drawdown": metrics.get("max_drawdown"),
            "expectancy_r": metrics.get("expectancy_r"),
            "metrics": json.dumps(_jsonable(metrics)),
            "regimes": regime_breakdown(res.trades).to_json(orient="records"),
            "monte_carlo": json.dumps(_jsonable(mc)) if mc else None,
        }
        row = {
            k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in row.items()
        }
        with self.engine.begin() as conn:
            conn.execute(runs_table.insert().values(**row))
        if not res.trades.empty:
            t = res.trades.copy()
            for c in ("signal_ts", "entry_ts", "exit_ts"):
                t[c] = pd.to_datetime(t[c], utc=True).astype("int64") // 10**9
            t.insert(0, "run_id", run_id)
            t.to_sql(TRADES_TABLE, self.engine, if_exists="append", index=False)
        return run_id

    def runs(self, kind: str | None = None, batch_id: str | None = None) -> pd.DataFrame:
        q = select(runs_table).order_by(runs_table.c.created_at.desc())
        if kind:
            q = q.where(runs_table.c.kind == kind)
        if batch_id:
            q = q.where(runs_table.c.batch_id == batch_id)
        with self.engine.connect() as conn:
            return pd.DataFrame(conn.execute(q).mappings().all())

    def trades(self, run_id: str) -> pd.DataFrame:
        with self.engine.connect() as conn:
            try:
                df = pd.read_sql(
                    f"SELECT * FROM {TRADES_TABLE} WHERE run_id = ?", conn, params=(run_id,)
                )
            except Exception:  # table not created yet
                return pd.DataFrame()
        for c in ("signal_ts", "entry_ts", "exit_ts"):
            if c in df:
                df[c] = pd.to_datetime(df[c], unit="s", utc=True)
        return df


# ------------------------------------------------------------------ leaderboard


@dataclass
class LeaderboardResult:
    table: pd.DataFrame
    batch_id: str
    csv_path: Path | None
    skipped: list[str]


def run_leaderboard(
    cfg: Config,
    store: CandleStore,
    *,
    strategies: list[str] | None = None,
    symbols: list[str] | None = None,
    timeframes: list[str] | None = None,
    save: bool = True,
    csv_dir: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> LeaderboardResult:
    """Every strategy x symbol x timeframe with DEFAULT params (no tuning, so no
    in-sample optimisation bias). Ranked by Sharpe among runs with >= min_trades."""
    bt = cfg.backtest
    names = strategies or list(STRATEGIES)
    syms = symbols or [*cfg.watchlist.nse, *cfg.watchlist.crypto]
    tfs = timeframes or cfg.timeframes
    results = ResultStore(store.engine) if save else None
    batch_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    rows, skipped = [], []

    for sym in syms:
        market = market_of(sym)
        for tf in tfs:
            todo = [n for n in names if STRATEGIES[n]().supports(market, tf)]
            if not todo:
                continue
            df = store.load(sym, tf)
            if len(df) < bt.min_bars:
                skipped.append(f"{sym} {tf}: {len(df)} bars (< {bt.min_bars})")
                continue
            for name in todo:
                if progress:
                    progress(f"{name} {sym} {tf}")
                try:
                    res = run_one(cfg, store, STRATEGIES[name](), sym, tf, df)
                    m = compute_metrics(res, bt.risk_free_pct / 100)
                    mc = monte_carlo(res.trades, bt.monte_carlo_runs)
                except Exception as exc:
                    logger.error("{} {} {}: {}", name, sym, tf, exc)
                    skipped.append(f"{name} {sym} {tf}: {exc}")
                    continue
                if results is not None:
                    results.save(res, m, "leaderboard", batch_id, mc)
                notes = []
                if m.get("trades", 0) < bt.min_trades:
                    notes.append(f"<{bt.min_trades} trades")
                if m.get("years", 0) < 1:
                    notes.append("<1 year of data")
                rows.append(
                    {
                        "strategy": name,
                        "symbol": sym,
                        "timeframe": tf,
                        **{k: m.get(k, np.nan) for k in LEADERBOARD_COLUMNS[4:-2]},
                        "mc_dd_p5": mc.get("dd_p5", np.nan),
                        "note": ", ".join(notes),
                    }
                )

    table = pd.DataFrame(rows, columns=LEADERBOARD_COLUMNS[1:])
    rankable = table["trades"] >= bt.min_trades
    table["_score"] = table["sharpe"].where(rankable)
    table = table.sort_values("_score", ascending=False, na_position="last").reset_index(drop=True)
    ranked = table["_score"].notna()
    table.insert(0, "rank", ranked.cumsum().where(ranked))  # unranked rows: rank NaN
    table = table.drop(columns="_score")

    csv_path = None
    if save and not table.empty:
        out = csv_dir or PROJECT_ROOT / "reports"
        out.mkdir(parents=True, exist_ok=True)
        csv_path = out / f"leaderboard_{batch_id}.csv"
        table.to_csv(csv_path, index=False)
    return LeaderboardResult(table, batch_id, csv_path, skipped)
