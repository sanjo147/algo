"""Walk-forward optimisation (rolling in-sample / out-of-sample).

History is cut into windows. For each window:
  train (in-sample, IS): try every parameter combo, keep the best by `objective`;
  test (out-of-sample, OOS): run ONLY that combo on the next, unseen bars.
Then the next window rolls forward by one test length. The OOS pieces glued
together are the honest estimate: those parameters were chosen without seeing
the bars they were scored on. A big gap between IS and OOS = overfitting.

Indicator warm-up: each run sees all history before its window (cut at the
window's end) but only trades inside it, so nothing after the window leaks in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sanjotracker.backtest.costs import CostModel
from sanjotracker.backtest.engine import TRADE_COLUMNS, BacktestResult, run_backtest
from sanjotracker.backtest.metrics import compute_metrics
from sanjotracker.strategies.base import Strategy, StrategyContext

OBJECTIVES = ("sharpe", "total_return", "expectancy_r", "profit_factor")


@dataclass
class WalkForwardResult:
    windows: pd.DataFrame  # one row per window: dates, best params, IS score, OOS score
    oos: BacktestResult  # all OOS pieces stitched together
    oos_metrics: dict[str, float]


def _score(m: dict[str, float], objective: str, min_trades: int) -> float:
    if m.get("trades", 0) < min_trades:
        return -np.inf  # too few trades to judge: never "best"
    v = m.get(objective, np.nan)
    return float(v) if np.isfinite(v) else -np.inf


def split_windows(
    n: int, warmup: int, n_windows: int, train_ratio: float
) -> list[tuple[int, int, int]]:
    """(train_start, test_start, test_end) index triples for rolling windows."""
    usable = n - warmup
    test_len = int(usable // (n_windows + train_ratio))
    train_len = int(test_len * train_ratio)
    if test_len < 2:
        raise ValueError("not enough bars for walk-forward; use fewer windows or more data")
    out = []
    for k in range(n_windows):
        tr = warmup + k * test_len
        te = tr + train_len
        out.append((tr, te, min(te + test_len, n)))
    return out


def walk_forward(
    df: pd.DataFrame,
    strategy_cls: type[Strategy],
    ctx: StrategyContext,
    cost: CostModel,
    *,
    n_windows: int = 5,
    train_ratio: float = 2.0,
    objective: str = "sharpe",
    min_trades: int = 5,
    warmup: int = 200,
    initial_capital: float = 100_000,
    risk_pct: float = 1.0,
    benchmark_df: pd.DataFrame | None = None,
) -> WalkForwardResult:
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}")
    df = df.reset_index(drop=True)
    kw = {"initial_capital": initial_capital, "risk_pct": risk_pct, "benchmark_df": benchmark_df}
    rows, pieces = [], []
    equity_parts: list[pd.Series] = []
    held_parts: list[pd.Series] = []
    close_parts: list[pd.Series] = []
    capital = initial_capital

    for k, (tr, te, end) in enumerate(split_windows(len(df), warmup, n_windows, train_ratio)):
        best, best_score, best_is = {}, -np.inf, {}
        for params in strategy_cls.grid():
            r = run_backtest(df, strategy_cls(**params), ctx, cost, start=tr, end=te, **kw)
            m = compute_metrics(r)
            s = _score(m, objective, min_trades)
            if s > best_score or not best:
                best, best_score, best_is = params, s, m
        oos = run_backtest(
            df,
            strategy_cls(**best),
            ctx,
            cost,
            start=te,
            end=end,
            **{**kw, "initial_capital": capital},
        )
        om = compute_metrics(oos)
        rows.append(
            {
                "window": k + 1,
                "train_from": df["timestamp"].iloc[tr],
                "test_from": df["timestamp"].iloc[te],
                "test_to": df["timestamp"].iloc[end - 1],
                "params": best,
                f"is_{objective}": best_is.get(objective, np.nan),
                f"oos_{objective}": om.get(objective, np.nan),
                "is_trades": best_is.get("trades", 0),
                "oos_trades": om.get("trades", 0),
                "oos_return": om.get("total_return", np.nan),
            }
        )
        pieces.append(oos.trades)
        equity_parts.append(oos.equity)
        held_parts.append(oos.in_market)
        close_parts.append(oos.close)
        capital = float(oos.equity.iloc[-1])  # compound across windows

    trades = (
        pd.concat([p for p in pieces if not p.empty], ignore_index=True)
        if any(not p.empty for p in pieces)
        else pd.DataFrame(columns=TRADE_COLUMNS)
    )
    stitched = BacktestResult(
        symbol=ctx.symbol,
        timeframe=ctx.timeframe,
        strategy=f"{strategy_cls.name} (walk-forward)",
        params={"objective": objective, "n_windows": n_windows, "train_ratio": train_ratio},
        cost_model=cost.name,
        initial_capital=initial_capital,
        trades=trades,
        equity=pd.concat(equity_parts),
        in_market=pd.concat(held_parts),
        close=pd.concat(close_parts),
    )
    return WalkForwardResult(pd.DataFrame(rows), stitched, compute_metrics(stitched))
