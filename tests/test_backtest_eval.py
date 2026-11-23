"""Metrics, regime breakdown, Monte Carlo, walk-forward, leaderboard + result storage."""

import numpy as np
import pandas as pd
import pytest

from sanjotracker.backtest.costs import ZeroCostModel
from sanjotracker.backtest.engine import TRADE_COLUMNS, BacktestResult
from sanjotracker.backtest.metrics import _max_drawdown, compute_metrics, regime_breakdown
from sanjotracker.backtest.montecarlo import monte_carlo
from sanjotracker.backtest.runner import ResultStore, run_leaderboard, run_one
from sanjotracker.backtest.walkforward import split_windows, walk_forward
from sanjotracker.config import PROJECT_ROOT, load_config
from sanjotracker.strategies import STRATEGIES, StrategyContext, get_strategy
from tests.conftest import T0, make_candles


def _result(equity: list[float], trades: pd.DataFrame | None = None) -> BacktestResult:
    idx = pd.date_range(T0, periods=len(equity), freq="1D", tz="UTC")
    eq = pd.Series(equity, index=idx, dtype=float)
    return BacktestResult(
        "X",
        "1d",
        "s",
        {},
        "zero",
        equity[0],
        trades if trades is not None else pd.DataFrame(columns=TRADE_COLUMNS),
        eq,
        pd.Series(True, index=idx),
        eq.copy(),
    )


def _trades(pnls: list[float], regimes: list[str] | None = None) -> pd.DataFrame:
    n = len(pnls)
    return pd.DataFrame(
        {
            "net_pnl": pnls,
            "gross_pnl": pnls,
            "fees": [0.0] * n,
            "r_multiple": [p / 100 for p in pnls],
            "bars_held": [3] * n,
            "return_pct": [p / 10_000 for p in pnls],
            "regime_trend": regimes or ["trending"] * n,
            "regime_volatility": ["normal"] * n,
            "regime_market": ["bull"] * n,
        }
    )


def test_drawdown_and_duration() -> None:
    eq = pd.Series(
        [100, 120, 90, 100, 130, 125],
        index=pd.date_range(T0, periods=6, freq="1D", tz="UTC"),
        dtype=float,
    )
    dd, days = _max_drawdown(eq)
    assert dd == pytest.approx(90 / 120 - 1)
    assert days == 3  # under water from day 2 until the new high on day 4


def test_trade_metrics() -> None:
    m = compute_metrics(_result([100, 110, 105, 121], _trades([200, -100, 300, -100])))
    assert m["total_return"] == pytest.approx(0.21)
    assert m["trades"] == 4 and m["win_rate"] == 0.5
    assert m["profit_factor"] == pytest.approx(500 / 200)
    assert m["expectancy_r"] == pytest.approx(0.75)
    assert m["avg_loss"] == -100


def test_regime_breakdown_groups() -> None:
    rb = regime_breakdown(_trades([100, -50, 200], ["trending", "ranging", "trending"]))
    row = rb[(rb["dimension"] == "trend") & (rb["regime"] == "trending")].iloc[0]
    assert row["trades"] == 2 and row["net_pnl"] == 300 and row["win_rate"] == 1.0


def test_monte_carlo_shuffle_keeps_final_return() -> None:
    t = _trades(list(np.random.default_rng(0).normal(20, 100, 60)))
    mc = monte_carlo(t, runs=300)
    assert mc["dd_p5"] <= mc["dd_p50"] <= mc["dd_p95"] <= 0
    assert mc["return_p5"] <= mc["return_p50"] <= mc["return_p95"]
    assert mc["actual_return"] == pytest.approx(np.prod(1 + t["return_pct"]) - 1)
    assert monte_carlo(_trades([10]))["trades"] == 1


def test_walk_forward_windows_never_overlap_test_data() -> None:
    w = split_windows(1000, 200, 4, 2.0)
    for tr, te, end in w:
        assert tr < te < end
    tests = [(te, end) for _, te, end in w]
    assert all(a[1] <= b[0] for a, b in zip(tests, tests[1:], strict=False))


def test_walk_forward_runs_and_reports_oos() -> None:
    df = make_candles(T0, 1200, "1h", seed=8)
    df[["open", "high", "low", "close"]] += 20 * np.sin(np.arange(1200) / 60)[:, None]
    ctx = StrategyContext("X/USDT", "1h", "UTC", "crypto")
    wf = walk_forward(df, STRATEGIES["ema_crossover"], ctx, ZeroCostModel(), n_windows=3)
    assert len(wf.windows) == 3
    assert all(set(p) <= set(STRATEGIES["ema_crossover"].param_grid) for p in wf.windows["params"])
    assert wf.oos.equity.index[0] == wf.windows["test_from"].iloc[0]
    assert wf.oos.equity.index.is_monotonic_increasing


def test_leaderboard_saves_db_and_csv(store, tmp_path) -> None:  # noqa: ANN001
    cfg = load_config(PROJECT_ROOT / "config.example.yaml")
    cfg.watchlist.nse, cfg.watchlist.crypto = [], ["AAA/USDT"]
    df = make_candles(T0, 800, "1d", seed=4)
    df[["open", "high", "low", "close"]] += 30 * np.sin(np.arange(800) / 50)[:, None]
    store.upsert("AAA/USDT", "1d", df, "test")
    lb = run_leaderboard(
        cfg, store, strategies=["ema_crossover", "rsi_mean_reversion"], csv_dir=tmp_path
    )
    t = lb.table
    assert set(t["strategy"]) == {"ema_crossover", "rsi_mean_reversion"}
    ranked = t[t["rank"].notna()]
    assert list(ranked["rank"]) == list(range(1, len(ranked) + 1))
    assert (t.loc[t["rank"].isna(), "trades"] < cfg.backtest.min_trades).all()
    assert lb.csv_path is not None and lb.csv_path.exists()
    rs = ResultStore(store.engine)
    runs = rs.runs(kind="leaderboard", batch_id=lb.batch_id)
    assert len(runs) == len(t)
    with_trades = runs[runs["trades"] > 0]
    if len(with_trades):
        tr = rs.trades(with_trades["run_id"].iloc[0])
        assert len(tr) == with_trades["trades"].iloc[0]
        assert str(tr["entry_ts"].dt.tz) == "UTC"


def test_run_one_needs_data(store) -> None:  # noqa: ANN001
    cfg = load_config(PROJECT_ROOT / "config.example.yaml")
    with pytest.raises(ValueError, match="no 1d candles"):
        run_one(cfg, store, get_strategy("ema_crossover"), "NONE/USDT", "1d")
