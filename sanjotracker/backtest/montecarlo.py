"""Monte Carlo trade-order shuffles for drawdown percentiles.

Two questions, two methods:
1. "How bad could the drawdown have been with the same trades in a different
   order?" -> SHUFFLE the trade sequence 1000x. (The final return doesn't change
   with order when returns compound, only the path does.)
2. "How much could the final result vary?" -> BOOTSTRAP: resample the trades
   with replacement 1000x.
Caveat: both assume trades are independent; a real losing streak in one regime
can be worse than the shuffles suggest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _max_dd(paths: np.ndarray) -> np.ndarray:
    """Max drawdown (negative fraction) of each row of equity paths."""
    peaks = np.maximum.accumulate(paths, axis=1)
    return (paths / peaks - 1).min(axis=1)


def monte_carlo(
    trades: pd.DataFrame, runs: int = 1000, seed: int = 0, ruin_dd: float = -0.5
) -> dict[str, float]:
    """Percentiles from shuffling / bootstrapping per-trade returns (`return_pct`)."""
    r = trades["return_pct"].dropna().to_numpy(dtype=float) if len(trades) else np.array([])
    if len(r) < 2:
        return {"trades": float(len(r))}
    rng = np.random.default_rng(seed)
    start = np.ones((runs, 1))

    shuffled = np.array([rng.permutation(r) for _ in range(runs)])
    dd_shuffle = _max_dd(np.hstack([start, np.cumprod(1 + shuffled, axis=1)]))

    boot = rng.choice(r, size=(runs, len(r)), replace=True)
    boot_paths = np.hstack([start, np.cumprod(1 + boot, axis=1)])
    final = boot_paths[:, -1] - 1
    dd_boot = _max_dd(boot_paths)

    actual = np.hstack([[1.0], np.cumprod(1 + r)])
    out = {
        "trades": float(len(r)),
        "runs": float(runs),
        "actual_max_dd": float(_max_dd(actual[None, :])[0]),
        "actual_return": float(actual[-1] - 1),
        # drawdown from reordering the same trades (5th pct = bad case)
        "dd_p5": float(np.percentile(dd_shuffle, 5)),
        "dd_p50": float(np.percentile(dd_shuffle, 50)),
        "dd_p95": float(np.percentile(dd_shuffle, 95)),
        # final return if the trade mix had come out differently
        "return_p5": float(np.percentile(final, 5)),
        "return_p50": float(np.percentile(final, 50)),
        "return_p95": float(np.percentile(final, 95)),
        "prob_loss": float((final < 0).mean()),
        "prob_dd_worse_than": float((dd_boot <= ruin_dd).mean()),
        "ruin_dd": ruin_dd,
    }
    return {k: round(v, 6) for k, v in out.items()}
