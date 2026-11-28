"""Performance metrics with regime breakdown.

Notes:
- Sharpe/Sortino use per-bar returns annualised by the measured bars-per-year
  (works for any NSE timeframe). Risk-free rate default 0;
  pass rf_annual=0.065 to compare against Indian T-bills.
- Expectancy is in R (net P&L / initial risk): +0.2R means you make 0.2x what
  you risk per trade on average, after costs.
- On short histories (e.g. 59 days of free 15m data) annualised numbers are
  very noisy; look at `years` and `trades` before trusting anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sanjotracker.backtest.engine import BacktestResult

YEAR_SECONDS = 365.25 * 24 * 3600


def _max_drawdown(eq: pd.Series) -> tuple[float, float]:
    """(max drawdown as a negative fraction, longest time under water in days)."""
    if eq.empty:
        return 0.0, 0.0
    peak = eq.cummax()
    dd = eq / peak - 1
    under = dd < 0
    longest, run_start = 0.0, None
    idx = eq.index
    for k, u in enumerate(under.to_numpy()):
        t = idx[k]
        if u and run_start is None:
            run_start = idx[k - 1] if k else t  # measured from the peak, not the first loss
        elif not u and run_start is not None:
            longest = max(longest, (t - run_start).total_seconds() / 86400)
            run_start = None
    if run_start is not None:
        longest = max(longest, (eq.index[-1] - run_start).total_seconds() / 86400)
    return float(dd.min()), round(longest, 1)


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b not in (0, 0.0) and np.isfinite(b) else float("nan")


def compute_metrics(res: BacktestResult, rf_annual: float = 0.0) -> dict[str, float]:
    eq = res.equity
    t = res.trades
    m: dict[str, float] = {}
    if len(eq) < 2:
        return {"trades": 0.0}
    years = (eq.index[-1] - eq.index[0]).total_seconds() / YEAR_SECONDS
    rets = eq.pct_change().dropna()
    bpy = len(rets) / years if years > 0 else np.nan
    total = eq.iloc[-1] / eq.iloc[0] - 1
    m["years"] = round(years, 2)
    m["total_return"] = total
    m["cagr"] = (1 + total) ** (1 / years) - 1 if years > 0 and total > -1 else np.nan
    rf_bar = (1 + rf_annual) ** (1 / bpy) - 1 if bpy and np.isfinite(bpy) else 0.0
    ex = rets - rf_bar
    sd = ex.std(ddof=0)
    downside = np.sqrt(np.mean(np.minimum(ex, 0) ** 2))
    m["sharpe"] = _safe_div(ex.mean(), sd) * np.sqrt(bpy) if sd > 0 else np.nan
    m["sortino"] = _safe_div(ex.mean(), downside) * np.sqrt(bpy) if downside > 0 else np.nan
    m["max_drawdown"], m["max_dd_days"] = _max_drawdown(eq)
    m["calmar"] = _safe_div(m["cagr"], abs(m["max_drawdown"])) if m["max_drawdown"] < 0 else np.nan
    m["exposure_pct"] = float(res.in_market.mean() * 100)

    n = len(t)
    m["trades"] = float(n)
    if n:
        wins, losses = t[t["net_pnl"] > 0], t[t["net_pnl"] <= 0]
        m["win_rate"] = len(wins) / n
        m["avg_win"] = float(wins["net_pnl"].mean()) if len(wins) else 0.0
        m["avg_loss"] = float(losses["net_pnl"].mean()) if len(losses) else 0.0
        gross_loss = abs(float(losses["net_pnl"].sum()))
        m["profit_factor"] = _safe_div(float(wins["net_pnl"].sum()), gross_loss)
        m["expectancy_r"] = float(t["r_multiple"].mean())
        m["avg_bars_held"] = float(t["bars_held"].mean())
        m["fees_total"] = float(t["fees"].sum())
        m["fees_pct_of_gross"] = _safe_div(m["fees_total"], float(t["gross_pnl"].abs().sum()))
    else:
        for k in (
            "win_rate",
            "avg_win",
            "avg_loss",
            "profit_factor",
            "expectancy_r",
            "avg_bars_held",
            "fees_total",
            "fees_pct_of_gross",
        ):
            m[k] = np.nan
    m["buy_hold_return"] = float(res.close.iloc[-1] / res.close.iloc[0] - 1)
    m["excess_vs_buy_hold"] = total - m["buy_hold_return"]
    return {k: (round(float(v), 6) if v is not None else np.nan) for k, v in m.items()}


def regime_breakdown(trades: pd.DataFrame) -> pd.DataFrame:
    """Trades grouped by trend / volatility / market regime at the signal bar."""
    cols = ["dimension", "regime", "trades", "win_rate", "avg_r", "net_pnl"]
    if trades.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for dim in ("regime_trend", "regime_volatility", "regime_market"):
        for reg, g in trades.groupby(dim):
            rows.append(
                {
                    "dimension": dim.removeprefix("regime_"),
                    "regime": reg,
                    "trades": len(g),
                    "win_rate": round(float((g["net_pnl"] > 0).mean()), 3),
                    "avg_r": round(float(g["r_multiple"].mean()), 3),
                    "net_pnl": round(float(g["net_pnl"].sum()), 2),
                }
            )
    return pd.DataFrame(rows, columns=cols)
