"""Event-driven backtest engine (signal on bar t, fill at bar t+1 open).

Per bar i, in this order:
1. fill orders decided at bar i-1's close at bar i's OPEN (+ slippage, + fees);
2. intrabar stop / target on bar i's high-low. Stop first if both are touched in
   the same bar (we can't know which came first -> assume the worse one). A gap
   through the stop fills at the open, not at the stop;
3. NSE intraday: square off at the close of the session's last bar (like MIS);
4. read the strategy's signals at bar i's close -> orders for bar i+1;
5. mark equity at bar i's close.

Sizing: risk `risk_pct` of current equity between fill and stop, capped at 100%
of equity (no leverage). NSE quantities are whole shares; crypto is fractional.
Shorts only where allowed (default: NSE intraday only — no shorting delivery or spot).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sanjotracker.analysis.regime import market_regime_at_times, regime_frame
from sanjotracker.backtest.costs import INTRADAY_TFS, CostModel
from sanjotracker.config import RegimeConfig
from sanjotracker.data.providers.base import TIMEFRAME_DELTAS
from sanjotracker.strategies.base import Strategy, StrategyContext, local_times, minutes_of_day

NSE_CLOSE_MIN = 15 * 60 + 30

TRADE_COLUMNS = [
    "symbol",
    "timeframe",
    "strategy",
    "direction",
    "signal_ts",
    "entry_ts",
    "entry_price",
    "stop",
    "target",
    "qty",
    "exit_ts",
    "exit_price",
    "exit_reason",
    "bars_held",
    "gross_pnl",
    "fees",
    "net_pnl",
    "r_multiple",
    "return_pct",
    "regime_trend",
    "regime_volatility",
    "regime_market",
]


@dataclass
class BacktestResult:
    symbol: str
    timeframe: str
    strategy: str
    params: dict
    cost_model: str
    initial_capital: float
    trades: pd.DataFrame
    equity: pd.Series  # equity at each bar close, indexed by bar timestamp (UTC)
    in_market: pd.Series  # True where a position was held at the bar close
    close: pd.Series  # the symbol's close over the same bars (for buy & hold)
    meta: dict = field(default_factory=dict)


@dataclass
class _Pos:
    direction: int
    qty: float
    entry_i: int
    signal_i: int
    entry_fill: float
    entry_fees: float
    stop: float
    target: float
    equity_at_entry: float


def _session_last(df: pd.DataFrame, timeframe: str) -> np.ndarray:
    """NSE: True for bars whose close reaches the 15:30 session end."""
    tf_min = TIMEFRAME_DELTAS[timeframe].total_seconds() / 60
    mins = minutes_of_day(local_times(df, "Asia/Kolkata"))
    return (mins + tf_min >= NSE_CLOSE_MIN).to_numpy()


def run_backtest(
    df: pd.DataFrame,
    strategy: Strategy,
    ctx: StrategyContext,
    cost: CostModel,
    *,
    initial_capital: float = 100_000,
    risk_pct: float = 1.0,
    allow_short: bool | None = None,
    eod_exit: bool | None = None,
    start: int = 0,
    end: int | None = None,
    benchmark_df: pd.DataFrame | None = None,
    regime_cfg: RegimeConfig | None = None,
    close_at_end: bool = True,
) -> BacktestResult:
    """Backtest one strategy on one symbol/timeframe.

    close_at_end=False leaves a still-open trade open and describes it in
    meta["open_position"] (paper trading needs that; backtests close it).

    Bars [start, end) are traded and reported; bars before `start` are only
    warm-up history for indicators. Data after `end` is cut off BEFORE the
    strategy runs, so it can't leak in.
    """
    nse_intraday = ctx.market == "nse" and ctx.timeframe in INTRADAY_TFS
    allow_short = nse_intraday if allow_short is None else allow_short
    eod_exit = nse_intraday if eod_exit is None else eod_exit
    fractional = ctx.market == "crypto"

    d = df.iloc[:end].reset_index(drop=True) if end is not None else df.reset_index(drop=True)
    n = len(d)
    sig = strategy.generate_signals(d, ctx).reset_index(drop=True)
    o, h, lo, c = (d[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    ts = pd.to_datetime(d["timestamp"], utc=True)
    entry_sig = sig["entry"].to_numpy()
    exit_sig = sig["exit"].to_numpy(dtype=bool)
    stop_sig = sig["stop"].to_numpy(dtype=float)
    tgt_sig = sig["target_r"].to_numpy(dtype=float)
    last_bar = _session_last(d, ctx.timeframe) if eod_exit else np.zeros(n, dtype=bool)
    day = local_times(d, "Asia/Kolkata").dt.date.to_numpy() if eod_exit else None
    max_hold = strategy.max_hold()

    reg = regime_frame(d, regime_cfg or RegimeConfig())
    close_times = ts + TIMEFRAME_DELTAS[ctx.timeframe]
    market = market_regime_at_times(
        benchmark_df, close_times, (regime_cfg or RegimeConfig()).benchmark_ema
    )

    cash = float(initial_capital)
    pos: _Pos | None = None
    pending_exit: str | None = None
    pending_entry: tuple[int, float, float, int] | None = None  # dir, stop, target_r, signal_i
    trades: list[dict] = []
    equity = np.full(n, float(initial_capital))
    held = np.zeros(n, dtype=bool)

    def close_pos(i: int, raw_price: float, reason: str) -> None:
        nonlocal cash, pos
        assert pos is not None
        side = "sell" if pos.direction == 1 else "buy"
        fill = cost.fill_price(side, raw_price)
        fees = cost.charges(side, fill, pos.qty)
        cash += pos.direction * pos.qty * fill - fees
        gross = pos.direction * (fill - pos.entry_fill) * pos.qty
        total_fees = pos.entry_fees + fees
        net = gross - total_fees
        risk = abs(pos.entry_fill - pos.stop) * pos.qty
        s = pos.signal_i
        trades.append(
            {
                "symbol": ctx.symbol,
                "timeframe": ctx.timeframe,
                "strategy": strategy.label(),
                "direction": "long" if pos.direction == 1 else "short",
                "signal_ts": ts.iloc[s],
                "entry_ts": ts.iloc[pos.entry_i],
                "entry_price": pos.entry_fill,
                "stop": pos.stop,
                "target": pos.target,
                "qty": pos.qty,
                "exit_ts": ts.iloc[i],
                "exit_price": fill,
                "exit_reason": reason,
                "bars_held": i - pos.entry_i,
                "gross_pnl": gross,
                "fees": total_fees,
                "net_pnl": net,
                "r_multiple": net / risk if risk > 0 else np.nan,
                "return_pct": net / pos.equity_at_entry,
                "regime_trend": reg["trend"].iloc[s],
                "regime_volatility": reg["volatility"].iloc[s],
                "regime_market": market[s],
            }
        )
        pos = None

    def open_pos(i: int, direction: int, stop: float, target_r: float, signal_i: int) -> None:
        nonlocal cash, pos
        side = "buy" if direction == 1 else "sell"
        fill = cost.fill_price(side, o[i])
        per_unit = (fill - stop) * direction
        if not per_unit > 0:  # opened beyond the stop (gap) -> no trade
            return
        eq = cash
        qty = min(eq * risk_pct / 100 / per_unit, eq * 0.999 / fill)
        qty = round(qty, 6) if fractional else math.floor(qty)
        if qty <= 0:
            return
        fees = cost.charges(side, fill, qty)
        cash -= direction * qty * fill + fees
        target = fill + direction * target_r * per_unit if np.isfinite(target_r) else np.nan
        pos = _Pos(direction, qty, i, signal_i, fill, fees, stop, target, eq)

    for i in range(n):
        # 1) orders from the previous close fill at this open
        if pending_exit and pos is not None:
            close_pos(i, o[i], pending_exit)
        if pending_entry and pos is None:
            open_pos(i, *pending_entry)
        pending_exit, pending_entry = None, None
        if pos is not None and eod_exit and day[i] != day[pos.entry_i]:
            close_pos(i, o[i], "eod_missed")  # data gap hid the session's last bar

        # 2) intrabar stop / target
        if pos is not None:
            if pos.direction == 1:
                if lo[i] <= pos.stop:
                    close_pos(i, min(o[i], pos.stop), "stop")
                elif np.isfinite(pos.target) and h[i] >= pos.target:
                    close_pos(i, max(o[i], pos.target), "target")
            else:
                if h[i] >= pos.stop:
                    close_pos(i, max(o[i], pos.stop), "stop")
                elif np.isfinite(pos.target) and lo[i] <= pos.target:
                    close_pos(i, min(o[i], pos.target), "target")

        # 3) intraday square-off at the session close
        if pos is not None and last_bar[i]:
            close_pos(i, c[i], "eod")

        # 4) signals at this close -> orders for the next open
        can_trade = start <= i < n - 1 and not last_bar[i]
        if pos is not None and i < n - 1:
            if exit_sig[i]:
                pending_exit = "signal"
                if (
                    can_trade
                    and entry_sig[i] == -pos.direction
                    and (allow_short or entry_sig[i] > 0)
                ):
                    pending_entry = (int(entry_sig[i]), stop_sig[i], tgt_sig[i], i)  # reverse
            elif max_hold and i - pos.entry_i + 1 >= max_hold:
                pending_exit = "time"
        elif pos is None and can_trade and entry_sig[i] != 0:
            direction = int(np.sign(entry_sig[i]))
            stop_ok = np.isfinite(stop_sig[i]) and (c[i] - stop_sig[i]) * direction > 0
            if stop_ok and (allow_short or direction > 0):
                pending_entry = (direction, stop_sig[i], tgt_sig[i], i)

        # 5) mark to market
        if pos is not None:
            held[i] = True
            equity[i] = cash + pos.direction * pos.qty * c[i]
        else:
            equity[i] = cash

    open_position = None
    if pos is not None and close_at_end:
        close_pos(n - 1, c[n - 1], "end")
        equity[n - 1] = cash
    elif pos is not None:
        s = pos.signal_i
        open_position = {
            "direction": "long" if pos.direction == 1 else "short",
            "signal_ts": ts.iloc[s],
            "entry_ts": ts.iloc[pos.entry_i],
            "entry_price": pos.entry_fill,
            "stop": pos.stop,
            "target": pos.target,
            "qty": pos.qty,
            "entry_fees": pos.entry_fees,
            "risk_amount": abs(pos.entry_fill - pos.stop) * pos.qty,
            "last_close": float(c[n - 1]),
            "unrealized_pnl": pos.direction * (c[n - 1] - pos.entry_fill) * pos.qty
            - pos.entry_fees,
            "regime_trend": reg["trend"].iloc[s],
            "regime_volatility": reg["volatility"].iloc[s],
            "regime_market": market[s],
        }

    idx = pd.DatetimeIndex(ts)
    rng = slice(start, n)
    tdf = pd.DataFrame(trades, columns=TRADE_COLUMNS)
    return BacktestResult(
        symbol=ctx.symbol,
        timeframe=ctx.timeframe,
        strategy=strategy.label(),
        params=strategy.params.model_dump(),
        cost_model=cost.name,
        initial_capital=initial_capital,
        trades=tdf,
        equity=pd.Series(equity[rng], index=idx[rng], name="equity"),
        in_market=pd.Series(held[rng], index=idx[rng], name="in_market"),
        close=pd.Series(c[rng], index=idx[rng], name="close"),
        meta={
            "allow_short": allow_short,
            "eod_exit": eod_exit,
            "bars": n - start,
            "open_position": open_position,
        },
    )
