"""Paper trader: virtual positions from strategy signals, never real orders.

How a run works, for each setup (strategy x symbol x timeframe):
1. Replay the strategy with the backtest engine over the stored CLOSED candles,
   trading only bars that closed after the setup was first registered (no
   back-dated trades). Same rules as backtests: signal at bar t close, fill at
   bar t+1 open, ATR stop / R target, stop wins ties, costs, 1% risk sizing.
2. Reconcile with the paper_trades table:
   - a replayed trade we don't have yet -> insert it (open or already closed);
   - one of our open trades that the replay has now closed -> close it;
   - closed trades are never rewritten.
3. Notify Telegram for each new entry and each exit.

Because the engine is causal (tested), replaying from the start each run gives the
same trades a live bar-by-bar trader would have taken.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from loguru import logger

from sanjotracker.alerts.telegram_bot import Notifier, NullNotifier
from sanjotracker.analysis.breakout import fmt_price
from sanjotracker.analysis.indicators import atr
from sanjotracker.backtest.costs import cost_model_for
from sanjotracker.backtest.engine import run_backtest
from sanjotracker.backtest.runner import make_context
from sanjotracker.config import Config, PaperSetup
from sanjotracker.data.providers.base import TIMEFRAME_DELTAS
from sanjotracker.data.store import CandleStore
from sanjotracker.data.updater import Updater, normalize_symbol
from sanjotracker.paper.book import PaperBook, epoch
from sanjotracker.strategies import STRATEGIES, Strategy, StrategyContext


class AtrRiskOverride(Strategy):
    """Wraps a strategy: same entries/exits, but stop = close -/+ stop_atr x ATR and
    target = target_r x risk (either can be None to keep the strategy's own)."""

    name = "atr_override"

    def __init__(self, inner: Strategy, stop_atr: float | None, target_r: float | None) -> None:
        self.inner = inner
        self.params = inner.params
        self.Params = inner.Params  # type: ignore[misc]
        self.stop_atr = stop_atr
        self.target_r = target_r

    def max_hold(self) -> int | None:
        return self.inner.max_hold()

    def supports(self, timeframe: str) -> bool:
        return self.inner.supports(timeframe)

    def label(self) -> str:
        extra = []
        if self.stop_atr is not None:
            extra.append(f"{self.stop_atr:g}ATR stop")
        if self.target_r is not None:
            extra.append(f"{self.target_r:g}R target")
        return self.inner.label() + (f" [{', '.join(extra)}]" if extra else "")

    def generate_signals(self, df: pd.DataFrame, ctx: StrategyContext) -> pd.DataFrame:
        sig = self.inner.generate_signals(df, ctx).copy()
        entries = sig["entry"].to_numpy() != 0
        if self.stop_atr is not None:
            a = atr(df).to_numpy()
            c = df["close"].to_numpy(dtype=float)
            stop = c - np.sign(sig["entry"].to_numpy()) * self.stop_atr * a
            sig["stop"] = np.where(entries, stop, sig["stop"])
        if self.target_r is not None:
            sig["target_r"] = np.where(entries, self.target_r, sig["target_r"])
        return sig


@dataclass
class PaperRunResult:
    setups: int = 0
    opened: list[dict] = field(default_factory=list)
    closed: list[dict] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)


def expand_symbols(setup: PaperSetup, cfg: Config) -> list[str]:
    out: list[str] = []
    for s in setup.symbols:
        out += list(cfg.watchlist.nse) if s == "watchlist" else [normalize_symbol(s)]
    return list(dict.fromkeys(out))


def build_strategy(setup: PaperSetup, cfg: Config) -> Strategy:
    if setup.strategy not in STRATEGIES:
        raise KeyError(f"unknown strategy {setup.strategy!r}")
    inner = STRATEGIES[setup.strategy](**setup.params)
    stop_atr = setup.stop_atr if setup.stop_atr is not None else cfg.paper.stop_atr
    target_r = setup.target_r if setup.target_r is not None else cfg.paper.target_r
    if stop_atr is None and target_r is None:
        return inner
    return AtrRiskOverride(inner, stop_atr, target_r)


def setup_id(strategy: Strategy, symbol: str, timeframe: str) -> str:
    key = json.dumps(
        [strategy.label(), strategy.params.model_dump(), symbol, timeframe], default=str
    )
    return hashlib.sha1(key.encode()).hexdigest()[:12]


# ------------------------------------------------------------------ messages


def format_entry(t: dict, tz: str = "Asia/Kolkata") -> str:
    when = pd.Timestamp(t["entry_ts"], unit="s", tz="UTC").tz_convert(tz).strftime("%d %b %H:%M")
    arrow = "🟢 LONG" if t["direction"] == "long" else "🔴 SHORT"
    tgt = (
        f" · target {fmt_price(t['target'])}"
        if t.get("target") and np.isfinite(t["target"])
        else ""
    )
    return (
        f"📝 <b>PAPER {arrow}</b> {t['symbol']} {t['timeframe']}\n"
        f"{t['label']}\n"
        f"Entry {fmt_price(t['entry_price'])} ({when}) · stop {fmt_price(t['stop'])}{tgt}\n"
        f"Qty {t['qty']:g} · risk {t['risk_amount']:,.0f}\n"
        f"<i>virtual trade, no real order</i>"
    )


def format_exit(t: dict, tz: str = "Asia/Kolkata") -> str:
    when = pd.Timestamp(t["exit_ts"], unit="s", tz="UTC").tz_convert(tz).strftime("%d %b %H:%M")
    icon = "✅" if (t.get("net_pnl") or 0) > 0 else "❌"
    return (
        f"📝 {icon} <b>PAPER EXIT</b> {t['symbol']} {t['timeframe']} ({t['direction']})\n"
        f"{t['label']}\n"
        f"{fmt_price(t['entry_price'])} → {fmt_price(t['exit_price'])} "
        f"({t['exit_reason']}, {when})\n"
        f"P&L {t['net_pnl']:+,.0f} · {t['r_multiple']:+.2f}R"
    )


# ------------------------------------------------------------------ trader


class PaperTrader:
    def __init__(
        self,
        cfg: Config,
        store: CandleStore,
        notifier: Notifier | None = None,
        updater: Updater | None = None,
        skip_update: set[str] | None = None,
    ) -> None:
        """`updater`: fetch new candles before each setup (symbols in `skip_update`
        are assumed fresh, e.g. the watchlist the scanner just refreshed)."""
        self.skip_update = skip_update or set()
        self.cfg = cfg
        self.store = store
        self.book = PaperBook(store.engine)
        self.notifier = notifier or NullNotifier()
        self.updater = updater

    def _notify(self, text: str) -> bool:
        if not self.cfg.paper.telegram or isinstance(self.notifier, NullNotifier):
            return False
        try:
            self.notifier.send_text(text)
            return True
        except Exception as exc:
            logger.error("paper notification failed: {}", exc)
            return False

    def run(
        self,
        timeframes: set[str] | None = None,
        now: datetime | None = None,
    ) -> PaperRunResult:
        """One paper step for every configured setup (optionally filtered)."""
        out = PaperRunResult()
        if not self.cfg.paper.enabled:
            return out
        now = now or datetime.now(UTC)
        for setup in self.cfg.paper.setups:
            if timeframes and setup.timeframe not in timeframes:
                continue
            try:
                strat = build_strategy(setup, self.cfg)
            except Exception as exc:
                out.errors[setup.strategy] = str(exc)
                continue
            for sym in expand_symbols(setup, self.cfg):
                if not strat.supports(setup.timeframe):
                    continue
                try:
                    self._run_one(strat, sym, setup.timeframe, now, out)
                    out.setups += 1
                except Exception as exc:  # one symbol must not stop the rest
                    logger.exception("paper {} {} {}", strat.label(), sym, setup.timeframe)
                    out.errors[f"{strat.label()} {sym} {setup.timeframe}"] = str(exc)
        logger.info(
            "paper step: {} setups, {} opened, {} closed, {} errors",
            out.setups,
            len(out.opened),
            len(out.closed),
            len(out.errors),
        )
        return out

    def _run_one(
        self, strat: Strategy, symbol: str, tf: str, now: datetime, out: PaperRunResult
    ) -> None:
        if self.updater is not None and symbol not in self.skip_update:
            self.updater.update(symbol, tf, now=now)
        sid = setup_id(strat, symbol, tf)
        start_ts = self.book.ensure_setup(
            {
                "setup_id": sid,
                "label": strat.label(),
                "strategy": strat.name
                if not isinstance(strat, AtrRiskOverride)
                else strat.inner.name,
                "symbol": symbol,
                "timeframe": tf,
                "params": json.dumps(strat.params.model_dump(), default=str),
            },
            epoch(now),
        )
        df = self.store.load(symbol, tf, limit=self.cfg.breakout.scan_history_bars + 2000)
        if df.empty:
            raise ValueError(f"no {tf} candles for {symbol}")
        close_times = pd.to_datetime(df["timestamp"], utc=True) + TIMEFRAME_DELTAS[tf]
        # first bar that closed AFTER the setup was registered (bars are sorted)
        start_idx = int((close_times <= pd.Timestamp(start_ts, unit="s", tz="UTC")).sum())
        if start_idx >= len(df):
            self.book.touch_setup(sid)
            return  # nothing has closed since the setup started
        bench = self.store.load(self.cfg.benchmarks.nse, "1d")
        res = run_backtest(
            df,
            strat,
            make_context(self.cfg, symbol, tf),
            cost_model_for(symbol, tf, self.cfg.costs),
            initial_capital=self.cfg.paper.capital,
            risk_pct=self.cfg.paper.risk_per_trade_pct,
            start=start_idx,
            benchmark_df=None if bench.empty else bench,
            close_at_end=False,
        )
        self._reconcile(sid, strat.label(), symbol, tf, res.trades, res.meta["open_position"], out)
        self.book.touch_setup(sid)

    def _reconcile(
        self,
        sid: str,
        label: str,
        symbol: str,
        tf: str,
        closed: pd.DataFrame,
        open_pos: dict | None,
        out: PaperRunResult,
    ) -> None:
        known = self.book.known_entries(sid)
        mine_open = self.book.open_trades(sid)
        base = {"setup_id": sid, "label": label, "symbol": symbol, "timeframe": tf}
        seen: set[int] = set()

        for t in closed.to_dict("records"):
            key = epoch(t["entry_ts"])
            seen.add(key)
            exit_vals = {
                "exit_ts": epoch(t["exit_ts"]),
                "exit_price": float(t["exit_price"]),
                "exit_reason": t["exit_reason"],
                "fees": float(t["fees"]),
                "net_pnl": float(t["net_pnl"]),
                "r_multiple": float(t["r_multiple"]),
            }
            if key in mine_open:  # our open trade has now been closed
                row = mine_open[key]
                self.book.close(row["id"], exit_vals)
                full = {**row, **exit_vals}
                out.closed.append(full)
                if self._notify(format_exit(full)):
                    self.book.mark_notified(row["id"], "exit")
            elif key not in known:  # opened and closed between two runs
                row = {
                    **base,
                    **self._entry_vals(t),
                    **exit_vals,
                    "status": "closed",
                }
                tid = self.book.insert(row)
                out.opened.append(row)
                out.closed.append(row)
                if self._notify(format_entry(row) + "\n\n" + format_exit(row)):
                    self.book.mark_notified(tid, "entry")
                    self.book.mark_notified(tid, "exit")

        if open_pos is not None:
            key = epoch(open_pos["entry_ts"])
            seen.add(key)
            if key not in known:
                row = {**base, **self._entry_vals(open_pos), "status": "open"}
                tid = self.book.insert(row)
                out.opened.append(row)
                if self._notify(format_entry(row)):
                    self.book.mark_notified(tid, "entry")

        # an open trade the replay no longer contains: the source data was revised
        for key, row in mine_open.items():
            if key not in seen:
                vals = {
                    "exit_ts": epoch(datetime.now(UTC)),
                    "exit_price": row["entry_price"],
                    "exit_reason": "data_revised",
                    "fees": 0.0,
                    "net_pnl": 0.0,
                    "r_multiple": 0.0,
                }
                self.book.close(row["id"], vals)
                logger.warning("paper trade {} closed: history was revised at source", row["id"])

    @staticmethod
    def _entry_vals(t: dict) -> dict:
        target = t.get("target")
        return {
            "direction": t["direction"],
            "signal_ts": epoch(t["signal_ts"]),
            "entry_ts": epoch(t["entry_ts"]),
            "entry_price": float(t["entry_price"]),
            "stop": float(t["stop"]),
            "target": float(target) if target is not None and np.isfinite(target) else None,
            "qty": float(t["qty"]),
            "risk_amount": float(
                t.get("risk_amount") or abs(t["entry_price"] - t["stop"]) * t["qty"]
            ),
            "regime_trend": t.get("regime_trend"),
            "regime_volatility": t.get("regime_volatility"),
            "regime_market": t.get("regime_market"),
        }


def paper_summary(book: PaperBook, store: CandleStore, capital: float) -> dict:
    """Account view: realized P&L, unrealized P&L at the last close, win rate, equity."""
    t = book.trades()
    closed = t[t["status"] == "closed"]
    opened = t[t["status"] == "open"].copy()
    unreal = 0.0
    if not opened.empty:
        lasts = []
        for r in opened.itertuples():
            last = store.load(r.symbol, r.timeframe, limit=1)
            px = float(last["close"].iloc[-1]) if len(last) else r.entry_price
            sign = 1 if r.direction == "long" else -1
            lasts.append((px, sign * (px - r.entry_price) * r.qty))
        opened["last"] = [p for p, _ in lasts]
        opened["unrealized_pnl"] = [u for _, u in lasts]
        opened["unrealized_r"] = opened["unrealized_pnl"] / opened["risk_amount"]
        unreal = float(opened["unrealized_pnl"].sum())
    realized = float(closed["net_pnl"].sum()) if len(closed) else 0.0
    return {
        "capital": capital,
        "realized": realized,
        "unrealized": unreal,
        "equity": capital + realized + unreal,
        "closed": closed,
        "open": opened,
        "win_rate": float((closed["net_pnl"] > 0).mean()) if len(closed) else np.nan,
        "avg_r": float(closed["r_multiple"].mean()) if len(closed) else np.nan,
    }
