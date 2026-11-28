"""Cached data helpers shared by the Streamlit pages in app/.

Everything here reads the local SQLite DB (fast, no network). Pages fetch new
candles only when the user clicks a "fetch latest" button.
Cache keys include the DB path so tests with a temporary DB never mix with real data.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

from sanjotracker.alerts.dashboard_feed import AlertFeed
from sanjotracker.analysis.breakout import detect_events, events_frame, fmt_price
from sanjotracker.analysis.levels import compute_levels, find_pivots, nearest_levels
from sanjotracker.analysis.regime import Regime, current_regime
from sanjotracker.backtest.metrics import compute_metrics, regime_breakdown
from sanjotracker.backtest.montecarlo import monte_carlo
from sanjotracker.backtest.runner import run_one
from sanjotracker.config import Config, load_config
from sanjotracker.data.store import CandleStore
from sanjotracker.data.updater import build_updater, session_tz_for
from sanjotracker.scanner import Scanner
from sanjotracker.strategies import STRATEGIES

IST = "Asia/Kolkata"
INTRADAY = ("5m", "15m", "1h")
EVENT_ICON = {
    "breakout": ":material/trending_up:",
    "breakdown": ":material/trending_down:",
    "retest": ":material/replay:",
    "failed_breakout": ":material/warning:",
    "failed_breakdown": ":material/warning:",
    "approaching": ":material/visibility:",
}


# ------------------------------------------------------------------ config / resources


@st.cache_resource
def _logging() -> bool:
    """Configure loguru once per server process (INFO to console, DEBUG to logs/)."""
    from sanjotracker.log import setup_logging

    setup_logging()
    return True


def cfg() -> Config:
    """Fresh config each rerun (a few ms), so Settings edits show up immediately."""
    _logging()
    return load_config()


@st.cache_resource
def _store(db_file: str) -> CandleStore:
    return CandleStore(db_file)


@st.cache_resource
def _feed(db_file: str) -> AlertFeed:
    return AlertFeed(engine=_store(db_file).engine)


def store() -> CandleStore:
    return _store(str(cfg().db_file))


def feed() -> AlertFeed:
    return _feed(str(cfg().db_file))


def db_key() -> str:
    return str(cfg().db_file)


def all_symbols(c: Config | None = None) -> list[str]:
    c = c or cfg()
    syms = [*c.watchlist.nse, c.benchmarks.nse]
    return list(dict.fromkeys(syms))


# ------------------------------------------------------------------ candles / analysis


@st.cache_data(ttl=60, max_entries=200, show_spinner=False)
def candles(db: str, symbol: str, timeframe: str, limit: int | None = None) -> pd.DataFrame:
    return _store(db).load(symbol, timeframe, limit=limit)


@st.cache_data(ttl=300, max_entries=100, show_spinner=False)
def levels_and_pivots(db: str, symbol: str, timeframe: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    c = cfg()
    df = candles(db, symbol, timeframe, c.breakout.scan_history_bars)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    lv = compute_levels(df, symbol, timeframe, c.levels, session_tz=session_tz_for(symbol))
    return lv, find_pivots(df, c.levels.pivot_left, c.levels.pivot_right)


@st.cache_data(ttl=300, max_entries=100, show_spinner=False)
def recent_events(db: str, symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
    """Breakout events over the last `bars` bars (each judged with only prior data)."""
    c = cfg()
    df = candles(db, symbol, timeframe, c.breakout.scan_history_bars)
    if len(df) < 50:
        return pd.DataFrame()
    ev = detect_events(
        df,
        symbol,
        timeframe,
        cfg=c.breakout,
        levels_cfg=c.levels,
        regime_cfg=c.regime,
        session_tz=session_tz_for(symbol),
        start=max(0, len(df) - bars),
    )
    return events_frame(ev)


@st.cache_data(ttl=300, max_entries=50, show_spinner=False)
def regime_of(db: str, symbol: str, timeframe: str = "1d") -> Regime:
    c = cfg()
    df = candles(db, symbol, timeframe)
    bench = candles(db, c.benchmarks.nse, "1d")
    return current_regime(df, timeframe, bench, c.regime)


@st.cache_data(ttl=60, show_spinner=False)
def watchlist_table(db: str, symbols: tuple[str, ...]) -> pd.DataFrame:
    """One row per symbol from daily data: price, day change, regime, nearest zones."""
    rows = []
    for sym in symbols:
        df = candles(db, sym, "1d", 400)
        if df.empty:
            rows.append({"symbol": sym, "status": "no data"})
            continue
        price = float(df["close"].iloc[-1])
        prev = float(df["close"].iloc[-2]) if len(df) > 1 else price
        lv, _ = levels_and_pivots(db, sym, "1d")
        near = nearest_levels(lv[lv["source"] == "pivot"], price, 1) if not lv.empty else lv
        below = near[near["zone_high"] <= price] if not near.empty else near
        above = near[near["zone_low"] >= price] if not near.empty else near
        r = regime_of(db, sym, "1d")
        rows.append(
            {
                "symbol": sym,
                "last": price,
                "change_pct": (price / prev - 1) * 100,
                "trend": r.trend,
                "volatility": r.volatility,
                "support": float(below["zone_high"].iloc[0]) if len(below) else np.nan,
                "resistance": float(above["zone_low"].iloc[0]) if len(above) else np.nan,
                "to_support_pct": (float(below["zone_high"].iloc[0]) / price - 1) * 100
                if len(below)
                else np.nan,
                "to_resistance_pct": (float(above["zone_low"].iloc[0]) / price - 1) * 100
                if len(above)
                else np.nan,
                "spark": df["close"].tail(30).round(4).tolist(),
                "last_bar": pd.Timestamp(df["timestamp"].iloc[-1]),
                "status": "ok",
            }
        )
    return pd.DataFrame(rows)


def fetch_latest(symbols: list[str], timeframes: list[str]) -> pd.DataFrame:
    """Network: bring candles up to date (button-triggered only). Clears caches."""
    c = cfg()
    up = build_updater(store(), c.providers)
    res = up.update_many(symbols, timeframes)
    st.cache_data.clear()
    return pd.DataFrame([r.model_dump() for r in res])


# ------------------------------------------------------------------ alerts / scanner


@st.cache_data(ttl=30, show_spinner=False)
def alerts_since(db: str, hours: int, limit: int = 500) -> pd.DataFrame:
    since = datetime.now(UTC) - timedelta(hours=hours)
    return _feed(db).recent(since=since, limit=limit)


@st.cache_data(ttl=300, max_entries=20, show_spinner=False)
def scan(
    db: str, style: str, lookback: int, counter_trend: bool
) -> tuple[pd.DataFrame, dict[str, str], dict[str, dict]]:
    sc = Scanner(cfg(), _store(db))
    ev = sc.scan(
        style,  # type: ignore[arg-type]
        update=False,
        lookback_bars=lookback,
        include_counter_trend=counter_trend,
    )
    return events_frame(ev), sc.errors, {k: v.model_dump() for k, v in sc.regimes.items()}


# ------------------------------------------------------------------ backtests


@st.cache_data(ttl=600, max_entries=50, show_spinner=False)
def backtest(db: str, strategy: str, params_json: str, symbol: str, timeframe: str) -> dict:
    """Run one backtest; returns plain data (picklable) for the page."""
    c = cfg()
    strat = STRATEGIES[strategy](**json.loads(params_json))
    res = run_one(c, _store(db), strat, symbol, timeframe)
    m = compute_metrics(res, c.backtest.risk_free_pct / 100)
    return {
        "label": res.strategy,
        "metrics": m,
        "trades": res.trades,
        "equity": res.equity,
        "close": res.close,
        "regimes": regime_breakdown(res.trades),
        "mc": monte_carlo(res.trades, c.backtest.monte_carlo_runs),
        "cost_model": res.cost_model,
        "meta": res.meta,
    }


# ------------------------------------------------------------------ formatting


def regime_badges(name: str, r: Regime) -> None:
    """Small coloured badges: market / trend / volatility."""
    with st.container(horizontal=True, vertical_alignment="center"):
        st.markdown(f"**{name}**")
        st.badge(
            r.market,
            icon=":material/trending_up:" if r.market == "bull" else ":material/trending_down:",
            color={"bull": "green", "bear": "red"}.get(r.market, "gray"),
            help="Benchmark close vs its 200-day EMA",
        )
        adx = "" if r.adx is None else f" · ADX {r.adx:.0f}"
        st.badge(
            f"{r.trend}{adx}",
            color={"trending": "blue", "ranging": "orange"}.get(r.trend, "gray"),
            help="ADX(14) > 25 trending, < 20 ranging",
        )
        st.badge(
            f"{r.volatility} vol",
            color={"low": "green", "high": "red"}.get(r.volatility, "gray"),
            help="ATR% percentile over the last year",
        )


def to_ist(s: pd.Series, fmt: str = "%d %b %H:%M") -> pd.Series:
    return pd.to_datetime(s, utc=True).dt.tz_convert(IST).dt.strftime(fmt)


__all__ = ["fmt_price"]


@st.cache_data(ttl=3600, max_entries=20, show_spinner=False)
def walkforward(db: str, strategy: str, symbol: str, timeframe: str) -> dict:
    from sanjotracker.backtest.costs import cost_model_for
    from sanjotracker.backtest.runner import make_context
    from sanjotracker.backtest.walkforward import walk_forward

    c = cfg()
    bt = c.backtest
    bench = candles(db, c.benchmarks.nse, "1d")
    wf = walk_forward(
        candles(db, symbol, timeframe),
        STRATEGIES[strategy],
        make_context(c, symbol, timeframe),
        cost_model_for(symbol, timeframe, c.costs),
        n_windows=bt.walkforward_windows,
        train_ratio=bt.walkforward_train_ratio,
        objective=bt.walkforward_objective,
        initial_capital=bt.initial_capital,
        risk_pct=bt.risk_per_trade_pct,
        benchmark_df=None if bench.empty else bench,
    )
    return {
        "windows": wf.windows,
        "metrics": wf.oos_metrics,
        "equity": wf.oos.equity,
        "close": wf.oos.close,
        "objective": bt.walkforward_objective,
    }


@st.cache_data(ttl=60, show_spinner=False)
def latest_leaderboard(db: str) -> tuple[pd.DataFrame, str | None]:
    """Rows of the most recent leaderboard batch, metrics expanded into columns."""
    from sanjotracker.backtest.runner import ResultStore

    runs = ResultStore(_store(db).engine).runs(kind="leaderboard")
    if runs.empty:
        return runs, None
    batch = runs["batch_id"].iloc[0]
    runs = runs[runs["batch_id"] == batch].reset_index(drop=True)
    m = pd.DataFrame([json.loads(x) for x in runs["metrics"]])
    mc = pd.DataFrame([json.loads(x) if x else {} for x in runs["monte_carlo"]])
    out = pd.concat([runs[["run_id", "strategy", "symbol", "timeframe"]], m], axis=1)
    out["mc_dd_p5"] = mc.get("dd_p5")
    return out, batch


# ------------------------------------------------------------------ paper trading


@st.cache_resource
def _paper_book(db_file: str):  # noqa: ANN202
    from sanjotracker.paper.book import PaperBook

    return PaperBook(_store(db_file).engine)


def paper_book():  # noqa: ANN201
    return _paper_book(db_key())


@st.cache_data(ttl=30, show_spinner=False)
def paper_account(db: str, capital: float) -> dict:
    from sanjotracker.paper.trader import paper_summary

    return paper_summary(_paper_book(db), _store(db), capital)


def paper_step(notify: bool = True) -> dict:
    """Network + Telegram: fetch new candles and advance paper trades (button only)."""
    from sanjotracker.alerts.telegram_bot import NullNotifier, build_notifier
    from sanjotracker.paper.trader import PaperTrader

    c = cfg()
    up = build_updater(store(), c.providers)
    notifier = build_notifier(c) if notify else NullNotifier()
    res = PaperTrader(c, store(), notifier, up).run()
    st.cache_data.clear()
    return {
        "setups": res.setups,
        "opened": len(res.opened),
        "closed": len(res.closed),
        "errors": res.errors,
    }
