"""Backtest: one strategy with costs, walk-forward, parameter heatmap, leaderboard."""

import json
import typing

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sanjotracker import dashboard as d
from sanjotracker.backtest.costs import cost_model_for
from sanjotracker.backtest.runner import run_leaderboard
from sanjotracker.charts import equity_chart
from sanjotracker.data.updater import market_of
from sanjotracker.strategies import STRATEGIES

c = d.cfg()
db = d.db_key()

mode = st.segmented_control(
    "View",
    ["Single run", "Walk-forward", "Leaderboard"],
    default="Single run",
    required=True,
    key="bt_mode",
    label_visibility="collapsed",
)


def pct(v: float | None, digits: int = 1) -> str:
    return "–" if v is None or pd.isna(v) else f"{v * 100:+.{digits}f}%"


def num(v: float | None, digits: int = 2) -> str:
    return "–" if v is None or pd.isna(v) else f"{v:.{digits}f}"


def pick_target(prefix: str) -> tuple[str, str, str]:
    """Strategy, symbol, timeframe selectors (only valid combinations offered)."""
    with st.container(horizontal=True, vertical_alignment="bottom"):
        name = st.selectbox(
            "Strategy",
            list(STRATEGIES),
            key=f"{prefix}_strategy",
            help="\n\n".join(f"**{k}**: {v.description}" for k, v in STRATEGIES.items()),
        )
        cls = STRATEGIES[name]
        syms = [s for s in d.all_symbols(c) if market_of(s) in cls.markets]
        symbol = st.selectbox("Symbol", syms, key=f"{prefix}_symbol")
        tfs = [t for t in ("5m", "15m", "1h", "1d", "1w") if t in cls.timeframes]
        tf = st.segmented_control(
            "Timeframe",
            tfs,
            default="1d" if "1d" in tfs else tfs[-1],
            required=True,
            key=f"{prefix}_tf_{name}",
        )
    return name, symbol, tf


def param_inputs(name: str) -> dict:
    """One input per strategy parameter, typed from the pydantic model."""
    cls = STRATEGIES[name]
    values: dict = {}
    fields = cls.Params.model_fields
    cols = st.columns(min(4, max(1, len(fields))))
    for i, (k, f) in enumerate(fields.items()):
        ann = f.annotation
        args = typing.get_args(ann)
        optional = type(None) in args
        base = next((a for a in args if a is not type(None)), ann) if args else ann
        key = f"bt_p_{name}_{k}"
        with cols[i % len(cols)]:
            if typing.get_origin(ann) is typing.Literal:
                opts = list(typing.get_args(ann))
                values[k] = st.selectbox(k, opts, index=opts.index(f.default), key=key)
            elif base is bool:
                values[k] = st.checkbox(k, value=f.default, key=key)
            elif base in (int, float):
                step = 1 if base is int else 0.1
                values[k] = st.number_input(
                    k,
                    value=None if f.default is None else base(f.default),  # 30 -> 30.0
                    step=step,
                    key=key,
                    placeholder="config default" if optional else None,
                    format="%d" if base is int else "%.2f",
                )
            else:
                values[k] = st.text_input(k, value=str(f.default), key=key)
    return values


# ------------------------------------------------------------------ single run
if mode == "Single run":
    name, symbol, tf = pick_target("bt")
    if d.candles(db, symbol, tf).empty:
        st.info(
            f"Run `python scripts/backfill.py --symbols {symbol} --timeframes {tf}` first.",
            title=f"No {tf} data for {symbol}",
        )
        st.stop()
    with st.form("bt_params", border=True):
        st.markdown("**Parameters**")
        params = param_inputs(name)
        submitted = st.form_submit_button(
            "Run backtest", type="primary", icon=":material/play_arrow:"
        )

    with st.spinner("Backtesting…"):
        res = d.backtest(db, name, json.dumps(params, default=str), symbol, tf)
    m, mc = res["metrics"], res["mc"]
    cm = cost_model_for(symbol, tf, c.costs)
    st.caption(
        f"{res['label']} · {symbol} {tf} · {m.get('years', 0):.1f} years · costs {cm.name} "
        f"(~{cm.round_trip_pct():.2%} per round trip) · "
        f"risk {c.backtest.risk_per_trade_pct}% per trade · "
        f"shorts {'on' if res['meta']['allow_short'] else 'off'} · fills at the next bar's open"
    )
    if m.get("trades", 0) < c.backtest.min_trades:
        st.warning(
            f"Only {int(m.get('trades', 0))} trades: too few to tell skill from luck.",
            icon=":material/warning:",
        )

    with st.container(horizontal=True):
        st.metric(
            "Total return",
            pct(m.get("total_return")),
            delta=f"{(m.get('excess_vs_buy_hold') or 0) * 100:+.1f}% vs buy & hold",
            delta_arrow="off",
            delta_color="normal",
            border=True,
        )
        st.metric("CAGR", pct(m.get("cagr")), border=True)
        st.metric("Sharpe", num(m.get("sharpe")), border=True)
        st.metric("Max drawdown", pct(m.get("max_drawdown")), border=True)
        st.metric("Win rate", pct(m.get("win_rate"), 0).lstrip("+"), border=True)
        st.metric("Expectancy", f"{num(m.get('expectancy_r'))} R", border=True)
        st.metric("Trades", int(m.get("trades", 0)), border=True)
        st.metric("Profit factor", num(m.get("profit_factor")), border=True)

    st.plotly_chart(
        equity_chart(res["equity"], res["close"], title="Equity vs buy & hold", height=520),
        alt="Strategy equity curve against buy and hold, with drawdown below",
    )

    t_trades, t_regime, t_mc, t_heat, t_all = st.tabs(
        ["Trades", "By regime", "Monte Carlo", "Parameter heatmap", "All metrics"]
    )
    with t_trades:
        tr = res["trades"]
        if tr.empty:
            st.caption("No trades.")
        else:
            view = tr[
                [
                    "direction",
                    "entry_ts",
                    "entry_price",
                    "stop",
                    "exit_ts",
                    "exit_price",
                    "exit_reason",
                    "bars_held",
                    "fees",
                    "net_pnl",
                    "r_multiple",
                    "regime_trend",
                    "regime_market",
                ]
            ].copy()
            for col in ("entry_ts", "exit_ts"):
                view[col] = d.to_ist(view[col], "%Y-%m-%d %H:%M")
            st.dataframe(
                view.iloc[::-1],
                hide_index=True,
                column_config={
                    "entry_price": st.column_config.NumberColumn(format="%.4g"),
                    "stop": st.column_config.NumberColumn(format="%.4g"),
                    "exit_price": st.column_config.NumberColumn(format="%.4g"),
                    "fees": st.column_config.NumberColumn(format="%.2f"),
                    "net_pnl": st.column_config.NumberColumn(format="%.2f"),
                    "r_multiple": st.column_config.NumberColumn("R", format="%+.2f"),
                },
                alt="Every trade, newest first",
            )
            st.download_button(
                "Download trades CSV",
                tr.to_csv(index=False),
                f"trades_{name}_{symbol}_{tf}.csv".replace("/", "-"),
                mime="text/csv",
                icon=":material/download:",
            )
    with t_regime:
        st.caption("Which market conditions the strategy works in (regime at the signal bar).")
        rb = res["regimes"]
        if rb.empty:
            st.caption("No trades.")
        else:
            st.dataframe(
                rb,
                hide_index=True,
                column_config={
                    "win_rate": st.column_config.NumberColumn(format="percent"),
                    "avg_r": st.column_config.NumberColumn("avg R", format="%+.2f"),
                    "net_pnl": st.column_config.NumberColumn(format="%.0f"),
                },
                alt="Trades grouped by trend, volatility and market regime",
            )
    with t_mc:
        if len(mc) <= 1:
            st.caption("Needs at least 2 trades.")
        else:
            st.caption(
                f"{int(mc['runs'])} reshuffles / resamples of the {int(mc['trades'])} trades. "
                "Plan for the bad case, not the actual one."
            )
            with st.container(horizontal=True):
                st.metric("Max DD, actual", pct(mc["actual_max_dd"]), border=True)
                st.metric("Max DD, bad 5%", pct(mc["dd_p5"]), border=True)
                st.metric("Max DD, median", pct(mc["dd_p50"]), border=True)
                st.metric("Return, 5th pct", pct(mc["return_p5"]), border=True)
                st.metric("Return, median", pct(mc["return_p50"]), border=True)
                st.metric("Chance of a loss", f"{mc['prob_loss']:.0%}", border=True)
    with t_heat:
        cls = STRATEGIES[name]
        grid = {
            k: v for k, v in cls.param_grid.items() if all(isinstance(x, int | float) for x in v)
        }
        if len(grid) < 2:
            st.caption("This strategy has fewer than two numeric parameters to sweep.")
        else:
            st.caption(
                "A good parameter sits on a broad plateau. A lone bright cell is usually "
                "overfitting; check it with walk-forward."
            )
            with st.container(horizontal=True, vertical_alignment="bottom"):
                keys = list(grid)
                px = st.selectbox("X parameter", keys, index=0, key=f"hm_x_{name}")
                py = st.selectbox(
                    "Y parameter", [k for k in keys if k != px], index=0, key=f"hm_y_{name}"
                )
                metric = st.selectbox(
                    "Metric",
                    ["sharpe", "total_return", "expectancy_r", "max_drawdown", "trades"],
                    key="hm_metric",
                )
                go_heat = st.button("Run grid", icon=":material/grid_on:", key="hm_run")
            if go_heat:
                xs, ys = grid[px], grid[py]
                z = np.full((len(ys), len(xs)), np.nan)
                bar = st.progress(0.0, text="Running…")
                for iy, vy in enumerate(ys):
                    for ix, vx in enumerate(xs):
                        p = {**params, px: vx, py: vy}
                        r = d.backtest(db, name, json.dumps(p, default=str), symbol, tf)
                        z[iy, ix] = r["metrics"].get(metric, np.nan)
                        bar.progress((iy * len(xs) + ix + 1) / (len(xs) * len(ys)))
                bar.empty()
                fig = go.Figure(
                    go.Heatmap(
                        z=z,
                        x=[str(v) for v in xs],
                        y=[str(v) for v in ys],
                        colorscale="RdYlGn",
                        text=np.round(z, 2),
                        texttemplate="%{text}",
                        colorbar={"title": metric},
                    )
                )
                fig.update_layout(
                    template="plotly_dark",
                    height=420,
                    xaxis_title=px,
                    yaxis_title=py,
                    title=f"{metric} by {px} × {py} (other parameters as set above)",
                )
                st.plotly_chart(fig, alt=f"Heatmap of {metric} across {px} and {py}")
    with t_all:
        st.dataframe(
            pd.DataFrame({"metric": list(m), "value": [m[k] for k in m]}),
            hide_index=True,
            alt="All backtest metrics",
        )

# ------------------------------------------------------------------ walk-forward
elif mode == "Walk-forward":
    st.caption(
        f"Splits history into {c.backtest.walkforward_windows} windows. In each, the best "
        f"parameters by {c.backtest.walkforward_objective} on the past "
        "(in-sample) are tested on the next, unseen bars (out-of-sample)."
    )
    name, symbol, tf = pick_target("wf")
    if st.button("Run walk-forward", type="primary", icon=":material/play_arrow:", key="wf_run"):
        n = len(STRATEGIES[name].grid()) * c.backtest.walkforward_windows
        with st.spinner(f"Running {n} in-sample backtests…"):
            st.session_state.wf_result = (name, symbol, tf, d.walkforward(db, name, symbol, tf))
    got = st.session_state.get("wf_result")
    if got and got[:3] == (name, symbol, tf):
        wf = got[3]
        m = wf["metrics"]
        full = d.backtest(db, name, "{}", symbol, tf)["metrics"]
        with st.container(horizontal=True):
            st.metric("OOS return", pct(m.get("total_return")), border=True)
            st.metric("OOS Sharpe", num(m.get("sharpe")), border=True)
            st.metric(
                "OOS expectancy",
                f"{num(m.get('expectancy_r'))} R",
                delta=f"full-history default: {num(full.get('expectancy_r'))} R",
                delta_arrow="off",
                delta_color="off",
                border=True,
            )
            st.metric("OOS trades", int(m.get("trades", 0)), border=True)
            st.metric("OOS max drawdown", pct(m.get("max_drawdown")), border=True)
        w = wf["windows"].copy()
        for col in ("train_from", "test_from", "test_to"):
            w[col] = d.to_ist(w[col], "%Y-%m-%d")
        w["params"] = w["params"].map(lambda p: ", ".join(f"{k}={v}" for k, v in p.items()))
        st.dataframe(w, hide_index=True, alt="Walk-forward windows with chosen parameters")
        st.plotly_chart(
            equity_chart(wf["equity"], wf["close"], title="Out-of-sample equity", height=480),
            alt="Equity curve made only of out-of-sample periods",
        )
        st.caption(
            "If in-sample scores are much better than out-of-sample, or the chosen "
            "parameters jump around between windows, the edge is mostly curve-fitting."
        )

# ------------------------------------------------------------------ leaderboard
else:
    lb, batch = d.latest_leaderboard(db)
    with st.container(horizontal=True, vertical_alignment="bottom"):
        st.caption(
            "Every strategy × symbol × timeframe with default parameters and full costs. "
            + (f"Latest batch: {batch}." if batch else "No leaderboard yet.")
        )
        if st.button("Run leaderboard (~1 min)", icon=":material/leaderboard:", key="lb_run"):
            with st.status("Running every strategy on every symbol…", expanded=True) as status:
                line = st.empty()
                res = run_leaderboard(c, d.store(), progress=lambda s: line.write(s))
                status.update(label=f"Done: {len(res.table)} runs", state="complete")
            d.latest_leaderboard.clear()
            st.rerun()
    if lb.empty:
        st.stop()

    with st.container(horizontal=True, vertical_alignment="bottom"):
        strategies = st.pills(
            "Strategies",
            sorted(lb["strategy"].unique()),
            selection_mode="multi",
            default=sorted(lb["strategy"].unique()),
            key="lb_strats",
        )
        tfs = st.pills(
            "Timeframes",
            sorted(lb["timeframe"].unique()),
            selection_mode="multi",
            default=sorted(lb["timeframe"].unique()),
            key="lb_tfs",
        )
        only_ranked = st.toggle(
            f"Only ≥ {c.backtest.min_trades} trades", value=True, key="lb_ranked"
        )
    view = lb[lb["strategy"].isin(strategies or []) & lb["timeframe"].isin(tfs or [])]
    if only_ranked:
        view = view[view["trades"] >= c.backtest.min_trades]
    view = view.sort_values("sharpe", ascending=False, na_position="last")
    cols = [
        "strategy",
        "symbol",
        "timeframe",
        "trades",
        "years",
        "total_return",
        "sharpe",
        "max_drawdown",
        "win_rate",
        "expectancy_r",
        "profit_factor",
        "buy_hold_return",
        "mc_dd_p5",
    ]
    st.dataframe(
        view[[col for col in cols if col in view]],
        hide_index=True,
        column_config={
            "total_return": st.column_config.NumberColumn(format="percent"),
            "max_drawdown": st.column_config.NumberColumn(format="percent"),
            "win_rate": st.column_config.NumberColumn(format="percent"),
            "buy_hold_return": st.column_config.NumberColumn("buy & hold", format="percent"),
            "mc_dd_p5": st.column_config.NumberColumn("MC bad-5% DD", format="percent"),
            "sharpe": st.column_config.NumberColumn(format="%.2f"),
            "expectancy_r": st.column_config.NumberColumn("exp. R", format="%+.2f"),
            "profit_factor": st.column_config.NumberColumn(format="%.2f"),
            "years": st.column_config.NumberColumn(format="%.1f"),
        },
        alt="Strategy leaderboard ranked by Sharpe ratio",
    )
    st.download_button(
        "Download CSV",
        view.to_csv(index=False),
        f"leaderboard_{batch}.csv",
        mime="text/csv",
        icon=":material/download:",
    )
    st.caption(
        "Ranked by Sharpe with default parameters. A top row is a candidate to "
        "walk-forward test and paper trade, not proof of an edge."
    )
