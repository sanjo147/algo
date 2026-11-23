"""Paper trading: virtual account, open positions, closed trades, per-setup stats."""

import pandas as pd
import streamlit as st

from sanjotracker import dashboard as d
from sanjotracker.charts import trade_figure
from sanjotracker.data.updater import market_of

c = d.cfg()
db = d.db_key()
p = c.paper

with st.container(horizontal=True, vertical_alignment="center"):
    risk = (f"{p.stop_atr:g}×ATR stop" if p.stop_atr is not None else "strategy stop") + (
        f", {p.target_r:g}R target" if p.target_r is not None else ", strategy exit"
    )
    st.caption(
        f"Virtual trades only, never real orders. {len(p.setups)} setups · "
        f"{p.risk_per_trade_pct}% risk of {p.capital:,.0f} per trade · {risk} · "
        "fills at the next bar's open with full costs. "
        + ("" if p.enabled else ":orange[Paper trading is switched off in Settings.]")
    )
    if st.button(
        "Step now",
        icon=":material/play_arrow:",
        key="pp_step",
        help="Fetch new candles, advance every setup and send Telegram notices. "
        "The scheduler does this automatically after each scan.",
    ):
        with st.spinner("Fetching candles and stepping paper trades…"):
            r = d.paper_step()
        st.toast(f"{r['setups']} setups · {r['opened']} opened · {r['closed']} closed")
        for k, v in r["errors"].items():
            st.warning(f"{k}: {v}", icon=":material/warning:")

acct = d.paper_account(db, p.capital)
open_, closed = acct["open"], acct["closed"]

with st.container(horizontal=True):
    st.metric(
        "Equity",
        f"{acct['equity']:,.0f}",
        delta=f"{(acct['equity'] / acct['capital'] - 1) * 100:+.2f}%",
        border=True,
    )
    st.metric("Realized P&L", f"{acct['realized']:+,.0f}", border=True)
    st.metric("Open P&L", f"{acct['unrealized']:+,.0f}", border=True)
    st.metric("Open positions", len(open_), border=True)
    st.metric("Closed trades", len(closed), border=True)
    st.metric(
        "Win rate",
        "–" if pd.isna(acct["win_rate"]) else f"{acct['win_rate']:.0%}",
        border=True,
    )
    st.metric("Avg R", "–" if pd.isna(acct["avg_r"]) else f"{acct['avg_r']:+.2f}", border=True)

if open_.empty and closed.empty:
    st.info(
        "Setups only trade bars that close after they were first registered, so the "
        "first trades appear once the scheduler (or **Step now**) has run on new candles.",
        title="No paper trades yet",
        icon=":material/hourglass_empty:",
    )

st.subheader("Open positions")
if open_.empty:
    st.caption("None.")
else:
    view = open_.assign(entry=d.to_ist(open_["entry_ts"], "%d %b %H:%M"))
    st.dataframe(
        view[
            [
                "id",
                "symbol",
                "timeframe",
                "label",
                "direction",
                "entry",
                "entry_price",
                "stop",
                "target",
                "last",
                "qty",
                "unrealized_pnl",
                "unrealized_r",
            ]
        ],
        hide_index=True,
        column_config={
            "entry_price": st.column_config.NumberColumn("entry @", format="%.4g"),
            "stop": st.column_config.NumberColumn(format="%.4g"),
            "target": st.column_config.NumberColumn(format="%.4g"),
            "last": st.column_config.NumberColumn(format="%.4g"),
            "unrealized_pnl": st.column_config.NumberColumn("open P&L", format="%+.0f"),
            "unrealized_r": st.column_config.NumberColumn("open R", format="%+.2f"),
        },
        alt="Open paper positions with unrealized profit at the last close",
    )

if not closed.empty:
    st.subheader("Realized P&L over time")
    eq = closed.sort_values("exit_ts")
    curve = pd.DataFrame(
        {
            "date": eq["exit_ts"].dt.tz_convert(d.IST).dt.tz_localize(None),
            "cumulative P&L": eq["net_pnl"].cumsum(),
        }
    )
    st.area_chart(
        curve, x="date", y="cumulative P&L", alt="Cumulative realized paper profit over time"
    )

    st.subheader("By setup")
    by = (
        closed.groupby(["label", "timeframe"])
        .agg(
            trades=("id", "count"),
            win_rate=("net_pnl", lambda s: (s > 0).mean()),
            avg_r=("r_multiple", "mean"),
            net_pnl=("net_pnl", "sum"),
        )
        .reset_index()
        .sort_values("net_pnl", ascending=False)
    )
    st.dataframe(
        by,
        hide_index=True,
        column_config={
            "win_rate": st.column_config.NumberColumn(format="percent"),
            "avg_r": st.column_config.NumberColumn("avg R", format="%+.2f"),
            "net_pnl": st.column_config.NumberColumn(format="%+.0f"),
        },
        alt="Closed paper trades grouped by strategy setup",
    )

    st.subheader("Closed trades")
    cv = closed.assign(
        entry=d.to_ist(closed["entry_ts"], "%d %b %H:%M"),
        exit=d.to_ist(closed["exit_ts"], "%d %b %H:%M"),
    )
    st.dataframe(
        cv[
            [
                "id",
                "symbol",
                "timeframe",
                "label",
                "direction",
                "entry",
                "entry_price",
                "exit",
                "exit_price",
                "exit_reason",
                "fees",
                "net_pnl",
                "r_multiple",
                "tags",
            ]
        ],
        hide_index=True,
        column_config={
            "entry_price": st.column_config.NumberColumn("entry @", format="%.4g"),
            "exit_price": st.column_config.NumberColumn("exit @", format="%.4g"),
            "fees": st.column_config.NumberColumn(format="%.2f"),
            "net_pnl": st.column_config.NumberColumn(format="%+.0f"),
            "r_multiple": st.column_config.NumberColumn("R", format="%+.2f"),
        },
        alt="Closed paper trades, newest first",
    )

all_trades = pd.concat([open_, closed], ignore_index=True)
if not all_trades.empty:
    st.subheader("Look at a trade")
    tid = st.selectbox(
        "Trade",
        all_trades["id"].tolist(),
        format_func=lambda i: (
            lambda r: (
                f"#{i} {r['symbol']} {r['timeframe']} {r['direction']} "
                f"({r['status']}) · {r['label']}"
            )
        )(all_trades.set_index("id").loc[i]),
        key="pp_trade",
    )
    row = all_trades.set_index("id").loc[tid].to_dict()
    row["id"] = tid
    candles = d.candles(db, row["symbol"], row["timeframe"])
    if not candles.empty:
        st.plotly_chart(
            trade_figure(candles, row, is_nse=market_of(row["symbol"]) == "nse"),
            alt=f"Chart of paper trade {tid} with entry, exit, stop and target",
        )
    st.page_link(
        "app_pages/journal.py", label="Add notes in the journal", icon=":material/edit_note:"
    )

with st.expander("Registered setups"):
    s = d.paper_book().setups()
    if s.empty:
        st.caption("None yet.")
    else:
        s["start_ts"] = d.to_ist(s["start_ts"], "%d %b %Y %H:%M")
        s["last_run_ts"] = d.to_ist(s["last_run_ts"], "%d %b %H:%M")
        st.dataframe(
            s[["label", "symbol", "timeframe", "start_ts", "last_run_ts"]],
            hide_index=True,
            column_config={"start_ts": "trades from", "last_run_ts": "last step"},
            alt="Paper setups and when each started trading",
        )
    st.caption("Add or change setups under `paper:` in config.yaml (see config.example.yaml).")
