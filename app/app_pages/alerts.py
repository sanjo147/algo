"""Home: live alert feed, market regime badges, watchlist table."""

from datetime import UTC, datetime

import pandas as pd
import streamlit as st

from sanjotracker import dashboard as d
from sanjotracker.charts import event_figure

c = d.cfg()
db = d.db_key()

# ------------------------------------------------------------------ regime header
with st.container(border=True):
    d.regime_badges("NIFTY 50", d.regime_of(db, c.benchmarks.nse, "1d"))


# ------------------------------------------------------------------ live feed (fragment)
@st.fragment(run_every="60s")
def alert_feed() -> None:
    d.alerts_since.clear()  # always read fresh rows on the timer
    day = d.alerts_since(db, 24)
    week = d.alerts_since(db, 24 * 7, 2000)
    with st.container(horizontal=True):
        st.metric("Alerts, last 24 h", len(day), border=True)
        st.metric(
            "Sent to Telegram (24 h)",
            int(day["telegram_sent"].sum()) if len(day) else 0,
            border=True,
        )
        st.metric("Alerts, last 7 days", len(week), border=True)
        last = (
            pd.Timestamp(week["created_at"].iloc[0]).tz_convert(d.IST).strftime("%d %b %H:%M")
            if len(week)
            else "–"
        )
        st.metric("Latest alert", last, border=True)

    with st.container(horizontal=True, vertical_alignment="bottom"):
        hours = st.segmented_control(
            "Period",
            [24, 72, 168],
            default=72,
            format_func=lambda h: f"{h // 24}d",
            key="feed_hours",
            required=True,
        )
        kinds = st.pills(
            "Events",
            list(d.EVENT_ICON),
            selection_mode="multi",
            default=list(d.EVENT_ICON),
            format_func=lambda k: k.replace("_", " "),
            key="feed_kinds",
        )
        min_conf = st.slider("Min confidence", 0, 100, 0, step=5, key="feed_conf")

    df = d.alerts_since(db, int(hours or 72), 1000)
    if not df.empty:
        df = df[df["event"].isin(kinds or []) & (df["confidence"] >= min_conf)]
    if df.empty:
        st.info(
            "Alerts appear here when the scheduler finds them (`python scripts/run_scheduler.py`).",
            title="No alerts for these filters",
            icon=":material/notifications_off:",
        )
        st.caption(f"Checked {datetime.now(UTC).astimezone().strftime('%H:%M:%S')}")
        return

    view = pd.DataFrame(
        {
            "time": d.to_ist(df["created_at"]),
            "symbol": df["symbol"],
            "tf": df["timeframe"],
            "event": df["event"].str.replace("_", " "),
            "dir": df["direction"],
            "price": df["price"],
            "zone": [
                f"{d.fmt_price(lo)}–{d.fmt_price(hi)}"
                for lo, hi in zip(df["zone_low"], df["zone_high"], strict=True)
            ],
            "volume": df["volume_ratio"],
            "confidence": df["confidence"],
            "telegram": df["telegram_sent"].astype(bool),
            "note": df["note"].fillna(""),
        }
    )
    sel = st.dataframe(
        view,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="feed_table",
        column_config={
            "price": st.column_config.NumberColumn(format="%.4g"),
            "volume": st.column_config.NumberColumn("vol ×", format="%.1f"),
            "confidence": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d"),
            "telegram": st.column_config.CheckboxColumn(),
        },
        alt="Recent alerts, newest first",
    )
    st.caption(
        f"Auto-refreshes every 60 s · checked "
        f"{datetime.now(UTC).astimezone().strftime('%H:%M:%S')} · select a row for details"
    )
    rows = sel.selection.rows if sel else []
    if rows:
        ev = d.feed().get(int(df["id"].iloc[rows[0]]))
        if ev is None:
            return
        with st.container(border=True):
            st.markdown(f"**{ev.symbol} {ev.timeframe} · {ev.event.value.replace('_', ' ')}**")
            left, right = st.columns([1, 2])
            with left:
                for reason in ev.reasons:
                    st.markdown(f"- {reason}")
                st.caption(ev.regime.label())
            with right:
                candles = d.candles(db, ev.symbol, ev.timeframe, 400)
                if candles.empty:
                    st.warning("No candles stored for this symbol/timeframe.")
                else:
                    lv, _ = d.levels_and_pivots(db, ev.symbol, ev.timeframe)
                    fig = event_figure(candles, lv, ev, is_nse=True)
                    st.plotly_chart(fig, alt=f"{ev.symbol} chart around the {ev.event.value}")


st.subheader("Alert feed")
alert_feed()

# ------------------------------------------------------------------ watchlist
st.subheader("Watchlist")
with st.container(horizontal=True, vertical_alignment="center"):
    st.caption("Daily data from the local database. Zones are the nearest pivot S/R.")
    if st.button("Fetch latest daily candles", icon=":material/refresh:", key="wl_fetch"):
        with st.spinner("Downloading new candles…"):
            res = d.fetch_latest(d.all_symbols(c), ["1d"])
        bad = res[res["error"].notna()]
        st.toast(
            f"Updated {len(res) - len(bad)} symbols, {len(bad)} failed", icon=":material/done:"
        )

wl = d.watchlist_table(db, tuple(d.all_symbols(c)))
if wl.empty or (wl["status"] == "no data").all():
    st.info(
        "Run `python scripts/backfill.py` to download history first.",
        title="No data yet",
        icon=":material/database:",
    )
else:
    wl = wl[wl["status"] == "ok"].copy()
    wl["last_bar"] = d.to_ist(wl["last_bar"], "%d %b")
    st.dataframe(
        wl.drop(columns=["status"]),
        hide_index=True,
        column_order=[
            "symbol",
            "market",
            "last",
            "change_pct",
            "spark",
            "trend",
            "volatility",
            "support",
            "to_support_pct",
            "resistance",
            "to_resistance_pct",
            "last_bar",
        ],
        column_config={
            "last": st.column_config.NumberColumn(format="%.4g"),
            "change_pct": st.column_config.NumberColumn("day %", format="%+.2f%%"),
            "spark": st.column_config.LineChartColumn("30 days"),
            "support": st.column_config.NumberColumn(format="%.4g"),
            "resistance": st.column_config.NumberColumn(format="%.4g"),
            "to_support_pct": st.column_config.NumberColumn("to S %", format="%+.1f%%"),
            "to_resistance_pct": st.column_config.NumberColumn("to R %", format="%+.1f%%"),
            "last_bar": "last bar",
        },
        alt="Watchlist with price, daily change, regime and nearest support and resistance",
    )
