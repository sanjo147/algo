"""Charts: candlesticks + S/R zones + pivots + breakout markers + indicators."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sanjotracker import dashboard as d
from sanjotracker.analysis.indicators import bollinger, ema, vwap
from sanjotracker.analysis.levels import nearest_levels
from sanjotracker.charts import levels_chart
from sanjotracker.data.updater import market_of, session_tz_for

c = d.cfg()
db = d.db_key()
symbols = d.all_symbols(c)

with st.container(horizontal=True, vertical_alignment="bottom"):
    symbol = st.selectbox("Symbol", symbols, key="ch_symbol")
    timeframe = st.segmented_control(
        "Timeframe", ["5m", "15m", "1h", "1d", "1w"], default="1d", required=True, key="ch_tf"
    )
    bars = st.select_slider("Bars", [100, 200, 300, 500, 800], value=300, key="ch_bars")
    overlays = st.pills(
        "Overlays",
        ["Pivots", "Events", "Alerts", "Reference levels", "EMA 200", "Bollinger", "VWAP"],
        selection_mode="multi",
        default=["Pivots", "Events", "Alerts"],
        key="ch_overlays",
    )
    if st.button("Fetch latest", icon=":material/refresh:", key="ch_fetch"):
        with st.spinner("Downloading new candles…"):
            res = d.fetch_latest([symbol], [timeframe])
        err = res["error"].dropna()
        if len(err):
            st.toast(f"Update failed: {err.iloc[0]}", icon=":material/error:")
        else:
            st.toast(f"+{int(res['new_rows'].iloc[0])} candles", icon=":material/done:")

overlays = overlays or []
df = d.candles(db, symbol, timeframe, c.breakout.scan_history_bars)
if df.empty:
    st.info(
        f"Run `python scripts/backfill.py --symbols {symbol} --timeframes {timeframe}` "
        "or press **Fetch latest**.",
        title=f"No {timeframe} candles for {symbol}",
        icon=":material/database:",
    )
    st.stop()

levels, pivots = d.levels_and_pivots(db, symbol, timeframe)
shown = df.tail(bars).reset_index(drop=True)
is_nse = market_of(symbol) == "nse"
intraday = timeframe in d.INTRADAY

markers = None
parts = []
if "Events" in overlays:
    with st.spinner("Finding breakout events (each bar judged with only earlier data)…"):
        ev = d.recent_events(db, symbol, timeframe, min(bars, 150))
    if not ev.empty:
        ev = ev[ev["event"] != "approaching"]
        parts.append(ev[["bar_ts", "price", "event", "direction"]])
if "Alerts" in overlays:
    al = d.alerts_since(db, 24 * 365, 2000)
    al = al[(al["symbol"] == symbol) & (al["timeframe"] == timeframe)]
    if not al.empty:
        parts.append(al[["bar_ts", "price", "event", "direction"]])
if parts:
    m = pd.concat(parts).drop_duplicates(["bar_ts", "event"])
    markers = pd.DataFrame(
        {
            "timestamp": m["bar_ts"],
            "price": m["price"],
            "label": m["event"].str.replace("_", " "),
            "direction": m["direction"],
        }
    )
    markers = markers[markers["timestamp"] >= shown["timestamp"].iloc[0]]

fig = levels_chart(
    shown,
    levels,
    title=f"{symbol} · {timeframe}",
    pivots=pivots if "Pivots" in overlays else None,
    display_tz=c.app.timezone_display,
    is_nse=is_nse,
    intraday=intraday,
    markers=markers,
    show_reference="Reference levels" in overlays,
)

x = pd.to_datetime(shown["timestamp"], utc=True).dt.tz_convert(d.IST).dt.tz_localize(None)
full_close = df["close"]
tail = slice(len(df) - len(shown), len(df))


def add_line(y: pd.Series, name: str, color: str, dash: str | None = None) -> None:
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y.iloc[tail].to_numpy(),
            name=name,
            line={"color": color, "width": 1, "dash": dash},
        ),
        row=1,
        col=1,
    )


if "EMA 200" in overlays:
    add_line(ema(full_close, 200), "EMA 200", "#e57373")
if "Bollinger" in overlays:
    bb = bollinger(full_close)
    add_line(bb["bb_upper"], "BB upper", "#90a4ae", "dot")
    add_line(bb["bb_lower"], "BB lower", "#90a4ae", "dot")
if "VWAP" in overlays and intraday:
    add_line(vwap(df, session_tz_for(symbol)), "VWAP", "#ba68c8")

st.plotly_chart(
    fig, alt=f"{symbol} {timeframe} candlestick chart with support and resistance zones"
)

price = float(df["close"].iloc[-1])
r = d.regime_of(db, symbol, timeframe)
left, right = st.columns([1, 2])
with left, st.container(border=True):
    st.markdown("**Now**")
    st.metric("Last close", d.fmt_price(price))
    d.regime_badges(timeframe, r)
    st.caption(
        f"Last bar {d.to_ist(df['timestamp'].tail(1), '%d %b %Y %H:%M').iloc[0]} IST · "
        f"{len(levels)} levels from {len(pivots)} confirmed pivots"
    )
with right, st.container(border=True):
    st.markdown("**Nearest levels**")
    near = nearest_levels(levels, price, 4) if not levels.empty else levels
    if near.empty:
        st.caption("No levels yet (needs more history).")
    else:
        near = near.sort_values("zone_low", ascending=False)
        near["distance_pct"] = ((near["zone_low"] + near["zone_high"]) / 2 / price - 1) * 100
        st.dataframe(
            near[
                ["type", "zone_low", "zone_high", "touches", "strength", "source", "distance_pct"]
            ],
            hide_index=True,
            column_config={
                "zone_low": st.column_config.NumberColumn(format="%.4g"),
                "zone_high": st.column_config.NumberColumn(format="%.4g"),
                "strength": st.column_config.ProgressColumn(
                    min_value=0, max_value=100, format="%d"
                ),
                "distance_pct": st.column_config.NumberColumn("distance", format="%+.2f%%"),
            },
            alt="Nearest support and resistance levels above and below the price",
        )
