"""Scanner: events on the latest closed bars and symbols sitting near strong levels."""

import pandas as pd
import streamlit as st

from sanjotracker import dashboard as d

c = d.cfg()
db = d.db_key()

with st.container(horizontal=True, vertical_alignment="bottom"):
    style = st.segmented_control(
        "Style",
        ["swing", "intraday"],
        default="swing",
        required=True,
        key="sc_style",
        help="Swing: 1d bars with the 1w trend. Intraday: 15m bars with the 1h trend.",
    )
    market = st.segmented_control(
        "Market", ["all", "nse", "crypto"], default="all", required=True, key="sc_market"
    )
    lookback = st.slider("Bars back", 1, 20, 3, key="sc_lookback")
    min_conf = st.slider("Min confidence", 0, 100, 0, step=5, key="sc_conf")
    counter = st.toggle("Counter-trend breakouts", key="sc_counter")
    fetch = st.button("Fetch latest + scan", icon=":material/refresh:", key="sc_fetch")

st_cfg = getattr(c.styles, style)
if fetch:
    syms = (
        [*c.watchlist.nse, *c.watchlist.crypto] if market == "all" else getattr(c.watchlist, market)
    )
    with st.spinner(f"Downloading {st_cfg.primary} and {st_cfg.trend} candles…"):
        d.fetch_latest(syms, [st_cfg.primary, st_cfg.trend])

with st.spinner("Scanning the watchlist…"):
    events, errors, regimes = d.scan(db, style, market, lookback, counter)

st.subheader("Events")
st.caption(
    f"{st_cfg.primary} bars, {st_cfg.trend} trend filter, last {lookback} closed bar(s). "
    "Each bar is judged only with data available at its close."
)
if not events.empty:
    events = events[events["confidence"] >= min_conf]
if events.empty:
    st.info("Nothing on these bars. Try more bars back or a lower confidence.", title="No events")
else:
    view = pd.DataFrame(
        {
            "bar": d.to_ist(
                events["bar_ts"], "%d %b" if st_cfg.primary in ("1d", "1w") else "%d %b %H:%M"
            ),
            "symbol": events["symbol"],
            "event": events["event"].str.replace("_", " "),
            "dir": events["direction"],
            "price": events["price"],
            "zone": [
                f"{d.fmt_price(lo)}–{d.fmt_price(hi)}"
                for lo, hi in zip(events["zone_low"], events["zone_high"], strict=True)
            ],
            "vol ×": events["volume_ratio"],
            "htf trend": events["trend_htf"],
            "confidence": events["confidence"],
            "why": events["reasons"],
        }
    )
    st.dataframe(
        view,
        hide_index=True,
        column_config={
            "price": st.column_config.NumberColumn(format="%.4g"),
            "vol ×": st.column_config.NumberColumn(format="%.1f"),
            "confidence": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d"),
            "why": st.column_config.TextColumn(width="large"),
        },
        alt="Breakout, retest and approaching events from the latest scan",
    )

st.subheader("Near strong levels")
st.caption("Daily data. Distance to the nearest pivot zone above and below, in % of price.")
wl = d.watchlist_table(db, tuple(d.all_symbols(c)))
if not wl.empty and "to_support_pct" in wl:
    wl = wl[wl["status"] == "ok"].copy()
    if market != "all":
        wl = wl[wl["market"] == market]
    wl["nearest_pct"] = wl[["to_support_pct", "to_resistance_pct"]].abs().min(axis=1)
    within = st.slider("Within %", 0.5, 10.0, 3.0, step=0.5, key="sc_within")
    near = wl[wl["nearest_pct"] <= within].sort_values("nearest_pct")
    st.dataframe(
        near[
            [
                "symbol",
                "last",
                "support",
                "to_support_pct",
                "resistance",
                "to_resistance_pct",
                "trend",
                "volatility",
            ]
        ],
        hide_index=True,
        column_config={
            "last": st.column_config.NumberColumn(format="%.4g"),
            "support": st.column_config.NumberColumn(format="%.4g"),
            "resistance": st.column_config.NumberColumn(format="%.4g"),
            "to_support_pct": st.column_config.NumberColumn("to S %", format="%+.1f%%"),
            "to_resistance_pct": st.column_config.NumberColumn("to R %", format="%+.1f%%"),
        },
        alt="Symbols within the chosen distance of a support or resistance zone",
    )

with st.expander("Regime per symbol and scan problems"):
    if regimes:
        st.dataframe(
            pd.DataFrame(regimes).T.drop(columns=["atr_pct_rank"], errors="ignore"),
            alt="Trend, volatility and market regime per symbol",
        )
    for k, v in errors.items():
        st.warning(f"{k}: {v}", icon=":material/warning:")
