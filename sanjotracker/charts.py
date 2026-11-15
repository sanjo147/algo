"""Plotly charts: candlesticks with S/R zones, pivots and indicators (dark theme).

Used by scripts/plot_levels.py now and by the Streamlit dashboard in Phase 6.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sanjotracker.analysis.indicators import ema

GREEN = "38, 166, 154"
RED = "239, 83, 80"
GREY = "160, 160, 160"


def _nse_rangebreaks(x: pd.Series, intraday: bool) -> list[dict]:
    """Hide weekends, NSE holidays and (intraday) the hours the market is closed."""
    days = pd.DatetimeIndex(x.dt.date.unique())
    all_bdays = pd.bdate_range(days.min(), days.max())
    missing = [d.strftime("%Y-%m-%d") for d in all_bdays.difference(days)]
    breaks: list[dict] = [{"bounds": ["sat", "mon"]}]
    if missing:
        breaks.append({"values": missing})
    if intraday:
        breaks.append({"bounds": [15.5, 9.25], "pattern": "hour"})
    return breaks


def levels_chart(
    df: pd.DataFrame,
    levels: pd.DataFrame,
    *,
    title: str,
    pivots: pd.DataFrame | None = None,
    display_tz: str = "Asia/Kolkata",
    is_nse: bool = True,
    intraday: bool = False,
    max_zones: int = 8,
) -> go.Figure:
    """Candlestick (+ volume, if any) with pivot zones shaded and reference levels dotted.

    Only zones/levels that overlap the plotted price range are drawn.
    """
    d = df.copy()
    d["x"] = pd.to_datetime(d["timestamp"], utc=True).dt.tz_convert(display_tz).dt.tz_localize(None)
    has_volume = bool(d["volume"].sum() > 0)  # indices like ^NSEI report no volume
    fig = make_subplots(
        rows=2 if has_volume else 1,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.8, 0.2] if has_volume else [1.0],
        vertical_spacing=0.02,
    )
    fig.add_trace(
        go.Candlestick(
            x=d["x"],
            open=d["open"],
            high=d["high"],
            low=d["low"],
            close=d["close"],
            name="price",
            increasing_line_color=f"rgb({GREEN})",
            decreasing_line_color=f"rgb({RED})",
        ),
        row=1,
        col=1,
    )
    for n, color in ((20, "#f5c542"), (50, "#42a5f5")):
        fig.add_trace(
            go.Scatter(
                x=d["x"],
                y=ema(df["close"], n).to_numpy(),
                name=f"EMA {n}",
                line={"width": 1, "color": color},
            ),
            row=1,
            col=1,
        )
    if has_volume:
        vol_colors = [
            f"rgba({GREEN},0.5)" if c >= o else f"rgba({RED},0.5)"
            for o, c in zip(d["open"], d["close"], strict=True)
        ]
        fig.add_trace(
            go.Bar(
                x=d["x"], y=d["volume"], marker_color=vol_colors, name="volume", showlegend=False
            ),
            row=2,
            col=1,
        )

    lo, hi = float(d["low"].min()), float(d["high"].max())
    pad = (hi - lo) * 0.05
    visible = levels[(levels["zone_high"] >= lo - pad) & (levels["zone_low"] <= hi + pad)]

    zones = visible[visible["source"] == "pivot"].head(max_zones)
    for z in zones.itertuples():
        rgb = GREEN if z.type == "S" else RED
        fig.add_hrect(
            y0=z.zone_low,
            y1=z.zone_high,
            row=1,
            col=1,
            line_width=0,
            fillcolor=f"rgba({rgb},{0.08 + 0.32 * z.strength / 100:.2f})",
            annotation_text=f"{z.type} {z.touches}x · {z.strength:.0f}",
            annotation_position="right",
            annotation_font_size=10,
            annotation_font_color=f"rgb({rgb})",
        )
    for r in visible[visible["source"] != "pivot"].itertuples():
        mid = (r.zone_low + r.zone_high) / 2
        fig.add_hline(
            y=mid,
            row=1,
            col=1,
            line_dash="dot",
            line_width=1,
            line_color=f"rgba({GREY},0.7)",
            annotation_text=r.source.replace("_", " "),
            annotation_position="top left",
            annotation_font_size=9,
            annotation_font_color=f"rgb({GREY})",
        )

    if pivots is not None and not pivots.empty:
        p = pivots[pivots["pivot_ts"] >= df["timestamp"].iloc[0]].copy()
        p["x"] = p["pivot_ts"].dt.tz_convert(display_tz).dt.tz_localize(None)
        for kind, symbol, color, shift in (
            ("high", "triangle-down", RED, 1.003),
            ("low", "triangle-up", GREEN, 0.997),
        ):
            k = p[p["kind"] == kind]
            fig.add_trace(
                go.Scatter(
                    x=k["x"],
                    y=k["price"] * shift,
                    mode="markers",
                    name=f"pivot {kind}",
                    marker={"symbol": symbol, "size": 7, "color": f"rgb({color})"},
                ),
                row=1,
                col=1,
            )

    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=800,
        xaxis_rangeslider_visible=False,
        legend={"orientation": "h", "y": 1.02, "x": 0},
        margin={"l": 60, "r": 110, "t": 70, "b": 30},
    )
    if is_nse:
        fig.update_xaxes(rangebreaks=_nse_rangebreaks(d["x"], intraday))
    return fig
