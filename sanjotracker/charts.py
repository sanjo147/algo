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
    markers: pd.DataFrame | None = None,
    height: int = 800,
    show_reference: bool = True,
    highlight: tuple[float, float, str] | None = None,
) -> go.Figure:
    """Candlestick (+ volume, if any) with pivot zones shaded and reference levels dotted.

    Only zones/levels that overlap the plotted price range are drawn.
    `markers` (optional): columns timestamp, price, label, direction ("up"/"down")
    for breakout/retest/... events.
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
    refs = visible[visible["source"] != "pivot"] if show_reference else visible.iloc[0:0]
    for r in refs.itertuples():
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

    if highlight is not None:
        lo_h, hi_h, label = highlight
        fig.add_hrect(
            y0=lo_h,
            y1=hi_h,
            row=1,
            col=1,
            line_width=1,
            line_color="rgba(255, 213, 79, 0.9)",
            fillcolor="rgba(255, 213, 79, 0.25)",
            annotation_text=label,
            annotation_position="left",
            annotation_font_size=11,
            annotation_font_color="#ffd54f",
        )

    if markers is not None and not markers.empty:
        mk = markers.copy()
        mk["x"] = (
            pd.to_datetime(mk["timestamp"], utc=True).dt.tz_convert(display_tz).dt.tz_localize(None)
        )
        for direction, symbol, color in (("up", "star", "#ffd54f"), ("down", "x", "#ff8a65")):
            k = mk[mk["direction"] == direction]
            if k.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=k["x"],
                    y=k["price"],
                    mode="markers+text",
                    text=k["label"],
                    textposition="top center" if direction == "up" else "bottom center",
                    textfont={"size": 11, "color": color},
                    name=f"events {direction}",
                    marker={"symbol": symbol, "size": 14, "color": color},
                ),
                row=1,
                col=1,
            )

    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=height,
        xaxis_rangeslider_visible=False,
        legend={"orientation": "h", "y": 1.0, "x": 0, "yanchor": "bottom"},
        margin={"l": 60, "r": 110, "t": 70, "b": 30},
    )
    if is_nse:
        fig.update_xaxes(rangebreaks=_nse_rangebreaks(d["x"], intraday))
    return fig


def event_figure(
    df: pd.DataFrame,
    levels: pd.DataFrame,
    event,  # noqa: ANN001  (analysis.breakout.Event; untyped to avoid an import cycle)
    *,
    bars: int = 120,
    display_tz: str = "Asia/Kolkata",
    is_nse: bool = True,
) -> go.Figure:
    """Small chart for one alert: the last `bars` candles up to the event bar."""
    upto = df[df["timestamp"] <= pd.Timestamp(event.bar_ts)].tail(bars).reset_index(drop=True)
    markers = pd.DataFrame(
        {
            "timestamp": [event.bar_ts],
            "price": [event.price],
            "label": [event.event.value.replace("_", " ")],
            "direction": [event.direction],
        }
    )
    title = f"{event.symbol} · {event.timeframe} · {event.event.value} ({event.confidence})"
    return levels_chart(
        upto,
        levels,
        title=title,
        display_tz=display_tz,
        is_nse=is_nse,
        intraday=event.timeframe in ("5m", "15m", "1h"),
        max_zones=4,
        markers=markers,
        height=500,
        show_reference=False,
        highlight=(event.zone_low, event.zone_high, f"{event.zone_source} zone"),
    )


def equity_chart(
    equity: pd.Series,
    close: pd.Series | None = None,
    *,
    title: str = "Equity",
    display_tz: str = "Asia/Kolkata",
    height: int = 600,
) -> go.Figure:
    """Equity curve vs buy & hold (scaled to the same start) with a drawdown panel."""
    x = equity.index.tz_convert(display_tz).tz_localize(None)
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.04
    )
    fig.add_trace(
        go.Scatter(x=x, y=equity, name="strategy", line={"color": "#42a5f5", "width": 2}),
        row=1,
        col=1,
    )
    if close is not None and len(close):
        bh = close / close.iloc[0] * equity.iloc[0]
        fig.add_trace(
            go.Scatter(x=x, y=bh, name="buy & hold", line={"color": f"rgb({GREY})", "width": 1}),
            row=1,
            col=1,
        )
    dd = (equity / equity.cummax() - 1) * 100
    fig.add_trace(
        go.Scatter(
            x=x,
            y=dd,
            name="drawdown %",
            fill="tozeroy",
            line={"color": f"rgb({RED})", "width": 1},
        ),
        row=2,
        col=1,
    )
    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=height,
        legend={"orientation": "h", "y": 1.0, "x": 0, "yanchor": "bottom"},
        margin={"l": 60, "r": 30, "t": 70, "b": 30},
    )
    fig.update_yaxes(title_text="drawdown %", row=2, col=1)
    return fig


def trade_figure(
    df: pd.DataFrame,
    trade: dict,
    *,
    display_tz: str = "Asia/Kolkata",
    is_nse: bool = True,
    bars_before: int = 60,
    bars_after: int = 20,
) -> go.Figure:
    """Candles around one trade with entry/exit markers and stop/target lines."""
    ts = pd.to_datetime(df["timestamp"], utc=True)
    i0 = int(ts.searchsorted(pd.Timestamp(trade["entry_ts"])))
    exit_ts = trade.get("exit_ts")
    i1 = (
        int(ts.searchsorted(pd.Timestamp(exit_ts)))
        if exit_ts is not None and pd.notna(exit_ts)
        else len(df) - 1
    )
    win = df.iloc[max(0, i0 - bars_before) : min(len(df), i1 + bars_after + 1)].reset_index(
        drop=True
    )
    up = trade["direction"] == "long"
    marks = [
        {
            "timestamp": trade["entry_ts"],
            "price": trade["entry_price"],
            "label": "entry",
            "direction": "up" if up else "down",
        }
    ]
    if exit_ts is not None and pd.notna(exit_ts):
        marks.append(
            {
                "timestamp": exit_ts,
                "price": trade["exit_price"],
                "label": f"exit ({trade.get('exit_reason')})",
                "direction": "down" if up else "up",
            }
        )
    empty = pd.DataFrame(columns=["zone_low", "zone_high", "type", "touches", "strength", "source"])
    fig = levels_chart(
        win,
        empty,
        title=f"{trade['symbol']} · {trade['timeframe']} · {trade.get('label', '')}",
        display_tz=display_tz,
        is_nse=is_nse,
        intraday=trade["timeframe"] in ("5m", "15m", "1h"),
        markers=pd.DataFrame(marks),
        height=480,
    )
    for y, name, color in (
        (trade.get("stop"), "stop", RED),
        (trade.get("target"), "target", GREEN),
    ):
        if y is not None and pd.notna(y):
            fig.add_hline(
                y=float(y),
                row=1,
                col=1,
                line_dash="dash",
                line_width=1,
                line_color=f"rgb({color})",
                annotation_text=name,
                annotation_position="top left",
                annotation_font_color=f"rgb({color})",
            )
    return fig
