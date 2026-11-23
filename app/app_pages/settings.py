"""Settings: edit the watchlist and thresholds. Saved to config.yaml (only the changed
lines are rewritten, comments are kept, a .bak copy is made, invalid values are refused)."""

import re

import streamlit as st

from sanjotracker import dashboard as d
from sanjotracker.config import (
    get_secrets,
    resolve_config_path,
    save_watchlist,
    update_config_values,
)

c = d.cfg()

st.caption(
    f"Reading `{resolve_config_path().name}` · database `{c.db_file.name}`. "
    "The scheduler picks up changes on its next run; the dashboard immediately."
)

NSE_RE = re.compile(r"^[A-Z0-9&\-_.]+\.NS$")
CRYPTO_RE = re.compile(r"^[A-Z0-9]+/[A-Z0-9]+$")

# ------------------------------------------------------------------ watchlist
with st.form("wl_form", border=True):
    st.subheader("Watchlist")
    nse = st.multiselect(
        "NSE stocks",
        c.watchlist.nse,
        default=c.watchlist.nse,
        accept_new_options=True,
        key="set_nse",
        help="Yahoo tickers with .NS, e.g. TCS.NS. Type to add a new one.",
    )
    crypto = st.multiselect(
        "Crypto pairs",
        c.watchlist.crypto,
        default=c.watchlist.crypto,
        accept_new_options=True,
        key="set_crypto",
        help="Binance pairs, e.g. DOGE/USDT. Type to add a new one.",
    )
    if st.form_submit_button("Save watchlist", type="primary", icon=":material/save:"):
        nse_u = [s.strip().upper() for s in nse]
        crypto_u = [s.strip().upper() for s in crypto]
        bad = [s for s in nse_u if not NSE_RE.match(s)] + [
            s for s in crypto_u if not CRYPTO_RE.match(s)
        ]
        if bad:
            st.error(
                f"Not valid: {', '.join(bad)}. NSE needs `.NS`, crypto needs `BASE/QUOTE`.",
                icon=":material/error:",
            )
        else:
            save_watchlist(list(dict.fromkeys(nse_u)), list(dict.fromkeys(crypto_u)))
            st.cache_data.clear()
            new = set(nse_u + crypto_u) - set(c.watchlist.nse + c.watchlist.crypto)
            st.success(
                "Saved."
                + (
                    f" Download history for the new symbols: `python scripts/backfill.py "
                    f"--symbols {' '.join(sorted(new))}`"
                    if new
                    else ""
                ),
                icon=":material/check:",
            )

# ------------------------------------------------------------------ thresholds
with st.form("thr_form", border=True):
    st.subheader("Thresholds")
    t_lv, t_bo, t_al, t_bt = st.tabs(["Levels", "Breakouts", "Alerts", "Backtest"])
    new: dict[str, object] = {}
    with t_lv:
        cols = st.columns(4)
        new["levels.pivot_left"] = cols[0].number_input(
            "Pivot left bars", 2, 20, c.levels.pivot_left, key="s_pl"
        )
        new["levels.pivot_right"] = cols[1].number_input(
            "Pivot right bars",
            2,
            20,
            c.levels.pivot_right,
            key="s_pr",
            help="A pivot is only known after this many bars have closed (higher = later, surer).",
        )
        new["levels.zone_width_atr"] = cols[2].number_input(
            "Zone width (× ATR)", 0.1, 3.0, c.levels.zone_width_atr, 0.1, key="s_zw"
        )
        new["levels.recency_half_life_bars"] = cols[3].number_input(
            "Recency half-life (bars)", 10, 1000, c.levels.recency_half_life_bars, 10, key="s_hl"
        )
    with t_bo:
        cols = st.columns(4)
        b = c.breakout
        new["breakout.volume_mult"] = cols[0].number_input(
            "Volume × average", 0.5, 5.0, b.volume_mult, 0.1, key="s_vm"
        )
        new["breakout.min_body_atr"] = cols[1].number_input(
            "Min body (× ATR)", 0.0, 3.0, b.min_body_atr, 0.1, key="s_mb"
        )
        new["breakout.approach_atr"] = cols[2].number_input(
            "Approaching within (× ATR)", 0.1, 3.0, b.approach_atr, 0.1, key="s_ap"
        )
        new["breakout.failed_within_bars"] = cols[3].number_input(
            "Failed if back inside within (bars)", 1, 20, b.failed_within_bars, key="s_fw"
        )
        cols = st.columns(4)
        new["breakout.retest_within_bars"] = cols[0].number_input(
            "Retest within (bars)", 2, 50, b.retest_within_bars, key="s_rw"
        )
        new["breakout.min_zone_touches"] = cols[1].number_input(
            "Min zone touches", 1, 10, b.min_zone_touches, key="s_mt"
        )
        new["breakout.min_zone_strength"] = cols[2].number_input(
            "Min zone strength", 0.0, 100.0, float(b.min_zone_strength), 5.0, key="s_ms"
        )
        new["breakout.reference_min_strength"] = cols[3].number_input(
            "Min reference-level strength",
            0.0,
            100.0,
            float(b.reference_min_strength),
            5.0,
            key="s_rs",
        )
    with t_al:
        a = c.alerts
        cols = st.columns(3)
        new["alerts.min_confidence"] = cols[0].slider(
            "Min confidence", 0, 100, a.min_confidence, 5, key="s_mc"
        )
        new["alerts.cooldown_minutes"] = cols[1].number_input(
            "Cooldown (minutes)",
            0,
            10_080,
            a.cooldown_minutes,
            30,
            key="s_cd",
            help="Same symbol + timeframe + event + zone alerts at most once per this window.",
        )
        new["alerts.daily_summary_time"] = cols[2].time_input(
            "Daily summary (IST)", a.daily_summary_time, key="s_ds"
        )
        new["alerts.events"] = st.pills(
            "Events that alert",
            list(d.EVENT_ICON),
            selection_mode="multi",
            default=[e for e in a.events if e in d.EVENT_ICON],
            format_func=lambda k: k.replace("_", " "),
            key="s_ev",
        )
        cols = st.columns(4, vertical_alignment="bottom")
        new["alerts.telegram_enabled"] = cols[0].toggle(
            "Telegram on", a.telegram_enabled, key="s_tg"
        )
        new["alerts.send_charts"] = cols[1].toggle("Attach chart", a.send_charts, key="s_sc")
        new["alerts.quiet_hours.enabled"] = cols[2].toggle(
            "Quiet hours", a.quiet_hours.enabled, key="s_qe"
        )
        q1, q2 = cols[3].columns(2)
        new["alerts.quiet_hours.start"] = q1.time_input("From", a.quiet_hours.start, key="s_qs")
        new["alerts.quiet_hours.end"] = q2.time_input("To", a.quiet_hours.end, key="s_qt")
    with t_bt:
        bt = c.backtest
        cols = st.columns(4)
        new["backtest.initial_capital"] = cols[0].number_input(
            "Starting capital", 1_000.0, 1e9, float(bt.initial_capital), 10_000.0, key="s_ic"
        )
        new["backtest.risk_per_trade_pct"] = cols[1].number_input(
            "Risk per trade (%)", 0.1, 5.0, bt.risk_per_trade_pct, 0.1, key="s_rp"
        )
        new["backtest.min_trades"] = cols[2].number_input(
            "Min trades to rank", 1, 500, bt.min_trades, key="s_mn"
        )
        new["backtest.walkforward_windows"] = cols[3].number_input(
            "Walk-forward windows", 2, 12, bt.walkforward_windows, key="s_wf"
        )

    if st.form_submit_button("Save thresholds", type="primary", icon=":material/save:"):
        current = c.model_dump()

        def cur(path: str) -> object:
            v: object = current
            for p in path.split("."):
                v = v[p]  # type: ignore[index]
            return v

        changed = {k: v for k, v in new.items() if v != cur(k)}
        if not changed:
            st.info("Nothing changed.", icon=":material/info:")
        else:
            try:
                update_config_values(changed)
            except Exception as exc:  # validation refused the new values
                st.error(f"Not saved: {exc}", icon=":material/error:")
            else:
                st.cache_data.clear()
                st.success(
                    "Saved: " + ", ".join(k.split(".", 1)[1] for k in changed),
                    icon=":material/check:",
                )

# ------------------------------------------------------------------ telegram + data
left, right = st.columns(2)
with left, st.container(border=True):
    st.subheader("Telegram")
    s = get_secrets()
    has_token, has_chat = s.telegram_bot_token is not None, bool(s.telegram_chat_id)
    st.badge(
        "bot token set" if has_token else "no bot token",
        color="green" if has_token else "red",
        icon=":material/key:",
    )
    st.badge(
        "chat id set" if has_chat else "no chat id",
        color="green" if has_chat else "red",
        icon=":material/chat:",
    )
    st.caption("Secrets live in `.env` only and are never shown here. See the README for setup.")
    if st.button("Send test message", disabled=not (has_token and has_chat), key="s_test"):
        from sanjotracker.alerts.telegram_bot import build_notifier

        try:
            build_notifier(c, s).send_text("✅ Test from the SanjoTracker dashboard.")
            st.toast("Sent. Check Telegram.", icon=":material/send:")
        except Exception as exc:
            st.error(f"Telegram error: {exc}", icon=":material/error:")
with right, st.container(border=True):
    st.subheader("Stored data")
    inv = d.store().inventory()
    if inv.empty:
        st.caption("Empty. Run `python scripts/backfill.py`.")
    else:
        st.dataframe(
            inv, hide_index=True, height=260, alt="Candles stored per symbol and timeframe"
        )
