"""Telegram sender and bot commands.

Sending uses the plain Bot HTTP API through httpx (synchronous, easy to test).
Commands use python-telegram-bot's polling Application and answer ONLY your own
chat id (anyone can find a bot and message it).

Commands: /start /watchlist /add SYMBOL /remove SYMBOL /levels SYMBOL /status

Security: the bot token is part of every Telegram URL, so errors are redacted
before logging and httpx's own request logging is silenced.
"""

from __future__ import annotations

import asyncio
import html
import logging
import time
from collections.abc import Callable
from typing import Protocol
from zoneinfo import ZoneInfo

import httpx
import pandas as pd
from loguru import logger
from pydantic import SecretStr

from sanjotracker.analysis.breakout import Event, fmt_price
from sanjotracker.analysis.levels import compute_levels, nearest_levels
from sanjotracker.charts import event_figure, levels_chart
from sanjotracker.config import Config, Secrets
from sanjotracker.data.store import CandleStore
from sanjotracker.data.updater import market_of, session_tz_for

TELEGRAM_API = "https://api.telegram.org"
CAPTION_LIMIT = 1024
MESSAGE_LIMIT = 4096

for _name in ("httpx", "httpcore", "telegram", "apscheduler"):
    logging.getLogger(_name).setLevel(logging.WARNING)  # httpx logs URLs = token

EMOJI = {
    "breakout": "🚀",
    "breakdown": "🔻",
    "retest": "🔁",
    "failed_breakout": "⚠️",
    "failed_breakdown": "⚠️",
    "approaching": "👀",
}
ARROW = {"up": "↑", "down": "↓"}


class TelegramError(RuntimeError):
    pass


# ------------------------------------------------------------------ formatting


def format_event(e: Event, tz: str = "Asia/Kolkata") -> str:
    """HTML message: emoji SYMBOL tf EVENT, price, zone, volume, confidence, reasons."""
    when = pd.Timestamp(e.bar_ts).tz_convert(ZoneInfo(tz))
    when_txt = (
        when.strftime("%d %b") if e.timeframe in ("1d", "1w") else when.strftime("%d %b %H:%M")
    )
    vol = "-" if e.volume_ratio is None else f"{e.volume_ratio:.1f}x avg"
    zone_kind = "resistance" if e.zone_type == "R" else "support"
    lines = [
        f"{EMOJI.get(e.event.value, '')} <b>{html.escape(e.symbol)}</b> {e.timeframe} · "
        f"<b>{e.event.value.replace('_', ' ').upper()}</b> {ARROW[e.direction]}",
        f"Price <b>{fmt_price(e.price)}</b>  (bar {when_txt})",
        f"Zone {html.escape(e.zone_text)} ({zone_kind})",
        f"Volume {vol} · Confidence <b>{e.confidence}</b>/100",
        "",
        *[f"• {html.escape(r)}" for r in e.reasons],
        f"<i>{html.escape(e.regime.label())}</i>",
    ]
    return "\n".join(lines)


def redact(text: str, token: str | None) -> str:
    return text.replace(token, "***") if token else text


# ------------------------------------------------------------------ charts


def render_png(fig, width: int = 900, height: int = 500) -> bytes:  # noqa: ANN001
    """Plotly -> PNG via kaleido (uses your installed Chrome/Edge)."""
    return fig.to_image(format="png", width=width, height=height, scale=1)


def event_chart_png(store: CandleStore, event: Event, cfg: Config) -> bytes:
    """PNG for an alert: candles up to the event bar, zones known before it, a marker."""
    df = store.load(event.symbol, event.timeframe, limit=cfg.breakout.scan_history_bars)
    before = df[df["timestamp"] < pd.Timestamp(event.bar_ts)]
    levels = compute_levels(
        before, event.symbol, event.timeframe, cfg.levels, session_tz=session_tz_for(event.symbol)
    )
    fig = event_figure(
        df,
        levels,
        event,
        display_tz=cfg.app.timezone_display,
        is_nse=market_of(event.symbol) == "nse",
    )
    return render_png(fig)


# ------------------------------------------------------------------ senders


class Notifier(Protocol):
    def send_text(self, text: str) -> None: ...
    def send_photo(self, png: bytes, caption: str) -> None: ...


class NullNotifier:
    """Used when Telegram is off or not configured: just logs."""

    def send_text(self, text: str) -> None:
        logger.info("[telegram off] {}", text.splitlines()[0] if text else "")

    def send_photo(self, png: bytes, caption: str) -> None:
        self.send_text(caption)


class TelegramNotifier:
    """Sends to one chat. Retries once on Telegram's 429 'slow down' reply."""

    def __init__(
        self,
        token: SecretStr,
        chat_id: str,
        client: httpx.Client | None = None,
        pause: float = 1.0,
    ) -> None:
        self._token = token.get_secret_value()
        self.chat_id = chat_id
        self.client = client or httpx.Client(timeout=30)
        self.pause = pause  # Telegram allows ~1 message/second per chat

    def _call(self, method: str, data: dict, files: dict | None = None) -> dict:
        url = f"{TELEGRAM_API}/bot{self._token}/{method}"
        for attempt in range(2):
            try:
                r = self.client.post(url, data=data, files=files)
            except httpx.HTTPError as exc:
                raise TelegramError(redact(f"{type(exc).__name__}: {exc}", self._token)) from None
            body = (
                r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
            )
            if r.status_code == 429 and attempt == 0:
                wait = float(body.get("parameters", {}).get("retry_after", 5))
                logger.warning("Telegram rate limit; waiting {}s", wait)
                time.sleep(wait)
                continue
            if not body.get("ok"):
                desc = body.get("description", f"HTTP {r.status_code}")
                raise TelegramError(redact(f"{method} failed: {desc}", self._token))
            if self.pause:
                time.sleep(self.pause)
            return body
        raise TelegramError(f"{method} failed after retry")

    def send_text(self, text: str) -> None:
        for i in range(0, len(text), MESSAGE_LIMIT):
            self._call(
                "sendMessage",
                {
                    "chat_id": self.chat_id,
                    "text": text[i : i + MESSAGE_LIMIT],
                    "parse_mode": "HTML",
                    "disable_web_page_preview": "true",
                },
            )

    def send_photo(self, png: bytes, caption: str) -> None:
        short = caption if len(caption) <= CAPTION_LIMIT else caption.split("\n\n")[0]
        self._call(
            "sendPhoto",
            {"chat_id": self.chat_id, "caption": short, "parse_mode": "HTML"},
            files={"photo": ("chart.png", png, "image/png")},
        )
        if short != caption:
            self.send_text(caption)


def build_notifier(cfg: Config, secrets: Secrets | None = None) -> Notifier:
    secrets = secrets or Secrets()
    if not cfg.alerts.telegram_enabled:
        return NullNotifier()
    if secrets.telegram_bot_token is None or not secrets.telegram_chat_id:
        logger.warning("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set in .env; Telegram is off")
        return NullNotifier()
    return TelegramNotifier(secrets.telegram_bot_token, secrets.telegram_chat_id)


def send_event(
    notifier: Notifier,
    event: Event,
    tz: str,
    chart: Callable[[Event], bytes] | None = None,
) -> None:
    """Send one alert; falls back to text-only if the chart can't be rendered."""
    text = format_event(event, tz)
    png = None
    if chart is not None:
        try:
            png = chart(event)
        except Exception as exc:  # no Chrome, kaleido hiccup... the text still goes out
            logger.warning("chart for {} failed ({}); sending text only", event.symbol, exc)
    if png:
        notifier.send_photo(png, text)
    else:
        notifier.send_text(text)


# ------------------------------------------------------------------ bot commands


HELP = (
    "<b>SanjoTracker</b> — S/R breakout alerts (analysis only, no orders).\n\n"
    "/watchlist — symbols being scanned\n"
    "/add SYMBOL — e.g. /add TCS.NS or /add DOGE/USDT\n"
    "/remove SYMBOL\n"
    "/levels SYMBOL — nearest support/resistance (daily) + chart\n"
    "/status — scheduler and data status"
)


class BotActions:
    """The plain (sync) functions behind each command, so they can be unit-tested."""

    def __init__(
        self,
        load_cfg: Callable[[], Config],
        store: CandleStore,
        save_watchlist: Callable[[list[str], list[str]], None],
        validate_symbol: Callable[[str], str | None],
        status_text: Callable[[], str],
    ) -> None:
        self.load_cfg = load_cfg
        self.store = store
        self.save_watchlist = save_watchlist
        self.validate_symbol = validate_symbol
        self.status_text = status_text

    def watchlist(self) -> str:
        cfg = self.load_cfg()
        nse = ", ".join(cfg.watchlist.nse) or "-"
        crypto = ", ".join(cfg.watchlist.crypto) or "-"
        return f"<b>NSE</b>: {html.escape(nse)}\n<b>Crypto</b>: {html.escape(crypto)}"

    def add(self, symbol: str) -> str:
        symbol = symbol.strip().upper()
        cfg = self.load_cfg()
        market = market_of(symbol)
        current = getattr(cfg.watchlist, market)
        if symbol in current:
            return f"{html.escape(symbol)} is already on the watchlist."
        err = self.validate_symbol(symbol)
        if err:
            return f"Could not add {html.escape(symbol)}: {html.escape(err)}"
        nse, crypto = list(cfg.watchlist.nse), list(cfg.watchlist.crypto)
        (nse if market == "nse" else crypto).append(symbol)
        self.save_watchlist(nse, crypto)
        return f"Added {html.escape(symbol)} ({market}). It will be in the next scan."

    def remove(self, symbol: str) -> str:
        symbol = symbol.strip().upper()
        cfg = self.load_cfg()
        nse, crypto = list(cfg.watchlist.nse), list(cfg.watchlist.crypto)
        for lst in (nse, crypto):
            if symbol in lst:
                lst.remove(symbol)
                self.save_watchlist(nse, crypto)
                return f"Removed {html.escape(symbol)}."
        return f"{html.escape(symbol)} is not on the watchlist."

    def levels(self, symbol: str, timeframe: str = "1d") -> tuple[str, bytes | None]:
        symbol = symbol.strip().upper()
        cfg = self.load_cfg()
        df = self.store.load(symbol, timeframe, limit=cfg.breakout.scan_history_bars)
        if df.empty:
            return f"No {timeframe} data for {html.escape(symbol)}. Try /add {symbol} first.", None
        lv = compute_levels(df, symbol, timeframe, cfg.levels, session_tz=session_tz_for(symbol))
        price = float(df["close"].iloc[-1])
        near = nearest_levels(lv, price, n=3)
        lines = [f"<b>{html.escape(symbol)}</b> {timeframe} · last close {fmt_price(price)}", ""]
        for r in near.sort_values("zone_low", ascending=False).itertuples():
            what = f"{r.touches}x" if r.source == "pivot" else r.source.replace("_", " ")
            lines.append(
                f"{'🟥' if r.zone_low >= price else '🟩'} {fmt_price(r.zone_low)}–"
                f"{fmt_price(r.zone_high)}  {what} · {r.strength:.0f}"
            )
        png = None
        try:
            fig = levels_chart(
                df.tail(150).reset_index(drop=True),
                lv,
                title=f"{symbol} · {timeframe}",
                display_tz=cfg.app.timezone_display,
                is_nse=market_of(symbol) == "nse",
                intraday=False,
                height=500,
            )
            png = render_png(fig)
        except Exception as exc:
            logger.warning("levels chart failed: {}", exc)
        return "\n".join(lines), png

    def status(self) -> str:
        return self.status_text()


def build_bot_app(token: SecretStr, chat_id: str, actions: BotActions):  # noqa: ANN201
    """python-telegram-bot Application with the commands, limited to `chat_id`."""
    from telegram import Update
    from telegram.constants import ParseMode
    from telegram.ext import Application, CommandHandler, ContextTypes, filters

    only_me = filters.Chat(chat_id=int(chat_id))
    app = Application.builder().token(token.get_secret_value()).build()

    def _arg(ctx: ContextTypes.DEFAULT_TYPE) -> str | None:
        return ctx.args[0] if ctx.args else None

    async def reply(update: Update, text: str) -> None:
        await update.effective_message.reply_text(text, parse_mode=ParseMode.HTML)

    async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        await reply(update, HELP)

    async def watchlist(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        await reply(update, await asyncio.to_thread(actions.watchlist))

    async def add(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        sym = _arg(ctx)
        if not sym:
            return await reply(update, "Usage: /add SYMBOL  (e.g. /add TCS.NS)")
        await reply(update, f"Checking {html.escape(sym.upper())}…")
        await reply(update, await asyncio.to_thread(actions.add, sym))

    async def remove(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        sym = _arg(ctx)
        if not sym:
            return await reply(update, "Usage: /remove SYMBOL")
        await reply(update, await asyncio.to_thread(actions.remove, sym))

    async def levels(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        sym = _arg(ctx)
        if not sym:
            return await reply(update, "Usage: /levels SYMBOL")
        text, png = await asyncio.to_thread(actions.levels, sym)
        if png:
            await update.effective_message.reply_photo(png, caption=text, parse_mode="HTML")
        else:
            await reply(update, text)

    async def status(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        await reply(update, await asyncio.to_thread(actions.status))

    async def on_error(update: object, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        logger.error("bot command failed: {}", redact(str(ctx.error), token.get_secret_value()))

    for name, fn in (
        ("start", start),
        ("help", start),
        ("watchlist", watchlist),
        ("add", add),
        ("remove", remove),
        ("levels", levels),
        ("status", status),
    ):
        app.add_handler(CommandHandler(name, fn, filters=only_me))
    app.add_error_handler(on_error)
    return app
