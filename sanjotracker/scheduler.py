"""APScheduler jobs (NSE market hours vs crypto 24/7). All times are IST.

| job              | when                                         | what                    |
|------------------|----------------------------------------------|-------------------------|
| nse_intraday     | Mon–Fri 09:32–15:32, every 15 min (:02 :17…) | 15m scan, NSE           |
| nse_swing        | Mon–Fri at daily_summary_time + 1 min (15:46)| 1d scan + daily summary |
| crypto_intraday  | every 15 min, 24/7 (:00:30, :15:30, …)       | 15m scan, crypto        |
| crypto_swing     | daily 05:35 (after the 00:00 UTC close)      | 1d scan, crypto         |

Each run is 2 min (NSE) / 30 s (crypto) after the bar closes, so the bar is final.
NSE jobs skip weekends and the holidays in config (`nse_market.holidays`).
Jobs run one at a time (single worker) so SQLite and the free APIs aren't hammered.
If the laptop sleeps, a missed run is done once on wake if still within the grace
time, otherwise skipped (no burst of catch-up scans).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, time, timedelta
from zoneinfo import ZoneInfo

from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger
from pydantic import BaseModel

from sanjotracker.alerts.dashboard_feed import AlertFeed
from sanjotracker.alerts.rules import AlertRules
from sanjotracker.alerts.telegram_bot import (
    EMOJI,
    Notifier,
    NullNotifier,
    event_chart_png,
    send_event,
)
from sanjotracker.analysis.breakout import Event
from sanjotracker.analysis.regime import Regime
from sanjotracker.config import Config, load_config
from sanjotracker.data.store import CandleStore
from sanjotracker.scanner import Market, Scanner, Style

IST = ZoneInfo("Asia/Kolkata")
JOBS: dict[str, tuple[Style, Market]] = {
    "nse_intraday": ("intraday", "nse"),
    "nse_swing": ("swing", "nse"),
    "crypto_intraday": ("intraday", "crypto"),
    "crypto_swing": ("swing", "crypto"),
}
# first/last time an NSE 15m bar can have closed (+2 min) on a trading day
NSE_INTRADAY_WINDOW = (time(9, 30), time(15, 35))


class RunInfo(BaseModel):
    job: str
    started: datetime
    seconds: float = 0
    events: int = 0
    stored: int = 0
    sent: int = 0
    errors: dict[str, str] = {}
    skipped: str | None = None


def nse_intraday_due(now: datetime, cfg: Config) -> str | None:
    """None if the NSE 15m scan should run now, else the reason to skip."""
    local = now.astimezone(IST)
    if not cfg.nse_market.is_trading_day(local.date()):
        return "not an NSE trading day"
    lo, hi = NSE_INTRADAY_WINDOW
    if not lo <= local.time() <= hi:
        return "outside NSE session"
    return None


class AlertService:
    """scan -> rules -> dashboard feed -> Telegram. One instance per process."""

    def __init__(
        self,
        notifier: Notifier,
        load_cfg: Callable[[], Config] = load_config,
        store: CandleStore | None = None,
        feed: AlertFeed | None = None,
        scanner_factory: Callable[[Config, CandleStore], Scanner] | None = None,
        chart: Callable[[Event], bytes] | None = None,
    ) -> None:
        self.load_cfg = load_cfg
        cfg = load_cfg()
        self.store = store or CandleStore(cfg.db_file)
        self.feed = feed or AlertFeed(engine=self.store.engine)
        self.notifier = notifier
        self.scanner_factory = scanner_factory or (lambda c, s: Scanner(c, s))
        self._chart = chart
        self.last_runs: dict[str, RunInfo] = {}
        self.benchmarks: dict[str, Regime] = {}
        self.started = datetime.now(UTC)

    def chart_fn(self, cfg: Config) -> Callable[[Event], bytes] | None:
        if not cfg.alerts.send_charts:
            return None
        return self._chart or (lambda e: event_chart_png(self.store, e, cfg))

    # ------------------------------------------------------------------ jobs

    def run_job(self, job: str, now: datetime | None = None, force: bool = False) -> RunInfo:
        """Run one scheduled job. `force` ignores the NSE session check (for --once)."""
        style, market = JOBS[job]
        now = now or datetime.now(UTC)
        cfg = self.load_cfg()  # re-read each run: picks up /add, /remove and edits
        info = RunInfo(job=job, started=now)
        if job == "nse_intraday" and not force:
            info.skipped = nse_intraday_due(now, cfg)
        if (
            job == "nse_swing"
            and not force
            and not cfg.nse_market.is_trading_day(now.astimezone(IST).date())
        ):
            info.skipped = "not an NSE trading day"
        if info.skipped:
            logger.info("{} skipped: {}", job, info.skipped)
            self.last_runs[job] = info
            return info

        t0 = datetime.now(UTC)
        scanner = self.scanner_factory(cfg, self.store)
        try:
            events = scanner.scan(style, market=market)
            self.benchmarks.update(scanner.benchmark_regimes(market))
        except Exception as exc:  # a scan crash must not kill the scheduler
            logger.exception("{} scan crashed", job)
            info.errors["scan"] = str(exc)
            events = []
        info.errors.update(scanner.errors)
        info.events = len(events)

        chart = self.chart_fn(cfg)
        for ev, decision, alert_id in AlertRules(
            cfg.alerts, self.feed, cfg.app.timezone_display
        ).process(events, now):
            info.stored += 1
            if decision.action != "send":
                continue
            if isinstance(self.notifier, NullNotifier):  # Telegram not configured / --no-telegram
                self.feed.mark_sent(alert_id, False, "telegram off")
                continue
            try:
                send_event(self.notifier, ev, cfg.app.timezone_display, chart)
                self.feed.mark_sent(alert_id, True)
                info.sent += 1
            except Exception as exc:
                logger.error("telegram send failed for {}: {}", ev.symbol, exc)
                self.feed.mark_sent(alert_id, False, f"telegram error: {exc}")

        if job == "nse_swing":
            self.send_daily_summary(now, cfg)
        info.seconds = round((datetime.now(UTC) - t0).total_seconds(), 1)
        self.last_runs[job] = info
        logger.info(
            "{} done in {}s: {} events, {} stored, {} sent, {} problems",
            job,
            info.seconds,
            info.events,
            info.stored,
            info.sent,
            len(info.errors),
        )
        return info

    # ------------------------------------------------------------------ summary / status

    def daily_summary(self, now: datetime, cfg: Config) -> str:
        local = now.astimezone(IST)
        day_start = datetime.combine(local.date(), time(0), IST)
        today = self.feed.recent(since=day_start, limit=500)
        lines = [f"📊 <b>Daily summary · {local:%a %d %b}</b>", ""]
        for sym, r in self.benchmarks.items():
            adx = "-" if r.adx is None else f"{r.adx:.0f}"
            lines.append(f"{sym}: {r.market} · {r.trend} (ADX {adx}) · {r.volatility} vol")
        if today.empty:
            lines += ["", "No alerts today."]
        else:
            counts = today["event"].value_counts()
            lines += [
                "",
                f"Alerts today: {len(today)}  "
                + "  ".join(f"{EMOJI.get(k, '')}{v}" for k, v in counts.items()),
                "",
                "<b>Top 5</b>",
            ]
            for r in today.sort_values("confidence", ascending=False).head(5).itertuples():
                lines.append(
                    f"{EMOJI.get(r.event, '')} {r.symbol} {r.timeframe} {r.event} · {r.confidence}"
                )
        problems = {k: v for i in self.last_runs.values() for k, v in i.errors.items()}
        if problems:
            lines += ["", f"⚠️ {len(problems)} data/scan problems (see logs)"]
        return "\n".join(lines)

    def send_daily_summary(self, now: datetime, cfg: Config) -> None:
        try:
            self.notifier.send_text(self.daily_summary(now, cfg))
        except Exception as exc:
            logger.error("daily summary failed: {}", exc)

    def status_text(self, scheduler: BackgroundScheduler | None = None) -> str:
        up = datetime.now(UTC) - self.started
        lines = [f"🟢 running for {str(up).split('.')[0]}", ""]
        for job in JOBS:
            r = self.last_runs.get(job)
            if r is None:
                last = "not run yet"
            elif r.skipped:
                last = f"skipped ({r.skipped})"
            else:
                last = (
                    f"{r.started.astimezone(IST):%d %b %H:%M} · {r.events} ev · "
                    f"{r.sent} sent · {len(r.errors)} problems"
                )
            nxt = ""
            if scheduler is not None and (j := scheduler.get_job(job)) and j.next_run_time:
                nxt = f" · next {j.next_run_time.astimezone(IST):%d %b %H:%M}"
            lines.append(f"<b>{job}</b>: {last}{nxt}")
        since = datetime.now(UTC) - timedelta(hours=24)
        lines += ["", f"alerts in last 24h: {len(self.feed.recent(since=since, limit=1000))}"]
        return "\n".join(lines)


# ------------------------------------------------------------------ scheduler


def build_scheduler(service: AlertService, cfg: Config) -> BackgroundScheduler:
    """BackgroundScheduler (IST) with the four scan jobs; call .start() on it."""
    sched = BackgroundScheduler(
        timezone=IST,
        executors={"default": ThreadPoolExecutor(1)},
        job_defaults={"coalesce": True, "max_instances": 1, "misfire_grace_time": 600},
    )
    summary = cfg.alerts.daily_summary_time
    swing_at = (datetime.combine(datetime.today(), summary) + timedelta(minutes=1)).time()
    triggers = {
        "nse_intraday": CronTrigger(
            day_of_week="mon-fri", hour="9-15", minute="2,17,32,47", timezone=IST
        ),
        "nse_swing": CronTrigger(
            day_of_week="mon-fri", hour=swing_at.hour, minute=swing_at.minute, timezone=IST
        ),
        "crypto_intraday": CronTrigger(minute="0,15,30,45", second=30, timezone=IST),
        "crypto_swing": CronTrigger(hour=5, minute=35, timezone=IST),
    }
    for job, trig in triggers.items():
        grace = 3600 if job.endswith("swing") else 600
        sched.add_job(service.run_job, trig, args=[job], id=job, name=job, misfire_grace_time=grace)
    return sched
