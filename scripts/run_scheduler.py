"""Start the alert scheduler (+ Telegram bot commands if .env has a token).

Examples:
    python scripts/run_scheduler.py                       # run forever (Ctrl+C to stop)
    python scripts/run_scheduler.py --keep-awake          # also stop Windows from sleeping
    python scripts/run_scheduler.py --once crypto_intraday  # one job now, then exit
    python scripts/run_scheduler.py --once nse_swing --no-telegram
    python scripts/run_scheduler.py --test-telegram       # send a test message
"""

from __future__ import annotations

import argparse
import ctypes
import sys
import time

from loguru import logger

from sanjotracker.alerts.telegram_bot import (
    BotActions,
    NullNotifier,
    build_bot_app,
    build_notifier,
)
from sanjotracker.config import get_secrets, load_config, save_watchlist
from sanjotracker.data.updater import build_updater
from sanjotracker.log import setup_logging
from sanjotracker.scheduler import JOBS, AlertService, build_scheduler

ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--once", choices=sorted(JOBS), help="run one job immediately and exit")
    p.add_argument("--no-telegram", action="store_true", help="log alerts instead of sending")
    p.add_argument("--no-bot", action="store_true", help="don't listen for bot commands")
    p.add_argument("--keep-awake", action="store_true", help="Windows: block sleep while running")
    p.add_argument("--test-telegram", action="store_true", help="send a test message and exit")
    return p.parse_args()


def keep_awake(on: bool) -> None:
    """Ask Windows not to sleep while this process runs (undone automatically on exit)."""
    if sys.platform != "win32":
        logger.warning("--keep-awake only works on Windows; use caffeinate/systemd-inhibit")
        return
    flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0)
    ctypes.windll.kernel32.SetThreadExecutionState(flags)
    logger.info("Windows sleep is {} while the scheduler runs", "blocked" if on else "allowed")


def validator(service: AlertService):  # noqa: ANN201
    """/add check: the symbol must return daily candles from its provider."""

    def check(symbol: str) -> str | None:
        cfg = service.load_cfg()
        updater = build_updater(
            service.store, {"nse": cfg.providers.nse, "crypto": cfg.providers.crypto}
        )
        res = updater.update(symbol, "1d")
        if res.error:
            return res.error
        if not res.total_rows:
            return "no data from the provider (check the ticker, e.g. TCS.NS)"
        return None

    return check


def main() -> int:
    args = parse_args()
    setup_logging()
    cfg = load_config()
    secrets = get_secrets()
    notifier = NullNotifier() if args.no_telegram else build_notifier(cfg, secrets)
    service = AlertService(notifier)

    if args.test_telegram:
        notifier.send_text("✅ SanjoTracker can reach you. Alerts will arrive here.")
        print("Sent (if Telegram is configured; otherwise see the log line above).")
        return 0
    if args.once:
        info = service.run_job(args.once, force=True)
        print(info.model_dump_json(indent=2))
        return 0

    sched = build_scheduler(service, cfg)
    sched.start()
    for job in sched.get_jobs():
        logger.info("{:<16} next run {}", job.id, job.next_run_time)
    if args.keep_awake:
        keep_awake(True)

    token, chat = secrets.telegram_bot_token, secrets.telegram_chat_id
    try:
        if not args.no_bot and not args.no_telegram and token and chat:
            actions = BotActions(
                load_cfg=load_config,
                store=service.store,
                save_watchlist=save_watchlist,
                validate_symbol=validator(service),
                status_text=lambda: service.status_text(sched),
            )
            logger.info("Telegram bot listening for commands (only from your chat id)")
            build_bot_app(token, chat, actions).run_polling(stop_signals=None)
        else:
            logger.info("Scheduler running without bot commands. Ctrl+C to stop.")
            while True:
                time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        sched.shutdown(wait=False)
        if args.keep_awake:
            keep_awake(False)
        logger.info("scheduler stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
