# SanjoTracker

Personal NSE + crypto tracker: support/resistance zones, breakout detection,
strategy backtests, and alerts to Telegram and a Streamlit dashboard.

> Analysis and alerts only. There is no order-placement code.
> Free yfinance NSE data is delayed ~15 min. Treat intraday alerts as practice
> until a broker API is plugged in.

## Status

| Phase | What | Status |
|---|---|---|
| 0 | Project skeleton, config, CI | ✅ |
| 1 | Data layer (providers, SQLite store, updater) | ✅ |
| 2 | Indicators + support/resistance | ✅ |
| 3 | Breakout scanner + regime | ✅ |
| 4 | Alerts + scheduler | ✅ |
| 5 | Strategies + backtester | ✅ |
| 6 | Streamlit dashboard | ⏳ |
| 7 | Paper trading + journal | ⏳ |

## Setup

Requires Python 3.11+.

```bash
git clone https://github.com/<your-username>/sanjotracker.git
cd sanjotracker
python -m venv .venv
```

Activate the venv:

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate
```

Install and configure:

```bash
pip install -r requirements-dev.txt
cp config.example.yaml config.yaml   # Windows: copy config.example.yaml config.yaml
cp .env.example .env                 # Windows: copy .env.example .env
```

Edit `config.yaml` (watchlist, thresholds, costs, holidays) and `.env`
(Telegram token and chat ID, set up in Phase 4). Both are git-ignored.

## Run

```bash
pytest -q                     # tests
ruff check . && ruff format . # lint + format
streamlit run app/Home.py     # dashboard (full version in Phase 6)
```

### Download market data

```bash
python scripts/backfill.py                                  # whole watchlist, all timeframes
python scripts/backfill.py --market crypto --timeframes 1d  # a subset
python scripts/backfill.py --symbols RELIANCE.NS --timeframes 15m
```

Re-running only fetches candles that are missing. Data goes to `data/sanjotracker.db`.

Free-data limits: yfinance keeps 5m/15m for ~60 days and 1h for ~2 years.
Run backfill (or the scheduler, Phase 4) at least every few weeks so intraday
history keeps growing in your local DB instead of leaving gaps.
Crypto data comes from `data-api.binance.vision`, Binance's official public
market-data host, because `api.binance.com` is blocked by some Indian ISPs.

### Check support / resistance visually

```bash
python scripts/plot_levels.py RELIANCE.NS 1d --open
python scripts/plot_levels.py BTC/USDT 15m --bars 400 --open
```

Saves `reports/levels_<symbol>_<tf>.html` (dark candlestick chart) and prints the
nearest zones. Shaded bands are pivot zones (green = support, red = resistance,
darker = stronger, label = touches · strength 0–100). Dotted lines are reference
levels: previous day/week/month high-low, 52-week high-low, round numbers.

How zones are built: a swing high/low is confirmed only after `pivot_right` bars
have closed (no lookahead). Nearby pivots are grouped into zones no wider than
`zone_width_atr` x ATR, and strength = touches x recency x relative volume.
Tune these in `config.yaml` under `levels:`.

### Scan for breakouts now

```bash
python scripts/scan_now.py swing                     # 1d bars, 1w trend filter
python scripts/scan_now.py intraday --market crypto  # 15m bars, 1h trend filter
python scripts/scan_now.py swing --lookback 5 --reasons
```

Prints the benchmark regime (NIFTY / BTC vs their 200 EMA, ADX trend, volatility)
and a table of events on the last closed bar (`--lookback N` for the last N):

| Event | Meaning |
|---|---|
| 🚀 breakout / 🔻 breakdown | close through a strong zone, volume > 1.5x avg, body > 0.5x ATR |
| 🔁 retest | after a breakout, price dips back into the zone and closes on the breakout side |
| ⚠️ failed_breakout / failed_breakdown | closes back inside the zone within 3 bars |
| 👀 approaching | within 0.5x ATR of a strong zone (or inside it) |

Each event has a confidence 0–100 made of listed reasons (`--reasons` shows them).
Breakouts against the higher-timeframe trend (EMA 20 vs 50) are hidden unless you
pass `--all`. All thresholds live in `config.yaml` under `breakout:` and `regime:`.

No lookahead: an event on bar t only uses zones that existed before bar t, bar t's
own close/volume, and higher-timeframe bars that had closed by then.

## Telegram alerts + scheduler (Phase 4)

### 1. Create your bot (2 minutes)

1. In Telegram, open **@BotFather** → send `/newbot` → pick a name and a username
   ending in `bot` (e.g. `sanjo_tracker_bot`). BotFather replies with a **token**
   like `123456789:AAH...`. Treat it like a password.
2. Open your new bot's chat and press **Start** (send it any message), so it is
   allowed to message you.
3. Get your **chat ID**: open
   `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates` in your browser and find
   `"chat":{"id":123456789`. (Or message **@userinfobot**, which replies with your id.)
4. Copy `.env.example` to `.env` and fill in:
   ```
   TELEGRAM_BOT_TOKEN=123456789:AAH...
   TELEGRAM_CHAT_ID=123456789
   ```
5. Test it:
   ```bash
   python scripts/run_scheduler.py --test-telegram
   ```

The bot answers **only your chat ID**. Commands: `/start`, `/watchlist`,
`/add TCS.NS`, `/remove TCS.NS`, `/levels RELIANCE.NS`, `/status`.
`/add` checks that the ticker returns data first. It edits only the `watchlist:`
section of `config.yaml` (created from the example if missing; the old file is
kept as `config.yaml.bak`).

### 2. Run the scheduler

```bash
python scripts/run_scheduler.py --keep-awake
```

| Job | When (IST) | Scan |
|---|---|---|
| nse_intraday | Mon–Fri 09:32–15:32, every 15 min | 15m bars, 1h trend |
| nse_swing | Mon–Fri 15:46 + daily summary | 1d bars, 1w trend |
| crypto_intraday | every 15 min, 24/7 | 15m bars, 1h trend |
| crypto_swing | daily 05:35 (after the 00:00 UTC close) | 1d bars, 1w trend |

NSE jobs skip weekends and `nse_market.holidays`. Keep that list updated each year
from the NSE holiday calendar.

Try one job right now without waiting: `python scripts/run_scheduler.py --once crypto_intraday`
(add `--no-telegram` to only log).

Alert rules (`alerts:` in config): confidence ≥ `min_confidence` (60), the same
symbol + timeframe + event + zone at most once per `cooldown_minutes` (240), never
twice for the same bar, optional quiet hours (stored for the dashboard, not sent).
Every alert is saved in the `alerts` table, which the dashboard reads in Phase 6.

### 3. Keep it running on a laptop

- `--keep-awake` stops Windows from sleeping **while the scheduler runs**. Normal
  sleep comes back when you stop it. Closing the lid can still sleep the laptop:
  Control Panel → Power Options → "Choose what closing the lid does" → *Do nothing*
  (when plugged in).
- Keep it plugged in. If it sleeps anyway, missed scans are skipped (a 15-min scan
  runs late only if the laptop wakes within 10 min; daily scans within 1 hour).
- Start on login: Task Scheduler → *Create Basic Task* → trigger *When I log on* →
  action *Start a program*:
  - Program: `D:\algo\sanjotracker\.venv\Scripts\pythonw.exe`
  - Arguments: `scripts\run_scheduler.py --keep-awake`
  - Start in: `D:\algo\sanjotracker`

  (`pythonw` runs without a console window; logs go to `logs\sanjotracker.log`.)
- Check it's alive any time with `/status` in Telegram.

## Backtesting (Phase 5)

```bash
python scripts/run_backtest.py --list                          # strategies + parameters
python scripts/run_backtest.py sr_breakout RELIANCE.NS 1d      # one backtest
python scripts/run_backtest.py ema_crossover BTC/USDT 1h --param fast=10 --plot
python scripts/run_backtest.py sr_breakout RELIANCE.NS 1d --walkforward
python scripts/run_backtest.py --leaderboard                   # everything, ~1 min
```

| Strategy | Idea | Timeframes |
|---|---|---|
| sr_breakout | close through a strong S/R zone on volume (same rules as the scanner) | 15m, 1h, 1d |
| ema_crossover | EMA 20/50 cross, ATR stop, exit on the opposite cross | all |
| rsi_mean_reversion | RSI < 30 above the 200 EMA (and mirror), exit on RSI recovery | all |
| orb_intraday | first close outside the 09:15–09:30 range, NSE only | 5m, 15m |
| vwap_pullback | trend + dip to session VWAP that closes back on the trend side | 5m, 15m |

How the engine works:

- Signals use bar t's close; orders fill at bar t+1's **open** with slippage.
- Every trade pays the fees in `costs:` (NSE delivery / intraday / Binance).
- Stops and targets trigger inside the bar. If a bar hits both, the stop counts.
- Each trade risks 1% of equity, with no leverage.
- Shorts only on NSE intraday, which is squared off at 15:30.

Output:

- **Metrics:** CAGR, Sharpe, Sortino, Calmar, max drawdown and its duration, win
  rate, profit factor, expectancy in R, exposure, and the result vs buy & hold.
- **Regime breakdown:** trend, volatility and market at the signal bar.
- **Monte Carlo:** drawdown percentiles from 1000 shuffles of the trade order.

Results go to the `backtest_runs` / `backtest_trades` tables and
`reports/leaderboard_*.csv`.

How to read it honestly:

- The **leaderboard uses default parameters only**. Tuning on the same data you
  report is curve-fitting.
- To tune, use `--walkforward`: it picks parameters on past windows and scores
  them on the next, unseen window. Expect the out-of-sample result to be much
  worse than the full-history one. That gap is the overfitting.
- `<20 trades` or `<1 year of data` rows are shown but not ranked. Free intraday
  data is only ~59 days, too short to judge.
- Fees matter. Binance spot costs ~0.30% per round trip; NSE delivery ~0.32%.
  On a 15m chart, that can be most of a stop distance.

More commands (dashboard) are added phase by phase.

## Layout

```
sanjotracker/
  config.py         typed config from config.yaml + .env
  data/providers/   the only place yfinance / ccxt are imported
  data/             SQLite store + incremental updater
  analysis/         indicators, S/R levels, breakouts, regime
  strategies/       strategy implementations
  backtest/         engine, costs, metrics, walk-forward, Monte Carlo
  alerts/           rules, Telegram, dashboard feed
app/                Streamlit dashboard
scripts/            CLI entry points
tests/              pytest
```

## Rules this codebase follows

- No lookahead: signals on bar *t* use data up to *t*'s close and fill at *t+1* open.
- Every backtest includes NSE / Binance costs from config.
- Secrets live only in `.env`.
- Timestamps are stored in UTC and shown in Asia/Kolkata.
