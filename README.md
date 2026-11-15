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
| 4 | Alerts + scheduler | ⏳ |
| 5 | Strategies + backtester | ⏳ |
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

More commands (scheduler, backtest) are added phase by phase.

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
