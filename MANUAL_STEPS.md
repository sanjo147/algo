# Manual steps after the NSE-only switch

SanjoTracker is now **Indian stock market (NSE) only**. Crypto is gone, and daily
data comes from **NSE's official MCP servers**, falling back to Yahoo when they
can't be reached. The code is done. These steps need you, because they depend on
your network, your accounts or your data. Run every command from
`D:\algo\sanjotracker` with the virtual environment active:

```powershell
cd D:\algo\sanjotracker
.venv\Scripts\Activate.ps1
```

---

## 1. Install the updated dependencies (2 min, required)

`ccxt` (crypto) was removed. `truststore` (lets Python use the Windows
certificate store) and `httpx` were added.

```powershell
pip install -r requirements.txt
pip uninstall -y ccxt
```

## 2. Check that NSE's MCP is reachable from your network (5 min, required)

On the network this was built on, `mcp.nseindia.in` (and the whole of
`nseindia.in`) is intercepted by a **FortiGate firewall**. That firewall swaps
in its own HTTPS certificate, which nothing trusts, so the app refuses to
connect (correctly) and uses Yahoo instead. Google and Yahoo were not
intercepted. This is typical of college, hostel and office Wi-Fi.

1. Connect to a different network: your **phone's mobile hotspot** or home broadband.
2. Run:
   ```powershell
   python scripts/check_nse_mcp.py
   ```
3. You want to see `PASS: NSE MCP works here`. The script also compares NSE's
   adjusted prices with Yahoo's. More than 0.5% apart means something is off:
   send me the output.

If it only works on another network:

- **Option A (simplest):** run the scheduler and backfill while on that network.
  On the firewalled network the app keeps working, just with Yahoo data.
- **Option B:** if the firewall belongs to your college or employer, ask their IT
  whether they provide an official root certificate to install, or whether they
  can exempt `nseindia.in` from HTTPS inspection. **Don't install a certificate
  from anyone else.** A trusted interception certificate lets that network read
  all your HTTPS traffic.
- Never "fix" this by turning certificate checking off.

## 3. Re-download your history (10–20 min, required, on the network that passed step 2)

Yahoo data used to be **dividend-adjusted**. It is now split/bonus-adjusted only,
the same as NSE's. Old and new candles therefore don't match, so start the
stored history fresh:

```powershell
python scripts/backfill.py --full
```

Expect about 20 small NSE requests per stock (5 years, 3 months per request).
`^NSEI` and all intraday timeframes still come from Yahoo. The `provider` column
of each candle records where it came from.

## 4. Delete the old crypto data (1 min, recommended)

Your database still has about 137,000 crypto rows: candles, 10 paper setups and
60 backtest runs. They're unused now. Look first, then delete. A backup copy of
the database is made automatically.

```powershell
python scripts/remove_crypto_data.py          # dry run: shows the counts
python scripts/remove_crypto_data.py --yes    # backs up the DB, then deletes
```

## 5. Restart anything that was running (1 min)

If `run_scheduler.py` or the dashboard is running, stop it (Ctrl+C) and start it
again, so it loads the new code. If you set up a Windows Task Scheduler task,
it picks up the new code at the next log-in.

```powershell
python scripts/run_scheduler.py --keep-awake
streamlit run app/Home.py
```

There are now two scheduled jobs: `nse_intraday` (every 15 min, 09:32–15:32) and
`nse_swing` (15:46 + daily summary). The crypto jobs are gone.

## 6. Update the NSE holiday list (5 min, once a year)

`config.example.yaml` only lists 25 Dec 2026 and 26 Jan 2027. Copy the full
2026/2027 list from **nseindia.com → Resources → Exchange Communication →
Holidays** into `nse_market: holidays:` in your `config.yaml`. On missing
holidays the scheduler would scan stale data.

(You don't have a `config.yaml` yet, so the example is in use. The first save
from the dashboard's Settings page or Telegram's `/add` creates one. Or copy it:
`copy config.example.yaml config.yaml`.)

## 7. Telegram (if not done yet, 5 min)

There is no `.env` file yet, so no alerts reach your phone. Follow
**README → Telegram alerts + scheduler → Create your bot**, then run
`python scripts/run_scheduler.py --test-telegram`.

## 8. Optional: use NSE's MCP directly in Claude (5 min)

The same official servers can be added to Claude, so you can ask questions like
"compare TCS and INFY over 6 months" in chat. This is separate from the app.

- **Claude desktop / claude.ai:** Settings → Connectors → *Add custom connector*.
  Add one connector per URL:
  - `https://mcp.nseindia.in/bhavcopy/cm/mcp` (end-of-day history, 5 years)
  - `https://mcp.nseindia.in/cmmkt/mcp` (live market data, ~1–3 min delayed)
- **Claude Code (terminal):**
  ```powershell
  claude mcp add --transport http nse-bhavcopy https://mcp.nseindia.in/bhavcopy/cm/mcp
  claude mcp add --transport http nse-live https://mcp.nseindia.in/cmmkt/mcp
  ```

Step 2's network limitation applies here too.

---

### What you don't need to do

- Change any code. The provider order is in `config.yaml → providers:`.
- Get an API key. NSE's MCP is free with no login. (A broker API such as
  Zerodha Kite is only needed later, for real-time intraday data.)

### Good to know

- **BSE:** BSE has no official MCP server. The only "NSE + BSE" MCP servers are
  third-party ones that scrape exchange websites, so SanjoTracker sticks to
  NSE's official one. Almost every BSE stock you'd trade is also on NSE.
- **Terms:** NSE's MCP data is for informational and educational use. A personal
  tracker is fine; don't resell the data or train AI models on it.
- **Intraday:** NSE's MCP has no intraday candles, so 5m/15m/1h stay on Yahoo
  (~15 min delayed) until you connect a broker API (Phase 9).
