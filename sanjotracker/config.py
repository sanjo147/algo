"""Load config.yaml + .env into typed pydantic models.

Usage:
    from sanjotracker.config import get_config, get_secrets
    cfg = get_config()
    cfg.watchlist.nse  # ["RELIANCE.NS", ...]

config.yaml is looked up first; if missing, config.example.yaml is used so tests
and CI run without any local setup. Secrets come only from environment / .env.
"""

from __future__ import annotations

import os
import re
import shutil
from datetime import date, time
from functools import lru_cache
from pathlib import Path

import yaml
from loguru import logger
from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VALID_TIMEFRAMES = {"5m", "15m", "1h", "1d", "1w"}


class AppSettings(BaseModel):
    timezone_display: str = "Asia/Kolkata"
    db_path: str = "data/sanjotracker.db"
    log_level: str = "INFO"


class Watchlist(BaseModel):
    nse: list[str] = Field(default_factory=list)


class Benchmarks(BaseModel):
    nse: str = "^NSEI"


class Providers(BaseModel):
    """Ordered fallback chains. NSE's MCP has daily data only, so intraday uses Yahoo."""

    daily: list[str] = Field(default_factory=lambda: ["nse_mcp", "yfinance_nse"])
    intraday: list[str] = Field(default_factory=lambda: ["yfinance_nse"])


class StyleTimeframes(BaseModel):
    primary: str
    trend: str
    entry: str | None = None

    @field_validator("primary", "trend", "entry")
    @classmethod
    def _check_tf(cls, v: str | None) -> str | None:
        if v is not None and v not in VALID_TIMEFRAMES:
            raise ValueError(f"unknown timeframe {v!r}; use one of {sorted(VALID_TIMEFRAMES)}")
        return v


class Styles(BaseModel):
    swing: StyleTimeframes
    intraday: StyleTimeframes


class NseMarket(BaseModel):
    open: time = time(9, 15)
    close: time = time(15, 30)
    weekdays: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4])
    holidays: list[date] = Field(default_factory=list)

    def is_trading_day(self, d: date) -> bool:
        """True if NSE is open on this (IST) calendar date."""
        return d.weekday() in self.weekdays and d not in self.holidays


class LevelsConfig(BaseModel):
    pivot_left: int = 5
    pivot_right: int = 5
    zone_width_atr: float = 0.5
    recency_half_life_bars: int = 100
    round_number_step_pct: float = 1.0


class BreakoutConfig(BaseModel):
    approach_atr: float = 0.5
    volume_mult: float = 1.5
    volume_ma_bars: int = 20
    min_body_atr: float = 0.5
    failed_within_bars: int = 3
    trend_ema_fast: int = 20
    trend_ema_slow: int = 50
    retest_within_bars: int = 10
    min_zone_strength: float = 30
    min_zone_touches: int = 2
    reference_min_strength: float = 50
    scan_history_bars: int = 1500


class RegimeConfig(BaseModel):
    adx_period: int = 14
    adx_trending: float = 25
    adx_ranging: float = 20
    vol_lookback_days: int = 252
    benchmark_ema: int = 200
    vol_low_pct: float = 0.25
    vol_high_pct: float = 0.75


class NseCosts(BaseModel):
    """Indian equity charges as fractions of turnover (0.001 = 0.1 %)."""

    brokerage_pct: float
    brokerage_max_per_order: float
    stt_buy_pct: float = 0.0
    stt_sell_pct: float
    exchange_txn_pct: float
    sebi_pct: float
    stamp_buy_pct: float
    gst_pct: float
    slippage_pct: float


class Costs(BaseModel):
    nse_intraday: NseCosts
    nse_delivery: NseCosts


class QuietHours(BaseModel):
    enabled: bool = False
    start: time = time(23, 0)
    end: time = time(7, 0)


class AlertsConfig(BaseModel):
    telegram_enabled: bool = True
    cooldown_minutes: int = 240
    min_confidence: int = Field(60, ge=0, le=100)
    quiet_hours: QuietHours = Field(default_factory=QuietHours)
    daily_summary_time: time = time(15, 45)
    events: list[str] = Field(
        default_factory=lambda: [
            "breakout",
            "breakdown",
            "retest",
            "failed_breakout",
            "failed_breakdown",
            "approaching",
        ]
    )
    send_charts: bool = True


class PaperSetup(BaseModel):
    """One paper-traded strategy. `symbols`: tickers, or "watchlist" for all of it."""

    strategy: str
    timeframe: str = "1d"
    symbols: list[str] = Field(default_factory=lambda: ["watchlist"])
    params: dict = Field(default_factory=dict)
    stop_atr: float | None = None  # None -> paper.stop_atr
    target_r: float | None = None  # None -> paper.target_r

    @field_validator("timeframe")
    @classmethod
    def _check_tf(cls, v: str) -> str:
        if v not in VALID_TIMEFRAMES:
            raise ValueError(f"unknown timeframe {v!r}")
        return v


class PaperConfig(BaseModel):
    enabled: bool = True
    capital: float = 100_000  # virtual capital each setup sizes from
    risk_per_trade_pct: float = 1.0
    stop_atr: float | None = 2.0  # ATR stop distance; None = the strategy's own stop
    target_r: float | None = 2.0  # target in R; None = the strategy's own target
    telegram: bool = True  # notify entries/exits
    setups: list[PaperSetup] = Field(
        default_factory=lambda: [
            PaperSetup(strategy="ema_crossover", timeframe="1d"),
            PaperSetup(strategy="sr_breakout", timeframe="1d"),
        ]
    )


class BacktestConfig(BaseModel):
    initial_capital: float = 100_000
    risk_per_trade_pct: float = 1.0
    min_trades: int = 20  # fewer trades than this = not ranked on the leaderboard
    min_bars: int = 300  # skip symbol/timeframes with less history
    monte_carlo_runs: int = 1000
    walkforward_windows: int = 5
    walkforward_train_ratio: float = 2.0  # train window = 2 x test window
    walkforward_objective: str = "sharpe"
    risk_free_pct: float = 0.0  # annual, for Sharpe/Sortino (e.g. 6.5 for Indian T-bills)


class Config(BaseModel):
    """Full application config (non-secret)."""

    app: AppSettings = Field(default_factory=AppSettings)
    watchlist: Watchlist
    benchmarks: Benchmarks = Field(default_factory=Benchmarks)
    providers: Providers = Field(default_factory=Providers)
    timeframes: list[str]
    styles: Styles
    nse_market: NseMarket = Field(default_factory=NseMarket)
    levels: LevelsConfig = Field(default_factory=LevelsConfig)
    breakout: BreakoutConfig = Field(default_factory=BreakoutConfig)
    regime: RegimeConfig = Field(default_factory=RegimeConfig)
    costs: Costs
    alerts: AlertsConfig = Field(default_factory=AlertsConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    paper: PaperConfig = Field(default_factory=PaperConfig)

    @field_validator("timeframes")
    @classmethod
    def _check_timeframes(cls, v: list[str]) -> list[str]:
        bad = set(v) - VALID_TIMEFRAMES
        if bad:
            raise ValueError(f"unknown timeframes: {sorted(bad)}")
        return v

    @property
    def db_file(self) -> Path:
        """Absolute path to the SQLite DB (relative paths resolve from the project root).
        $SANJOTRACKER_DB overrides it."""
        p = Path(os.environ.get(DB_ENV) or self.app.db_path)
        return p if p.is_absolute() else PROJECT_ROOT / p


class Secrets(BaseSettings):
    """Secrets read from environment variables or .env. Never log these."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    telegram_bot_token: SecretStr | None = None
    telegram_chat_id: str | None = None
    anthropic_api_key: SecretStr | None = None


CONFIG_ENV = "SANJOTRACKER_CONFIG"  # optional: use another config file
DB_ENV = "SANJOTRACKER_DB"  # optional: use another SQLite file (tests, experiments)


def resolve_config_path(path: str | Path | None = None) -> Path:
    """Config file to use: explicit path > $SANJOTRACKER_CONFIG > config.yaml > example."""
    if path is not None:
        return Path(path)
    if os.environ.get(CONFIG_ENV):
        return Path(os.environ[CONFIG_ENV])
    local = PROJECT_ROOT / "config.yaml"
    if local.exists():
        return local
    return PROJECT_ROOT / "config.example.yaml"


def load_config(path: str | Path | None = None) -> Config:
    """Parse a YAML config file into a validated Config."""
    cfg_path = resolve_config_path(path)
    logger.debug("Loading config from {}", cfg_path)
    with cfg_path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return Config.model_validate(raw)


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Cached app-wide config."""
    return load_config()


@lru_cache(maxsize=1)
def get_secrets() -> Secrets:
    """Cached secrets from .env / environment."""
    return Secrets()


def _watchlist_block(nse: list[str], comments: dict[str, str]) -> str:
    lines = ["watchlist:", "  nse:" if nse else "  nse: []"]
    for s in nse:
        c = comments.get(s)
        lines.append(f"    - {s}" + (f"        # {c}" if c else ""))
    return "\n".join(lines) + "\n"


def save_watchlist(nse: list[str], path: str | Path | None = None) -> Path:
    """Rewrite only the `watchlist:` block of config.yaml (other comments are kept).

    If config.yaml doesn't exist yet it is created from config.example.yaml.
    A copy of the previous file is kept as config.yaml.bak.
    """
    target = writable_config_path(path)
    text = target.read_text(encoding="utf-8")
    m = re.search(r"^watchlist:[^\n]*\n(?:(?:[ \t]+[^\n]*|[ \t]*)\n)*", text, flags=re.M)
    if m is None:
        raise ValueError(f"no top-level 'watchlist:' section in {target}")
    comments = dict(re.findall(r"^\s+-\s+(\S+)\s+#\s*(.*)$", m.group(0), flags=re.M))
    block = _watchlist_block(nse, comments)
    trailing = "\n" if m.group(0).endswith("\n\n") else ""
    new = text[: m.start()] + block + trailing + text[m.end() :]
    Config.model_validate(yaml.safe_load(new))  # never write a config that won't load
    shutil.copyfile(target, target.with_suffix(".yaml.bak"))
    target.write_text(new, encoding="utf-8")
    get_config.cache_clear()
    logger.info("watchlist saved to {} ({} symbols)", target, len(nse))
    return target


def writable_config_path(path: str | Path | None = None) -> Path:
    """The config file edits go to (never config.example.yaml; created from it if needed)."""
    if path is not None:
        target = Path(path)
    elif os.environ.get(CONFIG_ENV):
        target = Path(os.environ[CONFIG_ENV])
    else:
        target = PROJECT_ROOT / "config.yaml"
    if not target.exists():
        shutil.copyfile(PROJECT_ROOT / "config.example.yaml", target)
    return target


def _yaml_scalar(v: object) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, time):
        return f'"{v:%H:%M}"'
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_yaml_scalar(x) for x in v) + "]"
    if isinstance(v, str):
        return f'"{v}"' if (":" in v or v == "" or v.strip() != v) else v
    return repr(v) if isinstance(v, float) else str(v)


def _find_block(lines: list[str], start: int, end: int, key: str, indent: int) -> int | None:
    """Index of the line `<indent spaces>key:` within lines[start:end], else None."""
    pat = re.compile(rf"^ {{{indent}}}{re.escape(key)}:(\s|$)")
    for i in range(start, end):
        if pat.match(lines[i]):
            return i
    return None


def _block_end(lines: list[str], i: int, indent: int) -> int:
    """First line after the block opened at line i (next line with indent <= indent)."""
    j = i + 1
    while j < len(lines):
        ln = lines[j]
        if ln.strip() and not ln.lstrip().startswith("#"):
            if len(ln) - len(ln.lstrip(" ")) <= indent:
                break
        j += 1
    return j


def update_config_values(updates: dict[str, object], path: str | Path | None = None) -> Path:
    """Set dotted keys (e.g. "breakout.volume_mult", "alerts.quiet_hours.enabled") in
    config.yaml by editing only those lines, so comments and layout are kept.

    Values must be scalars, times or flat lists. Missing keys are added at the end
    of their section. The result is validated before writing; a .bak is kept.
    """
    target = writable_config_path(path)
    lines = target.read_text(encoding="utf-8").splitlines()
    for dotted, value in updates.items():
        parts = dotted.split(".")
        start, end, indent = 0, len(lines), 0
        for depth, key in enumerate(parts):
            i = _find_block(lines, start, end, key, indent)
            last = depth == len(parts) - 1
            if i is None:
                # create the missing key (and any missing parent) at the end of the block
                insert = end
                while insert > start and not lines[insert - 1].strip():
                    insert -= 1
                if last:
                    lines.insert(insert, f"{' ' * indent}{key}: {_yaml_scalar(value)}")
                    break
                lines.insert(insert, f"{' ' * indent}{key}:")
                i = insert
            if last:
                ln = lines[i]
                comment = ""
                m = re.search(r"\s+#.*$", ln.split(":", 1)[1])
                if m:
                    comment = m.group(0)
                lines[i] = f"{' ' * indent}{key}: {_yaml_scalar(value)}{comment}"
            else:
                start, end, indent = i + 1, _block_end(lines, i, indent), indent + 2
    new = "\n".join(lines) + "\n"
    Config.model_validate(yaml.safe_load(new))  # never write a config that won't load
    shutil.copyfile(target, target.with_suffix(".yaml.bak"))
    target.write_text(new, encoding="utf-8")
    get_config.cache_clear()
    logger.info("config updated ({} keys) in {}", len(updates), target)
    return target
