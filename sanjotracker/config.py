"""Load config.yaml + .env into typed pydantic models.

Usage:
    from sanjotracker.config import get_config, get_secrets
    cfg = get_config()
    cfg.watchlist.nse  # ["RELIANCE.NS", ...]

config.yaml is looked up first; if missing, config.example.yaml is used so tests
and CI run without any local setup. Secrets come only from environment / .env.
"""

from __future__ import annotations

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
    crypto: list[str] = Field(default_factory=list)


class Benchmarks(BaseModel):
    nse: str = "^NSEI"
    crypto: str = "BTC/USDT"


class Providers(BaseModel):
    nse: str = "yfinance_nse"
    crypto: str = "binance_ccxt"


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


class BinanceCosts(BaseModel):
    maker_pct: float
    taker_pct: float
    slippage_pct: float


class Costs(BaseModel):
    nse_intraday: NseCosts
    nse_delivery: NseCosts
    binance_spot: BinanceCosts


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


class BacktestConfig(BaseModel):
    initial_capital: float = 100_000
    risk_per_trade_pct: float = 1.0


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

    @field_validator("timeframes")
    @classmethod
    def _check_timeframes(cls, v: list[str]) -> list[str]:
        bad = set(v) - VALID_TIMEFRAMES
        if bad:
            raise ValueError(f"unknown timeframes: {sorted(bad)}")
        return v

    @property
    def db_file(self) -> Path:
        """Absolute path to the SQLite DB (relative paths resolve from the project root)."""
        p = Path(self.app.db_path)
        return p if p.is_absolute() else PROJECT_ROOT / p


class Secrets(BaseSettings):
    """Secrets read from environment variables or .env. Never log these."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    telegram_bot_token: SecretStr | None = None
    telegram_chat_id: str | None = None
    anthropic_api_key: SecretStr | None = None


def resolve_config_path(path: str | Path | None = None) -> Path:
    """Return the config file to use: explicit path > config.yaml > config.example.yaml."""
    if path is not None:
        return Path(path)
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
