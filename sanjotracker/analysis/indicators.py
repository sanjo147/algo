"""Technical indicators on OHLCV DataFrames (columns: timestamp, open, high, low, close, volume).

Every indicator is causal: the value at bar t uses only bars <= t, so they are
safe for signals and backtests. RSI / ATR / ADX use Wilder's smoothing
(an EMA with alpha = 1/n), which is what TradingView and most platforms show.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, n: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(n, min_periods=n).mean()


def ema(series: pd.Series, n: int) -> pd.Series:
    """Exponential moving average (span n), NaN until n bars exist."""
    out = series.ewm(span=n, adjust=False, min_periods=n).mean()
    return out


def _wilder(series: pd.Series, n: int) -> pd.Series:
    """Wilder smoothing = EMA with alpha 1/n."""
    return series.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Relative Strength Index, 0–100."""
    delta = close.diff()
    gain = _wilder(delta.clip(lower=0), n)
    loss = _wilder(-delta.clip(upper=0), n)
    rs = gain / loss
    out = 100 - 100 / (1 + rs)
    # all gains and no losses -> RSI 100 (rs is inf -> formula already gives 100;
    # 0/0 for a flat series -> leave as 50)
    out = out.where(~(gain.eq(0) & loss.eq(0)), 50.0)
    return out


def true_range(df: pd.DataFrame) -> pd.Series:
    """Max of (high-low, |high-prev close|, |low-prev close|)."""
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Average True Range (Wilder)."""
    return _wilder(true_range(df), n)


def adx(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    """ADX with +DI / -DI. ADX > 25 ≈ trending, < 20 ≈ ranging.

    Returns columns: plus_di, minus_di, adx.
    """
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    atr_n = atr(df, n)
    plus_di = 100 * _wilder(plus_dm, n) / atr_n
    minus_di = 100 * _wilder(minus_dm, n) / atr_n
    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    return pd.DataFrame(
        {"plus_di": plus_di, "minus_di": minus_di, "adx": _wilder(dx, n)}, index=df.index
    )


def vwap(df: pd.DataFrame, session_tz: str = "UTC") -> pd.Series:
    """Session VWAP that resets at each calendar day in `session_tz`.

    Use "Asia/Kolkata" for NSE (resets every trading day) and "UTC" for crypto.
    Only meaningful on intraday timeframes.
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3
    session = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(session_tz).dt.date
    pv = (typical * df["volume"]).groupby(session).cumsum()
    vol = df["volume"].groupby(session).cumsum()
    # zero-volume series (indices) -> fall back to typical price
    return (pv / vol.replace(0, np.nan)).fillna(typical)


def volume_ma(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """Simple moving average of volume."""
    return sma(df["volume"], n)


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands. Returns columns: bb_mid, bb_upper, bb_lower, bb_width (fraction of mid)."""
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    upper, lower = mid + k * sd, mid - k * sd
    return pd.DataFrame(
        {"bb_mid": mid, "bb_upper": upper, "bb_lower": lower, "bb_width": (upper - lower) / mid},
        index=close.index,
    )


def add_indicators(
    df: pd.DataFrame,
    *,
    ema_periods: tuple[int, ...] = (20, 50, 200),
    rsi_n: int = 14,
    atr_n: int = 14,
    adx_n: int = 14,
    vol_n: int = 20,
    session_tz: str = "UTC",
    intraday: bool = False,
) -> pd.DataFrame:
    """Return a copy of df with the standard indicator columns added."""
    out = df.copy()
    for p in ema_periods:
        out[f"ema_{p}"] = ema(out["close"], p)
    out["rsi"] = rsi(out["close"], rsi_n)
    out["atr"] = atr(out, atr_n)
    out["atr_pct"] = out["atr"] / out["close"] * 100
    out = out.join(adx(out, adx_n))
    out["vol_ma"] = volume_ma(out, vol_n)
    out["vol_ratio"] = out["volume"] / out["vol_ma"].replace(0, np.nan)
    out = out.join(bollinger(out["close"]))
    if intraday:
        out["vwap"] = vwap(out, session_tz)
    return out
