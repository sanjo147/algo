"""Strategy registry: get_strategy("ema_crossover", fast=10)."""

from __future__ import annotations

from typing import Any

from sanjotracker.strategies.base import Strategy, StrategyContext
from sanjotracker.strategies.ema_crossover import EmaCrossover
from sanjotracker.strategies.orb_intraday import OrbIntraday
from sanjotracker.strategies.rsi_mean_reversion import RsiMeanReversion
from sanjotracker.strategies.sr_breakout import SrBreakout
from sanjotracker.strategies.vwap_pullback import VwapPullback

STRATEGIES: dict[str, type[Strategy]] = {
    cls.name: cls for cls in (SrBreakout, EmaCrossover, RsiMeanReversion, OrbIntraday, VwapPullback)
}


def get_strategy(name: str, **params: Any) -> Strategy:
    if name not in STRATEGIES:
        raise KeyError(f"unknown strategy {name!r}; choose from {sorted(STRATEGIES)}")
    return STRATEGIES[name](**params)


__all__ = ["STRATEGIES", "Strategy", "StrategyContext", "get_strategy"]
