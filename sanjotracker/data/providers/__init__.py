"""Data providers. The only package allowed to import yfinance or talk to NSE's MCP.

Use `build_nse_provider(cfg.providers)` (or `get_provider(name)`) so callers never
import a concrete provider; imports happen lazily.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from sanjotracker.data.providers.base import DataProvider

if TYPE_CHECKING:
    from sanjotracker.config import Providers

_REGISTRY: dict[str, tuple[str, str]] = {
    "nse_mcp": ("sanjotracker.data.providers.nse_mcp", "NseMcpProvider"),
    "yfinance_nse": ("sanjotracker.data.providers.yfinance_nse", "YFinanceNSEProvider"),
    "kite": ("sanjotracker.data.providers.kite_stub", "KiteProvider"),
}


def available_providers() -> list[str]:
    return sorted(_REGISTRY)


def get_provider(name: str, **kwargs: object) -> DataProvider:
    """Instantiate a provider by its config name (see `providers:` in config.yaml)."""
    try:
        module_name, cls_name = _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown provider {name!r}; choose from {available_providers()}"
        ) from None
    cls = getattr(import_module(module_name), cls_name)
    return cls(**kwargs)


def build_nse_provider(providers: Providers) -> DataProvider:
    """The configured chains: daily (e.g. NSE MCP -> Yahoo) and intraday (Yahoo)."""
    from sanjotracker.data.providers.routed import RoutedProvider

    cache: dict[str, DataProvider] = {}

    def make(name: str) -> DataProvider:
        if name not in cache:
            cache[name] = get_provider(name)
        return cache[name]

    return RoutedProvider(
        daily=[make(n) for n in providers.daily], intraday=[make(n) for n in providers.intraday]
    )


__all__ = ["DataProvider", "available_providers", "build_nse_provider", "get_provider"]
