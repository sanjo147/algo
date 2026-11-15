"""Data providers. The only package allowed to import yfinance / ccxt / broker SDKs.

Use `get_provider(name)` so callers never import a concrete provider; the import
happens lazily, so e.g. ccxt is not loaded unless a crypto provider is requested.
"""

from __future__ import annotations

from importlib import import_module

from sanjotracker.data.providers.base import DataProvider

_REGISTRY: dict[str, tuple[str, str]] = {
    "yfinance_nse": ("sanjotracker.data.providers.yfinance_nse", "YFinanceNSEProvider"),
    "binance_ccxt": ("sanjotracker.data.providers.binance_ccxt", "BinanceCCXTProvider"),
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


__all__ = ["DataProvider", "available_providers", "get_provider"]
