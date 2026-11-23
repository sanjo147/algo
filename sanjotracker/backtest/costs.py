"""NSE and Binance fee/slippage models, driven by config (nothing hard-coded).

NSE charges per executed order (fractions of turnover = price x qty):
- brokerage = min(brokerage_pct x turnover, brokerage_max_per_order)  (0 for free delivery)
- STT: delivery on buy AND sell; intraday on sell only
- exchange transaction charge, SEBI fee: both sides
- stamp duty: buy side only
- GST = gst_pct x (brokerage + exchange + SEBI)
Binance spot: taker fee x turnover (backtests use market orders = taker).
Slippage is applied to the fill PRICE (buy higher, sell lower), not as a fee.

Which model a run uses: crypto -> binance_spot; NSE 1d/1w -> nse_delivery;
NSE 5m/15m/1h -> nse_intraday.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from sanjotracker.config import BinanceCosts, Costs, NseCosts

Side = Literal["buy", "sell"]
INTRADAY_TFS = {"5m", "15m", "1h"}


class CostModel(ABC):
    name: str = "base"
    slippage_pct: float = 0.0

    @abstractmethod
    def charges(self, side: Side, price: float, qty: float) -> float:
        """Total fees/taxes (in quote currency) for one executed order."""

    def fill_price(self, side: Side, price: float) -> float:
        """Price after slippage: you buy a bit higher and sell a bit lower."""
        return price * (1 + self.slippage_pct) if side == "buy" else price * (1 - self.slippage_pct)

    def round_trip_pct(self, price: float = 1000.0, qty: float = 100.0) -> float:
        """Fees + slippage of buy-then-sell as a fraction of turnover (for reports)."""
        t = price * qty
        fees = self.charges("buy", price, qty) + self.charges("sell", price, qty)
        return fees / t + 2 * self.slippage_pct


class NseCostModel(CostModel):
    def __init__(self, c: NseCosts, name: str) -> None:
        self.c = c
        self.name = name
        self.slippage_pct = c.slippage_pct

    def charges(self, side: Side, price: float, qty: float) -> float:
        c = self.c
        turnover = abs(price * qty)
        brokerage = c.brokerage_pct * turnover
        if c.brokerage_max_per_order > 0:
            brokerage = min(brokerage, c.brokerage_max_per_order)
        stt = (c.stt_buy_pct if side == "buy" else c.stt_sell_pct) * turnover
        exchange = c.exchange_txn_pct * turnover
        sebi = c.sebi_pct * turnover
        stamp = c.stamp_buy_pct * turnover if side == "buy" else 0.0
        gst = c.gst_pct * (brokerage + exchange + sebi)
        return brokerage + stt + exchange + sebi + stamp + gst


class BinanceCostModel(CostModel):
    def __init__(self, c: BinanceCosts) -> None:
        self.c = c
        self.name = "binance_spot"
        self.slippage_pct = c.slippage_pct

    def charges(self, side: Side, price: float, qty: float) -> float:
        return self.c.taker_pct * abs(price * qty)


class ZeroCostModel(CostModel):
    """Only for tests and 'what if there were no costs' comparisons."""

    name = "zero"

    def charges(self, side: Side, price: float, qty: float) -> float:
        return 0.0


def cost_model_for(symbol: str, timeframe: str, costs: Costs) -> CostModel:
    if "/" in symbol:
        return BinanceCostModel(costs.binance_spot)
    if timeframe in INTRADAY_TFS:
        return NseCostModel(costs.nse_intraday, "nse_intraday")
    return NseCostModel(costs.nse_delivery, "nse_delivery")
