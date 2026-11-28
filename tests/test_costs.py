"""Fee models: hand-checked NSE delivery and intraday charges."""

import pytest

from sanjotracker.backtest.costs import (
    NseCostModel,
    ZeroCostModel,
    cost_model_for,
)
from sanjotracker.config import PROJECT_ROOT, load_config

COSTS = load_config(PROJECT_ROOT / "config.example.yaml").costs


def test_nse_delivery_buy_and_sell_by_hand() -> None:
    m = NseCostModel(COSTS.nse_delivery, "nse_delivery")
    t = 1000 * 100  # ₹1,00,000 turnover
    exch, sebi = 0.0000297 * t, 0.000001 * t
    gst = 0.18 * (0 + exch + sebi)  # zero brokerage on delivery
    buy = 0.001 * t + exch + sebi + 0.00015 * t + gst  # STT + exch + SEBI + stamp + GST
    sell = 0.001 * t + exch + sebi + gst  # no stamp on sell
    assert m.charges("buy", 1000, 100) == pytest.approx(buy)
    assert m.charges("sell", 1000, 100) == pytest.approx(sell)


def test_nse_intraday_brokerage_cap_and_sell_only_stt() -> None:
    m = NseCostModel(COSTS.nse_intraday, "nse_intraday")
    big = 1_000_000  # 0.03% would be ₹300 -> capped at ₹20
    exch, sebi = 0.0000297 * big, 0.000001 * big
    buy = 20 + exch + sebi + 0.00003 * big + 0.18 * (20 + exch + sebi)
    sell = 20 + 0.00025 * big + exch + sebi + 0.18 * (20 + exch + sebi)
    assert m.charges("buy", 1000, 1000) == pytest.approx(buy)
    assert m.charges("sell", 1000, 1000) == pytest.approx(sell)
    small = m.charges("buy", 100, 10)  # ₹1,000 -> 0.03% = ₹0.30 brokerage, not ₹20
    assert small < 1


def test_slippage_moves_fill_against_you() -> None:
    m = NseCostModel(COSTS.nse_delivery, "nse_delivery")
    assert m.fill_price("buy", 100) == pytest.approx(100.05)
    assert m.fill_price("sell", 100) == pytest.approx(99.95)
    assert 0.003 < m.round_trip_pct() < 0.004  # ~0.32% delivery round trip


def test_model_selection() -> None:
    assert cost_model_for("TCS.NS", "1d", COSTS).name == "nse_delivery"
    assert cost_model_for("TCS.NS", "15m", COSTS).name == "nse_intraday"
    assert ZeroCostModel().charges("buy", 100, 10) == 0
