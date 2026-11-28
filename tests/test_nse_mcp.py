"""NSE MCP client + provider against a fake MCP server (no network), and the routed
provider's fallback / top-up logic."""

import json
from datetime import UTC, datetime, timedelta

import httpx
import numpy as np
import pandas as pd
import pytest

from sanjotracker.config import Providers
from sanjotracker.data.providers import build_nse_provider
from sanjotracker.data.providers.base import ProviderError, empty_ohlcv
from sanjotracker.data.providers.mcp_client import McpClient, McpError, McpToolError
from sanjotracker.data.providers.nse_mcp import (
    NseMcpProvider,
    UnsupportedSymbolError,
    adjust_for_actions,
    nse_symbol,
    price_multiplier,
    rows_to_frame,
    to_weekly,
)
from sanjotracker.data.providers.routed import RoutedProvider
from sanjotracker.data.updater import market_of, normalize_symbol
from tests.conftest import FakeProvider, make_candles

DAYS = pd.bdate_range("2025-06-02", "2026-11-27")  # weekdays, ~18 months


def _bhav_rows(sym: str) -> list[dict]:
    """RAW prices with a 1:1 bonus on 2026-03-02 (prices halve on the ex-date)."""
    rows = []
    for i, d in enumerate(DAYS):
        base = 2400 + i if d < pd.Timestamp("2026-03-02") else 1200 + i / 2
        rows.append(
            {
                "date": d.strftime("%Y-%m-%d"),
                "open": base,
                "high": base + 10,
                "low": base - 10,
                "close": base + 2,
                "prevClose": base,
                "volume": 1000,
                "totalTradedValue": 1.0,
                "deliveryQty": 500,
            }
        )
    return rows


class FakeNse:
    """Enough of NSE's MCP servers: sessions, tools/call, paging, soft errors."""

    def __init__(self, sse: bool = False, fail_first: int = 0, expire_once: bool = False) -> None:
        self.sse, self.fail_first, self.expire_once = sse, fail_first, expire_once
        self.calls: list[tuple[str, dict]] = []
        self.sessions = 0

    def _reply(self, msg_id: int, result: dict, headers: dict | None = None) -> httpx.Response:
        body = {"jsonrpc": "2.0", "id": msg_id, "result": result}
        if self.sse:
            text = f"event: message\ndata: {json.dumps(body)}\n\n"
            return httpx.Response(
                200, text=text, headers={"content-type": "text/event-stream", **(headers or {})}
            )
        return httpx.Response(200, json=body, headers=headers or {})

    @staticmethod
    def _tool(payload: object) -> dict:
        return {"content": [{"type": "text", "text": json.dumps(payload)}], "isError": False}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert "text/event-stream" in request.headers["accept"]
        msg = json.loads(request.content)
        if self.fail_first:
            self.fail_first -= 1
            return httpx.Response(502, text="bad gateway")
        method = msg.get("method")
        if method == "notifications/initialized":
            return httpx.Response(202)
        if method == "initialize":
            self.sessions += 1
            return self._reply(
                msg["id"],
                {"protocolVersion": "2025-06-18"},
                {"mcp-session-id": f"s{self.sessions}"},
            )
        assert request.headers.get("mcp-session-id"), "session header missing"
        if self.expire_once:
            self.expire_once = False
            return httpx.Response(404, text="unknown session")
        name, args = msg["params"]["name"], msg["params"]["arguments"]
        self.calls.append((name, args))
        if name == "get_stock_history":
            if args["symbol"] == "NOPE":
                return self._reply(msg["id"], self._tool({"error": "Symbol not found"}))
            end = DAYS[-1] if args["endDate"] == "today" else pd.Timestamp(args["endDate"])
            start = end - pd.DateOffset(months=args["months"])
            rows = [r for r in _bhav_rows(args["symbol"]) if start < pd.Timestamp(r["date"]) <= end]
            nxt = (start).strftime("%Y-%m-%d") if start > DAYS[0] else None
            return self._reply(
                msg["id"],
                self._tool({"symbol": args["symbol"], "data": rows, "next_end_date": nxt}),
            )
        if name == "get_corporate_actions":
            return self._reply(
                msg["id"],
                self._tool(
                    {
                        "actions": [
                            {
                                "exDate": "2026-03-02",
                                "actionType": "Bonus",
                                "purpose": "Bonus 1:1",
                                "adjustmentFactor": 2.0,
                            },
                            {
                                "exDate": "2026-08-01",
                                "actionType": "Dividend",
                                "purpose": "Dividend Rs 10",
                                "adjustmentFactor": 0.99,
                            },
                        ]
                    }
                ),
            )
        if name == "cm_get_stock_quote":
            return self._reply(msg["id"], self._tool({"stock": {"lastTradedPrice": 1415.5}}))
        return httpx.Response(400, text="unknown tool")


def _provider(fake: FakeNse) -> NseMcpProvider:
    def client(url: str) -> McpClient:
        return McpClient(url, client=httpx.Client(transport=httpx.MockTransport(fake)), backoff=0)

    return NseMcpProvider(
        bhavcopy=client("https://bhav"), live=client("https://live"), call_pause=0
    )


# ------------------------------------------------------------------ client


@pytest.mark.parametrize("sse", [False, True])
def test_client_session_and_tool_call(sse: bool) -> None:
    fake = FakeNse(sse=sse)
    c = McpClient("https://x", client=httpx.Client(transport=httpx.MockTransport(fake)))
    assert (
        c.call_tool("cm_get_stock_quote", {"symbol": "RELIANCE"})["stock"]["lastTradedPrice"]
        == 1415.5
    )
    assert c.session_id == "s1"


def test_client_soft_error_and_retries() -> None:
    fake = FakeNse(fail_first=2)
    c = McpClient("https://x", client=httpx.Client(transport=httpx.MockTransport(fake)), backoff=0)
    with pytest.raises(McpToolError, match="Symbol not found"):
        c.call_tool("get_stock_history", {"symbol": "NOPE", "months": 3, "endDate": "today"})
    assert fake.sessions == 1  # two 502s were retried, then one session


def test_client_reopens_expired_session() -> None:
    fake = FakeNse(expire_once=True)
    c = McpClient("https://x", client=httpx.Client(transport=httpx.MockTransport(fake)), backoff=0)
    c.call_tool("cm_get_stock_quote", {"symbol": "TCS"})
    assert fake.sessions == 2


def test_certificate_error_is_explained_not_retried() -> None:
    calls = []

    def boom(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        raise httpx.ConnectError("[SSL: CERTIFICATE_VERIFY_FAILED] unable to get local issuer")

    c = McpClient("https://x", client=httpx.Client(transport=httpx.MockTransport(boom)), backoff=0)
    with pytest.raises(McpError, match="inspecting HTTPS"):
        c.call_tool("cm_get_stock_quote", {"symbol": "TCS"})
    assert len(calls) == 1


# ------------------------------------------------------------------ provider


def test_history_pages_back_and_adjusts_for_bonus() -> None:
    fake = FakeNse()
    p = _provider(fake)
    start = datetime(2025, 7, 1, tzinfo=UTC)
    df = p.get_ohlcv("RELIANCE.NS", "1d", start=start, end=datetime(2026, 11, 28, tzinfo=UTC))
    pages = [a for n, a in fake.calls if n == "get_stock_history"]
    assert len(pages) >= 5 and pages[0]["symbol"] == "RELIANCE" and pages[0]["months"] == 3
    assert df["timestamp"].is_unique and df["timestamp"].is_monotonic_increasing
    # bar time = 00:00 IST, like Yahoo's daily bars
    assert (df["timestamp"].dt.tz_convert("Asia/Kolkata").dt.hour == 0).all()
    # continuous across the bonus: no 50% gap at the ex-date
    jump = df["close"].pct_change().abs().max()
    assert jump < 0.01
    # dividends are NOT adjusted (same convention as Yahoo with auto_adjust=False)
    last = df[df["timestamp"] >= pd.Timestamp("2026-08-01", tz="Asia/Kolkata")]
    assert last["close"].iloc[-1] == pytest.approx(1200 + (len(DAYS) - 1) / 2 + 2)


def test_weekly_from_daily() -> None:
    p = _provider(FakeNse())
    w = p.get_ohlcv(
        "TCS.NS",
        "1w",
        start=datetime(2026, 9, 1, tzinfo=UTC),
        end=datetime(2026, 11, 28, tzinfo=UTC),
    )
    local = w["timestamp"].dt.tz_convert("Asia/Kolkata")
    assert (local.dt.day_of_week == 0).all() and (local.dt.hour == 0).all()


def test_indices_and_unknown_symbols_are_unsupported() -> None:
    p = _provider(FakeNse())
    with pytest.raises(UnsupportedSymbolError):
        p.get_ohlcv("^NSEI", "1d")
    with pytest.raises(UnsupportedSymbolError):
        p.get_ohlcv("NOPE.NS", "1d")
    assert p.get_latest("RELIANCE.NS") == 1415.5


@pytest.mark.parametrize(("factor", "mult"), [(2.0, 0.5), (0.5, 0.5), (5.0, 0.2), (0, 1.0)])
def test_price_multiplier(factor: float, mult: float) -> None:
    assert price_multiplier(factor) == pytest.approx(mult)


def test_adjust_only_before_ex_date() -> None:
    df = rows_to_frame(_bhav_rows("X"))
    adj = adjust_for_actions(
        df, [{"exDate": "02-03-2026", "actionType": "SPLIT", "adjustmentFactor": 0.5}]
    )
    before = df["timestamp"] < pd.Timestamp("2026-03-02", tz="Asia/Kolkata")
    np.testing.assert_allclose(adj.loc[before, "close"], df.loc[before, "close"] * 0.5)
    np.testing.assert_allclose(adj.loc[~before, "close"], df.loc[~before, "close"])
    np.testing.assert_allclose(adj.loc[before, "volume"], df.loc[before, "volume"] * 2)


def test_symbol_helpers() -> None:
    assert nse_symbol("reliance.ns") == "RELIANCE"
    assert normalize_symbol("tcs") == "TCS.NS" and normalize_symbol("^NSEI") == "^NSEI"
    with pytest.raises(ValueError, match="NSE-only"):
        market_of("BTC/USDT")
    assert to_weekly(empty_ohlcv()).empty


# ------------------------------------------------------------------ routing


class Broken(FakeProvider):
    name = "broken"

    def get_ohlcv(self, *a, **k):  # noqa: ANN002, ANN003, ANN201
        raise ProviderError("connect error: certificate verify failed")


def _daily(n: int = 30) -> pd.DataFrame:
    return make_candles(datetime(2026, 10, 1, tzinfo=UTC), n, "1d", seed=1)


def test_router_falls_back_and_records_source() -> None:
    backup = FakeProvider(_daily())
    backup.name = "backup"
    primary = Broken(_daily())
    r = RoutedProvider(daily=[primary, backup], intraday=[backup])
    df = r.get_ohlcv("TCS.NS", "1d")
    assert len(df) == 30 and r.name == "backup"
    r.get_ohlcv("INFY.NS", "1d")
    assert "broken" in r._down  # unreachable primary is skipped for the rest of the run


def test_router_tops_up_newest_daily_bars() -> None:
    full = _daily(30)
    nse = FakeProvider(full.iloc[:28])  # NSE hasn't published the last 2 sessions yet
    nse.name = "nse_mcp"
    yahoo = FakeProvider(full)
    yahoo.name = "yfinance_nse"
    r = RoutedProvider(daily=[nse, yahoo], intraday=[yahoo])
    df = r.get_ohlcv("TCS.NS", "1d", end=full["timestamp"].iloc[-1] + timedelta(days=1))
    assert len(df) == 30 and r.name == "nse_mcp"
    pd.testing.assert_frame_equal(df.iloc[:28], full.iloc[:28].reset_index(drop=True))


def test_router_intraday_uses_intraday_chain() -> None:
    nse = Broken(_daily())
    yahoo = FakeProvider(_daily())
    r = RoutedProvider(daily=[nse, yahoo], intraday=[yahoo])
    assert r.chain("15m") == [yahoo]
    assert not r.get_ohlcv("TCS.NS", "15m").empty


def test_build_from_config() -> None:
    r = build_nse_provider(Providers())
    assert [p.name for p in r.daily] == ["nse_mcp", "yfinance_nse"]
    assert [p.name for p in r.intraday] == ["yfinance_nse"]
    assert r.daily[1] is r.intraday[0]  # one Yahoo instance shared


def test_router_history_is_the_longest_in_the_chain() -> None:
    r = build_nse_provider(Providers())
    assert r.max_history("1d") is None  # Yahoo is unlimited; NSE clamps its own requests
    assert r.max_history("15m") is not None  # Yahoo-only, ~60 days
