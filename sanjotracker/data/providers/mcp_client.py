"""Minimal synchronous MCP client (Model Context Protocol, streamable HTTP transport).

Enough to call tools on a remote MCP server such as NSE's official ones:
POST JSON-RPC 2.0 messages, accept either a JSON or an SSE (text/event-stream)
reply, and carry the Mcp-Session-Id header the server hands out at initialize.

TLS uses the operating system's certificate store when the `truststore` package is
installed (so a company/college root installed in Windows is honoured), otherwise
certifi. Certificate checks are never switched off.
"""

from __future__ import annotations

import json
import ssl
import time
from typing import Any

import httpx
from loguru import logger

from sanjotracker.data.providers.base import ProviderError

PROTOCOL_VERSION = "2025-06-18"
ACCEPT = "application/json, text/event-stream"  # NSE answers 400 to JSON-only


class McpError(ProviderError):
    """The MCP server could not be reached or returned an error."""


class McpToolError(McpError):
    """A tool ran but reported an error (bad symbol, no data...)."""


def _ssl_context() -> ssl.SSLContext | bool:
    try:
        import truststore

        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except ImportError:
        return True  # httpx default: certifi


def _explain(exc: Exception, url: str) -> str:
    text = str(exc)
    if "CERTIFICATE_VERIFY_FAILED" in text or "certificate" in text.lower():
        return (
            f"TLS certificate for {url} is not trusted. Your network is probably "
            "inspecting HTTPS traffic (e.g. a college/office firewall); try another "
            f"network such as a mobile hotspot. ({type(exc).__name__}: {text[:160]})"
        )
    return f"{type(exc).__name__}: {text[:200]}"


def _parse_body(resp: httpx.Response) -> dict:
    """JSON-RPC message from a JSON or SSE response (last `data:` event with an id)."""
    ctype = resp.headers.get("content-type", "")
    if "text/event-stream" in ctype:
        msgs = []
        for line in resp.text.splitlines():
            if line.startswith("data:"):
                payload = line[5:].strip()
                if payload:
                    msgs.append(json.loads(payload))
        replies = [m for m in msgs if "id" in m]
        if not replies:
            raise McpError("empty SSE reply from MCP server")
        return replies[-1]
    return resp.json()


def _tool_payload(result: dict) -> Any:
    """Decode a tools/call result: text content -> JSON (or plain text/number)."""
    parts = [c.get("text", "") for c in result.get("content", []) if c.get("type") == "text"]
    text = "".join(parts).strip()
    if result.get("isError"):
        raise McpToolError(text[:300] or "tool reported an error")
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return text
    if isinstance(data, dict) and data.get("error"):  # NSE reports some errors softly
        raise McpToolError(str(data["error"])[:300])
    return data


class McpClient:
    def __init__(
        self,
        url: str,
        *,
        client: httpx.Client | None = None,
        timeout: float = 90.0,
        retries: int = 3,
        backoff: float = 3.0,
        user_agent: str = "sanjotracker/0.1",
    ) -> None:
        self.url = url
        self.retries = retries
        self.backoff = backoff
        self.client = client or httpx.Client(timeout=timeout, verify=_ssl_context())
        self.headers = {
            "Content-Type": "application/json",
            "Accept": ACCEPT,
            "User-Agent": user_agent,
        }
        self.session_id: str | None = None
        self._next_id = 0

    # ------------------------------------------------------------------ transport

    def _post(self, message: dict) -> httpx.Response:
        headers = dict(self.headers)
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        return self.client.post(self.url, headers=headers, json=message)

    def _request(self, method: str, params: dict) -> dict:
        last: Exception | None = None
        for attempt in range(1, self.retries + 1):
            self._next_id += 1
            msg = {"jsonrpc": "2.0", "id": self._next_id, "method": method, "params": params}
            try:
                resp = self._post(msg)
                if resp.status_code == 404 and self.session_id and method != "initialize":
                    self.session_id = None  # session expired: open a new one and retry
                    self.initialize()
                    continue
                if resp.status_code in (429, 502, 503, 504):
                    raise McpError(f"HTTP {resp.status_code}")
                if resp.status_code >= 400:
                    raise McpError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                body = _parse_body(resp)
                if "error" in body:
                    raise McpError(f"{method}: {body['error'].get('message', body['error'])}")
                if sid := resp.headers.get("mcp-session-id"):
                    self.session_id = sid
                return body.get("result", {})
            except McpToolError:
                raise
            except (httpx.HTTPError, McpError, ValueError) as exc:
                last = exc
                if isinstance(exc, httpx.ConnectError) and "certificate" in str(exc).lower():
                    break  # retrying won't fix an untrusted certificate
                if attempt < self.retries:
                    wait = self.backoff * 2 ** (attempt - 1)
                    logger.warning("MCP {} failed ({}); retry in {:.0f}s", method, exc, wait)
                    time.sleep(wait)
        raise McpError(_explain(last, self.url) if last else f"{method} failed")

    # ------------------------------------------------------------------ protocol

    def initialize(self) -> dict:
        result = self._request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "sanjotracker", "version": "0.1"},
            },
        )
        try:  # notifications get no reply; failures here are harmless
            self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})
        except httpx.HTTPError:
            pass
        return result

    def _ensure(self) -> None:
        if self.session_id is None:
            self.initialize()

    def list_tools(self) -> list[dict]:
        self._ensure()
        return self._request("tools/list", {}).get("tools", [])

    def call_tool(self, name: str, arguments: dict | None = None) -> Any:
        self._ensure()
        return _tool_payload(
            self._request("tools/call", {"name": name, "arguments": arguments or {}})
        )
