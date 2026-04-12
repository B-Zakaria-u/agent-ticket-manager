"""
src/mcp/client.py — MCP server registry and tool loader.

Usage
-----
Register servers once at startup via env-var JSON or programmatically:

    MCP_SERVERS='[{"name":"filesystem","url":"http://localhost:3001","transport":"sse"}]'

Then in any agent:

    from src.mcp.client import MCPClientManager
    extra_tools = await MCPClientManager.get_tools(state.get("mcp_servers", []))

Supported transports
--------------------
  "sse"    — Server-Sent Events (streaming HTTP)  [default]
  "stdio"  — local process via stdin/stdout
  "http"   — plain HTTP POST (non-streaming)

Each server config dict
-----------------------
  name        : str   — human label (used in logging)
  url         : str   — base URL for sse / http transports
  command     : list  — argv for stdio transport   e.g. ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
  transport   : str   — "sse" | "stdio" | "http"   (default "sse")
  headers     : dict  — extra HTTP headers (auth tokens etc.)
  timeout     : int   — per-call timeout in seconds (default 30)
  enabled     : bool  — set False to skip without removing (default True)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

log = logging.getLogger(__name__)

# ── optional langchain-mcp-adapters import (graceful if not installed) ────────
try:
    from langchain_mcp_adapters.client import MultiServerMCPClient  # type: ignore
    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False
    log.warning(
        "langchain-mcp-adapters not installed — MCP tools disabled. "
        "Install with: pip install langchain-mcp-adapters"
    )


def _servers_from_env() -> list[dict]:
    """Parse MCP_SERVERS env-var (JSON array) if present."""
    raw = os.getenv("MCP_SERVERS", "").strip()
    if not raw:
        return []
    try:
        servers = json.loads(raw)
        if not isinstance(servers, list):
            raise ValueError("MCP_SERVERS must be a JSON array")
        return servers
    except Exception as exc:
        log.error("Failed to parse MCP_SERVERS env-var: %s", exc)
        return []


def _normalise(servers: list[dict]) -> dict[str, dict]:
    """
    Convert the list of server configs into the dict format expected by
    MultiServerMCPClient:   { name: { transport, url | command, ... } }
    """
    result: dict[str, dict] = {}
    for srv in servers:
        if not srv.get("enabled", True):
            continue
        name      = srv.get("name") or f"mcp_{id(srv)}"
        transport = srv.get("transport", "sse")
        entry: dict[str, Any] = {"transport": transport}

        if transport == "stdio":
            entry["command"] = srv["command"][0]
            entry["args"]    = srv["command"][1:]
            entry["env"]     = srv.get("env", {})
        else:
            entry["url"]     = srv["url"]
            if srv.get("headers"):
                entry["headers"] = srv["headers"]

        entry["read_timeout_seconds"] = srv.get("timeout", 30)
        result[name] = entry
    return result


class MCPClientManager:
    """
    Thin async façade over MultiServerMCPClient.

    All public methods are safe to call even when MCP is not installed —
    they just return an empty list.
    """

    # ── Public API ────────────────────────────────────────────────────────────

    @classmethod
    async def get_tools(
        cls,
        runtime_servers: list[dict] | None = None,
    ) -> list:
        """
        Return LangChain-compatible tools from all configured MCP servers.

        Merges:
          1. Servers declared in MCP_SERVERS env-var
          2. Servers passed in via `runtime_servers` (from GraphState)

        Returns [] if no servers are configured or the library is missing.
        """
        if not _MCP_AVAILABLE:
            return []

        all_servers: list[dict] = _servers_from_env() + (runtime_servers or [])
        if not all_servers:
            return []

        server_map = _normalise(all_servers)
        if not server_map:
            return []

        try:
            return await cls._load(server_map)
        except Exception as exc:
            log.error("MCP tool load failed: %s", exc)
            return []

    @classmethod
    async def list_servers(
        cls,
        runtime_servers: list[dict] | None = None,
    ) -> list[str]:
        """Return names of all enabled servers (env + runtime)."""
        all_servers = _servers_from_env() + (runtime_servers or [])
        return [
            s.get("name", f"mcp_{i}")
            for i, s in enumerate(all_servers)
            if s.get("enabled", True)
        ]

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    async def _load(server_map: dict[str, dict]) -> list:
        """Connect to all servers in parallel and collect their tools."""
        names = list(server_map)
        log.info("Connecting to MCP servers: %s", names)

        async with MultiServerMCPClient(server_map) as client:
            tools = client.get_tools()

        log.info(
            "Loaded %d MCP tool(s) from %d server(s): %s",
            len(tools),
            len(server_map),
            [t.name for t in tools],
        )
        return tools


# ── Convenience: synchronous wrapper for non-async call sites ─────────────────

def get_mcp_tools_sync(runtime_servers: list[dict] | None = None) -> list:
    """Blocking wrapper — use only outside an existing event loop."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Inside an existing loop (e.g. Jupyter) — schedule as task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    MCPClientManager.get_tools(runtime_servers),
                )
                return future.result()
        return loop.run_until_complete(MCPClientManager.get_tools(runtime_servers))
    except Exception as exc:
        log.error("get_mcp_tools_sync failed: %s", exc)
        return []