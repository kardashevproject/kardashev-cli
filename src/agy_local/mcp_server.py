"""Ultra-compact JSON-RPC 2.0 MCP Server (`agy-local-mcp`) for delegating tasks to local models.

Exposes 2 lightweight MCP tools (<250 tokens schema overhead):
1. `run_local_agent_task`: Spawns a local Antigravity `Agent` (LiteRT on-device or local OpenAI/vLLM/Ollama)
   with full workspace coding tools (`view_file`, `replace_file_content`, `write_to_file`, `run_command`,
   `grep`, `list_dir`) to execute a sub-task at zero cloud API cost.
2. `list_local_models`: Lists available presets, discovered local checkpoints, and live localhost servers.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any

from google.antigravity import Agent

from agy_local.builder import HarnessOptions, PolicyMode, build_agent_config
from agy_local.discovery import discover_local_models, probe_local_endpoints
from agy_local.presets import list_presets

MCP_TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "name": "run_local_agent_task",
        "description": "Delegate a coding/research task to a local on-device Antigravity agent (LiteRT Gemma 4, GLM-4, Qwen3) at zero API cost.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Actionable task instructions for the local agent.",
                },
                "preset": {
                    "type": "string",
                    "default": "gemma4-26b",
                    "description": "Preset key (e.g. gemma4-26b, glm4-9b-ollama, glm4-vllm, qwen3.6-35b-vllm).",
                },
                "workspace": {
                    "type": "string",
                    "description": "Workspace directory path (defaults to current working directory).",
                },
                "policy": {
                    "type": "string",
                    "enum": ["allow-all", "sandbox", "ask"],
                    "default": "allow-all",
                },
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "list_local_models",
        "description": "List local LiteRT/HuggingFace checkpoints, presets, and online localhost inference servers.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
]


async def _execute_local_agent_async(
    prompt: str,
    preset: str = "gemma4-26b",
    workspace: str | None = None,
    policy: str = "allow-all",
) -> dict[str, Any]:
    ws = os.path.abspath(os.path.expanduser(workspace)) if workspace else os.getcwd()
    opts = HarnessOptions(
        preset=preset,
        workspaces=[ws],
        policy_mode=PolicyMode(policy),
    )
    cfg = build_agent_config(opts)
    tool_calls_executed: list[str] = []
    chunks: list[str] = []

    async with Agent(config=cfg) as agent:
        response = await agent.chat(prompt)

        async def _collect_tools() -> None:
            async for call in response.tool_calls:
                tool_calls_executed.append(call.name)

        tool_task = asyncio.create_task(_collect_tools())
        async for token in response:
            chunks.append(token)
        await asyncio.gather(tool_task, return_exceptions=True)

    return {
        "preset": preset,
        "workspace": ws,
        "tool_calls": tool_calls_executed,
        "output": "".join(chunks).strip(),
    }


def run_local_agent_task(
    prompt: str,
    preset: str = "gemma4-26b",
    workspace: str | None = None,
    policy: str = "allow-all",
) -> dict[str, Any]:
    """Synchronous entrypoint for MCP tool 1: `run_local_agent_task`."""
    return asyncio.run(
        _execute_local_agent_async(
            prompt=prompt,
            preset=preset,
            workspace=workspace,
            policy=policy,
        )
    )


def list_local_models() -> dict[str, Any]:
    """Synchronous entrypoint for MCP tool 2: `list_local_models`."""
    models = discover_local_models()
    endpoints = probe_local_endpoints()
    presets = list_presets()
    return {
        "presets": [p.key for p in presets],
        "discovered_checkpoints": [
            {
                "name": m.name,
                "backend": m.backend.value,
                "size_gb": m.size_gb,
                "path": m.path,
            }
            for m in models
        ],
        "online_endpoints": [
            {"label": e.label, "base_url": e.base_url, "models": e.models}
            for e in endpoints
            if e.reachable
        ],
    }


def handle_mcp_request(request: dict[str, Any]) -> dict[str, Any] | None:
    """Processes a single JSON-RPC 2.0 MCP request."""
    method = request.get("method", "")
    req_id = request.get("id")
    params = request.get("params") or {}

    if method.startswith("notifications/"):
        return None

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "agy-local", "version": "0.1.0"},
            },
        }

    if method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": MCP_TOOLS_SCHEMA},
        }

    if method == "tools/call":
        tool_name = params.get("name")
        args = params.get("arguments") or {}
        try:
            if tool_name == "run_local_agent_task":
                res = run_local_agent_task(**args)
            elif tool_name == "list_local_models":
                res = list_local_models()
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
                }
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(res)}],
                    "structuredContent": res,
                    "isError": False,
                },
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": f"Error: {exc}"}],
                    "isError": True,
                },
            }

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def serve_stdio() -> None:
    """Runs the JSON-RPC 2.0 MCP server over standard input/output."""
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            err_resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {exc}"},
            }
            sys.stdout.write(json.dumps(err_resp) + "\n")
            sys.stdout.flush()
            continue

        resp = handle_mcp_request(req)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


def main() -> None:
    serve_stdio()


if __name__ == "__main__":
    main()
