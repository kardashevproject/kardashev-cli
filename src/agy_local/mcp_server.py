"""JSON-RPC 2.0 MCP Co-Processor Server (`agy-local-mcp`) for Frontier & Local Coding Agents.

Exposes 5 high-signal MCP tools so frontier agents (`Antigravity 2.0`, `Claude Code`, `Cursor`,
`OpenAI Codex`) can offload codebase scouting, fuzzy self-healing edits, test-fix-verify loops,
and full subagent tasks to local GPUs (`Gemma 4 26B`, `Qwen 3.6`, `GLM-4`):

1. `scout_codebase`: Fast AST symbol graph & surgical slice extractor (saves 60-80% frontier cloud input tokens).
2. `apply_smart_patch`: Fault-tolerant fuzzy search-and-replace (`heal_replace_file_args`) + immediate Python AST syntax check.
3. `verify_and_repair`: Runs a workspace verification command (`pytest`, `ruff`, etc.) locally and optionally uses the local agent to auto-repair failures.
4. `run_local_agent_task`: Spawns a local Antigravity `Agent` with the engineered CoT + self-healing harness to execute a coding subtask at zero cloud API cost.
5. `list_local_models`: Lists available presets, discovered local checkpoints, and live localhost servers.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from google.antigravity import Agent

from agy_local.ast_map import scout_workspace
from agy_local.builder import HarnessOptions, PolicyMode, build_agent_config
from agy_local.discovery import discover_local_models, probe_local_endpoints
from agy_local.harness_hooks import check_python_syntax, heal_replace_file_args
from agy_local.presets import list_presets

MCP_TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "name": "scout_codebase",
        "description": (
            "Index workspace AST symbols and return a ranked repo map plus surgical code slices "
            "with exact [StartLine-EndLine] spans for a query. Call this FIRST to save 60-80% cloud tokens."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Symbol names, file keywords, or natural-language task query.",
                },
                "workspace": {
                    "type": "string",
                    "description": "Workspace directory path (defaults to current working directory).",
                },
                "max_slices": {
                    "type": "integer",
                    "default": 5,
                    "description": "Maximum number of surgical code slices to return (default: 5).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "apply_smart_patch",
        "description": (
            "Apply a fault-tolerant, self-healing file edit. Automatically repairs off-by-N line "
            "ranges, normalizes indentation drift, performs fuzzy block matching (>=88%), and "
            "validates Python AST syntax."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "target_file": {
                    "type": "string",
                    "description": "Absolute or workspace-relative path to the file to edit.",
                },
                "target_content": {
                    "type": "string",
                    "description": "Target lines of code to replace.",
                },
                "replacement_content": {
                    "type": "string",
                    "description": "Replacement lines of code.",
                },
                "start_line": {
                    "type": "integer",
                    "default": 1,
                    "description": "Approximate 1-indexed starting line number.",
                },
                "end_line": {
                    "type": "integer",
                    "default": 999999,
                    "description": "Approximate 1-indexed ending line number.",
                },
            },
            "required": ["target_file", "target_content", "replacement_content"],
        },
    },
    {
        "name": "verify_and_repair",
        "description": (
            "Run a test or linter command (e.g. 'pytest -q && ruff check') locally. Converts noisy "
            "stdout into a compact structured report, and if auto_repair=true, runs the local GPU "
            "agent to fix failures automatically."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "default": "pytest -q",
                    "description": "Verification shell command to execute in the workspace.",
                },
                "workspace": {
                    "type": "string",
                    "description": "Workspace directory path (defaults to current working directory).",
                },
                "auto_repair": {
                    "type": "boolean",
                    "default": False,
                    "description": "If true and the command fails, invoke the local agent to repair the failure.",
                },
                "preset": {
                    "type": "string",
                    "default": "gemma4-26b",
                    "description": "Local model preset to use when auto_repair=true.",
                },
            },
        },
    },
    {
        "name": "run_local_agent_task",
        "description": (
            "Delegate a coding/research task to a local on-device Antigravity agent "
            "(LiteRT Gemma 4, GLM-4, Qwen3) with AST Repo-Map & Self-Healing Edits at zero API cost."
        ),
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


def scout_codebase(
    query: str,
    workspace: str | None = None,
    max_slices: int = 5,
) -> dict[str, Any]:
    """MCP tool 1: `scout_codebase` — fast AST symbol graph and surgical slice extractor."""
    ws = os.path.abspath(os.path.expanduser(workspace)) if workspace else os.getcwd()
    res = scout_workspace(ws, query, max_slices=max_slices)
    return res.to_dict()


def apply_smart_patch(
    target_file: str,
    target_content: str,
    replacement_content: str,
    start_line: int = 1,
    end_line: int = 999999,
) -> dict[str, Any]:
    """MCP tool 2: `apply_smart_patch` — self-healing fuzzy edit + Python AST syntax check."""
    fpath = Path(os.path.expanduser(target_file)).resolve()
    if not fpath.is_file():
        return {"applied": False, "error": f"File not found: {fpath}"}

    raw_args = {
        "TargetFile": str(fpath),
        "TargetContent": target_content,
        "ReplacementContent": replacement_content,
        "StartLine": start_line,
        "EndLine": end_line,
    }
    healed_args, heal_reason = heal_replace_file_args(raw_args)
    effective = healed_args if healed_args is not None else raw_args
    eff_target = str(effective["TargetContent"])
    eff_replacement = str(effective["ReplacementContent"])

    original_text = fpath.read_text(encoding="utf-8")
    if eff_target not in original_text:
        return {
            "applied": False,
            "file": str(fpath),
            "error": "TargetContent could not be matched even after fuzzy/indentation healing.",
        }

    updated_text = original_text.replace(eff_target, eff_replacement, 1)
    fpath.write_text(updated_text, encoding="utf-8")

    syntax_err = check_python_syntax(str(fpath))
    if syntax_err:
        # Rollback broken syntax edit automatically
        fpath.write_text(original_text, encoding="utf-8")
        return {
            "applied": False,
            "rolled_back": True,
            "file": str(fpath),
            "heal_applied": heal_reason,
            "syntax_error": syntax_err,
        }

    return {
        "applied": True,
        "file": str(fpath),
        "heal_applied": heal_reason or "exact_window_match",
        "start_line": effective["StartLine"],
        "end_line": effective["EndLine"],
        "syntax_valid": True,
    }


def verify_and_repair(
    command: str = "pytest -q",
    workspace: str | None = None,
    auto_repair: bool = False,
    preset: str = "gemma4-26b",
) -> dict[str, Any]:
    """MCP tool 3: `verify_and_repair` — executes verification command and compresses output."""
    ws = os.path.abspath(os.path.expanduser(workspace)) if workspace else os.getcwd()
    proc = subprocess.run(
        command,
        shell=True,
        cwd=ws,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    combined = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
    lines = combined.splitlines()
    tail_summary = "\n".join(lines[-35:]) if len(lines) > 35 else combined

    if proc.returncode == 0:
        return {
            "passed": True,
            "exit_code": 0,
            "command": command,
            "summary": lines[-1] if lines else "passed",
            "output_tail": "\n".join(lines[-10:]),
        }

    if not auto_repair:
        return {
            "passed": False,
            "exit_code": proc.returncode,
            "command": command,
            "failure_tail": tail_summary,
        }

    repair_prompt = (
        f"The verification command `{command}` failed in `{ws}` with exit code {proc.returncode}.\n"
        f"Failure output:\n```\n{tail_summary}\n```\n"
        f"Diagnose the root cause, apply a minimal fix using `replace_file_content`, and re-run `{command}`."
    )
    agent_res = run_local_agent_task(
        prompt=repair_prompt,
        preset=preset,
        workspace=ws,
        policy="allow-all",
    )
    recheck = subprocess.run(
        command,
        shell=True,
        cwd=ws,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    re_lines = ((recheck.stdout or "") + "\n" + (recheck.stderr or "")).strip().splitlines()
    return {
        "passed": recheck.returncode == 0,
        "initial_exit_code": proc.returncode,
        "final_exit_code": recheck.returncode,
        "auto_repaired": recheck.returncode == 0,
        "repair_tool_calls": agent_res.get("tool_calls", []),
        "repair_summary": agent_res.get("output", ""),
        "output_tail": "\n".join(re_lines[-15:]),
    }


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
    """Synchronous entrypoint for MCP tool 4: `run_local_agent_task`."""
    return asyncio.run(
        _execute_local_agent_async(
            prompt=prompt,
            preset=preset,
            workspace=workspace,
            policy=policy,
        )
    )


def list_local_models() -> dict[str, Any]:
    """Synchronous entrypoint for MCP tool 5: `list_local_models`."""
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
            if tool_name == "scout_codebase":
                res = scout_codebase(**args)
            elif tool_name == "apply_smart_patch":
                res = apply_smart_patch(**args)
            elif tool_name == "verify_and_repair":
                res = verify_and_repair(**args)
            elif tool_name == "run_local_agent_task":
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
