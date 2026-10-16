"""JSON-RPC 2.0 MCP Co-Processor Server (`kardashev-mcp`) for Frontier & Local Coding Agents.

Exposes 9 high-signal MCP tools so frontier agents (`Claude Code`, `Cursor`, `OpenAI Codex`,
`Antigravity 2.0`) can offload codebase scouting, fuzzy self-healing edits, test-fix-verify loops,
OpenCommons knowledge discovery, and full subagent tasks to local GPUs (`Gemma 4 26B`, `Qwen 3.6`, `GLM-4`):

1. `scout_codebase`: Fast AST symbol graph & surgical slice extractor (saves 60-80% frontier cloud input tokens).
2. `apply_smart_patch`: Fault-tolerant fuzzy search-and-replace (`heal_replace_file_args`) + immediate Python AST syntax check.
3. `verify_and_repair`: Runs a workspace verification command (`pytest`, `ruff`, etc.) locally and optionally uses the local agent to auto-repair failures.
4. `run_local_agent_task`: Spawns a local Kardashev Studio `Agent` with the engineered CoT + self-healing harness to execute a coding subtask at zero cloud API cost.
5. `list_local_models`: Lists available presets, discovered local checkpoints, and live localhost servers.
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from google.antigravity import Agent

from kardashev_cli.ast_map import scout_workspace
from kardashev_cli.builder import HarnessOptions, PolicyMode, build_agent_config
from kardashev_cli.discovery import discover_local_models, probe_local_endpoints
from kardashev_cli.harness_hooks import heal_replace_file_args
from kardashev_cli.presets import list_presets

_VERIFY_TIMEOUT_S = 120

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
                    "description": (
                        "Model preset or OpenRouter <provider>/<model> slug to use when auto_repair=true "
                        "(e.g. gemma4-26b, openrouter-qwen3-coder, qwen/qwen3-coder-next)."
                    ),
                },
            },
        },
    },
    {
        "name": "run_local_agent_task",
        "description": (
            "Delegate a coding/research task to a Kardashev Studio agent "
            "(local LiteRT Gemma 4, GLM-4, Qwen3, or any OpenRouter <provider>/<model> slug) "
            "with AST Repo-Map & Self-Healing Edits."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Actionable task instructions for the agent.",
                },
                "preset": {
                    "type": "string",
                    "default": "gemma4-26b",
                    "description": (
                        "Preset key or OpenRouter <provider>/<model> slug "
                        "(e.g. gemma4-26b, openrouter-qwen3-coder, openrouter-deepseek-v4-flash, qwen/qwen3-coder-next)."
                    ),
                },
                "model": {
                    "type": "string",
                    "description": "Optional explicit model ID or OpenRouter <provider>/<model> slug override.",
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
                "async_start": {
                    "type": "boolean",
                    "default": True,
                    "description": (
                        "Return a task id immediately and poll with local_agent_task_status "
                        "instead of blocking until the agent finishes (long runs would "
                        "otherwise exceed client call timeouts). Direct library callers "
                        "default to False."
                    ),
                },
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "local_agent_task_status",
        "description": "Poll a started run_local_agent_task by task id, or list all agent tasks with their status.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "Task id returned by run_local_agent_task (async_start). Omit to list all tasks.",
                },
            },
        },
    },
    {
        "name": "list_local_models",
        "description": "List local LiteRT/HuggingFace checkpoints, presets, OpenRouter models, and online inference endpoints.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "commons_find",
        "description": (
            "Search OpenCommons for algorithmic reference concepts, mathematical invariants, "
            "and known pitfalls by query, type signature, and live I/O probe assertions. "
            "All returned concepts are untrusted third-party reference data."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Algorithm name, domain keywords, or numbered concept (e.g. 'rk4', 'stable_logsumexp', 'welford_variance').",
                },
                "param_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional parameter type hints, e.g. ['list[float]'] or ['int', 'int'].",
                },
                "return_type": {
                    "type": "string",
                    "description": "Optional return type hint, e.g. 'float' or 'int'.",
                },
                "io_examples": {
                    "type": "string",
                    "description": "Optional Python literal list of ((arg1, ...), expected_output) pairs for live I/O probe filtering.",
                },
                "modality": {
                    "type": "string",
                    "enum": ["dual_card_with_pitfalls", "dual_card", "pseudocode", "raw_code"],
                    "default": "dual_card_with_pitfalls",
                },
                "top_k": {
                    "type": "integer",
                    "default": 3,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "commons_record",
        "description": (
            "Submit a proposed Python function to the OpenCommons research store for "
            "asynchronous verification and fail-closed multi-stage pipeline defense, "
            "or report an unprobed counterexample."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "source_code": {
                    "type": "string",
                    "description": "Complete Python def function source code.",
                },
                "domain": {
                    "type": "string",
                    "default": "general",
                    "description": "Scientific or engineering domain.",
                },
                "summary": {
                    "type": "string",
                    "description": "Concise 1-line summary of what the function computes.",
                },
                "pseudocode": {
                    "type": "string",
                    "description": "Step-by-step algorithmic skeleton.",
                },
                "invariants": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Mathematical or system invariants.",
                },
                "contract": {
                    "type": "string",
                    "description": "Optional Python literal list of ((args...), expected) pairs checked by the correctness engine.",
                },
                "teleological_intent": {
                    "type": "string",
                    "description": "Physical conservation law or scientific rationale.",
                },
                "dimensional_units": {
                    "type": "string",
                    "description": "Dimensional unit signature.",
                },
                "require_human_review": {
                    "type": "boolean",
                    "default": False,
                    "description": "If true, stage in ~/.opencommons/pending_queue.json for human `/commons review`.",
                },
                "counterexample_block_id": {
                    "type": "string",
                    "description": "Optional block_id to quarantine if reporting an unprobed (K+1)-th input failure.",
                },
                "counterexample_args": {
                    "type": "string",
                    "description": "Optional Python tuple literal of failing arguments for counterexample ratcheting.",
                },
                "counterexample_expected": {
                    "type": "string",
                    "description": "Python literal of the expected correct output for counterexample_args; required to quarantine.",
                },
                "sync_to_hub": {
                    "type": "boolean",
                    "default": True,
                    "description": "If true and the block passes 5-gate peer review, automatically gossip-sync with the remote OpenCommons Hub.",
                },
            },
        },
    },
    {
        "name": "commons_sync",
        "description": (
            "Synchronize the local OpenCommons store with the remote OpenCommons Hub "
            "(pull latest concepts, push local discoveries, or run bidirectional "
            "delta sync)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["bidirectional", "pull", "push"],
                    "default": "bidirectional",
                    "description": "Sync direction: 'bidirectional', 'pull', or 'push'.",
                },
                "hub_url": {
                    "type": "string",
                    "description": "Optional remote OpenCommons Hub URL.",
                },
                "force": {
                    "type": "boolean",
                    "default": False,
                    "description": "If true, bypass ETag cache check on pull.",
                },
            },
        },
    },
    {
        "name": "commons_status",
        "description": (
            "Return live OpenCommons store statistics, active Concept "
            "Equivalence Classes (CIDs), quarantined counterexamples, and pending proposals."
        ),
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


def _check_python_syntax_text(source: str, display_name: str) -> str | None:
    """In-memory Python AST syntax check against `source` (no file I/O)."""
    try:
        ast.parse(source, filename=display_name)
        return None
    except SyntaxError as exc:
        return f"SyntaxError in {Path(display_name).name} at line {exc.lineno}, col {exc.offset}: {exc.msg}"


def _find_occurrences(text: str, target: str) -> list[tuple[int, int, int]]:
    """Returns every `(char_index, start_line, end_line)` occurrence of `target` within `text`."""
    occurrences: list[tuple[int, int, int]] = []
    target_newlines = target.count("\n")
    search_from = 0
    while True:
        idx = text.find(target, search_from)
        if idx == -1:
            break
        start_line = text.count("\n", 0, idx) + 1
        end_line = start_line + target_newlines
        occurrences.append((idx, start_line, end_line))
        search_from = idx + 1
    return occurrences


def _select_occurrence(
    occurrences: list[tuple[int, int, int]],
    hint_start: int,
    hint_end: int,
    hint_provided: bool,
) -> tuple[tuple[int, int, int] | None, list[tuple[int, int]]]:
    """Picks the occurrence whose line span best matches the `[hint_start, hint_end]` hint.

    Returns `(chosen_occurrence, candidate_spans)`. `chosen_occurrence` is `None` when the hint
    is missing/default or two or more occurrences are equidistant, in which case
    `candidate_spans` lists every ambiguous candidate's `(start_line, end_line)`.
    """
    if not hint_provided:
        return (None, [(o[1], o[2]) for o in occurrences])

    def _distance(occ: tuple[int, int, int]) -> int:
        _idx, o_start, o_end = occ
        if o_end >= hint_start and o_start <= hint_end:
            return 0  # overlaps the hint window
        if o_start > hint_end:
            return o_start - hint_end
        return hint_start - o_end

    best_distance = min(_distance(o) for o in occurrences)
    tied = [o for o in occurrences if _distance(o) == best_distance]
    if len(tied) > 1:
        return (None, [(o[1], o[2]) for o in occurrences])
    return (tied[0], [])


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
    occurrences = _find_occurrences(original_text, eff_target)
    if not occurrences:
        return {
            "applied": False,
            "file": str(fpath),
            "error": "TargetContent could not be matched even after fuzzy/indentation healing.",
        }

    if len(occurrences) == 1:
        chosen = occurrences[0]
    else:
        hint_provided = not (start_line == 1 and end_line == 999999)
        chosen, candidate_spans = _select_occurrence(occurrences, start_line, end_line, hint_provided)
        if chosen is None:
            return {
                "applied": False,
                "file": str(fpath),
                "error": (
                    f"TargetContent matches {len(occurrences)} locations and start_line/end_line "
                    "does not unambiguously pick one. Pass a more specific start_line/end_line "
                    "hint or a more unique TargetContent."
                ),
                "candidate_line_spans": [f"{s}-{e}" for s, e in candidate_spans],
            }

    char_idx, actual_start_line, actual_end_line = chosen
    updated_text = original_text[:char_idx] + eff_replacement + original_text[char_idx + len(eff_target) :]

    syntax_err = None
    if str(fpath).endswith(".py"):
        syntax_err = _check_python_syntax_text(updated_text, str(fpath))
    if syntax_err:
        return {
            "applied": False,
            "file": str(fpath),
            "heal_applied": heal_reason,
            "syntax_error": syntax_err,
        }

    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=str(fpath.parent), prefix=f".{fpath.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as tmp_file:
            tmp_file.write(updated_text)
        os.replace(tmp_path, fpath)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise

    return {
        "applied": True,
        "file": str(fpath),
        "heal_applied": heal_reason or "exact_window_match",
        "start_line": actual_start_line,
        "end_line": actual_end_line,
        "syntax_valid": True,
    }


class _VerifyRunResult:
    """Lightweight holder for a capped, process-group-safe verification run."""

    __slots__ = ("returncode", "stderr", "stdout", "timed_out")

    def __init__(self, returncode: int | None, stdout: str, stderr: str, timed_out: bool) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.timed_out = timed_out


def _run_verify_command(
    command: str, ws: str, timeout_s: int | None = None
) -> _VerifyRunResult:
    """Runs `command` in its own process group so a timeout can kill the whole tree.

    Returns a structured `_VerifyRunResult` instead of raising `subprocess.TimeoutExpired`.
    `timeout_s` defaults to the module-level `_VERIFY_TIMEOUT_S`, resolved at call time (not
    bound as a mutable default argument) so callers/tests can override it dynamically.
    """
    if timeout_s is None:
        timeout_s = _VERIFY_TIMEOUT_S
    proc = subprocess.Popen(
        command,
        shell=True,
        cwd=ws,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
        return _VerifyRunResult(proc.returncode, stdout or "", stderr or "", False)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        stdout, stderr = proc.communicate()
        return _VerifyRunResult(None, stdout or "", stderr or "", True)


def verify_and_repair(
    command: str = "pytest -q",
    workspace: str | None = None,
    auto_repair: bool = False,
    preset: str = "gemma4-26b",
) -> dict[str, Any]:
    """MCP tool 3: `verify_and_repair` — executes verification command and compresses output."""
    ws = os.path.abspath(os.path.expanduser(workspace)) if workspace else os.getcwd()
    run = _run_verify_command(command, ws)
    combined = ((run.stdout or "") + "\n" + (run.stderr or "")).strip()
    lines = combined.splitlines()
    tail_summary = "\n".join(lines[-35:]) if len(lines) > 35 else combined

    if run.timed_out:
        return {
            "passed": False,
            "timed_out": True,
            "exit_code": None,
            "command": command,
            "failure_tail": tail_summary,
        }

    if run.returncode == 0:
        return {
            "passed": True,
            "timed_out": False,
            "exit_code": 0,
            "command": command,
            "summary": lines[-1] if lines else "passed",
            "output_tail": "\n".join(lines[-10:]),
        }

    if not auto_repair:
        return {
            "passed": False,
            "timed_out": False,
            "exit_code": run.returncode,
            "command": command,
            "failure_tail": tail_summary,
        }

    repair_prompt = (
        f"The verification command `{command}` failed in `{ws}` with exit code {run.returncode}.\n"
        f"Failure output:\n```\n{tail_summary}\n```\n"
        f"Diagnose the root cause, apply a minimal fix using `replace_file_content`, and re-run `{command}`."
    )
    agent_res = run_local_agent_task(
        prompt=repair_prompt,
        preset=preset,
        workspace=ws,
        policy="allow-all",
    )
    recheck = _run_verify_command(command, ws)
    if recheck.timed_out:
        return {
            "passed": False,
            "timed_out": True,
            "initial_exit_code": run.returncode,
            "final_exit_code": None,
            "auto_repaired": False,
            "repair_tool_calls": agent_res.get("tool_calls", []),
            "repair_summary": agent_res.get("output", ""),
        }
    re_combined = ((recheck.stdout or "") + "\n" + (recheck.stderr or "")).strip()
    re_lines = re_combined.splitlines()
    return {
        "passed": recheck.returncode == 0,
        "timed_out": False,
        "initial_exit_code": run.returncode,
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
    model: str | None = None,
) -> dict[str, Any]:
    ws = os.path.abspath(os.path.expanduser(workspace)) if workspace else os.getcwd()
    effective_preset = model if (model and preset == "gemma4-26b") else preset
    opts = HarnessOptions(
        preset=effective_preset,
        model=model,
        workspaces=[ws],
        policy_mode=PolicyMode(policy),
    )
    cfg = build_agent_config(opts)
    tool_calls_executed: list[str] = []
    chunks: list[str] = []

    from kardashev_cli.ast_map import build_turn_zero_context
    from kardashev_cli.harness_hooks import get_harness_telemetry

    effective_prompt = prompt
    if opts.enable_ast_repo_map:
        ctx_block, _n_files, _n_syms, _n_slices = build_turn_zero_context(ws, prompt)
        if ctx_block:
            get_harness_telemetry().ast_maps_injected += 1
            effective_prompt = f"{ctx_block}\n\n[User Task]:\n{prompt}"

    async with Agent(config=cfg) as agent:
        response = await agent.chat(effective_prompt)

        async def _collect_tools() -> None:
            async for call in response.tool_calls:
                tool_calls_executed.append(call.name)

        tool_task = asyncio.create_task(_collect_tools())
        async for token in response:
            chunks.append(token)
        await asyncio.gather(tool_task, return_exceptions=True)

    return {
        "preset": effective_preset,
        "model": getattr(cfg, "model", None) or getattr(cfg, "model_path", None),
        "workspace": ws,
        "tool_calls": tool_calls_executed,
        "output": "".join(chunks).strip(),
    }


_AGENT_TASKS: dict[str, dict[str, Any]] = {}
_AGENT_TASKS_LOCK = threading.Lock()


def _start_agent_task(
    prompt: str,
    preset: str = "gemma4-26b",
    workspace: str | None = None,
    policy: str = "allow-all",
    model: str | None = None,
) -> dict[str, Any]:
    """Starts `run_local_agent_task` on a background thread and returns immediately.

    The MCP tool previously blocked for the whole agent run, so any caller with
    a client-side timeout lost the result (a long task exited the client's call
    window even though the server was still working). Started tasks are polled
    with the `local_agent_task_status` tool instead.
    """
    task_id = "lat_" + uuid.uuid4().hex[:12]
    entry: dict[str, Any] = {
        "task_id": task_id,
        "status": "running",
        "started_at": time.time(),
        "preset": preset,
        "model": model,
        "workspace": workspace,
        "result": None,
        "error": None,
    }
    with _AGENT_TASKS_LOCK:
        _AGENT_TASKS[task_id] = entry

    def _worker() -> None:
        try:
            outcome = asyncio.run(
                _execute_local_agent_async(
                    prompt=prompt,
                    preset=preset,
                    workspace=workspace,
                    policy=policy,
                    model=model,
                )
            )
            entry["status"] = "succeeded"
            entry["result"] = outcome
        except BaseException as exc:  # noqa: BLE001 - daemon thread boundary
            entry["status"] = "failed"
            entry["error"] = f"{type(exc).__name__}: {exc}"

    threading.Thread(target=_worker, name=task_id, daemon=True).start()
    return {
        "task_id": task_id,
        "status": "started",
        "note": "poll with local_agent_task_status; results survive caller timeouts",
    }


def run_local_agent_task(
    prompt: str,
    preset: str = "gemma4-26b",
    workspace: str | None = None,
    policy: str = "allow-all",
    model: str | None = None,
    async_start: bool = False,
) -> dict[str, Any]:
    """Entrypoint for MCP tool 4: `run_local_agent_task`.

    With `async_start=False` (the default for direct library callers) this blocks
    until the agent finishes, as before. The MCP dispatch layer passes
    `async_start=True` so tool calls return a task id immediately instead of
    exceeding client call timeouts.
    """
    if async_start:
        return _start_agent_task(
            prompt=prompt,
            preset=preset,
            workspace=workspace,
            policy=policy,
            model=model,
        )
    return asyncio.run(
        _execute_local_agent_async(
            prompt=prompt,
            preset=preset,
            workspace=workspace,
            policy=policy,
            model=model,
        )
    )


def agent_task_status(task_id: str | None = None) -> dict[str, Any]:
    """Synchronous entrypoint for MCP tool `local_agent_task_status`.

    `task_id` returns one task; omitting it returns every task this server
    process knows about (newest last).
    """
    with _AGENT_TASKS_LOCK:
        snapshot = {k: dict(v) for k, v in _AGENT_TASKS.items()}
    if task_id is not None:
        entry = snapshot.get(task_id)
        if entry is None:
            return {"found": False, "task_id": task_id}
        return {"found": True, **entry}
    tasks = list(snapshot.values())
    return {
        "found": bool(tasks),
        "count": len(tasks),
        "statuses": {t["status"] for t in tasks} if tasks else set(),
        "tasks": tasks,
    }


def list_local_models() -> dict[str, Any]:
    """Synchronous entrypoint for MCP tool 5: `list_local_models`."""
    from kardashev_cli.discovery import probe_openrouter_endpoint

    models = discover_local_models()
    endpoints = probe_local_endpoints()
    presets = list_presets()
    or_status = probe_openrouter_endpoint()
    return {
        "presets": [p.key for p in presets],
        "openrouter_configured": or_status.reachable,
        "openrouter_models": or_status.models,
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
        from kardashev_cli.opencommons import _contribution_instructions

        result: dict[str, Any] = {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "kardashev-mcp", "version": "0.2.0"},
        }
        instructions = _contribution_instructions()
        if instructions:
            result["instructions"] = instructions
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

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
                args.setdefault("async_start", True)
                res = run_local_agent_task(**args)
            elif tool_name in ("local_agent_task_status", "agent_task_status"):
                res = agent_task_status(task_id=args.get("task_id"))
            elif tool_name == "list_local_models":
                res = list_local_models()
            elif tool_name in ("commons_find", "find"):
                from kardashev_cli.opencommons import execute_commons_find

                res = execute_commons_find(args)
            elif tool_name in ("commons_record", "record"):
                from kardashev_cli.opencommons import execute_commons_record

                res = execute_commons_record(args)
            elif tool_name in ("commons_sync", "sync"):
                from kardashev_cli.opencommons import execute_commons_sync

                res = execute_commons_sync(args)
            elif tool_name in ("commons_status", "status"):
                from kardashev_cli.opencommons import execute_commons_status

                res = execute_commons_status()
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
