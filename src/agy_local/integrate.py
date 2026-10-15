"""Frontier Coding Agent Integrator (`agy-local integrate`).

Automatically configures frontier coding agents (**Google Antigravity 2.0**, **Claude Code**,
**Cursor**, and **OpenAI Codex CLI**) to use `agy-local-mcp` as a local GPU co-processor &
AST verification harness:

1. Registers `agy-local-mcp` in the target agent's MCP configuration file (`mcp.json`,
   `claude_desktop_config.json` / `~/.claude.json`, `~/.cursor/mcp.json`, `~/.codex/config.json`).
2. Installs a dedicated **Co-Processor Routing Skill / Rule** (`agy-local-coprocessor`) so the
   frontier agent automatically knows *when* and *how* to offload:
   - `scout_codebase` (AST repo-map & surgical code slice extraction — saves 60-80% cloud input tokens)
   - `apply_smart_patch` (Self-healing fuzzy edit with AST syntax verification)
   - `verify_and_repair` (Local test/linter loop with on-device auto-repair)
   - `run_local_agent_task` (Delegate full coding subtasks to local Gemma 4 / Qwen / GLM-4)
"""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

COPROCESSOR_SKILL_MARKDOWN = """\
---
name: agy-local-coprocessor
description: Offload codebase scouting, AST repo-mapping, fuzzy self-healing file patches, and test-fix-verify loops to the local agy-local GPU co-processor MCP server (saving 60-80% frontier cloud tokens).
---

# agy-local Co-Processor & Verification Protocol

When the `agy-local` MCP server is connected, use its high-speed local tools to accelerate your workflow and conserve context window tokens:

1. **Zero-Token Codebase Scouting (`scout_codebase`)**:
   - **ALWAYS call `scout_codebase(workspace, query)` first** when exploring an unfamiliar repository or searching for classes, functions, or call sites.
   - It indexes the repository AST locally in milliseconds and returns ranked file paths, exact symbol signatures with `[StartLine-EndLine]` coordinates, and surgical code slices—eliminating 5+ turns of `list_dir` and `view_file`.
2. **Fault-Tolerant Self-Healing Edits (`apply_smart_patch`)**:
   - When a standard file edit fails due to line-number drift or indentation mismatch, call `apply_smart_patch(target_file, target_content, replacement_content, start_line, end_line)`.
   - It automatically heals off-by-N line ranges, normalizes indentation, applies fuzzy AST block matching, and validates Python syntax via `ast.parse()`.
3. **Local Test-Fix-Verify Loop (`verify_and_repair`)**:
   - After completing a code change, call `verify_and_repair(workspace, command="pytest -q")` to run tests locally.
   - If `auto_repair=true` is passed, the local on-device model (`Gemma 4 26B` / `Qwen 3.6`) will automatically diagnose and repair trivial lint/import/test failures locally before returning the final green status.
4. **Subagent Task Delegation (`run_local_agent_task`)**:
   - Offload routine unit-test generation, docstring writing, or multi-file boilerplate refactors to the local GPU model at zero cloud API cost.
"""


@dataclass
class IntegrationResult:
    """Represents the result of integrating `agy-local` into a target frontier coding agent."""

    target: str
    mcp_config_path: str
    skill_path: str | None
    binary_command: str
    status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resolve_mcp_command() -> str:
    user_bin = Path.home() / ".local" / "bin" / "agy-local-mcp"
    if user_bin.exists():
        return str(user_bin)
    which_bin = shutil.which("agy-local-mcp")
    if which_bin:
        return which_bin
    venv_bin = Path(sys.executable).parent / "agy-local-mcp"
    if venv_bin.exists():
        return str(venv_bin)
    return "agy-local-mcp"


def _upsert_mcp_json(config_path: Path, mcp_cmd: str, key_name: str = "mcpServers") -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {}
    if config_path.is_file():
        try:
            loaded = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
        except (OSError, json.JSONDecodeError):
            data = {}

    servers = data.get(key_name)
    if not isinstance(servers, dict):
        servers = {}
        data[key_name] = servers

    servers["agy-local"] = {
        "command": mcp_cmd,
        "args": [],
    }
    config_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def integrate_frontier_agent(
    target: str,
    *,
    home_dir: Path | None = None,
) -> list[IntegrationResult]:
    """Installs `agy-local-mcp` and co-processor routing skills into the specified frontier agent(s).

    Supported targets: `"antigravity"`, `"claude"`, `"cursor"`, `"codex"`, or `"all"`.
    """
    home = (home_dir or Path.home()).resolve()
    mcp_cmd = _resolve_mcp_command()
    targets = (
        ["antigravity", "claude", "cursor", "codex"]
        if target == "all"
        else [target.lower()]
    )
    results: list[IntegrationResult] = []

    for tgt in targets:
        if tgt == "antigravity":
            mcp_file = home / ".gemini" / "antigravity" / "mcp_config.json"
            _upsert_mcp_json(mcp_file, mcp_cmd)
            skill_file = (
                home
                / ".gemini"
                / "antigravity"
                / "skills"
                / "agy-local-coprocessor"
                / "SKILL.md"
            )
            skill_file.parent.mkdir(parents=True, exist_ok=True)
            skill_file.write_text(COPROCESSOR_SKILL_MARKDOWN, encoding="utf-8")
            results.append(
                IntegrationResult(
                    target="antigravity",
                    mcp_config_path=str(mcp_file),
                    skill_path=str(skill_file),
                    binary_command=mcp_cmd,
                    status="installed",
                )
            )
        elif tgt == "claude":
            mcp_file = home / ".claude" / "mcp_servers.json"
            _upsert_mcp_json(mcp_file, mcp_cmd)
            skill_file = home / ".claude" / "skills" / "agy-local-coprocessor" / "SKILL.md"
            skill_file.parent.mkdir(parents=True, exist_ok=True)
            skill_file.write_text(COPROCESSOR_SKILL_MARKDOWN, encoding="utf-8")
            results.append(
                IntegrationResult(
                    target="claude",
                    mcp_config_path=str(mcp_file),
                    skill_path=str(skill_file),
                    binary_command=mcp_cmd,
                    status="installed",
                )
            )
        elif tgt == "cursor":
            mcp_file = home / ".cursor" / "mcp.json"
            _upsert_mcp_json(mcp_file, mcp_cmd)
            rule_file = home / ".cursor" / "rules" / "agy-local-coprocessor.mdc"
            rule_file.parent.mkdir(parents=True, exist_ok=True)
            rule_file.write_text(COPROCESSOR_SKILL_MARKDOWN, encoding="utf-8")
            results.append(
                IntegrationResult(
                    target="cursor",
                    mcp_config_path=str(mcp_file),
                    skill_path=str(rule_file),
                    binary_command=mcp_cmd,
                    status="installed",
                )
            )
        elif tgt == "codex":
            mcp_file = home / ".codex" / "mcp.json"
            _upsert_mcp_json(mcp_file, mcp_cmd)
            skill_file = home / ".codex" / "instructions" / "agy-local-coprocessor.md"
            skill_file.parent.mkdir(parents=True, exist_ok=True)
            skill_file.write_text(COPROCESSOR_SKILL_MARKDOWN, encoding="utf-8")
            results.append(
                IntegrationResult(
                    target="codex",
                    mcp_config_path=str(mcp_file),
                    skill_path=str(skill_file),
                    binary_command=mcp_cmd,
                    status="installed",
                )
            )
        else:
            raise ValueError(
                f"Unsupported integration target '{tgt}'. Choose from: antigravity, claude, cursor, codex, all."
            )

    return results
