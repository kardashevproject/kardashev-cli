"""Frontier Coding Agent Integrator (`kardashev integrate`).

Automatically configures external coding agents (**Claude Code**, **Cursor**, **OpenAI Codex CLI**,
and **Google Antigravity**) to use `kardashev-mcp` and `opencommons-mcp` as a local GPU co-processor &
OpenCommons research engine:

1. Registers `kardashev-cli` (`kardashev-mcp`) and `opencommons` (`opencommons-mcp`) in the target
   agent's genuine configuration file:
   - Claude Code: `~/.claude.json` (`mcpServers` block)
   - OpenAI Codex CLI: `~/.codex/config.toml` (`[mcp_servers.*]` tables)
   - Google Antigravity: `~/.gemini/config/mcp_config.json` (`mcpServers` block)
   - Cursor: `~/.cursor/mcp.json` (`mcpServers` block)
2. Installs a dedicated **Co-Processor Routing Skill / Rule** (`kardashev-coprocessor`) across all targets:
   - Claude Code: `~/.claude/skills/kardashev-coprocessor/SKILL.md`
   - Codex: `~/.codex/skills/kardashev-coprocessor/SKILL.md` & `~/.agents/skills/kardashev-coprocessor/SKILL.md`
   - Antigravity: `~/.gemini/config/skills/kardashev-coprocessor/SKILL.md`
   - Cursor: `~/.cursor/rules/kardashev-coprocessor.mdc`
3. Guarantees safety and hygiene:
   - Atomic `.bak` backup creation before modifying any existing configuration file.
   - Atomic writes via tempfile replacement to prevent partial writes.
   - Purges deprecated legacy `agy-local` MCP entries and `agy-local-coprocessor` skills across all targets.
   - Live health verification (`kardashev integrate --check`) with stdio JSON-RPC handshakes.
"""

from __future__ import annotations

import json
import select
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import tomllib

COPROCESSOR_SKILL_MARKDOWN = """\
---
name: kardashev-coprocessor
description: Offload codebase scouting, AST repo-mapping, fuzzy self-healing file patches, OpenCommons concept retrieval/contribution, and test-fix-verify loops to the local Kardashev CLI MCP server (saving 60-80% cloud tokens).
---

# Kardashev Co-Processor & OpenCommons Protocol

When the `kardashev-cli` or `opencommons` MCP server is connected, use its high-speed local tools to accelerate your workflow and conserve context window tokens:

1. **Zero-Token Codebase Scouting (`scout_codebase`)**:
   - **ALWAYS call `scout_codebase(workspace, query)` first** when exploring an unfamiliar repository or searching for classes, functions, or call sites.
   - It indexes the repository AST locally in milliseconds and returns ranked file paths, exact symbol signatures with `[StartLine-EndLine]` coordinates, and surgical code slices—eliminating 5+ turns of directory listing and file reads.
2. **Fault-Tolerant Self-Healing Edits (`apply_smart_patch`)**:
   - When a standard file edit fails due to line-number drift or indentation mismatch, call `apply_smart_patch(target_file, target_content, replacement_content, start_line, end_line)`.
   - It automatically heals off-by-N line ranges, normalizes indentation, applies fuzzy AST block matching, and validates Python syntax via `ast.parse()`.
3. **Local Test-Fix-Verify Loop (`verify_and_repair`)**:
   - After completing a code change, call `verify_and_repair(workspace, command="pytest -q")` to run tests locally.
   - If `auto_repair=true` is passed, the local on-device model (`Gemma 4 26B` / `Qwen 3.6`) will automatically diagnose and repair trivial lint/import/test failures locally before returning the final green status.
4. **Subagent Task Delegation (`run_local_agent_task`)**:
   - Offload routine unit-test generation, docstring writing, or multi-file boilerplate refactors to the local GPU model at zero cloud API cost.
5. **OpenCommons Research Retrieval & Contribution (`commons_find`, `commons_record`, `commons_sync`)**:
   - Call `commons_find(query)` (or `find(query)`) when implementing numerical, scientific, cryptographic, or algorithmic functions to retrieve verified Dual Concept Cards, invariants, and Refuted Pitfalls in `<5ms`.
   - Call `commons_record(source_code, domain, summary, invariants)` (or `record(...)`) when you synthesize a reusable, verified Python function (or report a `(K+1)`-th counterexample via `counterexample_block_id` + `counterexample_args`) to participate in the OpenCommons research loop.
"""


@dataclass
class IntegrationResult:
    """Represents the result of integrating `kardashev-cli` into a target coding agent."""

    target: str
    mcp_config_path: str
    skill_path: str | None
    binary_command: str
    status: str
    backup_path: str | None = None
    skill_paths: list[str] = field(default_factory=list)
    purged_items: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HandshakeResult:
    """Result of testing an MCP server via stdio JSON-RPC handshake."""

    command: str
    success: bool
    server_name: str | None = None
    server_version: str | None = None
    tools: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TargetCheckResult:
    """Health check outcome for an individual frontier coding agent platform."""

    target: str
    config_file: str
    config_exists: bool
    kardashev_configured: bool
    opencommons_configured: bool
    legacy_purged: bool
    skill_file: str | None
    skill_exists: bool
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return (
            self.config_exists
            and self.kardashev_configured
            and self.opencommons_configured
            and self.legacy_purged
            and (self.skill_exists if self.skill_file else True)
            and not self.errors
        )

    def to_dict(self) -> dict[str, Any]:
        res = asdict(self)
        res["ok"] = self.ok
        return res


@dataclass
class CheckSummary:
    """Aggregated health check summary for all checked platforms and MCP servers."""

    targets: list[TargetCheckResult]
    kardashev_handshake: HandshakeResult
    opencommons_handshake: HandshakeResult

    @property
    def all_ok(self) -> bool:
        return (
            all(t.ok for t in self.targets)
            and self.kardashev_handshake.success
            and self.opencommons_handshake.success
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "all_ok": self.all_ok,
            "targets": [t.to_dict() for t in self.targets],
            "kardashev_handshake": self.kardashev_handshake.to_dict(),
            "opencommons_handshake": self.opencommons_handshake.to_dict(),
        }

    def format_table(self) -> str:
        lines: list[str] = []
        lines.append("=" * 88)
        lines.append("          Kardashev Frontier Agent Integration & MCP Health Check")
        lines.append("=" * 88)
        header = f"{'Platform':<13} {'Config File':<34} {'Config':<8} {'Servers':<9} {'Skill':<8} {'Legacy':<8} {'Status':<6}"
        lines.append(header)
        lines.append("-" * 88)

        for t in self.targets:
            cfg_disp = t.config_file
            if len(cfg_disp) > 33:
                cfg_disp = "..." + cfg_disp[-30:]
            cfg_stat = "OK" if t.config_exists else "MISSING"
            srv_stat = "OK (2)" if (t.kardashev_configured and t.opencommons_configured) else "INCOMPLETE"
            skl_stat = "OK" if t.skill_exists else "MISSING"
            leg_stat = "CLEAN" if t.legacy_purged else "DIRTY"
            status = "PASS" if t.ok else "FAIL"
            row = f"{t.target:<13} {cfg_disp:<34} {cfg_stat:<8} {srv_stat:<9} {skl_stat:<8} {leg_stat:<8} {status:<6}"
            lines.append(row)

        lines.append("-" * 88)
        lines.append("Live MCP Stdio Handshakes:")

        def _fmt_hs(name: str, hs: HandshakeResult) -> str:
            if hs.success:
                tool_list = ", ".join(hs.tools[:4])
                suffix = f" (+{len(hs.tools) - 4} more)" if len(hs.tools) > 4 else ""
                return f"  ✓ [{name}] {hs.server_name} v{hs.server_version} ({len(hs.tools)} tools: {tool_list}{suffix})"
            return f"  ✗ [{name}] FAILED: {hs.error}"

        lines.append(_fmt_hs("kardashev-mcp", self.kardashev_handshake))
        lines.append(_fmt_hs("opencommons-mcp", self.opencommons_handshake))
        lines.append("=" * 88)

        overall = "ALL SYSTEMS OPERATIONAL" if self.all_ok else "ATTENTION: INTEGRATION ISSUES DETECTED"
        lines.append(f"Overall Status: {overall}")
        return "\n".join(lines)


def _resolve_mcp_command(binary_name: str = "kardashev-mcp") -> str:
    """Resolves the absolute path to an MCP binary using deterministic precedence."""
    user_bin = Path.home() / ".local" / "bin" / binary_name
    if user_bin.exists():
        return str(user_bin)
    which_bin = shutil.which(binary_name)
    if which_bin:
        return which_bin
    venv_bin = Path(sys.executable).parent / binary_name
    if venv_bin.exists():
        return str(venv_bin)
    return binary_name


def _create_atomic_backup(target_path: Path) -> Path | None:
    """Creates an atomic `.bak` backup copy of target_path before modifying it."""
    if not target_path.is_file():
        return None
    backup_path = target_path.with_name(f"{target_path.name}.bak")
    backup_tmp = target_path.with_name(f"{target_path.name}.bak.tmp")
    shutil.copy2(target_path, backup_tmp)
    backup_tmp.replace(backup_path)
    return backup_path


def _atomic_write_text(target_path: Path, content: str) -> None:
    """Atomically writes content to target_path via temporary file replacement."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_name(f"{target_path.name}.tmp")
    temp_path.write_text(content, encoding="utf-8")
    temp_path.replace(target_path)


def _upsert_codex_toml(
    config_path: Path,
    kardashev_cmd: str,
    opencommons_cmd: str,
    *,
    dry_run: bool = False,
    force: bool = False,
) -> tuple[Path | None, list[str]]:
    """Upserts `[mcp_servers.kardashev-cli]` and `[mcp_servers.opencommons]` into Codex config.toml.

    Preserves all other sections and formatting. Purges deprecated `agy-local` sections.
    """
    backup_path: Path | None = None
    purged: list[str] = []

    content = ""
    if config_path.is_file():
        try:
            content = config_path.read_text(encoding="utf-8")
        except OSError:
            content = ""

    lines = content.splitlines(keepends=True)
    sections: list[tuple[str, list[str]]] = []
    current_key = ""
    current_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]") and not stripped.startswith("[["):
            raw_header = stripped[1:-1].strip()
            norm_header = raw_header.replace('"', "").replace("'", "")
            sections.append((current_key, current_lines))
            current_key = norm_header
            current_lines = [line]
        else:
            current_lines.append(line)
    sections.append((current_key, current_lines))

    new_sections: list[list[str]] = []
    has_kardashev = False
    has_opencommons = False

    k_content = [
        "[mcp_servers.kardashev-cli]\n",
        f"command = {json.dumps(str(kardashev_cmd))}\n",
        "args = []\n\n",
    ]
    o_content = [
        "[mcp_servers.opencommons]\n",
        f"command = {json.dumps(str(opencommons_cmd))}\n",
        "args = []\n\n",
    ]

    for key, slines in sections:
        if key in ("mcp_servers.agy-local", "mcp_servers.agy_local"):
            purged.append(f"Purged [{key}] from {config_path}")
            continue
        if key == "mcp_servers.kardashev-cli":
            new_sections.append(k_content)
            has_kardashev = True
        elif key == "mcp_servers.opencommons":
            new_sections.append(o_content)
            has_opencommons = True
        else:
            new_sections.append(slines)

    if not has_kardashev:
        new_sections.append(k_content)
    if not has_opencommons:
        new_sections.append(o_content)

    new_content = "".join("".join(s) for s in new_sections)
    # Validate resulting TOML syntax
    tomllib.loads(new_content)

    if not dry_run:
        if config_path.is_file():
            backup_path = _create_atomic_backup(config_path)
        _atomic_write_text(config_path, new_content)

    return backup_path, purged


def _upsert_mcp_json(
    config_path: Path,
    kardashev_cmd: str,
    opencommons_cmd: str,
    *,
    is_claude: bool = False,
    dry_run: bool = False,
    force: bool = False,
) -> tuple[Path | None, list[str]]:
    """Upserts MCP server registrations into JSON configuration files.

    Used for Claude Code (`~/.claude.json`), Antigravity (`~/.gemini/config/mcp_config.json`),
    and Cursor (`~/.cursor/mcp.json`).
    """
    backup_path: Path | None = None
    purged: list[str] = []

    data: dict[str, Any] = {}
    if config_path.is_file():
        try:
            loaded = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
        except (OSError, json.JSONDecodeError):
            data = {}

    servers = data.get("mcpServers")
    if not isinstance(servers, dict):
        servers = {}
        data["mcpServers"] = servers

    if "agy-local" in servers:
        del servers["agy-local"]
        purged.append(f"Purged 'agy-local' from {config_path}")
    if "agy_local" in servers:
        del servers["agy_local"]
        purged.append(f"Purged 'agy_local' from {config_path}")

    if is_claude:
        servers["kardashev-cli"] = {
            "type": "stdio",
            "command": kardashev_cmd,
            "args": [],
            "env": {},
        }
        servers["opencommons"] = {
            "type": "stdio",
            "command": opencommons_cmd,
            "args": [],
            "env": {},
        }
    else:
        servers["kardashev-cli"] = {
            "command": kardashev_cmd,
            "args": [],
        }
        servers["opencommons"] = {
            "command": opencommons_cmd,
            "args": [],
        }

    new_content = json.dumps(data, indent=2) + "\n"

    if not dry_run:
        if config_path.is_file():
            backup_path = _create_atomic_backup(config_path)
        _atomic_write_text(config_path, new_content)

    return backup_path, purged


def purge_legacy_installations(
    home_dir: Path | None = None,
    *,
    dry_run: bool = False,
) -> list[str]:
    """Purges deprecated legacy `agy-local` MCP entries and `agy-local-coprocessor` skills across all targets."""
    home = (home_dir or Path.home()).resolve()
    purged: list[str] = []

    legacy_dirs = [
        home / ".claude" / "skills" / "agy-local-coprocessor",
        home / ".codex" / "skills" / "agy-local-coprocessor",
        home / ".agents" / "skills" / "agy-local-coprocessor",
        home / ".gemini" / "config" / "skills" / "agy-local-coprocessor",
        home / ".gemini" / "antigravity" / "skills" / "agy-local-coprocessor",
    ]
    for d in legacy_dirs:
        if d.is_dir():
            purged.append(f"Removed legacy skill directory: {d}")
            if not dry_run:
                shutil.rmtree(d, ignore_errors=True)

    legacy_files = [
        home / ".codex" / "instructions" / "agy-local-coprocessor.md",
        home / ".cursor" / "rules" / "agy-local-coprocessor.mdc",
        home / ".cursor" / "rules" / "agy-local-coprocessor.md",
    ]
    for f in legacy_files:
        if f.is_file():
            purged.append(f"Removed legacy instruction/rule file: {f}")
            if not dry_run:
                try:
                    f.unlink()
                except OSError:
                    pass

    # Purge agy-local entries from legacy or auxiliary MCP configs
    aux_configs = [
        (home / ".claude" / "mcp_servers.json", "mcpServers"),
        (home / ".gemini" / "antigravity" / "mcp_config.json", "mcpServers"),
        (home / ".codex" / "mcp.json", "mcpServers"),
    ]
    for cfg, key in aux_configs:
        if cfg.is_file():
            try:
                data = json.loads(cfg.read_text(encoding="utf-8"))
                if isinstance(data, dict) and isinstance(data.get(key), dict):
                    servers = data[key]
                    altered = False
                    for name in ["agy-local", "agy_local"]:
                        if name in servers:
                            del servers[name]
                            altered = True
                            purged.append(f"Purged legacy '{name}' from {cfg}")
                    if altered and not dry_run:
                        _create_atomic_backup(cfg)
                        _atomic_write_text(cfg, json.dumps(data, indent=2) + "\n")
            except (OSError, json.JSONDecodeError):
                pass

    return purged


def perform_mcp_handshake(
    command: str | list[str],
    *,
    required_tools: list[str] | None = None,
    timeout: float = 5.0,
) -> HandshakeResult:
    """Executes a live stdio JSON-RPC 2.0 handshake against an MCP server."""
    cmd_args: list[str]
    if isinstance(command, str):
        resolved = _resolve_mcp_command(command)
        cmd_args = shlex.split(resolved) if " " in resolved and not Path(resolved).exists() else [resolved]
    else:
        cmd_args = list(command)

    cmd_str = " ".join(cmd_args)

    try:
        proc = subprocess.Popen(
            cmd_args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return HandshakeResult(command=cmd_str, success=False, error=f"Process launch error: {exc}")

    try:
        # Step 1: Send initialize request
        init_req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "kardashev-integrate-check", "version": "1.0"},
            },
        }
        if proc.stdin is None or proc.stdout is None:
            return HandshakeResult(command=cmd_str, success=False, error="I/O pipes unavailable")

        proc.stdin.write(json.dumps(init_req) + "\n")
        proc.stdin.flush()

        rlist, _, _ = select.select([proc.stdout], [], [], timeout)
        if not rlist:
            proc.kill()
            return HandshakeResult(
                command=cmd_str,
                success=False,
                error=f"Timeout waiting for 'initialize' response after {timeout}s",
            )

        init_line = proc.stdout.readline()
        if not init_line:
            proc.kill()
            stderr_out = proc.stderr.read() if proc.stderr else ""
            return HandshakeResult(
                command=cmd_str,
                success=False,
                error=f"Server closed connection unexpectedly on initialize. Stderr: {stderr_out.strip()}",
            )

        init_resp = json.loads(init_line)
        if init_resp.get("jsonrpc") != "2.0":
            proc.kill()
            return HandshakeResult(
                command=cmd_str,
                success=False,
                error=f"Non-JSON-RPC 2.0 response: {init_line.strip()}",
            )

        server_info = init_resp.get("result", {}).get("serverInfo", {})
        server_name = server_info.get("name")
        server_version = server_info.get("version")

        # Step 2: Send initialized notification
        init_notif = {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
        proc.stdin.write(json.dumps(init_notif) + "\n")
        proc.stdin.flush()

        # Step 3: Send tools/list request
        tools_req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
        proc.stdin.write(json.dumps(tools_req) + "\n")
        proc.stdin.flush()

        rlist, _, _ = select.select([proc.stdout], [], [], timeout)
        if not rlist:
            proc.kill()
            return HandshakeResult(
                command=cmd_str,
                success=False,
                server_name=server_name,
                server_version=server_version,
                error=f"Timeout waiting for 'tools/list' response after {timeout}s",
            )

        tools_line = proc.stdout.readline()
        tools_resp = json.loads(tools_line)
        tool_objs = tools_resp.get("result", {}).get("tools", [])
        tool_names = [t.get("name") for t in tool_objs if isinstance(t, dict) and "name" in t]

        # Verify required tools
        if required_tools:
            missing = [r for r in required_tools if r not in tool_names]
            if missing:
                proc.terminate()
                return HandshakeResult(
                    command=cmd_str,
                    success=False,
                    server_name=server_name,
                    server_version=server_version,
                    tools=tool_names,
                    error=f"Missing required tools: {missing}",
                )

        proc.terminate()
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            proc.kill()

        return HandshakeResult(
            command=cmd_str,
            success=True,
            server_name=server_name,
            server_version=server_version,
            tools=tool_names,
        )

    except (OSError, ValueError, TimeoutError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        try:
            proc.kill()
        except OSError:
            pass
        return HandshakeResult(command=cmd_str, success=False, error=f"Handshake error: {exc}")


def integrate_frontier_agent(
    target: str | list[str] = "all",
    *,
    home_dir: Path | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> list[IntegrationResult]:
    """Installs `kardashev-mcp`, `opencommons-mcp`, and co-processor routing skills into the specified agent(s).

    Supported targets: `"claude"`, `"cursor"`, `"codex"`, `"antigravity"`, or `"all"`.
    """
    home = (home_dir or Path.home()).resolve()
    kardashev_cmd = _resolve_mcp_command("kardashev-mcp")
    opencommons_cmd = _resolve_mcp_command("opencommons-mcp")

    if target == "all":
        targets = ["claude", "cursor", "codex", "antigravity"]
    elif isinstance(target, list):
        targets = [t.lower() for t in target]
    else:
        targets = [target.lower()]

    # Unconditionally purge legacy items
    purged_items = purge_legacy_installations(home_dir=home, dry_run=dry_run)

    results: list[IntegrationResult] = []

    for tgt in targets:
        if tgt == "claude":
            mcp_file = home / ".claude.json"
            bak, p_mcp = _upsert_mcp_json(
                mcp_file,
                kardashev_cmd,
                opencommons_cmd,
                is_claude=True,
                dry_run=dry_run,
                force=force,
            )
            desktop_cfg = home / ".config" / "Claude" / "claude_desktop_config.json"
            if desktop_cfg.is_file():
                _upsert_mcp_json(
                    desktop_cfg,
                    kardashev_cmd,
                    opencommons_cmd,
                    is_claude=False,
                    dry_run=dry_run,
                    force=force,
                )
            skill_file = home / ".claude" / "skills" / "kardashev-coprocessor" / "SKILL.md"
            if not dry_run:
                _atomic_write_text(skill_file, COPROCESSOR_SKILL_MARKDOWN)
            results.append(
                IntegrationResult(
                    target="claude",
                    mcp_config_path=str(mcp_file),
                    skill_path=str(skill_file),
                    binary_command=kardashev_cmd,
                    status="dry-run" if dry_run else "installed",
                    backup_path=str(bak) if bak else None,
                    skill_paths=[str(skill_file)],
                    purged_items=p_mcp + purged_items,
                )
            )

        elif tgt == "codex":
            mcp_file = home / ".codex" / "config.toml"
            bak, p_mcp = _upsert_codex_toml(
                mcp_file,
                kardashev_cmd,
                opencommons_cmd,
                dry_run=dry_run,
                force=force,
            )
            skill1 = home / ".codex" / "skills" / "kardashev-coprocessor" / "SKILL.md"
            skill2 = home / ".agents" / "skills" / "kardashev-coprocessor" / "SKILL.md"
            if not dry_run:
                _atomic_write_text(skill1, COPROCESSOR_SKILL_MARKDOWN)
                _atomic_write_text(skill2, COPROCESSOR_SKILL_MARKDOWN)
            results.append(
                IntegrationResult(
                    target="codex",
                    mcp_config_path=str(mcp_file),
                    skill_path=str(skill1),
                    binary_command=kardashev_cmd,
                    status="dry-run" if dry_run else "installed",
                    backup_path=str(bak) if bak else None,
                    skill_paths=[str(skill1), str(skill2)],
                    purged_items=p_mcp + purged_items,
                )
            )

        elif tgt == "antigravity":
            mcp_file = home / ".gemini" / "config" / "mcp_config.json"
            bak, p_mcp = _upsert_mcp_json(
                mcp_file,
                kardashev_cmd,
                opencommons_cmd,
                is_claude=False,
                dry_run=dry_run,
                force=force,
            )
            skill_file = home / ".gemini" / "config" / "skills" / "kardashev-coprocessor" / "SKILL.md"
            if not dry_run:
                _atomic_write_text(skill_file, COPROCESSOR_SKILL_MARKDOWN)
            results.append(
                IntegrationResult(
                    target="antigravity",
                    mcp_config_path=str(mcp_file),
                    skill_path=str(skill_file),
                    binary_command=kardashev_cmd,
                    status="dry-run" if dry_run else "installed",
                    backup_path=str(bak) if bak else None,
                    skill_paths=[str(skill_file)],
                    purged_items=p_mcp + purged_items,
                )
            )

        elif tgt == "cursor":
            mcp_file = home / ".cursor" / "mcp.json"
            bak, p_mcp = _upsert_mcp_json(
                mcp_file,
                kardashev_cmd,
                opencommons_cmd,
                is_claude=False,
                dry_run=dry_run,
                force=force,
            )
            rule_file = home / ".cursor" / "rules" / "kardashev-coprocessor.mdc"
            if not dry_run:
                _atomic_write_text(rule_file, COPROCESSOR_SKILL_MARKDOWN)
            results.append(
                IntegrationResult(
                    target="cursor",
                    mcp_config_path=str(mcp_file),
                    skill_path=str(rule_file),
                    binary_command=kardashev_cmd,
                    status="dry-run" if dry_run else "installed",
                    backup_path=str(bak) if bak else None,
                    skill_paths=[str(rule_file)],
                    purged_items=p_mcp + purged_items,
                )
            )

        else:
            raise ValueError(
                f"Unsupported integration target '{tgt}'. Choose from: claude, cursor, codex, antigravity, all."
            )

    return results


def check_integrations(
    targets: str | list[str] = "all",
    *,
    home_dir: Path | None = None,
    timeout: float = 5.0,
) -> CheckSummary:
    """Verifies that each frontier coding agent target has valid configurations and passes live MCP stdio handshakes."""
    home = (home_dir or Path.home()).resolve()
    kardashev_cmd = _resolve_mcp_command("kardashev-mcp")
    opencommons_cmd = _resolve_mcp_command("opencommons-mcp")

    if targets == "all":
        tgt_list = ["claude", "cursor", "codex", "antigravity"]
    elif isinstance(targets, list):
        tgt_list = [t.lower() for t in targets]
    else:
        tgt_list = [targets.lower()]

    target_results: list[TargetCheckResult] = []

    for tgt in tgt_list:
        errors: list[str] = []
        if tgt == "claude":
            cfg = home / ".claude.json"
            skl = home / ".claude" / "skills" / "kardashev-coprocessor" / "SKILL.md"
            k_ok = False
            o_ok = False
            leg_purged = True

            if cfg.is_file():
                try:
                    data = json.loads(cfg.read_text(encoding="utf-8"))
                    servers = data.get("mcpServers", {})
                    k_ok = "kardashev-cli" in servers
                    o_ok = "opencommons" in servers
                    if "agy-local" in servers or "agy_local" in servers:
                        leg_purged = False
                        errors.append("Deprecated 'agy-local' MCP entry found")
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    errors.append(f"Failed to parse Claude JSON: {exc}")
            else:
                errors.append(f"Missing config file {cfg}")

            skl_ok = skl.is_file()
            if not skl_ok:
                errors.append(f"Missing skill file {skl}")

            target_results.append(
                TargetCheckResult(
                    target="claude",
                    config_file=str(cfg),
                    config_exists=cfg.is_file(),
                    kardashev_configured=k_ok,
                    opencommons_configured=o_ok,
                    legacy_purged=leg_purged,
                    skill_file=str(skl),
                    skill_exists=skl_ok,
                    errors=errors,
                )
            )

        elif tgt == "codex":
            cfg = home / ".codex" / "config.toml"
            skl1 = home / ".codex" / "skills" / "kardashev-coprocessor" / "SKILL.md"
            skl2 = home / ".agents" / "skills" / "kardashev-coprocessor" / "SKILL.md"
            k_ok = False
            o_ok = False
            leg_purged = True

            if cfg.is_file():
                try:
                    data = tomllib.loads(cfg.read_text(encoding="utf-8"))
                    servers = data.get("mcp_servers", {})
                    k_ok = "kardashev-cli" in servers
                    o_ok = "opencommons" in servers
                    if "agy-local" in servers or "agy_local" in servers:
                        leg_purged = False
                        errors.append("Deprecated 'agy-local' MCP table found in TOML")
                except (OSError, ValueError, tomllib.TOMLDecodeError) as exc:
                    errors.append(f"Failed to parse Codex TOML: {exc}")
            else:
                errors.append(f"Missing config file {cfg}")

            skl_ok = skl1.is_file() or skl2.is_file()
            if not skl_ok:
                errors.append(f"Missing skill file in {skl1} and {skl2}")

            target_results.append(
                TargetCheckResult(
                    target="codex",
                    config_file=str(cfg),
                    config_exists=cfg.is_file(),
                    kardashev_configured=k_ok,
                    opencommons_configured=o_ok,
                    legacy_purged=leg_purged,
                    skill_file=str(skl1),
                    skill_exists=skl_ok,
                    errors=errors,
                )
            )

        elif tgt == "antigravity":
            cfg = home / ".gemini" / "config" / "mcp_config.json"
            skl = home / ".gemini" / "config" / "skills" / "kardashev-coprocessor" / "SKILL.md"
            k_ok = False
            o_ok = False
            leg_purged = True

            if cfg.is_file():
                try:
                    data = json.loads(cfg.read_text(encoding="utf-8"))
                    servers = data.get("mcpServers", {})
                    k_ok = "kardashev-cli" in servers
                    o_ok = "opencommons" in servers
                    if "agy-local" in servers or "agy_local" in servers:
                        leg_purged = False
                        errors.append("Deprecated 'agy-local' MCP entry found")
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    errors.append(f"Failed to parse Antigravity config: {exc}")
            else:
                errors.append(f"Missing config file {cfg}")

            skl_ok = skl.is_file()
            if not skl_ok:
                errors.append(f"Missing skill file {skl}")

            target_results.append(
                TargetCheckResult(
                    target="antigravity",
                    config_file=str(cfg),
                    config_exists=cfg.is_file(),
                    kardashev_configured=k_ok,
                    opencommons_configured=o_ok,
                    legacy_purged=leg_purged,
                    skill_file=str(skl),
                    skill_exists=skl_ok,
                    errors=errors,
                )
            )

        elif tgt == "cursor":
            cfg = home / ".cursor" / "mcp.json"
            rule = home / ".cursor" / "rules" / "kardashev-coprocessor.mdc"
            k_ok = False
            o_ok = False
            leg_purged = True

            if cfg.is_file():
                try:
                    data = json.loads(cfg.read_text(encoding="utf-8"))
                    servers = data.get("mcpServers", {})
                    k_ok = "kardashev-cli" in servers
                    o_ok = "opencommons" in servers
                    if "agy-local" in servers or "agy_local" in servers:
                        leg_purged = False
                        errors.append("Deprecated 'agy-local' MCP entry found")
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    errors.append(f"Failed to parse Cursor mcp.json: {exc}")
            else:
                errors.append(f"Missing config file {cfg}")

            rule_ok = rule.is_file()
            if not rule_ok:
                errors.append(f"Missing rule file {rule}")

            target_results.append(
                TargetCheckResult(
                    target="cursor",
                    config_file=str(cfg),
                    config_exists=cfg.is_file(),
                    kardashev_configured=k_ok,
                    opencommons_configured=o_ok,
                    legacy_purged=leg_purged,
                    skill_file=str(rule),
                    skill_exists=rule_ok,
                    errors=errors,
                )
            )

    # Perform live MCP handshakes
    k_hs = perform_mcp_handshake(
        kardashev_cmd,
        required_tools=["commons_find", "commons_record", "commons_sync", "commons_status"],
        timeout=timeout,
    )
    o_hs = perform_mcp_handshake(
        opencommons_cmd,
        required_tools=["find", "record", "sync", "status"],
        timeout=timeout,
    )

    return CheckSummary(
        targets=target_results,
        kardashev_handshake=k_hs,
        opencommons_handshake=o_hs,
    )
