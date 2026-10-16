"""Hermetic unit tests for Frontier Agent Integrations (`kardashev integrate`).

Tests coverage:
- Fresh registration on all 4 platforms (Claude, Codex, Antigravity, Cursor)
- Atomic `.bak` backup creation on reconfiguration
- Unconditional purge of deprecated `agy-local` MCP entries and `agy-local-coprocessor` skills
- `--dry-run` flag verifying non-destructive preview
- Live stdio JSON-RPC handshake (`initialize` + `tools/list`) against kardashev-mcp and opencommons-mcp
- Health check verification (`kardashev integrate --check`)
- CLI command dispatch with various flags
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import tomllib

from kardashev_cli.cli import main as cli_main
from kardashev_cli.integrate import (
    check_integrations,
    integrate_frontier_agent,
    perform_mcp_handshake,
)


def test_fresh_registration_all_platforms(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    results = integrate_frontier_agent("all", home_dir=tmp_path)
    assert len(results) == 4
    assert all(r.status == "installed" for r in results)

    # 1. Claude Code
    claude_cfg = tmp_path / ".claude.json"
    assert claude_cfg.is_file()
    claude_data = json.loads(claude_cfg.read_text(encoding="utf-8"))
    assert "mcpServers" in claude_data
    assert "kardashev-cli" in claude_data["mcpServers"]
    assert "opencommons" in claude_data["mcpServers"]
    assert claude_data["mcpServers"]["kardashev-cli"]["type"] == "stdio"
    assert claude_data["mcpServers"]["opencommons"]["type"] == "stdio"
    claude_skill = tmp_path / ".claude" / "skills" / "kardashev-coprocessor" / "SKILL.md"
    assert claude_skill.is_file()
    assert "# Kardashev Co-Processor" in claude_skill.read_text(encoding="utf-8")

    # 2. Codex
    codex_cfg = tmp_path / ".codex" / "config.toml"
    assert codex_cfg.is_file()
    codex_data = tomllib.loads(codex_cfg.read_text(encoding="utf-8"))
    assert "mcp_servers" in codex_data
    assert "kardashev-cli" in codex_data["mcp_servers"]
    assert "opencommons" in codex_data["mcp_servers"]
    codex_skill1 = tmp_path / ".codex" / "skills" / "kardashev-coprocessor" / "SKILL.md"
    codex_skill2 = tmp_path / ".agents" / "skills" / "kardashev-coprocessor" / "SKILL.md"
    assert codex_skill1.is_file()
    assert codex_skill2.is_file()

    # 3. Antigravity
    agy_cfg = tmp_path / ".gemini" / "config" / "mcp_config.json"
    assert agy_cfg.is_file()
    agy_data = json.loads(agy_cfg.read_text(encoding="utf-8"))
    assert "mcpServers" in agy_data
    assert "kardashev-cli" in agy_data["mcpServers"]
    assert "opencommons" in agy_data["mcpServers"]
    agy_skill = tmp_path / ".gemini" / "config" / "skills" / "kardashev-coprocessor" / "SKILL.md"
    assert agy_skill.is_file()

    # 4. Cursor
    cursor_cfg = tmp_path / ".cursor" / "mcp.json"
    assert cursor_cfg.is_file()
    cursor_data = json.loads(cursor_cfg.read_text(encoding="utf-8"))
    assert "mcpServers" in cursor_data
    assert "kardashev-cli" in cursor_data["mcpServers"]
    assert "opencommons" in cursor_data["mcpServers"]
    cursor_rule = tmp_path / ".cursor" / "rules" / "kardashev-coprocessor.mdc"
    assert cursor_rule.is_file()

    # No backups created on fresh installation
    assert not (tmp_path / ".claude.json.bak").exists()
    assert not (tmp_path / ".codex" / "config.toml.bak").exists()
    assert not (tmp_path / ".gemini" / "config" / "mcp_config.json.bak").exists()
    assert not (tmp_path / ".cursor" / "mcp.json.bak").exists()


def test_atomic_backup_creation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    # Initial integration
    integrate_frontier_agent("all", home_dir=tmp_path)

    # Stamp custom markers into the existing files
    claude_cfg = tmp_path / ".claude.json"
    c_data = json.loads(claude_cfg.read_text(encoding="utf-8"))
    c_data["marker"] = "pre-backup-claude"
    claude_cfg.write_text(json.dumps(c_data, indent=2) + "\n", encoding="utf-8")

    codex_cfg = tmp_path / ".codex" / "config.toml"
    codex_cfg.write_text(codex_cfg.read_text(encoding="utf-8") + '\ncustom_marker = "pre-backup-codex"\n')

    agy_cfg = tmp_path / ".gemini" / "config" / "mcp_config.json"
    a_data = json.loads(agy_cfg.read_text(encoding="utf-8"))
    a_data["marker"] = "pre-backup-agy"
    agy_cfg.write_text(json.dumps(a_data, indent=2) + "\n", encoding="utf-8")

    cursor_cfg = tmp_path / ".cursor" / "mcp.json"
    cur_data = json.loads(cursor_cfg.read_text(encoding="utf-8"))
    cur_data["marker"] = "pre-backup-cursor"
    cursor_cfg.write_text(json.dumps(cur_data, indent=2) + "\n", encoding="utf-8")

    # Re-run integration: should create atomic .bak files
    results = integrate_frontier_agent("all", home_dir=tmp_path)
    assert len(results) == 4
    assert all(r.backup_path is not None for r in results)

    claude_bak = tmp_path / ".claude.json.bak"
    assert claude_bak.is_file()
    assert json.loads(claude_bak.read_text(encoding="utf-8")).get("marker") == "pre-backup-claude"

    codex_bak = tmp_path / ".codex" / "config.toml.bak"
    assert codex_bak.is_file()
    assert "pre-backup-codex" in codex_bak.read_text(encoding="utf-8")

    agy_bak = tmp_path / ".gemini" / "config" / "mcp_config.json.bak"
    assert agy_bak.is_file()
    assert json.loads(agy_bak.read_text(encoding="utf-8")).get("marker") == "pre-backup-agy"

    cursor_bak = tmp_path / ".cursor" / "mcp.json.bak"
    assert cursor_bak.is_file()
    assert json.loads(cursor_bak.read_text(encoding="utf-8")).get("marker") == "pre-backup-cursor"


def test_legacy_agy_local_purge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    # Setup legacy configs containing agy-local
    claude_cfg = tmp_path / ".claude.json"
    claude_cfg.parent.mkdir(parents=True, exist_ok=True)
    claude_cfg.write_text(
        json.dumps({"mcpServers": {"agy-local": {"command": "legacy-agy-bin"}}}, indent=2),
        encoding="utf-8",
    )

    codex_cfg = tmp_path / ".codex" / "config.toml"
    codex_cfg.parent.mkdir(parents=True, exist_ok=True)
    codex_cfg.write_text(
        '[mcp_servers.agy-local]\ncommand = "legacy-agy-bin"\n\n[other]\nkey = "val"\n',
        encoding="utf-8",
    )

    agy_cfg = tmp_path / ".gemini" / "config" / "mcp_config.json"
    agy_cfg.parent.mkdir(parents=True, exist_ok=True)
    agy_cfg.write_text(
        json.dumps({"mcpServers": {"agy-local": {"command": "legacy-agy-bin"}}}, indent=2),
        encoding="utf-8",
    )

    cursor_cfg = tmp_path / ".cursor" / "mcp.json"
    cursor_cfg.parent.mkdir(parents=True, exist_ok=True)
    cursor_cfg.write_text(
        json.dumps({"mcpServers": {"agy-local": {"command": "legacy-agy-bin"}}}, indent=2),
        encoding="utf-8",
    )

    # Setup legacy skill directories and files
    legacy_skills = [
        tmp_path / ".claude" / "skills" / "agy-local-coprocessor" / "SKILL.md",
        tmp_path / ".codex" / "skills" / "agy-local-coprocessor" / "SKILL.md",
        tmp_path / ".agents" / "skills" / "agy-local-coprocessor" / "SKILL.md",
        tmp_path / ".codex" / "instructions" / "agy-local-coprocessor.md",
        tmp_path / ".gemini" / "config" / "skills" / "agy-local-coprocessor" / "SKILL.md",
        tmp_path / ".gemini" / "antigravity" / "skills" / "agy-local-coprocessor" / "SKILL.md",
        tmp_path / ".cursor" / "rules" / "agy-local-coprocessor.mdc",
    ]
    for s in legacy_skills:
        s.parent.mkdir(parents=True, exist_ok=True)
        s.write_text("legacy skill content\n", encoding="utf-8")

    # Run integration
    results = integrate_frontier_agent("all", home_dir=tmp_path)
    assert len(results) == 4

    # Verify all legacy skill directories and files are purged
    for s in legacy_skills:
        assert not s.exists()

    # Verify agy-local removed from all configs
    claude_data = json.loads(claude_cfg.read_text(encoding="utf-8"))
    assert "agy-local" not in claude_data["mcpServers"]
    assert "kardashev-cli" in claude_data["mcpServers"]

    codex_data = tomllib.loads(codex_cfg.read_text(encoding="utf-8"))
    assert "agy-local" not in codex_data["mcp_servers"]
    assert "kardashev-cli" in codex_data["mcp_servers"]
    assert codex_data.get("other", {}).get("key") == "val"

    agy_data = json.loads(agy_cfg.read_text(encoding="utf-8"))
    assert "agy-local" not in agy_data["mcpServers"]
    assert "kardashev-cli" in agy_data["mcpServers"]

    cursor_data = json.loads(cursor_cfg.read_text(encoding="utf-8"))
    assert "agy-local" not in cursor_data["mcpServers"]
    assert "kardashev-cli" in cursor_data["mcpServers"]


def test_dry_run_does_not_modify_disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    results = integrate_frontier_agent("all", home_dir=tmp_path, dry_run=True)
    assert len(results) == 4
    assert all(r.status == "dry-run" for r in results)

    # Check nothing was written
    assert not (tmp_path / ".claude.json").exists()
    assert not (tmp_path / ".codex" / "config.toml").exists()
    assert not (tmp_path / ".gemini" / "config" / "mcp_config.json").exists()
    assert not (tmp_path / ".cursor" / "mcp.json").exists()
    assert not (tmp_path / ".claude" / "skills").exists()


def test_individual_platform_targeting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    # Target only claude
    r_claude = integrate_frontier_agent("claude", home_dir=tmp_path)
    assert len(r_claude) == 1
    assert r_claude[0].target == "claude"
    assert (tmp_path / ".claude.json").is_file()
    assert not (tmp_path / ".codex" / "config.toml").exists()

    # Target codex and cursor as a list
    r_multi = integrate_frontier_agent(["codex", "cursor"], home_dir=tmp_path)
    assert len(r_multi) == 2
    assert (tmp_path / ".codex" / "config.toml").is_file()
    assert (tmp_path / ".cursor" / "mcp.json").is_file()
    assert not (tmp_path / ".gemini" / "config" / "mcp_config.json").exists()


def test_mcp_handshake_live() -> None:
    # 1. kardashev-mcp handshake
    k_hs = perform_mcp_handshake(
        "kardashev-mcp",
        required_tools=["commons_find", "commons_record", "commons_sync", "commons_status"],
        timeout=5.0,
    )
    assert k_hs.success is True, f"Handshake failed: {k_hs.error}"
    assert k_hs.server_name == "kardashev-mcp"
    assert k_hs.server_version is not None
    assert "commons_find" in k_hs.tools
    assert "commons_record" in k_hs.tools
    assert "scout_codebase" in k_hs.tools
    assert "apply_smart_patch" in k_hs.tools

    # 2. opencommons-mcp handshake
    o_hs = perform_mcp_handshake(
        "opencommons-mcp",
        required_tools=["find", "record", "sync", "status"],
        timeout=5.0,
    )
    assert o_hs.success is True, f"Handshake failed: {o_hs.error}"
    assert o_hs.server_name == "opencommons-mcp"
    assert "find" in o_hs.tools
    assert "record" in o_hs.tools


def test_mcp_handshake_errors() -> None:
    # Nonexistent binary
    res = perform_mcp_handshake("nonexistent-mcp-bin-404", timeout=1.0)
    assert res.success is False
    assert "error" in res.error.lower() or "not found" in res.error.lower() or "launch error" in res.error.lower()

    # Missing required tools
    res2 = perform_mcp_handshake("kardashev-mcp", required_tools=["completely_imaginary_tool_xyz"], timeout=5.0)
    assert res2.success is False
    assert "Missing required tools" in (res2.error or "")


def test_check_integrations_health(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    # Before integration: should report failures/missing
    summary_pre = check_integrations("all", home_dir=tmp_path)
    assert summary_pre.all_ok is False
    assert any(not t.ok for t in summary_pre.targets)

    # Run integration
    integrate_frontier_agent("all", home_dir=tmp_path)

    # After integration: should report all_ok is True
    summary_post = check_integrations("all", home_dir=tmp_path)
    assert summary_post.all_ok is True
    for t in summary_post.targets:
        assert t.ok is True
        assert t.config_exists is True
        assert t.kardashev_configured is True
        assert t.opencommons_configured is True
        assert t.skill_exists is True

    table_str = summary_post.format_table()
    assert "ALL SYSTEMS OPERATIONAL" in table_str
    assert "kardashev-mcp" in table_str


def test_cli_integrate_commands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    # 1. Preview with --dry-run
    ret_dry = cli_main(["integrate", "--dry-run"])
    assert ret_dry == 0
    assert not (tmp_path / ".claude.json").exists()

    # 2. Targeted integration --claude
    ret_claude = cli_main(["integrate", "--claude"])
    assert ret_claude == 0
    assert (tmp_path / ".claude.json").is_file()

    # 3. Full integration --all
    ret_all = cli_main(["integrate", "--all"])
    assert ret_all == 0

    # 4. Check status --check
    ret_check = cli_main(["integrate", "--check"])
    assert ret_check == 0

    # 5. Check status with --json
    ret_json = cli_main(["integrate", "--check", "--json"])
    assert ret_json == 0
