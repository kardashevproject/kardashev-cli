"""Unit tests for kardashev-cli presets, discovery, builder, TUI, and CLI."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from google.antigravity import LiteRTAgentConfig, LocalOpenAIAgentConfig

from kardashev_cli.builder import HarnessOptions, PolicyMode, build_agent_config
from kardashev_cli.cli import main
from kardashev_cli.discovery import discover_local_models
from kardashev_cli.mcp_server import handle_mcp_request
from kardashev_cli.presets import BackendKind, get_preset, list_presets
from kardashev_cli.serve import compute_serve_plan
from kardashev_cli.tui import (
    KardashevCompleter,
    KardashevRichTUI,
    expand_at_file_mentions,
    format_tool_badge,
)


def test_list_and_get_presets() -> None:
    presets = list_presets()
    assert len(presets) >= 8
    gemma = get_preset("gemma4-26b")
    assert gemma.backend == BackendKind.LITERT
    assert gemma.resolved_model_path() is not None
    assert gemma.resolved_model_path().endswith("model.litertlm")

    glm = get_preset("glm4-9b-ollama")
    assert glm.backend == BackendKind.OPENAI
    assert glm.default_base_url == "http://127.0.0.1:11434/v1"

    with pytest.raises(KeyError):
        get_preset("non-existent-preset")


def test_build_litert_config(tmp_path: Path) -> None:
    opts = HarnessOptions(
        preset="gemma4-26b",
        workspaces=[str(tmp_path)],
        policy_mode=PolicyMode.ALLOW_ALL,
        speculative_decoding=True,
    )
    cfg = build_agent_config(opts)
    assert isinstance(cfg, LiteRTAgentConfig)
    assert cfg.enable_speculative_decoding is True
    assert str(tmp_path) in cfg.workspaces
    assert cfg.compaction_config is not None
    assert cfg.compaction_config.token_threshold == 40960


def test_build_openai_config_with_glm_preset(tmp_path: Path) -> None:
    opts = HarnessOptions(
        preset="glm4-vllm",
        workspaces=[str(tmp_path)],
        policy_mode=PolicyMode.SANDBOX,
        with_chronosml_mcp=True,
        token_threshold=32000,
    )
    cfg = build_agent_config(opts)
    assert isinstance(cfg, LocalOpenAIAgentConfig)
    assert cfg.model == "THUDM/glm-4-9b-chat"
    assert cfg.base_url == "http://127.0.0.1:8000/v1"
    assert cfg.capabilities.run_command_config.enable_sandbox is True
    assert len(cfg.mcp_servers) == 1
    assert cfg.mcp_servers[0].name == "chronosml"
    assert cfg.compaction_config.token_threshold == 32000


def test_discover_local_models(tmp_path: Path) -> None:
    litert_dir = tmp_path / "litert" / "gemma4-26b"
    litert_dir.mkdir(parents=True)
    fake_model = litert_dir / "model.litertlm"
    fake_model.write_bytes(b"0" * 1024)

    hf_dir = tmp_path / "hf" / "models--THUDM--glm-4-9b-chat"
    hf_dir.mkdir(parents=True)
    (hf_dir / "weights.bin").write_bytes(b"1" * 2048)

    custom_dir = tmp_path / "custom" / "qwen3.6-35b-a3b-fp8"
    custom_dir.mkdir(parents=True)
    (custom_dir / "config.json").write_text("{}", encoding="utf-8")

    found = discover_local_models(
        litert_dir=tmp_path / "litert",
        hf_cache_dir=tmp_path / "hf",
        extra_model_dirs=[tmp_path / "custom"],
    )
    names = {m.name for m in found}
    assert "gemma4-26b" in names
    assert "THUDM/glm-4-9b-chat" in names
    assert "qwen3.6-35b-a3b-fp8" in names


def test_cli_dry_run_and_subcommands(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["chat", "--preset", "gemma4-26b", "--dry-run"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["config_class"] == "LiteRTAgentConfig"
    assert out["backend"] == "litert"

    rc2 = main(["presets", "--json"])
    assert rc2 == 0
    presets_out = json.loads(capsys.readouterr().out)
    assert any(p["key"] == "glm4-9b-ollama" for p in presets_out)


def test_compute_serve_plan_and_cli(capsys: pytest.CaptureFixture[str]) -> None:
    plan = compute_serve_plan(preset="qwen3.6-35b-vllm")
    assert plan.engine == "vllm"
    assert plan.port == 8000
    assert 0.25 <= plan.gpu_memory_utilization <= 0.90
    assert plan.max_model_len >= 4096
    assert "--gpu-memory-utilization" in plan.command

    with pytest.raises(ValueError, match="LiteRT"):
        compute_serve_plan(preset="gemma4-26b")

    rc = main(["serve", "--preset", "glm4-vllm", "--dry-run"])
    assert rc == 0
    serve_out = json.loads(capsys.readouterr().out)
    assert serve_out["engine"] == "vllm"
    assert serve_out["served_model_name"] == "THUDM/glm-4-9b-chat"


def test_mcp_server_jsonrpc(tmp_path: Path) -> None:
    init_resp = handle_mcp_request({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert init_resp is not None
    assert init_resp["result"]["serverInfo"]["name"] == "kardashev-mcp"

    tools_resp = handle_mcp_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert tools_resp is not None
    tool_names = {t["name"] for t in tools_resp["result"]["tools"]}
    assert tool_names == {
        "scout_codebase",
        "apply_smart_patch",
        "verify_and_repair",
        "run_local_agent_task",
        "local_agent_task_status",
        "list_local_models",
        "commons_find",
        "commons_record",
        "commons_sync",
        "commons_status",
    }

    call_resp = handle_mcp_request(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "list_local_models", "arguments": {}},
        }
    )
    assert call_resp is not None
    assert call_resp["result"]["isError"] is False
    structured = call_resp["result"]["structuredContent"]
    assert "gemma4-26b" in structured["presets"]

    # Test scout_codebase and apply_smart_patch over MCP JSON-RPC
    sample = tmp_path / "calc.py"
    sample.write_text(
        "class Calculator:\n"
        "    def compute_value(self, x: int) -> int:\n"
        "        return x + 1\n",
        encoding="utf-8",
    )
    scout_resp = handle_mcp_request(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "scout_codebase",
                "arguments": {"query": "compute_value Calculator", "workspace": str(tmp_path)},
            },
        }
    )
    assert scout_resp is not None
    assert scout_resp["result"]["isError"] is False
    ranked_paths = [f["path"] for f in scout_resp["result"]["structuredContent"]["ranked_files"]]
    assert "calc.py" in ranked_paths

    patch_resp = handle_mcp_request(
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "apply_smart_patch",
                "arguments": {
                    "target_file": str(sample),
                    "target_content": "def compute_value(self, x: int) -> int:\n    return x + 1\n",
                    "replacement_content": "def compute_value(self, x: int) -> int:\n    return x * 10\n",
                    "start_line": 99,
                    "end_line": 100,
                },
            },
        }
    )
    assert patch_resp is not None
    assert patch_resp["result"]["isError"] is False
    assert "return x * 10" in sample.read_text(encoding="utf-8")


def test_tui_tool_badges_diff_and_at_mentions(tmp_path: Path) -> None:
    from google.antigravity import types
    from prompt_toolkit.document import Document

    # 1. Test Edit diff badge
    edit_tc = types.ToolCall(
        name="replace_file_content",
        args={
            "TargetFile": str(tmp_path / "app.py"),
            "TargetContent": "x = 1\ny = 2\n",
            "ReplacementContent": "x = 10\ny = 2\n",
        },
    )
    badge, summary, diff_preview = format_tool_badge(edit_tc)
    assert badge == "Edit"
    assert "app.py" in summary
    assert diff_preview is not None
    assert "-x = 1" in diff_preview
    assert "+x = 10" in diff_preview

    # 2. Test @file mention expansion
    sample_file = tmp_path / "hello.py"
    sample_file.write_text("print('hi')\n", encoding="utf-8")
    augmented, attached = expand_at_file_mentions("Explain @hello.py please", str(tmp_path))
    assert attached == ["hello.py"]
    assert "<attached_file path=\"hello.py\">" in augmented
    assert "print('hi')" in augmented

    # 3. Test slash & @file tab completion
    completer = KardashevCompleter(str(tmp_path))
    slash_completions = [
        c.text for c in completer.get_completions(Document("/pre", cursor_position=4), None)
    ]
    assert "/preset" in slash_completions

    file_completions = [
        c.text for c in completer.get_completions(Document("check @hel", cursor_position=10), None)
    ]
    assert "@hello.py" in file_completions

    # 4. Test welcome banner rendering
    opts = HarnessOptions(preset="gemma4-26b", workspaces=[str(tmp_path)])
    tui = KardashevRichTUI(options=opts)
    cfg = build_agent_config(opts)
    panel = tui.render_welcome_banner(cfg)
    assert panel is not None
    assert "Kardashev Studio" in str(panel.title)


def test_tui_advanced_features_slices_ralph_and_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio
    import subprocess

    from google.antigravity import types
    from google.antigravity.hooks import hooks as hooks_mod

    import kardashev_cli.tui as tui_mod
    from kardashev_cli.tui import (
        RalphVerifyStopHook,
        ToolEventRecord,
        TurnTranscript,
        create_git_checkpoint,
        render_context_gauge,
        restore_git_checkpoint,
        summarize_tool_result,
    )

    # 1. Test @file:start-end line slicing
    sliced_file = tmp_path / "code.py"
    sliced_file.write_text("line1\nline2\nline3\nline4\nline5\n", encoding="utf-8")
    aug, attached = expand_at_file_mentions("Inspect @code.py:2-4", str(tmp_path))
    assert attached == ["code.py:2-4"]
    assert "line2\nline3\nline4" in aug
    assert "line1" not in aug
    assert "line5" not in aug

    # 2. Test git checkpoint & rewind in a temporary git repo.
    # Git env vars (GIT_DIR, GIT_INDEX_FILE, ...) leak into test subprocesses whenever this
    # runs under a git hook or CI: the tmp-repo git calls would then target the outer repo
    # (a 2026-10-15 incident had a test's `commit -m init` landing in the real worktree).
    # Scrubbing them makes the tmp repo hermetic.
    for key in list(os.environ):
        if key.startswith("GIT_"):
            monkeypatch.delenv(key, raising=False)
    subprocess.run(["git", "-C", str(tmp_path), "init"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "code.py"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "--no-verify", "-m", "init"], check=True, capture_output=True)

    cp = create_git_checkpoint(str(tmp_path), index=1, turn=1, label="initial")
    assert cp is not None
    sliced_file.write_text("corrupted\n", encoding="utf-8")
    aug_diff, att_diff = expand_at_file_mentions("Review @git:diff", str(tmp_path))
    assert "@git:diff" in att_diff
    assert "+corrupted" in aug_diff

    assert restore_git_checkpoint(str(tmp_path), cp) is True
    assert "line1\nline2" in sliced_file.read_text(encoding="utf-8")

    # 3. Test PostToolCallHook result recording & context gauge
    opts = HarnessOptions(preset="gemma4-26b", workspaces=[str(tmp_path)])
    tui = KardashevRichTUI(options=opts)
    tui._active_turn_tools.append(
        ToolEventRecord(call_id="tc-1", name="view_file", badge="Read", summary="code.py")
    )
    tui.record_tool_result(
        types.ToolResult(name="view_file", id="tc-1", step_id="s-1", result="line1\nline2\nline3")
    )
    assert tui._active_turn_tools[0].completed_at is not None
    assert tui._active_turn_tools[0].result_preview == "✓ line1 (+2 lines)"
    assert tui._active_turn_tools[0].is_error is False

    err_summary, is_err = summarize_tool_result(
        types.ToolResult(name="run_command", id="tc-2", step_id="s-2", error="Command exited with code 1")
    )
    assert is_err is True
    assert "Command exited with code 1" in err_summary

    gauge_str = render_context_gauge(20480, 40960)
    assert "50.0%" in gauge_str

    # 4. Test RalphVerifyStopHook (fail -> CONTINUE, pass -> ALLOW_STOP)
    async def _run_ralph_checks() -> None:
        tui_mod._ACTIVE_TUI = tui
        try:
            tui.ralph_verify_cmd = "false"
            hook = RalphVerifyStopHook()
            res_fail = await hook.run(
                hooks_mod.HookContext(),
                types.StopArgs(response_text="done", trajectory_id="t1", continuation_count=0),
            )
            assert res_fail.decision == types.StopDecision.CONTINUE

            tui.ralph_verify_cmd = "true"
            res_pass = await hook.run(
                hooks_mod.HookContext(),
                types.StopArgs(response_text="done", trajectory_id="t1", continuation_count=0),
            )
            assert res_pass.decision == types.StopDecision.ALLOW_STOP
        finally:
            tui_mod._ACTIVE_TUI = None

    asyncio.run(_run_ralph_checks())

    # 5. Test Markdown session export
    tui.transcripts.append(
        TurnTranscript(
            turn_index=1,
            user_prompt="Check code.py",
            thinking="Checking file...",
            assistant_response="Looks clean.",
            tools=[{"badge": "Read", "summary": "code.py", "duration_ms": 12.0, "result_preview": "✓ line1"}],
            duration_s=1.25,
            prompt_tokens=100,
            output_tokens=25,
        )
    )
    cfg = build_agent_config(opts)
    report_path = tui.export_markdown(tmp_path / "report.md", cfg)
    assert report_path.is_file()
    assert "Looks clean." in report_path.read_text(encoding="utf-8")


def test_cli_updater_config_and_completions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from kardashev_cli.updater import UserConfig, detect_install_layout, generate_shell_completion

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    # 1. UserConfig persistence via CLI
    assert main(["config", "set", "default_preset", "glm4-9b-ollama"]) == 0
    assert main(["config", "set", "speculative_decoding", "false"]) == 0
    capsys.readouterr()

    loaded = UserConfig.load()
    assert loaded.default_preset == "glm4-9b-ollama"
    assert loaded.speculative_decoding is False

    assert main(["config", "reset"]) == 0
    capsys.readouterr()
    assert UserConfig.load().default_preset == "gemma4-26b"

    # 2. Shell completions
    for sh in ("bash", "zsh", "fish"):
        script = generate_shell_completion(sh)
        assert "kardashev" in script
        assert "gemma4-26b" in script

    assert main(["completions", "zsh"]) == 0
    assert "#compdef kardashev ksh" in capsys.readouterr().out

    # 3. Install layout & --version
    mode, root = detect_install_layout()
    assert mode in ("git-editable", "managed-venv", "pip")
    assert root is not None
    assert main(["--version"]) == 0
    assert "kardashev 0.2.0" in capsys.readouterr().out


def test_ast_repo_map_and_self_healing_hooks(tmp_path: Path) -> None:
    import asyncio
    import copy

    from google.antigravity import types
    from google.antigravity.hooks import hooks as hooks_mod

    from kardashev_cli.ast_map import parse_python_symbols, render_compact_repo_map, scout_workspace
    from kardashev_cli.harness_hooks import (
        SelfHealingEditHook,
        get_harness_telemetry,
        heal_replace_file_args,
        reset_harness_telemetry,
    )
    from kardashev_cli.integrate import integrate_frontier_agent

    # 1. AST Repo-Map & Scout
    py_file = tmp_path / "engine.py"
    src_text = (
        "class QuantumRouter:\n"
        "    \"\"\"Routes quantum states.\"\"\"\n"
        "    def dispatch_packet(self, payload: bytes) -> bool:\n"
        "        return len(payload) > 0\n\n"
        "def helper_fn(a: int, b: int) -> int:\n"
        "    return a + b\n"
    )
    py_file.write_text(src_text, encoding="utf-8")
    summary = parse_python_symbols(src_text, "engine.py")
    assert any(s.name == "QuantumRouter" and s.kind == "class" for s in summary.symbols)
    assert any(s.name == "dispatch_packet" and s.kind == "method" for s in summary.symbols)

    repo_map = render_compact_repo_map(tmp_path)
    assert "engine.py" in repo_map
    assert "QuantumRouter" in repo_map

    scout_res = scout_workspace(tmp_path, "dispatch_packet QuantumRouter")
    assert any(f["path"] == "engine.py" for f in scout_res.ranked_files)
    assert any("dispatch_packet" in s["symbol"] for s in scout_res.surgical_slices)

    # 2. Self-Healing Edit Engine (Stage A: wrong line numbers, Stage B: indentation shift, Stage C: fuzzy match)
    # Stage A: exact match at lines 3-4, but model hallucinated lines 50-55
    args_a = {
        "TargetFile": str(py_file),
        "StartLine": 50,
        "EndLine": 55,
        "TargetContent": "    def dispatch_packet(self, payload: bytes) -> bool:\n        return len(payload) > 0",
        "ReplacementContent": "    def dispatch_packet(self, payload: bytes) -> bool:\n        return len(payload) >= 1",
    }
    healed_a, reason_a = heal_replace_file_args(args_a)
    assert healed_a is not None and reason_a is not None
    assert "exact match" in reason_a
    assert healed_a["StartLine"] == 3
    assert healed_a["EndLine"] == 4

    # Stage B: model forgot class method 4-space indentation
    args_b = {
        "TargetFile": str(py_file),
        "StartLine": 1,
        "EndLine": 2,
        "TargetContent": "def dispatch_packet(self, payload: bytes) -> bool:\n    return len(payload) > 0",
        "ReplacementContent": "def dispatch_packet(self, payload: bytes) -> bool:\n    return True",
    }
    healed_b, reason_b = heal_replace_file_args(args_b)
    assert healed_b is not None and reason_b is not None
    assert "indentation" in reason_b
    assert healed_b["TargetContent"].startswith("    def dispatch_packet")
    assert healed_b["ReplacementContent"].startswith("    def dispatch_packet")

    # Stage C: slight typo in target content (fuzzy similarity >= 0.88)
    args_c = {
        "TargetFile": str(py_file),
        "StartLine": 1,
        "EndLine": 10,
        "TargetContent": "    def dispatch_packet(self, payload: bytes) -> bool:\n        return len(payload) > 0  #",
        "ReplacementContent": "    def dispatch_packet(self, payload: bytes) -> bool:\n        return False",
    }
    _healed_c, reason_c = heal_replace_file_args(args_c)
    assert reason_c is not None and "fuzzy block match" in reason_c

    # Stage D: markdown code fences + relative path resolution
    args_d = {
        "TargetFile": "engine.py",
        "StartLine": 90,
        "EndLine": 95,
        "TargetContent": "```python\ndef helper_fn(a: int, b: int) -> int:\n    return a + b\n```",
        "ReplacementContent": "```python\ndef helper_fn(a: int, b: int) -> int:\n    return a * b\n```",
    }
    healed_d, reason_d = heal_replace_file_args(args_d, workspace=tmp_path)
    assert healed_d is not None and reason_d is not None
    assert "stripped markdown fences" in reason_d
    assert healed_d["TargetFile"] == str(py_file.resolve())
    assert "```" not in healed_d["ReplacementContent"]

    # Stage E: duplicate identical blocks in a file -> picks occurrence closest to StartLine
    dup_file = tmp_path / "dup.py"
    dup_file.write_text(
        "class First:\n    def reset(self):\n        self.x = 0\n\n"
        + ("\n" * 10)
        + "class Second:\n    def reset(self):\n        self.x = 0\n",
        encoding="utf-8",
    )
    healed_dup, _reason_dup = heal_replace_file_args(
        {
            "TargetFile": str(dup_file),
            "StartLine": 20,
            "EndLine": 22,
            "TargetContent": "    def reset(self):\n        self.x = 0",
            "ReplacementContent": "    def reset(self):\n        self.x = 1",
        }
    )
    assert healed_dup is not None
    assert healed_dup["StartLine"] == 16

    # Turn-0 context builder (AST Repo-Map + high-confidence surgical code slices)
    from kardashev_cli.ast_map import build_turn_zero_context

    ctx_block, n_files, n_syms, n_slices = build_turn_zero_context(
        tmp_path, "dispatch_packet QuantumRouter"
    )
    assert "<ast_repo_map" in ctx_block
    assert "<surgical_code_slices>" in ctx_block
    assert n_files >= 1 and n_syms >= 2 and n_slices >= 1

    # 3. Verify Hook deepcopy compatibility, Relative-Path Auto-Heal & Loop-Breaker
    hook = SelfHealingEditHook(workspace=str(tmp_path))
    copied_hook = copy.deepcopy(hook)
    assert isinstance(copied_hook, SelfHealingEditHook)

    async def _test_loop_breaker() -> None:
        reset_harness_telemetry()
        ctx = hooks_mod.HookContext()
        # Relative path in view_file should be normalized to absolute path automatically
        rel_tc = types.ToolCall(name="view_file", args={"file_path": "engine.py"})
        rel_res = await hook.run(ctx, rel_tc)
        assert rel_res.allow is True
        assert rel_res.modified_args is not None
        assert rel_res.modified_args["file_path"] == str(py_file.resolve())

        tc = types.ToolCall(name="run_command", args={"CommandLine": "pytest"})
        r1 = await hook.run(ctx, tc)
        assert r1.allow is True
        r2 = await hook.run(ctx, tc)
        assert r2.allow is True
        r3 = await hook.run(ctx, tc)
        assert r3.allow is False
        assert "Loop-Breaker" in (r3.message or "")
        assert get_harness_telemetry().loop_blocks_count >= 1

    asyncio.run(_test_loop_breaker())

    # 4. Frontier Agent Integrator
    home_mock = tmp_path / "mock_home"
    home_mock.mkdir()
    statuses = integrate_frontier_agent("all", home_dir=home_mock)
    assert len(statuses) == 4
    assert all(s.status == "installed" for s in statuses)
    assert (home_mock / ".gemini" / "config" / "mcp_config.json").is_file()
    assert (home_mock / ".claude.json").is_file()
    assert (home_mock / ".cursor" / "mcp.json").is_file()
    assert (home_mock / ".codex" / "config.toml").is_file()

    # 5. C++ Dawn/WebGPU stderr filter
    import os

    from kardashev_cli.tui import suppress_noisy_cpp_stderr

    with suppress_noisy_cpp_stderr():
        os.write(
            2,
            b"Warning: maxDynamicUniformBuffersPerPipelineLayout artificially reduced from 500000 to 16\n",
        )


def test_opencommons_research_engine_and_agent_participation(tmp_path: Path) -> None:
    from kardashev_cli.cli import main as cli_main
    from kardashev_cli.harness_hooks import get_harness_telemetry, reset_harness_telemetry
    from kardashev_cli.mcp_server import handle_mcp_request
    from kardashev_cli.opencommons import (
        approve_pending_proposal,
        build_turn_zero_commons_context,
        execute_commons_find,
        execute_commons_record,
        extract_candidate_functions_from_source,
        fingerprint_function,
        handle_opencommons_mcp_request,
        load_pending_proposals,
        load_persistent_store,
        reject_pending_proposal,
    )

    reset_harness_telemetry()
    store_file = tmp_path / "store.json"
    pending_file = tmp_path / "pending.json"

    # 1. Behavioral fingerprint merges iterative vs recursive GCD into the same equivalence class
    src_iter = """
def gcd_iter(a: int, b: int) -> int:
    x, y = abs(a), abs(b)
    while y != 0:
        x, y = y, x % y
    return x
"""
    src_rec = """
def gcd_rec(u: int, v: int) -> int:
    u, v = abs(u), abs(v)
    if v == 0:
        return u
    return gcd_rec(v, u % v)
"""
    fp1 = fingerprint_function(src_iter)
    fp2 = fingerprint_function(src_rec)
    assert fp1.sid != fp2.sid
    assert fp1.fp == fp2.fp

    # 2. Tiered recording: clean pure function with contract+invariant evidence passes the
    # correctness gate (verified) and then auto-commits via 5-gate peer review.
    rec_clean = execute_commons_record(
        {
            "source_code": "def cube_int(n: int) -> int:\n    return n * n * n\n",
            "domain": "math",
            "summary": "Compute exact integer cube",
            "invariants": ["f(-n) == -f(n)"],
            "contract": "[((2,), 8), ((3,), 27), ((-2,), -8)]",
            "teleological_intent": "Isotropic 3D volume scaling",
            "dimensional_units": "m -> m^3",
        },
        store_path=store_file,
        pending_path=pending_file,
    )
    assert rec_clean["status"] == "accepted"
    assert rec_clean["trust_tier"] == "peer_reviewed"
    assert rec_clean["verification"]["status"] == "verified"
    assert get_harness_telemetry().commons_concepts_recorded == 1

    # 3. A degenerate constant stub has no contract/invariant/oracle evidence behind it, so the
    # correctness gate finds it unverifiable (not provably wrong, but not trusted either) and
    # stages it for review instead of either accepting or outright quarantining it.
    rec_stub = execute_commons_record(
        {
            "source_code": "def fake_hash(data: bytes) -> int:\n    return 0\n",
            "domain": "crypto",
        },
        store_path=store_file,
        pending_path=pending_file,
    )
    assert rec_stub["status"] == "queued_for_review"
    assert rec_stub["stage_reason"] == "needs_evidence"

    # 4. Pending Review Queue: explicit review request or domain-specific inputs stage in pending_queue.json
    rec_queued = execute_commons_record(
        {
            "source_code": "def custom_energy_scale(x: float) -> float:\n    return x * 2.5 + 1.0\n",
            "domain": "physics",
            "summary": "Linear affine energy calibration",
            "require_human_review": True,
        },
        store_path=store_file,
        pending_path=pending_file,
    )
    assert rec_queued["status"] == "queued_for_review"
    prop_id = rec_queued["proposal_id"]
    # rec_stub (fake_hash) is also sitting in the queue from step 3 above.
    pending_list = load_pending_proposals(pending_file)
    assert len(pending_list) == 2
    assert {p.func_name for p in pending_list} == {"fake_hash", "custom_energy_scale"}

    # Approve the pending proposal -> moves into persistent store
    app_res = approve_pending_proposal(prop_id, store_path=store_file, pending_path=pending_file)
    assert app_res["approved"] is True
    # fake_hash is still sitting in the queue; only custom_energy_scale was approved.
    assert len(load_pending_proposals(pending_file)) == 1
    find_after = execute_commons_find({"query": "custom_energy_scale"}, store_path=store_file)
    assert find_after["hit_count"] >= 1

    # Test reject_pending_proposal as well
    rec_to_reject = execute_commons_record(
        {
            "source_code": "def draft_fn(x: int) -> int:\n    return x + 99\n",
            "require_human_review": True,
        },
        store_path=store_file,
        pending_path=pending_file,
    )
    rej_res = reject_pending_proposal(rec_to_reject["proposal_id"], pending_path=pending_file)
    assert rej_res["rejected"] is True

    # 5. Turn-0 Commons Context Injection (`<opencommons_knowledge>`)
    ctx_xml, hits = build_turn_zero_commons_context(
        "implement numerically stable logsumexp and welford variance",
        store_path=store_file,
    )
    assert hits >= 1
    assert "<opencommons_knowledge" in ctx_xml
    assert "stable_logsumexp" in ctx_xml

    # Unrelated query incurs zero token overhead
    empty_xml, empty_hits = build_turn_zero_commons_context(
        "change css button color to blue",
        store_path=store_file,
    )
    assert empty_hits == 0
    assert empty_xml == ""

    # 6. Autonomous Post-Edit Function Extraction
    sample_module = """
def harmonic_mean_pair(a: float, b: float) -> float:
    \"\"\"Compute harmonic mean of two positive numbers.\"\"\"
    if a + b == 0.0:
        return 0.0
    return (2.0 * a * b) / (a + b)
"""
    extracted = extract_candidate_functions_from_source(
        sample_module,
        store_path=store_file,
        pending_path=pending_file,
    )
    assert len(extracted) == 1
    # No contract/invariants are auto-attached by extraction, so the correctness gate has no
    # evidence to verify against and stages it for review rather than trusting it outright.
    assert extracted[0]["status"] == "queued_for_review"

    # 7. Monotonic Counterexample Ratchet
    store = load_persistent_store(store_file)
    trojan_code = """
def trojan_cube(n: int) -> int:
    if n == 1337:
        raise RuntimeError("hidden trap")
    return n * n * n
"""
    b_trojan, _, _ = store.ingest(trojan_code, domain="math")
    quarantined = store.report_counterexample(b_trojan.block_id, (1337,))
    assert quarantined is True
    assert b_trojan.block_id in store.quarantined_ids

    # 8. Test both `opencommons-mcp` and `kardashev-mcp` JSON-RPC 2.0 tool calls
    oc_init = handle_opencommons_mcp_request({"jsonrpc": "2.0", "id": 1, "method": "initialize"}, store_path=store_file)
    assert oc_init is not None and oc_init["result"]["serverInfo"]["name"] == "opencommons-mcp"

    ksh_tools = handle_mcp_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert ksh_tools is not None
    t_names = [t["name"] for t in ksh_tools["result"]["tools"]]
    assert "commons_find" in t_names
    assert "commons_record" in t_names

    # 9. Test `kardashev commons` CLI subcommands
    assert cli_main(["commons", "status", "--store", str(store_file), "--json"]) == 0
    assert cli_main(["commons", "find", "logsumexp", "--store", str(store_file), "--json"]) == 0


def test_opencommons_zero_trust_sandbox_and_network_gossip_hub(tmp_path: Path) -> None:
    import threading

    import pytest
    from rich.console import Console

    from kardashev_cli.builder import HarnessOptions, build_agent_config
    from kardashev_cli.opencommons import (
        UnsafeConceptCodeError,
        create_opencommons_hub_server,
        execute_commons_record,
        load_persistent_store,
        pull_from_remote_hub,
        push_to_remote_hub,
        verify_ast_safety_static,
    )
    from kardashev_cli.tui import KardashevRichTUI

    # 1. Layer-1 Static AST Capability Sandbox blocks imports, dunders, and forbidden calls
    with pytest.raises(UnsafeConceptCodeError, match="Import statements"):
        verify_ast_safety_static("def bad(x: int) -> int:\n    import os\n    return x\n")

    with pytest.raises(UnsafeConceptCodeError, match="Dunder attribute"):
        verify_ast_safety_static("def escape(x: int) -> int:\n    return x.__class__.__base__\n")

    with pytest.raises(UnsafeConceptCodeError, match="forbidden by security policy"):
        verify_ast_safety_static("def run_eval(x: int) -> int:\n    return eval('x + 1')\n")

    # 2. Live Multi-Threaded HTTP CRDT Gossip Hub: Agent A pushes -> Hub verifies -> Agent B pulls delta
    hub_store_path = tmp_path / "hub_store.json"
    agent_a_store = tmp_path / "agent_a.json"
    agent_b_store = tmp_path / "agent_b.json"

    # Initialize Hub on an ephemeral localhost port (port=0)
    srv = create_opencommons_hub_server(host="127.0.0.1", port=0, store_path=hub_store_path)
    host, port = srv.server_address[:2]
    hub_url = f"http://{host}:{port}"
    srv_thread = threading.Thread(target=srv.serve_forever, daemon=True)
    srv_thread.start()

    try:
        # Agent A records Kahan compensated summation locally
        kahan_src = """
def kahan_compensated_sum(values: list) -> float:
    total = 0.0
    c = 0.0
    for x in values:
        y = float(x) - c
        t = total + y
        c = (t - total) - y
        total = t
    return total
"""
        rec_a = execute_commons_record(
            {
                "source_code": kahan_src,
                "domain": "numerical_analysis",
                "summary": "O(1) error Kahan summation",
                "invariants": ["abs(f([1.0, 1e-16, -1.0]) - 1e-16) <= 1e-16"],
                "contract": "[(([1.0, 2.0, 3.0],), 6.0), (([0.1, 0.2],), 0.30000000000000004), (([],), 0.0)]",
            },
            store_path=agent_a_store,
            pending_path=tmp_path / "pending_a.json",
        )
        assert rec_a["status"] == "accepted"

        # Agent A pushes its local store to the live HTTP CRDT Hub
        push_res = push_to_remote_hub(hub_url=hub_url, store_path=agent_a_store)
        assert push_res["ok"] is True
        assert push_res["accepted_by_hub"] >= 1

        # Agent B pulls from the HTTP CRDT Hub -> receives kahan_compensated_sum via Merkle delta
        pull_res = pull_from_remote_hub(hub_url=hub_url, store_path=agent_b_store, force=True)
        assert pull_res["ok"] is True
        assert pull_res["new_blocks_pulled"] >= 1

        b_store = load_persistent_store(agent_b_store)
        hits = b_store.find("kahan_compensated_sum", top_k=1)
        assert len(hits) == 1
        assert "kahan_compensated_sum" in hits[0].block.block_id

        # Second pull with matching ETag/Merkle root hits the 0-block fast-path
        pull_cached = pull_from_remote_hub(hub_url=hub_url, store_path=agent_b_store, force=False)
        assert pull_cached["ok"] is True
        assert pull_cached["up_to_date"] is True
        assert pull_cached["new_blocks_pulled"] == 0

        # External agent harness connects directly to Hub's Streamable HTTP MCP endpoint (`POST /mcp`)
        import urllib.request

        mcp_req = urllib.request.Request(
            f"{hub_url}/mcp",
            data=json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 42,
                    "method": "tools/call",
                    "params": {"name": "find", "arguments": {"query": "kahan_compensated_sum"}},
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(mcp_req, timeout=5.0) as mcp_http_resp:
            mcp_body = json.loads(mcp_http_resp.read().decode("utf-8"))
            assert mcp_body["id"] == 42
            assert mcp_body["result"]["isError"] is False
            assert mcp_body["result"]["structuredContent"]["hit_count"] >= 1
    finally:
        srv.shutdown()
        srv.server_close()

    # 3. Verify TUI connection error panel renders cleanly without raising when an OpenAI server is offline
    buf_console = Console(record=True, width=100)
    opts = HarnessOptions(backend="openai", model="qwen3.6-35b-a3b-fp8", base_url="http://127.0.0.1:59999/v1")
    cfg = build_agent_config(opts)
    tui = KardashevRichTUI(options=opts, console=buf_console)
    tui._render_connection_error(
        RuntimeError('Post "http://127.0.0.1:59999/v1/chat/completions": dial tcp 127.0.0.1:59999: connect: connection refused'),
        cfg,
    )
    rendered = buf_console.export_text()
    assert "Inference Endpoint Unreachable" in rendered
    assert "kardashev serve --model qwen3.6-35b-a3b-fp8" in rendered


def test_openrouter_presets_and_authenticated_bridge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kardashev_cli.builder import (
        AuthenticatedOpenAIAgentConfig,
        HarnessOptions,
        build_agent_config,
        resolve_api_key,
    )
    from kardashev_cli.integrate import integrate_frontier_agent
    from kardashev_cli.presets import OPENROUTER_BASE_URL, get_preset

    # 1. Curated OpenRouter presets
    qwen_coder = get_preset("openrouter-qwen3-coder")
    assert qwen_coder.model_id == "qwen/qwen3-coder-next"
    assert qwen_coder.default_base_url == OPENROUTER_BASE_URL
    assert qwen_coder.recommended_vram_gb == 0

    # 2. Dynamic <provider>/<model> slug resolution
    dyn_slug = get_preset("deepseek/deepseek-v4.1-flash")
    assert dyn_slug.model_id == "deepseek/deepseek-v4.1-flash"
    assert dyn_slug.default_base_url == OPENROUTER_BASE_URL

    prefixed = get_preset("openrouter:google/gemini-4-argon")
    assert prefixed.model_id == "google/gemini-4-argon"
    assert prefixed.default_base_url == OPENROUTER_BASE_URL

    # 3. AuthenticatedOpenAIAgentConfig with OPENROUTER_API_KEY
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-or-key")
    assert resolve_api_key(OPENROUTER_BASE_URL) == "test-or-key"

    cfg = build_agent_config(
        HarnessOptions(preset="qwen/qwen3-coder-next", workspaces=[str(tmp_path)])
    )
    assert isinstance(cfg, AuthenticatedOpenAIAgentConfig)
    assert cfg.model == "qwen/qwen3-coder-next"
    assert cfg.base_url == OPENROUTER_BASE_URL
    assert cfg.api_key == "test-or-key"

    # 4. Claude Desktop config integration when ~/.config/Claude/claude_desktop_config.json exists
    desktop_dir = tmp_path / ".config" / "Claude"
    desktop_dir.mkdir(parents=True)
    desktop_cfg = desktop_dir / "claude_desktop_config.json"
    desktop_cfg.write_text(json.dumps({"preferences": {"keepAwakeEnabled": True}}), encoding="utf-8")

    integrate_frontier_agent("claude", home_dir=tmp_path)
    updated_desktop = json.loads(desktop_cfg.read_text(encoding="utf-8"))
    assert updated_desktop["preferences"]["keepAwakeEnabled"] is True
    assert "kardashev-cli" in updated_desktop["mcpServers"]
    assert "opencommons" in updated_desktop["mcpServers"]


def test_tui_step_update_tool_badges_and_results() -> None:
    """Verifies format_tool_badge, summarize_tool_result, and extract_step_tool_result_preview handle localharness StepUpdate fields."""
    from google.antigravity import types
    from google.antigravity.connections.local import types as local_types

    from kardashev_cli.tui import (
        HarnessOptions,
        KardashevRichTUI,
        extract_step_tool_result_preview,
        format_tool_badge,
        summarize_tool_result,
    )

    # 1. StepUpdate snake_case fields for run_command, view_file, edit_file, create_file, list_dir, find_file, search_dir
    tc_bash = types.ToolCall(
        name="run_command",
        args={"command_line": "ls -lt ~/.claude/projects", "working_dir": "/tmp/work"},
        id="traj:1",
        step_id="traj:1",
    )
    badge, summary, diff = format_tool_badge(tc_bash)
    assert badge == "Bash"
    assert "ls -lt ~/.claude/projects" in summary
    assert "(in work)" in summary
    assert diff is None

    tc_read = types.ToolCall(
        name="view_file",
        args={"file_path": "/tmp/work/src/main.py", "start_line": 10, "end_line": 40},
        id="traj:2",
        step_id="traj:2",
    )
    badge_r, summary_r, _ = format_tool_badge(tc_read)
    assert badge_r == "Read"
    assert summary_r == "/tmp/work/src/main.py:10-40"

    tc_edit = types.ToolCall(
        name="edit_file",
        args={
            "file_path": "/tmp/work/src/main.py",
            "diff_block": {
                "start_line": 5,
                "end_line": 6,
                "lines": [
                    {"text": "old_val = 1", "action": "LINE_ACTION_DELETE"},
                    {"text": "new_val = 2", "action": "LINE_ACTION_INSERT"},
                ],
            },
        },
        id="traj:3",
        step_id="traj:3",
    )
    badge_e, summary_e, diff_e = format_tool_badge(tc_edit)
    assert badge_e == "Edit"
    assert summary_e == "/tmp/work/src/main.py"
    assert diff_e is not None
    assert "-old_val = 1" in diff_e
    assert "+new_val = 2" in diff_e

    # 2. Structured local_types results in summarize_tool_result
    tr_cmd = types.ToolResult(
        name="run_command",
        id="call_1",
        step_id="traj:1",
        result=local_types.RunCommandResult(output="total 564\nfile1.py\nfile2.py"),
    )
    sum_str, is_err = summarize_tool_result(tr_cmd)
    assert not is_err
    assert "✓ total 564 (+2 lines)" == sum_str

    # 3. Direct StepUpdate DONE result extraction
    tc_bash_done = types.ToolCall(
        name="run_command",
        args={"command_line": "echo hello", "exit_code": 0, "combined_output": "hello world\nline 2"},
        id="traj:1",
        step_id="traj:1",
    )
    fake_done_step = type("FakeStep", (), {"status": types.StepStatus.DONE, "error": ""})()
    prev, err = extract_step_tool_result_preview(tc_bash_done, fake_done_step)
    assert not err
    assert prev == "✓ hello world (+1 lines)"

    # 4. Early PostToolCallHook followed by StepUpdate populates summary in-place without duplicates
    tui = KardashevRichTUI(options=HarnessOptions())
    tui.record_tool_result(tr_cmd)
    assert len(tui._active_turn_tools) == 1
    assert tui._active_turn_tools[0].step_id == "traj:1"
    assert tui._active_turn_tools[0].result_preview == "✓ total 564 (+2 lines)"


def test_tui_multi_step_interleaved_timeline_and_proxy_broken_pipe(tmp_path: Path) -> None:
    """Verifies multi-step turns never concatenate sentences ('one.The') and _BridgeHandler suppresses BrokenPipeError."""
    import asyncio
    import http.server
    import socket
    import threading

    from google.antigravity import types
    from rich.console import Console

    from kardashev_cli.builder import AuthenticatedOpenAIConnectionStrategy, HarnessOptions
    from kardashev_cli.tui import KardashevRichTUI

    async def _run() -> None:
        # 1. Simulate a 2-step turn with tool call between model text segments
        class _FakeConversation:
            last_turn_usage = None
            total_usage = None
            conversation_id = "test-conv"

            def __init__(self) -> None:
                self.compaction_indices: list[int] = []
                self.sent_prompts: list[str] = []
                self.cancelled = False

            async def send(self, prompt: str) -> None:
                self.sent_prompts.append(prompt)

            async def cancel(self) -> None:
                self.cancelled = True

            async def receive_steps(self):
                # Step 1: initial text + empty-args tool call start + populated-args tool call done
                yield types.Step(
                    id="traj:1",
                    step_index=1,
                    trajectory_id="traj",
                    type=types.StepType.TEXT_RESPONSE,
                    source=types.StepSource.MODEL,
                    target=types.StepTarget.USER,
                    status=types.StepStatus.ACTIVE,
                    content_delta="Let me find the latest one.",
                )
                yield types.Step(
                    id="traj:2",
                    step_index=2,
                    trajectory_id="traj",
                    type=types.StepType.TOOL_CALL,
                    source=types.StepSource.MODEL,
                    target=types.StepTarget.ENVIRONMENT,
                    status=types.StepStatus.ACTIVE,
                    tool_calls=[
                        types.ToolCall(name="run_command", args={}, id="traj:2", step_id="traj:2")
                    ],
                )
                yield types.Step(
                    id="traj:2",
                    step_index=2,
                    trajectory_id="traj",
                    type=types.StepType.TOOL_CALL,
                    source=types.StepSource.MODEL,
                    target=types.StepTarget.ENVIRONMENT,
                    status=types.StepStatus.DONE,
                    tool_calls=[
                        types.ToolCall(
                            name="run_command",
                            args={
                                "command_line": "ls -la ~/.claude/projects",
                                "exit_code": 0,
                                "combined_output": "project-a\nproject-b",
                            },
                            id="traj:2",
                            step_id="traj:2",
                        )
                    ],
                )
                # Step 2: second model text response after tool execution
                yield types.Step(
                    id="traj:3",
                    step_index=3,
                    trajectory_id="traj",
                    type=types.StepType.TEXT_RESPONSE,
                    source=types.StepSource.MODEL,
                    target=types.StepTarget.USER,
                    status=types.StepStatus.DONE,
                    content_delta="The most recent activity is in project-a.",
                    is_complete_response=True,
                )

        class _FakeAgent:
            def __init__(self) -> None:
                self.conversation = _FakeConversation()

        buf = Console(record=True, width=110)
        opts = HarnessOptions(
            backend="openai",
            model="z-ai/glm-5.3-prime",
            base_url="http://127.0.0.1:8000/v1",
            workspaces=[str(tmp_path)],
            enable_opencommons=False,
            enable_ast_repo_map=False,
        )
        tui = KardashevRichTUI(options=opts, console=buf)
        tui.auto_inject_ast_map = False
        fake_agent = _FakeAgent()
        await tui._run_turn(fake_agent, "check latest chat", opts)  # type: ignore[arg-type]

        assert len(tui.transcripts) == 1
        tr = tui.transcripts[0]
        # Sentences across distinct steps must be separated by \n\n, never glued as "one.The"
        assert "one.The" not in tr.assistant_response
        assert tr.assistant_response == "Let me find the latest one.\n\nThe most recent activity is in project-a."
        # Tool call badge must have updated from empty args to the full command_line
        assert len(tr.tools) == 1
        assert tr.tools[0]["badge"] == "Bash"
        assert "ls -la ~/.claude/projects" in tr.tools[0]["summary"]
        assert tr.tools[0]["result_preview"] == "✓ project-a (+1 lines)"

        # Simulate model switch / reconnect continuity context
        tui._synced_transcript_count = 0
        prior_ctx = tui._build_prior_turns_context()
        assert "<prior_conversation_history>" in prior_ctx
        assert "check latest chat" in prior_ctx

        # 2. Test AuthenticatedOpenAIConnectionStrategy proxy suppresses BrokenPipeError when client disconnects mid-stream
        class _UpstreamHandler(http.server.BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for _ in range(10):
                    try:
                        self.wfile.write(b"data: " + (b"x" * 2048) + b"\n\n")
                        self.wfile.flush()
                    except OSError:
                        break

            def log_message(self, format: str, *args: object) -> None:
                return

        upstream_srv = http.server.HTTPServer(("127.0.0.1", 0), _UpstreamHandler)
        up_host, up_port = upstream_srv.server_address[:2]
        up_thread = threading.Thread(target=upstream_srv.serve_forever, daemon=True)
        up_thread.start()

        strat = AuthenticatedOpenAIConnectionStrategy(
            base_url=f"http://{up_host}:{up_port}/v1",
            model_name="test-model",
            api_key="test-key",
        )
        # Start only the proxy portion by mocking super().__aenter__
        from google.antigravity.connections.local import local_openai_connection as _loc_oa

        orig_aenter = _loc_oa.LocalOpenAIConnectionStrategy.__aenter__

        async def _noop_aenter(self_obj: object) -> None:
            return None

        _loc_oa.LocalOpenAIConnectionStrategy.__aenter__ = _noop_aenter  # type: ignore[method-assign]
        try:
            await strat.__aenter__()
            proxy_host, proxy_port = strat._proxy_server.server_address[:2]
            # Connect a raw socket, send HTTP POST, read first bytes, and abruptly close to trigger BrokenPipeError
            with socket.create_connection((proxy_host, proxy_port), timeout=3.0) as sock:
                req_bytes = (
                    b"POST /v1/chat/completions HTTP/1.1\r\n"
                    b"Host: 127.0.0.1\r\n"
                    b"Content-Type: application/json\r\n"
                    b"Content-Length: 2\r\n\r\n{}"
                )
                sock.sendall(req_bytes)
                _ = sock.recv(64)
                # Abrupt RST/close while proxy is still streaming chunks
        finally:
            await strat.__aexit__(None, None, None)
            _loc_oa.LocalOpenAIConnectionStrategy.__aenter__ = orig_aenter  # type: ignore[method-assign]
            upstream_srv.shutdown()
            upstream_srv.server_close()

    asyncio.run(_run())


