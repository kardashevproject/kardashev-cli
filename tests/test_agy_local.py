"""Unit tests for agy-local presets, discovery, builder, and CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from google.antigravity import LiteRTAgentConfig, LocalOpenAIAgentConfig

from agy_local.builder import HarnessOptions, PolicyMode, build_agent_config
from agy_local.cli import main
from agy_local.discovery import discover_local_models
from agy_local.mcp_server import handle_mcp_request
from agy_local.presets import BackendKind, get_preset, list_presets
from agy_local.serve import compute_serve_plan
from agy_local.tui import AgyCompleter, AgyRichTUI, expand_at_file_mentions, format_tool_badge


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
    assert init_resp["result"]["serverInfo"]["name"] == "agy-local"

    tools_resp = handle_mcp_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert tools_resp is not None
    tool_names = {t["name"] for t in tools_resp["result"]["tools"]}
    assert tool_names == {
        "scout_codebase",
        "apply_smart_patch",
        "verify_and_repair",
        "run_local_agent_task",
        "list_local_models",
        "commons_find",
        "commons_record",
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
    completer = AgyCompleter(str(tmp_path))
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
    tui = AgyRichTUI(options=opts)
    cfg = build_agent_config(opts)
    panel = tui.render_welcome_banner(cfg)
    assert panel is not None


def test_tui_advanced_features_slices_ralph_and_export(tmp_path: Path) -> None:
    import asyncio
    import subprocess

    from google.antigravity import types
    from google.antigravity.hooks import hooks as hooks_mod

    import agy_local.tui as tui_mod
    from agy_local.tui import (
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

    # 2. Test git checkpoint & rewind in a temporary git repo
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
    tui = AgyRichTUI(options=opts)
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
    from agy_local.updater import UserConfig, detect_install_layout, generate_shell_completion

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
        assert "agy-local" in script
        assert "gemma4-26b" in script

    assert main(["completions", "zsh"]) == 0
    assert "#compdef agy-local agy" in capsys.readouterr().out

    # 3. Install layout & --version
    mode, root = detect_install_layout()
    assert mode in ("git-editable", "managed-venv", "pip")
    assert root is not None
    assert main(["--version"]) == 0
    assert "agy-local 0.1.0" in capsys.readouterr().out


def test_ast_repo_map_and_self_healing_hooks(tmp_path: Path) -> None:
    import asyncio
    import copy

    from google.antigravity import types
    from google.antigravity.hooks import hooks as hooks_mod

    from agy_local.ast_map import parse_python_symbols, render_compact_repo_map, scout_workspace
    from agy_local.harness_hooks import (
        SelfHealingEditHook,
        get_harness_telemetry,
        heal_replace_file_args,
        reset_harness_telemetry,
    )
    from agy_local.integrate import integrate_frontier_agent

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
    from agy_local.ast_map import build_turn_zero_context

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
    assert (home_mock / ".gemini" / "antigravity" / "mcp_config.json").is_file()
    assert (home_mock / ".claude" / "mcp_servers.json").is_file()
    assert (home_mock / ".cursor" / "mcp.json").is_file()
    assert (home_mock / ".codex" / "mcp.json").is_file()

    # 5. C++ Dawn/WebGPU stderr filter
    import os

    from agy_local.tui import suppress_noisy_cpp_stderr

    with suppress_noisy_cpp_stderr():
        os.write(
            2,
            b"Warning: maxDynamicUniformBuffersPerPipelineLayout artificially reduced from 500000 to 16\n",
        )


def test_opencommons_research_engine_and_agent_participation(tmp_path: Path) -> None:
    from agy_local.cli import main as cli_main
    from agy_local.harness_hooks import get_harness_telemetry, reset_harness_telemetry
    from agy_local.mcp_server import handle_mcp_request
    from agy_local.opencommons import (
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

    # 2. Tiered recording: clean pure function auto-commits via 5-gate peer review
    rec_clean = execute_commons_record(
        {
            "source_code": "def cube_int(n: int) -> int:\n    return n * n * n\n",
            "domain": "math",
            "summary": "Compute exact integer cube",
            "invariants": ["cube_int(-n) == -cube_int(n)"],
            "teleological_intent": "Isotropic 3D volume scaling",
            "dimensional_units": "m -> m^3",
        },
        store_path=store_file,
        pending_path=pending_file,
    )
    assert rec_clean["status"] == "accepted"
    assert rec_clean["trust_tier"] == "peer_reviewed"
    assert get_harness_telemetry().commons_concepts_recorded == 1

    # 3. Degenerate constant stub is quarantined by Gate 2
    rec_stub = execute_commons_record(
        {
            "source_code": "def fake_hash(data: bytes) -> int:\n    return 0\n",
            "domain": "crypto",
        },
        store_path=store_file,
        pending_path=pending_file,
    )
    assert rec_stub["status"] == "quarantined"

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
    pending_list = load_pending_proposals(pending_file)
    assert len(pending_list) == 1
    assert pending_list[0].func_name == "custom_energy_scale"

    # Approve the pending proposal -> moves into persistent store
    app_res = approve_pending_proposal(prop_id, store_path=store_file, pending_path=pending_file)
    assert app_res["approved"] is True
    assert len(load_pending_proposals(pending_file)) == 0
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
    assert extracted[0]["status"] == "accepted"

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

    # 8. Test both `opencommons-mcp` and `agy-local-mcp` JSON-RPC 2.0 tool calls
    oc_init = handle_opencommons_mcp_request({"jsonrpc": "2.0", "id": 1, "method": "initialize"}, store_path=store_file)
    assert oc_init is not None and oc_init["result"]["serverInfo"]["name"] == "opencommons-mcp"

    agy_tools = handle_mcp_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert agy_tools is not None
    t_names = [t["name"] for t in agy_tools["result"]["tools"]]
    assert "commons_find" in t_names
    assert "commons_record" in t_names

    # 9. Test `agy-local commons` CLI subcommands
    assert cli_main(["commons", "status", "--store", str(store_file), "--json"]) == 0
    assert cli_main(["commons", "find", "logsumexp", "--store", str(store_file), "--json"]) == 0




