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


def test_mcp_server_jsonrpc() -> None:
    init_resp = handle_mcp_request({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert init_resp is not None
    assert init_resp["result"]["serverInfo"]["name"] == "agy-local"

    tools_resp = handle_mcp_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert tools_resp is not None
    tool_names = {t["name"] for t in tools_resp["result"]["tools"]}
    assert tool_names == {"run_local_agent_task", "list_local_models"}

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



