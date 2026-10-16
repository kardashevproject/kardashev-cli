"""Tests for the TUI interaction core: input queue, keys, palette, resume, todos."""

from __future__ import annotations

import asyncio
import os
import pty
import time

from kardashev_cli.tui import KardashevRichTUI
from kardashev_cli.tui_input import TerminalTypeAhead, TypeAheadBuffer

# ---------------------------------------------------------------- input queue


def test_buffer_collects_draft_without_loss() -> None:
    buf = TypeAheadBuffer()
    assert buf.feed("hello wor") is True
    assert buf.draft == "hello wor"
    buf.feed("ld")
    assert buf.draft == "hello world"
    assert buf.queued == []


def test_buffer_enter_commits_lines_in_order() -> None:
    buf = TypeAheadBuffer()
    buf.feed("first\rsecond\nthird")
    assert buf.queued == ["first", "second"]
    assert buf.draft == "third"
    assert buf.pop_next() == "first"
    assert buf.pop_next() == "second"
    assert buf.pop_next() is None
    assert buf.take_draft() == "third"
    assert buf.draft == ""


def test_buffer_drops_blank_lines() -> None:
    buf = TypeAheadBuffer()
    buf.feed("\r  \r\r")
    assert buf.queued == []


def test_buffer_editing_keys() -> None:
    buf = TypeAheadBuffer()
    buf.feed("abc\x7f")
    assert buf.draft == "ab"
    buf.feed("\x15")
    assert buf.draft == ""
    buf.feed("a\tb")
    assert buf.draft == "a b"


def test_buffer_swallows_escape_sequences() -> None:
    buf = TypeAheadBuffer()
    # Up arrow, F5 (CSI with params), SS3 sequence, then real text.
    buf.feed("\x1b[A\x1b[15~\x1bOPhi")
    assert buf.draft == "hi"


def test_buffer_escape_sequence_split_across_reads() -> None:
    buf = TypeAheadBuffer()
    buf.feed("\x1b")
    buf.feed("[")
    buf.feed("Bx")
    assert buf.draft == "x"


def test_buffer_preserves_unicode() -> None:
    buf = TypeAheadBuffer()
    buf.feed("café → ok")
    assert buf.draft == "café → ok"


def test_buffer_demote_merges_for_editing() -> None:
    buf = TypeAheadBuffer()
    buf.feed("one\rtwo\rthr")
    buf.demote_to_draft()
    assert buf.queued == []
    assert buf.draft == "one two thr"


def test_buffer_display_line_idle_and_active() -> None:
    buf = TypeAheadBuffer()
    assert buf.display_line() == ""
    buf.feed("queued text\r")
    assert buf.display_line().startswith("queued 1")
    assert "queued text" in buf.display_line()
    buf.feed("typing")
    assert "draft: typing" in buf.display_line()


def test_buffer_display_line_truncates_to_width() -> None:
    buf = TypeAheadBuffer()
    buf.feed("x" * 300)
    line = buf.display_line(40)
    assert len(line) <= 40
    assert line.startswith("draft: …")


# ------------------------------------------------------------ terminal capture


def test_terminal_type_ahead_is_noop_without_tty() -> None:
    read_fd, write_fd = os.pipe()
    try:
        ta = TerminalTypeAhead(TypeAheadBuffer(), fd=read_fd)
        assert TerminalTypeAhead.available(read_fd) is False

        async def go() -> bool:
            return ta.start(asyncio.get_running_loop())

        assert asyncio.run(go()) is False
        ta.stop()  # safe even though never started
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_terminal_type_ahead_captures_and_restores_termios() -> None:
    import termios

    master, slave = pty.openpty()
    try:
        before = termios.tcgetattr(slave)
        buf = TypeAheadBuffer()
        changes: list[int] = []
        ta = TerminalTypeAhead(buf, fd=slave, on_change=lambda: changes.append(1))

        async def go() -> None:
            assert ta.start(asyncio.get_running_loop()) is True
            # Echo and canonical mode are off while capturing; signals stay on.
            during = termios.tcgetattr(slave)
            assert not (during[3] & termios.ECHO)
            assert not (during[3] & termios.ICANON)
            assert during[3] & termios.ISIG
            os.write(master, "ship it\rnext café".encode())
            for _ in range(50):
                await asyncio.sleep(0.01)
                if buf.draft:
                    break
            ta.stop()

        asyncio.run(go())
        assert buf.queued == ["ship it"]
        assert buf.draft == "next café"
        assert changes
        after = termios.tcgetattr(slave)
        assert after[3] == before[3]
    finally:
        os.close(master)
        os.close(slave)


def test_terminal_type_ahead_drains_pending_bytes_on_stop() -> None:
    master, slave = pty.openpty()
    try:
        buf = TypeAheadBuffer()
        ta = TerminalTypeAhead(buf, fd=slave)

        async def go() -> None:
            assert ta.start(asyncio.get_running_loop())
            os.write(master, b"late keys")
            time.sleep(0.05)  # noqa: ASYNC251 — arrive but never let the loop poll the reader
            ta.stop()

        asyncio.run(go())
        assert buf.draft == "late keys"
    finally:
        os.close(master)
        os.close(slave)


def test_terminal_type_ahead_suspend_resume_roundtrip() -> None:
    import termios

    master, slave = pty.openpty()
    try:
        before = termios.tcgetattr(slave)
        ta = TerminalTypeAhead(TypeAheadBuffer(), fd=slave)

        async def go() -> None:
            assert ta.start(asyncio.get_running_loop())
            ta.suspend()
            assert termios.tcgetattr(slave)[3] == before[3]
            ta.resume()
            assert not (termios.tcgetattr(slave)[3] & termios.ECHO)
            ta.stop()

        asyncio.run(go())
        assert termios.tcgetattr(slave)[3] == before[3]
    finally:
        os.close(master)
        os.close(slave)


# ------------------------------------------------------------- TUI integration


def _make_tui(tmp_path) -> KardashevRichTUI:
    from kardashev_cli.builder import HarnessOptions

    opts = HarnessOptions(preset="gemma4-26b", workspaces=[str(tmp_path)])
    return KardashevRichTUI(options=opts)


def test_tui_has_input_queue(tmp_path) -> None:
    tui = _make_tui(tmp_path)
    assert isinstance(tui.input_queue, TypeAheadBuffer)
    assert tui.type_ahead is None
    tui.input_queue.feed("do the next thing\r")
    assert tui.input_queue.pop_next() == "do the next thing"


# ------------------------------------------------ 1. Streaming & fence balance


def test_balance_markdown_fences_progressive_streaming() -> None:
    from kardashev_cli.tui_core import balance_markdown_fences

    # Plain text without fences is untouched
    assert balance_markdown_fences("Hello world") == "Hello world"

    # Unclosed backtick fence is auto-closed for flicker-free syntax highlighting
    partial = "Here is a function:\n```python\ndef add(a, b):\n    return a + b"
    balanced = balance_markdown_fences(partial)
    assert balanced.endswith("\n```")
    assert balanced.count("```") == 2

    # Once closed by the stream, no extra fence is appended
    closed = partial + "\n```\nDone!"
    assert balance_markdown_fences(closed) == closed

    # Tilde fence is balanced with matching tildes
    partial_tilde = "~~~bash\necho hi"
    assert balance_markdown_fences(partial_tilde).endswith("\n~~~")


# -------------------------------------------------------- 2. Activity row line


def test_single_activity_line_and_elapsed_tokens_formatting() -> None:
    from kardashev_cli.tui_core import (
        DARK_THEME,
        LIGHT_THEME,
        current_activity,
        format_activity_line,
        format_elapsed,
        format_tokens,
    )

    assert format_elapsed(0.42) == "0.4s"
    assert format_elapsed(12.3) == "12.3s"
    assert format_elapsed(125.0) == "2m05s"
    assert format_tokens(950) == "950"
    assert format_tokens(14_200) == "14.2k"
    assert format_tokens(1_500_000) == "1.5M"

    assert current_activity(["Bash", "Read"], thinking=True, writing=True) == "running Bash +1"
    assert current_activity([], thinking=False, writing=True) == "responding"
    assert current_activity([], thinking=True, writing=False) == "thinking"
    assert current_activity([], thinking=False, writing=False) == "working"

    line = format_activity_line(
        model="gemma4-26b",
        session_tokens=4200,
        turn_est_tokens=300,
        ctx_limit=32768,
        elapsed_s=3.4,
        activity="running Edit",
        gauge="[██░░░░░░░░] 14%",
        spinner="⠹",
        theme=DARK_THEME,
        extra_badges=[("todos 1/2", LIGHT_THEME.accent)],
    )
    assert "\n" not in line.plain
    assert "running Edit" in line.plain
    assert "gemma4-26b" in line.plain
    assert "4.5k/32.8k tok" in line.plain
    assert "todos 1/2" in line.plain
    assert "3.4s" in line.plain


# ----------------------------------------------------- 3. Tool-call collapse


def test_tool_call_blocks_collapsed_by_default_and_expandable(tmp_path) -> None:
    from rich.console import Console
    from rich.panel import Panel

    from kardashev_cli.tui import ToolEventRecord

    buf = Console(record=True, width=100)
    tui = _make_tui(tmp_path)
    tui.console = buf

    rec = ToolEventRecord(
        call_id="call:1",
        step_id="step:1",
        name="edit_file",
        badge="Edit",
        summary="src/app.py",
        started_at=10.0,
        completed_at=10.045,
        result_preview="✓ updated src/app.py (+3 lines)\nextra line that should not show in collapsed mode",
        full_output="line 1\nline 2\nline 3",
        diff_preview="--- a/src/app.py\n+++ b/src/app.py\n@@ -1 +1 @@\n-old\n+new",
    )

    # Collapsed by default: 1 header line + 1 tail line, NO diff Panel
    assert tui.expand_tools is False
    collapsed_out: list[object] = []
    tui._render_tool_record(rec, collapsed_out)
    assert len(collapsed_out) == 2
    assert not any(isinstance(x, Panel) for x in collapsed_out)
    header_plain = collapsed_out[0].plain  # type: ignore[attr-defined]
    tail_plain = collapsed_out[1].plain  # type: ignore[attr-defined]
    assert "Edit(src/app.py)" in header_plain
    assert "[45ms]" in header_plain
    assert "[Alt+O]" in header_plain
    assert "\n" not in tail_plain
    assert "extra line" not in tail_plain

    # Expanded via /expand or Alt+O: includes full output lines + syntax-highlighted diff Panel
    tui.expand_tools = True
    expanded_out: list[object] = []
    tui._render_tool_record(rec, expanded_out)
    assert any(isinstance(x, Panel) for x in expanded_out)
    assert len(expanded_out) >= 4


# ----------------------------------------- 4. Esc-interrupt & keybindings


def test_double_esc_and_ctrl_c_trigger_interrupt_on_buffer() -> None:
    buf = TypeAheadBuffer()
    # Single Esc followed by printable char is not an interrupt
    buf.feed("\x1ba")
    assert buf.interrupt_requested is False

    # Double Esc (\x1b\x1b) requests an interrupt without polluting draft
    buf.feed("working draft\x1b\x1b")
    assert buf.interrupt_requested is True
    assert buf.consume_interrupt() is True
    assert buf.interrupt_requested is False
    assert "working draft" in buf.draft

    # Ctrl+C (\x03) also triggers interrupt_requested
    buf.feed("\x03")
    assert buf.consume_interrupt() is True


def test_keybindings_registered_for_thoughts_tools_todos_policy_help(tmp_path) -> None:
    tui = _make_tui(tmp_path)
    kb = tui._build_key_bindings()
    bound_keys = {tuple(str(k.value if hasattr(k, "value") else k) for k in b.keys) for b in kb.bindings}
    assert ("escape", "c-m") in bound_keys  # Alt+Enter: insert newline
    assert ("escape", "t") in bound_keys  # Alt+T: thoughts
    assert ("escape", "o") in bound_keys  # Alt+O: expand tools
    assert ("escape", "d") in bound_keys  # Alt+D: collapse todos
    assert ("escape", "p") in bound_keys  # Alt+P: cycle policy
    assert ("escape", "h") in bound_keys  # Alt+H: help footer


# --------------------------------------------------------- 5. Slash palette


def test_slash_palette_completeness_and_unknown_prefix_rejection(tmp_path) -> None:
    from prompt_toolkit.document import Document
    from rich.console import Console

    from kardashev_cli.tui import SLASH_COMMANDS, KardashevCompleter
    from kardashev_cli.tui_core import SLASH_ALIASES, validate_slash_command

    # Required commands must be in SLASH_COMMANDS with concise descriptions
    for required in (
        "/help",
        "/status",
        "/resume",
        "/todos",
        "/expand",
        "/theme",
        "/copy",
        "/save",
        "/load",
        "/preset",
        "/model",
    ):
        assert required in SLASH_COMMANDS
        assert len(SLASH_COMMANDS[required]) >= 10

    # Completer surfaces known commands and rejects unknown prefixes inline
    completer = KardashevCompleter(str(tmp_path))
    known_hits = list(completer.get_completions(Document("/res", cursor_position=4), None))
    assert any(c.text == "/resume" for c in known_hits)

    unknown_hits = list(completer.get_completions(Document("/statuz", cursor_position=7), None))
    assert len(unknown_hits) == 1
    meta_str = str(unknown_hits[0].display_meta).lower()
    assert "unknown command" in meta_str

    # validate_slash_command resolves aliases and suggests closest matches on typos
    ok, canon, err = validate_slash_command("/todo add ship tui", SLASH_COMMANDS, SLASH_ALIASES)
    assert ok is True
    assert canon == "/todos"
    assert err is None

    bad_ok, bad_canon, bad_err = validate_slash_command("/statuz", SLASH_COMMANDS, SLASH_ALIASES)
    assert bad_ok is False
    assert bad_canon is None
    assert "Did you mean" in bad_err
    assert "/status" in bad_err

    # Submitting an unknown slash command in TUI prints the inline suggestion and returns False
    buf = Console(record=True, width=100)
    tui = _make_tui(tmp_path)
    tui.console = buf
    should_exit = asyncio.run(tui._handle_slash_command("/statuz", None, tui.options))  # type: ignore[arg-type]
    assert should_exit is False
    rendered = buf.export_text()
    assert "Unknown command '/statuz'" in rendered
    assert "/status" in rendered


# -------------------------------------------------------- 6. Session resume


def test_session_save_and_resume_picker_and_restore(tmp_path) -> None:
    from rich.console import Console

    from kardashev_cli.tui import TurnTranscript

    sessions_dir = tmp_path / "sessions"
    buf = Console(record=True, width=110)
    tui = _make_tui(tmp_path)
    tui.console = buf
    tui._sessions_dir_override = sessions_dir

    tui.transcripts.append(
        TurnTranscript(
            turn_index=1,
            user_prompt="refactor the router",
            thinking="analyzing routes",
            assistant_response="Refactored router module.",
            tools=[{"badge": "Edit", "summary": "router.py", "duration_ms": 18.0, "result_preview": "✓ ok", "is_error": False}],
            duration_s=1.2,
            prompt_tokens=120,
            output_tokens=80,
        )
    )
    tui.todos.add("Write router unit tests", status="in_progress")
    tui.todos.add("Update API docs", status="done")

    # Save session via /save
    asyncio.run(tui._handle_slash_command("/save alpha-run", None, tui.options))  # type: ignore[arg-type]
    saved_file = sessions_dir / "alpha-run.json"
    assert saved_file.is_file()

    # Welcome banner surfaces Recent Sessions on launch
    banner_buf = Console(record=True, width=110)
    banner_buf.print(tui.render_welcome_banner(tui.options))
    banner_text = banner_buf.export_text()
    assert "Recent Sessions" in banner_text
    assert "alpha-run" in banner_text

    # Fresh TUI instance: /resume lists sessions, /resume 1 restores turns + tokens + todos
    tui2 = _make_tui(tmp_path)
    buf2 = Console(record=True, width=110)
    tui2.console = buf2
    tui2._sessions_dir_override = sessions_dir

    asyncio.run(tui2._handle_slash_command("/resume", None, tui2.options))  # type: ignore[arg-type]
    picker_out = buf2.export_text()
    assert "alpha-run" in picker_out
    assert "refactor the router" in picker_out

    asyncio.run(tui2._handle_slash_command("/resume 1", None, tui2.options))  # type: ignore[arg-type]
    assert tui2.session_turns == 1
    assert tui2.total_prompt_tokens == 120
    assert tui2.total_output_tokens == 80
    assert len(tui2.todos.items) == 2
    assert tui2.todos.items[0].text == "Write router unit tests"
    assert tui2.todos.items[0].status == "in_progress"
    assert tui2.todos.items[1].status == "done"


# ---------------------------------------------------------- 7. Todos widget


def test_todos_tracker_auto_ingest_and_slash_commands(tmp_path) -> None:
    from rich.console import Console

    tui = _make_tui(tmp_path)
    buf = Console(record=True, width=100)
    tui.console = buf

    # Auto-ingest from markdown checklist emitted by assistant
    md_sample = (
        "Plan for this turn:\n"
        "- [x] Inspect existing TUI layout\n"
        "- [-] Wire live activity bar\n"
        "- [ ] Add unit tests\n"
    )
    changed = tui.todos.ingest_from_text(md_sample)
    assert changed == 3
    assert tui.todos.counts() == (1, 1, 1)
    assert tui.todos.summary_badge() == "todos 1/3 · 1 wip"

    # Auto-ingest update when task finishes in later markdown chunk
    tui.todos.ingest_from_text("- [x] Wire live activity bar\n")
    assert tui.todos.counts() == (2, 0, 1)

    # Slash commands: /todos done, /todos collapse
    asyncio.run(tui._handle_slash_command("/todos done 3", None, tui.options))  # type: ignore[arg-type]
    assert tui.todos.counts() == (3, 0, 0)

    asyncio.run(tui._handle_slash_command("/todos collapse", None, tui.options))  # type: ignore[arg-type]
    assert tui.todos.collapsed is True
    collapsed_panel = tui.todos.render(tui.theme)
    assert collapsed_panel is not None


# ------------------------------------------------------- 8. Edit visibility


def test_edit_diff_preview_and_three_line_approval_prompt(monkeypatch) -> None:
    from google.antigravity import types
    from rich.console import Console

    import kardashev_cli.tui as tui_mod
    from kardashev_cli.tui import _confirm_tool_handler, format_tool_badge
    from kardashev_cli.tui_core import format_approval_prompt, is_file_modifying_tool

    assert is_file_modifying_tool("edit_file") is True
    assert is_file_modifying_tool("write_to_file") is True
    assert is_file_modifying_tool("apply_smart_patch") is True
    assert is_file_modifying_tool("view_file") is False

    # Approval prompt is strictly <= 3 lines
    prompt_text = format_approval_prompt("Edit", "src/main.py", has_diff=True)
    lines = prompt_text.plain.splitlines()
    assert 1 <= len(lines) <= 3
    assert "[y]es" in prompt_text.plain
    assert "[a]lways" in prompt_text.plain

    # Write and Patch tool calls generate unified diffs
    tc_write = types.ToolCall(
        name="write_to_file",
        args={"TargetFile": "hello.py", "CodeContent": "def greet():\n    return 'hi'\n"},
        id="tc:w1",
    )
    badge_w, summary_w, diff_w = format_tool_badge(tc_write)
    assert badge_w == "Write"
    assert "hello.py" in summary_w
    assert diff_w is not None
    assert "+++ b/hello.py" in diff_w
    assert "+def greet():" in diff_w

    # _confirm_tool_handler renders the syntax-highlighted diff BEFORE prompting
    rec_console = Console(record=True, width=100)
    monkeypatch.setattr(tui_mod, "_CONSOLE", rec_console)

    async def _fake_input(prompt_str: str = "") -> str:
        rec_console.print(prompt_str, end="")
        return "y"

    monkeypatch.setattr(tui_mod, "_nonblocking_input", _fake_input)
    decision = asyncio.run(_confirm_tool_handler(tc_write))
    assert decision is True
    out = rec_console.export_text()
    diff_pos = out.find("+++ b/hello.py")
    allow_pos = out.find("Allow")
    assert diff_pos != -1 and allow_pos != -1
    assert diff_pos < allow_pos


# ----------------------------------------------------------------- 9. Theme


def test_theme_tokens_and_zero_todo_residue(tmp_path) -> None:
    from pathlib import Path

    from rich.console import Console

    from kardashev_cli.tui_core import DARK_THEME, LIGHT_THEME, get_theme

    assert get_theme("dark") == DARK_THEME
    assert get_theme("light") == LIGHT_THEME
    assert DARK_THEME.syntax_theme != LIGHT_THEME.syntax_theme

    tui = _make_tui(tmp_path)
    tui.console = Console(record=True, width=100)
    asyncio.run(tui._handle_slash_command("/theme light", None, tui.options))  # type: ignore[arg-type]
    assert tui.theme.name == "light"
    asyncio.run(tui._handle_slash_command("/theme dark", None, tui.options))  # type: ignore[arg-type]
    assert tui.theme.name == "dark"

    # Verify zero TODO(owner...) residue in TUI source files
    src_dir = Path(__import__("kardashev_cli.tui").tui.__file__).parent
    for fname in ("tui.py", "tui_core.py", "tui_input.py"):
        content = (src_dir / fname).read_text(encoding="utf-8")
        assert "TODO(" not in content, f"Found unresolved TODO( in {fname}"


def test_diff_stats_and_copy_code_block(tmp_path) -> None:
    from rich.console import Console

    from kardashev_cli.tui import ToolEventRecord, TurnTranscript
    from kardashev_cli.tui_core import compute_diff_stats, extract_last_code_block

    diff_text = "--- a/app.py\n+++ b/app.py\n@@ -1,2 +1,3 @@\n-old_line\n+new_line_1\n+new_line_2\n"
    assert compute_diff_stats(diff_text) == (2, 1)

    tui = _make_tui(tmp_path)
    rec = ToolEventRecord(
        call_id="tc:diff",
        name="replace_file_content",
        badge="Edit",
        summary="app.py",
        diff_preview=diff_text,
        completed_at=1.0,
    )
    out: list = []
    tui._render_tool_record(rec, out)
    assert "+2" in out[0].plain and "-1" in out[0].plain

    md = "Here is code:\n```python\nprint(1)\n```\nAnd final:\n```bash\npytest -q\n```"
    assert extract_last_code_block(md) == "pytest -q"

    buf = Console(record=True, width=100)
    tui.console = buf
    tui.transcripts.append(
        TurnTranscript(
            turn_index=1,
            user_prompt="run tests",
            thinking="",
            assistant_response=md,
            tools=[],
            duration_s=0.5,
            prompt_tokens=10,
            output_tokens=10,
        )
    )
    target_out = tmp_path / "copied.sh"
    asyncio.run(tui._handle_slash_command(f"/copy {target_out}", None, tui.options))  # type: ignore[arg-type]
    assert target_out.read_text(encoding="utf-8").strip() == "pytest -q"
    assert "Wrote 1 line(s)" in buf.export_text()


