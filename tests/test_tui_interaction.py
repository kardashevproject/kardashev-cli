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
            time.sleep(0.05)  # arrive but never let the loop poll the reader
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
