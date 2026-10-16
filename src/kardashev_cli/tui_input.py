"""Type-ahead input queue for the interactive TUI.

While a turn is streaming, the main prompt is not active, so keystrokes would
otherwise be echoed raw by the terminal and lost to the next prompt. This module
buffers them instead:

* ``TypeAheadBuffer`` is a pure, terminal-free line editor. Printable characters
  extend the draft, Enter commits the draft as a queued line, Backspace and
  Ctrl+U edit it, and escape sequences (arrows, function keys) are swallowed.
* ``TerminalTypeAhead`` feeds the buffer from a real tty. It switches the
  terminal to cbreak mode (no echo, no line buffering) while keeping signal
  generation on, so Ctrl+C still interrupts the turn. The original terminal
  settings are restored on ``stop()`` and around any blocking ``input()`` prompt.

Nothing here touches the network or the agent; it only moves keystrokes.
"""

from __future__ import annotations

import asyncio
import codecs
import contextlib
import os
import select
import sys
from collections import deque
from collections.abc import Callable

_BACKSPACE = ("\x7f", "\x08")
_CLEAR_DRAFT = "\x15"  # Ctrl+U
_ESC = "\x1b"


class TypeAheadBuffer:
    """Character-level draft editor plus a FIFO of committed lines."""

    def __init__(self) -> None:
        self.draft: str = ""
        self._queued: deque[str] = deque()
        # Escape-sequence parser state: None, "esc" (just saw ESC), "csi" (inside CSI/SS3).
        self._esc_state: str | None = None

    # -- feeding ---------------------------------------------------------
    def feed(self, text: str) -> bool:
        """Process decoded input. Returns True when visible state changed."""
        changed = False
        for ch in text:
            if self._esc_state == "esc":
                # ESC followed by [ or O opens a sequence; anything else is a lone Alt-chord.
                self._esc_state = "csi" if ch in ("[", "O") else None
                continue
            if self._esc_state == "csi":
                # Parameter and intermediate bytes are 0x20-0x3f; a final byte is 0x40-0x7e.
                if "@" <= ch <= "~":
                    self._esc_state = None
                continue
            if ch == _ESC:
                self._esc_state = "esc"
                continue
            if ch in ("\r", "\n"):
                line = self.draft.strip()
                self.draft = ""
                if line:
                    self._queued.append(line)
                changed = True
                continue
            if ch in _BACKSPACE:
                if self.draft:
                    self.draft = self.draft[:-1]
                    changed = True
                continue
            if ch == _CLEAR_DRAFT:
                if self.draft:
                    self.draft = ""
                    changed = True
                continue
            if ch == "\t":
                self.draft += " "
                changed = True
                continue
            if ch.isprintable():
                self.draft += ch
                changed = True
            # all other control characters are ignored
        return changed

    # -- consuming -------------------------------------------------------
    @property
    def queued(self) -> list[str]:
        return list(self._queued)

    def pop_next(self) -> str | None:
        """Pop the oldest committed line, or None."""
        return self._queued.popleft() if self._queued else None

    def take_draft(self) -> str:
        """Return and clear the unsent partial draft (used to pre-fill the prompt)."""
        text, self.draft = self.draft, ""
        return text

    def demote_to_draft(self) -> None:
        """Merge queued lines and the draft back into one editable draft.

        Used after a user interrupt: the user stopped the turn, so queued text is
        handed back for editing instead of being sent automatically.
        """
        parts = [*self._queued, self.draft]
        self._queued.clear()
        self.draft = " ".join(p for p in parts if p)

    def has_pending(self) -> bool:
        return bool(self._queued or self.draft)

    def clear(self) -> None:
        self._queued.clear()
        self.draft = ""
        self._esc_state = None

    # -- display ---------------------------------------------------------
    def display_line(self, width: int = 80) -> str:
        """One subtle status line describing buffered input, or '' when idle."""
        if not self.has_pending():
            return ""
        n = len(self._queued)
        head = f"queued {n} · " if n else ""
        tail_src = self._queued[-1] if (n and not self.draft) else self.draft
        label = f"{head}{'next: ' if (n and not self.draft) else 'draft: '}"
        budget = max(8, width - len(label) - 4)
        shown = tail_src if len(tail_src) <= budget else "…" + tail_src[-(budget - 1):]
        cursor = "▏" if self.draft else ""
        return f"{label}{shown}{cursor}"


class TerminalTypeAhead:
    """Feed a ``TypeAheadBuffer`` from a tty while a turn runs."""

    def __init__(
        self,
        buffer: TypeAheadBuffer,
        fd: int | None = None,
        on_change: Callable[[], None] | None = None,
    ) -> None:
        self.buffer = buffer
        self.fd = sys.stdin.fileno() if fd is None else fd
        self.on_change = on_change
        self._loop: asyncio.AbstractEventLoop | None = None
        self._saved_attrs: list | None = None
        self._active = False  # user intends capture on (start called, stop not yet)
        self._reading = False  # reader + cbreak currently installed
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="ignore")

    @staticmethod
    def available(fd: int | None = None) -> bool:
        try:
            return os.isatty(sys.stdin.fileno() if fd is None else fd)
        except (OSError, ValueError):
            return False

    def start(self, loop: asyncio.AbstractEventLoop | None = None) -> bool:
        """Begin capturing. Returns False (a no-op) when the fd is not a tty."""
        if self._active:
            return self._reading
        if not self.available(self.fd):
            return False
        self._loop = loop or asyncio.get_running_loop()
        self._active = True
        return self._install()

    def stop(self) -> None:
        """Drain anything already typed, then restore the terminal."""
        if not self._active:
            return
        if self._reading:
            self._drain()
        self._uninstall()
        self._active = False
        self._loop = None

    def suspend(self) -> None:
        """Restore cooked mode so a blocking ``input()`` prompt behaves normally."""
        if self._active and self._reading:
            self._drain()
            self._uninstall()

    def resume(self) -> None:
        if self._active and not self._reading:
            self._install()

    # -- internals -------------------------------------------------------
    def _install(self) -> bool:
        import termios
        import tty

        loop = self._loop
        if loop is None:
            return False
        try:
            self._saved_attrs = termios.tcgetattr(self.fd)
            # cbreak: no echo, no line buffering, ISIG kept so Ctrl+C still raises SIGINT.
            tty.setcbreak(self.fd)
            loop.add_reader(self.fd, self._on_readable)
        except (OSError, termios.error, NotImplementedError, ValueError):
            self._restore_attrs()
            return False
        self._reading = True
        return True

    def _uninstall(self) -> None:
        loop = self._loop
        if loop is not None:
            with contextlib.suppress(Exception):
                loop.remove_reader(self.fd)
        self._restore_attrs()
        self._reading = False

    def _restore_attrs(self) -> None:
        if self._saved_attrs is None:
            return
        import termios

        with contextlib.suppress(OSError, termios.error):
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self._saved_attrs)
        self._saved_attrs = None

    def _on_readable(self) -> None:
        try:
            data = os.read(self.fd, 4096)
        except (BlockingIOError, InterruptedError):
            return
        except OSError:
            data = b""
        if not data:
            # EOF on the tty: stop watching so the loop does not spin.
            if self._loop is not None:
                with contextlib.suppress(Exception):
                    self._loop.remove_reader(self.fd)
            return
        self._ingest(data)

    def _ingest(self, data: bytes) -> None:
        text = self._decoder.decode(data)
        if self.buffer.feed(text) and self.on_change is not None:
            with contextlib.suppress(Exception):
                self.on_change()

    def _drain(self) -> None:
        """Read whatever is already waiting without blocking."""
        for _ in range(64):
            try:
                ready, _, _ = select.select([self.fd], [], [], 0)
            except (OSError, ValueError):
                return
            if not ready:
                return
            try:
                data = os.read(self.fd, 4096)
            except (BlockingIOError, InterruptedError):
                return
            except OSError:
                return
            if not data:
                return
            self._ingest(data)
