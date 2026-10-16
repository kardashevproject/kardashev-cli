"""Pure helpers and state models for the TUI interaction core.

Nothing in this module touches the terminal, the network, or the live agent. It
holds deterministic text/render builders and state models (theme tokens, key
bindings, streaming markdown fence balancer, single-line activity bar, todos
widget, session resume picker, slash validation, and 3-line approval prompts)
so every behavior can be unit tested without a live session.
"""

from __future__ import annotations

import difflib
import json
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ----------------------------------------------------------------------- theme


@dataclass(frozen=True)
class TUITheme:
    """Brand palette tokens for dark and light terminal themes.

    Derived from The Kardashev Project dark palette (--bg #000000, --fg #ffffff,
    --muted #8a8a8e, --dim #3a3a3d) and OpenCommons paper palette (--bg #f6f4ef,
    --ink #141413, --muted #5e5d59, --rule #dcd8ce).
    """

    name: str
    fg: str
    muted: str
    dim: str
    accent: str
    accent_alt: str
    success: str
    warning: str
    error: str
    border: str
    border_accent: str
    code_theme: str
    toolbar_style: str
    prompt_style: str
    model_tag_style: str

    @property
    def label(self) -> str:
        return "Kardashev Obsidian Dark" if self.name == "dark" else "OpenCommons Paper Light"

    @property
    def syntax_theme(self) -> str:
        return self.code_theme

    @property
    def border_muted(self) -> str:
        return self.border

    @property
    def assistant_border(self) -> str:
        return "green" if self.name == "dark" else "#2b8a3e"

    @property
    def tool_badge(self) -> str:
        return self.accent

    @property
    def accent_secondary(self) -> str:
        return self.accent_alt

    @property
    def prompt_symbol(self) -> str:
        return self.prompt_style

    @property
    def model_tag(self) -> str:
        return self.model_tag_style

    @property
    def toolbar_bg(self) -> str:
        parts = self.toolbar_style.split()
        for p in parts:
            if p.startswith("bg:"):
                return p[3:]
        return "#141416"

    @property
    def toolbar_fg(self) -> str:
        parts = self.toolbar_style.split()
        for p in parts:
            if not p.startswith("bg:"):
                return p
        return "#8a8a8e"


DARK_THEME = TUITheme(
    name="dark",
    fg="#ffffff",
    muted="#8a8a8e",
    dim="dim",
    accent="bold cyan",
    accent_alt="bold bright_magenta",
    success="bold green",
    warning="bold yellow",
    error="bold red",
    border="#3a3a3d",
    border_accent="cyan",
    code_theme="monokai",
    toolbar_style="bg:#141416 #8a8a8e",
    prompt_style="ansibrightcyan bold",
    model_tag_style="ansibrightblack",
)

LIGHT_THEME = TUITheme(
    name="light",
    fg="#141413",
    muted="#5e5d59",
    dim="dim",
    accent="bold #0b7285",
    accent_alt="bold #862e9c",
    success="bold #2b8a3e",
    warning="bold #d9480f",
    error="bold #c92a2a",
    border="#dcd8ce",
    border_accent="#0b7285",
    code_theme="friendly",
    toolbar_style="bg:#efece4 #5e5d59",
    prompt_style="ansicyan bold",
    model_tag_style="ansibrightblack",
)

THEMES: dict[str, TUITheme] = {
    "dark": DARK_THEME,
    "light": LIGHT_THEME,
}


def get_theme(name: str | None = None) -> TUITheme:
    """Resolve a TUITheme by name or ``KARDASHEV_THEME`` environment variable."""
    raw = (name or os.environ.get("KARDASHEV_THEME") or "dark").strip().lower()
    return THEMES.get(raw, DARK_THEME)


# ------------------------------------------------------------------ key bindings

# (keys, what it does). Single source for the help footer; the live bindings are
# registered in ``KardashevRichTUI._build_key_bindings`` and a test keeps them in sync.
KEY_BINDINGS: tuple[tuple[str, str], ...] = (
    ("Enter", "submit"),
    ("Alt+Enter", "newline"),
    ("Up/Down", "history"),
    ("Ctrl+C / Esc Esc", "interrupt turn (Ctrl+C twice to quit)"),
    ("Ctrl+D", "quit"),
    ("Alt+T", "toggle thoughts"),
    ("Alt+O", "toggle tool details"),
    ("Alt+D", "toggle todos"),
    ("Alt+P", "cycle policy"),
    ("F1/Alt+H", "toggle this help"),
)


def build_help_footer(
    commands_or_theme: Mapping[str, str] | TUITheme | None = None,
    bindings: Sequence[tuple[str, str]] = KEY_BINDINGS,
    width: int = 100,
) -> Any:
    """Compact help footer: returns a styled Rich Panel when given a TUITheme,
    or plain text lines ``list[str]`` when given a commands Mapping.
    """
    if isinstance(commands_or_theme, TUITheme) or commands_or_theme is None:
        th = commands_or_theme or DARK_THEME
        txt = Text()
        for idx, (k, d) in enumerate(bindings):
            if idx > 0:
                txt.append("  ·  ", style=th.dim)
            txt.append(k, style=th.accent)
            txt.append(f" {d}", style=th.muted)
        return Panel(
            txt,
            title=f"[{th.accent}]Keyboard Shortcuts[/{th.accent}] [{th.dim}]· F1 / Alt+H to hide[/{th.dim}]",
            border_style=th.border,
            padding=(0, 1),
        )

    commands = commands_or_theme
    lines: list[str] = []
    lines.append("  ".join(f"{k} {d}" for k, d in bindings))
    names = sorted(commands)
    row = ""
    for name in names:
        piece = name if not row else f"  {name}"
        if row and len(row) + len(piece) > width:
            lines.append(row)
            row = name
        else:
            row += piece
    if row:
        lines.append(row)
    return lines


# -------------------------------------------------------- streaming markdown


def balance_markdown_fences(text: str) -> str:
    """Close any unclosed fenced code block in streaming Markdown text.

    While tokens stream, an assistant response often ends mid-code-block before
    the closing ````` or ``~~~`` arrives. Appending a matching closing fence
    allows Rich ``Markdown`` to syntax-highlight the in-flight code block on
    every live frame without flickering between plain text and code blocks.
    """
    if not text:
        return text
    open_marker: str | None = None
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("```", "~~~")):
            fence_char = stripped[0]
            run_len = 0
            for ch in stripped:
                if ch == fence_char:
                    run_len += 1
                else:
                    break
            if run_len >= 3:
                marker = fence_char * run_len
                if open_marker is None:
                    open_marker = marker
                elif stripped.startswith(open_marker) and stripped[len(open_marker) :].strip() == "":
                    open_marker = None
    if open_marker is not None:
        sep = "" if text.endswith("\n") else "\n"
        return f"{text}{sep}{open_marker}"
    return text


def compute_diff_stats(diff_text: str | None) -> tuple[int, int]:
    """Count ``(added_lines, removed_lines)`` in a unified diff block (Codex / Claude Code style)."""
    if not diff_text:
        return (0, 0)
    added = 0
    removed = 0
    for ln in diff_text.splitlines():
        if ln.startswith(("+++", "---", "@@")):
            continue
        if ln.startswith("+"):
            added += 1
        elif ln.startswith("-"):
            removed += 1
    return (added, removed)


def extract_last_code_block(markdown_text: str) -> str:
    """Extract the last fenced code block from an assistant Markdown response (or full text if none)."""
    if not markdown_text:
        return ""
    blocks = re.findall(r"```(?:[^\n]*)\n(.*?)```", markdown_text, flags=re.DOTALL)
    if blocks:
        return blocks[-1].strip("\r\n")
    return markdown_text.strip()


# ------------------------------------------------------------------ activity line


def format_elapsed(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rest = divmod(int(seconds), 60)
    return f"{minutes}m{rest:02d}s"


def format_tokens(n: int) -> str:
    if n < 1000:
        return str(n)
    if n < 1_000_000:
        return f"{n / 1000:.1f}k"
    return f"{n / 1_000_000:.1f}M"


def current_activity(
    running_tools: Iterable[str],
    thinking: bool,
    writing: bool,
) -> str:
    """Name what the agent is doing right now, most specific first."""
    tools = [t for t in running_tools if t]
    if tools:
        extra = f" +{len(tools) - 1}" if len(tools) > 1 else ""
        return f"running {tools[0]}{extra}"
    if writing:
        return "responding"
    if thinking:
        return "thinking"
    return "working"


def format_activity_line(
    model: str,
    session_tokens: int,
    turn_tokens_est: int = 0,
    ctx_limit: int = 32768,
    elapsed_s: float = 0.0,
    activity: str = "working",
    gauge_markup: str = "",
    spinner: str = "⠋",
    badges: Sequence[str] = (),
    theme: TUITheme = DARK_THEME,
    *,
    turn_est_tokens: int | None = None,
    gauge: str | None = None,
    extra_badges: Sequence[tuple[str, str] | str] = (),
) -> Text:
    """One status row: spinner, activity, model, tokens + gauge, badges, elapsed."""
    turn_est = turn_est_tokens if turn_est_tokens is not None else turn_tokens_est
    gauge_str = gauge if gauge is not None else gauge_markup

    line = Text()
    line.append(f"  {spinner} ", style=theme.accent)
    line.append(activity, style="bold")
    line.append(" · ", style=theme.dim)
    line.append(model, style=theme.muted)
    line.append(" · ", style=theme.dim)
    tok = format_tokens(session_tokens + turn_est)
    lim_tok = format_tokens(ctx_limit)
    approx = "~" if turn_est else ""
    line.append(f"{approx}{tok}/{lim_tok} tok ", style=theme.muted)
    if gauge_str:
        line.append_text(Text.from_markup(gauge_str))
    for badge in badges:
        if badge:
            line.append(" · ", style=theme.dim)
            line.append(badge, style=theme.muted)
    for eb in extra_badges:
        if isinstance(eb, tuple) and len(eb) == 2:
            b_text, b_style = eb
            if b_text:
                line.append(" · ", style=theme.dim)
                line.append(b_text, style=b_style)
        elif isinstance(eb, str) and eb:
            line.append(" · ", style=theme.dim)
            line.append(eb, style=theme.muted)
    line.append(" · ", style=theme.dim)
    line.append(format_elapsed(elapsed_s), style=theme.muted)
    return line


# --------------------------------------------------------------- slash palette

SLASH_ALIASES: dict[str, str] = {
    "/quit": "/exit",
    "/q": "/exit",
    "/upgrade": "/update",
    "/todo": "/todos",
    "/tools": "/expand",
    "/sessions": "/resume",
    "/presets": "/preset",
    "/?": "/help",
}


def validate_slash_command(
    raw_cmd: str,
    commands: Mapping[str, str],
    aliases: Mapping[str, str] = SLASH_ALIASES,
) -> tuple[bool, str | None, str | None]:
    """Validate a slash command prefix against known commands and aliases.

    Returns ``(is_valid, canonical_cmd, error_message)``.
    """
    stripped = raw_cmd.strip()
    if not stripped.startswith("/"):
        return (True, None, None)
    parts = stripped.split()
    cmd = parts[0].lower()
    if cmd in commands:
        return (True, cmd, None)
    if cmd in aliases and aliases[cmd] in commands:
        return (True, aliases[cmd], None)
    candidates = difflib.get_close_matches(cmd, list(commands.keys()) + list(aliases.keys()), n=2, cutoff=0.5)
    hint = f" Did you mean {', '.join(candidates)}?" if candidates else ""
    return (False, None, f"Unknown command '{cmd}'.{hint} Type /help for available commands.")


# ---------------------------------------------------------------- todos widget

_TODO_LINE_RE = re.compile(
    r"^\s*(?:[-*+]|\d+[.)])\s*\[(?P<mark>[ xX✓✔\-~◐])\]\s+(?P<text>.+?)\s*$"
)

VALID_TODO_STATUSES = ("pending", "in_progress", "completed", "done")


class _CallableCounts(tuple):  # type: ignore[type-arg]
    """Tuple ``(done, in_progress, pending)`` that can also be called as ``counts()``."""

    def __call__(self) -> tuple[int, int, int]:
        return tuple(self)  # type: ignore[return-value]


@dataclass
class TodoItem:
    """Single task item in the session checklist."""

    id: str
    text: str
    status: str = "pending"  # "pending" | "in_progress" | "done"

    def __post_init__(self) -> None:
        if self.status in ("completed", "done", "complete"):
            self.status = "done"
        elif self.status in ("in_progress", "wip", "active"):
            self.status = "in_progress"
        else:
            self.status = "pending"

    def to_dict(self) -> dict[str, str]:
        return {"id": self.id, "text": self.text, "status": self.status}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], fallback_idx: int = 1) -> TodoItem:
        raw_status = str(data.get("status", "pending")).strip().lower()
        if raw_status in ("done", "complete", "completed", "checked", "true"):
            status = "done"
        elif raw_status in ("in_progress", "in-progress", "active", "wip", "doing", "running"):
            status = "in_progress"
        else:
            status = "pending"
        text = str(data.get("text") or data.get("content") or data.get("title") or "").strip()
        item_id = str(data.get("id") or f"todo-{fallback_idx}")
        return cls(id=item_id, text=text, status=status)


@dataclass
class TodoTracker:
    """Session-scoped task checklist with markdown/tool ingestion and persistence."""

    items: list[TodoItem] = field(default_factory=list)
    collapsed: bool = False
    _next_seq: int = 1

    def add(self, text: str, status: str = "pending") -> TodoItem:
        clean = text.strip()
        if not clean:
            raise ValueError("Todo text must not be empty")
        norm_status = (
            "done"
            if status in ("done", "completed", "complete")
            else ("in_progress" if status in ("in_progress", "wip", "active") else "pending")
        )
        for existing in self.items:
            if existing.text.lower() == clean.lower():
                existing.status = norm_status
                return existing
        item_id = str(self._next_seq)
        self._next_seq += 1
        item = TodoItem(id=item_id, text=clean, status=norm_status)
        self.items.append(item)
        return item

    def resolve(self, index_or_id: int | str) -> TodoItem | None:
        if isinstance(index_or_id, int):
            if 1 <= index_or_id <= len(self.items):
                return self.items[index_or_id - 1]
            return None
        key = str(index_or_id).strip()
        if key.isdigit():
            idx = int(key)
            if 1 <= idx <= len(self.items):
                return self.items[idx - 1]
        for item in self.items:
            if item.id == key or item.id == f"todo-{key}" or item.text.lower().startswith(key.lower()):
                return item
        return None

    def set_status(self, index_or_id: int | str, status: str) -> TodoItem | None:
        target = self.resolve(index_or_id)
        if target is None:
            return None
        if status in ("done", "complete", "completed"):
            target.status = "done"
        elif status in ("wip", "in_progress", "in-progress", "active", "start"):
            target.status = "in_progress"
        else:
            target.status = "pending"
        return target

    def remove(self, index_or_id: int | str) -> TodoItem | None:
        target = self.resolve(index_or_id)
        if target is None:
            return None
        self.items.remove(target)
        return target

    def clear(self, completed_only: bool = False) -> int:
        before = len(self.items)
        if completed_only:
            self.items = [it for it in self.items if it.status != "done"]
        else:
            self.items.clear()
        return before - len(self.items)

    def toggle_collapsed(self) -> bool:
        self.collapsed = not self.collapsed
        return self.collapsed

    @property
    def counts(self) -> _CallableCounts:
        """Return ``(done_count, in_progress_count, pending_count)`` (both property & callable)."""
        done = sum(1 for it in self.items if it.status == "done")
        wip = sum(1 for it in self.items if it.status == "in_progress")
        pending = len(self.items) - done - wip
        return _CallableCounts((done, wip, pending))

    def summary_badge(self) -> str:
        """Short token for the activity line when todos exist, else ''."""
        if not self.items:
            return ""
        done, wip, _pending = self.counts
        total = len(self.items)
        active_part = f" · {wip} wip" if wip else ""
        return f"todos {done}/{total}{active_part}"

    def ingest_from_text(self, text: str) -> int:
        """Extract markdown checklist lines (``- [ ] ...``, ``- [x] ...``, ``- [-] ...``).

        Returns the number of items added or updated.
        """
        if not text:
            return 0
        changed = 0
        in_fence = False
        for line in text.splitlines():
            stripped = line.lstrip()
            if stripped.startswith(("```", "~~~")):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            m = _TODO_LINE_RE.match(line)
            if not m:
                continue
            mark = m.group("mark")
            item_text = m.group("text").strip()
            if not item_text:
                continue
            if mark in ("x", "X", "✓", "✔"):
                status = "done"
            elif mark in ("-", "~", "◐"):
                status = "in_progress"
            else:
                status = "pending"
            existing = next((it for it in self.items if it.text.lower() == item_text.lower()), None)
            if existing is not None:
                if existing.status != status:
                    existing.status = status
                    changed += 1
            else:
                self.add(item_text, status=status)
                changed += 1
        return changed

    def ingest_from_tool_call(self, name: str, args: Mapping[str, Any] | None) -> int:
        """Update checklist when the model calls a todo/checklist tool."""
        if not args:
            return 0
        lower = name.lower()
        if "todo" not in lower and "checklist" not in lower and "task_list" not in lower:
            return 0
        raw_todos = args.get("todos") or args.get("items") or args.get("tasks")
        if not isinstance(raw_todos, list):
            return 0
        new_items: list[TodoItem] = []
        for idx, entry in enumerate(raw_todos, start=1):
            if isinstance(entry, Mapping):
                item = TodoItem.from_dict(entry, fallback_idx=idx)
                if item.text:
                    new_items.append(item)
            elif isinstance(entry, str) and entry.strip():
                new_items.append(TodoItem(id=str(idx), text=entry.strip()))
        if not new_items:
            return 0
        self.items = new_items
        self._next_seq = len(self.items) + 1
        return len(new_items)

    def to_list(self) -> list[dict[str, str]]:
        return [it.to_dict() for it in self.items]

    def load_list(self, raw: Sequence[Mapping[str, Any]]) -> None:
        loaded: list[TodoItem] = []
        for idx, entry in enumerate(raw, start=1):
            if isinstance(entry, Mapping):
                item = TodoItem.from_dict(entry, fallback_idx=idx)
                if item.text:
                    loaded.append(item)
        self.items = loaded
        self._next_seq = len(self.items) + 1

    def save_to_file(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "collapsed": self.collapsed,
            "todos": self.to_list(),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load_from_file(self, path: Path) -> bool:
        if not path.is_file():
            return False
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if isinstance(data, Mapping):
            self.collapsed = bool(data.get("collapsed", False))
            raw_list = data.get("todos", [])
            if isinstance(raw_list, list):
                self.load_list(raw_list)
                return True
        elif isinstance(data, list):
            self.load_list(data)
            return True
        return False

    def render(self, theme: TUITheme = DARK_THEME) -> Panel | Text | None:
        """Render the todos checklist as a collapsible panel (or None when empty)."""
        if not self.items:
            return None
        done, wip, _pending = self.counts
        total = len(self.items)
        if self.collapsed:
            line = Text()
            line.append("  ▸ ", style=theme.accent)
            line.append(f"Todos ({done}/{total} done", style="bold")
            if wip:
                line.append(f" · {wip} active", style=theme.warning)
            line.append(") ", style="bold")
            line.append("· Alt+D or /todos to expand", style=theme.dim)
            return Panel(
                line,
                border_style=theme.border,
                padding=(0, 1),
            )

        lines: list[Text] = []
        for idx, item in enumerate(self.items, start=1):
            row = Text()
            if item.status == "done":
                row.append("  ✓ ", style=theme.success)
                row.append(f"{idx}. {item.text}", style=f"strike {theme.muted}")
            elif item.status == "in_progress":
                row.append("  ◐ ", style=theme.warning)
                row.append(f"{idx}. {item.text}", style="bold")
            else:
                row.append("  ☐ ", style=theme.muted)
                row.append(f"{idx}. {item.text}", style=theme.fg)
            lines.append(row)

        body = Text("\n").join(lines)
        return Panel(
            body,
            title=f"[{theme.accent}]▾ Todos ({done}/{total})[/{theme.accent}] [{theme.dim}]· Alt+D collapse[/{theme.dim}]",
            border_style=theme.border,
            padding=(0, 1),
        )


# ------------------------------------------------------------- session resume


@dataclass(frozen=True)
class SessionSummary:
    """Metadata summary for a saved TUI session."""

    session_id: str
    path: Path
    saved_at: str
    preset: str | None
    model: str | None
    turn_count: int
    last_prompt: str
    todo_count: int = 0


def list_recent_sessions(sessions_dir: Path, limit: int = 10) -> list[SessionSummary]:
    """Return saved sessions in ``sessions_dir`` ordered newest first."""
    if not sessions_dir.is_dir():
        return []
    files = sorted(
        (p for p in sessions_dir.glob("*.json") if not p.name.endswith(".todos.json")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    summaries: list[SessionSummary] = []
    for path in files[: max(1, limit)]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, Mapping):
            continue
        turns = data.get("turns")
        turn_list = turns if isinstance(turns, list) else []
        last_prompt = ""
        if turn_list and isinstance(turn_list[-1], Mapping):
            raw_p = str(turn_list[-1].get("user_prompt", "")).strip()
            last_prompt = " ".join(raw_p.splitlines())[:72]
        todos = data.get("todos")
        todo_count = len(todos) if isinstance(todos, list) else 0
        summaries.append(
            SessionSummary(
                session_id=path.stem,
                path=path,
                saved_at=str(data.get("saved_at", "")),
                preset=data.get("preset"),
                model=data.get("model"),
                turn_count=len(turn_list),
                last_prompt=last_prompt,
                todo_count=todo_count,
            )
        )
    return summaries


def resolve_session_target(
    first: str | Path,
    second: Path | str,
    recent: Sequence[SessionSummary] | None = None,
) -> Path | None:
    """Resolve a 1-based picker index, ``latest``, or session name to a JSON file path.

    Accepts either ``(query, sessions_dir)`` or ``(sessions_dir, query)``.
    """
    if isinstance(first, Path):
        sessions_dir = first
        query = str(second)
    else:
        query = str(first)
        sessions_dir = Path(second)

    items = list(recent) if recent is not None else list_recent_sessions(sessions_dir, limit=25)
    clean = query.strip()
    if not clean:
        return None
    if clean.lower() in ("latest", "last"):
        return items[0].path if items else None
    if clean.isdigit():
        idx = int(clean)
        if 1 <= idx <= len(items):
            return items[idx - 1].path
    direct = sessions_dir / (clean if clean.endswith(".json") else f"{clean}.json")
    if direct.is_file():
        return direct
    for item in items:
        if item.session_id.lower() == clean.lower():
            return item.path
    return None


def render_sessions_picker(
    sessions: Sequence[SessionSummary],
    theme: TUITheme = DARK_THEME,
) -> Panel:
    """Render a numbered picker table for launch or ``/resume``."""
    tbl = Table(border_style=theme.border, box=None, pad_edge=False)
    tbl.add_column("#", style=theme.accent, justify="right", no_wrap=True)
    tbl.add_column("Session", style="bold", no_wrap=True)
    tbl.add_column("Model / Preset", style=theme.muted, no_wrap=True)
    tbl.add_column("Turns", justify="right", style=theme.muted)
    tbl.add_column("Saved", style=theme.dim, no_wrap=True)
    tbl.add_column("Last Prompt", style=theme.fg)
    for idx, s in enumerate(sessions, start=1):
        model_label = s.model or s.preset or "default"
        turn_label = f"{s.turn_count}" + (f" ({s.todo_count} todo)" if s.todo_count else "")
        tbl.add_row(
            f"[{idx}]",
            s.session_id,
            str(model_label),
            turn_label,
            s.saved_at or "—",
            s.last_prompt or "—",
        )
    return Panel(
        tbl,
        title=f"[{theme.accent}]Recent Sessions[/{theme.accent}] [{theme.dim}]· /resume <#> or /resume <name>[/{theme.dim}]",
        border_style=theme.border,
        padding=(0, 1),
    )


# --------------------------------------------- edit visibility & approval prompt

_FILE_MODIFYING_TOOLS: frozenset[str] = frozenset(
    {
        "replace_file_content",
        "edit_file",
        "write_to_file",
        "create_file",
        "apply_smart_patch",
        "multi_edit",
        "edit",
        "write",
        "builtintools.replace_file_content",
        "builtintools.edit_file",
        "builtintools.write_to_file",
        "builtintools.create_file",
    }
)


def is_file_modifying_tool(name: str) -> bool:
    """Return True when ``name`` is a file-modifying tool call."""
    clean = str(name).strip().lower()
    if ":" in clean:
        clean = clean.rsplit(":", 1)[-1]
    return clean in _FILE_MODIFYING_TOOLS


def format_approval_prompt(
    badge: str,
    summary: str,
    reason: str = "",
    theme: TUITheme = DARK_THEME,
    *,
    has_diff: bool = False,
) -> Text:
    """Build a compact approval prompt guaranteed to fit in at most 3 lines."""
    lines: list[Text] = []
    header = Text()
    header.append("  ? Allow ", style=theme.warning)
    header.append(badge, style=theme.accent)
    if summary:
        short_sum = " ".join(summary.splitlines()).strip()
        if len(short_sum) > 72:
            short_sum = short_sum[:69] + "..."
        header.append(f"({short_sum})", style=theme.fg)
    if has_diff:
        header.append(" [diff shown above]", style=theme.dim)
    header.append("  [y]es · [n]o · [a]lways", style=theme.muted)
    lines.append(header)

    if reason and reason.strip():
        clean_reason = " ".join(reason.strip().splitlines())
        if len(clean_reason) > 92:
            clean_reason = clean_reason[:89] + "..."
        r_line = Text()
        r_line.append(f"    reason: {clean_reason}", style=theme.muted)
        lines.append(r_line)

    # Strictly cap at 2 header lines so the interactive prompt line ("Allow execution? [y/N]: ")
    # brings total height to at most 3 lines.
    return Text("\n").join(lines[:2])
