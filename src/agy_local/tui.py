"""Rich Terminal UI (TUI) for agy-local inspired by Claude Code & OpenAI Codex CLI.

State-of-the-art interactive agent harness featuring:
- Claude Code / Codex style header card with live workspace, git branch & dirty counter,
  active model & backend, safety policy badge, autonomous Ralph loop status, and MCP indicators.
- Real-time Rich `Live` streaming:
  - Collapsible/dimmed reasoning (`💭 Thinking`) panel.
  - Live tool call timeline (`⏺ Bash(...)`, `⏺ Read(...)`, `⏺ Edit(...)`, `⏺ Write(...)`,
    `⏺ Grep(...)`, `⏺ ListDir(...)`) paired with real-time `PostToolCallHook` result previews,
    execution durations (`ms`), and error diagnostics.
  - Automatic syntax-highlighted unified diff previews (`+` green, `-` red) on file edits.
  - Live Markdown rendering with syntax-highlighted code blocks.
- Workspace Git Checkpoint & Rewind Engine (`/checkpoint`, `/rewind`, `/undo`, `/diff`):
  - Snapshots workspace state before agent edits and lets you rewind to any turn checkpoint.
- Autonomous Verification Loop (`/ralph`, `StopHook`):
  - Automatically runs a verification command (e.g. `pytest -q && ruff check`) when the model
    finishes a turn; if verification fails, feeds the failure output back into the model to
    self-repair up to `max_iterations`.
- Session Persistence & Export (`/save`, `/load`, `/export`, `/compact`):
  - Save/restore session transcripts & state in `~/.local/share/agy-local/sessions/` or export
    clean Markdown reports.
- Interactive `prompt_toolkit` prompt with keybindings (`Alt+T` toggle thoughts, `Alt+P` cycle
  policy), `/slash` command & sub-argument tab completion, `@file:start-end` & `@git:diff`
  inline context injection, and `!cmd` shell execution.
"""

from __future__ import annotations

import asyncio
import contextlib
import difflib
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from google.antigravity import Agent, types
from google.antigravity.hooks import hooks as hooks_mod
from google.antigravity.hooks import policy as policy_mod
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style as PtStyle
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from agy_local.ast_map import build_workspace_index, render_compact_repo_map, scout_workspace
from agy_local.builder import HarnessOptions, PolicyMode, build_agent_config
from agy_local.discovery import discover_local_models, probe_local_endpoints
from agy_local.harness_hooks import get_harness_telemetry
from agy_local.integrate import integrate_frontier_agent
from agy_local.presets import PRESETS, BackendKind, list_presets
from agy_local.updater import UserConfig, check_for_updates, perform_self_update

_NOISY_STDERR_SUBSTRINGS: tuple[str, ...] = (
    "maxDynamicUniformBuffersPerPipelineLayout artificially reduced",
    "maxDynamicStorageBuffersPerPipelineLayout artificially reduced",
)


@contextlib.contextmanager
def suppress_noisy_cpp_stderr() -> Iterator[None]:
    """Filters harmless C++ Dawn/WebGPU pipeline layout warnings written directly to fd 2."""
    try:
        orig_fd = os.dup(2)
    except OSError:
        yield
        return

    r_fd, w_fd = os.pipe()
    os.dup2(w_fd, 2)
    os.close(w_fd)

    def _forward() -> None:
        buf = b""
        while True:
            try:
                chunk = os.read(r_fd, 4096)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line_bytes, buf = buf.split(b"\n", 1)
                    line_str = line_bytes.decode("utf-8", errors="replace")
                    if not any(pat in line_str for pat in _NOISY_STDERR_SUBSTRINGS):
                        os.write(orig_fd, line_bytes + b"\n")
            except OSError:
                break
        if buf:
            line_str = buf.decode("utf-8", errors="replace")
            if not any(pat in line_str for pat in _NOISY_STDERR_SUBSTRINGS):
                with contextlib.suppress(OSError):
                    os.write(orig_fd, buf)
        with contextlib.suppress(OSError):
            os.close(r_fd)
            os.close(orig_fd)

    reader = threading.Thread(target=_forward, daemon=True)
    reader.start()
    try:
        yield
    finally:
        with contextlib.suppress(OSError):
            sys.stderr.flush()
            os.dup2(orig_fd, 2)
        reader.join(timeout=0.5)

SLASH_COMMANDS: dict[str, str] = {
    "/help": "Show interactive slash commands, keybindings, @file mentions, and !shell shortcuts",
    "/status": "Show session token usage, context window utilization bar, turn count, and telemetry",
    "/commons": "Inspect, search, review, or sync OpenCommons research (/commons \\[status|find|review|approve|reject|pull|push|seed])",
    "/ast": "Render or toggle automatic AST Repo-Map context injection (/ast \\[show|on|off])",
    "/scout": "Run the AST Context Scout on a query to extract surgical code slices (/scout <query>)",
    "/integrate": "Install Kardashev Co-Processor MCP & skill into frontier agents (/integrate \\[antigravity|claude|cursor|codex|all])",
    "/preset": "Switch to a curated model preset (e.g., /preset gemma4-26b or /preset glm4-9b-ollama)",
    "/model": "Switch model ID or OpenAI base URL (e.g., /model qwen3-vl-8b http://127.0.0.1:8000/v1)",
    "/policy": "Switch tool execution policy mode (/policy allow-all | ask | sandbox)",
    "/ralph": "Configure autonomous self-verification loop (e.g., /ralph pytest -q | /ralph off)",
    "/checkpoint": "Create or list workspace git checkpoints (/checkpoint \\[create|list])",
    "/rewind": "Rewind workspace files to a numbered checkpoint (/rewind <index>)",
    "/diff": "Show colorized git diff of uncommitted changes in the workspace",
    "/undo": "Revert unstaged file modifications in the workspace (with confirmation)",
    "/compact": "Summarize conversation history to free up context window tokens",
    "/save": "Save session transcript and metadata (/save \\[name])",
    "/load": "Load and resume a saved session (/load <name>)",
    "/export": "Export full conversation & tool trace to a Markdown file (/export \\[path.md])",
    "/config": "View or save persistent user defaults in ~/.config/agy-local/config.json",
    "/update": "Self-update kardashev-cli from GitHub main and refresh ~/.local/bin symlinks",
    "/models": "Scan local disk checkpoints and probe localhost OpenAI servers",
    "/doctor": "Run hardware, GPU, LiteRT, and MCP diagnostics",
    "/thoughts": "Toggle display of model reasoning/thinking traces (or press Alt+T)",
    "/clear": "Clear conversation history and terminal screen",
    "/exit": "Exit interactive session (or Ctrl+D)",
}


@dataclass
class ToolEventRecord:
    """Tracks a single tool invocation from `ToolCall` through `PostToolCallHook` completion."""

    call_id: str
    name: str
    badge: str
    summary: str
    diff_preview: str | None = None
    started_at: float = field(default_factory=time.perf_counter)
    completed_at: float | None = None
    result_preview: str | None = None
    is_error: bool = False

    @property
    def duration_ms(self) -> float | None:
        if self.completed_at is None:
            return None
        return max(0.0, (self.completed_at - self.started_at) * 1000.0)


@dataclass
class WorkspaceCheckpoint:
    """Represents a git stash/tree snapshot created before or during a session."""

    index: int
    turn: int
    timestamp: str
    label: str
    commit_oid: str


@dataclass
class TurnTranscript:
    """Stores a completed turn for `/save`, `/load`, `/export`, and `/compact`."""

    turn_index: int
    user_prompt: str
    thinking: str
    assistant_response: str
    tools: list[dict[str, Any]]
    duration_s: float
    prompt_tokens: int
    output_tokens: int


# Module-level active TUI pointer used by stateless SDK hooks (avoids pickling RLock on Console).
_ACTIVE_TUI: AgyRichTUI | None = None


def _git_branch_and_dirty(workspace: str) -> tuple[str | None, int]:
    """Returns `(branch_name, dirty_file_count)` if `workspace` is inside a git repo."""
    try:
        br = subprocess.run(
            ["git", "-C", workspace, "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=1.5,
            check=False,
        )
        if br.returncode != 0:
            return None, 0
        branch = br.stdout.strip()
        st = subprocess.run(
            ["git", "-C", workspace, "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=1.5,
            check=False,
        )
        dirty = len([ln for ln in st.stdout.splitlines() if ln.strip()]) if st.returncode == 0 else 0
        return branch, dirty
    except (OSError, subprocess.SubprocessError):
        return None, 0


def create_git_checkpoint(workspace: str, index: int, turn: int, label: str) -> WorkspaceCheckpoint | None:
    """Creates a non-destructive git snapshot OID (via `git stash create` or `HEAD`) for `/rewind`."""
    try:
        stash_proc = subprocess.run(
            ["git", "-C", workspace, "stash", "create", f"agy-local-checkpoint-{index}"],
            capture_output=True,
            text=True,
            timeout=3.0,
            check=False,
        )
        oid = stash_proc.stdout.strip()
        if not oid:
            head_proc = subprocess.run(
                ["git", "-C", workspace, "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=2.0,
                check=False,
            )
            if head_proc.returncode != 0:
                return None
            oid = head_proc.stdout.strip()
        ts = time.strftime("%H:%M:%S")
        return WorkspaceCheckpoint(index=index, turn=turn, timestamp=ts, label=label, commit_oid=oid)
    except (OSError, subprocess.SubprocessError):
        return None


def restore_git_checkpoint(workspace: str, checkpoint: WorkspaceCheckpoint) -> bool:
    """Restores tracked workspace files to the state captured in `checkpoint.commit_oid`."""
    try:
        proc = subprocess.run(
            ["git", "-C", workspace, "checkout", checkpoint.commit_oid, "--", "."],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
        return proc.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def format_tool_badge(call: types.ToolCall) -> tuple[str, str, str | None]:
    """Formats a ToolCall into `(badge_label, summary_arg, diff_preview)` in Claude Code / Codex style."""
    name = str(call.name)
    args = call.args or {}

    if name in ("run_command", "BuiltinTools.RUN_COMMAND"):
        cmd = str(args.get("CommandLine", "")).strip()
        cwd = str(args.get("Cwd", ""))
        short_cmd = cmd if len(cmd) <= 88 else cmd[:85] + "..."
        sub = f"{short_cmd}  (in {os.path.basename(cwd) or cwd})" if cwd else short_cmd
        return ("Bash", sub, None)

    if name in ("view_file", "BuiltinTools.VIEW_FILE"):
        path = str(args.get("AbsolutePath", ""))
        start = args.get("StartLine")
        end = args.get("EndLine")
        loc = f":{start}-{end}" if start or end else ""
        return ("Read", f"{path}{loc}", None)

    if name in ("replace_file_content", "BuiltinTools.REPLACE_FILE_CONTENT"):
        path = str(args.get("TargetFile", ""))
        old = str(args.get("TargetContent", ""))
        new = str(args.get("ReplacementContent", ""))
        diff_lines = list(
            difflib.unified_diff(
                old.splitlines(),
                new.splitlines(),
                fromfile=f"a/{os.path.basename(path)}",
                tofile=f"b/{os.path.basename(path)}",
                lineterm="",
                n=2,
            )
        )
        diff_text = "\n".join(diff_lines[:28])
        if len(diff_lines) > 28:
            diff_text += f"\n... ({len(diff_lines) - 28} more diff lines)"
        return ("Edit", path, diff_text or None)

    if name in ("write_to_file", "BuiltinTools.WRITE_TO_FILE"):
        path = str(args.get("TargetFile", ""))
        content = str(args.get("CodeContent", ""))
        line_count = len(content.splitlines())
        preview_lines = [f"+ {ln}" for ln in content.splitlines()[:12]]
        if line_count > 12:
            preview_lines.append(f"... (+{line_count - 12} more lines)")
        return ("Write", f"{path} ({line_count} lines)", "\n".join(preview_lines) or None)

    if name in ("list_dir", "BuiltinTools.LIST_DIR"):
        path = str(args.get("DirectoryPath", args.get("Path", ".")))
        return ("ListDir", path, None)

    if name in ("grep_search", "grep", "BuiltinTools.GREP"):
        query = str(args.get("Query", args.get("Pattern", "")))
        search_path = str(args.get("SearchPath", "."))
        return ("Grep", f"'{query}' in {search_path}", None)

    # Generic / MCP tool call
    compact_args = ", ".join(f"{k}={repr(v)[:40]}" for k, v in list(args.items())[:3])
    return (name, compact_args, None)


def summarize_tool_result(tool_result: types.ToolResult) -> tuple[str, bool]:
    """Formats a `types.ToolResult` into `(one_line_summary, is_error)` for the live TUI tree."""
    if tool_result.error or tool_result.exception:
        err_text = str(tool_result.error or tool_result.exception).strip().splitlines()[0]
        return (f"✗ {err_text[:96]}", True)
    raw = str(tool_result.result or "").strip()
    if not raw:
        return ("✓ completed", False)
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    line_count = len(raw.splitlines())
    first_line = lines[0][:84] if lines else "done"
    if line_count > 1:
        return (f"✓ {first_line} (+{line_count - 1} lines)", False)
    return (f"✓ {first_line}", False)


def expand_at_file_mentions(prompt_text: str, workspace: str) -> tuple[str, list[str]]:
    """Detects `@path/to/file`, `@path/to/file:10-50`, `@git:diff`, and `@git:status` mentions."""
    tokens = prompt_text.split()
    attached_files: list[str] = []
    context_blocks: list[str] = []

    for tok in tokens:
        if not tok.startswith("@") or len(tok) < 2:
            continue
        raw_spec = tok[1:].strip(",.:;!?")
        if not raw_spec:
            continue

        # Special @git:diff and @git:status context providers
        if raw_spec in ("git:diff", "git:status"):
            git_args = ["diff"] if raw_spec == "git:diff" else ["status", "--short"]
            try:
                proc = subprocess.run(
                    ["git", "-C", workspace, *git_args],
                    capture_output=True,
                    text=True,
                    timeout=3.0,
                    check=False,
                )
                git_out = proc.stdout.strip() or "(empty)"
                if len(git_out) > 20000:
                    git_out = git_out[:20000] + "\n... [truncated]"
                attached_files.append(f"@{raw_spec}")
                context_blocks.append(f"<git_context type=\"{raw_spec}\">\n{git_out}\n</git_context>")
            except (OSError, subprocess.SubprocessError):
                pass
            continue

        # Parse optional :start-end line range (e.g. @src/agy_local/cli.py:10-45)
        line_start: int | None = None
        line_end: int | None = None
        m = re.match(r"^(.+?):(\d+)(?:-(\d+))?$", raw_spec)
        if m:
            rel_candidate = m.group(1)
            line_start = int(m.group(2))
            line_end = int(m.group(3)) if m.group(3) else line_start
        else:
            rel_candidate = raw_spec

        candidate_path = Path(workspace) / rel_candidate
        if not candidate_path.is_file():
            candidate_path = Path(os.path.expanduser(rel_candidate))
        if candidate_path.is_file():
            try:
                text = candidate_path.read_text(encoding="utf-8")
                if line_start is not None:
                    all_lines = text.splitlines()
                    s_idx = max(0, line_start - 1)
                    e_idx = min(len(all_lines), line_end or line_start)
                    text = "\n".join(all_lines[s_idx:e_idx])
                if len(text) > 24000:
                    text = text[:24000] + "\n... [truncated]"
                rel_display = (
                    str(candidate_path.relative_to(workspace))
                    if candidate_path.is_relative_to(workspace)
                    else str(candidate_path)
                )
                range_suffix = f":{line_start}-{line_end}" if line_start is not None else ""
                label = f"{rel_display}{range_suffix}"
                attached_files.append(label)
                context_blocks.append(f"<attached_file path=\"{label}\">\n{text}\n</attached_file>")
            except (UnicodeDecodeError, OSError):
                pass

    if not context_blocks:
        return prompt_text, []
    augmented = "\n\n".join(context_blocks) + "\n\n" + prompt_text
    return augmented, attached_files


class AgyCompleter(Completer):
    """Provides tab completion for `/slash` commands, presets, policies, and `@file` / `@git:` mentions."""

    def __init__(self, workspace: str) -> None:
        self.workspace = workspace

    def get_completions(self, document: Any, complete_event: Any) -> Any:
        text_before = document.text_before_cursor
        word = document.get_word_before_cursor(WORD=True)

        # 1. Slash command completion at start of line
        if text_before.lstrip().startswith("/"):
            parts = text_before.lstrip().split()
            if len(parts) <= 1 and not text_before.endswith(" "):
                prefix = parts[0] if parts else "/"
                for cmd, desc in SLASH_COMMANDS.items():
                    if cmd.startswith(prefix):
                        yield Completion(
                            cmd,
                            start_position=-len(prefix),
                            display=cmd,
                            display_meta=desc,
                        )
                return

            # Sub-argument completion for /preset, /policy, /ralph, /checkpoint
            cmd_name = parts[0].lower()
            cur_arg = parts[1] if len(parts) > 1 and not text_before.endswith(" ") else ""
            if cmd_name == "/preset":
                for p in list_presets():
                    if p.key.startswith(cur_arg):
                        yield Completion(
                            p.key,
                            start_position=-len(cur_arg),
                            display=p.key,
                            display_meta=f"{p.backend.value} · {p.model_id}",
                        )
                return
            if cmd_name == "/policy":
                for mode in ("allow-all", "ask", "sandbox"):
                    if mode.startswith(cur_arg):
                        yield Completion(mode, start_position=-len(cur_arg), display=mode)
                return
            if cmd_name == "/ralph":
                for opt in ("off", "pytest -q", "ruff check src tests && pytest -q"):
                    if opt.startswith(cur_arg):
                        yield Completion(opt, start_position=-len(cur_arg), display=opt)
                return
            if cmd_name == "/checkpoint":
                for opt in ("create", "list"):
                    if opt.startswith(cur_arg):
                        yield Completion(opt, start_position=-len(cur_arg), display=opt)
                return
            if cmd_name == "/commons":
                for opt in ("status", "find", "review", "approve", "reject", "seed"):
                    if opt.startswith(cur_arg):
                        yield Completion(opt, start_position=-len(cur_arg), display=opt)
                return

        # 2. @file and @git: completion
        if word.startswith("@"):
            partial = word[1:]
            for special, meta in (("git:diff", "git uncommitted diff"), ("git:status", "git working tree status")):
                if special.startswith(partial):
                    yield Completion(
                        f"@{special}",
                        start_position=-len(word),
                        display=f"@{special}",
                        display_meta=meta,
                    )
            root = Path(self.workspace)
            try:
                search_dir = (root / os.path.dirname(partial)).resolve()
                prefix_name = os.path.basename(partial)
                if search_dir.is_dir() and search_dir.is_relative_to(root.resolve()):
                    for child in sorted(search_dir.iterdir()):
                        if child.name.startswith(".") and not prefix_name.startswith("."):
                            continue
                        if child.name in (".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache"):
                            continue
                        if child.name.startswith(prefix_name):
                            rel = str(child.relative_to(root.resolve()))
                            if child.is_dir():
                                rel += "/"
                            yield Completion(
                                f"@{rel}",
                                start_position=-len(word),
                                display=f"@{rel}",
                                display_meta="dir" if child.is_dir() else "file",
                            )
            except OSError:
                pass


class RichAskQuestionHook(hooks_mod.OnInteractionHook):
    """Rich-styled interactive question handler when the model invokes `ask_question`."""

    async def run(
        self, context: hooks_mod.HookContext, data: types.AskQuestionInteractionSpec
    ) -> hooks_mod.QuestionHookResult:
        console = Console()
        responses: list[types.QuestionResponse] = []
        for q in data.questions:
            console.print()
            console.print(Panel(f"[bold cyan]? {q.question}[/bold cyan]", border_style="cyan"))
            options = list(q.options) if hasattr(q, "options") else []
            for idx, opt in enumerate(options):
                console.print(f"  [bold yellow]{idx + 1}.[/bold yellow] {opt.text}")
            ans = await _nonblocking_input("  Select option # or type answer: ")
            ans = ans.strip()
            if not ans:
                responses.append(types.QuestionResponse(skipped=True))
                continue
            matched_id = None
            if options:
                try:
                    sel = int(ans) - 1
                    if 0 <= sel < len(options):
                        matched_id = options[sel].id
                except ValueError:
                    pass
            if matched_id:
                responses.append(types.QuestionResponse(selected_option_ids=[matched_id]))
            else:
                responses.append(types.QuestionResponse(freeform_response=ans))
        return hooks_mod.QuestionHookResult(responses=responses)


class RichPostToolCallHook(hooks_mod.PostToolCallHook):
    """Captures real-time tool completion results and durations for the live TUI tree."""

    async def run(self, context: hooks_mod.HookContext, data: types.ToolResult) -> None:
        if _ACTIVE_TUI is not None:
            _ACTIVE_TUI.record_tool_result(data)


class RichCompactionHook(hooks_mod.OnCompactionHook):
    """Notifies the active TUI whenever automatic KV-cache / context compaction occurs."""

    async def run(self, context: hooks_mod.HookContext, data: types.Step) -> None:
        if _ACTIVE_TUI is not None:
            _ACTIVE_TUI.compaction_notifications += 1


class RalphVerifyStopHook(hooks_mod.StopHook):
    """Autonomous verification StopHook (`/ralph <cmd>`) that runs a verification command before stopping."""

    async def run(
        self, context: hooks_mod.HookContext, data: types.StopArgs
    ) -> types.StopHookResult:
        tui = _ACTIVE_TUI
        if tui is None or not tui.ralph_verify_cmd:
            return types.StopHookResult(decision=types.StopDecision.ALLOW_STOP)
        if data.continuation_count >= tui.ralph_max_iterations:
            return types.StopHookResult(decision=types.StopDecision.ALLOW_STOP)

        workspace = tui.options.workspaces[0] if tui.options.workspaces else os.getcwd()
        proc = await asyncio.to_thread(
            subprocess.run,
            tui.ralph_verify_cmd,
            shell=True,
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if proc.returncode == 0:
            tui.last_ralph_status = f"✓ PASS ({tui.ralph_verify_cmd})"
            return types.StopHookResult(decision=types.StopDecision.ALLOW_STOP)

        combined_out = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
        tail_out = "\n".join(combined_out.splitlines()[-30:])
        tui.last_ralph_status = (
            f"↻ Retry {data.continuation_count + 1}/{tui.ralph_max_iterations} "
            f"(exit {proc.returncode})"
        )
        return types.StopHookResult(
            decision=types.StopDecision.CONTINUE,
            reason=(
                f"Autonomous verification command `{tui.ralph_verify_cmd}` failed with exit code "
                f"{proc.returncode}. Inspect the failure output below and fix the root cause:\n\n"
                f"```\n{tail_out}\n```"
            ),
        )


async def _nonblocking_input(prompt_text: str) -> str:
    loop = asyncio.get_running_loop()
    fut = loop.create_future()

    def _read() -> None:
        try:
            res = input(prompt_text)
            if not fut.cancelled():
                loop.call_soon_threadsafe(fut.set_result, res)
        except BaseException as exc:  # noqa: BLE001
            if not fut.cancelled():
                loop.call_soon_threadsafe(fut.set_exception, exc)

    t = threading.Thread(target=_read, daemon=True)
    t.start()
    return await fut


async def _confirm_tool_handler(tc: types.ToolCall, reason: str = "") -> bool:
    console = Console()
    badge, summary, diff_preview = format_tool_badge(tc)
    console.print()
    body = f"[bold yellow]{badge}[/bold yellow] [white]{summary}[/white]"
    if reason:
        body += f"\n[dim]Reason: {reason}[/dim]"
    console.print(Panel(body, title="[bold yellow]Permission Required[/bold yellow]", border_style="yellow"))
    if diff_preview:
        console.print(Syntax(diff_preview, "diff", theme="monokai", word_wrap=True))
    ans = await _nonblocking_input("  Allow execution? [y/N]: ")
    return ans.strip().lower() in ("y", "yes")


def render_context_gauge(used_tokens: int, limit_tokens: int, width: int = 16) -> str:
    """Renders a visual Unicode progress bar for context window utilization."""
    pct = min(1.0, max(0.0, used_tokens / max(1, limit_tokens)))
    filled = round(pct * width)
    bar = "█" * filled + "░" * (width - filled)
    color = "green" if pct < 0.65 else ("yellow" if pct < 0.85 else "red")
    return f"[{color}]{bar}[/{color}] {pct * 100:.1f}%"


class AgyRichTUI:
    """Claude Code & OpenAI Codex inspired interactive TUI session manager."""

    def __init__(
        self,
        options: HarnessOptions,
        console: Console | None = None,
    ) -> None:
        self.options = options
        self.console = console or Console()
        self.show_thoughts = True
        self.auto_inject_ast_map = True
        self.session_turns = 0
        self.session_tool_calls = 0
        self.total_prompt_tokens = 0
        self.total_output_tokens = 0
        self.compaction_notifications = 0
        self.ralph_verify_cmd: str | None = None
        self.ralph_max_iterations: int = 3
        self.last_ralph_status: str | None = None
        self.checkpoints: list[WorkspaceCheckpoint] = []
        self.transcripts: list[TurnTranscript] = []
        self._active_turn_tools: list[ToolEventRecord] = []
        self._restart_requested = False

    def record_tool_result(self, result: types.ToolResult) -> None:
        """Matches a `PostToolCallHook` result to the most recent in-flight `ToolEventRecord`."""
        summary, is_err = summarize_tool_result(result)
        now = time.perf_counter()
        for rec in reversed(self._active_turn_tools):
            if (result.id and rec.call_id == result.id) or (
                rec.completed_at is None and rec.name == str(result.name)
            ):
                rec.completed_at = now
                rec.result_preview = summary
                rec.is_error = is_err
                return
        # Fallback if no matching call_id was seen yet
        self._active_turn_tools.append(
            ToolEventRecord(
                call_id=result.id or str(result.name),
                name=str(result.name),
                badge=str(result.name),
                summary="",
                completed_at=now,
                result_preview=summary,
                is_error=is_err,
            )
        )

    def _active_model_summary(self, cfg: Any) -> tuple[str, str, int]:
        cls_name = type(cfg).__name__
        compaction = getattr(cfg, "compaction_config", None)
        ctx_limit = getattr(compaction, "token_threshold", None) or 40960
        if hasattr(cfg, "model_path"):
            path = str(getattr(cfg, "model_path", ""))
            model_short = (
                os.path.basename(os.path.dirname(path))
                if path.endswith("model.litertlm")
                else os.path.basename(path)
            )
            dev = str(getattr(cfg, "backend", "gpu")).replace("LiteRTBackend.", "")
            spec = " + speculative" if getattr(cfg, "enable_speculative_decoding", False) else ""
            return (model_short or "litert", f"LiteRT ({dev}{spec})", ctx_limit)
        model_name = str(getattr(cfg, "model", "openai-model"))
        base_url = str(getattr(cfg, "base_url", "http://127.0.0.1:8000/v1"))
        return (model_name, f"{cls_name} @ {base_url}", ctx_limit)

    def render_welcome_banner(self, cfg: Any) -> Panel:
        """Builds the Claude Code / Codex style welcome status card."""
        workspace = self.options.workspaces[0] if self.options.workspaces else os.getcwd()
        branch, dirty = _git_branch_and_dirty(workspace)
        model_label, backend_label, ctx_limit = self._active_model_summary(cfg)
        policy_val = (
            self.options.policy_mode.value
            if isinstance(self.options.policy_mode, PolicyMode)
            else str(self.options.policy_mode)
        )
        policy_color = {
            "allow-all": "bold green",
            "sandbox": "bold cyan",
            "ask": "bold yellow",
        }.get(policy_val, "bold white")

        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="bold dim", justify="right")
        grid.add_column()

        git_badge = ""
        if branch:
            dirty_str = f" [yellow](*{dirty} modified)[/yellow]" if dirty else " [green](clean)[/green]"
            git_badge = f"  [dim]git:[/dim][bold magenta]{branch}[/bold magenta]{dirty_str}"

        grid.add_row("Model", f"[bold cyan]{model_label}[/bold cyan]  [dim]· {backend_label}[/dim]")
        grid.add_row("Workspace", f"[bold white]{workspace}[/bold white]{git_badge}")
        mcp_list = ["chronosml"] if self.options.with_chronosml_mcp else []
        if self.options.with_opencommons_mcp:
            mcp_list.append("opencommons")
        mcp_list.extend(name for name, _, _ in self.options.extra_mcp_stdio)
        mcp_str = ", ".join(mcp_list) if mcp_list else "none (use --with-chronosml to enable)"
        ralph_badge = (
            f"  [dim]· Ralph Verify:[/dim] [bold green]{self.ralph_verify_cmd}[/bold green]"
            if self.ralph_verify_cmd
            else ""
        )
        grid.add_row(
            "Safety & Context",
            f"[{policy_color}]{policy_val.upper()}[/{policy_color}]  [dim]· compaction ceiling:[/dim] {ctx_limit:,} tok  [dim]· MCP:[/dim] {mcp_str}{ralph_badge}",
        )
        ast_state = "ON" if self.auto_inject_ast_map else "OFF"
        oc_state = "ON" if self.options.enable_opencommons else "OFF"
        grid.add_row(
            "Cognitive Engine",
            f"[bold green]4-Phase CoT[/bold green] · [bold cyan]AST Repo-Map ({ast_state})[/bold cyan] · [bold yellow]Self-Healing Fuzzy Edits[/bold yellow] · [bold bright_magenta]OpenCommons ({oc_state})[/bold bright_magenta]",
        )
        grid.add_row(
            "Shortcuts",
            "[dim][bold]/help[/bold] commands · [bold]/commons[/bold] research · [bold]/scout[/bold] AST search · [bold]/integrate[/bold] frontier agents · [bold]@file:1-50[/bold] · [bold]!cmd[/bold][/dim]",
        )
        upd = check_for_updates(force=False)
        if upd.update_available:
            grid.add_row(
                "Update",
                f"[bold yellow]⬆ {upd.message}[/bold yellow] [dim](type /update)[/dim]",
            )

        return Panel(
            grid,
            title="[bold bright_white]✦ Kardashev CLI · Local & Distributed Agent Studio[/bold bright_white] [dim](kardashev / ksh / agy v0.2.0)[/dim]",
            subtitle="[dim]The Kardashev Project · Engineered AST, CoT, OpenCommons & Self-Healing Harness[/dim]",
            border_style="bright_blue",
            padding=(0, 1),
        )

    def _render_connection_error(self, exc: Exception, cfg: Any) -> None:
        """Renders a diagnostic panel when a local/remote inference endpoint fails without crashing the TUI."""
        model_label, backend_label, _ = self._active_model_summary(cfg)
        base_url = str(getattr(cfg, "base_url", "http://127.0.0.1:8000/v1"))
        err_str = str(exc)
        is_conn_refused = "connection refused" in err_str.lower() or "failed to send http request" in err_str.lower()

        if is_conn_refused:
            body = (
                f"[bold red]✗ Could not connect to OpenAI inference server at `{base_url}`[/bold red]\n"
                f"[dim]{err_str}[/dim]\n\n"
                f"[bold white]How to resolve:[/bold white]\n"
                f"1. Start the local server in another terminal:\n"
                f"   [bold green]kardashev serve --model {model_label}[/bold green]\n"
                f"2. Or switch immediately to an in-process LiteRT GPU/NPU model (zero server required):\n"
                f"   [bold cyan]/preset gemma4-26b[/bold cyan]  or  [bold cyan]/models[/bold cyan]"
            )
            title = "[bold red]Inference Endpoint Unreachable[/bold red]"
        else:
            body = (
                f"[bold red]✗ Model execution error on `{model_label}` ({backend_label}):[/bold red]\n"
                f"[dim]{err_str}[/dim]\n\n"
                f"[dim]Tip: Use `/preset gemma4-26b` or `/models` to switch models, or `/doctor` to inspect hardware.[/dim]"
            )
            title = "[bold red]Agent Execution Error[/bold red]"

        self.console.print()
        self.console.print(Panel(body, title=title, border_style="red", padding=(0, 1)))
        self.console.print()

    def _upgrade_interactive_config(self, cfg: Any) -> Any:
        hooks_list = list(getattr(cfg, "hooks", []) or [])
        if not any(isinstance(h, RichAskQuestionHook) for h in hooks_list):
            hooks_list.append(RichAskQuestionHook())
        if not any(isinstance(h, RichPostToolCallHook) for h in hooks_list):
            hooks_list.append(RichPostToolCallHook())
        if not any(isinstance(h, RichCompactionHook) for h in hooks_list):
            hooks_list.append(RichCompactionHook())
        if not any(isinstance(h, RalphVerifyStopHook) for h in hooks_list):
            hooks_list.append(RalphVerifyStopHook())

        capabilities = cfg.capabilities
        if capabilities.agent_behavior != types.AgentBehavior.INTERACTIVE:
            capabilities = capabilities.model_copy(
                update={"agent_behavior": types.AgentBehavior.INTERACTIVE}
            )

        policies_list = list(getattr(cfg, "policies", []) or [])
        upgraded_policies = []
        for p in policy_mod.flatten_policies(policies_list):
            if (
                p.tool == types.BuiltinTools.RUN_COMMAND.value
                and p.decision == policy_mod.Decision.DENY
                and p.when is None
            ):
                upgraded_policies.append(
                    policy_mod.ask_user(
                        types.BuiltinTools.RUN_COMMAND.value,
                        handler=_confirm_tool_handler,
                        name=p.name or "interactive_confirm",
                    )
                )
            else:
                upgraded_policies.append(p)

        return cfg.model_copy(
            update={
                "hooks": hooks_list,
                "capabilities": capabilities,
                "policies": upgraded_policies,
            }
        )

    def _sessions_dir(self) -> Path:
        d = Path.home() / ".local" / "share" / "agy-local" / "sessions"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def export_markdown(self, target_path: Path, cfg: Any) -> Path:
        """Exports the current session transcript and tool trace to a Markdown report."""
        model_label, backend_label, _ = self._active_model_summary(cfg)
        lines = [
            f"# Antigravity Local Studio Session (`{model_label}`)",
            "",
            f"- **Backend**: `{backend_label}`",
            f"- **Turns**: {self.session_turns}",
            f"- **Tool Calls**: {self.session_tool_calls}",
            f"- **Tokens**: {self.total_prompt_tokens:,} prompt / {self.total_output_tokens:,} output",
            "",
            "---",
            "",
        ]
        for tr in self.transcripts:
            lines.append(f"## Turn {tr.turn_index} ({tr.duration_s:.2f}s)")
            lines.append("")
            lines.append("### User")
            lines.append(tr.user_prompt)
            lines.append("")
            if tr.thinking:
                lines.append("<details><summary>💭 Thinking Trace</summary>\n")
                lines.append(tr.thinking)
                lines.append("\n</details>\n")
            if tr.tools:
                lines.append("### Tool Calls")
                for tc in tr.tools:
                    dur = f" ({tc['duration_ms']:.0f}ms)" if tc.get("duration_ms") is not None else ""
                    res = f" → {tc.get('result_preview', '')}" if tc.get("result_preview") else ""
                    lines.append(f"- **`{tc['badge']}({tc['summary']})`**{dur}{res}")
                lines.append("")
            lines.append("### Assistant")
            lines.append(tr.assistant_response)
            lines.append("")
            lines.append("---")
            lines.append("")
        target_path.write_text("\n".join(lines), encoding="utf-8")
        return target_path

    async def _handle_slash_command(self, raw_cmd: str, agent: Agent, cfg: Any) -> bool:
        """Handles `/...` commands. Returns True if session should exit or restart."""
        parts = raw_cmd.strip().split()
        cmd = parts[0].lower()
        args = parts[1:]
        workspace = self.options.workspaces[0] if self.options.workspaces else os.getcwd()

        if cmd in ("/exit", "/quit"):
            self.console.print("[dim]Session closed.[/dim]")
            return True

        if cmd == "/help":
            tbl = Table(title="Interactive Commands, Context Mentions & Keybindings", border_style="bright_blue")
            tbl.add_column("Shortcut / Command", style="bold cyan", no_wrap=True)
            tbl.add_column("Description", style="white")
            for k, desc in SLASH_COMMANDS.items():
                tbl.add_row(k, desc)
            tbl.add_row("@path/to/file[:start-end]", "Inline a workspace file or line slice into your prompt")
            tbl.add_row("@git:diff / @git:status", "Inline live uncommitted git diff or status into your prompt")
            tbl.add_row("!<shell command>", "Execute a local shell command immediately without calling the model")
            tbl.add_row("Alt+T / Alt+P", "Keybindings: toggle thinking display (Alt+T) or cycle policy (Alt+P)")
            self.console.print(tbl)
            return False

        if cmd == "/status":
            model_label, backend_label, ctx_limit = self._active_model_summary(cfg)
            usage = agent.conversation.total_usage
            p_tok = (usage.prompt_token_count or 0) if usage and usage.prompt_token_count else self.total_prompt_tokens
            c_tok = (
                (usage.candidates_token_count or 0)
                if usage and usage.candidates_token_count
                else self.total_output_tokens
            )
            t_tok = (usage.thoughts_token_count or 0) if usage else 0
            tot_tok = (usage.total_token_count or 0) if usage and usage.total_token_count else (p_tok + c_tok)
            gauge = render_context_gauge(tot_tok, ctx_limit)
            htel = get_harness_telemetry()

            tbl = Table(title="Session Telemetry & Engineered Harness Status", border_style="cyan")
            tbl.add_column("Metric", style="bold dim")
            tbl.add_column("Value", style="white")
            tbl.add_row("Active Model", f"[bold cyan]{model_label}[/bold cyan] ({backend_label})")
            tbl.add_row("Conversation ID", agent.conversation.conversation_id or "(local)")
            tbl.add_row("Completed Turns", str(self.session_turns))
            tbl.add_row("Executed Tool Calls", str(self.session_tool_calls))
            tbl.add_row("Prompt / Output / Thought Tokens", f"{p_tok:,} in · {c_tok:,} out · {t_tok:,} thought")
            tbl.add_row("Context Window Utilization", f"{gauge}  ({tot_tok:,} / {ctx_limit:,} tok)")
            tbl.add_row(
                "Self-Healing Edits & Loop Blocks",
                f"[bold green]{htel.healed_edits_count} healed[/bold green] · "
                f"[bold yellow]{htel.loop_blocks_count} loop(s) blocked[/bold yellow] · "
                f"[cyan]{htel.ast_maps_injected} AST map(s)[/cyan]",
            )
            tbl.add_row(
                "OpenCommons Research Activity",
                f"[bold magenta]{htel.commons_cards_injected} card(s) injected[/bold magenta] · "
                f"[bold green]{htel.commons_concepts_recorded} recorded[/bold green] · "
                f"[yellow]{htel.commons_proposals_queued} queued for review[/yellow] · "
                f"[red]{htel.commons_counterexamples_reported} counterexample(s)[/red]",
            )
            tbl.add_row(
                "Compaction Events",
                str(max(len(agent.conversation.compaction_indices), self.compaction_notifications)),
            )
            tbl.add_row("Workspace Checkpoints", f"{len(self.checkpoints)} saved (use /checkpoint or /rewind)")
            tbl.add_row(
                "Autonomous Ralph Loop",
                f"[green]{self.ralph_verify_cmd}[/green]" if self.ralph_verify_cmd else "[dim]disabled (/ralph <cmd>)[/dim]",
            )
            self.console.print(tbl)
            return False

        if cmd == "/clear":
            agent.conversation.clear_history()
            self.transcripts.clear()
            self.console.clear()
            self.console.print(self.render_welcome_banner(cfg))
            self.console.print("[green]✓ Cleared conversation history.[/green]")
            return False

        if cmd == "/thoughts":
            self.show_thoughts = not self.show_thoughts
            state = "ENABLED" if self.show_thoughts else "HIDDEN"
            self.console.print(f"[cyan]✓ Reasoning trace display: {state}[/cyan]")
            return False

        if cmd == "/ralph":
            if not args:
                state = self.ralph_verify_cmd or "disabled"
                self.console.print(
                    f"[cyan]Autonomous Ralph verification command:[/cyan] [bold]{state}[/bold]\n"
                    "[dim]Usage: /ralph <shell_command> (e.g., `/ralph pytest -q`) or `/ralph off`[/dim]"
                )
                return False
            if args[0].lower() in ("off", "none", "disable"):
                self.ralph_verify_cmd = None
                self.last_ralph_status = None
                self.console.print("[yellow]✓ Disabled autonomous Ralph verification loop.[/yellow]")
                return False
            self.ralph_verify_cmd = " ".join(args)
            self.console.print(
                f"[green]✓ Enabled autonomous Ralph verification loop:[/green] [bold cyan]{self.ralph_verify_cmd}[/bold cyan] "
                f"(max {self.ralph_max_iterations} auto-repair iterations per turn)"
            )
            return False

        if cmd == "/checkpoint":
            sub = args[0].lower() if args else "list"
            if sub == "create":
                label = " ".join(args[1:]) if len(args) > 1 else f"manual-turn-{self.session_turns}"
                cp = create_git_checkpoint(workspace, len(self.checkpoints) + 1, self.session_turns, label)
                if cp:
                    self.checkpoints.append(cp)
                    self.console.print(
                        f"[green]✓ Created checkpoint #{cp.index}[/green] ([cyan]{cp.commit_oid[:8]}[/cyan]) — {cp.label}"
                    )
                else:
                    self.console.print("[yellow]⚠ Could not create git checkpoint (not a git repository?).[/yellow]")
                return False

            if not self.checkpoints:
                self.console.print("[dim]No checkpoints recorded yet. Use `/checkpoint create [label]` or run a turn.[/dim]")
                return False
            tbl = Table(title="Workspace Git Checkpoints", border_style="cyan")
            tbl.add_column("#", style="bold cyan", justify="right")
            tbl.add_column("Turn", justify="right")
            tbl.add_column("Time", style="dim")
            tbl.add_column("Git OID", style="magenta")
            tbl.add_column("Label")
            for cp in self.checkpoints:
                tbl.add_row(str(cp.index), str(cp.turn), cp.timestamp, cp.commit_oid[:10], cp.label)
            self.console.print(tbl)
            return False

        if cmd == "/rewind":
            if not args:
                self.console.print("[yellow]Usage: /rewind <checkpoint_number> (type /checkpoint to list)[/yellow]")
                return False
            try:
                target_idx = int(args[0])
            except ValueError:
                self.console.print("[red]Checkpoint number must be an integer.[/red]")
                return False
            matched = next((cp for cp in self.checkpoints if cp.index == target_idx), None)
            if not matched:
                self.console.print(f"[red]Checkpoint #{target_idx} not found. Type /checkpoint to list.[/red]")
                return False
            if restore_git_checkpoint(workspace, matched):
                self.console.print(
                    f"[green]✓ Rewound workspace files to checkpoint #{matched.index} "
                    f"({matched.commit_oid[:8]} · {matched.label}).[/green]"
                )
            else:
                self.console.print("[red]✗ Failed to restore git checkpoint.[/red]")
            return False

        if cmd == "/compact":
            if not self.transcripts:
                self.console.print("[dim]Nothing to compact yet.[/dim]")
                return False
            summary_bullets = []
            for tr in self.transcripts[-8:]:
                short_u = tr.user_prompt.splitlines()[0][:100]
                short_a = tr.assistant_response.splitlines()[0][:140] if tr.assistant_response else "(tool actions)"
                summary_bullets.append(f"- Turn {tr.turn_index}: User asked `{short_u}` -> Assistant: `{short_a}`")
            compact_seed = (
                "Here is a concise summary of our earlier conversation turns prior to manual compaction:\n"
                + "\n".join(summary_bullets)
            )
            agent.conversation.clear_history()
            self.total_prompt_tokens = len(compact_seed) // 4
            self.total_output_tokens = 0
            self.compaction_notifications += 1
            await agent.conversation.send(
                f"{compact_seed}\n\nAcknowledge compaction in one short line and await my next instruction."
            )
            async for _ in agent.conversation.receive_steps():
                pass
            self.console.print(
                f"[green]✓ Compacted {len(self.transcripts)} turn(s) into a concise context summary.[/green]"
            )
            return False

        if cmd == "/save":
            name = args[0] if args else time.strftime("session-%Y%m%d-%H%M%S")
            if not name.endswith(".json"):
                name += ".json"
            out_file = self._sessions_dir() / name
            payload = {
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "preset": self.options.preset,
                "backend": self.options.backend,
                "model": self.options.model,
                "turns": [tr.__dict__ for tr in self.transcripts],
            }
            out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            self.console.print(f"[green]✓ Saved session ({len(self.transcripts)} turns) to {out_file}[/green]")
            return False

        if cmd == "/load":
            if not args:
                files = sorted(self._sessions_dir().glob("*.json"))
                if not files:
                    self.console.print("[dim]No saved sessions found in ~/.local/share/agy-local/sessions/[/dim]")
                    return False
                tbl = Table(title="Saved Sessions", border_style="cyan")
                tbl.add_column("Session File", style="bold cyan")
                tbl.add_column("Size", justify="right", style="dim")
                for f in files:
                    tbl.add_row(f.name, f"{f.stat().st_size:,} B")
                self.console.print(tbl)
                return False
            name = args[0] if args[0].endswith(".json") else f"{args[0]}.json"
            in_file = self._sessions_dir() / name
            if not in_file.is_file():
                self.console.print(f"[red]Saved session '{in_file}' not found.[/red]")
                return False
            data = json.loads(in_file.read_text(encoding="utf-8"))
            loaded_turns = data.get("turns", [])
            self.transcripts = [TurnTranscript(**t) for t in loaded_turns]
            self.session_turns = len(self.transcripts)
            self.console.print(
                f"[green]✓ Loaded {len(self.transcripts)} turn(s) from {in_file.name}.[/green]"
            )
            return False

        if cmd == "/export":
            out_name = args[0] if args else f"agy-session-{int(time.time())}.md"
            target = Path(workspace) / out_name if not os.path.isabs(out_name) else Path(out_name)
            self.export_markdown(target, cfg)
            self.console.print(f"[green]✓ Exported Markdown session report to {target}[/green]")
            return False

        if cmd == "/policy":
            if not args or args[0] not in ("allow-all", "ask", "sandbox"):
                self.console.print("[yellow]Usage: /policy <allow-all|ask|sandbox>[/yellow]")
                return False
            self.options.policy_mode = PolicyMode(args[0])
            self._restart_requested = True
            self.console.print(f"[green]✓ Switched safety policy to {args[0].upper()} (reloading harness)...[/green]")
            return True

        if cmd == "/preset":
            if not args:
                tbl = Table(title="Available Model Presets", border_style="bright_blue")
                tbl.add_column("Preset Key", style="bold cyan")
                tbl.add_column("Backend", style="magenta")
                tbl.add_column("VRAM", justify="right")
                tbl.add_column("Target Model")
                tbl.add_column("Description", style="dim")
                for p in list_presets():
                    vram = f"{p.recommended_vram_gb} GB" if p.recommended_vram_gb else "Cloud"
                    tbl.add_row(p.key, p.backend.value, vram, p.model_id, p.description)
                self.console.print(tbl)
                return False
            target_key = args[0].lower()
            if target_key not in PRESETS:
                self.console.print(f"[red]Unknown preset '{target_key}'. Type /preset to list all.[/red]")
                return False
            preset_obj = PRESETS[target_key]
            if preset_obj.backend == BackendKind.OPENAI:
                from agy_local.discovery import probe_endpoint

                ep = await asyncio.to_thread(probe_endpoint, preset_obj.default_base_url, preset_obj.key, 0.8)
                if not ep.reachable and "--force" not in args:
                    self.console.print(
                        Panel(
                            f"[bold yellow]⚠ Preset `{target_key}` uses `{preset_obj.default_base_url}` which is currently OFFLINE.[/bold yellow]\n\n"
                            f"• Start the server first:\n"
                            f"  [bold green]kardashev serve --model {preset_obj.model_id}[/bold green]\n"
                            f"• Or stay on in-process LiteRT GPU/NPU:\n"
                            f"  [bold cyan]/preset gemma4-26b[/bold cyan]\n"
                            f"[dim](Pass `/preset {target_key} --force` to switch anyway.)[/dim]",
                            title="[bold yellow]Preset Endpoint Offline[/bold yellow]",
                            border_style="yellow",
                        )
                    )
                    return False
            self.options.preset = target_key
            self.options.backend = None
            self.options.model = None
            self.options.model_path = None
            self.options.base_url = None
            self._restart_requested = True
            self.console.print(f"[green]✓ Switching active preset to '{target_key}'...[/green]")
            return True

        if cmd == "/model":
            if not args:
                self.console.print("[yellow]Usage: /model <model_id_or_litert_path> [base_url][/yellow]")
                return False
            target_model = args[0]
            if target_model.lower() in PRESETS and len(args) == 1:
                self.options.preset = target_model.lower()
                self.options.backend = None
                self.options.model = None
                self.options.model_path = None
                self.options.base_url = None
                self._restart_requested = True
                self.console.print(f"[green]✓ Switching active preset to '{target_model.lower()}'...[/green]")
                return True

            if target_model.endswith(".litertlm"):
                self.options.preset = None
                self.options.backend = "litert"
                self.options.model_path = target_model
                self.options.model = None
                self._restart_requested = True
                self.console.print(f"[green]✓ Switching LiteRT model to '{target_model}'...[/green]")
                return True

            # Check if a discovered local LiteRT model matches by stem
            discovered = discover_local_models()
            litert_match = next(
                (m for m in discovered if m.backend == BackendKind.LITERT and m.name.lower() == target_model.lower()),
                None,
            )
            if litert_match and len(args) == 1:
                self.options.preset = None
                self.options.backend = "litert"
                self.options.model_path = litert_match.path
                self.options.model = None
                self._restart_requested = True
                self.console.print(f"[green]✓ Switching to local LiteRT model '{litert_match.name}'...[/green]")
                return True

            target_url = args[1] if len(args) > 1 else (self.options.base_url or "http://127.0.0.1:8000/v1")
            from agy_local.discovery import probe_endpoint

            ep = await asyncio.to_thread(probe_endpoint, target_url, "Target Endpoint", 0.8)
            if not ep.reachable:
                # Check if any other local server (e.g. Ollama on 11434) is online and serves this model
                all_eps = await asyncio.to_thread(probe_local_endpoints, 0.6)
                alt_ep = next(
                    (e for e in all_eps if e.reachable and any(target_model.lower() in m.lower() for m in e.models)),
                    None,
                )
                if alt_ep is not None and len(args) == 1:
                    target_url = alt_ep.base_url
                else:
                    self.console.print(
                        Panel(
                            f"[bold yellow]⚠ Local inference server at `{target_url}` is currently OFFLINE.[/bold yellow]\n\n"
                            f"Model [bold cyan]{target_model}[/bold cyan] requires a running OpenAI-compatible server (vLLM / Ollama / SGLang).\n"
                            f"• To start the server in another terminal, run:\n"
                            f"  [bold green]kardashev serve --model {target_model}[/bold green]\n"
                            f"• Or stay on in-process LiteRT GPU/NPU (no server needed):\n"
                            f"  [bold cyan]/preset gemma4-26b[/bold cyan]",
                            title="[bold yellow]Inference Server Offline[/bold yellow]",
                            border_style="yellow",
                        )
                    )
                    if "--force" not in args:
                        self.console.print(
                            "[dim]Staying on your current active model so your session isn't interrupted. "
                            "(Pass `/model <name> <url> --force` to switch anyway.)[/dim]"
                        )
                        return False

            self.options.preset = None
            self.options.backend = "openai"
            self.options.model = target_model
            self.options.model_path = None
            self.options.base_url = target_url
            self._restart_requested = True
            self.console.print(f"[green]✓ Switching model to '{target_model}' @ {target_url}...[/green]")
            return True

        if cmd == "/models":
            models = discover_local_models()
            endpoints = probe_local_endpoints()
            tbl = Table(title="Discovered Local Checkpoints", border_style="cyan")
            tbl.add_column("Model Name", style="bold white")
            tbl.add_column("Backend", style="magenta")
            tbl.add_column("Size", justify="right", style="green")
            tbl.add_column("Source", style="dim")
            for m in models:
                tbl.add_row(m.name, m.backend.value, f"{m.size_gb:.2f} GB", m.source)
            self.console.print(tbl)

            ep_tbl = Table(title="Localhost OpenAI Inference Endpoints", border_style="bright_blue")
            ep_tbl.add_column("Status")
            ep_tbl.add_column("Server")
            ep_tbl.add_column("Base URL", style="dim")
            ep_tbl.add_column("Models")
            for e in endpoints:
                badge = "[bold green]● ONLINE[/bold green]" if e.reachable else "[dim]○ OFFLINE[/dim]"
                ep_tbl.add_row(badge, e.label, e.base_url, ", ".join(e.models) if e.models else "—")
            self.console.print(ep_tbl)
            return False

        if cmd == "/doctor":
            from agy_local.cli import _cmd_doctor

            class _DummyArgs:
                json = False

            _cmd_doctor(_DummyArgs())  # type: ignore[arg-type]
            return False

        if cmd == "/diff":
            proc = await asyncio.to_thread(
                subprocess.run,
                ["git", "-C", workspace, "diff"],
                capture_output=True,
                text=True,
                check=False,
            )
            diff_out = proc.stdout.strip()
            if not diff_out:
                self.console.print("[dim]No unstaged git changes in workspace.[/dim]")
            else:
                self.console.print(
                    Panel(
                        Syntax(diff_out[:12000], "diff", theme="monokai", word_wrap=True),
                        title=f"[bold cyan]git diff ({workspace})[/bold cyan]",
                        border_style="cyan",
                    )
                )
            return False

        if cmd == "/undo":
            proc = await asyncio.to_thread(
                subprocess.run,
                ["git", "-C", workspace, "status", "--short"],
                capture_output=True,
                text=True,
                check=False,
            )
            if not proc.stdout.strip():
                self.console.print("[dim]Working tree is already clean.[/dim]")
                return False
            self.console.print(Panel(proc.stdout.strip(), title="[yellow]Modified Files[/yellow]", border_style="yellow"))
            ans = (await _nonblocking_input("  Discard unstaged modifications via `git checkout -- .`? [y/N]: ")).strip().lower()
            if ans in ("y", "yes"):
                await asyncio.to_thread(subprocess.run, ["git", "-C", workspace, "checkout", "--", "."], check=False)
                self.console.print("[green]✓ Reverted unstaged changes.[/green]")
            return False

        if cmd in ("/update", "/upgrade"):
            self.console.print("[cyan]Checking for updates from GitHub main...[/cyan]")
            ok, msg = await asyncio.to_thread(perform_self_update, include_litert=True)
            color = "green" if ok else "red"
            icon = "✓" if ok else "✗"
            self.console.print(f"[{color}]{icon} {msg}[/{color}]")
            return False

        if cmd == "/config":
            ucfg = UserConfig.load()
            if not args or args[0] == "show":
                tbl = Table(title=f"User Defaults ({UserConfig.config_path()})", border_style="cyan")
                tbl.add_column("Key", style="bold cyan")
                tbl.add_column("Value", style="white")
                for k, v in ucfg.__dict__.items():
                    tbl.add_row(k, repr(v))
                self.console.print(tbl)
                return False
            if args[0] == "save":
                ucfg.default_preset = self.options.preset or ucfg.default_preset
                ucfg.policy_mode = (
                    self.options.policy_mode.value
                    if isinstance(self.options.policy_mode, PolicyMode)
                    else str(self.options.policy_mode)
                )
                ucfg.show_thoughts = self.show_thoughts
                ucfg.ralph_verify_cmd = self.ralph_verify_cmd
                ucfg.with_chronosml_mcp = self.options.with_chronosml_mcp
                ucfg.with_opencommons_mcp = self.options.with_opencommons_mcp
                saved = ucfg.save()
                self.console.print(f"[green]✓ Saved current session settings as user defaults in {saved}[/green]")
                return False
            self.console.print("[yellow]Usage: /config [show|save][/yellow]")
            return False

        if cmd == "/ast":
            sub = args[0].lower() if args else "show"
            if sub in ("on", "enable"):
                self.auto_inject_ast_map = True
                self.console.print("[green]✓ Enabled automatic AST Repo-Map injection.[/green]")
                return False
            if sub in ("off", "disable"):
                self.auto_inject_ast_map = False
                self.console.print("[yellow]✓ Disabled automatic AST Repo-Map injection.[/yellow]")
                return False
            idx = await asyncio.to_thread(build_workspace_index, workspace)
            rmap = render_compact_repo_map(idx, query=" ".join(args[1:]), max_chars=4000)
            self.console.print(
                Panel(
                    Syntax(rmap, "markdown", theme="monokai", word_wrap=True),
                    title=f"[bold cyan]AST Repo-Map ({len(idx)} files indexed)[/bold cyan]",
                    border_style="cyan",
                )
            )
            return False

        if cmd == "/scout":
            if not args:
                self.console.print("[yellow]Usage: /scout <query_or_symbol>[/yellow]")
                return False
            query_str = " ".join(args)
            res = await asyncio.to_thread(scout_workspace, workspace, query_str, max_slices=4)
            tbl = Table(
                title=(
                    f"AST Context Scout: '{query_str}' "
                    f"(~{res.estimated_tokens_saved:,} raw tokens saved)"
                ),
                border_style="cyan",
            )
            tbl.add_column("File", style="bold cyan")
            tbl.add_column("Symbol", style="bold yellow")
            tbl.add_column("Lines", justify="right", style="magenta")
            tbl.add_column("Score", justify="right", style="green")
            for sl in res.surgical_slices:
                tbl.add_row(sl["path"], sl["symbol"], f"L{sl['start_line']}-{sl['end_line']}", str(sl["score"]))
            self.console.print(tbl)
            return False

        if cmd == "/commons":
            from agy_local.opencommons import (
                DEFAULT_PENDING_PATH,
                DEFAULT_STORE_PATH,
                approve_pending_proposal,
                load_pending_proposals,
                load_persistent_store,
                pull_from_remote_hub,
                push_to_remote_hub,
                reject_pending_proposal,
                save_persistent_store,
                seed_default_scientific_primitives,
            )

            sub = args[0].lower() if args else "status"
            if sub == "seed":
                store = await asyncio.to_thread(load_persistent_store)
                added = await asyncio.to_thread(seed_default_scientific_primitives, store)
                merged = await asyncio.to_thread(save_persistent_store, store)
                self.console.print(
                    f"[green]✓ Seeded {added} foundational primitives ({len(merged.canonical_block_by_cid)} active concepts in {DEFAULT_STORE_PATH}).[/green]"
                )
                return False

            if sub == "pull":
                hub_url = args[1] if len(args) > 1 else None
                res_pull = await asyncio.to_thread(pull_from_remote_hub, hub_url)
                if res_pull.get("ok"):
                    self.console.print(
                        f"[green]✓ Pulled from {res_pull['hub_url']} ({res_pull['protocol']} · {res_pull['elapsed_ms']}ms): "
                        f"+{res_pull['new_blocks_pulled']} new block(s), {res_pull['active_concepts_cid']} active concepts (Merkle: {res_pull['merkle_root']}).[/green]"
                    )
                else:
                    self.console.print(f"[red]✗ Pull failed: {res_pull.get('error')}[/red]")
                return False

            if sub == "push":
                hub_url = args[1] if len(args) > 1 else None
                res_push = await asyncio.to_thread(push_to_remote_hub, hub_url)
                if res_push.get("ok"):
                    self.console.print(
                        f"[green]✓ Pushed to {res_push['hub_url']} ({res_push['elapsed_ms']}ms): "
                        f"accepted={res_push['accepted_by_hub']}, total_active={res_push['total_active_concepts']} (Merkle: {res_push['merkle_root']}).[/green]"
                    )
                else:
                    self.console.print(f"[red]✗ Push failed: {res_push.get('error')}[/red]")
                return False

            if sub == "find":
                q = " ".join(args[1:]).strip()
                if not q:
                    self.console.print("[yellow]Usage: /commons find <query>[/yellow]")
                    return False
                store = await asyncio.to_thread(load_persistent_store)
                hits = store.find(query=q, top_k=3)
                if not hits:
                    self.console.print(f"[dim]No OpenCommons concepts matched '{q}'.[/dim]")
                    return False
                for h in hits:
                    card_str = h.block.render_for_agent("dual_card_with_pitfalls")
                    self.console.print(
                        Panel(
                            Syntax(card_str, "python", theme="monokai", word_wrap=True),
                            title=f"[bold magenta]{h.block.block_id}[/bold magenta] [dim](score={h.score} · {h.latency_ms}ms)[/dim]",
                            border_style="magenta",
                        )
                    )
                return False

            if sub == "review":
                pending = await asyncio.to_thread(load_pending_proposals)
                if not pending:
                    self.console.print("[dim]✓ Pending OpenCommons review queue is empty.[/dim]")
                    return False
                tbl = Table(title=f"Pending OpenCommons Agent Proposals ({DEFAULT_PENDING_PATH})", border_style="yellow")
                tbl.add_column("Proposal ID", style="bold yellow")
                tbl.add_column("Function", style="bold cyan")
                tbl.add_column("Domain", style="magenta")
                tbl.add_column("Reason", style="dim")
                tbl.add_column("Summary")
                for p in pending:
                    tbl.add_row(p.proposal_id, p.func_name, p.domain, p.stage_reason, p.summary)
                self.console.print(tbl)
                self.console.print(
                    "[dim]Use `/commons approve <proposal_id_or_func>` or `/commons reject <proposal_id_or_func>`[/dim]"
                )
                return False

            if sub == "approve":
                if len(args) < 2:
                    self.console.print("[yellow]Usage: /commons approve <proposal_id_or_func_name>[/yellow]")
                    return False
                res_app = await asyncio.to_thread(approve_pending_proposal, args[1])
                if res_app.get("approved"):
                    self.console.print(
                        f"[green]✓ Approved `{res_app['block_id']}` -> committed to OpenCommons ({res_app['total_active_concepts']} active concepts).[/green]"
                    )
                else:
                    self.console.print(f"[red]✗ {res_app.get('error')}[/red]")
                return False

            if sub == "reject":
                if len(args) < 2:
                    self.console.print("[yellow]Usage: /commons reject <proposal_id_or_func_name>[/yellow]")
                    return False
                res_rej = await asyncio.to_thread(reject_pending_proposal, args[1])
                if res_rej.get("rejected"):
                    self.console.print(
                        f"[yellow]✓ Rejected proposal `{res_rej['proposal_id']}` ({res_rej['func_name']}).[/yellow]"
                    )
                else:
                    self.console.print(f"[red]✗ {res_rej.get('error')}[/red]")
                return False

            # Default: /commons status
            store = await asyncio.to_thread(load_persistent_store)
            pending = await asyncio.to_thread(load_pending_proposals)
            manifest = store.compute_merkle_manifest()
            tbl = Table(title=f"OpenCommons Research Store ({DEFAULT_STORE_PATH})", border_style="magenta")
            tbl.add_column("Block ID", style="bold cyan")
            tbl.add_column("CID", style="magenta")
            tbl.add_column("Domain", style="green")
            tbl.add_column("Tier", style="yellow")
            tbl.add_column("Summary")
            for _cid, bid in sorted(store.canonical_block_by_cid.items()):
                b = store.blocks_by_id[bid]
                tbl.add_row(b.block_id, b.cid, b.domain, b.trust_tier, b.summary[:68])
            self.console.print(tbl)
            self.console.print(
                f"  [dim]Active equivalence classes:[/dim] [bold green]{len(store.canonical_block_by_cid)}[/bold green] · "
                f"[dim]Quarantined:[/dim] [red]{len(store.quarantined_ids)}[/red] · "
                f"[dim]Pending review:[/dim] [yellow]{len(pending)}[/yellow] · "
                f"[dim]Merkle Root:[/dim] [cyan]{manifest['merkle_root']}[/cyan]\n"
                f"  [dim]Commands: `/commons find <q>`, `/commons review`, `/commons pull [url]`, `/commons push [url]`[/dim]"
            )
            return False

        if cmd == "/integrate":
            target = args[0].lower() if args else "all"
            try:
                int_results = integrate_frontier_agent(target)
                tbl = Table(title="Frontier Coding Agent Co-Processor Integration", border_style="green")
                tbl.add_column("Target Agent", style="bold cyan")
                tbl.add_column("MCP Config", style="white")
                tbl.add_column("Installed Co-Processor Skill / Rule", style="dim")
                for r in int_results:
                    tbl.add_row(r.target, r.mcp_config_path, r.skill_path or "—")
                self.console.print(tbl)
            except ValueError as exc:
                self.console.print(f"[red]✗ {exc}[/red]")
            return False

        self.console.print(f"[yellow]Unknown command '{cmd}'. Type /help for available commands.[/yellow]")
        return False

    async def _run_turn(self, agent: Agent, user_text: str, cfg: Any) -> None:
        from agy_local.harness_hooks import reset_turn_tool_signatures

        reset_turn_tool_signatures()
        workspace = self.options.workspaces[0] if self.options.workspaces else os.getcwd()
        augmented_prompt, attached = expand_at_file_mentions(user_text, workspace)
        if attached:
            self.console.print(
                f"  [dim]📎 Attached {', '.join(f'[cyan]{f}[/cyan]' for f in attached)}[/dim]"
            )

        # On Turn 0, prepend query-matched OpenCommons Dual Concept Cards + AST Repo-Map + surgical slices
        if self.options.enable_opencommons and self.session_turns == 0:
            from agy_local.opencommons import build_turn_zero_commons_context

            oc_block, oc_hits = await asyncio.to_thread(build_turn_zero_commons_context, user_text)
            if oc_block and oc_hits > 0:
                augmented_prompt = f"{oc_block}\n\n{augmented_prompt}"
                get_harness_telemetry().commons_cards_injected += oc_hits
                self.console.print(
                    f"  [dim]🧬 Retrieved {oc_hits} verified OpenCommons Dual Concept Card(s)[/dim]"
                )

        if self.auto_inject_ast_map and self.session_turns == 0:
            from agy_local.ast_map import build_turn_zero_context

            ctx_block, n_files, n_syms, n_slices = await asyncio.to_thread(
                build_turn_zero_context, workspace, user_text
            )
            if ctx_block:
                augmented_prompt = f"{ctx_block}\n\n{augmented_prompt}"
                htel = get_harness_telemetry()
                htel.ast_maps_injected += 1
                slice_tag = f" · {n_slices} surgical slice(s)" if n_slices else ""
                self.console.print(
                    f"  [dim]🧭 Grounded with AST Repo-Map ({n_files} files · {n_syms} symbols{slice_tag})[/dim]"
                )

        # Auto-snapshot git state before the turn in case the agent edits files
        cp = create_git_checkpoint(
            workspace,
            len(self.checkpoints) + 1,
            self.session_turns + 1,
            f"pre-turn-{self.session_turns + 1}: {user_text[:36]}",
        )
        if cp:
            self.checkpoints.append(cp)

        t0 = time.perf_counter()
        thought_buf: list[str] = []
        text_buf: list[str] = []
        self._active_turn_tools = []
        seen_tool_ids: set[str] = set()
        self.last_ralph_status = None

        def _render_live_frame() -> Group:
            renderables: list[Any] = []
            elapsed = time.perf_counter() - t0

            if thought_buf and self.show_thoughts:
                full_thought = "".join(thought_buf).strip()
                t_lines = full_thought.splitlines()
                clipped = "\n".join(t_lines[-6:]) if len(t_lines) > 6 else full_thought
                renderables.append(
                    Panel(
                        Text(clipped, style="italic dim"),
                        title=f"[dim]💭 Thinking ({elapsed:.1f}s)[/dim]",
                        border_style="dim",
                        padding=(0, 1),
                    )
                )

            for rec in self._active_turn_tools:
                line = Text()
                icon = "✗" if rec.is_error else ("⏺" if rec.completed_at else "⠋")
                icon_style = "bold red" if rec.is_error else ("bold cyan" if rec.completed_at else "bold yellow")
                line.append(f"  {icon} ", style=icon_style)
                line.append(f"{rec.badge}", style="bold bright_cyan")
                line.append(f"({rec.summary})", style="white")
                if rec.duration_ms is not None:
                    line.append(f"  [{rec.duration_ms:.0f}ms]", style="dim")
                renderables.append(line)

                if rec.result_preview:
                    res_line = Text()
                    res_style = "red" if rec.is_error else "dim green"
                    res_line.append(f"    ⎿  {rec.result_preview}", style=res_style)
                    renderables.append(res_line)

                if rec.diff_preview:
                    renderables.append(
                        Panel(
                            Syntax(rec.diff_preview, "diff", theme="monokai", word_wrap=True),
                            border_style="dim",
                            padding=(0, 1),
                        )
                    )

            if self.last_ralph_status:
                ralph_line = Text()
                ralph_line.append(f"  🔁 Ralph Verify: {self.last_ralph_status}", style="bold yellow")
                renderables.append(ralph_line)

            if text_buf:
                md_text = "".join(text_buf)
                renderables.append(
                    Panel(
                        Markdown(md_text),
                        title="[bold bright_green]✦ Assistant[/bold bright_green]",
                        border_style="green",
                        padding=(0, 1),
                    )
                )
            elif not renderables:
                spinner_line = Text()
                spinner_line.append("  ⠋ ", style="bold cyan")
                spinner_line.append(f"Processing on local model ({elapsed:.1f}s)...", style="dim")
                renderables.append(spinner_line)

            return Group(*renderables)

        try:
            await agent.conversation.send(augmented_prompt)
        except Exception as exc:  # noqa: BLE001
            self._render_connection_error(exc, cfg)
            return

        with Live(_render_live_frame(), console=self.console, refresh_per_second=12, transient=False) as live:
            try:
                async for step in agent.conversation.receive_steps():
                    is_model = step.source == types.StepSource.MODEL
                    is_target_user = step.target == types.StepTarget.USER

                    if is_model and step.thinking_delta:
                        thought_buf.append(step.thinking_delta)

                    if is_model and is_target_user and step.content_delta:
                        text_buf.append(step.content_delta)
                    elif step.is_complete_response and step.content and not text_buf:
                        text_buf.append(step.content)

                    if step.tool_calls:
                        for tc in step.tool_calls:
                            tc_key = tc.id or f"{tc.name}:{tc.args}"
                            if tc_key not in seen_tool_ids:
                                seen_tool_ids.add(tc_key)
                                self.session_tool_calls += 1
                                badge, summary, diff_preview = format_tool_badge(tc)
                                self._active_turn_tools.append(
                                    ToolEventRecord(
                                        call_id=tc_key,
                                        name=str(tc.name),
                                        badge=badge,
                                        summary=summary,
                                        diff_preview=diff_preview,
                                    )
                                )

                    live.update(_render_live_frame())
            except KeyboardInterrupt:
                await agent.conversation.cancel()
                self.console.print("\n[yellow]⚠ Turn cancelled by user (Ctrl+C).[/yellow]")
                return
            except Exception as exc:  # noqa: BLE001
                live.stop()
                self._render_connection_error(exc, cfg)
                return

        dt = max(0.001, time.perf_counter() - t0)
        self.session_turns += 1

        # Compute turn telemetry
        turn_usage = agent.conversation.last_turn_usage
        out_chars = len("".join(text_buf)) + len("".join(thought_buf))
        est_out_tokens = max(1, out_chars // 4) if out_chars else 0
        if turn_usage and turn_usage.candidates_token_count:
            out_tok = (turn_usage.candidates_token_count or 0) + (turn_usage.thoughts_token_count or 0)
            in_tok = turn_usage.prompt_token_count or 0
        else:
            out_tok = est_out_tokens
            in_tok = max(1, len(augmented_prompt) // 4)

        self.total_prompt_tokens += in_tok
        self.total_output_tokens += out_tok
        tok_per_s = out_tok / dt if out_tok > 0 else 0.0

        self.transcripts.append(
            TurnTranscript(
                turn_index=self.session_turns,
                user_prompt=user_text,
                thinking="".join(thought_buf).strip(),
                assistant_response="".join(text_buf).strip(),
                tools=[
                    {
                        "badge": r.badge,
                        "summary": r.summary,
                        "duration_ms": r.duration_ms,
                        "result_preview": r.result_preview,
                        "is_error": r.is_error,
                    }
                    for r in self._active_turn_tools
                ],
                duration_s=dt,
                prompt_tokens=in_tok,
                output_tokens=out_tok,
            )
        )

        _, _, ctx_limit = self._active_model_summary(cfg)
        tot_ctx = self.total_prompt_tokens + self.total_output_tokens
        pct = min(100.0, (tot_ctx / max(1, ctx_limit)) * 100.0)
        footer = Text()
        footer.append(f"  ⏱ {dt:.2f}s", style="dim")
        if tok_per_s > 0:
            footer.append(f" · ⚡ {tok_per_s:.1f} tok/s", style="dim cyan")
        footer.append(f" · 📊 ~{out_tok:,} out tok", style="dim")
        if self._active_turn_tools:
            footer.append(f" · 🛠 {len(self._active_turn_tools)} tool(s)", style="dim yellow")
        footer.append(f" · ctx {tot_ctx:,}/{ctx_limit:,} ({pct:.1f}%)", style="dim")
        if self.compaction_notifications:
            footer.append(f" · 🗜 {self.compaction_notifications} compacted", style="dim magenta")
        self.console.print(footer)
        self.console.print()

    def _build_key_bindings(self) -> KeyBindings:
        kb = KeyBindings()

        @kb.add("escape", "t")
        def _toggle_thoughts(event: Any) -> None:
            self.show_thoughts = not self.show_thoughts
            event.app.invalidate()

        @kb.add("escape", "p")
        def _cycle_policy(event: Any) -> None:
            order = [PolicyMode.ALLOW_ALL, PolicyMode.SANDBOX, PolicyMode.ASK]
            cur = (
                self.options.policy_mode
                if isinstance(self.options.policy_mode, PolicyMode)
                else PolicyMode(str(self.options.policy_mode))
            )
            nxt = order[(order.index(cur) + 1) % len(order)]
            self.options.policy_mode = nxt
            event.app.invalidate()

        return kb

    async def run(self) -> int:
        """Main loop supporting hot-reloading presets/models/policies via `/preset`, `/model`, `/policy`."""
        global _ACTIVE_TUI
        _ACTIVE_TUI = self

        history_path = Path.home() / ".cache" / "agy-local" / "prompt_history.txt"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        workspace = self.options.workspaces[0] if self.options.workspaces else os.getcwd()

        pt_style = PtStyle.from_dict(
            {
                "prompt": "ansibrightcyan bold",
                "model_tag": "ansibrightblack",
                "bottom-toolbar": "bg:#1a1b26 #a9b1d6",
            }
        )
        prompt_session: PromptSession[str] | None = None
        if sys.stdin.isatty():
            prompt_session = PromptSession(
                history=FileHistory(str(history_path)),
                completer=AgyCompleter(workspace),
                key_bindings=self._build_key_bindings(),
                style=pt_style,
                complete_while_typing=False,
            )

        try:
            while True:
                self._restart_requested = False
                base_cfg = build_agent_config(self.options)
                cfg = self._upgrade_interactive_config(base_cfg)
                model_label, backend_label, ctx_limit = self._active_model_summary(cfg)

                self.console.print(self.render_welcome_banner(cfg))

                agent_cm = Agent(cfg)
                try:
                    with (
                        self.console.status(
                            f"[bold cyan]⚡ Initializing {model_label}[/bold cyan] "
                            f"[dim]({backend_label} · warming up KV cache)...[/dim]",
                            spinner="dots",
                        ),
                        suppress_noisy_cpp_stderr(),
                    ):
                        agent = await agent_cm.__aenter__()
                except (KeyboardInterrupt, asyncio.CancelledError):
                    self.console.print("\n[dim]Startup cancelled.[/dim]")
                    return 130

                try:
                    ctrl_c_streak = 0
                    while True:
                        try:
                            branch, dirty = _git_branch_and_dirty(workspace)
                            git_info = f" | git:{branch}{'*' if dirty else ''}" if branch else ""
                            pol = (
                                self.options.policy_mode.value
                                if isinstance(self.options.policy_mode, PolicyMode)
                                else str(self.options.policy_mode)
                            )
                            tot_ctx = self.total_prompt_tokens + self.total_output_tokens
                            ctx_pct = min(100.0, (tot_ctx / max(1, ctx_limit)) * 100.0)
                            thoughts_flag = "on" if self.show_thoughts else "off"
                            ralph_flag = f" | ralph:{self.ralph_verify_cmd}" if self.ralph_verify_cmd else ""
                            toolbar_html = HTML(
                                f" <b>{model_label}</b> | policy:<b>{pol}</b> | ctx:<b>{ctx_pct:.0f}%</b> | "
                                f"thoughts:<b>{thoughts_flag}</b>{git_info}{ralph_flag} | "
                                f"<b>/help</b> · <b>@file:1-20</b> · <b>@git:diff</b> · <b>!cmd</b> · <b>Alt+T</b>/<b>Alt+P</b> "
                            )
                            prompt_html = HTML(f"<model_tag>[{model_label}]</model_tag> <prompt>❯ </prompt>")

                            if prompt_session is not None:
                                user_input = await prompt_session.prompt_async(
                                    prompt_html,
                                    bottom_toolbar=toolbar_html,
                                )
                            else:
                                user_input = await _nonblocking_input("❯ ")
                            ctrl_c_streak = 0
                        except KeyboardInterrupt:
                            ctrl_c_streak += 1
                            if ctrl_c_streak >= 2:
                                break
                            self.console.print(
                                "  [dim]Press Ctrl+C again, Ctrl+D, or type /exit to quit.[/dim]"
                            )
                            continue
                        except (EOFError, asyncio.CancelledError):
                            break

                        text = user_input.strip()
                        if not text:
                            continue

                        if text.lower() in ("exit", "quit"):
                            break

                        # Direct !shell shortcut (like Claude Code / Codex CLI)
                        if text.startswith("!"):
                            shell_cmd = text[1:].strip()
                            if shell_cmd:
                                self.console.print(Rule(f"[bold yellow]$ {shell_cmd}[/bold yellow]", style="yellow"))
                                try:
                                    proc = await asyncio.to_thread(
                                        subprocess.run,
                                        shell_cmd,
                                        shell=True,
                                        cwd=workspace,
                                        check=False,
                                    )
                                    status_color = "green" if proc.returncode == 0 else "red"
                                    self.console.print(f"[{status_color}]↳ exit code {proc.returncode}[/{status_color}]\n")
                                except (KeyboardInterrupt, asyncio.CancelledError):
                                    self.console.print("\n[yellow]⚡ Shell command interrupted.[/yellow]\n")
                            continue

                        # Slash commands
                        if text.startswith("/"):
                            should_break = await self._handle_slash_command(text, agent, cfg)
                            if should_break:
                                break
                            continue

                        try:
                            await self._run_turn(agent, text, cfg)
                        except (KeyboardInterrupt, asyncio.CancelledError):
                            self.console.print("\n  [yellow]⚡ Turn interrupted by user (Ctrl+C).[/yellow]\n")
                finally:
                    with (
                        self.console.status(
                            f"[dim]Shutting down {model_label} engine...[/dim]",
                            spinner="dots",
                        ),
                        suppress_noisy_cpp_stderr(),
                        contextlib.suppress(KeyboardInterrupt, asyncio.CancelledError, Exception),
                    ):
                        await asyncio.shield(agent_cm.__aexit__(None, None, None))
                    if not self._restart_requested:
                        self.console.print("[dim]✓ Session closed cleanly.[/dim]")

                if not self._restart_requested:
                    break
        except (KeyboardInterrupt, asyncio.CancelledError):
            self.console.print("\n[dim]✓ Session closed cleanly.[/dim]")
        finally:
            if _ACTIVE_TUI is self:
                _ACTIVE_TUI = None

        return 0
