"""Rich Terminal UI (TUI) for agy-local inspired by Claude Code & OpenAI Codex CLI.

Features:
- Claude Code / Codex style header card with live workspace, git branch, active model,
  hardware backend, policy badge, and MCP server indicators.
- Real-time streaming with Rich `Live`:
  - Collapsible/dimmed reasoning (`Thinking`) block.
  - Inline diffs for `replace_file_content` / `write_to_file` (`+` green, `-` red).
  - Structured tool call badges (`Bash(...)`, `Read(...)`, `Edit(...)`, `Grep(...)`, `ListDir(...)`).
  - Live Markdown rendering with syntax-highlighted code blocks.
- Interactive `prompt_toolkit` prompt with:
  - Slash command completion (`/help`, `/model`, `/preset`, `/policy`, `/status`, `/doctor`,
    `/models`, `/diff`, `/undo`, `/compact`, `/clear`, `/bash`, `/exit`).
  - `@file` workspace path tab-completion & inline file context injection.
  - `!cmd` direct shell execution shortcut.
  - Telemetry footer after each turn (`⏱ latency · tok/s · prompt/output tokens · tool count`).
"""

from __future__ import annotations

import asyncio
import difflib
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from google.antigravity import Agent, types
from google.antigravity.hooks import hooks as hooks_mod
from google.antigravity.hooks import policy as policy_mod
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style as PtStyle
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from agy_local.builder import HarnessOptions, PolicyMode, build_agent_config
from agy_local.discovery import discover_local_models, probe_local_endpoints
from agy_local.presets import PRESETS, list_presets

SLASH_COMMANDS: dict[str, str] = {
    "/help": "Show interactive slash commands, @file mentions, and !shell shortcuts",
    "/status": "Show session token usage, context window utilization, turn count, and active tools",
    "/preset": "Switch to a curated model preset (e.g., /preset gemma4-26b or /preset glm4-9b-ollama)",
    "/model": "Switch model ID or OpenAI base URL (e.g., /model qwen3-vl-8b http://127.0.0.1:8000/v1)",
    "/policy": "Switch tool execution policy mode (/policy allow-all | ask | sandbox)",
    "/models": "Scan local disk checkpoints and probe localhost OpenAI servers",
    "/doctor": "Run hardware, GPU, LiteRT, and MCP diagnostics",
    "/diff": "Show colorized git diff of uncommitted changes in the workspace",
    "/undo": "Revert unstaged file modifications in the workspace (with confirmation)",
    "/thoughts": "Toggle display of model reasoning/thinking traces",
    "/clear": "Clear conversation history and terminal screen",
    "/exit": "Exit agy-local session (or Ctrl+D)",
}


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


def expand_at_file_mentions(prompt_text: str, workspace: str) -> tuple[str, list[str]]:
    """Detects `@path/to/file` mentions in user input and attaches file contents inline."""
    tokens = prompt_text.split()
    attached_files: list[str] = []
    context_blocks: list[str] = []

    for tok in tokens:
        if not tok.startswith("@") or len(tok) < 2:
            continue
        rel_candidate = tok[1:].strip(",.:;!?")
        candidate_path = Path(workspace) / rel_candidate
        if not candidate_path.is_file():
            candidate_path = Path(os.path.expanduser(rel_candidate))
        if candidate_path.is_file():
            try:
                text = candidate_path.read_text(encoding="utf-8")
                if len(text) > 24000:
                    text = text[:24000] + "\n... [truncated]"
                rel_display = (
                    str(candidate_path.relative_to(workspace))
                    if candidate_path.is_relative_to(workspace)
                    else str(candidate_path)
                )
                attached_files.append(rel_display)
                context_blocks.append(f"<attached_file path=\"{rel_display}\">\n{text}\n</attached_file>")
            except (UnicodeDecodeError, OSError):
                pass

    if not context_blocks:
        return prompt_text, []
    augmented = "\n\n".join(context_blocks) + "\n\n" + prompt_text
    return augmented, attached_files


class AgyCompleter(Completer):
    """Provides tab completion for `/slash` commands, preset names, and `@file` workspace paths."""

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

            # Sub-argument completion for /preset and /policy
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

        # 2. @file path completion
        if word.startswith("@"):
            partial = word[1:]
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
        self.session_turns = 0
        self.session_tool_calls = 0
        self.total_prompt_tokens = 0
        self.total_output_tokens = 0
        self._restart_requested = False

    def _active_model_summary(self, cfg: Any) -> tuple[str, str, int]:
        cls_name = type(cfg).__name__
        compaction = getattr(cfg, "compaction_config", None)
        ctx_limit = getattr(compaction, "token_threshold", None) or 40960
        if hasattr(cfg, "model_path"):
            path = str(getattr(cfg, "model_path", ""))
            model_short = os.path.basename(os.path.dirname(path)) if path.endswith("model.litertlm") else os.path.basename(path)
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
        mcp_list.extend(name for name, _, _ in self.options.extra_mcp_stdio)
        mcp_str = ", ".join(mcp_list) if mcp_list else "none (use --with-chronosml to enable)"
        grid.add_row(
            "Safety & Context",
            f"[{policy_color}]{policy_val.upper()}[/{policy_color}]  [dim]· compaction ceiling:[/dim] {ctx_limit:,} tok  [dim]· MCP:[/dim] {mcp_str}",
        )
        grid.add_row(
            "Shortcuts",
            "[dim][bold]/help[/bold] commands · [bold]@file[/bold] attach file · [bold]!cmd[/bold] run shell · [bold]/preset[/bold] switch model · [bold]/diff[/bold] git diff[/dim]",
        )

        return Panel(
            grid,
            title="[bold bright_white]✦ Antigravity Local Studio[/bold bright_white] [dim](agy-local v0.1.0)[/dim]",
            subtitle="[dim]The Kardashev Project · Open & On-Device Agent Harness[/dim]",
            border_style="bright_blue",
            padding=(0, 1),
        )

    def _upgrade_interactive_config(self, cfg: Any) -> Any:
        hooks_list = list(getattr(cfg, "hooks", []) or [])
        if not any(isinstance(h, RichAskQuestionHook) for h in hooks_list):
            hooks_list.append(RichAskQuestionHook())

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

    def _handle_slash_command(self, raw_cmd: str, agent: Agent, cfg: Any) -> bool:
        """Handles `/...` commands. Returns True if session should exit or restart."""
        parts = raw_cmd.strip().split()
        cmd = parts[0].lower()
        args = parts[1:]
        workspace = self.options.workspaces[0] if self.options.workspaces else os.getcwd()

        if cmd in ("/exit", "/quit"):
            self.console.print("[dim]Session closed.[/dim]")
            return True

        if cmd == "/help":
            tbl = Table(title="Interactive Commands & Shortcuts", border_style="bright_blue", show_lines=False)
            tbl.add_column("Command", style="bold cyan", no_wrap=True)
            tbl.add_column("Description", style="white")
            for k, desc in SLASH_COMMANDS.items():
                tbl.add_row(k, desc)
            tbl.add_row("@path/to/file", "Inline a workspace file directly into your prompt (Tab-completable)")
            tbl.add_row("!<shell command>", "Execute a local shell command immediately without calling the model")
            self.console.print(tbl)
            return False

        if cmd == "/status":
            model_label, backend_label, ctx_limit = self._active_model_summary(cfg)
            usage = agent.conversation.total_usage
            p_tok = (usage.prompt_token_count or 0) if usage else self.total_prompt_tokens
            c_tok = (usage.candidates_token_count or 0) if usage else self.total_output_tokens
            t_tok = (usage.thoughts_token_count or 0) if usage else 0
            tot_tok = (usage.total_token_count or 0) if usage else (p_tok + c_tok)
            pct = min(100.0, (tot_tok / max(1, ctx_limit)) * 100.0)

            tbl = Table(title="Session Telemetry & Harness Status", border_style="cyan")
            tbl.add_column("Metric", style="bold dim")
            tbl.add_column("Value", style="white")
            tbl.add_row("Active Model", f"[bold cyan]{model_label}[/bold cyan] ({backend_label})")
            tbl.add_row("Conversation ID", agent.conversation.conversation_id or "(local)")
            tbl.add_row("Completed Turns", str(self.session_turns))
            tbl.add_row("Executed Tool Calls", str(self.session_tool_calls))
            tbl.add_row("Prompt / Output / Thought Tokens", f"{p_tok:,} in · {c_tok:,} out · {t_tok:,} thought")
            tbl.add_row("Context Window Utilization", f"{tot_tok:,} / {ctx_limit:,} tokens ({pct:.1f}%)")
            tbl.add_row("Compaction Events", str(len(agent.conversation.compaction_indices)))
            self.console.print(tbl)
            return False

        if cmd == "/clear":
            agent.conversation.clear_history()
            self.console.clear()
            self.console.print(self.render_welcome_banner(cfg))
            self.console.print("[green]✓ Cleared conversation history.[/green]")
            return False

        if cmd == "/thoughts":
            self.show_thoughts = not self.show_thoughts
            state = "ENABLED" if self.show_thoughts else "HIDDEN"
            self.console.print(f"[cyan]✓ Reasoning trace display: {state}[/cyan]")
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
            self.options.preset = None
            if target_model.endswith(".litertlm"):
                self.options.backend = "litert"
                self.options.model_path = target_model
                self.options.model = None
            else:
                self.options.backend = "openai"
                self.options.model = target_model
                self.options.model_path = None
                if len(args) > 1:
                    self.options.base_url = args[1]
            self._restart_requested = True
            self.console.print(f"[green]✓ Switching model to '{target_model}'...[/green]")
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
            proc = subprocess.run(
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
            proc = subprocess.run(
                ["git", "-C", workspace, "status", "--short"],
                capture_output=True,
                text=True,
                check=False,
            )
            if not proc.stdout.strip():
                self.console.print("[dim]Working tree is already clean.[/dim]")
                return False
            self.console.print(Panel(proc.stdout.strip(), title="[yellow]Modified Files[/yellow]", border_style="yellow"))
            ans = input("  Discard unstaged modifications via `git checkout -- .`? [y/N]: ").strip().lower()
            if ans in ("y", "yes"):
                subprocess.run(["git", "-C", workspace, "checkout", "--", "."], check=False)
                self.console.print("[green]✓ Reverted unstaged changes.[/green]")
            return False

        self.console.print(f"[yellow]Unknown command '{cmd}'. Type /help for available commands.[/yellow]")
        return False

    async def _run_turn(self, agent: Agent, user_text: str, cfg: Any) -> None:
        workspace = self.options.workspaces[0] if self.options.workspaces else os.getcwd()
        augmented_prompt, attached = expand_at_file_mentions(user_text, workspace)
        if attached:
            self.console.print(
                f"  [dim]📎 Attached {', '.join(f'[cyan]{f}[/cyan]' for f in attached)}[/dim]"
            )

        t0 = time.perf_counter()
        thought_buf: list[str] = []
        text_buf: list[str] = []
        tool_events: list[tuple[str, str, str | None]] = []
        seen_tool_ids: set[str] = set()

        def _render_live_frame() -> Group:
            renderables: list[Any] = []
            elapsed = time.perf_counter() - t0

            if thought_buf and self.show_thoughts:
                full_thought = "".join(thought_buf).strip()
                # Show last 6 lines during live streaming to keep screen compact
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

            for badge, summary, diff_preview in tool_events:
                line = Text()
                line.append("  ⏺ ", style="bold cyan")
                line.append(f"{badge}", style="bold bright_cyan")
                line.append(f"({summary})", style="white")
                renderables.append(line)
                if diff_preview:
                    renderables.append(
                        Panel(
                            Syntax(diff_preview, "diff", theme="monokai", word_wrap=True),
                            border_style="dim",
                            padding=(0, 1),
                        )
                    )

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

        await agent.conversation.send(augmented_prompt)

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
                                tool_events.append(format_tool_badge(tc))

                    live.update(_render_live_frame())
            except KeyboardInterrupt:
                await agent.conversation.cancel()
                self.console.print("\n[yellow]⚠ Turn cancelled by user (Ctrl+C).[/yellow]")
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

        _, _, ctx_limit = self._active_model_summary(cfg)
        footer = Text()
        footer.append(f"  ⏱ {dt:.2f}s", style="dim")
        if tok_per_s > 0:
            footer.append(f" · ⚡ {tok_per_s:.1f} tok/s", style="dim cyan")
        footer.append(f" · 📊 ~{out_tok:,} out tok", style="dim")
        if tool_events:
            footer.append(f" · 🛠 {len(tool_events)} tool call(s)", style="dim yellow")
        footer.append(f" · ctx {(self.total_prompt_tokens + self.total_output_tokens):,}/{ctx_limit:,}", style="dim")
        self.console.print(footer)
        self.console.print()

    async def run(self) -> int:
        """Main loop supporting hot-reloading presets/models/policies via `/preset`, `/model`, `/policy`."""
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
                style=pt_style,
                complete_while_typing=False,
            )

        while True:
            self._restart_requested = False
            base_cfg = build_agent_config(self.options)
            cfg = self._upgrade_interactive_config(base_cfg)
            model_label, backend_label, _ = self._active_model_summary(cfg)

            self.console.print(self.render_welcome_banner(cfg))

            async with Agent(cfg) as agent:
                while True:
                    try:
                        branch, dirty = _git_branch_and_dirty(workspace)
                        git_info = f" | git:{branch}{'*' if dirty else ''}" if branch else ""
                        pol = (
                            self.options.policy_mode.value
                            if isinstance(self.options.policy_mode, PolicyMode)
                            else str(self.options.policy_mode)
                        )
                        toolbar_html = HTML(
                            f" <b>{model_label}</b> ({backend_label}) | policy:<b>{pol}</b>{git_info} | "
                            f"Tab: complete · <b>/help</b> commands · <b>@file</b> attach · <b>!cmd</b> shell "
                        )
                        prompt_html = HTML(f"<model_tag>[{model_label}]</model_tag> <prompt>❯ </prompt>")

                        if prompt_session is not None:
                            user_input = await prompt_session.prompt_async(
                                prompt_html,
                                bottom_toolbar=toolbar_html,
                            )
                        else:
                            user_input = await _nonblocking_input("❯ ")
                    except (EOFError, KeyboardInterrupt):
                        self.console.print("\n[dim]Session closed.[/dim]")
                        return 0

                    text = user_input.strip()
                    if not text:
                        continue

                    if text.lower() in ("exit", "quit"):
                        self.console.print("[dim]Session closed.[/dim]")
                        return 0

                    # Direct !shell shortcut (like Claude Code / Codex CLI)
                    if text.startswith("!"):
                        shell_cmd = text[1:].strip()
                        if shell_cmd:
                            self.console.print(Rule(f"[bold yellow]$ {shell_cmd}[/bold yellow]", style="yellow"))
                            proc = await asyncio.to_thread(
                                subprocess.run,
                                shell_cmd,
                                shell=True,
                                cwd=workspace,
                                check=False,
                            )
                            status_color = "green" if proc.returncode == 0 else "red"
                            self.console.print(f"[{status_color}]↳ exit code {proc.returncode}[/{status_color}]\n")
                        continue

                    # Slash commands
                    if text.startswith("/"):
                        should_break = self._handle_slash_command(text, agent, cfg)
                        if should_break:
                            break
                        continue

                    await self._run_turn(agent, text, cfg)

            if not self._restart_requested:
                break

        return 0
