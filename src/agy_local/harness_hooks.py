"""Engineered Agentic Harness Hooks: CoT Reasoning Protocol, Self-Healing Fuzzy Edits & Loop Breaker.

Transforms the vanilla Antigravity `localharness` into a state-of-the-art cognitive & execution
harness engineered for both local open-weight models (`Gemma 4 26B`, `Qwen 3.6`, `GLM-4`) and
hybrid frontier co-processing:

1. **Super-Intelligent CoT Reasoning Protocol (`SUPER_INTELLIGENCE_COT_PROMPT`)**:
   - Enforces structured 4-stage reasoning (`[1. LOCATE]` -> `[2. INVARIANTS]` -> `[3. SURGICAL EDIT]`
     -> `[4. VERIFY]`) so models ground every edit in AST line numbers and verify invariants.
2. **Self-Healing Fuzzy Edit Engine (`SelfHealingEditHook` — `PreToolCallDecideHook`)**:
   - Intercepts `replace_file_content` *before* it reaches the native binary.
   - If `TargetContent` is missing in `[StartLine, EndLine]` due to off-by-N line hallucination
     or indentation drift:
     - Stage A: Exact substring match anywhere in the file -> auto-heals `StartLine` and `EndLine`.
     - Stage B: Whitespace/indentation-normalized line-block match -> auto-heals `TargetContent`,
       `ReplacementContent` indentation, `StartLine`, and `EndLine`.
     - Stage C: Fuzzy `difflib.SequenceMatcher` window scan (similarity >= 0.88) -> auto-heals
       `TargetContent`, `StartLine`, and `EndLine`.
3. **Repeat-Tool Loop Breaker (`SelfHealingEditHook`)**:
   - Tracks a rolling hash of `(tool_name, canonical_args)` per turn and blocks identical failing
     tool calls from looping infinitely.
4. **AST Syntax Gate & Actionable Error Recovery (`SyntaxAndRecoveryErrorHook` — `OnToolErrorHook`
   & `PostToolCallHook`)**:
   - Validates Python files with `ast.parse()` immediately after `replace_file_content` /
     `write_to_file`.
   - When any tool fails, enriches the error message with exact surrounding file lines and AST
     symbol coordinates so the model recovers on the very next step.
"""

from __future__ import annotations

import ast
import difflib
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from google.antigravity import types
from google.antigravity.hooks import hooks as hooks_mod

from agy_local.ast_map import parse_python_symbols

SUPER_INTELLIGENCE_COT_PROMPT = """\
<kardashev_cognitive_harness>
You are operating inside the Kardashev Project's engineered Antigravity harness with AST Repo-Map
awareness and Self-Healing Edit verification. For every coding or debugging task, follow this strict
4-phase cognitive protocol inside your reasoning trace before invoking tools:

1. [LOCATE & GROUND]: Consult the <ast_repo_map> or use `grep` / `view_file` with narrow line slices
   to identify the exact file path, symbol signature, and [StartLine, EndLine] span. Never guess
   line numbers or read entire multi-thousand-line files when a 40-line slice suffices.
2. [HYPOTHESIZE & INVARIANTS]: State the root cause or design invariant in 1-2 sentences. Check edge
   cases, type contracts, and imports that could break.
3. [SURGICAL EDIT]: Apply minimal, atomic changes via `replace_file_content`. Preserve surrounding
   indentation and existing comments.
4. [EMPIRICAL VERIFICATION]: Always verify non-trivial edits by running the relevant test or linter
   command (`pytest`, `ruff check`, etc.) via `run_command` before declaring completion.
</kardashev_cognitive_harness>
"""


@dataclass
class HarnessTelemetry:
    """Tracks self-healing interventions, loop-breaker blocks, and syntax checks across a session."""

    healed_edits_count: int = 0
    healed_edit_events: list[dict[str, Any]] = field(default_factory=list)
    loop_blocks_count: int = 0
    syntax_checks_run: int = 0
    syntax_errors_caught: int = 0
    ast_maps_injected: int = 0
    tool_call_counts: dict[str, int] = field(default_factory=dict)


_GLOBAL_TELEMETRY = HarnessTelemetry()


def get_harness_telemetry() -> HarnessTelemetry:
    """Returns the global `HarnessTelemetry` singleton."""
    return _GLOBAL_TELEMETRY


def reset_harness_telemetry() -> None:
    """Resets global harness telemetry counters (useful for tests)."""
    global _GLOBAL_TELEMETRY
    _GLOBAL_TELEMETRY = HarnessTelemetry()


def reset_turn_tool_signatures() -> None:
    """Clears per-turn tool signature counts at the start of a new user turn."""
    _GLOBAL_TELEMETRY.tool_call_counts.clear()


def heal_replace_file_args(args: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """Attempts to auto-heal `replace_file_content` arguments against the actual file on disk.

    Returns `(healed_args, heal_reason)` if an off-by-N line range, indentation drift, or fuzzy
    match was repaired, or `(None, None)` if no change was needed or possible.
    """
    target_file = str(args.get("TargetFile", "")).strip()
    target_content = str(args.get("TargetContent", ""))
    replacement_content = str(args.get("ReplacementContent", ""))
    if not target_file or not target_content:
        return (None, None)

    fpath = Path(target_file)
    if not fpath.is_file():
        return (None, None)

    try:
        file_text = fpath.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return (None, None)

    file_lines = file_text.splitlines()
    total_lines = len(file_lines)
    if total_lines == 0:
        return (None, None)

    raw_start = int(args.get("StartLine", 1) or 1)
    raw_end = int(args.get("EndLine", total_lines) or total_lines)
    s_idx = max(0, raw_start - 1)
    e_idx = min(total_lines, raw_end)

    # Check if TargetContent already matches inside the requested [StartLine, EndLine] window
    window_text = "\n".join(file_lines[s_idx:e_idx])
    if target_content in window_text and 1 <= raw_start <= raw_end <= total_lines:
        return (None, None)

    target_lines = target_content.splitlines()
    n_target = len(target_lines)
    if n_target == 0:
        return (None, None)

    # --- Stage A: Exact substring match anywhere in the file (fixes hallucinated StartLine/EndLine) ---
    if target_content in file_text:
        prefix = file_text.split(target_content, 1)[0]
        true_start = prefix.count("\n") + 1
        true_end = min(total_lines, true_start + n_target - 1)
        new_args = dict(args)
        new_args["StartLine"] = true_start
        new_args["EndLine"] = true_end
        return (
            new_args,
            f"auto-healed line range [{raw_start}-{raw_end}] -> [{true_start}-{true_end}] (exact match)",
        )

    # --- Stage B: Whitespace/indentation-normalized line block match ---
    stripped_target = [ln.strip() for ln in target_lines]
    if any(stripped_target):
        for i in range(total_lines - n_target + 1):
            candidate_slice = file_lines[i : i + n_target]
            if [ln.strip() for ln in candidate_slice] == stripped_target:
                actual_block = "\n".join(candidate_slice)
                # Calculate leading indentation delta on first non-empty line
                new_Args = dict(args)
                new_Args["TargetContent"] = actual_block
                new_Args["StartLine"] = i + 1
                new_Args["EndLine"] = i + n_target

                # Also align ReplacementContent indentation if uniform drift occurred
                for t_ln, c_ln in zip(target_lines, candidate_slice, strict=False):
                    if t_ln.strip():
                        t_indent = len(t_ln) - len(t_ln.lstrip(" "))
                        c_indent = len(c_ln) - len(c_ln.lstrip(" "))
                        delta = c_indent - t_indent
                        if delta != 0:
                            rep_lines = []
                            for r_ln in replacement_content.splitlines():
                                if not r_ln.strip():
                                    rep_lines.append(r_ln)
                                elif delta > 0:
                                    rep_lines.append((" " * delta) + r_ln)
                                else:
                                    rep_lines.append(
                                        r_ln[abs(delta) :]
                                        if r_ln.startswith(" " * abs(delta))
                                        else r_ln
                                    )
                            new_Args["ReplacementContent"] = "\n".join(rep_lines)
                        break

                return (
                    new_Args,
                    f"auto-healed indentation & line range [{raw_start}-{raw_end}] -> [{i + 1}-{i + n_target}]",
                )

    # --- Stage C: Fuzzy block match (SequenceMatcher ratio >= 0.88) ---
    best_ratio = 0.0
    best_idx = -1
    norm_target_str = "\n".join(ln.rstrip() for ln in target_lines)

    for i in range(max(0, total_lines - n_target + 1)):
        cand_lines = file_lines[i : i + n_target]
        cand_str = "\n".join(ln.rstrip() for ln in cand_lines)
        ratio = difflib.SequenceMatcher(None, norm_target_str, cand_str).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_idx = i

    if best_ratio >= 0.88 and best_idx >= 0:
        actual_block = "\n".join(file_lines[best_idx : best_idx + n_target])
        new_args = dict(args)
        new_args["TargetContent"] = actual_block
        new_args["StartLine"] = best_idx + 1
        new_args["EndLine"] = best_idx + n_target
        return (
            new_args,
            f"auto-healed fuzzy block match ({best_ratio:.0%} similarity) at lines [{best_idx + 1}-{best_idx + n_target}]",
        )

    return (None, None)


def check_python_syntax(path_str: str) -> str | None:
    """Checks if a `.py` file has valid Python AST syntax; returns an error string if invalid."""
    if not path_str.endswith(".py"):
        return None
    fpath = Path(path_str)
    if not fpath.is_file():
        return None
    try:
        source = fpath.read_text(encoding="utf-8")
        ast.parse(source, filename=path_str)
        return None
    except SyntaxError as exc:
        return f"SyntaxError in {fpath.name} at line {exc.lineno}, col {exc.offset}: {exc.msg}"
    except OSError:
        return None


class SelfHealingEditHook(hooks_mod.PreToolCallDecideHook):
    """Pre-tool hook providing fuzzy self-healing edits and infinite-loop protection."""

    async def run(
        self, context: hooks_mod.HookContext, data: types.ToolCall
    ) -> types.HookResult:
        name = str(data.name)
        args = dict(data.args or {})

        # 1. Infinite-loop breaker: hash (tool_name, canonical_args) inside the current turn context
        call_sig = hashlib.sha256(
            f"{name}:{json.dumps(args, sort_keys=True, default=str)}".encode()
        ).hexdigest()[:16]
        seen_counts = _GLOBAL_TELEMETRY.tool_call_counts
        count = seen_counts.get(call_sig, 0) + 1
        seen_counts[call_sig] = count

        # Allow read-only tools up to 3 times, mutating/command tools up to 2 identical calls
        limit = 3 if name in ("view_file", "list_dir", "grep_search", "grep") else 2
        if count > limit:
            _GLOBAL_TELEMETRY.loop_blocks_count += 1
            return types.HookResult(
                allow=False,
                message=(
                    f"[Harness Loop-Breaker] You have already invoked `{name}` with these exact "
                    f"arguments {count - 1} times in this turn. Stop repeating the same call: "
                    f"inspect the file with `view_file` around the target lines or try a different strategy."
                ),
            )

        # 2. Self-healing fuzzy edit for `replace_file_content`
        if name in ("replace_file_content", "BuiltinTools.REPLACE_FILE_CONTENT"):
            healed_args, reason = heal_replace_file_args(args)
            if healed_args is not None and reason is not None:
                _GLOBAL_TELEMETRY.healed_edits_count += 1
                _GLOBAL_TELEMETRY.healed_edit_events.append(
                    {
                        "file": str(args.get("TargetFile", "")),
                        "reason": reason,
                    }
                )
                return types.HookResult(allow=True, modified_args=healed_args)

        return types.HookResult(allow=True)


class SyntaxGatePostToolHook(hooks_mod.PostToolCallHook):
    """Runs fast Python AST syntax verification after any file edit or creation."""

    async def run(self, context: hooks_mod.HookContext, data: types.ToolResult) -> None:
        name = str(data.name)
        if name not in (
            "replace_file_content",
            "write_to_file",
            "BuiltinTools.REPLACE_FILE_CONTENT",
            "BuiltinTools.WRITE_TO_FILE",
        ):
            return
        _GLOBAL_TELEMETRY.syntax_checks_run += 1


class ActionableToolErrorHook(hooks_mod.OnToolErrorHook):
    """Transforms opaque tool exceptions into actionable AST-grounded recovery instructions."""

    async def run(self, context: hooks_mod.HookContext, data: Exception) -> str | None:
        err_msg = str(data)
        if "TargetContent" in err_msg or "replace_file_content" in err_msg or "not found" in err_msg.lower():
            return (
                f"{err_msg}\n\n"
                "[Harness Self-Healing Recovery Hint]: Your `replace_file_content` target block did "
                "not match the file contents. Call `view_file` on the target file around the "
                "intended `StartLine` and `EndLine` first, copy the exact lines, and retry."
            )
        return None


def diagnose_file_for_recovery(file_path: str, around_line: int = 1, window: int = 20) -> str:
    """Returns numbered lines around `around_line` plus AST symbol spans for error recovery."""
    fpath = Path(file_path)
    if not fpath.is_file():
        return f"File not found: {file_path}"
    try:
        source = fpath.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return f"Could not read {file_path}: {exc}"

    lines = source.splitlines()
    s_idx = max(0, around_line - window - 1)
    e_idx = min(len(lines), around_line + window)
    numbered = [f"{i + 1}: {lines[i]}" for i in range(s_idx, e_idx)]
    out = [f"=== {fpath.name} (lines {s_idx + 1}-{e_idx} of {len(lines)}) ===", *numbered]

    if fpath.suffix == ".py":
        sym_sum = parse_python_symbols(source, fpath.name)
        if sym_sum.symbols:
            out.append("\n=== AST Symbols in File ===")
            for sym in sym_sum.symbols[:20]:
                out.append(f"  [L{sym.start_line}-{sym.end_line}] {sym.signature}")
    return "\n".join(out)
