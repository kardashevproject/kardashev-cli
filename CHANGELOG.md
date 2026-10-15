# Changelog

All notable changes to `agy-local` will be documented in this file.

## [0.1.0] - 2026-10-15

### Added
- **4-Phase Super-Intelligence Cognitive Harness (`src/agy_local/harness_hooks.py`)**:
  - `SUPER_INTELLIGENCE_COT_PROMPT`: Enforces a 4-phase engineering protocol (`[1. LOCATE & GROUND]`, `[2. HYPOTHESIZE & INVARIANTS]`, `[3. SURGICAL EDIT]`, `[4. EMPIRICAL VERIFICATION]`) across both local models and subagents.
  - `SelfHealingEditHook` (`PreToolCallDecideHook`): 3-stage self-healing fuzzy matcher for `replace_file_content` (Stage A exact line-shift repair, Stage B indentation alignment, Stage C $\ge 88\%$ `SequenceMatcher` fuzzy block repair) + per-turn SHA-256 `(tool, args)` infinite-loop breaker.
  - `SyntaxGatePostToolHook` & `ActionableToolErrorHook`: Immediate Python AST syntax verification and AST-grounded recovery diagnostics on tool failure.
- **AST Repo-Map & Surgical Context Scout (`src/agy_local/ast_map.py`)**:
  - Zero-dependency AST & regex symbol indexer (`parse_python_symbols`, `parse_generic_symbols`, `render_compact_repo_map`, `scout_workspace`) with automatic Turn-0 AST Repo-Map injection and CLI/TUI commands (`agy-local scout "<query>"`, `/ast`, `/scout`).
- **One-Command Frontier Agent Co-Processor Integration (`src/agy_local/integrate.py`)**:
  - `agy-local integrate [antigravity|claude|cursor|codex|all]` and `/integrate`: Auto-registers `agy-local-mcp` and installs the `agy-local-coprocessor` skill/rules into Antigravity 2.0, Claude Code, Cursor, and OpenAI Codex CLI.
  - Expanded `agy-local-mcp` JSON-RPC 2.0 server with 5 tools: `scout_codebase`, `apply_smart_patch`, `verify_and_repair`, `run_local_agent_task`, and `list_local_models`.
- **Claude Code / OpenAI Codex Inspired Rich Terminal Studio (`AgyRichTUI`)**:
  - Live header card with active model, backend (`LiteRT GPU/NPU` or `LocalOpenAIAgentConfig`), git branch & dirty file counter, safety policy badge, Ralph verification loop badge, and MCP server status.
  - Real-time Rich `Live` streaming with collapsible `💭 Thinking` traces, structured tool badges (`Bash`, `Read`, `Edit`, `Write`, `Grep`, `ListDir`), live `PostToolCallHook` result trees (`⎿ ✓`), and syntax-highlighted unified diff previews for file edits.
  - Autonomous self-verification loop (`/ralph` & `--ralph` via `RalphVerifyStopHook`), git workspace checkpoints & time-travel rewind (`/checkpoint`, `/rewind`), `@file:start-end` & `@git:diff` context injection, and session save/load/export/compact.
- **CLI Installer, Self-Updater & Persistent Config (`install.sh`, `src/agy_local/updater.py`)**:
  - One-line installer (`install.sh`), short `agy` alias, in-place self-updater (`agy-local update` / `/update`), persistent user config (`agy-local config`), and shell tab-completions (`agy-local completions [bash|zsh|fish]`).
