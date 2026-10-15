# Changelog

All notable changes to `agy-local` will be documented in this file.

## [0.1.0] - 2026-10-15

### Added
- **Claude Code / OpenAI Codex Inspired Rich Terminal Studio (`AgyRichTUI`)**:
  - Live header card with active model, backend (`LiteRT GPU/NPU` or `LocalOpenAIAgentConfig`), git branch & dirty file counter, safety policy badge, and MCP server status.
  - Real-time Rich `Live` streaming with collapsible `💭 Thinking` traces, structured tool badges (`Bash`, `Read`, `Edit`, `Write`, `Grep`, `ListDir`), and syntax-highlighted unified diff previews for file edits.
  - Interactive `prompt_toolkit` session with `Tab` completion for `/slash` commands (`/help`, `/status`, `/preset`, `/model`, `/policy`, `/models`, `/doctor`, `/diff`, `/undo`, `/thoughts`, `/clear`, `/exit`), `@file` inline context attachment, and `!cmd` shell execution.
  - Per-turn telemetry footer (`⏱ latency · ⚡ tok/s · 📊 output tokens · 🛠 tool calls · ctx usage`).
- **Autonomous CLI, Server Launcher & MCP Server (`agy-local` / `agy-local-mcp`)**:
  - `agy-local exec`: Executes single-shot autonomous coding tasks with live streaming of tokens, reasoning (`--show-thoughts`), and tool calls (`--show-tools`).
  - `agy-local serve`: Launches local vLLM or Ollama servers with ChronosML auto-tuned `--gpu-memory-utilization` and `--max-model-len`.
  - `agy-local mcp` (`agy-local-mcp`): JSON-RPC 2.0 `stdio` MCP server exposing `run_local_subagent`, `list_local_presets`, and `probe_local_models`.
  - `agy-local presets`, `agy-local models`, `agy-local doctor`, and `agy-local import-litert`.
