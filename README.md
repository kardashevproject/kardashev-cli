# agy-local

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Antigravity SDK](https://img.shields.io/badge/Antigravity-localharness-success.svg)](https://github.com/google-antigravity/antigravity-sdk-python)

**`agy-local`** is an open-source CLI, interactive terminal REPL, and configuration harness that lets you run the full **Google Antigravity `localharness`** (`view_file`, `replace_file_content`, `write_to_file`, `run_command`, `grep`, `list_dir`, MCP servers, and skills) backed by **any local or open-weight model**:

- **Managed On-Device LiteRT Runtime (`LiteRTAgentConfig`)**: Run `Gemma 4 26B A4B` natively on GPU (CUDA / Apple Metal) or NPU with zero API keys, speculative decoding, and automatic 64k KV-cache compaction (`token_threshold=40960`).
- **OpenAI-Compatible Local & Open Model Servers (`LocalOpenAIAgentConfig`)**: Connect the Antigravity harness to **vLLM**, **Ollama**, **SGLang**, **LM Studio**, or **Zhipu GLM API** endpoints with automatic `.lightweight()` tool and prompt optimizations.
- **Curated Open-Weight Presets**: Built-in presets for `gemma4-26b`, `glm4-9b-ollama`, `glm4-vllm`, `glm-zhipu-api`, `qwen3.6-35b-vllm`, `qwen3.8-27b-vllm`, `qwen3-vl-8b-vllm`, `gpt-oss-20b-vllm`, and `lm-studio`.
- **Zero-Config Local Discovery (`agy-local models` & `agy-local doctor`)**: Automatically scans `~/.litert-lm/models`, `~/.cache/huggingface/hub`, and `~/models`, and probes local ports (`8000`, `11434`, `1234`, `30000`) to detect ready-to-run checkpoints and live servers.

---

## Quickstart

### 1. Install `agy-local`

```bash
git clone https://github.com/kardashevproject/agy-local.git
cd agy-local
pip install -e ".[litert]"
```

### 2. Import the Gemma 4 26B LiteRT Checkpoint (On-Device GPU/NPU)

```bash
agy-local import-litert --preset gemma4-26b
# Or directly via litert-lm:
# litert-lm import --from-huggingface-repo=litert-community/gemma-4-26B-A4B-it-litert-lm gemma-4-26B-A4B-it-web.litertlm gemma4-26b
```

### 3. Launch the Interactive Antigravity Studio TUI (`agy-local chat`)

```bash
# Run on-device with Gemma 4 26B via LiteRT:
agy-local chat --preset gemma4-26b

# Run with GLM-4 via Ollama:
agy-local chat --preset glm4-9b-ollama

# Run with Qwen3.6-35B-A3B-FP8 or GLM-4 on local vLLM:
agy-local chat --preset qwen3.6-35b-vllm

# Point to any custom OpenAI-compatible server + attach ChronosML MCP:
agy-local chat --backend openai --model glm4:9b --base-url http://127.0.0.1:11434/v1 --with-chronosml
```

---

## Interactive Rich TUI (`AgyRichTUI`)

Inspired by **Claude Code** and **OpenAI Codex CLI**, `agy-local chat` launches a state-of-the-art, keyboard-driven terminal studio (`AgyRichTUI`) featuring:

- **Live Workspace, Git, Context Gauge & Ralph Header**: Displays the active model, hardware backend (`LiteRT GPU/NPU` or `LocalOpenAIAgentConfig`), git branch & dirty file counter, safety policy badge (`ALLOW-ALL`, `SANDBOX`, `ASK`), autonomous Ralph verification loop status, and active MCP servers.
- **Real-Time Streaming with Thinking, Live Tool Results & Unified Diffs**:
  - Collapsible reasoning (`💭 Thinking`) panel streamed in real time.
  - Structured tool call timeline (`⏺ Bash(...)`, `⏺ Read(...)`, `⏺ Edit(...)`, `⏺ Write(...)`, `⏺ Grep(...)`, `⏺ ListDir(...)`) paired with live `PostToolCallHook` execution durations (`[ms]`) and result trees (`⎿ ✓ ...` / `⎿ ✗ ...`).
  - Automatic syntax-highlighted unified diff previews (`+` green, `-` red) whenever the model edits or writes files.
  - Per-turn telemetry footer (`⏱ latency · ⚡ tok/s · 📊 output tokens · 🛠 tool calls · ctx usage % · 🗜 compactions`).
- **Autonomous Self-Verification Loop (`/ralph` & `--ralph`)**:
  - Powered by `RalphVerifyStopHook` (`StopHook`): automatically executes a verification shell command (e.g. `pytest -q && ruff check`) before allowing the agent turn to finish. If tests fail, the failure output is automatically injected back into the agent to self-repair.
- **Workspace Git Checkpoints & Time-Travel Rewind (`/checkpoint` & `/rewind`)**:
  - Automatically snapshots the workspace tree before every turn and lets you rewind to any numbered checkpoint via `/rewind <index>`.
- **Interactive Slash Commands, Context Providers & Keybindings** (with `Tab` completion):

| Shortcut / Command | Description |
|---|---|
| `/help` | Display all interactive slash commands, keybindings, `@file` mentions, and `!shell` shortcuts. |
| `/status` | Inspect session token usage, Unicode context utilization gauge (`█░`), turn count, checkpoints, and compactions. |
| `/preset [key]` | List all presets or hot-swap the active model preset mid-session (e.g., `/preset gemma4-26b`). |
| `/model <id> [url]` | Hot-swap to any model ID, `.litertlm` path, or OpenAI base URL mid-session. |
| `/policy <mode>` | Switch safety policy on the fly (`allow-all`, `ask`, `sandbox`) or press `Alt+P`. |
| `/ralph <cmd\|off>` | Enable or disable autonomous self-verification loop (e.g., `/ralph pytest -q`). |
| `/checkpoint [create\|list]` | Create a named git workspace snapshot or list all turn checkpoints. |
| `/rewind <index>` | Rewind workspace files to a numbered checkpoint from `/checkpoint list`. |
| `/compact` | Summarize earlier turns and reset the conversation window to free context tokens. |
| `/save [name]` / `/load [name]` | Save or restore session transcripts in `~/.local/share/agy-local/sessions/`. |
| `/export [path.md]` | Export the full session transcript, thinking traces, and tool timeline to Markdown. |
| `/models` | Scan local disk checkpoints and probe localhost OpenAI inference endpoints. |
| `/doctor` | Run hardware, GPU, LiteRT, and MCP diagnostics inside the session. |
| `/diff` | Render a syntax-highlighted `git diff` of uncommitted workspace changes. |
| `/undo` | Revert unstaged modifications in the workspace (with interactive confirmation). |
| `/thoughts` | Toggle visibility of live reasoning/thinking traces (or press `Alt+T`). |
| `/clear` | Clear conversation history and reset the terminal view. |
| `/exit` | Exit the session (or press `Ctrl+D`). |
| `@path/to/file[:start-end]` | Inline a full workspace file or exact line slice (e.g. `@src/agy_local/cli.py:1-50`). |
| `@git:diff` / `@git:status` | Inline live uncommitted git diff or working tree status directly into your prompt. |
| `!<shell command>` | Execute a local shell command immediately in the workspace without invoking the model. |
| `Alt+T` / `Alt+P` | Instant keybindings to toggle thinking traces (`Alt+T`) or cycle safety policy (`Alt+P`). |

---

## CLI Reference

| Subcommand | Description |
|---|---|
| `agy-local chat` | Launch the Rich Terminal UI (`AgyRichTUI`) with full Antigravity coding tools (pass `--plain` for minimal REPL). |
| `agy-local exec "<prompt>"` | Execute a single autonomous coding task and stream tokens, thoughts (`--show-thoughts`), and tool calls (`--show-tools`). |
| `agy-local serve` | Launch a local vLLM or Ollama server (`--preset qwen3.6-35b-vllm`) with ChronosML auto-computed `--gpu-memory-utilization` and `--max-model-len`. |
| `agy-local mcp` | Run the JSON-RPC 2.0 `stdio` MCP server (`agy-local-mcp`) so frontier agents can delegate tasks to local models at zero API cost. |
| `agy-local presets` | List all built-in LiteRT, GLM-4, Qwen3, and GPT-OSS presets (`--json` supported). |
| `agy-local models` | Scan local disk (`~/.litert-lm`, `~/.cache/huggingface`, `~/models`) and probe local inference ports (`8000`, `11434`, `1234`, `30000`). |
| `agy-local doctor` | Verify Python, `litert-lm`, CUDA GPU availability, `chronosml-mcp`, and active local model endpoints. |
| `agy-local import-litert` | Download and register a LiteRT checkpoint from HuggingFace. |

---

## Plug-and-Play MCP Server (`agy-local-mcp`)

Expose local on-device models (`gemma4-26b`, `qwen3.6-35b-vllm`, `glm4-9b-ollama`) as a zero-cost subagent tool inside **Antigravity 2.0**, **Claude Code**, or **Cursor**:

```json
{
  "mcpServers": {
    "agy-local": {
      "command": "agy-local-mcp",
      "args": []
    }
  }
}
```

---

## Python SDK Usage

```python
import asyncio
from agy_local import AgyRichTUI, HarnessOptions, PolicyMode

async def main():
    tui = AgyRichTUI(
        HarnessOptions(
            preset="gemma4-26b",          # Or "glm4-9b-ollama", "qwen3.6-35b-vllm"
            workspaces=["."],
            policy_mode=PolicyMode.ALLOW_ALL,
            speculative_decoding=True,
            with_chronosml_mcp=True,
        )
    )
    await tui.run()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## License

Apache License 2.0 — Copyright (c) 2026 The Kardashev Project.
