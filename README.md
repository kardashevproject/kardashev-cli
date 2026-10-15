# kardashev-cli (`kardashev` / `ksh` / `agy`)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Antigravity SDK](https://img.shields.io/badge/Antigravity-localharness-success.svg)](https://github.com/google-antigravity/antigravity-sdk-python)

**`kardashev-cli`** (CLI executables **`kardashev`**, **`ksh`**, **`agy`**, and **`agy-local`**) is The Kardashev Project's flagship open-source CLI, interactive terminal studio, distributed OpenCommons CRDT research engine, and coding harness that lets you run the full **Google Antigravity `localharness`** (`view_file`, `replace_file_content`, `write_to_file`, `run_command`, `grep`, `list_dir`, MCP servers, and skills) backed by **any local or open-weight model**:

- **Managed On-Device LiteRT Runtime (`LiteRTAgentConfig`)**: Run `Gemma 4 26B A4B` natively on GPU (CUDA / Apple Metal) or NPU with zero API keys, speculative decoding, and automatic 64k KV-cache compaction (`token_threshold=40960`).
- **OpenAI-Compatible Local & Open Model Servers (`LocalOpenAIAgentConfig`)**: Connect the Antigravity harness to **vLLM**, **Ollama**, **SGLang**, **LM Studio**, or **Zhipu GLM API** endpoints with automatic `.lightweight()` tool and prompt optimizations, plus pre-flight endpoint liveness probing and zero-crash diagnostics if a local server is offline.
- **Curated Open-Weight Presets**: Built-in presets for `gemma4-26b`, `glm4-9b-ollama`, `glm4-vllm`, `glm-zhipu-api`, `qwen3.6-35b-vllm`, `qwen3.8-27b-vllm`, `qwen3-vl-8b-vllm`, `gpt-oss-20b-vllm`, and `lm-studio`.
- **Zero-Config Local Discovery (`kardashev models` & `kardashev doctor`)**: Automatically scans `~/.litert-lm/models`, `~/.cache/huggingface/hub`, and `~/models`, and probes local ports (`8000`, `11434`, `1234`, `30000`) to detect ready-to-run checkpoints and live servers.

- **4-Phase Super-Intelligence Cognitive Harness (`SUPER_INTELLIGENCE_COT_PROMPT`)**: Replaces generic tool prompts with a strict 4-phase engineering protocol (`[1. LOCATE & GROUND]` $\rightarrow$ `[2. HYPOTHESIZE & INVARIANTS]` $\rightarrow$ `[3. SURGICAL EDIT]` $\rightarrow$ `[4. EMPIRICAL VERIFICATION & COMMONS CONTRIBUTION]`) plus automatic Turn-0 AST Repo-Map and `<opencommons_knowledge>` injection.
- **Distributed OpenCommons Research Engine & Zero-Trust CRDT Gossip Hub (`src/agy_local/opencommons.py` & `opencommons-mcp`)**:
  - **Fast Structural & Behavioral Fingerprinting (`<5ms`)**: Alpha-normalized De Bruijn AST Merkle hash (`sid`) for $O(1)$ structural deduplication plus 16-probe canonical execution fingerprinting (`fp -> cid`) that merges distinct algorithms computing the same mathematical function into a single Canonical Equivalence Class.
  - **3-Layer Zero-Trust Security Sandbox**: Blocks `ast.Import`/`ast.ImportFrom`, `ast.Global`/`ast.Nonlocal`, all `__dunder__` attribute/identifier escapes (`__subclasses__`, `__globals__`, `__builtins__`), and forbidden dynamic builtins (`eval`, `exec`, `open`, `getattr`), paired with a pure-math `SAFE_EXEC_BUILTINS` whitelist and lazy execution compilation (`0` untrusted network code executed on pull).
  - **16-Bucket Merkle Tree + HTTP `ETag` (`304`) & Gzipped Delta Sync (`kardashev commons pull|push|serve-hub`)**: Synchronize verified research concepts globally via static GitHub Raw CDN (`commons/store.json`) or a live multi-threaded HTTP CRDT Gossip Hub (`kardashev commons serve-hub`).
  - **5-Gate Adversarial Peer Review & Tiered Commit Policy**: Automatically commits agent-contributed Python functions into `~/.opencommons/store.json` (via atomic `fcntl`-locked G-Set join-semilattice CRDT merge $\sqcup$) when all 5 adversarial gates pass, and stages domain-specific or human-review proposals into `~/.opencommons/pending_queue.json` for `/commons review`.
  - **Monotonic Counterexample Ratchet & Autonomous Post-Edit Harvesting**: Agents can report $(K+1)$-th input counterexamples to quarantine flawed blocks and attach `PitfallWarning`s, while `SyntaxGatePostToolHook` automatically harvests newly written, type-annotated pure helper functions after verified file edits.
- **3-Stage Self-Healing Fuzzy Edit Engine (`SelfHealingEditHook` + `SyntaxGatePostToolHook`)**: Automatically heals hallucinated `StartLine`/`EndLine` coordinates (Stage A), class-method indentation drift (Stage B), and minor whitespace/typo divergence $\ge 88\%$ (Stage C) before `replace_file_content` executes, backed by an AST syntax gate and per-turn SHA-256 infinite-loop breaker.
- **One-Command Frontier Agent Co-Processor (`kardashev integrate`)**: Connects `kardashev-mcp` / `agy-local-mcp` and installs the co-processor skill into **Antigravity 2.0**, **Claude Code**, **Cursor**, and **OpenAI Codex CLI**, cutting cloud input tokens by 60–80% via `scout_codebase`, `apply_smart_patch`, `commons_find`, and `commons_record`.

---

## Quickstart

### 1. One-Line CLI Install (Claude Code / Codex Style)

```bash
curl -fsSL https://raw.githubusercontent.com/kardashevproject/kardashev-cli/main/install.sh | bash
```

Or install via `uv tool` / `pip`:
```bash
uv tool install "git+https://github.com/kardashevproject/kardashev-cli.git"
# Or from a local clone:
git clone https://github.com/kardashevproject/kardashev-cli.git
cd kardashev-cli && pip install -e ".[litert]"
```

This installs **`kardashev`**, **`ksh`**, **`agy`**, **`agy-local`**, **`kardashev-mcp`**, **`agy-local-mcp`**, and **`opencommons-mcp`** into `~/.local/bin`.

### 2. Integrate with Frontier Coding Agents (`Antigravity`, `Claude Code`, `Cursor`, `Codex`)

Turn your local GPU/CPU into a high-speed AST Scout, Self-Healing Patch Engine, OpenCommons Knowledge Base, and Verification Co-Processor for frontier coding agents in one command:

```bash
kardashev integrate antigravity   # Or: claude, cursor, codex, all
```

### 3. Self-Update & Global OpenCommons Sync Anytime

```bash
kardashev update                  # Pulls latest CLI commits, upgrades dependencies, refreshes symlinks
kardashev commons pull            # Pulls latest verified OpenCommons concepts via Merkle-root + ETag sync
kardashev commons push <hub_url>  # Pushes local peer-reviewed discoveries to an OpenCommons CRDT Hub
kardashev commons serve-hub       # Launches a multi-threaded HTTP CRDT Gossip Hub on port 8765
```

### 4. Import the Gemma 4 26B LiteRT Checkpoint (On-Device GPU/NPU)

```bash
kardashev import-litert --preset gemma4-26b
```

### 5. Launch the Interactive Kardashev Studio TUI (`kardashev` / `ksh` / `agy`)

```bash
# Launch directly with your saved user defaults (~/.config/agy-local/config.json):
kardashev   # or: ksh / agy

# Run on-device with Gemma 4 26B via LiteRT + autonomous pytest verification loop:
kardashev chat --preset gemma4-26b --ralph "pytest -q"

# Run with GLM-4 via Ollama or Qwen3.6-35B on local vLLM:
kardashev chat --preset glm4-9b-ollama
kardashev chat --preset qwen3.6-35b-vllm
```

---

## Interactive Rich TUI (`AgyRichTUI`)

Inspired by **Claude Code** and **OpenAI Codex CLI**, `kardashev chat` (`ksh` / `agy`) launches a state-of-the-art, keyboard-driven terminal studio (`AgyRichTUI`) featuring:

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
| `/status` | Inspect session token usage, Unicode context utilization gauge (`█░`), self-healing edit stats, OpenCommons stats, checkpoints, and compactions. |
| `/ast [query]` | Render the ranked Aider-style AST Repository Map (`classes`, `methods`, `functions`, `[start-end]` lines). |
| `/scout <query>` | Run the Surgical Context Scout to extract ranked AST symbols and exact code slices (`<2,000 tokens`). |
| `/commons [action]` | Manage the OpenCommons research engine (`status`, `find <query>`, `review`, `approve <id>`, `reject <id>`, `pull [url]`, `push [url]`, `seed`). |
| `/integrate [target]` | Auto-configure `kardashev-mcp` / `agy-local-mcp` and co-processor skills into `antigravity`, `claude`, `cursor`, `codex`, or `all`. |
| `/preset [key]` | List all presets or hot-swap the active model preset mid-session (e.g., `/preset gemma4-26b`). |
| `/model <id> [url]` | Hot-swap to any model ID, `.litertlm` path, or OpenAI base URL mid-session (with automatic endpoint liveness check). |
| `/policy <mode>` | Switch safety policy on the fly (`allow-all`, `ask`, `sandbox`) or press `Alt+P`. |
| `/ralph <cmd\|off>` | Enable or disable autonomous self-verification loop (e.g., `/ralph pytest -q`). |
| `/checkpoint [create\|list]` | Create a named git workspace snapshot or list all turn checkpoints. |
| `/rewind <index>` | Rewind workspace files to a numbered checkpoint from `/checkpoint list`. |
| `/compact` | Summarize earlier turns and reset the conversation window to free context tokens. |
| `/save [name]` / `/load [name]` | Save or restore session transcripts in `~/.local/share/agy-local/sessions/`. |
| `/export [path.md]` | Export the full session transcript, thinking traces, and tool timeline to Markdown. |
| `/config [show\|save]` | Inspect or save current session settings as persistent user defaults in `~/.config/agy-local/config.json`. |
| `/update` | Self-update `kardashev-cli` in-place from GitHub `main` and refresh CLI symlinks. |
| `/models` | Scan local disk checkpoints and probe localhost OpenAI inference endpoints. |
| `/doctor` | Run hardware, GPU, LiteRT, OpenCommons, and MCP diagnostics inside the session. |
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
| `agy` / `agy-local [chat]` | Launch the Rich Terminal UI (`AgyRichTUI`) with full Antigravity coding tools (pass `--plain` for minimal REPL). |
| `agy-local exec "<prompt>"` | Execute a single autonomous coding task and stream tokens, thoughts (`--show-thoughts`), and tool calls (`--show-tools`). |
| `agy-local scout "<query>"` | Run the AST Repo-Map & Surgical Context Scout on a workspace (`--json` and `--max-slices` supported). |
| `agy-local commons [action]` | Query and manage the OpenCommons research engine (`status`, `find`, `record`, `review`, `approve`, `reject`, `seed`, `sync`, `mcp`). |
| `agy-local integrate [target]` | Auto-register `agy-local-mcp` & co-processor skills into `antigravity`, `claude`, `cursor`, `codex`, or `all`. |
| `agy-local update` (`--upgrade`) | Self-update `agy-local` from GitHub `main`, upgrade dependencies, and refresh `~/.local/bin` symlinks (`--check` supported). |
| `agy-local config [show\|set\|reset]` | View or set persistent user defaults in `~/.config/agy-local/config.json`. |
| `agy-local completions [bash\|zsh\|fish]` | Output shell tab-completion scripts (`eval "$(agy-local completions zsh)"`). |
| `agy-local serve` | Launch a local vLLM or Ollama server (`--preset qwen3.6-35b-vllm`) with ChronosML auto-computed `--gpu-memory-utilization` and `--max-model-len`. |
| `agy-local mcp` | Run the JSON-RPC 2.0 `stdio` MCP server (`agy-local-mcp`) so frontier agents can delegate tasks to local models at zero API cost. |
| `agy-local presets` | List all built-in LiteRT, GLM-4, Qwen3, and GPT-OSS presets (`--json` supported). |
| `agy-local models` | Scan local disk (`~/.litert-lm`, `~/.cache/huggingface`, `~/models`) and probe local inference ports (`8000`, `11434`, `1234`, `30000`). |
| `agy-local doctor` | Verify Python, `litert-lm`, CUDA GPU availability, `chronosml-mcp`, `opencommons-mcp`, and active local model endpoints. |
| `agy-local import-litert` | Download and register a LiteRT checkpoint from HuggingFace. |

---

## Frontier Agent Co-Processor MCP Server (`agy-local-mcp`)

Run `agy-local integrate all` (or configure `agy-local-mcp` manually) to expose 7 zero-cost local tools to **Antigravity 2.0**, **Claude Code**, **Cursor**, and **OpenAI Codex**:

| MCP Tool | Purpose for Frontier Agents |
|---|---|
| `scout_codebase` | Builds a ranked AST Repository Map + surgical code slices (`<2,000 tokens`) for a natural-language or symbol query, saving 60–80% of frontier input tokens. |
| `apply_smart_patch` | Applies a code replacement using the 3-stage self-healing fuzzy matcher (exact line-shift, indentation alignment, or $\ge 88\%$ fuzzy block match) with automatic Python AST syntax rollback. |
| `verify_and_repair` | Runs a test/linter command (`pytest`, `ruff`, `mypy`) and returns structured failure diagnostics or invokes the local agent to repair failures. |
| `commons_find` | Queries the persistent OpenCommons knowledge base (`~/.opencommons/store.json`) in `<5ms` by natural-language query, parameter/return types, and live `<2ms` I/O probe assertions, returning Dual Concept Cards + Refuted Pitfalls. |
| `commons_record` | Contributes a verified Python function to OpenCommons (computing AST Merkle `sid`, 16-probe behavioral `fp/cid`, and 5-gate adversarial peer review; auto-commits if all 5 gates pass or stages in `~/.opencommons/pending_queue.json`) or reports a $(K+1)$-th input counterexample. |
| `run_local_agent_task` | Delegates boilerplate generation, docstrings, or sub-tasks to an on-device model (`gemma4-26b`, `qwen3.6-35b-vllm`, `glm4-9b-ollama`) at zero API cost. |
| `list_local_models` | Lists all built-in presets, discovered local checkpoints, and active localhost OpenAI inference servers. |

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
