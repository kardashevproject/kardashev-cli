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

### 3. Launch the Interactive Antigravity Harness (`run_interactive_loop`)

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

## CLI Reference

| Subcommand | Description |
|---|---|
| `agy-local chat` | Launch the interactive terminal REPL (`run_interactive_loop`) with full Antigravity coding tools. |
| `agy-local exec "<prompt>"` | Execute a single autonomous coding task and stream tokens, thoughts (`--show-thoughts`), and tool calls (`--show-tools`). |
| `agy-local presets` | List all built-in LiteRT, GLM-4, Qwen3, and GPT-OSS presets (`--json` supported). |
| `agy-local models` | Scan local disk (`~/.litert-lm`, `~/.cache/huggingface`, `~/models`) and probe local inference ports (`8000`, `11434`, `1234`, `30000`). |
| `agy-local doctor` | Verify Python, `litert-lm`, CUDA GPU availability, `chronosml-mcp`, and active local model endpoints. |
| `agy-local import-litert` | Download and register a LiteRT checkpoint from HuggingFace. |

---

## Python SDK Usage

```python
import asyncio
from google.antigravity.utils.interactive import run_interactive_loop
from agy_local import HarnessOptions, PolicyMode, build_agent_config

async def main():
    config = build_agent_config(
        HarnessOptions(
            preset="gemma4-26b",          # Or "glm4-9b-ollama", "qwen3.6-35b-vllm"
            workspaces=["."],
            policy_mode=PolicyMode.ALLOW_ALL,
            speculative_decoding=True,
            with_chronosml_mcp=True,
        )
    )
    await run_interactive_loop(config)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## License

Apache License 2.0 — Copyright (c) 2026 The Kardashev Project.
