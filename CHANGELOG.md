# Changelog

All notable changes to `agy-local` will be documented in this file.

## [0.1.0] - 2026-10-15

### Added
- **Interactive & Autonomous Harness CLI (`agy-local`)**:
  - `agy-local chat`: Launches `google.antigravity.utils.interactive.run_interactive_loop` backed by any local or open-weight model with full Antigravity tool access (`view_file`, `replace_file_content`, `write_to_file`, `run_command`, `grep`, `list_dir`).
  - `agy-local exec`: Executes single-shot autonomous coding tasks with live streaming of tokens, reasoning (`--show-thoughts`), and tool calls (`--show-tools`).
  - `agy-local presets`: Curated presets for `gemma4-26b` (LiteRT), `glm4-9b-ollama`, `glm4-vllm`, `glm-zhipu-api`, `qwen3.6-35b-vllm`, `qwen3.8-27b-vllm`, `qwen3-vl-8b-vllm`, `gpt-oss-20b-vllm`, and `lm-studio`.
  - `agy-local models` & `agy-local doctor`: Hardware, checkpoint, and local OpenAI server auto-discovery across `~/.litert-lm/models`, `~/.cache/huggingface/hub`, `~/models`, and localhost inference ports.
  - `agy-local import-litert`: One-command LiteRT checkpoint downloader and importer.
  - Native `--with-chronosml` flag to attach the `chronosml-mcp` server for zero-guesswork ML runtime and VRAM estimation.
