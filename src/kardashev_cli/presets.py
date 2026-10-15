"""Curated model and backend presets for kardashev-cli."""

from __future__ import annotations

import enum
import os
from dataclasses import dataclass


class BackendKind(str, enum.Enum):
    """Supported execution backends for the Kardashev CLI agent runtime."""

    LITERT = "litert"
    OPENAI = "openai"


@dataclass(frozen=True)
class ModelPreset:
    """Declarative preset for a local or open-weight model."""

    key: str
    family: str
    backend: BackendKind
    description: str
    model_id: str
    default_base_url: str | None = None
    default_model_path: str | None = None
    hf_repo: str | None = None
    litert_filename: str | None = None
    recommended_vram_gb: int = 16
    context_window: int = 65536

    def resolved_model_path(self) -> str | None:
        """Returns the expanded absolute path if `default_model_path` is set."""
        if not self.default_model_path:
            return None
        return os.path.abspath(os.path.expanduser(self.default_model_path))


PRESETS: dict[str, ModelPreset] = {
    # 1. LiteRT On-Device Models
    "gemma4-26b": ModelPreset(
        key="gemma4-26b",
        family="Gemma 4",
        backend=BackendKind.LITERT,
        description="Google Gemma 4 26B-A4B MoE on-device via LiteRT-LM (64k context, GPU/NPU)",
        model_id="gemma4-26b",
        default_model_path="~/.litert-lm/models/gemma4-26b/model.litertlm",
        hf_repo="litert-community/gemma-4-26B-A4B-it-litert-lm",
        litert_filename="gemma-4-26B-A4B-it-web.litertlm",
        recommended_vram_gb=24,
        context_window=65536,
    ),
    # 2. GLM Open Models (Ollama / vLLM / SGLang / Zhipu OpenAI-compatible)
    "glm4-9b-ollama": ModelPreset(
        key="glm4-9b-ollama",
        family="GLM-4",
        backend=BackendKind.OPENAI,
        description="THUDM GLM-4 9B Chat served locally via Ollama",
        model_id="glm4:9b",
        default_base_url="http://127.0.0.1:11434/v1",
        hf_repo="THUDM/glm-4-9b-chat",
        recommended_vram_gb=12,
        context_window=131072,
    ),
    "glm4-vllm": ModelPreset(
        key="glm4-vllm",
        family="GLM-4",
        backend=BackendKind.OPENAI,
        description="THUDM GLM-4 / GLM-4.5 served locally via vLLM OpenAI server",
        model_id="THUDM/glm-4-9b-chat",
        default_base_url="http://127.0.0.1:8000/v1",
        hf_repo="THUDM/glm-4-9b-chat",
        recommended_vram_gb=20,
        context_window=131072,
    ),
    "glm-zhipu-api": ModelPreset(
        key="glm-zhipu-api",
        family="GLM-4",
        backend=BackendKind.OPENAI,
        description="Zhipu AI OpenAI-compatible endpoint for GLM-4-Plus / GLM-4-Air",
        model_id="glm-4-plus",
        default_base_url="https://open.bigmodel.cn/api/paas/v4",
        recommended_vram_gb=0,
        context_window=131072,
    ),
    # 3. Open-Weight Qwen3 & GPT-OSS Presets (vLLM & Ollama)
    "qwen3.6-35b-vllm": ModelPreset(
        key="qwen3.6-35b-vllm",
        family="Qwen 3.6",
        backend=BackendKind.OPENAI,
        description="Qwen3.6-35B-A3B-FP8 MoE served via local vLLM",
        model_id="Qwen/Qwen3.6-35B-A3B-FP8",
        default_base_url="http://127.0.0.1:8000/v1",
        hf_repo="Qwen/Qwen3.6-35B-A3B-FP8",
        recommended_vram_gb=40,
        context_window=131072,
    ),
    "qwen3.8-27b-vllm": ModelPreset(
        key="qwen3.8-27b-vllm",
        family="Qwen 3.8",
        backend=BackendKind.OPENAI,
        description="Qwen3.8-27B dense model served via local vLLM",
        model_id="Qwen/Qwen3.8-27B",
        default_base_url="http://127.0.0.1:8000/v1",
        hf_repo="Qwen/Qwen3.8-27B",
        recommended_vram_gb=56,
        context_window=131072,
    ),
    "qwen3-vl-8b-vllm": ModelPreset(
        key="qwen3-vl-8b-vllm",
        family="Qwen 3 VL",
        backend=BackendKind.OPENAI,
        description="Qwen3-VL-8B-Instruct served via local vLLM (--served-model-name qwen3-vl-8b)",
        model_id="qwen3-vl-8b",
        default_base_url="http://127.0.0.1:8000/v1",
        hf_repo="Qwen/Qwen3-VL-8B-Instruct",
        recommended_vram_gb=18,
        context_window=32768,
    ),
    "gpt-oss-20b-vllm": ModelPreset(
        key="gpt-oss-20b-vllm",
        family="GPT-OSS",
        backend=BackendKind.OPENAI,
        description="OpenAI gpt-oss-20b open-weight model served via local vLLM",
        model_id="openai/gpt-oss-20b",
        default_base_url="http://127.0.0.1:8000/v1",
        hf_repo="openai/gpt-oss-20b",
        recommended_vram_gb=24,
        context_window=65536,
    ),
    "gemma4-26b-ollama": ModelPreset(
        key="gemma4-26b-ollama",
        family="Gemma 4",
        backend=BackendKind.OPENAI,
        description="Google Gemma 4 26B served via local Ollama",
        model_id="gemma4:26b",
        default_base_url="http://127.0.0.1:11434/v1",
        recommended_vram_gb=24,
        context_window=65536,
    ),
    "lm-studio": ModelPreset(
        key="lm-studio",
        family="LM Studio",
        backend=BackendKind.OPENAI,
        description="Any GGUF/MLX model loaded in LM Studio local OpenAI server",
        model_id="gemma-4-26B-A4B-it",
        default_base_url="http://127.0.0.1:1234/v1",
        recommended_vram_gb=16,
        context_window=65536,
    ),
}


def get_preset(key: str) -> ModelPreset:
    """Retrieves a preset by key or raises KeyError with helpful suggestions."""
    normalized = key.strip().lower()
    if normalized in PRESETS:
        return PRESETS[normalized]
    available = ", ".join(sorted(PRESETS.keys()))
    raise KeyError(f"Unknown preset '{key}'. Available presets: {available}")


def list_presets() -> list[ModelPreset]:
    """Returns all available presets sorted by key."""
    return [PRESETS[k] for k in sorted(PRESETS.keys())]
