"""Local vLLM & Ollama launcher (`kardashev serve`) with ChronosML Roofline & VRAM auto-tuning."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from typing import Any

from kardashev_cli.presets import BackendKind, get_preset


@dataclass(frozen=True)
class ServePlan:
    """Resolved launch plan for a local vLLM or Ollama inference server."""

    engine: str
    model_target: str
    served_model_name: str
    host: str
    port: int
    max_model_len: int
    gpu_memory_utilization: float
    estimated_model_gb: float
    available_mem_gb: float
    oom_risk: bool
    command: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _probe_chronosml_memory(
    model_target: str,
    seq_len: int,
    fallback_vram_gb: float,
) -> tuple[float, float, bool]:
    """Queries `chronosml` (in-process or via CLI) for peak model memory & available hardware memory."""
    try:
        from chronosml.estimator import estimate_task  # type: ignore[import-not-found]

        report = estimate_task(
            model_path_or_config=model_target,
            task_type="llm_generate",
            num_samples=1,
            batch_size=1,
            seq_len=min(seq_len, 4096),
            max_new_tokens=128,
            run_micro_pilot=False,
        )
        return (
            round(float(report.peak_mem_gb), 2),
            round(float(report.available_mem_gb), 2),
            bool(report.oom_risk),
        )
    except (ImportError, RuntimeError, ValueError, TypeError, OSError):
        pass

    chronos_bin = shutil.which("chronosml")
    if chronos_bin:
        try:
            proc = subprocess.run(
                [
                    chronos_bin,
                    "estimate",
                    "--task-type",
                    "llm_generate",
                    "--model",
                    model_target,
                    "--seq-len",
                    str(min(seq_len, 4096)),
                    "--json",
                ],
                capture_output=True,
                text=True,
                timeout=8,
                check=False,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                data = json.loads(proc.stdout)
                return (
                    round(float(data.get("peak_mem_gb", fallback_vram_gb)), 2),
                    round(float(data.get("available_mem_gb", 64.0)), 2),
                    bool(data.get("oom_risk", False)),
                )
        except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError):
            pass

    # Fallback: read Linux /proc/meminfo MemAvailable if available
    avail_gb = 64.0
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    avail_gb = round(kb / (1024**2), 2)
                    break
    except OSError:
        pass

    oom = fallback_vram_gb > (avail_gb * 0.88)
    return round(fallback_vram_gb, 2), avail_gb, oom


def _find_vllm_binary() -> str:
    """Finds the `vllm` executable in PATH or common local virtual environments."""
    in_path = shutil.which("vllm")
    if in_path:
        return in_path
    home = os.path.expanduser("~")
    candidates = [
        os.path.join(home, ".local", "bin", "vllm"),
        os.path.join(home, "venv-vllm", "bin", "vllm"),
        os.path.join(home, ".venv-vllm", "bin", "vllm"),
        os.path.join(home, "llm-lab", "autolab", "data", "venv-vllm", "bin", "vllm"),
    ]
    for cand in candidates:
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return "vllm"


def _resolve_local_model_target(model_or_name: str) -> tuple[str, float | None]:
    """Resolves a short model name (e.g. `qwen3.6-35b-a3b-fp8`) to its on-disk directory path and size in GB if present."""
    expanded = os.path.expanduser(model_or_name)
    if os.path.exists(expanded):
        from pathlib import Path

        from kardashev_cli.discovery import _dir_size_and_has_weights

        sz_bytes, _ = _dir_size_and_has_weights(Path(expanded))
        sz_gb = round(sz_bytes / (1024**3), 2) if sz_bytes > 0 else None
        return (os.path.abspath(expanded), sz_gb)

    from kardashev_cli.discovery import discover_local_models

    for dm in discover_local_models():
        if dm.name.lower() == model_or_name.lower() or dm.name.lower().endswith(f"/{model_or_name.lower()}"):
            return (dm.path, dm.size_gb if dm.size_gb > 0 else None)

    return (model_or_name, None)


def compute_serve_plan(
    *,
    preset: str | None = None,
    model: str | None = None,
    engine: str | None = None,
    host: str = "127.0.0.1",
    port: int | None = None,
    max_model_len: int | None = None,
    gpu_memory_utilization: float | None = None,
    tensor_parallel_size: int = 1,
    enforce_eager: bool = False,
) -> ServePlan:
    """Computes an optimal vLLM or Ollama launch plan using ChronosML memory telemetry."""
    resolved_model = model
    resolved_engine = engine
    fallback_vram = 16.0
    default_ctx = 32768
    served_name: str | None = None

    # If the user passed a preset name via --model, resolve the preset automatically
    if not preset and model and model.lower() in ("gemma4-26b", "glm4-9b-ollama", "glm4-vllm", "qwen3.6-35b-vllm", "qwen3.8-27b-vllm", "qwen3-vl-8b-vllm", "gpt-oss-20b-vllm"):
        preset = model.lower()
        resolved_model = None

    if preset:
        p = get_preset(preset)
        if p.backend == BackendKind.LITERT:
            raise ValueError(
                f"Preset '{preset}' uses the on-device LiteRT backend, which does not require "
                "an external server. Run `kardashev chat --preset gemma4-26b` directly."
            )
        if not resolved_model:
            resolved_model = p.resolved_model_path() or p.hf_repo or p.model_id
        served_name = p.model_id
        fallback_vram = float(p.recommended_vram_gb or 16.0)
        default_ctx = min(p.context_window, 32768)
        if not resolved_engine:
            if p.default_base_url and ":11434" in p.default_base_url:
                resolved_engine = "ollama"
            else:
                resolved_engine = "vllm"

    if not resolved_model:
        resolved_model = "THUDM/glm-4-9b-chat"
    if not served_name:
        served_name = os.path.basename(resolved_model.rstrip("/")) or resolved_model

    # Resolve short local model names via the local model scanner
    resolved_model, disk_size_gb = _resolve_local_model_target(resolved_model)
    if disk_size_gb is not None and disk_size_gb > 0:
        fallback_vram = max(fallback_vram, round(disk_size_gb * 1.10, 2))

    if not resolved_engine:
        resolved_engine = "ollama" if ":" in resolved_model and "/" not in resolved_model else "vllm"

    target_seq_len = max_model_len or default_ctx
    est_gb, avail_gb, oom_risk = _probe_chronosml_memory(
        model_target=resolved_model,
        seq_len=target_seq_len,
        fallback_vram_gb=fallback_vram,
    )
    if disk_size_gb is not None and est_gb < disk_size_gb:
        est_gb = disk_size_gb

    # Auto-compute safe gpu_memory_utilization and max_model_len if not explicitly overridden
    if gpu_memory_utilization is None:
        if avail_gb > 0:
            # Reserve 25% KV-cache headroom over model weight footprint, clamped to [0.25, 0.90]
            needed_ratio = (est_gb * 1.25) / avail_gb
            gpu_util = round(min(0.90, max(0.25, needed_ratio)), 2)
        else:
            gpu_util = 0.80
    else:
        gpu_util = round(float(gpu_memory_utilization), 2)

    if max_model_len is None:
        headroom_gb = max(0.0, (avail_gb * gpu_util) - est_gb)
        if headroom_gb < 4.0:
            resolved_max_len = 4096
        elif headroom_gb < 10.0:
            resolved_max_len = 8192
        elif headroom_gb < 20.0:
            resolved_max_len = 16384
        else:
            resolved_max_len = min(default_ctx, 32768)
    else:
        resolved_max_len = int(max_model_len)

    if resolved_engine == "ollama":
        resolved_port = port or 11434
        cmd = ["ollama", "run", resolved_model]
    else:
        resolved_port = port or 8000
        vllm_bin = _find_vllm_binary()
        cmd = [
            vllm_bin,
            "serve",
            resolved_model,
            "--served-model-name",
            served_name,
            "--host",
            host,
            "--port",
            str(resolved_port),
            "--max-model-len",
            str(resolved_max_len),
            "--gpu-memory-utilization",
            str(gpu_util),
        ]
        if tensor_parallel_size > 1:
            cmd.extend(["--tensor-parallel-size", str(tensor_parallel_size)])
        if enforce_eager:
            cmd.append("--enforce-eager")

    return ServePlan(
        engine=resolved_engine,
        model_target=resolved_model,
        served_model_name=served_name,
        host=host,
        port=resolved_port,
        max_model_len=resolved_max_len,
        gpu_memory_utilization=gpu_util,
        estimated_model_gb=est_gb,
        available_mem_gb=avail_gb,
        oom_risk=oom_risk,
        command=cmd,
    )
