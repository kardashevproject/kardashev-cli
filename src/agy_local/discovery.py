"""Auto-discovery for local model checkpoints and OpenAI-compatible inference servers."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from agy_local.presets import BackendKind

DEFAULT_PROBE_ENDPOINTS: tuple[tuple[str, str], ...] = (
    ("vLLM / SGLang (8000)", "http://127.0.0.1:8000/v1"),
    ("Ollama (11434)", "http://127.0.0.1:11434/v1"),
    ("LM Studio (1234)", "http://127.0.0.1:1234/v1"),
    ("SGLang (30000)", "http://127.0.0.1:30000/v1"),
)


@dataclass(frozen=True)
class DiscoveredModel:
    """Represents a local model checkpoint found on disk."""

    name: str
    backend: BackendKind
    path: str
    size_bytes: int
    source: str

    @property
    def size_gb(self) -> float:
        return round(self.size_bytes / (1024**3), 2)


@dataclass(frozen=True)
class DiscoveredEndpoint:
    """Represents a live OpenAI-compatible inference server discovered on localhost."""

    label: str
    base_url: str
    reachable: bool
    models: list[str] = field(default_factory=list)
    error: str | None = None


def _dir_size_bytes(path: Path) -> int:
    total = 0
    try:
        if path.is_file():
            return path.stat().st_size
        for entry in path.rglob("*"):
            if entry.is_file() and not entry.is_symlink():
                try:
                    total += entry.stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return total


def discover_local_models(
    litert_dir: str | Path | None = None,
    hf_cache_dir: str | Path | None = None,
    extra_model_dirs: list[str | Path] | None = None,
) -> list[DiscoveredModel]:
    """Scans local LiteRT, HuggingFace Hub, and custom model directories."""
    discovered: list[DiscoveredModel] = []

    # 1. Scan LiteRT-LM models (~/.litert-lm/models)
    lt_root = Path(os.path.expanduser(str(litert_dir or "~/.litert-lm/models")))
    if lt_root.is_dir():
        for lt_file in sorted(lt_root.rglob("*.litertlm")):
            try:
                size = lt_file.stat().st_size
            except OSError:
                size = 0
            model_name = lt_file.parent.name if lt_file.name == "model.litertlm" else lt_file.stem
            discovered.append(
                DiscoveredModel(
                    name=model_name,
                    backend=BackendKind.LITERT,
                    path=str(lt_file.resolve()),
                    size_bytes=size,
                    source="litert-lm",
                )
            )

    # 2. Scan HuggingFace cache (~/.cache/huggingface/hub)
    hf_root = Path(os.path.expanduser(str(hf_cache_dir or "~/.cache/huggingface/hub")))
    if hf_root.is_dir():
        for entry in sorted(hf_root.iterdir()):
            if entry.is_dir() and entry.name.startswith("models--"):
                parts = entry.name.split("--")
                repo_id = "/".join(parts[1:]) if len(parts) >= 3 else entry.name
                size = _dir_size_bytes(entry)
                if size > 0:
                    discovered.append(
                        DiscoveredModel(
                            name=repo_id,
                            backend=BackendKind.OPENAI,
                            path=str(entry.resolve()),
                            size_bytes=size,
                            source="huggingface-cache",
                        )
                    )

    # 3. Scan ~/models or extra directories
    custom_dirs = extra_model_dirs if extra_model_dirs is not None else ["~/models"]
    for raw_dir in custom_dirs:
        m_root = Path(os.path.expanduser(str(raw_dir)))
        if not m_root.is_dir():
            continue
        for child in sorted(m_root.iterdir()):
            if not child.is_dir() or child.name in ("hf", "logs", ".locks"):
                continue
            if (child / "config.json").is_file() or any(child.glob("*.safetensors")):
                discovered.append(
                    DiscoveredModel(
                        name=child.name,
                        backend=BackendKind.OPENAI,
                        path=str(child.resolve()),
                        size_bytes=_dir_size_bytes(child),
                        source="local-dir",
                    )
                )

    return discovered


def probe_endpoint(
    base_url: str,
    label: str = "Custom Endpoint",
    timeout_sec: float = 1.5,
) -> DiscoveredEndpoint:
    """Probes an OpenAI-compatible `/v1/models` endpoint to check liveness and model IDs."""
    normalized = base_url.rstrip("/")
    models_url = f"{normalized}/models" if normalized.endswith("/v1") else f"{normalized}/v1/models"
    req = urllib.request.Request(models_url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            data = payload.get("data", []) if isinstance(payload, dict) else []
            model_ids = [
                str(item.get("id"))
                for item in data
                if isinstance(item, dict) and item.get("id")
            ]
            return DiscoveredEndpoint(
                label=label,
                base_url=normalized,
                reachable=True,
                models=model_ids,
            )
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
        return DiscoveredEndpoint(
            label=label,
            base_url=normalized,
            reachable=False,
            models=[],
            error=str(exc),
        )


def probe_local_endpoints(timeout_sec: float = 1.2) -> list[DiscoveredEndpoint]:
    """Probes standard local OpenAI-compatible servers (vLLM, Ollama, LM Studio, SGLang)."""
    return [
        probe_endpoint(base_url=url, label=label, timeout_sec=timeout_sec)
        for label, url in DEFAULT_PROBE_ENDPOINTS
    ]
