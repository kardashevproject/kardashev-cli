"""agy-local: Open-source local & open-weight model harness for Google Antigravity."""

from agy_local.builder import HarnessOptions, PolicyMode, build_agent_config
from agy_local.discovery import (
    DiscoveredEndpoint,
    DiscoveredModel,
    discover_local_models,
    probe_local_endpoints,
)
from agy_local.presets import PRESETS, BackendKind, ModelPreset, get_preset, list_presets

__version__ = "0.1.0"

__all__ = [
    "PRESETS",
    "BackendKind",
    "DiscoveredEndpoint",
    "DiscoveredModel",
    "HarnessOptions",
    "ModelPreset",
    "PolicyMode",
    "__version__",
    "build_agent_config",
    "discover_local_models",
    "get_preset",
    "list_presets",
    "probe_local_endpoints",
]
