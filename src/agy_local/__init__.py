"""agy-local: Open-source local & open-weight model harness for Google Antigravity."""

from agy_local.builder import HarnessOptions, PolicyMode, build_agent_config
from agy_local.discovery import (
    DiscoveredEndpoint,
    DiscoveredModel,
    discover_local_models,
    probe_local_endpoints,
)
from agy_local.mcp_server import handle_mcp_request, list_local_models, run_local_agent_task
from agy_local.presets import PRESETS, BackendKind, ModelPreset, get_preset, list_presets
from agy_local.serve import ServePlan, compute_serve_plan

__version__ = "0.1.0"

__all__ = [
    "PRESETS",
    "BackendKind",
    "DiscoveredEndpoint",
    "DiscoveredModel",
    "HarnessOptions",
    "ModelPreset",
    "PolicyMode",
    "ServePlan",
    "__version__",
    "build_agent_config",
    "compute_serve_plan",
    "discover_local_models",
    "get_preset",
    "handle_mcp_request",
    "list_local_models",
    "list_presets",
    "probe_local_endpoints",
    "run_local_agent_task",
]
