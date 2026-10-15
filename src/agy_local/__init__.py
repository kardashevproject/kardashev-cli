"""agy-local: Open-source local & open-weight model harness for Google Antigravity."""

__version__ = "0.1.0"

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
from agy_local.tui import AgyRichTUI, expand_at_file_mentions, format_tool_badge
from agy_local.updater import (
    UpdateStatus,
    UserConfig,
    check_for_updates,
    generate_shell_completion,
    perform_self_update,
)

__all__ = [
    "PRESETS",
    "AgyRichTUI",
    "BackendKind",
    "DiscoveredEndpoint",
    "DiscoveredModel",
    "HarnessOptions",
    "ModelPreset",
    "PolicyMode",
    "ServePlan",
    "UpdateStatus",
    "UserConfig",
    "__version__",
    "build_agent_config",
    "check_for_updates",
    "compute_serve_plan",
    "discover_local_models",
    "expand_at_file_mentions",
    "format_tool_badge",
    "generate_shell_completion",
    "get_preset",
    "handle_mcp_request",
    "list_local_models",
    "list_presets",
    "perform_self_update",
    "probe_local_endpoints",
    "run_local_agent_task",
]
