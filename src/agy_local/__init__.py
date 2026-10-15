"""kardashev-cli (agy-local): Engineered local & open-weight coding agent harness with OpenCommons CRDT research."""

__version__ = "0.2.0"

from agy_local.ast_map import (
    FileSymbolSummary,
    ScoutResult,
    SymbolNode,
    build_workspace_index,
    render_compact_repo_map,
    scout_workspace,
)
from agy_local.builder import HarnessOptions, PolicyMode, build_agent_config
from agy_local.discovery import (
    DiscoveredEndpoint,
    DiscoveredModel,
    discover_local_models,
    probe_local_endpoints,
)
from agy_local.harness_hooks import (
    SUPER_INTELLIGENCE_COT_PROMPT,
    ActionableToolErrorHook,
    HarnessTelemetry,
    SelfHealingEditHook,
    SyntaxGatePostToolHook,
    check_python_syntax,
    get_harness_telemetry,
    heal_replace_file_args,
    reset_harness_telemetry,
)
from agy_local.integrate import IntegrationResult, integrate_frontier_agent
from agy_local.mcp_server import (
    apply_smart_patch,
    handle_mcp_request,
    list_local_models,
    run_local_agent_task,
    scout_codebase,
    verify_and_repair,
)
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
    "SUPER_INTELLIGENCE_COT_PROMPT",
    "ActionableToolErrorHook",
    "AgyRichTUI",
    "BackendKind",
    "DiscoveredEndpoint",
    "DiscoveredModel",
    "FileSymbolSummary",
    "HarnessOptions",
    "HarnessTelemetry",
    "IntegrationResult",
    "ModelPreset",
    "PolicyMode",
    "ScoutResult",
    "SelfHealingEditHook",
    "ServePlan",
    "SymbolNode",
    "SyntaxGatePostToolHook",
    "UpdateStatus",
    "UserConfig",
    "__version__",
    "apply_smart_patch",
    "build_agent_config",
    "build_workspace_index",
    "check_for_updates",
    "check_python_syntax",
    "compute_serve_plan",
    "discover_local_models",
    "expand_at_file_mentions",
    "format_tool_badge",
    "generate_shell_completion",
    "get_harness_telemetry",
    "get_preset",
    "handle_mcp_request",
    "heal_replace_file_args",
    "integrate_frontier_agent",
    "list_local_models",
    "list_presets",
    "perform_self_update",
    "probe_local_endpoints",
    "render_compact_repo_map",
    "reset_harness_telemetry",
    "run_local_agent_task",
    "scout_codebase",
    "scout_workspace",
    "verify_and_repair",
]
