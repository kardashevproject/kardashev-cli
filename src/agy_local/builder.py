"""Declarative builder for Google Antigravity local agent configurations."""

from __future__ import annotations

import enum
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from google.antigravity import (
    AgentConfig,
    CompactionConfig,
    LiteRTAgentConfig,
    LiteRTBackend,
    LocalOpenAIAgentConfig,
    types,
)
from google.antigravity.hooks import policy

from agy_local.harness_hooks import (
    SUPER_INTELLIGENCE_COT_PROMPT,
    ActionableToolErrorHook,
    SelfHealingEditHook,
    SyntaxGatePostToolHook,
)
from agy_local.presets import BackendKind, get_preset


class PolicyMode(str, enum.Enum):
    """Tool execution permission policy modes."""

    ASK = "ask"
    ALLOW_ALL = "allow-all"
    SANDBOX = "sandbox"


@dataclass
class HarnessOptions:
    """Configuration options for building an Antigravity local agent harness."""

    preset: str | None = None
    backend: BackendKind | str | None = None
    model: str | None = None
    model_path: str | None = None
    base_url: str | None = None
    litert_backend: str = "gpu"
    speculative_decoding: bool = False
    cache_dir: str | None = None
    download_if_missing: bool = False
    workspaces: list[str] = field(default_factory=lambda: [os.getcwd()])
    policy_mode: PolicyMode | str = PolicyMode.ALLOW_ALL
    lightweight: bool = True
    token_threshold: int | None = None
    system_instructions: str | None = None
    enable_cot_harness: bool = True
    enable_self_healing_edits: bool = True
    enable_ast_repo_map: bool = True
    skills_paths: list[str] | None = None
    with_chronosml_mcp: bool = False
    with_opencommons_mcp: bool = False
    extra_mcp_stdio: list[tuple[str, str, list[str]]] = field(default_factory=list)
    custom_tools: list[Callable[..., Any]] = field(default_factory=list)


def _resolve_policies_and_sandbox(
    mode: PolicyMode,
) -> tuple[list[policy.Policy], bool]:
    if mode == PolicyMode.ALLOW_ALL:
        return [policy.allow_all()], False
    if mode == PolicyMode.SANDBOX:
        return [policy.allow_all()], True
    # ASK mode: default interactive confirmation for run_command
    return [policy.confirm_run_command()], False


def _build_mcp_servers(options: HarnessOptions) -> list[types.McpServerConfig]:
    servers: list[types.McpServerConfig] = []
    if options.with_chronosml_mcp:
        chronos_bin = shutil.which("chronosml-mcp") or "chronosml-mcp"
        servers.append(
            types.McpStdioServer(
                name="chronosml",
                command=chronos_bin,
                args=[],
            )
        )
    if options.with_opencommons_mcp:
        opencommons_bin = shutil.which("opencommons-mcp") or "opencommons-mcp"
        servers.append(
            types.McpStdioServer(
                name="opencommons",
                command=opencommons_bin,
                args=[],
            )
        )
    for name, command, args in options.extra_mcp_stdio:
        servers.append(
            types.McpStdioServer(
                name=name,
                command=command,
                args=args,
            )
        )
    return servers


def build_agent_config(options: HarnessOptions) -> AgentConfig:
    """Builds a ready-to-run `LiteRTAgentConfig` or `LocalOpenAIAgentConfig`."""
    resolved_backend: BackendKind | None = (
        BackendKind(options.backend) if options.backend else None
    )
    resolved_model = options.model
    resolved_model_path = options.model_path
    resolved_base_url = options.base_url

    if options.preset:
        preset_obj = get_preset(options.preset)
        if resolved_backend is None:
            resolved_backend = preset_obj.backend
        if resolved_model is None:
            resolved_model = preset_obj.model_id
        if resolved_model_path is None and preset_obj.default_model_path:
            resolved_model_path = preset_obj.resolved_model_path()
        if resolved_base_url is None and preset_obj.default_base_url:
            resolved_base_url = preset_obj.default_base_url

    if resolved_backend is None:
        if resolved_model_path or (resolved_model and resolved_model.endswith(".litertlm")):
            resolved_backend = BackendKind.LITERT
        elif resolved_base_url or resolved_model:
            resolved_backend = BackendKind.OPENAI
        else:
            # Default to gemma4-26b LiteRT preset if nothing specified
            default_preset = get_preset("gemma4-26b")
            resolved_backend = default_preset.backend
            resolved_model = default_preset.model_id
            resolved_model_path = default_preset.resolved_model_path()

    policy_mode = (
        options.policy_mode
        if isinstance(options.policy_mode, PolicyMode)
        else PolicyMode(options.policy_mode)
    )
    policies, enable_sandbox = _resolve_policies_and_sandbox(policy_mode)
    workspaces = [os.path.abspath(os.path.expanduser(w)) for w in options.workspaces]
    mcp_servers = _build_mcp_servers(options)
    compaction_cfg = (
        CompactionConfig(token_threshold=options.token_threshold)
        if options.token_threshold is not None
        else None
    )

    hooks_list: list[Any] = []
    if options.enable_self_healing_edits:
        primary_ws = workspaces[0] if workspaces else None
        hooks_list.extend(
            [
                SelfHealingEditHook(workspace=primary_ws),
                SyntaxGatePostToolHook(),
                ActionableToolErrorHook(),
            ]
        )

    combined_instructions = ""
    if options.enable_cot_harness:
        combined_instructions = SUPER_INTELLIGENCE_COT_PROMPT
    if options.system_instructions:
        combined_instructions = (
            f"{combined_instructions}\n\n{options.system_instructions}".strip()
            if combined_instructions
            else options.system_instructions
        )

    common_kwargs: dict[str, Any] = {
        "workspaces": workspaces,
        "policies": policies,
    }
    if hooks_list:
        common_kwargs["hooks"] = hooks_list
    if combined_instructions:
        common_kwargs["system_instructions"] = combined_instructions
    if options.skills_paths:
        common_kwargs["skills_paths"] = [
            os.path.abspath(os.path.expanduser(p)) for p in options.skills_paths
        ]
    if mcp_servers:
        common_kwargs["mcp_servers"] = mcp_servers
    if options.custom_tools:
        common_kwargs["tools"] = list(options.custom_tools)
    if compaction_cfg is not None:
        common_kwargs["compaction_config"] = compaction_cfg

    if resolved_backend == BackendKind.LITERT:
        target_path = resolved_model_path or resolved_model
        if not target_path:
            target_path = get_preset("gemma4-26b").resolved_model_path()
        expanded_path = os.path.abspath(os.path.expanduser(str(target_path)))

        cache_dir = options.cache_dir
        if cache_dir:
            cache_dir = os.path.abspath(os.path.expanduser(cache_dir))

        cfg = LiteRTAgentConfig(
            model_path=expanded_path,
            backend=LiteRTBackend(options.litert_backend.lower()),
            enable_speculative_decoding=options.speculative_decoding,
            cache_dir=cache_dir,
            download_if_missing=options.download_if_missing,
            **common_kwargs,
        )
        if enable_sandbox:
            caps = cfg.capabilities.model_copy(
                update={
                    "run_command_config": types.RunCommandConfig(enable_sandbox=True),
                }
            )
            cfg = cfg.model_copy(update={"capabilities": caps})
        return cfg

    # BackendKind.OPENAI
    base_url = resolved_base_url or "http://127.0.0.1:8000/v1"
    model_name = resolved_model or "qwen3-vl-8b"

    openai_cfg = LocalOpenAIAgentConfig(
        model=model_name,
        base_url=base_url,
        **common_kwargs,
    )
    if options.lightweight:
        openai_cfg = openai_cfg.lightweight()

    if enable_sandbox:
        caps = openai_cfg.capabilities.model_copy(
            update={
                "run_command_config": types.RunCommandConfig(enable_sandbox=True),
            }
        )
        openai_cfg = openai_cfg.model_copy(update={"capabilities": caps})

    return openai_cfg
