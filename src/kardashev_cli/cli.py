"""CLI entrypoint for Kardashev CLI (`kardashev` / `ksh`)."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence

from google.antigravity import Agent
from google.antigravity.utils.interactive import run_interactive_loop

from kardashev_cli import __version__
from kardashev_cli.ast_map import scout_workspace
from kardashev_cli.builder import HarnessOptions, PolicyMode, build_agent_config
from kardashev_cli.discovery import discover_local_models, probe_local_endpoints
from kardashev_cli.harness_hooks import get_harness_telemetry
from kardashev_cli.integrate import integrate_frontier_agent
from kardashev_cli.mcp_server import serve_stdio
from kardashev_cli.opencommons import load_pending_proposals, load_persistent_store
from kardashev_cli.presets import BackendKind, get_preset, list_presets
from kardashev_cli.serve import compute_serve_plan
from kardashev_cli.tui import KardashevRichTUI
from kardashev_cli.updater import (
    UserConfig,
    check_for_updates,
    ensure_bin_symlinks,
    generate_shell_completion,
    perform_self_update,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kardashev",
        description=(
            "Kardashev Studio: Autonomous scientific & engineering harness for local LiteRT models "
            "(Gemma 4 26B), OpenAI-compatible servers (vLLM, Ollama, SGLang, GLM-4, Qwen), and OpenCommons."
        ),
    )
    parser.add_argument(
        "-V",
        "--version",
        action="store_true",
        help="Show kardashev-cli version and installation mode.",
    )
    parser.add_argument(
        "--upgrade",
        action="store_true",
        help="Self-update kardashev-cli from GitHub main and refresh CLI symlinks.",
    )
    sub = parser.add_subparsers(dest="command")

    # Common harness arguments helper
    def _add_harness_args(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "-p",
            "--preset",
            help="Preset key (e.g., gemma4-26b, glm4-9b-ollama, glm4-vllm, qwen3.6-35b-vllm, qwen3-vl-8b-vllm).",
        )
        p.add_argument(
            "-b",
            "--backend",
            choices=["litert", "openai"],
            help="Override backend ('litert' or 'openai').",
        )
        p.add_argument(
            "-m",
            "--model",
            help="Model identifier (for OpenAI servers) or path to .litertlm file.",
        )
        p.add_argument(
            "--model-path",
            help="Explicit path to a .litertlm checkpoint file.",
        )
        p.add_argument(
            "--base-url",
            help="OpenAI-compatible server base URL (e.g., http://127.0.0.1:8000/v1 or http://127.0.0.1:11434/v1).",
        )
        p.add_argument(
            "-w",
            "--workspace",
            action="append",
            dest="workspaces",
            help="Workspace directory to mount (defaults to current working directory). Can be passed multiple times.",
        )
        p.add_argument(
            "--policy",
            choices=["allow-all", "ask", "sandbox"],
            default="allow-all",
            help="Tool execution safety policy mode (default: allow-all).",
        )
        p.add_argument(
            "--litert-device",
            choices=["gpu", "npu", "cpu"],
            default="gpu",
            help="Hardware backend for LiteRT inference (default: gpu).",
        )
        p.add_argument(
            "--speculative",
            action="store_true",
            help="Enable speculative decoding (multi-token prediction) on LiteRT.",
        )
        p.add_argument(
            "--download-if-missing",
            action="store_true",
            help="Automatically download LiteRT model weights if missing.",
        )
        p.add_argument(
            "--no-lightweight",
            action="store_true",
            help="Disable .lightweight() preset on OpenAI-compatible models.",
        )
        p.add_argument(
            "--token-threshold",
            type=int,
            help="Override context compaction token threshold.",
        )
        p.add_argument(
            "--system-instructions",
            help="Custom system instructions for the agent.",
        )
        p.add_argument(
            "--with-chronosml",
            action="store_true",
            help="Attach the ChronosML MCP server (chronosml-mcp) for zero-guesswork ML runtime/VRAM estimation.",
        )
        p.add_argument(
            "--with-opencommons",
            action="store_true",
            help="Attach the OpenCommons MCP server (opencommons-mcp) for shared verified concept retrieval and recording.",
        )
        p.add_argument(
            "--dry-run",
            action="store_true",
            help="Print the resolved AgentConfig summary and exit without launching.",
        )

    # 1. chat (interactive Rich TUI or plain REPL)
    chat_p = sub.add_parser(
        "chat",
        help="Launch the interactive Kardashev Studio TUI in your terminal.",
    )
    chat_p.add_argument(
        "--plain",
        action="store_true",
        help="Use the minimal stdin/stdout run_interactive_loop instead of the Rich TUI.",
    )
    chat_p.add_argument(
        "--ralph",
        help="Autonomous verification shell command executed before stopping each turn (e.g. 'pytest -q').",
    )
    chat_p.add_argument(
        "--ralph-max-iter",
        type=int,
        default=3,
        help="Maximum autonomous self-repair iterations per turn when --ralph is enabled (default: 3).",
    )
    _add_harness_args(chat_p)

    # 2. exec (single-shot autonomous prompt)
    exec_p = sub.add_parser(
        "exec",
        help="Execute a single autonomous task prompt with the full Kardashev Studio tool harness.",
    )
    exec_p.add_argument("prompt", help="Task prompt to execute.")
    exec_p.add_argument(
        "--show-thoughts",
        action="store_true",
        help="Stream model reasoning/thinking tokens to stderr.",
    )
    exec_p.add_argument(
        "--show-tools",
        action="store_true",
        help="Print tool calls as they execute.",
    )
    _add_harness_args(exec_p)

    # 3. presets
    presets_p = sub.add_parser("presets", help="List curated local & open model presets.")
    presets_p.add_argument("--json", action="store_true", help="Output as JSON.")

    # 4. models (local disk & server discovery)
    models_p = sub.add_parser(
        "models",
        help="Discover local LiteRT/HuggingFace checkpoints and active local OpenAI servers.",
    )
    models_p.add_argument("--json", action="store_true", help="Output as JSON.")

    # 5. doctor
    doctor_p = sub.add_parser(
        "doctor",
        help="Check local GPU, LiteRT-LM runtime, cached models, and active inference endpoints.",
    )
    doctor_p.add_argument("--json", action="store_true", help="Output as JSON.")

    # 6. import-litert
    imp_p = sub.add_parser(
        "import-litert",
        help="Import a LiteRT-LM checkpoint (defaults to gemma4-26b from HuggingFace).",
    )
    imp_p.add_argument(
        "--preset",
        default="gemma4-26b",
        help="LiteRT preset to import (default: gemma4-26b).",
    )

    # 7. serve (ChronosML-tuned local vLLM / Ollama launcher)
    serve_p = sub.add_parser(
        "serve",
        help="Launch a local vLLM or Ollama server with ChronosML auto-tuned VRAM & context limits.",
    )
    serve_p.add_argument(
        "-p",
        "--preset",
        help="OpenAI-compatible preset to serve (e.g. qwen3.6-35b-vllm, glm4-vllm, gpt-oss-20b-vllm).",
    )
    serve_p.add_argument(
        "-m",
        "--model",
        help="Explicit HuggingFace repo, local model directory, or Ollama tag.",
    )
    serve_p.add_argument(
        "--engine",
        choices=["vllm", "ollama"],
        help="Inference server engine ('vllm' or 'ollama').",
    )
    serve_p.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1).")
    serve_p.add_argument("--port", type=int, help="Bind port (default: 8000 for vLLM, 11434 for Ollama).")
    serve_p.add_argument("--max-model-len", type=int, help="Override max context length.")
    serve_p.add_argument(
        "--gpu-memory-utilization",
        type=float,
        help="Override GPU memory utilization fraction (0.1 to 0.98).",
    )
    serve_p.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel GPU count (default: 1).",
    )
    serve_p.add_argument("--enforce-eager", action="store_true", help="Pass --enforce-eager to vLLM.")
    serve_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the ChronosML-computed ServePlan JSON and exit without launching.",
    )

    # 8. mcp (JSON-RPC 2.0 stdio MCP server)
    sub.add_parser(
        "mcp",
        help="Run the kardashev-mcp JSON-RPC 2.0 stdio MCP server for zero-cost local agent delegation.",
    )

    # 9. update / upgrade (self-updater)
    upd_p = sub.add_parser(
        "update",
        aliases=["upgrade"],
        help="Check for updates or self-update kardashev-cli from GitHub main.",
    )
    upd_p.add_argument(
        "--check",
        action="store_true",
        help="Only check if a newer version/commit is available without installing.",
    )
    upd_p.add_argument(
        "--json",
        action="store_true",
        help="Output update check status as JSON.",
    )

    # 10. config (manage ~/.config/kardashev/config.json defaults)
    cfg_p = sub.add_parser(
        "config",
        help="View or modify persistent user defaults in ~/.config/kardashev/config.json.",
    )
    cfg_p.add_argument(
        "action",
        nargs="?",
        choices=["show", "set", "reset", "path"],
        default="show",
        help="Action to perform (default: show).",
    )
    cfg_p.add_argument("key", nargs="?", help="Config key to set (e.g., default_preset, policy_mode).")
    cfg_p.add_argument("value", nargs="?", help="Value to set.")

    # 11. completions (generate bash/zsh/fish tab completions)
    comp_p = sub.add_parser(
        "completions",
        help="Generate shell tab-completion script for bash, zsh, or fish.",
    )
    comp_p.add_argument(
        "shell",
        nargs="?",
        choices=["bash", "zsh", "fish"],
        default="bash",
        help="Target shell (default: bash).",
    )

    # 12. integrate (install Kardashev Co-Processor MCP & skill into frontier coding agents)
    int_p = sub.add_parser(
        "integrate",
        help="Auto-configure Claude Code, Cursor, Codex, or Antigravity with the Kardashev Co-Processor MCP & skill.",
    )
    int_p.add_argument(
        "target",
        nargs="?",
        choices=["antigravity", "claude", "cursor", "codex", "all"],
        default="all",
        help="Target frontier coding agent (default: all).",
    )
    int_p.add_argument("--json", action="store_true", help="Output as JSON.")

    # 13. scout (fast AST Repo-Map & surgical code slice extractor)
    scout_p = sub.add_parser(
        "scout",
        help="Run the AST Repo-Map & surgical code slice extractor on a workspace query.",
    )
    scout_p.add_argument("query", nargs="*", help="Search query or symbol names.")
    scout_p.add_argument("-w", "--workspace", default=os.getcwd(), help="Workspace directory.")
    scout_p.add_argument("--max-slices", type=int, default=5, help="Max code slices to extract.")
    scout_p.add_argument("--json", action="store_true", help="Output full ScoutResult as JSON.")

    # 14. commons (OpenCommons Research Store, Networked CRDT Gossip Hub & Pending Review Queue CLI)
    com_p = sub.add_parser(
        "commons",
        help="Manage the OpenCommons Concept Commons store, sync over HTTP CRDT Gossip, or review agent proposals.",
    )
    com_p.add_argument(
        "action",
        nargs="?",
        choices=[
            "status",
            "find",
            "record",
            "review",
            "approve",
            "reject",
            "seed",
            "sync",
            "pull",
            "push",
            "serve-hub",
            "mcp",
            "settings",
            "dashboard",
        ],
        default="status",
        help="Action to perform (default: status).",
    )
    com_p.add_argument(
        "target",
        nargs="*",
        help="Search query, file path, proposal ID, hub URL, replica JSON path, or 'show'/'set KEY VALUE' for settings.",
    )
    com_p.add_argument("--no-browser", action="store_true", help="Don't open a browser tab for 'dashboard'.")
    com_p.add_argument("--domain", default="general", help="Domain when recording a Python file.")
    com_p.add_argument("--summary", default="", help="Summary when recording a Python file.")
    com_p.add_argument("--modality", default="dual_card_with_pitfalls", help="Rendering modality for find.")
    com_p.add_argument("--top-k", type=int, default=3, help="Max hits for find.")
    com_p.add_argument("--store", default=None, help="Override path to store.json.")
    com_p.add_argument("--host", default="127.0.0.1", help="Bind host for serve-hub (default: 127.0.0.1).")
    com_p.add_argument("--port", type=int, default=8765, help="Bind port for serve-hub (default: 8765).")
    com_p.add_argument("--force", action="store_true", help="Force full pull ignoring cached ETag.")
    com_p.add_argument("--json", action="store_true", help="Output as JSON.")

    return parser


def _options_from_args(args: argparse.Namespace) -> HarnessOptions:
    user_cfg = UserConfig.load()
    workspaces = args.workspaces if getattr(args, "workspaces", None) else [os.getcwd()]
    preset = getattr(args, "preset", None)
    backend = getattr(args, "backend", None)
    model = getattr(args, "model", None)
    model_path = getattr(args, "model_path", None)
    if not any([preset, backend, model, model_path]):
        preset = user_cfg.default_preset
    policy_arg = getattr(args, "policy", None) or user_cfg.policy_mode
    speculative = bool(getattr(args, "speculative", False)) or user_cfg.speculative_decoding
    with_chronos = bool(getattr(args, "with_chronosml", False)) or user_cfg.with_chronosml_mcp
    with_opencommons = bool(getattr(args, "with_opencommons", False)) or user_cfg.with_opencommons_mcp

    return HarnessOptions(
        preset=preset,
        backend=backend,
        model=model,
        model_path=model_path,
        base_url=getattr(args, "base_url", None),
        litert_backend=getattr(args, "litert_device", "gpu"),
        speculative_decoding=speculative,
        download_if_missing=bool(getattr(args, "download_if_missing", False)),
        workspaces=workspaces,
        policy_mode=PolicyMode(policy_arg),
        lightweight=not bool(getattr(args, "no_lightweight", False)),
        token_threshold=getattr(args, "token_threshold", None),
        system_instructions=getattr(args, "system_instructions", None),
        with_chronosml_mcp=with_chronos,
        with_opencommons_mcp=with_opencommons,
    )


def _describe_config(cfg: object) -> dict[str, object]:
    cls_name = type(cfg).__name__
    info: dict[str, object] = {
        "config_class": cls_name,
        "workspaces": getattr(cfg, "workspaces", []),
    }
    if hasattr(cfg, "model_path"):
        info["backend"] = "litert"
        info["model_path"] = getattr(cfg, "model_path", None)
        info["litert_device"] = str(getattr(cfg, "backend", "gpu"))
        info["speculative_decoding"] = getattr(cfg, "enable_speculative_decoding", False)
    else:
        info["backend"] = "openai"
        info["model"] = getattr(cfg, "model", None)
        info["base_url"] = getattr(cfg, "base_url", None)
    compaction = getattr(cfg, "compaction_config", None)
    if compaction:
        info["compaction_token_threshold"] = getattr(compaction, "token_threshold", None)
    return info


async def _cmd_chat(args: argparse.Namespace) -> int:
    opts = _options_from_args(args)
    cfg = build_agent_config(opts)
    summary = _describe_config(cfg)

    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return 0

    if getattr(args, "plain", False):
        await run_interactive_loop(cfg)
        return 0

    tui = KardashevRichTUI(options=opts)
    if getattr(args, "ralph", None):
        tui.ralph_verify_cmd = args.ralph
    tui.ralph_max_iterations = getattr(args, "ralph_max_iter", 3)
    return await tui.run()


async def _cmd_exec(args: argparse.Namespace) -> int:
    opts = _options_from_args(args)
    cfg = build_agent_config(opts)
    summary = _describe_config(cfg)

    if args.dry_run:
        summary["prompt"] = args.prompt
        print(json.dumps(summary, indent=2))
        return 0

    effective_prompt = args.prompt
    if opts.enable_ast_repo_map and opts.workspaces:
        from kardashev_cli.ast_map import build_turn_zero_context

        ctx_block, _n_files, _n_syms, _n_slices = build_turn_zero_context(
            opts.workspaces[0], args.prompt
        )
        if ctx_block:
            get_harness_telemetry().ast_maps_injected += 1
            effective_prompt = (
                f"{ctx_block}\n\n"
                f"[User Task]:\n{args.prompt}"
            )

    if opts.enable_opencommons:
        from kardashev_cli.opencommons import build_turn_zero_commons_context

        oc_block, oc_hits = build_turn_zero_commons_context(args.prompt)
        if oc_block and oc_hits > 0:
            get_harness_telemetry().commons_cards_injected += oc_hits
            effective_prompt = f"{oc_block}\n\n{effective_prompt}"

    from kardashev_cli.tui import suppress_noisy_cpp_stderr

    try:
        with suppress_noisy_cpp_stderr():
            async with Agent(config=cfg) as agent:
                response = await agent.chat(effective_prompt)

                tasks: list[asyncio.Task[None]] = []
                if args.show_thoughts:
                    async def _stream_thoughts() -> None:
                        async for thought in response.thoughts:
                            sys.stderr.write(f"[Thinking] {thought}")
                            sys.stderr.flush()

                    tasks.append(asyncio.create_task(_stream_thoughts()))

                if args.show_tools:
                    async def _stream_tools() -> None:
                        async for call in response.tool_calls:
                            sys.stderr.write(f"\n[ToolCall] {call.name}({call.args})\n")
                            sys.stderr.flush()

                    tasks.append(asyncio.create_task(_stream_tools()))

                async for token in response:
                    sys.stdout.write(token)
                    sys.stdout.flush()
                print()

                for t in tasks:
                    await asyncio.gather(t, return_exceptions=True)
    except (KeyboardInterrupt, asyncio.CancelledError):
        sys.stderr.write("\n[Interrupted]\n")
        return 130
    except Exception as exc:  # noqa: BLE001
        err_msg = str(exc)
        if "connection refused" in err_msg.lower() or "failed to send http request" in err_msg.lower():
            base_url = getattr(cfg, "base_url", "http://127.0.0.1:8000/v1")
            model_name = getattr(cfg, "model", "local-model")
            sys.stderr.write(
                f"\n✗ Inference endpoint unreachable at {base_url} for model '{model_name}'.\n"
                f"  • Start the local server first: kardashev serve --model {model_name}\n"
                f"  • Or use in-process LiteRT GPU/NPU: kardashev exec -p gemma4-26b \"...\"\n"
            )
            return 1
        sys.stderr.write(f"\n✗ Agent execution error: {err_msg}\n")
        return 1

    return 0


def _cmd_presets(args: argparse.Namespace) -> int:
    presets = list_presets()
    if args.json:
        out = [
            {
                "key": p.key,
                "family": p.family,
                "backend": p.backend.value,
                "model_id": p.model_id,
                "default_base_url": p.default_base_url,
                "default_model_path": p.resolved_model_path(),
                "recommended_vram_gb": p.recommended_vram_gb,
                "context_window": p.context_window,
                "description": p.description,
            }
            for p in presets
        ]
        print(json.dumps(out, indent=2))
        return 0

    print(f"{'PRESET KEY':<22} {'BACKEND':<8} {'VRAM':<7} {'MODEL / TARGET':<32} DESCRIPTION")
    print("-" * 110)
    for p in presets:
        vram = f"{p.recommended_vram_gb}GB" if p.recommended_vram_gb else "Cloud"
        print(f"{p.key:<22} {p.backend.value:<8} {vram:<7} {p.model_id:<32} {p.description}")
    return 0


def _cmd_models(args: argparse.Namespace) -> int:
    models = discover_local_models()
    endpoints = probe_local_endpoints()

    if args.json:
        print(
            json.dumps(
                {
                    "local_models": [
                        {
                            "name": m.name,
                            "backend": m.backend.value,
                            "path": m.path,
                            "size_gb": m.size_gb,
                            "source": m.source,
                        }
                        for m in models
                    ],
                    "endpoints": [
                        {
                            "label": e.label,
                            "base_url": e.base_url,
                            "reachable": e.reachable,
                            "models": e.models,
                        }
                        for e in endpoints
                    ],
                },
                indent=2,
            )
        )
        return 0

    print("=== Discovered Local Checkpoints ===")
    if not models:
        print("  (No local LiteRT or HuggingFace checkpoints detected)")
    else:
        print(f"  {'NAME':<36} {'BACKEND':<9} {'SIZE':<9} {'SOURCE':<18} PATH")
        print("  " + "-" * 104)
        for m in models:
            print(
                f"  {m.name:<36} {m.backend.value:<9} {f'{m.size_gb:.2f} GB':<9} "
                f"{m.source:<18} {m.path}"
            )

    print("\n=== Active OpenAI-Compatible Local Servers ===")
    for e in endpoints:
        status = "ONLINE " if e.reachable else "OFFLINE"
        models_str = ", ".join(e.models) if e.models else "(none)"
        print(f"  [{status}] {e.label:<22} {e.base_url:<30} Models: {models_str}")
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    litert_bin = shutil.which("litert-lm") or (
        os.path.join(os.path.dirname(sys.executable), "litert-lm")
        if os.path.exists(os.path.join(os.path.dirname(sys.executable), "litert-lm"))
        else None
    )
    ensure_bin_symlinks()
    nvidia_smi = shutil.which("nvidia-smi")
    chronos_mcp = shutil.which("chronosml-mcp")
    opencommons_mcp = shutil.which("opencommons-mcp")
    models = discover_local_models()
    endpoints = probe_local_endpoints()
    oc_store = load_persistent_store()
    oc_pending = load_pending_proposals()

    report = {
        "python": sys.executable,
        "litert_lm_cli": litert_bin,
        "nvidia_gpu_detected": bool(nvidia_smi),
        "chronosml_mcp_detected": bool(chronos_mcp),
        "opencommons_mcp_detected": bool(opencommons_mcp),
        "opencommons_active_concepts": len(oc_store.canonical_block_by_cid),
        "opencommons_pending_proposals": len(oc_pending),
        "discovered_models_count": len(models),
        "online_endpoints": [e.base_url for e in endpoints if e.reachable],
    }

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print("=== Kardashev CLI Environment Doctor ===")
    print(f"  Python interpreter   : {report['python']}")
    print(f"  litert-lm CLI        : {report['litert_lm_cli'] or 'MISSING (pip install litert-lm)'}")
    print(f"  NVIDIA GPU (CUDA)    : {'DETECTED' if report['nvidia_gpu_detected'] else 'Not found'}")
    print(f"  ChronosML MCP        : {chronos_mcp or 'Not in PATH'}")
    print(
        f"  OpenCommons Engine   : {len(oc_store.canonical_block_by_cid)} active concept(s), "
        f"{len(oc_pending)} pending review ({opencommons_mcp or 'in-process'})"
    )
    print(f"  Cached Local Models  : {len(models)} found")
    print(f"  Live Local Servers   : {len(report['online_endpoints'])} online")
    return 0


def _cmd_import_litert(args: argparse.Namespace) -> int:
    preset = get_preset(args.preset)
    if preset.backend != BackendKind.LITERT or not preset.hf_repo or not preset.litert_filename:
        print(f"Error: Preset '{preset.key}' is not an importable LiteRT preset.", file=sys.stderr)
        return 1

    litert_bin = shutil.which("litert-lm")
    if not litert_bin:
        candidate = os.path.join(os.path.dirname(sys.executable), "litert-lm")
        if os.path.exists(candidate):
            litert_bin = candidate
    if not litert_bin:
        print("Error: 'litert-lm' CLI not found. Install with: pip install litert-lm", file=sys.stderr)
        return 1

    cmd = [
        litert_bin,
        "import",
        f"--from-huggingface-repo={preset.hf_repo}",
        preset.litert_filename,
        preset.model_id,
    ]
    print(f"Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, check=False)
    return proc.returncode


def _cmd_serve(args: argparse.Namespace) -> int:
    plan = compute_serve_plan(
        preset=args.preset,
        model=args.model,
        engine=args.engine,
        host=args.host,
        port=args.port,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=args.enforce_eager,
    )
    if args.dry_run:
        print(json.dumps(plan.to_dict(), indent=2))
        return 0

    print("============================================================")
    print(f"  Kardashev CLI Server Launcher ({plan.engine})")
    print(f"  Model Target : {plan.model_target} (as '{plan.served_model_name}')")
    print(f"  Endpoint     : http://{plan.host}:{plan.port}/v1")
    print(
        f"  Memory Guard : ~{plan.estimated_model_gb:.2f} GB model / "
        f"{plan.available_mem_gb:.2f} GB avail (util={plan.gpu_memory_utilization}, "
        f"max_len={plan.max_model_len}, oom_risk={plan.oom_risk})"
    )
    print(f"  Command      : {' '.join(plan.command)}")
    print("============================================================")

    proc = subprocess.run(plan.command, check=False)
    return proc.returncode


def _cmd_update(args: argparse.Namespace) -> int:
    status = check_for_updates(force=True)
    if getattr(args, "json", False):
        print(json.dumps(status.to_dict(), indent=2))
        return 0
    if getattr(args, "check", False):
        print(status.message)
        return 0

    print(f"Checking for updates ({status.install_mode})...")
    ok, msg = perform_self_update(include_litert=True)
    if ok:
        print(f"✓ {msg}")
        return 0
    print(f"✗ {msg}", file=sys.stderr)
    return 1


def _cmd_config(args: argparse.Namespace) -> int:
    cfg = UserConfig.load()
    action = getattr(args, "action", "show")
    if action == "path":
        print(UserConfig.config_path())
        return 0
    if action == "reset":
        fresh = UserConfig()
        path = fresh.save()
        print(f"✓ Reset user configuration to defaults at {path}")
        return 0
    if action == "set":
        key = getattr(args, "key", None)
        val = getattr(args, "value", None)
        if not key or val is None:
            print("Usage: kardashev config set <key> <value>", file=sys.stderr)
            return 1
        if not hasattr(cfg, key):
            valid = ", ".join(UserConfig.__dataclass_fields__.keys())
            print(f"Error: Unknown config key '{key}'. Valid keys: {valid}", file=sys.stderr)
            return 1
        current = getattr(cfg, key)
        if isinstance(current, bool):
            parsed: object = str(val).lower() in ("1", "true", "yes", "on")
        elif val.lower() in ("none", "null", "off") and key == "ralph_verify_cmd":
            parsed = None
        else:
            parsed = val
        setattr(cfg, key, parsed)
        path = cfg.save()
        print(f"✓ Set {key} = {parsed!r} in {path}")
        return 0

    print(json.dumps(cfg.__dict__, indent=2))
    return 0


def _cmd_integrate(args: argparse.Namespace) -> int:
    ensure_bin_symlinks()
    target = getattr(args, "target", "all") or "all"
    results = integrate_frontier_agent(target)
    if getattr(args, "json", False):
        print(json.dumps([r.to_dict() for r in results], indent=2))
        return 0
    print("=== Kardashev CLI Frontier Agent Co-Processor Integration ===")
    for r in results:
        print(f"  ✓ [{r.target.upper()}]")
        print(f"      MCP Config : {r.mcp_config_path}")
        print(f"      Skill/Rule : {r.skill_path}")
        print(f"      Executable : {r.binary_command}")
    return 0


def _cmd_scout(args: argparse.Namespace) -> int:
    query = " ".join(getattr(args, "query", []) or [])
    ws = getattr(args, "workspace", os.getcwd())
    max_slices = int(getattr(args, "max_slices", 5))
    res = scout_workspace(ws, query, max_slices=max_slices)
    if getattr(args, "json", False):
        print(json.dumps(res.to_dict(), indent=2))
        return 0
    print(res.compact_repo_map)
    if res.surgical_slices:
        print(
            f"\n=== Surgical Code Slices ({len(res.surgical_slices)} hit(s) · "
            f"~{res.estimated_tokens_saved:,} raw tokens saved) ==="
        )
        for sl in res.surgical_slices:
            print(f"\n--- {sl['path']} :: {sl['symbol']} (L{sl['start_line']}-{sl['end_line']}) ---")
            print(sl["code"])
    return 0


def _cmd_commons(args: argparse.Namespace) -> int:
    from pathlib import Path as _Path

    from kardashev_cli.opencommons import (
        DEFAULT_PENDING_PATH,
        DEFAULT_STORE_PATH,
        approve_pending_proposal,
        create_opencommons_hub_server,
        execute_commons_record,
        load_pending_proposals,
        load_persistent_store,
        pull_from_remote_hub,
        push_to_remote_hub,
        reject_pending_proposal,
        run_opencommons_stdio_server,
        save_persistent_store,
        seed_default_scientific_primitives,
    )

    action = getattr(args, "action", "status") or "status"
    targets = list(getattr(args, "target", []) or [])
    store_path = getattr(args, "store", None)

    if action == "settings":
        from kardashev_cli import contrib_settings

        sub = targets[0] if targets else "show"
        if sub == "show":
            print(json.dumps(contrib_settings.load_settings().to_dict(), indent=2))
            return 0
        if sub == "set":
            if len(targets) < 3:
                print("Usage: kardashev commons settings set KEY VALUE", file=sys.stderr)
                return 1
            key, value = targets[1], " ".join(targets[2:])
            try:
                current = contrib_settings.load_settings()
                updated = contrib_settings.apply_update(current, key, value)
                contrib_settings.save_settings(updated)
            except contrib_settings.ContribSettingsError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                return 1
            print(json.dumps(updated.to_dict(), indent=2))
            return 0
        print("Usage: kardashev commons settings show|set KEY VALUE", file=sys.stderr)
        return 1

    if action == "dashboard":
        from kardashev_cli.contrib_dashboard import start_dashboard, stop_dashboard

        port = int(getattr(args, "port", 0) or 0)
        no_browser = bool(getattr(args, "no_browser", False))
        srv, thread = start_dashboard(port=port, open_browser=not no_browser)
        host, actual_port = srv.server_address[:2]
        print(f"=== OpenCommons contribution dashboard listening on http://{host}:{actual_port} ===")
        print("  (127.0.0.1 only; Ctrl+C to stop)")
        try:
            while thread.is_alive():
                thread.join(timeout=1)
        except KeyboardInterrupt:
            print("\nShutting down dashboard...")
        finally:
            stop_dashboard(srv, thread)
        return 0

    if action == "mcp":
        return run_opencommons_stdio_server(store_path=store_path)

    if action == "pull":
        hub_url = targets[0] if targets else None
        res = pull_from_remote_hub(
            hub_url=hub_url,
            store_path=store_path,
            force=bool(getattr(args, "force", False)),
        )
        print(json.dumps(res, indent=2))
        return 0 if res.get("ok") else 1

    if action == "push":
        hub_url = targets[0] if targets else None
        res = push_to_remote_hub(hub_url=hub_url, store_path=store_path)
        print(json.dumps(res, indent=2))
        return 0 if res.get("ok") else 1

    if action == "serve-hub":
        host = getattr(args, "host", "127.0.0.1") or "127.0.0.1"
        port = int(getattr(args, "port", 8765) or 8765)
        srv = create_opencommons_hub_server(host=host, port=port, store_path=store_path)
        actual_host, actual_port = srv.server_address[:2]
        print(f"=== OpenCommons CRDT Gossip Hub listening on http://{actual_host}:{actual_port} ===")
        print("  • GET  /v1/manifest  (16-bucket Merkle tree + ETag)")
        print("  • GET  /v1/store     (Full gzipped store replica)")
        print("  • POST /v1/sync      (Zero-Trust 5-gate verification + CRDT join)")
        print("  • POST /mcp          (Streamable HTTP MCP JSON-RPC 2.0 endpoint)")
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down OpenCommons Hub...")
        finally:
            srv.server_close()
        return 0

    if action == "seed":
        store = load_persistent_store(store_path)
        added = seed_default_scientific_primitives(store)
        merged = save_persistent_store(store, store_path)
        payload = {
            "seeded": added,
            "total_active_concepts": len(merged.canonical_block_by_cid),
            "store_path": str(_Path(store_path or DEFAULT_STORE_PATH).expanduser()),
        }
        print(json.dumps(payload, indent=2))
        return 0

    if action == "find":
        query = " ".join(targets).strip()
        if not query:
            print("Usage: kardashev commons find <query>", file=sys.stderr)
            return 1
        store = load_persistent_store(store_path)
        hits = store.find(query=query, top_k=int(getattr(args, "top_k", 3)))
        if getattr(args, "json", False):
            print(
                json.dumps(
                    [
                        {
                            "block_id": h.block.block_id,
                            "cid": h.block.cid,
                            "score": h.score,
                            "latency_ms": h.latency_ms,
                            "card": h.block.render_for_agent(getattr(args, "modality", "dual_card_with_pitfalls")),
                        }
                        for h in hits
                    ],
                    indent=2,
                )
            )
            return 0
        for h in hits:
            print(h.block.render_for_agent(mode=getattr(args, "modality", "dual_card_with_pitfalls")))
            print("-" * 60)
        return 0

    if action == "record":
        if not targets:
            print("Usage: kardashev commons record <file.py_or_code>", file=sys.stderr)
            return 1
        raw_arg = targets[0]
        src = _Path(raw_arg).read_text(encoding="utf-8") if _Path(raw_arg).is_file() else " ".join(targets)
        res = execute_commons_record(
            {
                "source_code": src,
                "domain": getattr(args, "domain", "general"),
                "summary": getattr(args, "summary", ""),
            },
            store_path=store_path,
        )
        print(json.dumps(res, indent=2))
        return 0

    if action == "review":
        pending = load_pending_proposals()
        if getattr(args, "json", False):
            print(json.dumps([p.to_dict() for p in pending], indent=2))
            return 0
        if not pending:
            print("✓ OpenCommons pending review queue is empty.")
            return 0
        print(f"=== Pending OpenCommons Agent Proposals ({len(pending)}) ===")
        for p in pending:
            print(f"  [{p.proposal_id}] {p.func_name} ({p.domain}) — {p.summary} [{p.stage_reason}]")
        return 0

    if action == "approve":
        if not targets:
            print("Usage: kardashev commons approve <proposal_id_or_func_name>", file=sys.stderr)
            return 1
        res = approve_pending_proposal(targets[0], store_path=store_path)
        print(json.dumps(res, indent=2))
        return 0 if res.get("approved") else 1

    if action == "reject":
        if not targets:
            print("Usage: kardashev commons reject <proposal_id_or_func_name>", file=sys.stderr)
            return 1
        res = reject_pending_proposal(targets[0])
        print(json.dumps(res, indent=2))
        return 0 if res.get("rejected") else 1

    if action == "sync":
        if not targets:
            print("Usage: kardashev commons sync <other_replica.json>", file=sys.stderr)
            return 1
        other = load_persistent_store(targets[0], auto_seed_if_empty=False)
        merged = save_persistent_store(other, path=store_path, crdt_merge_existing=True)
        print(
            json.dumps(
                {
                    "status": "crdt_merged",
                    "active_concepts": len(merged.canonical_block_by_cid),
                    "quarantined_blocks": len(merged.quarantined_ids),
                },
                indent=2,
            )
        )
        return 0

    # Default: status
    store = load_persistent_store(store_path)
    pending = load_pending_proposals()
    manifest = store.compute_merkle_manifest()
    payload = {
        "store_path": str(_Path(store_path or DEFAULT_STORE_PATH).expanduser()),
        "pending_queue_path": str(DEFAULT_PENDING_PATH),
        "total_blocks": len(store.blocks_by_id),
        "active_equivalence_classes_cid": len(store.canonical_block_by_cid),
        "quarantined_blocks": len(store.quarantined_ids),
        "pending_proposals_count": len(pending),
        "peer_review_executions": store.peer_review_executions,
        "sid_dedup_hits": store.sid_dedup_hits,
        "merkle_root": manifest["merkle_root"],
    }
    print(json.dumps(payload, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args_list = list(argv) if argv is not None else sys.argv[1:]

    if args_list in (["-V"], ["--version"]):
        status = check_for_updates(force=False)
        sha_info = f" ({status.local_commit[:7]})" if status.local_commit else ""
        print(f"kardashev {__version__}{sha_info} [{status.install_mode}]")
        return 0

    if args_list == ["--upgrade"]:
        args_list = ["update"]

    if not args_list:
        args_list = ["chat"]
    args = parser.parse_args(args_list)

    if getattr(args, "version", False):
        status = check_for_updates(force=False)
        sha_info = f" ({status.local_commit[:7]})" if status.local_commit else ""
        print(f"kardashev {__version__}{sha_info} [{status.install_mode}]")
        return 0

    try:
        if getattr(args, "upgrade", False) or args.command in ("update", "upgrade"):
            return _cmd_update(args)
        if args.command == "config":
            return _cmd_config(args)
        if args.command == "completions":
            print(generate_shell_completion(getattr(args, "shell", "bash")))
            return 0
        if args.command == "integrate":
            return _cmd_integrate(args)
        if args.command == "scout":
            return _cmd_scout(args)
        if args.command == "commons":
            return _cmd_commons(args)
        if args.command == "presets":
            return _cmd_presets(args)
        if args.command == "models":
            return _cmd_models(args)
        if args.command == "doctor":
            ensure_bin_symlinks()
            return _cmd_doctor(args)
        if args.command == "import-litert":
            return _cmd_import_litert(args)
        if args.command == "serve":
            return _cmd_serve(args)
        if args.command == "mcp":
            serve_stdio()
            return 0
        if args.command == "exec":
            return asyncio.run(_cmd_exec(args))
        if args.command == "chat":
            return asyncio.run(_cmd_chat(args))
    except (KeyboardInterrupt, asyncio.CancelledError):
        return 130
    except BrokenPipeError:
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyboardInterrupt, asyncio.CancelledError):
        sys.exit(130)
