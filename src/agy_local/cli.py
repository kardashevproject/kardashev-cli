"""CLI entrypoint for agy-local."""

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

from agy_local.builder import HarnessOptions, PolicyMode, build_agent_config
from agy_local.discovery import discover_local_models, probe_local_endpoints
from agy_local.presets import BackendKind, get_preset, list_presets


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agy-local",
        description=(
            "Run the Google Antigravity coding harness with local LiteRT models "
            "(Gemma 4 26B) or OpenAI-compatible servers (vLLM, Ollama, SGLang, LM Studio, GLM-4, Qwen)."
        ),
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
            "--dry-run",
            action="store_true",
            help="Print the resolved Antigravity AgentConfig summary and exit without launching.",
        )

    # 1. chat (interactive REPL via run_interactive_loop)
    chat_p = sub.add_parser(
        "chat",
        help="Launch the interactive Antigravity REPL (run_interactive_loop) in your terminal.",
    )
    _add_harness_args(chat_p)

    # 2. exec (single-shot autonomous prompt)
    exec_p = sub.add_parser(
        "exec",
        help="Execute a single autonomous task prompt with the full Antigravity tool harness.",
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

    return parser


def _options_from_args(args: argparse.Namespace) -> HarnessOptions:
    workspaces = args.workspaces if getattr(args, "workspaces", None) else [os.getcwd()]
    return HarnessOptions(
        preset=getattr(args, "preset", None),
        backend=getattr(args, "backend", None),
        model=getattr(args, "model", None),
        model_path=getattr(args, "model_path", None),
        base_url=getattr(args, "base_url", None),
        litert_backend=getattr(args, "litert_device", "gpu"),
        speculative_decoding=bool(getattr(args, "speculative", False)),
        download_if_missing=bool(getattr(args, "download_if_missing", False)),
        workspaces=workspaces,
        policy_mode=PolicyMode(getattr(args, "policy", "allow-all")),
        lightweight=not bool(getattr(args, "no_lightweight", False)),
        token_threshold=getattr(args, "token_threshold", None),
        system_instructions=getattr(args, "system_instructions", None),
        with_chronosml_mcp=bool(getattr(args, "with_chronosml", False)),
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

    print("============================================================")
    print("  Antigravity Local Harness (agy-local)")
    print(f"  Config  : {summary['config_class']} ({summary['backend']})")
    if summary["backend"] == "litert":
        print(f"  Model   : {summary.get('model_path')}")
        print(f"  Device  : {summary.get('litert_device')}")
    else:
        print(f"  Model   : {summary.get('model')}")
        print(f"  Server  : {summary.get('base_url')}")
    print(f"  Workdir : {', '.join(str(w) for w in summary['workspaces'])}")
    print("============================================================")

    await run_interactive_loop(cfg)
    return 0


async def _cmd_exec(args: argparse.Namespace) -> int:
    opts = _options_from_args(args)
    cfg = build_agent_config(opts)
    summary = _describe_config(cfg)

    if args.dry_run:
        summary["prompt"] = args.prompt
        print(json.dumps(summary, indent=2))
        return 0

    async with Agent(config=cfg) as agent:
        response = await agent.chat(args.prompt)

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
    nvidia_smi = shutil.which("nvidia-smi")
    chronos_mcp = shutil.which("chronosml-mcp")
    models = discover_local_models()
    endpoints = probe_local_endpoints()

    report = {
        "python": sys.executable,
        "litert_lm_cli": litert_bin,
        "nvidia_gpu_detected": bool(nvidia_smi),
        "chronosml_mcp_detected": bool(chronos_mcp),
        "discovered_models_count": len(models),
        "online_endpoints": [e.base_url for e in endpoints if e.reachable],
    }

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print("=== agy-local Environment Doctor ===")
    print(f"  Python interpreter   : {report['python']}")
    print(f"  litert-lm CLI        : {report['litert_lm_cli'] or 'MISSING (pip install litert-lm)'}")
    print(f"  NVIDIA GPU (CUDA)    : {'DETECTED' if report['nvidia_gpu_detected'] else 'Not found'}")
    print(f"  ChronosML MCP        : {chronos_mcp or 'Not in PATH'}")
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args_list = list(argv) if argv is not None else sys.argv[1:]
    if not args_list:
        args_list = ["chat"]
    args = parser.parse_args(args_list)

    if args.command == "presets":
        return _cmd_presets(args)
    if args.command == "models":
        return _cmd_models(args)
    if args.command == "doctor":
        return _cmd_doctor(args)
    if args.command == "import-litert":
        return _cmd_import_litert(args)
    if args.command == "exec":
        return asyncio.run(_cmd_exec(args))
    if args.command == "chat":
        return asyncio.run(_cmd_chat(args))

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
