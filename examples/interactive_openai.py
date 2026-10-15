#!/usr/bin/env python3
"""Run an interactive Antigravity tool harness backed by GLM-4, Qwen3.6, or any OpenAI-compatible server."""

import asyncio
import os
import sys

from google.antigravity import LocalOpenAIAgentConfig
from google.antigravity.hooks import policy
from google.antigravity.utils.interactive import run_interactive_loop


async def main() -> None:
    model = os.environ.get("AGY_LOCAL_MODEL", "glm4:9b")
    base_url = os.environ.get("AGY_LOCAL_BASE_URL", "http://127.0.0.1:11434/v1")
    workspace = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.getcwd()

    config = LocalOpenAIAgentConfig(
        model=model,
        base_url=base_url,
        workspaces=[workspace],
        policies=[policy.allow_all()],
    ).lightweight()

    print(f"Launching Antigravity Interactive Loop with {model} @ {base_url}")
    print(f"Workspace: {workspace}")
    await run_interactive_loop(config)


if __name__ == "__main__":
    asyncio.run(main())
