#!/usr/bin/env python3
"""Run an interactive Antigravity tool harness backed by Gemma 4 26B on-device via LiteRT."""

import asyncio
import os
import sys

from google.antigravity import LiteRTAgentConfig
from google.antigravity.hooks import policy
from google.antigravity.utils.interactive import run_interactive_loop


async def main() -> None:
    workspace = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.getcwd()
    model_path = os.path.expanduser("~/.litert-lm/models/gemma4-26b/model.litertlm")

    config = LiteRTAgentConfig(
        model_path=model_path,
        workspaces=[workspace],
        policies=[policy.allow_all()],
    )

    print(f"Launching Antigravity Interactive Loop with LiteRT ({model_path})")
    print(f"Workspace: {workspace}")
    await run_interactive_loop(config)


if __name__ == "__main__":
    asyncio.run(main())
