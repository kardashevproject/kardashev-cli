"""Default out-of-process executor for untrusted concept code.

Submitted source code is never executed in the calling (verifier) process.
This module always hands it to a fresh ``python -I`` subprocess with CPU,
memory, and wall-clock limits, no network-relevant environment variables,
and a minimal stdin/stdout JSON protocol. Another pipeline (e.g. the
OpenCommons sandbox/L0-L8 security layers) may supply a different
``Executor`` callable with the same signature instead of this one.
"""

from __future__ import annotations

import json
import subprocess
import sys

from .types import Outcome

# Harness run inside the child interpreter. It reads one JSON object from
# stdin describing the source, the function to call, and a list of call
# argument-tuples, and writes one JSON object of per-call outcomes to stdout.
# Resource limits are applied here, before the untrusted source is compiled.
_HARNESS_SOURCE = r"""
import json
import math
import sys


def _limit_resources(cpu_seconds, mem_bytes):
    try:
        import resource
    except ImportError:
        return
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    except (ValueError, OSError):
        pass
    try:
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    except (ValueError, OSError):
        pass
    try:
        resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))
    except (ValueError, OSError):
        pass


def main():
    payload = json.loads(sys.stdin.read())
    _limit_resources(int(payload.get("cpu_seconds", 5)), int(payload.get("mem_bytes", 512 * 1024 * 1024)))

    source = payload["source"]
    func_name = payload["func"]
    calls = payload["calls"]

    outcomes = []
    # `math` is provided as an implicit global (OpenCommons concept blocks are stored and
    # eventually run without their own `import` statement -- the storage layer provides `math`
    # the same way; see kardashev_cli.opencommons's SAFE_EXEC_BUILTINS/`math` globals).
    namespace = {"__name__": "__opencommons_concept__", "__builtins__": __builtins__, "math": math}
    try:
        code = compile(source, "<concept>", "exec")
        exec(code, namespace)
        fn = namespace[func_name]
    except Exception as exc:  # noqa: BLE001 - must report, never crash silently
        for _ in calls:
            outcomes.append({"ok": False, "value": None, "error_type": type(exc).__name__ + ": load failed"})
        print(json.dumps({"outcomes": outcomes}))
        return

    for args in calls:
        try:
            result = fn(*args)
            outcomes.append({"ok": True, "value": result, "error_type": None})
        except Exception as exc:  # noqa: BLE001 - captured as a data outcome
            outcomes.append({"ok": False, "value": None, "error_type": type(exc).__name__})

    print(json.dumps({"outcomes": outcomes}))


main()
"""

#: Environment handed to the child process: no inherited secrets, no proxy
#: configuration that could be mistaken for network access being available.
_MINIMAL_ENV = {"PATH": "/usr/bin:/bin"}

DEFAULT_CPU_SECONDS = 5
DEFAULT_MEM_BYTES = 512 * 1024 * 1024


def subprocess_executor(
    source: str,
    func_name: str,
    args_list: list[tuple],
    timeout_s: float,
    *,
    cpu_seconds: int = DEFAULT_CPU_SECONDS,
    mem_bytes: int = DEFAULT_MEM_BYTES,
) -> list[Outcome]:
    """Run ``func_name(*args)`` for each entry of ``args_list`` in a sandboxed subprocess.

    Never raises on behalf of the submitted code: a crash, a timeout, or a
    load failure all come back as failing :class:`Outcome` entries instead.
    """
    payload = json.dumps(
        {
            "source": source,
            "func": func_name,
            "calls": [list(a) for a in args_list],
            "cpu_seconds": cpu_seconds,
            "mem_bytes": mem_bytes,
        }
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", _HARNESS_SOURCE],
            check=False,
            input=payload,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=_MINIMAL_ENV,
        )
    except subprocess.TimeoutExpired:
        return [Outcome(ok=False, value=None, error_type="TimeoutExpired") for _ in args_list]

    if proc.returncode != 0 or not proc.stdout.strip():
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        error_type = detail[-1][:200] if detail else "SubprocessError"
        return [Outcome(ok=False, value=None, error_type=error_type) for _ in args_list]

    try:
        parsed = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return [Outcome(ok=False, value=None, error_type="MalformedExecutorOutput") for _ in args_list]

    outcomes = []
    for raw in parsed.get("outcomes", []):
        outcomes.append(Outcome(ok=raw.get("ok", False), value=raw.get("value"), error_type=raw.get("error_type")))
    # Defensive: always return exactly one outcome per requested call.
    while len(outcomes) < len(args_list):
        outcomes.append(Outcome(ok=False, value=None, error_type="MissingOutcome"))
    return outcomes[: len(args_list)]
