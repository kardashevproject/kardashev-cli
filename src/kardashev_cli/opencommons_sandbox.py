"""Out-of-process isolated execution worker for OpenCommons (Layer 7).

Executes untrusted candidate concepts and stored code blocks inside an unprivileged
subprocess communicating strictly over typed stdin/stdout JSON pipes:
- POSIX resource caps: RLIMIT_AS (256MB), RLIMIT_CPU (2s), RLIMIT_NPROC (1),
  RLIMIT_FSIZE (0), RLIMIT_CORE (0).
- Network isolation: unshare(CLONE_NEWUSER | CLONE_NEWNET) + socket monkeypatching.
- Dual-run determinism probing comparing runs with IEEE 754 quantized canonical serialization.
- Branch coverage tracing via sys.settrace on AST function body statements.
- Tamper-proof stdout redirection (fds 1/2 -> /dev/null, 128-bit authentic nonce).
- Zero in-process code execution in host process.
"""

from __future__ import annotations

import ast
import contextlib
import ctypes
import json
import math
import os
import resource
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Data Contracts (PROJECT.md § Interface Contracts)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Probe:
    """A single execution probe containing argument values and optional keyword arguments."""

    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: Any) -> Probe:
        if isinstance(raw, Probe):
            return raw
        if isinstance(raw, (tuple, list)):
            return cls(args=tuple(raw))
        if isinstance(raw, dict) and "args" in raw:
            return cls(args=tuple(raw.get("args", ())), kwargs=dict(raw.get("kwargs", {})))
        return cls(args=(raw,))


@dataclass
class ProbeResult:
    """Detailed execution result for an individual probe within a suite."""

    success: bool
    result: Any = None
    serialized_result: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    mutated_args: bool = False
    serialized_mutated_args: str = ""


@dataclass
class SandboxResult:
    """Aggregated result of running sandboxed execution probes on candidate code."""

    success: bool
    is_deterministic: bool
    branch_coverage: float
    outputs: list[Any]
    probe_results: list[ProbeResult] = field(default_factory=list)
    error_code: str | None = None  # "TIMEOUT", "OOM", "CRASH", "DISCREPANCY", "SANDBOX_VIOLATION", None
    error_message: str | None = None
    execution_time_ms: float = 0.0


@dataclass(frozen=True)
class SandboxConfig:
    """Resource limits and isolation settings for the sandbox worker."""

    as_limit_bytes: int = 256 * 1024 * 1024  # 256 MB virtual memory cap
    cpu_limit_seconds: int = 2  # 2.0s CPU time cap (SIGXCPU)
    wall_timeout_seconds: float = 2.5  # 2.5s wall clock timeout
    fsize_limit_bytes: int = 0  # 0 bytes (disallows regular file creation/writes)
    nproc_limit: int = 1  # 1 process (prevents fork bombs and thread creation)
    core_limit_bytes: int = 0  # 0 bytes (prevents core dumps)
    max_output_bytes: int = 64 * 1024  # 64 KB output cap
    isolate_network: bool = True  # CLONE_NEWNET namespace + socket monkeypatch
    min_branch_coverage: float = 0.0  # Threshold for branch coverage check


# ---------------------------------------------------------------------------
# Canonical Serialization (Float Quantization & Deterministic Sorting)
# ---------------------------------------------------------------------------


def canonical_serialize(val: Any) -> str:
    """Deterministically serialize arbitrary Python values with float quantization."""
    if val is None or isinstance(val, (bool, int, str)):
        return repr(val)
    if isinstance(val, float):
        if math.isnan(val):
            return "float:NaN"
        if math.isinf(val):
            return "float:Inf" if val > 0 else "float:-Inf"
        if abs(val) < 1e-12:
            return "float:0.000000000"
        return f"float:{val:.12g}"
    if isinstance(val, (bytes, bytearray)):
        return f"bytes:{bytes(val).hex()}"
    if isinstance(val, tuple):
        return "tuple:(" + ",".join(canonical_serialize(x) for x in val) + ")"
    if isinstance(val, list):
        return "list:[" + ",".join(canonical_serialize(x) for x in val) + "]"
    if isinstance(val, set):
        items = sorted(canonical_serialize(x) for x in val)
        return "set:{" + ",".join(items) + "}"
    if isinstance(val, dict):
        items = sorted((canonical_serialize(k), canonical_serialize(v)) for k, v in val.items())
        return "dict:{" + ",".join(f"{k}:{v}" for k, v in items) + "}"
    if hasattr(val, "tolist"):
        return "array:" + canonical_serialize(val.tolist())
    return f"obj:{type(val).__name__}:{val!r}"


# ---------------------------------------------------------------------------
# Typed Tagged JSON Encoding / Decoding (IPC Transport)
# ---------------------------------------------------------------------------


def safe_sandbox_encode(val: Any) -> Any:
    """Encodes arbitrary Python primitives and collections for JSON IPC."""
    if val is None or isinstance(val, (bool, int, str)):
        return val
    if isinstance(val, float):
        if math.isnan(val):
            return {"__type__": "float", "val": "nan"}
        if math.isinf(val):
            return {"__type__": "float", "val": "inf" if val > 0 else "-inf"}
        return val
    if isinstance(val, (bytes, bytearray)):
        return {"__type__": "bytes", "hex": bytes(val).hex()}
    if isinstance(val, tuple):
        return {"__type__": "tuple", "items": [safe_sandbox_encode(x) for x in val]}
    if isinstance(val, set):
        return {"__type__": "set", "items": [safe_sandbox_encode(x) for x in sorted(val, key=lambda x: str(x))]}
    if isinstance(val, list):
        return [safe_sandbox_encode(x) for x in val]
    if isinstance(val, dict):
        return {
            "__type__": "dict",
            "items": [[safe_sandbox_encode(k), safe_sandbox_encode(v)] for k, v in val.items()],
        }
    return repr(val)


def safe_sandbox_decode(obj: Any) -> Any:
    """Decodes tagged JSON IPC structures back into Python primitives and collections."""
    if isinstance(obj, dict) and "__type__" in obj:
        t = obj["__type__"]
        if t == "float":
            v = obj["val"]
            if v == "nan":
                return float("nan")
            if v == "inf":
                return float("inf")
            if v == "-inf":
                return float("-inf")
            return float(v)
        if t == "bytes":
            return bytes.fromhex(obj["hex"])
        if t == "tuple":
            return tuple(safe_sandbox_decode(x) for x in obj["items"])
        if t == "set":
            return {safe_sandbox_decode(x) for x in obj["items"]}
        if t == "dict":
            return {safe_sandbox_decode(k): safe_sandbox_decode(v) for k, v in obj["items"]}
    if isinstance(obj, list):
        return [safe_sandbox_decode(x) for x in obj]
    return obj


# ---------------------------------------------------------------------------
# Preexec Limit Application
# ---------------------------------------------------------------------------


def _apply_rlimits(config: SandboxConfig) -> None:
    """Executed in the child process immediately after fork() and before exec()."""
    # 1. Network isolation via unshare(CLONE_NEWUSER | CLONE_NEWNET) if requested
    if config.isolate_network:
        with contextlib.suppress(OSError, AttributeError):
            clone_newnet = 0x40000000
            clone_newuser = 0x10000000
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            libc.unshare(clone_newuser | clone_newnet)

    # 2. Address space limit (RLIMIT_AS <= 256MB)
    with contextlib.suppress(ValueError, OSError):
        resource.setrlimit(resource.RLIMIT_AS, (config.as_limit_bytes, config.as_limit_bytes))

    # 3. CPU time limit (RLIMIT_CPU <= 2s)
    with contextlib.suppress(ValueError, OSError):
        resource.setrlimit(resource.RLIMIT_CPU, (config.cpu_limit_seconds, config.cpu_limit_seconds + 1))

    # 4. File size limit (RLIMIT_FSIZE <= 0 bytes)
    with contextlib.suppress(ValueError, OSError):
        resource.setrlimit(resource.RLIMIT_FSIZE, (config.fsize_limit_bytes, config.fsize_limit_bytes))

    # 5. Core dump limit (RLIMIT_CORE <= 0)
    with contextlib.suppress(ValueError, OSError):
        resource.setrlimit(resource.RLIMIT_CORE, (config.core_limit_bytes, config.core_limit_bytes))

    # 6. Process count limit (RLIMIT_NPROC <= 1)
    with contextlib.suppress(ValueError, OSError):
        resource.setrlimit(resource.RLIMIT_NPROC, (config.nproc_limit, config.nproc_limit))


# ---------------------------------------------------------------------------
# AST Branch Analysis & Coverage
# ---------------------------------------------------------------------------


def extract_function_ast_info(
    source_code: str, func_name: str
) -> tuple[set[int], list[tuple[str, int, set[int], set[int]]]]:
    """Extracts executable line numbers and branch targets for the target function."""
    tree = ast.parse(source_code)
    fn_node: ast.FunctionDef | None = None
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name == func_name:
            fn_node = n
            break

    if fn_node is None:
        raise ValueError(f"Function '{func_name}' not found in source code")

    executable_lines: set[int] = set()
    branches: list[tuple[str, int, set[int], set[int]]] = []

    # Iterate strictly over statements in the function body
    for stmt_idx, stmt in enumerate(fn_node.body):
        # Skip leading docstrings
        if (
            stmt_idx == 0
            and isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        ):
            continue

        for node in ast.walk(stmt):
            if hasattr(node, "lineno"):
                executable_lines.add(node.lineno)

            if isinstance(node, ast.If):
                true_lines = {
                    n.lineno
                    for n in ast.walk(node)
                    if hasattr(n, "lineno") and any(n in ast.walk(b) for b in node.body)
                }
                false_lines = (
                    {
                        n.lineno
                        for n in ast.walk(node)
                        if hasattr(n, "lineno") and any(n in ast.walk(b) for b in node.orelse)
                    }
                    if node.orelse
                    else set()
                )
                branches.append(("If", node.lineno, true_lines, false_lines))

            elif isinstance(node, (ast.For, ast.While)):
                body_lines = {
                    n.lineno
                    for n in ast.walk(node)
                    if hasattr(n, "lineno") and any(n in ast.walk(b) for b in node.body)
                }
                orelse_lines = (
                    {
                        n.lineno
                        for n in ast.walk(node)
                        if hasattr(n, "lineno") and any(n in ast.walk(b) for b in node.orelse)
                    }
                    if node.orelse
                    else set()
                )
                branches.append(("Loop", node.lineno, body_lines, orelse_lines))

    return executable_lines, branches


def compute_coverage_ratio(
    executed_lines: set[int],
    executable_lines: set[int],
    branches: list[tuple[str, int, set[int], set[int]]],
) -> float:
    """Computes branch coverage as ratio of exercised executable lines and branch paths."""
    if not executable_lines:
        return 1.0

    line_cov = len(executed_lines.intersection(executable_lines)) / len(executable_lines)
    if not branches:
        return line_cov

    branch_points_hit = 0
    total_branch_decisions = len(branches) * 2

    for _b_type, b_lineno, true_lines, false_lines in branches:
        if b_lineno in executed_lines:
            if any(ln in executed_lines for ln in true_lines):
                branch_points_hit += 1
            if false_lines:
                if any(ln in executed_lines for ln in false_lines):
                    branch_points_hit += 1
            else:
                branch_points_hit += 1

    branch_cov = branch_points_hit / total_branch_decisions
    return round((line_cov * 0.5) + (branch_cov * 0.5), 4)


# ---------------------------------------------------------------------------
# Worker Bootstrap (Runs inside child process)
# ---------------------------------------------------------------------------

WORKER_BOOTSTRAP_CODE = r'''
import sys
import os
import json
import io
import math
import copy

def run_worker():
    # 1. Save communication pipe to parent and redirect fds 1 & 2 to /dev/null
    try:
        real_stdout_fd = os.dup(1)
        worker_out = os.fdopen(real_stdout_fd, "w", encoding="utf-8")
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        os.close(devnull)
    except Exception as exc:
        sys.stderr.write(f"Worker fd redirect failed: {exc}\n")
        sys.exit(1)

    # 2. Defense-in-depth: Block network socket creation
    try:
        import socket
        def _prohibited_socket(*args, **kwargs):
            raise PermissionError("Network socket creation is prohibited in sandbox")
        socket.socket = _prohibited_socket
    except Exception:
        pass

    # Helper tagged encoder/decoder
    def _tag_decode(obj):
        if isinstance(obj, dict) and "__type__" in obj:
            t = obj["__type__"]
            if t == "float":
                v = obj["val"]
                if v == "nan": return float("nan")
                if v == "inf": return float("inf")
                if v == "-inf": return float("-inf")
                return float(v)
            if t == "bytes":
                return bytes.fromhex(obj["hex"])
            if t == "tuple":
                return tuple(_tag_decode(x) for x in obj["items"])
            if t == "set":
                return {_tag_decode(x) for x in obj["items"]}
            if t == "dict":
                return {_tag_decode(k): _tag_decode(v) for k, v in obj["items"]}
        if isinstance(obj, list):
            return [_tag_decode(x) for x in obj]
        return obj

    def _tag_encode(val):
        if val is None or isinstance(val, (bool, int, str)):
            return val
        if isinstance(val, float):
            if math.isnan(val): return {"__type__": "float", "val": "nan"}
            if math.isinf(val): return {"__type__": "float", "val": "inf" if val > 0 else "-inf"}
            return val
        if isinstance(val, (bytes, bytearray)):
            return {"__type__": "bytes", "hex": bytes(val).hex()}
        if isinstance(val, tuple):
            return {"__type__": "tuple", "items": [_tag_encode(x) for x in val]}
        if isinstance(val, set):
            return {"__type__": "set", "items": [_tag_encode(x) for x in sorted(val, key=lambda x: str(x))]}
        if isinstance(val, list):
            return [_tag_encode(x) for x in val]
        if isinstance(val, dict):
            return {"__type__": "dict", "items": [[_tag_encode(k), _tag_encode(v)] for k, v in val.items()]}
        return repr(val)

    def _canon_ser(val):
        if val is None or isinstance(val, (bool, int, str)):
            return repr(val)
        if isinstance(val, float):
            if math.isnan(val):
                return "float:NaN"
            if math.isinf(val):
                return "float:Inf" if val > 0 else "float:-Inf"
            if abs(val) < 1e-12:
                return "float:0.000000000"
            return f"float:{val:.12g}"
        if isinstance(val, (bytes, bytearray)):
            return f"bytes:{bytes(val).hex()}"
        if isinstance(val, tuple):
            return "tuple:(" + ",".join(_canon_ser(x) for x in val) + ")"
        if isinstance(val, list):
            return "list:[" + ",".join(_canon_ser(x) for x in val) + "]"
        if isinstance(val, set):
            return "set:{" + ",".join(sorted(_canon_ser(x) for x in val)) + "}"
        if isinstance(val, dict):
            return "dict:{" + ",".join(f"{_canon_ser(k)}:{_canon_ser(v)}" for k, v in sorted(val.items(), key=lambda kv: str(kv[0]))) + "}"
        return repr(val)

    # 3. Read request from stdin
    try:
        raw_in = sys.stdin.read()
        req = json.loads(raw_in)
    except Exception as exc:
        worker_out.write(json.dumps({
            "success": False,
            "is_deterministic": False,
            "branch_coverage": 0.0,
            "outputs": [],
            "probe_results": [],
            "error_code": "CRASH",
            "error_message": f"Worker input parse failure: {exc}"
        }) + "\n")
        worker_out.flush()
        sys.exit(0)

    source_code = req["source_code"]
    func_name = req["func_name"]
    raw_probes = req.get("probes", [])
    dependency_sources = req.get("dependency_sources", [])
    nonce = req.get("nonce", "")

    ALLOWED_IMPORTS = frozenset({
        "math", "itertools", "functools", "collections", "heapq", "bisect",
        "operator", "dataclasses", "typing", "string", "re"
    })
    import importlib
    _real_import = __import__
    def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        root_pkg = name.split(".")[0]
        if root_pkg not in ALLOWED_IMPORTS:
            raise ImportError(f"Import of module '{name}' is forbidden in sandbox")
        return _real_import(name, globals, locals, fromlist, level)

    # 4. Safe Namespace Execution Setup
    safe_builtins = {
        "__import__": _safe_import,
        "abs": abs, "all": all, "any": any, "ascii": ascii, "bin": bin,
        "bool": bool, "bytearray": bytearray, "bytes": bytes, "chr": chr,
        "complex": complex, "dict": dict, "divmod": divmod, "enumerate": enumerate,
        "filter": filter, "float": float, "format": format, "frozenset": frozenset,
        "hex": hex, "int": int, "isinstance": isinstance, "issubclass": issubclass,
        "iter": iter, "len": len, "list": list, "map": map, "max": max,
        "min": min, "next": next, "oct": oct, "ord": ord, "pow": pow,
        "range": range, "repr": repr, "reversed": reversed, "round": round,
        "set": set, "slice": slice, "sorted": sorted, "str": str, "sum": sum,
        "tuple": tuple, "zip": zip, "math": math, "print": print,
        "ArithmeticError": ArithmeticError, "AssertionError": AssertionError,
        "AttributeError": AttributeError, "IndexError": IndexError,
        "KeyError": KeyError, "OverflowError": OverflowError,
        "TypeError": TypeError, "ValueError": ValueError, "ZeroDivisionError": ZeroDivisionError
    }
    exec_env = {"__builtins__": safe_builtins, "math": math}
    for mod_name in req.get("inject_modules", []):
        try:
            exec_env[mod_name] = _real_import(mod_name)
        except Exception:
            pass
    if "open" in req.get("inject_builtins", []):
        import builtins as _b
        exec_env["open"] = _b.open

    # 5. Compile and execute dependency sources
    for dep_idx, dep_src in enumerate(dependency_sources):
        try:
            dep_compiled = compile(dep_src, f"<dep_concept_{dep_idx}>", "exec")
            exec(dep_compiled, exec_env)
        except Exception as exc:
            worker_out.write(json.dumps({
                "nonce": nonce,
                "success": False,
                "is_deterministic": False,
                "branch_coverage": 0.0,
                "outputs": [],
                "probe_results": [],
                "error_code": "CRASH",
                "error_message": f"Dependency compilation error: {type(exc).__name__}: {exc}"
            }) + "\n")
            worker_out.flush()
            sys.exit(0)

    # 6. Compile candidate source code
    try:
        compiled = compile(source_code, "<sandboxed_concept>", "exec")
        exec(compiled, exec_env)
    except Exception as exc:
        worker_out.write(json.dumps({
            "nonce": nonce,
            "success": False,
            "is_deterministic": False,
            "branch_coverage": 0.0,
            "outputs": [],
            "probe_results": [],
            "error_code": "CRASH",
            "error_message": f"Compilation error: {type(exc).__name__}: {exc}"
        }) + "\n")
        worker_out.flush()
        sys.exit(0)

    fn = exec_env.get(func_name)
    if not callable(fn):
        worker_out.write(json.dumps({
            "nonce": nonce,
            "success": False,
            "is_deterministic": False,
            "branch_coverage": 0.0,
            "outputs": [],
            "probe_results": [],
            "error_code": "CRASH",
            "error_message": f"Object '{func_name}' is not callable"
        }) + "\n")
        worker_out.flush()
        sys.exit(0)

    # 7. Line and Branch Tracing
    executed_lines = set()
    def _trace(frame, event, arg):
        if event == "line" and frame.f_code.co_filename == "<sandboxed_concept>":
            executed_lines.add(frame.f_lineno)
        return _trace

    # 8. Determinism Probing & Output Collection
    outputs_run1 = []
    probe_results = []
    discrepancy_detected = False
    discrepancy_detail = None
    fatal_error = None
    valid_outputs = 0

    sys.settrace(_trace)
    try:
        for idx, probe in enumerate(raw_probes):
            p_args_decoded = _tag_decode(probe.get("args", [])) if isinstance(probe, dict) else _tag_decode(probe)
            p_kwargs_decoded = _tag_decode(probe.get("kwargs", {})) if isinstance(probe, dict) else {}

            # Run 1
            args_1 = copy.deepcopy(p_args_decoded)
            kwargs_1 = copy.deepcopy(p_kwargs_decoded)
            before_serialized = [_canon_ser(a) for a in args_1]
            pr_success = False
            pr_result = None
            pr_err_type = None
            pr_err_msg = None
            ser1 = ""

            try:
                res1 = fn(*args_1, **kwargs_1)
                ser1 = _canon_ser(res1)
                pr_success = True
                pr_result = _tag_encode(res1)
                valid_outputs += 1
            except Exception as exc:
                ser1 = f"ERR:{type(exc).__name__}:{exc}"
                pr_err_type = type(exc).__name__
                pr_err_msg = str(exc)

            after_serialized = [_canon_ser(a) for a in args_1]
            mutated = (after_serialized != before_serialized)
            state_str = "|".join(after_serialized) if mutated else ""

            # Run 2 (Determinism verification)
            args_2 = copy.deepcopy(p_args_decoded)
            kwargs_2 = copy.deepcopy(p_kwargs_decoded)
            try:
                res2 = fn(*args_2, **kwargs_2)
                ser2 = _canon_ser(res2)
            except Exception as exc:
                ser2 = f"ERR:{type(exc).__name__}:{exc}"

            if ser1 != ser2:
                discrepancy_detected = True
                discrepancy_detail = f"Probe {idx} produced non-deterministic outputs: run1={ser1!r} != run2={ser2!r}"

            outputs_run1.append(pr_result if pr_success else None)
            probe_results.append({
                "success": pr_success,
                "result": pr_result,
                "serialized_result": ser1 if pr_success else None,
                "error_type": pr_err_type,
                "error_message": pr_err_msg,
                "mutated_args": mutated,
                "serialized_mutated_args": state_str,
            })

            if discrepancy_detected:
                break
    except Exception as exc:
        fatal_error = f"Fatal probe execution error: {type(exc).__name__}: {exc}"
    finally:
        sys.settrace(None)

    # 9. Build Response
    if discrepancy_detected:
        res = {
            "nonce": nonce,
            "success": False,
            "is_deterministic": False,
            "executed_lines": sorted(list(executed_lines)),
            "outputs": outputs_run1,
            "probe_results": probe_results,
            "error_code": "DISCREPANCY",
            "error_message": discrepancy_detail
        }
    elif fatal_error:
        res = {
            "nonce": nonce,
            "success": False,
            "is_deterministic": False,
            "executed_lines": sorted(list(executed_lines)),
            "outputs": outputs_run1,
            "probe_results": probe_results,
            "error_code": "CRASH",
            "error_message": fatal_error
        }
    elif valid_outputs == 0 and len(raw_probes) > 0:
        res = {
            "nonce": nonce,
            "success": False,
            "is_deterministic": True,
            "executed_lines": sorted(list(executed_lines)),
            "outputs": outputs_run1,
            "probe_results": probe_results,
            "error_code": "CRASH",
            "error_message": "All probes raised exceptions"
        }
    else:
        res = {
            "nonce": nonce,
            "success": True,
            "is_deterministic": True,
            "executed_lines": sorted(list(executed_lines)),
            "outputs": outputs_run1,
            "probe_results": probe_results,
            "error_code": None,
            "error_message": None
        }

    worker_out.write(json.dumps(res) + "\n")
    worker_out.flush()

if __name__ == "__main__":
    run_worker()
'''


# ---------------------------------------------------------------------------
# Sandbox Runner Engine
# ---------------------------------------------------------------------------


class SandboxRunner:
    """Out-of-process isolated execution runner for untrusted concepts."""

    def __init__(self, config: SandboxConfig | None = None) -> None:
        self.config = config or SandboxConfig()

    def run_sandboxed_probes(
        self,
        source_code: str,
        func_name: str,
        probes: Sequence[Probe | tuple[Any, ...] | list[Any]],
        dependency_sources: Sequence[str] | None = None,
        timeout_seconds: float | None = None,
        inject_modules: list[str] | None = None,
        inject_builtins: list[str] | None = None,
    ) -> SandboxResult:
        """Executes candidate code inside an isolated unprivileged subprocess.

        Main process NEVER executes candidate code. All interactions are via JSON IPC.
        """
        t0 = time.perf_counter()

        # 1. Parse AST to extract executable lines and branches
        try:
            executable_lines, branches = extract_function_ast_info(source_code, func_name)
        except (SyntaxError, ValueError, TypeError) as exc:
            return SandboxResult(
                success=False,
                is_deterministic=False,
                branch_coverage=0.0,
                outputs=[],
                error_code="CRASH",
                error_message=f"Pre-execution AST analysis failed: {exc}",
                execution_time_ms=(time.perf_counter() - t0) * 1000.0,
            )

        # 2. Normalize and encode probes
        norm_probes = []
        for p in probes:
            if isinstance(p, Probe):
                norm_probes.append({
                    "args": safe_sandbox_encode(list(p.args)),
                    "kwargs": safe_sandbox_encode(p.kwargs),
                })
            elif isinstance(p, (tuple, list)):
                norm_probes.append({
                    "args": safe_sandbox_encode(list(p)),
                    "kwargs": {},
                })
            elif isinstance(p, dict) and "args" in p:
                norm_probes.append({
                    "args": safe_sandbox_encode(list(p.get("args", []))),
                    "kwargs": safe_sandbox_encode(dict(p.get("kwargs", {}))),
                })
            else:
                norm_probes.append({
                    "args": safe_sandbox_encode([p]),
                    "kwargs": {},
                })

        nonce = os.urandom(16).hex()
        request_payload = {
            "nonce": nonce,
            "source_code": source_code,
            "func_name": func_name,
            "probes": norm_probes,
            "dependency_sources": list(dependency_sources) if dependency_sources else [],
            "inject_modules": inject_modules or [],
            "inject_builtins": inject_builtins or [],
        }
        encoded_request = json.dumps(request_payload).encode("utf-8")

        # 3. Setup isolated subprocess with OS resource caps in preexec_fn
        effective_timeout = (
            timeout_seconds if timeout_seconds is not None else self.config.wall_timeout_seconds
        )
        config = self.config

        def _child_preexec() -> None:
            _apply_rlimits(config)

        cmd = [sys.executable, "-c", WORKER_BOOTSTRAP_CODE]

        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=_child_preexec,  # noqa: PLW1509
                cwd="/tmp",
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return SandboxResult(
                success=False,
                is_deterministic=False,
                branch_coverage=0.0,
                outputs=[],
                error_code="CRASH",
                error_message=f"Failed to spawn sandbox subprocess: {exc}",
                execution_time_ms=(time.perf_counter() - t0) * 1000.0,
            )

        # 4. Communicate with wall-clock timeout
        stdout_bytes = b""
        stderr_bytes = b""
        timed_out = False
        try:
            stdout_bytes, stderr_bytes = proc.communicate(
                input=encoded_request,
                timeout=effective_timeout,
            )
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            with contextlib.suppress(OSError, subprocess.SubprocessError):
                stdout_bytes, stderr_bytes = proc.communicate(timeout=0.5)

        returncode = proc.returncode
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # 5. Handle Process Failure Modes (Timeout, OOM, Signals)
        if timed_out or returncode in (-24, 137, -9):  # SIGXCPU, SIGKILL
            return SandboxResult(
                success=False,
                is_deterministic=False,
                branch_coverage=0.0,
                outputs=[],
                error_code="TIMEOUT",
                error_message=f"Sandbox execution timed out (returncode={returncode})",
                execution_time_ms=elapsed_ms,
            )

        # Check for OOM / Segfault / Resource violation
        if returncode in (-11, 139):  # SIGSEGV
            return SandboxResult(
                success=False,
                is_deterministic=False,
                branch_coverage=0.0,
                outputs=[],
                error_code="SANDBOX_VIOLATION",
                error_message=f"Process terminated by signal (SIGSEGV returncode={returncode})",
                execution_time_ms=elapsed_ms,
            )

        if returncode in (-25,):  # SIGXFSZ (File size limit exceeded)
            return SandboxResult(
                success=False,
                is_deterministic=False,
                branch_coverage=0.0,
                outputs=[],
                error_code="SANDBOX_VIOLATION",
                error_message="Sandbox file size limit violation (SIGXFSZ write attempt blocked)",
                execution_time_ms=elapsed_ms,
            )

        # Parse worker JSON output
        if not stdout_bytes:
            stderr_text = stderr_bytes.decode("utf-8", errors="replace")
            if "MemoryError" in stderr_text:
                return SandboxResult(
                    success=False,
                    is_deterministic=False,
                    branch_coverage=0.0,
                    outputs=[],
                    error_code="OOM",
                    error_message=f"Memory limit exceeded (RLIMIT_AS <= 256MB): {stderr_text[:200]}",
                    execution_time_ms=elapsed_ms,
                )
            return SandboxResult(
                success=False,
                is_deterministic=False,
                branch_coverage=0.0,
                outputs=[],
                error_code="CRASH",
                error_message=f"Subprocess terminated unexpectedly (returncode={returncode}): {stderr_text[:200]}",
                execution_time_ms=elapsed_ms,
            )

        # Output size cap check
        if len(stdout_bytes) > self.config.max_output_bytes:
            return SandboxResult(
                success=False,
                is_deterministic=False,
                branch_coverage=0.0,
                outputs=[],
                error_code="SANDBOX_VIOLATION",
                error_message=f"Sandbox output exceeded {self.config.max_output_bytes} bytes",
                execution_time_ms=elapsed_ms,
            )

        try:
            data = json.loads(stdout_bytes.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            return SandboxResult(
                success=False,
                is_deterministic=False,
                branch_coverage=0.0,
                outputs=[],
                error_code="CRASH",
                error_message=f"Failed to decode worker JSON: {exc}",
                execution_time_ms=elapsed_ms,
            )

        # Verify authentic nonce
        if data.get("nonce") != nonce:
            return SandboxResult(
                success=False,
                is_deterministic=False,
                branch_coverage=0.0,
                outputs=[],
                error_code="SANDBOX_VIOLATION",
                error_message="Worker nonce mismatch / stdout tampering detected",
                execution_time_ms=elapsed_ms,
            )

        # 6. Calculate Branch Coverage
        executed_lines = set(data.get("executed_lines", []))
        coverage_ratio = compute_coverage_ratio(executed_lines, executable_lines, branches)

        success = bool(data.get("success", False))
        error_code = data.get("error_code")
        error_message = data.get("error_message")

        # Check coverage threshold if configured
        if success and coverage_ratio < self.config.min_branch_coverage:
            success = False
            error_code = "SANDBOX_VIOLATION"
            error_message = (
                f"Branch coverage {coverage_ratio:.2f} is below threshold {self.config.min_branch_coverage:.2f}"
            )

        # Reconstruct decoded outputs and ProbeResult objects
        raw_outputs = data.get("outputs", [])
        decoded_outputs = [safe_sandbox_decode(o) for o in raw_outputs]

        probe_results: list[ProbeResult] = []
        for pr_dict in data.get("probe_results", []):
            pr_res = safe_sandbox_decode(pr_dict.get("result"))
            probe_results.append(
                ProbeResult(
                    success=bool(pr_dict.get("success", False)),
                    result=pr_res,
                    serialized_result=pr_dict.get("serialized_result"),
                    error_type=pr_dict.get("error_type"),
                    error_message=pr_dict.get("error_message"),
                    mutated_args=bool(pr_dict.get("mutated_args", False)),
                    serialized_mutated_args=pr_dict.get("serialized_mutated_args", ""),
                )
            )

        return SandboxResult(
            success=success,
            is_deterministic=bool(data.get("is_deterministic", False)),
            branch_coverage=coverage_ratio,
            outputs=decoded_outputs,
            probe_results=probe_results,
            error_code=error_code,
            error_message=error_message,
            execution_time_ms=elapsed_ms,
        )


# Global convenience function matching PROJECT.md interface contract
def run_sandboxed_probes(
    source_code: str,
    func_name: str,
    probes: Sequence[Probe | tuple[Any, ...] | list[Any]],
    config: SandboxConfig | None = None,
    dependency_sources: Sequence[str] | None = None,
    timeout_seconds: float | None = None,
    inject_modules: list[str] | None = None,
    inject_builtins: list[str] | None = None,
) -> SandboxResult:
    """Convenience wrapper for running sandboxed probes."""
    runner = SandboxRunner(config)
    return runner.run_sandboxed_probes(
        source_code=source_code,
        func_name=func_name,
        probes=probes,
        dependency_sources=dependency_sources,
        timeout_seconds=timeout_seconds,
        inject_modules=inject_modules,
        inject_builtins=inject_builtins,
    )
