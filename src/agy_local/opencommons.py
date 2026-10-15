"""Native OpenCommons Research Engine & Agent Participation Harness for `agy-local`.

Integrates The Kardashev Project's **Open-Science Concept Commons (`OpenCommons`)** directly into
the `agy-local` cognitive harness so local open-weight models (`Gemma 4 26B`, `Qwen 3.6`, `GLM-4`)
and connected frontier agents (`Antigravity 2.0`, `Claude Code`, `Cursor`, `Codex`) actively
retrieve verified scientific/algorithmic knowledge and contribute back new discoveries:

1. **Fast Structural & Behavioral Fingerprinting (`<5ms`)**:
   - Alpha-normalized De Bruijn AST Merkle hash (`sid`) for $O(1)$ structural deduplication.
   - 16-probe stratified canonical execution fingerprint (`fp -> cid`) that merges distinct
     algorithms computing the same mathematical function into a single Canonical Equivalence Class.
2. **5-Gate Asynchronous Adversarial Peer Review**:
   - Audits contributed Python functions for (1) zero valid executions, (2) degenerate constant
     stub outputs (F12/F19 spec gaming), (3) silent in-place caller argument mutation (F80),
     (4) explicit contract I/O mismatches, and (5) 32-probe adversarial/boundary invariant checks.
3. **Tiered Research Participation & Pending Review Queue (`~/.opencommons/`)**:
   - **Auto-Commit Tier**: Functions that pass all 5 peer-review gates are immediately committed
     into `~/.opencommons/store.json` (via atomic `fcntl` file-locked G-Set join-semilattice CRDT
     merge $\\sqcup$) with `trust_tier="peer_reviewed"`.
   - **Pending Review Queue (`~/.opencommons/pending_queue.json`)**: Uncertain, stateful, or
     domain-specific functions (or agent-proposed `PitfallWarning` updates that need human
     inspection) are staged in the pending review queue for `/commons review`,
     `agy commons approve <id>`, or `agy commons reject <id>`.
4. **Monotonic Counterexample Ratchet**:
   - When an agent discovers an unprobed $(K+1)$-th input $x_{K+1}$ that breaks an existing
     `ConceptBlock`, `report_counterexample` verifies the failure, permanently quarantines the
     refuted block in $Q$, promotes the next valid representative for that `cid`, and attaches a
     `PitfallWarning` so no future agent repeats the dead-end.
5. **Turn-0 Commons Context Injection & Autonomous Post-Edit Extraction**:
   - `build_turn_zero_commons_context(query)` injects matching Dual Concept Cards + Refuted
     Pitfalls (`<opencommons_knowledge>`) into Turn 0 alongside the AST Repo-Map.
   - `extract_candidate_functions_from_edit(file_path, new_content)` harvests newly written,
     type-annotated helper functions after agent edits/verifications and submits them through the
     tiered OpenCommons pipeline.
"""

from __future__ import annotations

import ast
import builtins
import contextlib
import copy
import fcntl
import gzip
import hashlib
import http.server
import json
import math
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

ServingMode = Literal["raw_code", "pseudocode", "dual_card", "dual_card_with_pitfalls"]
TrustTier = Literal["probe_verified", "peer_reviewed", "quarantined", "pending_review"]

DEFAULT_STORE_PATH = Path(
    os.environ.get("OPENCOMMONS_STORE_PATH", "~/.opencommons/store.json")
).expanduser()

DEFAULT_PENDING_PATH = Path(
    os.environ.get("OPENCOMMONS_PENDING_PATH", "~/.opencommons/pending_queue.json")
).expanduser()

DEFAULT_HUB_URL = os.environ.get(
    "OPENCOMMONS_HUB_URL",
    "https://raw.githubusercontent.com/kardashevproject/kardashev-cli/main/commons/store.json",
)

MAX_AST_NODES = 500
FORBIDDEN_CALL_NAMES = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "open",
        "input",
        "breakpoint",
        "__import__",
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
        "vars",
        "dir",
        "help",
        "memoryview",
        "exit",
        "quit",
    }
)

SAFE_EXEC_BUILTINS: dict[str, Any] = {
    "abs": abs,
    "all": all,
    "any": any,
    "bin": bin,
    "bool": bool,
    "bytearray": bytearray,
    "bytes": bytes,
    "callable": callable,
    "chr": chr,
    "complex": complex,
    "dict": dict,
    "divmod": divmod,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "frozenset": frozenset,
    "hasattr": hasattr,
    "hash": hash,
    "hex": hex,
    "int": int,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "iter": iter,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "next": next,
    "oct": oct,
    "ord": ord,
    "pow": pow,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "set": set,
    "slice": slice,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "type": type,
    "zip": zip,
    "Exception": Exception,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "ZeroDivisionError": ZeroDivisionError,
    "ArithmeticError": ArithmeticError,
    "OverflowError": OverflowError,
    "IndexError": IndexError,
    "KeyError": KeyError,
    "RuntimeError": RuntimeError,
    "AssertionError": AssertionError,
    "StopIteration": StopIteration,
}


class UnsafeConceptCodeError(ValueError):
    """Raised when a candidate Python concept block violates the static AST security policy."""


def verify_ast_safety_static(source_code: str) -> ast.Module:
    """Layer-1 Zero-Trust Static AST Capability Verifier (<0.05ms).

    Rejects:
    - Any `import` or `from ... import` statement (`ast.Import`, `ast.ImportFrom`)
    - Any dunder identifier or attribute access containing `__` (blocks `__subclasses__`,
      `__globals__`, `__builtins__`, `__code__`, `__dict__`, `__import__` sandbox escapes)
    - Calls to forbidden dynamic execution or I/O builtins (`eval`, `exec`, `open`, `getattr`, etc.)
    - Syntax-bomb DoS trees exceeding `MAX_AST_NODES` (500 nodes)
    """
    tree = ast.parse(source_code)
    for node_count, node in enumerate(ast.walk(tree), start=1):
        if node_count > MAX_AST_NODES:
            raise UnsafeConceptCodeError(
                f"AST exceeds maximum allowed complexity ({node_count} > {MAX_AST_NODES} nodes)."
            )
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise UnsafeConceptCodeError("Import statements are forbidden in OpenCommons pure concept blocks.")
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            raise UnsafeConceptCodeError("Global/nonlocal state mutation is forbidden in OpenCommons blocks.")
        if isinstance(node, ast.Name):
            if "__" in node.id:
                raise UnsafeConceptCodeError(f"Dunder identifier '{node.id}' is forbidden by security policy.")
            if node.id in FORBIDDEN_CALL_NAMES:
                raise UnsafeConceptCodeError(f"Builtin '{node.id}' is forbidden by security policy.")
        if isinstance(node, ast.Attribute) and "__" in node.attr:
            raise UnsafeConceptCodeError(f"Dunder attribute '.{node.attr}' is forbidden by security policy.")
    return tree


_TOKEN_RE = re.compile(r"[a-zA-Z]+|\d+")
_ALPHANUM_CONCEPT_RE = re.compile(r"([a-zA-Z]+)(\d+)")
COMMUTATIVE_OPS = (ast.Add, ast.Mult, ast.BitAnd, ast.BitOr, ast.BitXor)
COMMUTATIVE_CMP = (ast.Eq, ast.NotEq)


def tokenize_concept_query(text: str) -> tuple[set[str], dict[str, set[str]]]:
    """Tokenize identifier/query text while preserving numbered-concept stems (F87 lesson).

    Discriminates numbered concepts such as `base58` vs `base64`, `crc16` vs `crc32`, `rk4` vs `rk2`.
    """
    lowered = text.lower()
    numbered_stems: dict[str, set[str]] = {}
    for m in _ALPHANUM_CONCEPT_RE.finditer(lowered):
        stem, num = m.group(1), m.group(2)
        if len(stem) >= 2:
            numbered_stems.setdefault(stem, set()).add(num)

    raw_parts = re.split(r"[^a-z0-9]+", lowered)
    tokens: set[str] = set()
    for p in raw_parts:
        if not p:
            continue
        tokens.add(p)
        for sub in _TOKEN_RE.findall(p):
            if len(sub) >= 2 or sub.isdigit():
                tokens.add(sub)
    return tokens, numbered_stems


class _AlphaNormalizer(ast.NodeTransformer):
    """Normalizes parameter and local variable names to canonical De Bruijn positional names ($p0, $v0)."""

    def __init__(self, func_name: str) -> None:
        super().__init__()
        self.func_name = func_name
        self.var_map: dict[str, str] = {}
        self.param_count = 0
        self.local_count = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node = copy.copy(node)
        node.name = "$self"
        node.returns = None
        node.decorator_list = []

        body = list(node.body)
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body = body[1:]

        new_args = copy.copy(node.args)
        norm_args = []
        for arg in new_args.args:
            cname = f"$p{self.param_count}"
            self.param_count += 1
            self.var_map[arg.arg] = cname
            norm_args.append(ast.arg(arg=cname, annotation=None))
        new_args.args = norm_args
        new_args.vararg = None
        new_args.kwarg = None
        node.args = new_args
        node.body = [self.visit(stmt) for stmt in body]
        return node

    def visit_Name(self, node: ast.Name) -> ast.AST:
        node = copy.copy(node)
        if node.id == self.func_name:
            node.id = "$self"
        elif isinstance(node.ctx, ast.Store):
            if node.id not in self.var_map:
                self.var_map[node.id] = f"$v{self.local_count}"
                self.local_count += 1
            node.id = self.var_map[node.id]
        elif node.id in self.var_map:
            node.id = self.var_map[node.id]
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST:
        if node.value is None:
            return ast.Pass()
        return self.visit(ast.Assign(targets=[node.target], value=node.value))

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        node = copy.copy(node)
        node.left = self.visit(node.left)
        node.right = self.visit(node.right)
        if isinstance(node.op, COMMUTATIVE_OPS):
            d_left = ast.dump(node.left, include_attributes=False)
            d_right = ast.dump(node.right, include_attributes=False)
            if d_left > d_right:
                node.left, node.right = node.right, node.left
        return node

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        node = copy.copy(node)
        node.left = self.visit(node.left)
        node.comparators = [self.visit(c) for c in node.comparators]
        if len(node.ops) == 1 and isinstance(node.ops[0], COMMUTATIVE_CMP):
            d_left = ast.dump(node.left, include_attributes=False)
            d_right = ast.dump(node.comparators[0], include_attributes=False)
            if d_left > d_right:
                node.left, node.comparators[0] = node.comparators[0], node.left
        return node


def compute_structural_sid(source_code: str) -> tuple[str, str, tuple[str, ...], str, tuple[str, ...]]:
    """Parse Python source code, verify static AST security, and compute Merkle sid."""
    tree = verify_ast_safety_static(source_code)
    func_nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if not func_nodes:
        raise ValueError("No top-level FunctionDef found in source code.")
    fn = func_nodes[-1]
    func_name = fn.name

    param_types: list[str] = []
    for a in fn.args.args:
        if a.annotation is not None:
            param_types.append(ast.unparse(a.annotation).strip())
        else:
            param_types.append("Any")

    ret_type = ast.unparse(fn.returns).strip() if fn.returns is not None else "Any"

    builtins_set = set(dir(builtins))
    called: list[str] = []
    for sub in ast.walk(fn):
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
            cname = sub.func.id
            if cname != func_name and cname not in builtins_set and cname not in called:
                called.append(cname)

    norm_fn = _AlphaNormalizer(func_name).visit(copy.deepcopy(fn))
    canonical_ast = ast.dump(norm_fn, include_attributes=False)
    digest = hashlib.sha256(canonical_ast.encode("utf-8")).hexdigest()[:16]
    return f"sid:{digest}", func_name, tuple(param_types), ret_type, tuple(called)


def _canonical_serialize(val: Any) -> str:
    """Deterministically serialize arbitrary Python values (with float tolerance quantization)."""
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
        return "tuple:(" + ",".join(_canonical_serialize(x) for x in val) + ")"
    if isinstance(val, list):
        return "list:[" + ",".join(_canonical_serialize(x) for x in val) + "]"
    if isinstance(val, set):
        items = sorted(_canonical_serialize(x) for x in val)
        return "set:{" + ",".join(items) + "}"
    if isinstance(val, dict):
        items = sorted((_canonical_serialize(k), _canonical_serialize(v)) for k, v in val.items())
        return "dict:{" + ",".join(f"{k}:{v}" for k, v in items) + "}"
    if hasattr(val, "tolist"):
        return "array:" + _canonical_serialize(val.tolist())
    return f"obj:{type(val).__name__}:{val!r}"


def _build_probe_values_for_type(type_str: str) -> list[Any]:
    """Return 16 canonical stratified probe values for a given parameter type hint."""
    t = type_str.lower().replace(" ", "")
    if t in ("int",):
        return [0, 1, -1, 2, 3, 5, 7, 10, 12, 15, 16, 42, 100, 255, 1024, -7]
    if t in ("float",):
        return [0.0, 1.0, -1.0, 0.5, 2.0, -0.5, 0.25, 3.5, 10.0, 100.0, 0.1, 0.9, -2.5, 4.0, 8.0, 0.01]
    if t in ("bool",):
        return [False, True] * 8
    if t in ("str",):
        return [
            "",
            "a",
            "ab",
            "aba",
            "hello",
            "hello world",
            "12345",
            "A_B_C",
            "   spaces   ",
            "foo_bar_baz",
            "racecar",
            "alpha123beta",
            "test@example.com",
            "2026-10-14",
            "001100",
            "xyz",
        ]
    if t in ("bytes",):
        return [
            b"",
            b"\x00",
            b"\x01\x02",
            b"hello",
            b"\xff\x00\xff",
            b"123456789",
            b"commons",
            b"\xde\xad\xbe\xef",
            b"abc",
            b"\x00\x01\x02\x03",
            b"packet_payload",
            b"\x80\x7f",
            b"test",
            b"AAAA",
            b"\x10\x20\x30",
            b"end",
        ]
    if t in ("bytearray",):
        return [bytearray(b) for b in _build_probe_values_for_type("bytes")]
    if t.startswith(("list[float]", "sequence[float]")):
        return [
            [1.0, 2.0, 3.0],
            [0.0, 0.0],
            [-1.0, 0.0, 1.0],
            [10.0, 20.0, 30.0, 40.0],
            [0.5, 1.5, 2.5],
            [3.0, 1.0, 2.0],
            [5.0],
            [2.0, 4.0, 6.0, 8.0],
            [0.1, 0.2, 0.7],
            [-5.0, -2.0, -1.0],
            [1.0, 1.0, 1.0, 1.0],
            [100.0, 1.0],
            [0.25, 0.75],
            [1.0, 4.0, 9.0, 16.0],
            [2.5, -2.5],
            [0.0, 1.0],
            [1000.0, 1001.0, 1002.0],
            [-1000.0, -1001.0, -1002.0],
            [1e300, 1e300],
            [0.0, 0.0, 0.0],
            [42.0],
            [],
            [1e-300, -1e-300],
            [1e15, 1.0, -1e15],
        ]
    if t.startswith(("list[str]", "sequence[str]")):
        return [
            [],
            ["a"],
            ["a", "b", "a"],
            ["hello", "world"],
            ["foo", "bar", "baz"],
            ["x", "y", "z"],
            ["", "nonempty"],
            ["alpha", "beta"],
            ["one", "two", "two", "three"],
            ["cat", "dog"],
            ["a", "aa", "aaa"],
            ["test"],
            ["p", "q", "p", "q"],
            ["1", "2", "3"],
            ["same", "same"],
            ["first", "last"],
        ]
    if t.startswith("list[dict") or t == "list[dict]":
        return [
            [{"id": 1, "val": 10}, {"id": 2, "val": 20}],
            [{"id": 1, "val": 5}],
            [],
            [{"id": 1, "val": 10}, {"id": 1, "val": 15}],
            [{"a": 1, "b": 2}, {"a": 3, "b": 4}],
            [{"score": 0.9, "name": "x"}, {"score": 0.2, "name": "y"}],
            [{"k": "a", "n": 1}, {"k": "b", "n": 2}, {"k": "a", "n": 3}],
            [{"x": 0, "y": 0}],
        ] * 2
    if t.startswith("dict"):
        return [
            {},
            {"a": 1, "b": 2},
            {"x": 10, "y": 20, "z": 30},
            {"k1": 0, "k2": -1},
            {"alpha": 5, "beta": 5},
            {"one": 1},
            {"a": 100, "b": 50, "c": 25},
            {"key": 42},
        ] * 2
    if t.startswith("tuple"):
        return [
            (0, 0),
            (1, 2),
            (3, 4),
            (-1, 5),
            (10, 20),
            (100, 200),
            (7, 3),
            (8, 8),
        ] * 2
    return [
        [],
        [1],
        [1, 2, 3],
        [3, 1, 2],
        [3, 1, 3, 2, 1],
        [0, 0, 0],
        [-3, -1, 0, 2, 5],
        [10, 20, 30, 40],
        [5, 4, 3, 2, 1],
        [2, 2, 3, 3, 3],
        [100, -100],
        [1, 3, 5, 7, 9],
        [2, 4, 6, 8],
        [42],
        [0, 1, 0, 1],
        [9, 1, 8, 2, 7],
    ]


def generate_canonical_probes(param_types: tuple[str, ...], n_probes: int = 16) -> list[tuple[Any, ...]]:
    """Construct n_probes deterministic argument tuples matching param_types."""
    if not param_types:
        return [()]
    per_param = [_build_probe_values_for_type(pt) for pt in param_types]
    probes: list[tuple[Any, ...]] = []
    for i in range(n_probes):
        args = []
        for p_idx, plist in enumerate(per_param):
            offset = (i + p_idx * 3) if i >= 4 else i
            args.append(copy.deepcopy(plist[offset % len(plist)]))
        probes.append(tuple(args))
    return probes


@dataclass(frozen=True)
class FingerprintResult:
    """Result of structural + behavioral fingerprinting on a native Python function."""

    sid: str
    fp: str
    func_name: str
    param_types: tuple[str, ...]
    return_type: str
    dependencies: tuple[str, ...]
    valid_probe_count: int
    total_probe_count: int
    mutates_arguments: bool
    is_constant_output: bool
    elapsed_ms: float
    sample_outputs: tuple[str, ...]


def fingerprint_function(
    source_code: str,
    extra_globals: dict[str, Any] | None = None,
    n_probes: int = 16,
) -> FingerprintResult:
    """Compute structural Merkle sid and execution-based behavioral fingerprint fp in <5ms."""
    t0 = time.perf_counter()
    sid, func_name, param_types, return_type, deps = compute_structural_sid(source_code)

    ns: dict[str, Any] = {"__builtins__": SAFE_EXEC_BUILTINS, "math": math}
    if extra_globals:
        ns.update(extra_globals)
    exec(compile(source_code, "<commons_block>", "exec"), ns)  # noqa: S102
    fn: Callable[..., Any] = ns[func_name]

    probes = generate_canonical_probes(param_types, n_probes=n_probes)
    outcome_tokens: list[str] = []
    valid_outputs: list[str] = []
    mutates_args = False

    for probe_args in probes:
        before_serialized = [_canonical_serialize(a) for a in probe_args]
        working_args = [copy.deepcopy(a) for a in probe_args]
        try:
            res = fn(*working_args)
            after_serialized = [_canonical_serialize(a) for a in working_args]
            if after_serialized != before_serialized:
                mutates_args = True
            res_str = _canonical_serialize(res)
            state_str = "|".join(after_serialized) if mutates_args else ""
            token = f"OK:{res_str}:M({state_str})"
            outcome_tokens.append(token)
            valid_outputs.append(res_str)
        except Exception as exc:  # noqa: BLE001
            outcome_tokens.append(f"ERR:{type(exc).__name__}")

    valid_count = len(valid_outputs)
    unique_valid = len(set(valid_outputs))
    is_constant = (
        valid_count >= 4
        and unique_valid == 1
        and return_type.lower() not in ("bool", "none")
        and len(param_types) >= 1
    )

    if valid_count > 0:
        fp_digest = hashlib.sha256("\n".join(outcome_tokens).encode("utf-8")).hexdigest()[:16]
        fp = f"fp:{fp_digest}"
    else:
        fp = f"fp_struct:{sid[4:]}"

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return FingerprintResult(
        sid=sid,
        fp=fp,
        func_name=func_name,
        param_types=param_types,
        return_type=return_type,
        dependencies=deps,
        valid_probe_count=valid_count,
        total_probe_count=len(probes),
        mutates_arguments=mutates_args,
        is_constant_output=is_constant,
        elapsed_ms=round(elapsed_ms, 4),
        sample_outputs=tuple(valid_outputs[:4]),
    )


@dataclass(frozen=True)
class PeerReviewVerdict:
    """Outcome of the 5-gate adversarial peer-review audit on a contributed concept block."""

    trust_tier: TrustTier
    passed: bool
    reasons: tuple[str, ...]
    counterexample: str | None = None


def run_async_peer_review(
    source_code: str,
    fp_result: FingerprintResult,
    contract_io: list[tuple[tuple[Any, ...], Any]] | None = None,
    invariant_fn: Callable[[tuple[Any, ...], Any], bool] | None = None,
    reference_fn: Callable[..., Any] | None = None,
    allow_mutation: bool = False,
    extra_globals: dict[str, Any] | None = None,
) -> PeerReviewVerdict:
    """Audit a contributed block across 5 adversarial gates (F12/F19, F32/F33/F65, F78, F80)."""
    reasons: list[str] = []

    if fp_result.valid_probe_count == 0 and not contract_io:
        return PeerReviewVerdict(
            trust_tier="quarantined",
            passed=False,
            reasons=("zero_valid_probes: function raised exceptions on all 16 canonical probes",),
        )

    if fp_result.is_constant_output:
        return PeerReviewVerdict(
            trust_tier="quarantined",
            passed=False,
            reasons=(f"degenerate_constant_output: returned {fp_result.sample_outputs[0]} on all probes",),
        )

    if fp_result.mutates_arguments and not allow_mutation and fp_result.return_type.lower() != "none":
        return PeerReviewVerdict(
            trust_tier="quarantined",
            passed=False,
            reasons=("silent_argument_mutation: mutated caller input container in-place while returning a value",),
        )

    verify_ast_safety_static(source_code)
    ns: dict[str, Any] = {"__builtins__": SAFE_EXEC_BUILTINS, "math": math}
    if extra_globals:
        ns.update(extra_globals)
    exec(compile(source_code, "<peer_review>", "exec"), ns)  # noqa: S102
    fn: Callable[..., Any] = ns[fp_result.func_name]

    if contract_io:
        for args, expected in contract_io:
            try:
                actual = fn(*copy.deepcopy(args))
                if _canonical_serialize(actual) != _canonical_serialize(expected):
                    return PeerReviewVerdict(
                        trust_tier="quarantined",
                        passed=False,
                        reasons=("contract_io_mismatch",),
                        counterexample=f"args={args!r} -> got {actual!r}, expected {expected!r}",
                    )
            except Exception as exc:  # noqa: BLE001
                return PeerReviewVerdict(
                    trust_tier="quarantined",
                    passed=False,
                    reasons=(f"contract_io_exception:{type(exc).__name__}",),
                    counterexample=f"args={args!r} raised {exc!r}",
                )

    extended_probes = generate_canonical_probes(fp_result.param_types, n_probes=32)
    for probe_args in extended_probes:
        if invariant_fn is not None:
            try:
                out = fn(*copy.deepcopy(probe_args))
                if not invariant_fn(probe_args, out):
                    return PeerReviewVerdict(
                        trust_tier="quarantined",
                        passed=False,
                        reasons=("invariant_violation",),
                        counterexample=f"args={probe_args!r} -> output={out!r} violated invariant",
                    )
            except Exception as exc:  # noqa: BLE001
                return PeerReviewVerdict(
                    trust_tier="quarantined",
                    passed=False,
                    reasons=(f"adversarial_probe_crash:{type(exc).__name__}",),
                    counterexample=f"args={probe_args!r} raised {exc!r}",
                )

        if reference_fn is not None:
            try:
                ref_out = reference_fn(*copy.deepcopy(probe_args))
                ref_err = None
            except Exception as exc:  # noqa: BLE001
                ref_out = None
                ref_err = type(exc).__name__

            try:
                cand_out = fn(*copy.deepcopy(probe_args))
                cand_err = None
            except Exception as exc:  # noqa: BLE001
                cand_out = None
                cand_err = type(exc).__name__

            if ref_err != cand_err or (
                ref_err is None and _canonical_serialize(cand_out) != _canonical_serialize(ref_out)
            ):
                return PeerReviewVerdict(
                    trust_tier="quarantined",
                    passed=False,
                    reasons=("differential_refutation_mismatch",),
                    counterexample=(
                        f"args={probe_args!r} -> candidate={cand_out!r} (err={cand_err}), "
                        f"reference={ref_out!r} (err={ref_err})"
                    ),
                )

    reasons.append("passed_all_peer_review_gates")
    return PeerReviewVerdict(
        trust_tier="peer_reviewed",
        passed=True,
        reasons=tuple(reasons),
        counterexample=None,
    )


@dataclass(frozen=True)
class PitfallWarning:
    """A documented dead-end, numerical instability, or edge-case counterexample attached to a concept."""

    pitfall_id: str
    naive_pattern: str
    failure_mechanism: str
    counterexample_input: str
    verified_fix: str


@dataclass(frozen=True)
class ConceptBlock:
    """A unit of scientific or algorithmic knowledge stored in the Open-Science Concept Commons."""

    block_id: str
    cid: str
    sid: str
    fp: str
    name: str
    domain: str
    summary: str
    pseudocode: str
    invariants: tuple[str, ...]
    source_code: str
    param_types: tuple[str, ...]
    return_type: str
    dependencies: tuple[str, ...] = ()
    pitfalls: tuple[PitfallWarning, ...] = ()
    teleological_intent: str = ""
    dimensional_units: str = ""
    trust_tier: TrustTier = "probe_verified"
    contributor_agent: str = "agy-local"
    wave: int = 1
    created_at_ms: float = field(default_factory=lambda: time.time() * 1000.0)

    def render_for_agent(self, mode: ServingMode = "dual_card_with_pitfalls") -> str:
        """Render the concept in the requested representation modality."""
        sig = f"{self.name}({', '.join(self.param_types)}) -> {self.return_type}"
        if mode == "raw_code":
            return f"# Concept: {sig} [{self.cid}]\n{self.source_code.strip()}"

        intent_line = f"\n  Intent    : {self.teleological_intent}" if self.teleological_intent else ""
        units_line = f"\n  Units     : {self.dimensional_units}" if self.dimensional_units else ""

        if mode == "pseudocode":
            inv_str = "; ".join(self.invariants) if self.invariants else "total on domain"
            return (
                f"CONCEPT {sig} [{self.cid}]\n"
                f"  Summary   : {self.summary}{intent_line}{units_line}\n"
                f"  Algorithm : {self.pseudocode}\n"
                f"  Invariants: {inv_str}"
            )

        if mode == "dual_card":
            inv_str = "; ".join(self.invariants) if self.invariants else "total on domain"
            return (
                f"CONCEPT {sig} [{self.cid} | tier={self.trust_tier}]\n"
                f"  Summary   : {self.summary}{intent_line}{units_line}\n"
                f"  Algorithm : {self.pseudocode}\n"
                f"  Invariants: {inv_str}\n"
                f"  Reference :\n{self.source_code.strip()}"
            )

        inv_str = "; ".join(self.invariants) if self.invariants else "total on domain"
        pitfall_lines = ""
        if self.pitfalls:
            warnings = [
                f"    - AVOID '{p.naive_pattern}': {p.failure_mechanism} "
                f"(fails at {p.counterexample_input}) -> {p.verified_fix}"
                for p in self.pitfalls
            ]
            pitfall_lines = "\n  Refuted Pitfalls / Dead-Ends:\n" + "\n".join(warnings)
        return (
            f"CONCEPT {sig} [{self.cid} | tier={self.trust_tier}]\n"
            f"  Summary   : {self.summary}{intent_line}{units_line}\n"
            f"  Algorithm : {self.pseudocode}\n"
            f"  Invariants: {inv_str}{pitfall_lines}\n"
            f"  Reference :\n{self.source_code.strip()}"
        )

    def estimate_tokens(self, mode: ServingMode = "dual_card_with_pitfalls") -> int:
        rendered = self.render_for_agent(mode=mode)
        return max(1, round(len(rendered) / 3.8))


@dataclass(frozen=True)
class RetrievalHit:
    """A ranked concept retrieval result with behavioral probe verification status."""

    block: ConceptBlock
    score: float
    io_probe_verified: bool
    bundle_blocks: tuple[ConceptBlock, ...]
    latency_ms: float


@dataclass
class PendingProposal:
    """A staged agent research proposal awaiting human review or additional test verification."""

    proposal_id: str
    func_name: str
    domain: str
    summary: str
    pseudocode: str
    invariants: list[str]
    source_code: str
    teleological_intent: str
    dimensional_units: str
    pitfalls: list[dict[str, str]]
    contributor_agent: str
    stage_reason: str
    sid: str
    fp: str
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ"))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OpenCommonsStore:
    """Fast, meaning-indexed Open-Science Concept Commons with G-Set CRDT merge."""

    def __init__(self) -> None:
        self.blocks_by_id: dict[str, ConceptBlock] = {}
        self.cid_by_fp: dict[str, str] = {}
        self.canonical_block_by_cid: dict[str, str] = {}
        self.quarantined_ids: set[str] = set()
        self.naming_collisions: list[tuple[str, str, str]] = []
        self._token_index: dict[str, set[str]] = {}
        self._name_to_cids: dict[str, set[str]] = {}
        self._compiled_cache: dict[str, Callable[..., Any]] = {}
        self._block_id_by_sid: dict[str, str] = {}
        self.peer_review_executions: int = 0
        self.sid_dedup_hits: int = 0

    def ingest(
        self,
        source_code: str,
        domain: str = "general",
        summary: str = "",
        pseudocode: str = "",
        invariants: tuple[str, ...] = (),
        pitfalls: tuple[PitfallWarning, ...] = (),
        teleological_intent: str = "",
        dimensional_units: str = "",
        contributor_agent: str = "agy-local",
        wave: int = 1,
        run_peer_review: bool = False,
        contract_io: list[tuple[tuple[Any, ...], Any]] | None = None,
        invariant_fn: Callable[[tuple[Any, ...], Any], bool] | None = None,
        reference_fn: Callable[..., Any] | None = None,
        allow_mutation: bool = False,
    ) -> tuple[ConceptBlock, FingerprintResult, PeerReviewVerdict | None]:
        """Ingest a native Python concept block in <5ms, computing sid and behavioral fp."""
        sid, func_name, _param_types, _return_type, _deps = compute_structural_sid(source_code)
        block_id = f"{func_name}@{sid[4:12]}"

        if sid in self._block_id_by_sid and self._block_id_by_sid[sid] in self.blocks_by_id:
            existing_block = self.blocks_by_id[self._block_id_by_sid[sid]]
            if not contract_io and not invariant_fn and not reference_fn:
                self.sid_dedup_hits += 1
                cached_fp = FingerprintResult(
                    sid=existing_block.sid,
                    fp=existing_block.fp,
                    func_name=existing_block.name,
                    param_types=existing_block.param_types,
                    return_type=existing_block.return_type,
                    dependencies=existing_block.dependencies,
                    valid_probe_count=16,
                    total_probe_count=16,
                    mutates_arguments=False,
                    is_constant_output=False,
                    elapsed_ms=0.05,
                    sample_outputs=(),
                )
                cached_verdict = (
                    PeerReviewVerdict(
                        trust_tier=existing_block.trust_tier,
                        passed=(existing_block.trust_tier != "quarantined"),
                        reasons=(),
                    )
                    if run_peer_review
                    else None
                )
                return existing_block, cached_fp, cached_verdict

        extra_globals = self._build_dependency_globals()
        fp_res = fingerprint_function(source_code, extra_globals=extra_globals)

        if fp_res.fp in self.cid_by_fp:
            cid = self.cid_by_fp[fp_res.fp]
        else:
            cid = f"cid:{fp_res.fp.split(':', 1)[1]}"
            self.cid_by_fp[fp_res.fp] = cid

        auto_summary = summary or f"{fp_res.func_name}({', '.join(fp_res.param_types)}) -> {fp_res.return_type}"
        auto_pseudo = pseudocode or f"Compute {fp_res.func_name} over {', '.join(fp_res.param_types)}"

        verdict: PeerReviewVerdict | None = None
        trust_tier: TrustTier = "probe_verified"
        if run_peer_review:
            self.peer_review_executions += 1
            verdict = run_async_peer_review(
                source_code=source_code,
                fp_result=fp_res,
                contract_io=contract_io,
                invariant_fn=invariant_fn,
                reference_fn=reference_fn,
                allow_mutation=allow_mutation,
                extra_globals=extra_globals,
            )
            trust_tier = verdict.trust_tier

        block = ConceptBlock(
            block_id=block_id,
            cid=cid,
            sid=fp_res.sid,
            fp=fp_res.fp,
            name=fp_res.func_name,
            domain=domain,
            summary=auto_summary,
            pseudocode=auto_pseudo,
            invariants=invariants,
            source_code=source_code,
            param_types=fp_res.param_types,
            return_type=fp_res.return_type,
            dependencies=fp_res.dependencies,
            pitfalls=pitfalls,
            teleological_intent=teleological_intent,
            dimensional_units=dimensional_units,
            trust_tier=trust_tier,
            contributor_agent=contributor_agent,
            wave=wave,
        )

        self._block_id_by_sid[sid] = block_id
        self._insert_block(block)
        return block, fp_res, verdict

    def report_counterexample(
        self,
        block_id: str,
        failing_args: tuple[Any, ...],
        expected_output: Any | None = None,
        invariant_fn: Callable[[tuple[Any, ...], Any], bool] | None = None,
        failure_description: str = "unprobed_17th_input_counterexample",
        verified_fix: str = "Handle boundary input explicitly before evaluation",
    ) -> bool:
        """Monotonic Counterexample Ratchet against the (K+1)-th Input Probe Blind Spot."""
        if block_id not in self.blocks_by_id:
            return False
        block = self.blocks_by_id[block_id]
        fn = self._get_compiled_fn(block_id)
        refuted = False
        if fn is None:
            refuted = True
        else:
            try:
                out = fn(*copy.deepcopy(failing_args))
                if expected_output is not None and _canonical_serialize(out) != _canonical_serialize(expected_output):
                    refuted = True
                if invariant_fn is not None and not invariant_fn(failing_args, out):
                    refuted = True
            except Exception:  # noqa: BLE001
                refuted = True

        if not refuted:
            return False

        new_pitfall = PitfallWarning(
            pitfall_id=f"ratchet_{len(block.pitfalls) + 1}",
            naive_pattern=block.name,
            failure_mechanism=failure_description,
            counterexample_input=repr(failing_args),
            verified_fix=verified_fix,
        )
        quarantined_block = replace(
            block,
            trust_tier="quarantined",
            pitfalls=(*block.pitfalls, new_pitfall),
        )
        self.blocks_by_id[block_id] = quarantined_block
        self.quarantined_ids.add(block_id)

        if self.canonical_block_by_cid.get(block.cid) == block_id:
            del self.canonical_block_by_cid[block.cid]
            for cand_id, cand in self.blocks_by_id.items():
                if cand.cid == block.cid and cand_id not in self.quarantined_ids:
                    self.canonical_block_by_cid[block.cid] = cand_id
                    break
        return True

    def _build_dependency_globals(self) -> dict[str, Any]:
        ns: dict[str, Any] = {"__builtins__": SAFE_EXEC_BUILTINS, "math": math}
        for bid in self.canonical_block_by_cid.values():
            if bid in self.quarantined_ids:
                continue
            b = self.blocks_by_id[bid]
            fn = self._get_compiled_fn(bid)
            if fn is not None:
                ns[b.name] = fn
        return ns

    def _get_compiled_fn(self, block_id: str) -> Callable[..., Any] | None:
        """Lazily compiles and caches a concept block inside the restricted SAFE_EXEC_BUILTINS sandbox."""
        if block_id in self._compiled_cache:
            return self._compiled_cache[block_id]
        block = self.blocks_by_id.get(block_id)
        if block is None or block.trust_tier == "quarantined":
            return None
        with contextlib.suppress(Exception):
            verify_ast_safety_static(block.source_code)
            ns: dict[str, Any] = {"__builtins__": SAFE_EXEC_BUILTINS, "math": math}
            for dep_name in block.dependencies:
                dep_cids = self._name_to_cids.get(dep_name, set())
                for dcid in dep_cids:
                    dbid = self.canonical_block_by_cid.get(dcid)
                    if dbid and dbid != block_id and dbid not in self.quarantined_ids:
                        dep_fn = self._get_compiled_fn(dbid)
                        if dep_fn is not None:
                            ns[dep_name] = dep_fn
                        break
            exec(compile(block.source_code, f"<{block.block_id}>", "exec"), ns)  # noqa: S102
            if block.name in ns and callable(ns[block.name]):
                self._compiled_cache[block.block_id] = ns[block.name]
                return ns[block.name]
        return None

    def _insert_block(self, block: ConceptBlock) -> None:
        self.blocks_by_id[block.block_id] = block
        if block.trust_tier == "quarantined":
            self.quarantined_ids.add(block.block_id)
            return

        self.cid_by_fp.setdefault(block.fp, block.cid)

        existing_cids = self._name_to_cids.setdefault(block.name, set())
        for existing_cid in existing_cids:
            if existing_cid != block.cid:
                pair = (block.name, min(existing_cid, block.cid), max(existing_cid, block.cid))
                if pair not in self.naming_collisions:
                    self.naming_collisions.append(pair)
        existing_cids.add(block.cid)

        if block.cid not in self.canonical_block_by_cid:
            self.canonical_block_by_cid[block.cid] = block.block_id
        else:
            curr = self.blocks_by_id[self.canonical_block_by_cid[block.cid]]
            if (block.trust_tier == "peer_reviewed" and curr.trust_tier != "peer_reviewed") or (
                len(block.pitfalls) > len(curr.pitfalls)
            ):
                self.canonical_block_by_cid[block.cid] = block.block_id

        index_text = f"{block.name} {block.domain} {block.summary} {block.pseudocode}"
        tokens, _ = tokenize_concept_query(index_text)
        for tok in tokens:
            self._token_index.setdefault(tok, set()).add(block.cid)

    def compute_merkle_manifest(self) -> dict[str, Any]:
        """Compute deterministic Merkle root and 16 hex-prefix bucket hashes for O(delta) network sync."""
        buckets: dict[str, list[str]] = {f"{i:x}": [] for i in range(16)}
        for bid in sorted(self.blocks_by_id.keys()):
            blk = self.blocks_by_id[bid]
            q_flag = "Q" if bid in self.quarantined_ids else "A"
            entry = f"{bid}:{blk.sid}:{blk.cid}:{q_flag}:{len(blk.pitfalls)}"
            prefix = blk.sid[4] if len(blk.sid) > 4 and blk.sid[4] in buckets else "0"
            buckets[prefix].append(entry)

        bucket_hashes: dict[str, str] = {}
        for k, items in buckets.items():
            bucket_hashes[k] = hashlib.sha256("\n".join(items).encode("utf-8")).hexdigest()[:16]

        root_material = "|".join(f"{k}:{bucket_hashes[k]}" for k in sorted(bucket_hashes.keys()))
        merkle_root = hashlib.sha256(root_material.encode("utf-8")).hexdigest()[:24]
        return {
            "schema_version": "1.0.0",
            "merkle_root": merkle_root,
            "total_blocks": len(self.blocks_by_id),
            "active_concepts_cid": len(self.canonical_block_by_cid),
            "quarantined_blocks": len(self.quarantined_ids),
            "bucket_hashes": bucket_hashes,
        }

    def find(
        self,
        query: str,
        param_types: tuple[str, ...] | None = None,
        return_type: str | None = None,
        io_examples: list[tuple[tuple[Any, ...], Any]] | None = None,
        top_k: int = 3,
        include_bundle: bool = True,
        min_score: float = 2.5,
        enforce_numbered_discrimination: bool = True,
    ) -> list[RetrievalHit]:
        """Retrieve verified concepts matching query, signature, and live I/O probe assertions in <5ms."""
        t0 = time.perf_counter()
        q_tokens, q_num_stems = tokenize_concept_query(query)

        candidate_cids: set[str] = set()
        for tok in q_tokens:
            if tok in self._token_index:
                candidate_cids.update(self._token_index[tok])
        if not candidate_cids:
            candidate_cids = set(self.canonical_block_by_cid.keys())

        scored: list[tuple[float, bool, ConceptBlock]] = []
        for cid in candidate_cids:
            bid = self.canonical_block_by_cid.get(cid)
            if not bid or bid in self.quarantined_ids:
                continue
            block = self.blocks_by_id[bid]

            name_tokens, b_num_stems = tokenize_concept_query(f"{block.name} {block.summary}")
            doc_tokens, _ = tokenize_concept_query(f"{block.domain} {block.pseudocode}")

            if enforce_numbered_discrimination and q_num_stems:
                numbered_conflict = False
                for stem, q_nums in q_num_stems.items():
                    if stem in b_num_stems and b_num_stems[stem] != q_nums:
                        numbered_conflict = True
                        break
                if numbered_conflict:
                    continue

            name_overlap = len(q_tokens & name_tokens)
            doc_overlap = len(q_tokens & doc_tokens)
            score = 3.0 * name_overlap + 1.0 * doc_overlap

            if block.name.lower() in query.lower():
                score += 4.0

            if param_types is not None:
                if len(param_types) != len(block.param_types):
                    continue
                type_matches = sum(
                    1
                    for qp, bp in zip(param_types, block.param_types, strict=False)
                    if qp.lower() == bp.lower() or "any" in (qp.lower(), bp.lower())
                )
                score += 1.5 * type_matches

            if return_type is not None and return_type.lower() != "any":
                if block.return_type.lower() != return_type.lower() and block.return_type.lower() != "any":
                    continue
                score += 2.0

            io_verified = False
            if io_examples:
                fn = self._get_compiled_fn(block.block_id)
                if fn is None:
                    continue
                all_passed = True
                for args, expected_out in io_examples:
                    try:
                        actual_out = fn(*copy.deepcopy(args))
                        if _canonical_serialize(actual_out) != _canonical_serialize(expected_out):
                            all_passed = False
                            break
                    except Exception:  # noqa: BLE001
                        all_passed = False
                        break
                if not all_passed:
                    continue
                io_verified = True
                score += 10.0

            if score >= min_score:
                scored.append((score, io_verified, block))

        scored.sort(key=lambda item: (item[1], item[0]), reverse=True)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        hits: list[RetrievalHit] = []
        for score, io_ver, block in scored[:top_k]:
            bundle = self._resolve_bundle(block) if include_bundle else (block,)
            hits.append(
                RetrievalHit(
                    block=block,
                    score=round(score, 3),
                    io_probe_verified=io_ver,
                    bundle_blocks=bundle,
                    latency_ms=round(elapsed_ms, 4),
                )
            )
        return hits

    def _resolve_bundle(self, root_block: ConceptBlock) -> tuple[ConceptBlock, ...]:
        visited: set[str] = set()
        ordered: list[ConceptBlock] = []

        def _dfs(b: ConceptBlock) -> None:
            if b.cid in visited:
                return
            visited.add(b.cid)
            for dep_name in b.dependencies:
                dep_cids = self._name_to_cids.get(dep_name, set())
                for dcid in dep_cids:
                    dbid = self.canonical_block_by_cid.get(dcid)
                    if dbid and dbid not in self.quarantined_ids:
                        _dfs(self.blocks_by_id[dbid])
                        break
            ordered.append(b)

        _dfs(root_block)
        return tuple(ordered)

    def merge(self, other: OpenCommonsStore) -> OpenCommonsStore:
        """State-based G-Set join-semilattice CRDT merge (L_self ⊔ L_other)."""
        merged = OpenCommonsStore()
        all_blocks = sorted(
            [*self.blocks_by_id.values(), *other.blocks_by_id.values()],
            key=lambda b: (b.block_id, b.cid),
        )
        for b in all_blocks:
            if b.block_id not in merged.blocks_by_id or (
                b.trust_tier == "quarantined" and b.block_id not in merged.quarantined_ids
            ):
                merged._insert_block(b)
        return merged


FOUNDATIONAL_SEED_BLOCKS: list[dict[str, object]] = [
    {
        "domain": "numerical_analysis",
        "summary": "Shifted numerically stable LogSumExp preventing exp(x) overflow for x >= 710",
        "pseudocode": "1. m = max(xs); 2. return m + log(sum(exp(x - m) for x in xs))",
        "invariants": ("LSE(x + c) == LSE(x) + c", "max(xs) <= LSE(xs) <= max(xs) + log(N)"),
        "teleological_intent": "Log-partition function free-energy normalization without IEEE-754 overflow",
        "dimensional_units": "logits: nats -> nats",
        "pitfalls": (
            PitfallWarning(
                pitfall_id="lse_overflow",
                naive_pattern="math.log(sum(math.exp(x) for x in xs))",
                failure_mechanism="OverflowError when max(xs) >= 710.0",
                counterexample_input="[1000.0, 1001.0, 1002.0]",
                verified_fix="Subtract m = max(xs) before exponentiating",
            ),
        ),
        "source_code": """
def stable_logsumexp(xs: list[float]) -> float:
    if not xs:
        return 0.0
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))
""",
    },
    {
        "domain": "numerical_analysis",
        "summary": "Welford single-pass online variance avoiding subtractive cancellation at kappa >= 10^8",
        "pseudocode": "1. Track running mean and M2 = sum((x - mean_old)*(x - mean_new)); 2. return M2 / n",
        "invariants": ("Var(X + c) == Var(X)", "Var(X) >= 0"),
        "teleological_intent": "Orthogonal running residual accumulation preserving mantissa bits on shifted streams",
        "dimensional_units": "xs: [U] -> U^2",
        "pitfalls": (
            PitfallWarning(
                pitfall_id="variance_cancellation",
                naive_pattern="sum(x*x)/n - (sum(x)/n)**2",
                failure_mechanism="Severe precision loss from subtractive cancellation when kappa >= 10^8",
                counterexample_input="[1e12 + 1.0, 1e12 + 2.0, 1e12 + 3.0]",
                verified_fix="Use Welford online delta * delta2 update",
            ),
        ),
        "source_code": """
def welford_variance(xs: list[float]) -> float:
    if not xs:
        return 0.0
    count = 0
    mean = 0.0
    m2 = 0.0
    for x in xs:
        count += 1
        delta = x - mean
        mean += delta / count
        delta2 = x - mean
        m2 += delta * delta2
    return m2 / count
""",
    },
    {
        "domain": "hamiltonian_dynamics",
        "summary": "Symplectic Velocity Verlet / Leapfrog step preserving phase-space volume (det J == 1)",
        "pseudocode": "1. v_half = v - 0.5*dt*k*x; 2. x_new = x + dt*v_half; 3. v_new = v_half - 0.5*dt*k*x_new",
        "invariants": ("det(Jacobian) == 1.0", "bounded energy oscillation O(dt^2)"),
        "teleological_intent": "Liouville theorem phase-space volume conservation in Hamiltonian mechanics",
        "dimensional_units": "x: m, v: m/s, k: 1/s^2, dt: s -> tuple[m, m/s]",
        "pitfalls": (
            PitfallWarning(
                pitfall_id="euler_energy_drift",
                naive_pattern="x_new = x + dt*v; v_new = v - dt*k*x",
                failure_mechanism="Explicit Euler inflates phase-space area det J = 1 + k*dt^2 > 1",
                counterexample_input="(1.0, 0.0, 1.0, 0.1) over 1000 steps",
                verified_fix="Use half-step kick-drift-kick symplectic update",
            ),
        ),
        "source_code": """
def symplectic_leapfrog_step(x: float, v: float, k: float, dt: float) -> tuple[float, float]:
    v_half = v - 0.5 * dt * k * x
    x_new = x + dt * v_half
    v_new = v_half - 0.5 * dt * k * x_new
    return (x_new, v_new)
""",
    },
    {
        "domain": "cryptography_and_framing",
        "summary": "32-bit FNV-1a deterministic byte payload hash with standard offset basis 0x811c9dc5",
        "pseudocode": "1. h = 0x811c9dc5; 2. for b in data: h = ((h ^ b) * 0x01000193) & 0xffffffff",
        "invariants": ("0 <= h <= 0xffffffff", "avalanche on single-bit flip"),
        "teleological_intent": "Non-cryptographic uniform hash distribution over byte frames",
        "dimensional_units": "data: list[int] -> uint32",
        "pitfalls": (
            PitfallWarning(
                pitfall_id="wrong_fnv_offset",
                naive_pattern="h = 0",
                failure_mechanism="Zero initialization collapses all-zero prefixes to hash 0",
                counterexample_input="[0, 0, 0]",
                verified_fix="Initialize with FNV-1a 32-bit offset basis 0x811c9dc5",
            ),
        ),
        "source_code": """
def fnv1a_32_ints(data: list[int]) -> int:
    h = 0x811C9DC5
    for b in data:
        h = ((h ^ (b & 0xFF)) * 0x01000193) & 0xFFFFFFFF
    return h
""",
    },
]


def serialize_block(block: ConceptBlock) -> dict[str, Any]:
    d = asdict(block)
    d["invariants"] = list(block.invariants)
    d["param_types"] = list(block.param_types)
    d["dependencies"] = list(block.dependencies)
    d["pitfalls"] = [asdict(p) for p in block.pitfalls]
    return d


def deserialize_block(data: dict[str, Any]) -> ConceptBlock:
    pitfalls = tuple(PitfallWarning(**p) for p in data.get("pitfalls", []))
    return ConceptBlock(
        block_id=data["block_id"],
        cid=data["cid"],
        sid=data["sid"],
        fp=data["fp"],
        name=data["name"],
        domain=data.get("domain", "general"),
        summary=data.get("summary", ""),
        pseudocode=data.get("pseudocode", ""),
        invariants=tuple(data.get("invariants", [])),
        source_code=data["source_code"],
        param_types=tuple(data.get("param_types", [])),
        return_type=data.get("return_type", "Any"),
        dependencies=tuple(data.get("dependencies", [])),
        pitfalls=pitfalls,
        teleological_intent=data.get("teleological_intent", ""),
        dimensional_units=data.get("dimensional_units", ""),
        trust_tier=data.get("trust_tier", "probe_verified"),
        contributor_agent=data.get("contributor_agent", "agy-local"),
        wave=int(data.get("wave", 1)),
        created_at_ms=float(data.get("created_at_ms", 0.0)),
    )


def store_to_dict(store: OpenCommonsStore) -> dict[str, Any]:
    blocks = [
        serialize_block(store.blocks_by_id[bid])
        for bid in sorted(store.blocks_by_id.keys())
    ]
    return {
        "schema_version": "1.0.0",
        "blocks": blocks,
        "quarantined_ids": sorted(store.quarantined_ids),
        "peer_review_executions": store.peer_review_executions,
        "sid_dedup_hits": store.sid_dedup_hits,
    }


def store_from_dict(
    payload: dict[str, Any],
    *,
    verify_network_safety: bool = False,
) -> OpenCommonsStore:
    store = OpenCommonsStore()
    q_set = set(payload.get("quarantined_ids", []))
    for b_dict in payload.get("blocks", []):
        if not isinstance(b_dict, dict):
            continue
        if verify_network_safety:
            try:
                src = str(b_dict.get("source_code", ""))
                computed_sid, _, _, _, _ = compute_structural_sid(src)
                if b_dict.get("sid") and str(b_dict["sid"]) != computed_sid:
                    continue
                b_dict = dict(b_dict)
                b_dict["sid"] = computed_sid
            except (UnsafeConceptCodeError, SyntaxError, ValueError):
                continue
        if b_dict.get("block_id") in q_set:
            b_dict = dict(b_dict)
            b_dict["trust_tier"] = "quarantined"
        block = deserialize_block(b_dict)
        store._block_id_by_sid[block.sid] = block.block_id
        store._insert_block(block)
    store.quarantined_ids.update(q_set & set(store.blocks_by_id.keys()))
    store.peer_review_executions = int(payload.get("peer_review_executions", 0))
    store.sid_dedup_hits = int(payload.get("sid_dedup_hits", 0))
    return store


def seed_default_scientific_primitives(store: OpenCommonsStore) -> int:
    """Seed an OpenCommonsStore instance in-place with the 4 foundational scientific primitives."""
    count = 0
    for item in FOUNDATIONAL_SEED_BLOCKS:
        store.ingest(
            source_code=str(item["source_code"]),
            domain=str(item["domain"]),
            summary=str(item["summary"]),
            pseudocode=str(item["pseudocode"]),
            invariants=tuple(item["invariants"]),  # type: ignore[arg-type]
            pitfalls=tuple(item["pitfalls"]),  # type: ignore[arg-type]
            teleological_intent=str(item["teleological_intent"]),
            dimensional_units=str(item["dimensional_units"]),
            run_peer_review=True,
        )
        count += 1
    return count


def load_persistent_store(
    path: Path | str | None = None,
    *,
    auto_seed_if_empty: bool = True,
) -> OpenCommonsStore:
    """Load an OpenCommonsStore from disk (auto-seeding the 4 foundational primitives on first use)."""
    target = Path(path).expanduser().resolve() if path else DEFAULT_STORE_PATH.resolve()
    if not target.is_file():
        store = OpenCommonsStore()
        if auto_seed_if_empty:
            seed_default_scientific_primitives(store)
            save_persistent_store(store, target, crdt_merge_existing=False)
        return store
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
        store = store_from_dict(raw)
        if auto_seed_if_empty and not store.blocks_by_id:
            seed_default_scientific_primitives(store)
            save_persistent_store(store, target, crdt_merge_existing=False)
        return store
    except (OSError, ValueError, KeyError, TypeError):
        return OpenCommonsStore()


def save_persistent_store(
    store: OpenCommonsStore,
    path: Path | str | None = None,
    crdt_merge_existing: bool = True,
) -> OpenCommonsStore:
    """Atomically save an OpenCommonsStore to disk, merging with concurrent updates via CRDT `⊔`."""
    target = Path(path).expanduser().resolve() if path else DEFAULT_STORE_PATH.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.with_suffix(target.suffix + ".lock")

    with open(lock_path, "w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            final_store = store
            if crdt_merge_existing and target.is_file():
                try:
                    on_disk = store_from_dict(json.loads(target.read_text(encoding="utf-8")))
                    final_store = on_disk.merge(store)
                    final_store.peer_review_executions = max(
                        on_disk.peer_review_executions, store.peer_review_executions
                    )
                    final_store.sid_dedup_hits = max(on_disk.sid_dedup_hits, store.sid_dedup_hits)
                except (OSError, ValueError, KeyError, TypeError):
                    final_store = store

            serialized = json.dumps(store_to_dict(final_store), indent=2, sort_keys=True)
            fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=".oc_tmp_", suffix=".json")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as tmp_f:
                    tmp_f.write(serialized + "\n")
                os.replace(tmp_name, target)
            finally:
                if os.path.exists(tmp_name):
                    os.unlink(tmp_name)
            return final_store
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _decode_http_body(raw_bytes: bytes, content_encoding: str | None) -> dict[str, Any]:
    if content_encoding and "gzip" in content_encoding.lower():
        raw_bytes = gzip.decompress(raw_bytes)
    return json.loads(raw_bytes.decode("utf-8"))


def pull_from_remote_hub(
    hub_url: str | None = None,
    *,
    store_path: Path | str | None = None,
    timeout: float = 8.0,
    force: bool = False,
) -> dict[str, Any]:
    """Pull verified OpenCommons updates over the network using Merkle-root & delta sync."""
    t0 = time.perf_counter()
    target_url = (hub_url or DEFAULT_HUB_URL).rstrip("/")
    local_store = load_persistent_store(store_path)
    before_blocks = len(local_store.blocks_by_id)
    before_cids = len(local_store.canonical_block_by_cid)
    local_manifest = local_store.compute_merkle_manifest()

    headers: dict[str, str] = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "User-Agent": "kardashev-opencommons/1.0",
    }
    if not force:
        headers["If-None-Match"] = f'"{local_manifest["merkle_root"]}"'

    # If pointing to a live OpenCommons Hub (as opposed to a static .json file), try /v1/sync-delta first
    if not target_url.endswith(".json"):
        delta_endpoint = f"{target_url}/v1/sync-delta"
        req_payload = {
            "mode": "pull_only",
            "merkle_root": "" if force else local_manifest["merkle_root"],
            "known_block_ids": [] if force else sorted(local_store.blocks_by_id.keys()),
            "known_quarantined_ids": sorted(local_store.quarantined_ids),
        }
        try:
            body_bytes = gzip.compress(json.dumps(req_payload).encode("utf-8"))
            req = urllib.request.Request(
                delta_endpoint,
                data=body_bytes,
                headers={
                    **headers,
                    "Content-Type": "application/json",
                    "Content-Encoding": "gzip",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                resp_data = _decode_http_body(resp.read(), resp.headers.get("Content-Encoding"))
                delta_store = store_from_dict(resp_data, verify_network_safety=True)
                merged = local_store.merge(delta_store)
                saved = save_persistent_store(merged, store_path)
                elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                return {
                    "ok": True,
                    "protocol": "delta_sync",
                    "hub_url": target_url,
                    "up_to_date": resp_data.get("up_to_date", False),
                    "new_blocks_pulled": len(saved.blocks_by_id) - before_blocks,
                    "active_concepts_cid": len(saved.canonical_block_by_cid),
                    "new_cids_added": len(saved.canonical_block_by_cid) - before_cids,
                    "merkle_root": saved.compute_merkle_manifest()["merkle_root"],
                    "elapsed_ms": elapsed_ms,
                }
        except urllib.error.HTTPError as http_err:
            if http_err.code == 304:
                return {
                    "ok": True,
                    "protocol": "etag_304",
                    "hub_url": target_url,
                    "up_to_date": True,
                    "new_blocks_pulled": 0,
                    "active_concepts_cid": before_cids,
                    "merkle_root": local_manifest["merkle_root"],
                    "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                }
        except (urllib.error.URLError, OSError, ValueError):
            pass

    # Fallback: GET static store JSON (e.g. GitHub Raw CDN or /v1/store)
    store_url = target_url if target_url.endswith(".json") else f"{target_url}/v1/store"
    try:
        req = urllib.request.Request(store_url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            remote_dict = _decode_http_body(resp.read(), resp.headers.get("Content-Encoding"))
            remote_store = store_from_dict(remote_dict, verify_network_safety=True)
            merged = local_store.merge(remote_store)
            saved = save_persistent_store(merged, store_path)
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            return {
                "ok": True,
                "protocol": "static_cdn_get",
                "hub_url": store_url,
                "up_to_date": len(saved.blocks_by_id) == before_blocks,
                "new_blocks_pulled": len(saved.blocks_by_id) - before_blocks,
                "active_concepts_cid": len(saved.canonical_block_by_cid),
                "new_cids_added": len(saved.canonical_block_by_cid) - before_cids,
                "merkle_root": saved.compute_merkle_manifest()["merkle_root"],
                "elapsed_ms": elapsed_ms,
            }
    except urllib.error.HTTPError as http_err:
        if http_err.code == 304:
            return {
                "ok": True,
                "protocol": "etag_304",
                "hub_url": store_url,
                "up_to_date": True,
                "new_blocks_pulled": 0,
                "active_concepts_cid": before_cids,
                "merkle_root": local_manifest["merkle_root"],
                "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            }
        return {"ok": False, "hub_url": store_url, "error": f"HTTP {http_err.code}: {http_err.reason}"}
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return {"ok": False, "hub_url": store_url, "error": str(exc)}


def push_to_remote_hub(
    hub_url: str | None = None,
    *,
    store_path: Path | str | None = None,
    timeout: float = 10.0,
) -> dict[str, Any]:
    """Push local peer-reviewed OpenCommons discoveries & counterexamples to a remote CRDT Hub."""
    t0 = time.perf_counter()
    target_url = (hub_url or DEFAULT_HUB_URL).rstrip("/")
    local_store = load_persistent_store(store_path)
    local_manifest = local_store.compute_merkle_manifest()

    sync_endpoint = target_url if target_url.endswith("/v1/sync-delta") else f"{target_url}/v1/sync-delta"
    payload = {
        "mode": "bidirectional",
        "merkle_root": local_manifest["merkle_root"],
        "known_block_ids": sorted(local_store.blocks_by_id.keys()),
        "known_quarantined_ids": sorted(local_store.quarantined_ids),
        "blocks": [serialize_block(b) for b in local_store.blocks_by_id.values()],
        "quarantined_ids": sorted(local_store.quarantined_ids),
    }

    try:
        compressed = gzip.compress(json.dumps(payload).encode("utf-8"))
        req = urllib.request.Request(
            sync_endpoint,
            data=compressed,
            headers={
                "Content-Type": "application/json",
                "Content-Encoding": "gzip",
                "Accept-Encoding": "gzip",
                "User-Agent": "kardashev-opencommons/1.0",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp_data = _decode_http_body(resp.read(), resp.headers.get("Content-Encoding"))
            delta_down = store_from_dict(resp_data, verify_network_safety=True)
            merged = local_store.merge(delta_down)
            saved = save_persistent_store(merged, store_path)
            return {
                "ok": True,
                "hub_url": sync_endpoint,
                "accepted_by_hub": resp_data.get("accepted_count", 0),
                "rejected_unsafe_or_failed": resp_data.get("rejected_count", 0),
                "total_active_concepts": len(saved.canonical_block_by_cid),
                "merkle_root": saved.compute_merkle_manifest()["merkle_root"],
                "payload_bytes_sent": len(compressed),
                "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            }
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return {"ok": False, "hub_url": sync_endpoint, "error": str(exc)}


def process_hub_sync_request(
    req_payload: dict[str, Any],
    *,
    store_path: Path | str | None = None,
) -> dict[str, Any]:
    """Server-side handler for `/v1/sync-delta` that audits incoming blocks through the AST security gate
    + 5-gate adversarial peer review before merging into the Hub's CRDT store, and returns only the
    delta blocks missing on the client.
    """
    hub_store = load_persistent_store(store_path)
    hub_manifest = hub_store.compute_merkle_manifest()
    client_root = str(req_payload.get("merkle_root", ""))
    client_known_ids = set(req_payload.get("known_block_ids", []))
    mode = str(req_payload.get("mode", "bidirectional"))

    accepted_count = 0
    rejected_count = 0

    if mode == "bidirectional" and req_payload.get("blocks"):
        for b_dict in req_payload.get("blocks", []):
            if not isinstance(b_dict, dict):
                continue
            bid = str(b_dict.get("block_id", ""))
            if bid in hub_store.blocks_by_id and bid not in req_payload.get("quarantined_ids", []):
                continue
            try:
                src = str(b_dict.get("source_code", ""))
                verify_ast_safety_static(src)
                pitfalls = tuple(
                    PitfallWarning(**pf)
                    for pf in b_dict.get("pitfalls", [])
                    if isinstance(pf, dict)
                )
                blk, _fp, verdict = hub_store.ingest(
                    source_code=src,
                    domain=str(b_dict.get("domain", "general")),
                    summary=str(b_dict.get("summary", "")),
                    pseudocode=str(b_dict.get("pseudocode", "")),
                    invariants=tuple(str(x) for x in b_dict.get("invariants", [])),
                    pitfalls=pitfalls,
                    teleological_intent=str(b_dict.get("teleological_intent", "")),
                    dimensional_units=str(b_dict.get("dimensional_units", "")),
                    contributor_agent=str(b_dict.get("contributor_agent", "remote-agent")),
                    run_peer_review=True,
                )
                if b_dict.get("trust_tier") == "quarantined" or bid in req_payload.get("quarantined_ids", []):
                    hub_store.quarantined_ids.add(blk.block_id)
                    hub_store.blocks_by_id[blk.block_id] = replace(blk, trust_tier="quarantined")
                    accepted_count += 1
                elif verdict and verdict.passed:
                    accepted_count += 1
                else:
                    hub_store.blocks_by_id.pop(blk.block_id, None)
                    hub_store.quarantined_ids.discard(blk.block_id)
                    rejected_count += 1
            except (UnsafeConceptCodeError, SyntaxError, ValueError, TypeError):
                rejected_count += 1

        if accepted_count > 0:
            hub_store = save_persistent_store(hub_store, store_path)
            hub_manifest = hub_store.compute_merkle_manifest()

    if client_root == hub_manifest["merkle_root"] and accepted_count == 0:
        return {
            "schema_version": "1.0.0",
            "up_to_date": True,
            "merkle_root": hub_manifest["merkle_root"],
            "accepted_count": 0,
            "rejected_count": rejected_count,
            "blocks": [],
            "quarantined_ids": sorted(hub_store.quarantined_ids),
        }

    # Compute O(delta) missing blocks for client: ΔB_down = B_hub \ B_client
    delta_blocks = [
        serialize_block(hub_store.blocks_by_id[bid])
        for bid in sorted(hub_store.blocks_by_id.keys())
        if bid not in client_known_ids or bid in hub_store.quarantined_ids
    ]
    return {
        "schema_version": "1.0.0",
        "up_to_date": len(delta_blocks) == 0,
        "merkle_root": hub_manifest["merkle_root"],
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "blocks": delta_blocks,
        "quarantined_ids": sorted(hub_store.quarantined_ids),
    }


def create_opencommons_hub_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    store_path: Path | str | None = None,
) -> http.server.ThreadingHTTPServer:
    """Create a multi-threaded HTTP OpenCommons CRDT Gossip Hub server (`GET /v1/manifest`, `GET /v1/store`, `POST /v1/sync-delta`)."""

    class _HubHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_json(self, status_code: int, payload: dict[str, Any], etag: str | None = None) -> None:
            raw = json.dumps(payload).encode("utf-8")
            accept_enc = self.headers.get("Accept-Encoding", "")
            use_gzip = "gzip" in accept_enc.lower()
            body = gzip.compress(raw) if use_gzip else raw

            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            if use_gzip:
                self.send_header("Content-Encoding", "gzip")
            if etag:
                self.send_header("ETag", f'"{etag}"')
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            store = load_persistent_store(store_path)
            manifest = store.compute_merkle_manifest()
            client_etag = (self.headers.get("If-None-Match") or "").strip().strip('"')
            if client_etag and client_etag == manifest["merkle_root"]:
                self.send_response(304)
                self.end_headers()
                return

            if self.path.startswith("/v1/manifest"):
                self._send_json(200, manifest, etag=manifest["merkle_root"])
                return
            if self.path.startswith("/v1/store") or self.path.endswith(".json"):
                self._send_json(200, store_to_dict(store), etag=manifest["merkle_root"])
                return
            self._send_json(404, {"error": f"Unknown endpoint: {self.path}"})

        def do_POST(self) -> None:
            if not self.path.startswith("/v1/sync"):
                self._send_json(404, {"error": f"Unknown endpoint: {self.path}"})
                return
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length) if length > 0 else b"{}"
            try:
                req_data = _decode_http_body(raw_body, self.headers.get("Content-Encoding"))
                resp_payload = process_hub_sync_request(req_data, store_path=store_path)
                self._send_json(200, resp_payload, etag=resp_payload.get("merkle_root"))
            except (ValueError, KeyError, OSError) as exc:
                self._send_json(400, {"error": str(exc)})

    return http.server.ThreadingHTTPServer((host, port), _HubHandler)


def load_pending_proposals(pending_path: Path | str | None = None) -> list[PendingProposal]:
    """Loads the pending review queue of agent-proposed concepts."""
    target = Path(pending_path).expanduser().resolve() if pending_path else DEFAULT_PENDING_PATH.resolve()
    if not target.is_file():
        return []
    try:
        items = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(items, list):
            return []
        return [PendingProposal(**it) for it in items if isinstance(it, dict)]
    except (OSError, ValueError, TypeError):
        return []


def save_pending_proposals(
    proposals: list[PendingProposal],
    pending_path: Path | str | None = None,
) -> Path:
    """Atomically saves the pending review queue."""
    target = Path(pending_path).expanduser().resolve() if pending_path else DEFAULT_PENDING_PATH.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = [p.to_dict() for p in proposals]
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target


def enqueue_pending_proposal(
    proposal: PendingProposal,
    pending_path: Path | str | None = None,
) -> PendingProposal:
    """Adds or updates a proposal in the pending review queue (deduplicated by structural `sid`)."""
    existing = load_pending_proposals(pending_path)
    for p in existing:
        if p.sid == proposal.sid:
            return p
    existing.append(proposal)
    save_pending_proposals(existing, pending_path)
    return proposal


def approve_pending_proposal(
    proposal_id: str,
    *,
    store_path: Path | str | None = None,
    pending_path: Path | str | None = None,
) -> dict[str, Any]:
    """Approves a pending proposal by ID (or prefix) and ingests it into the persistent OpenCommons store."""
    proposals = load_pending_proposals(pending_path)
    matched: PendingProposal | None = None
    remaining: list[PendingProposal] = []
    for p in proposals:
        if matched is None and (p.proposal_id == proposal_id or p.func_name == proposal_id or p.proposal_id.startswith(proposal_id)):
            matched = p
        else:
            remaining.append(p)

    if matched is None:
        return {"approved": False, "error": f"Proposal '{proposal_id}' not found in pending queue."}

    store = load_persistent_store(store_path)
    pitfalls = tuple(PitfallWarning(**pf) for pf in matched.pitfalls)
    block, _fp_res, _verdict = store.ingest(
        source_code=matched.source_code,
        domain=matched.domain,
        summary=matched.summary,
        pseudocode=matched.pseudocode,
        invariants=tuple(matched.invariants),
        pitfalls=pitfalls,
        teleological_intent=matched.teleological_intent,
        dimensional_units=matched.dimensional_units,
        contributor_agent=matched.contributor_agent,
        run_peer_review=False,
    )
    # Explicitly elevate human-approved block to peer_reviewed
    elevated = replace(block, trust_tier="peer_reviewed")
    store.blocks_by_id[block.block_id] = elevated
    store.quarantined_ids.discard(block.block_id)
    store.canonical_block_by_cid[block.cid] = block.block_id

    merged = save_persistent_store(store, store_path)
    save_pending_proposals(remaining, pending_path)
    return {
        "approved": True,
        "proposal_id": matched.proposal_id,
        "block_id": elevated.block_id,
        "cid": elevated.cid,
        "trust_tier": elevated.trust_tier,
        "remaining_pending": len(remaining),
        "total_active_concepts": len(merged.canonical_block_by_cid),
    }


def reject_pending_proposal(
    proposal_id: str,
    *,
    pending_path: Path | str | None = None,
) -> dict[str, Any]:
    """Removes a proposal from the pending review queue."""
    proposals = load_pending_proposals(pending_path)
    removed: PendingProposal | None = None
    remaining: list[PendingProposal] = []
    for p in proposals:
        if removed is None and (p.proposal_id == proposal_id or p.func_name == proposal_id or p.proposal_id.startswith(proposal_id)):
            removed = p
        else:
            remaining.append(p)
    if removed is None:
        return {"rejected": False, "error": f"Proposal '{proposal_id}' not found in pending queue."}
    save_pending_proposals(remaining, pending_path)
    return {
        "rejected": True,
        "proposal_id": removed.proposal_id,
        "func_name": removed.func_name,
        "remaining_pending": len(remaining),
    }


def execute_commons_find(
    arguments: dict[str, Any],
    store_path: Path | str | None = None,
) -> dict[str, Any]:
    """Execute the `commons_find` (or `find`) tool against the persistent OpenCommons store."""
    store = load_persistent_store(store_path)
    query = str(arguments.get("query", ""))
    raw_pt = arguments.get("param_types")
    param_types = tuple(str(x) for x in raw_pt) if isinstance(raw_pt, list) else None
    return_type = str(arguments["return_type"]) if arguments.get("return_type") else None

    io_examples = None
    if arguments.get("io_examples"):
        try:
            parsed_io = ast.literal_eval(str(arguments["io_examples"]))
            if isinstance(parsed_io, list):
                io_examples = parsed_io
        except (ValueError, SyntaxError, TypeError):
            io_examples = None

    modality: ServingMode = arguments.get("modality", "dual_card_with_pitfalls")
    top_k = int(arguments.get("top_k", 3))

    hits = store.find(
        query=query,
        param_types=param_types,
        return_type=return_type,
        io_examples=io_examples,
        top_k=top_k,
        include_bundle=True,
    )
    rendered_hits = []
    for h in hits:
        bundle_cards = [b.render_for_agent(mode=modality) for b in h.bundle_blocks]
        rendered_hits.append(
            {
                "block_id": h.block.block_id,
                "cid": h.block.cid,
                "sid": h.block.sid,
                "trust_tier": h.block.trust_tier,
                "score": h.score,
                "io_probe_verified": h.io_probe_verified,
                "latency_ms": h.latency_ms,
                "bundle_size": len(h.bundle_blocks),
                "rendered_card": "\n\n".join(bundle_cards),
            }
        )
    return {
        "store_concepts": len(store.canonical_block_by_cid),
        "hit_count": len(rendered_hits),
        "hits": rendered_hits,
    }


def execute_commons_record(
    arguments: dict[str, Any],
    store_path: Path | str | None = None,
    pending_path: Path | str | None = None,
) -> dict[str, Any]:
    """Execute the tiered `commons_record` (or `record`) tool:
    - If reporting a (K+1)-th counterexample (`counterexample_block_id` + `counterexample_args`),
      runs the Monotonic Counterexample Ratchet.
    - If submitting a Python function (`source_code`), runs 16-probe behavioral fingerprinting and
      5-gate adversarial peer review:
      - Passes all 5 gates -> Auto-committed into `store.json` (`status="accepted"`).
      - Fails due to non-pure/domain-specific probes (or `require_human_review=True`) -> Staged in
        `pending_queue.json` (`status="queued_for_review"`) so the user can inspect via `/commons review`.
      - Fails due to degenerate constant output or silent caller mutation -> Quarantined (`status="quarantined"`).
    """
    from agy_local.harness_hooks import get_harness_telemetry

    store = load_persistent_store(store_path)
    htel = get_harness_telemetry()

    if arguments.get("counterexample_block_id") and arguments.get("counterexample_args"):
        bid = str(arguments["counterexample_block_id"])
        failing_args = ast.literal_eval(str(arguments["counterexample_args"]))
        if not isinstance(failing_args, tuple):
            failing_args = (failing_args,)
        failure_desc = str(arguments.get("failure_description", "unprobed_17th_input_counterexample"))
        verified_fix = str(arguments.get("verified_fix", "Handle boundary input explicitly"))
        quarantined = store.report_counterexample(
            block_id=bid,
            failing_args=failing_args,
            failure_description=failure_desc,
            verified_fix=verified_fix,
        )
        save_persistent_store(store, store_path)
        if quarantined:
            htel.commons_counterexamples_reported += 1
        return {
            "action": "report_counterexample",
            "block_id": bid,
            "quarantined": quarantined,
            "active_quarantined_count": len(store.quarantined_ids),
        }

    source_code = str(arguments.get("source_code") or arguments.get("code") or "").strip()
    if not source_code:
        return {"error": "source_code is required when recording a new concept block."}

    invariants_raw = arguments.get("invariants") or []
    invariants = tuple(str(x) for x in invariants_raw) if isinstance(invariants_raw, list) else ()
    domain = str(arguments.get("domain", "general"))
    summary = str(arguments.get("summary", ""))
    pseudocode = str(arguments.get("pseudocode", ""))
    teleological_intent = str(arguments.get("teleological_intent", ""))
    dimensional_units = str(arguments.get("dimensional_units", ""))
    contributor = str(arguments.get("contributor_agent", "agy-local"))
    require_review = bool(arguments.get("require_human_review", False))

    pitfalls_raw = arguments.get("pitfalls") or []
    pitfalls_list: list[PitfallWarning] = []
    if isinstance(pitfalls_raw, list):
        for idx, pf in enumerate(pitfalls_raw):
            if isinstance(pf, dict):
                pitfalls_list.append(
                    PitfallWarning(
                        pitfall_id=str(pf.get("pitfall_id", f"pf_{idx + 1}")),
                        naive_pattern=str(pf.get("naive_pattern", "")),
                        failure_mechanism=str(pf.get("failure_mechanism", "")),
                        counterexample_input=str(pf.get("counterexample_input", "")),
                        verified_fix=str(pf.get("verified_fix", "")),
                    )
                )

    if require_review:
        sid, func_name, _, _, _ = compute_structural_sid(source_code)
        prop = enqueue_pending_proposal(
            PendingProposal(
                proposal_id=f"prop_{uuid.uuid4().hex[:8]}",
                func_name=func_name,
                domain=domain,
                summary=summary or f"Proposed function {func_name}",
                pseudocode=pseudocode,
                invariants=list(invariants),
                source_code=source_code,
                teleological_intent=teleological_intent,
                dimensional_units=dimensional_units,
                pitfalls=[asdict(p) for p in pitfalls_list],
                contributor_agent=contributor,
                stage_reason="explicit_human_review_requested",
                sid=sid,
                fp=f"fp_pending:{sid[4:]}",
            ),
            pending_path=pending_path,
        )
        htel.commons_proposals_queued += 1
        return {
            "action": "stage_pending",
            "status": "queued_for_review",
            "proposal_id": prop.proposal_id,
            "func_name": func_name,
            "sid": sid,
            "stage_reason": prop.stage_reason,
        }

    block, fp_res, verdict = store.ingest(
        source_code=source_code,
        domain=domain,
        summary=summary,
        pseudocode=pseudocode,
        invariants=invariants,
        pitfalls=tuple(pitfalls_list),
        teleological_intent=teleological_intent,
        dimensional_units=dimensional_units,
        contributor_agent=contributor,
        run_peer_review=True,
    )
    passed = verdict.passed if verdict else True
    reasons = list(verdict.reasons) if verdict else []

    # Tiered routing: if it failed ONLY because generic canonical probes raised domain exceptions
    # (`zero_valid_probes`), queue it in pending_queue.json for human `/commons review` rather than
    # losing a legitimate domain-specific discovery!
    if not passed and any("zero_valid_probes" in r for r in reasons):
        # Remove from store's quarantined set and stage for human review
        store.blocks_by_id.pop(block.block_id, None)
        store.quarantined_ids.discard(block.block_id)
        prop = enqueue_pending_proposal(
            PendingProposal(
                proposal_id=f"prop_{uuid.uuid4().hex[:8]}",
                func_name=block.name,
                domain=domain,
                summary=block.summary,
                pseudocode=block.pseudocode,
                invariants=list(invariants),
                source_code=source_code,
                teleological_intent=teleological_intent,
                dimensional_units=dimensional_units,
                pitfalls=[asdict(p) for p in pitfalls_list],
                contributor_agent=contributor,
                stage_reason="domain_specific_inputs_need_review",
                sid=block.sid,
                fp=block.fp,
            ),
            pending_path=pending_path,
        )
        htel.commons_proposals_queued += 1
        return {
            "action": "stage_pending",
            "status": "queued_for_review",
            "proposal_id": prop.proposal_id,
            "block_id": block.block_id,
            "sid": block.sid,
            "stage_reason": prop.stage_reason,
            "quarantine_reasons": reasons,
        }

    merged_store = save_persistent_store(store, store_path)
    if passed:
        htel.commons_concepts_recorded += 1

    return {
        "action": "ingest",
        "status": "accepted" if passed else "quarantined",
        "block_id": block.block_id,
        "cid": block.cid,
        "sid": block.sid,
        "fp": block.fp,
        "trust_tier": block.trust_tier,
        "passed_peer_review": passed,
        "quarantine_reasons": reasons,
        "elapsed_ms": fp_res.elapsed_ms,
        "total_active_concepts": len(merged_store.canonical_block_by_cid),
    }


def build_turn_zero_commons_context(
    query: str,
    *,
    store_path: Path | str | None = None,
    top_k: int = 2,
    min_score: float = 3.0,
) -> tuple[str, int]:
    """Queries OpenCommons on Turn 0 for verified Dual Concept Cards & Pitfalls matching `query`.

    Returns `(xml_context_block, hit_count)`. If no relevant concept matches (`score < min_score`),
    returns `("", 0)` so unrelated tasks incur zero token overhead.
    """
    if not query.strip():
        return ("", 0)
    store = load_persistent_store(store_path)
    hits = store.find(query=query, top_k=top_k, min_score=min_score, include_bundle=True)
    if not hits:
        return ("", 0)

    cards: list[str] = []
    for h in hits:
        for b in h.bundle_blocks:
            cards.append(b.render_for_agent(mode="dual_card_with_pitfalls"))

    joined = "\n\n".join(cards)
    xml_block = (
        f'<opencommons_knowledge hits="{len(hits)}" store_concepts="{len(store.canonical_block_by_cid)}">\n'
        f"The following peer-reviewed scientific/algorithmic building blocks and refuted pitfalls were "
        f"retrieved from the OpenCommons knowledge base in <5ms. Reuse their verified invariants and "
        f"avoid the documented dead-ends:\n\n"
        f"{joined}\n"
        f"</opencommons_knowledge>"
    )
    return (xml_block, len(hits))


def extract_candidate_functions_from_source(
    source_code: str,
    *,
    domain: str = "workspace_discovery",
    contributor_agent: str = "agy-local-auto",
    store_path: Path | str | None = None,
    pending_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Scans Python source code for standalone, type-annotated pure helper functions and submits them
    through the tiered OpenCommons pipeline (auto-committing 5-gate peer-reviewed functions).
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return []

    results: list[dict[str, Any]] = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name.startswith("_") or node.name.startswith("test_"):
            continue
        if not node.args.args or node.returns is None:
            continue
        # Require explicit type annotations on all parameters and return type
        if any(a.annotation is None for a in node.args.args):
            continue
        # Skip methods (`self` / `cls`) or async/generator functions
        if node.args.args[0].arg in ("self", "cls"):
            continue
        if any(isinstance(sub, (ast.Yield, ast.YieldFrom, ast.Await, ast.Global, ast.Nonlocal)) for sub in ast.walk(node)):
            continue
        # Keep concise algorithmic building blocks (3 to 45 lines)
        end_ln = getattr(node, "end_lineno", node.lineno + 5)
        if not (3 <= (end_ln - node.lineno + 1) <= 45):
            continue

        fn_src = ast.unparse(node)
        doc = ast.get_docstring(node) or ""
        first_doc_line = doc.strip().splitlines()[0] if doc.strip() else f"Verified function {node.name}"

        with contextlib.suppress(Exception):
            res = execute_commons_record(
                {
                    "source_code": fn_src,
                    "domain": domain,
                    "summary": first_doc_line[:140],
                    "contributor_agent": contributor_agent,
                },
                store_path=store_path,
                pending_path=pending_path,
            )
            results.append(res)
    return results


OPENCOMMONS_MCP_TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "name": "find",
        "description": (
            "Search OpenCommons for verified scientific/algorithmic building blocks by query, "
            "type signature, and live <2ms I/O probe assertions. Returns Dual Concept Cards "
            "and topological dependency bundles."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Algorithm name, domain keywords, or numbered concept (e.g. 'rk4', 'stable_logsumexp', 'crc32').",
                },
                "param_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional parameter type hints, e.g. ['list[float]'] or ['int', 'int'].",
                },
                "return_type": {
                    "type": "string",
                    "description": "Optional return type hint, e.g. 'float' or 'int'.",
                },
                "io_examples": {
                    "type": "string",
                    "description": "Optional Python literal list of ((arg1, ...), expected_output) pairs for live I/O probe filtering.",
                },
                "modality": {
                    "type": "string",
                    "enum": ["dual_card_with_pitfalls", "dual_card", "pseudocode", "raw_code"],
                    "default": "dual_card_with_pitfalls",
                },
                "top_k": {
                    "type": "integer",
                    "default": 3,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "record",
        "description": (
            "Contribute a verified Python function to OpenCommons (computes AST Merkle sid, "
            "16-probe behavioral fp/cid, and runs 5-gate adversarial peer review; auto-commits "
            "if all 5 gates pass or stages in the pending review queue) or report a counterexample."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "source_code": {
                    "type": "string",
                    "description": "Complete Python def function source code.",
                },
                "domain": {
                    "type": "string",
                    "default": "general",
                    "description": "Scientific/engineering domain (e.g. 'numerical_analysis', 'cryptography', 'systems').",
                },
                "summary": {
                    "type": "string",
                    "description": "Concise 1-line summary of what the function computes.",
                },
                "pseudocode": {
                    "type": "string",
                    "description": "Step-by-step algorithmic skeleton decoupled from surface variable names.",
                },
                "invariants": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Mathematical or systems invariants.",
                },
                "teleological_intent": {
                    "type": "string",
                    "description": "Physical conservation law or engineering rationale behind the transformation.",
                },
                "dimensional_units": {
                    "type": "string",
                    "description": "Dimensional unit signature.",
                },
                "require_human_review": {
                    "type": "boolean",
                    "default": False,
                    "description": "If true, stage in ~/.opencommons/pending_queue.json for human `/commons review`.",
                },
                "counterexample_block_id": {
                    "type": "string",
                    "description": "Optional block_id to quarantine if reporting an unprobed (K+1)-th input failure.",
                },
                "counterexample_args": {
                    "type": "string",
                    "description": "Optional Python tuple literal of failing arguments for counterexample ratcheting.",
                },
            },
        },
    },
]


def handle_opencommons_mcp_request(
    req: dict[str, Any],
    store_path: Path | str | None = None,
    pending_path: Path | str | None = None,
) -> dict[str, Any] | None:
    """Process a single JSON-RPC 2.0 request for the standalone `opencommons-mcp` server."""
    method = req.get("method", "")
    req_id = req.get("id")

    if method.startswith("notifications/"):
        return None

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "opencommons-mcp", "version": "1.0.0"},
            },
        }

    if method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": OPENCOMMONS_MCP_TOOLS_SCHEMA},
        }

    if method == "tools/call":
        params = req.get("params", {}) or {}
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {}) or {}
        try:
            if tool_name in ("find", "commons_find"):
                res_payload = execute_commons_find(arguments, store_path=store_path)
            elif tool_name in ("record", "commons_record"):
                res_payload = execute_commons_record(
                    arguments,
                    store_path=store_path,
                    pending_path=pending_path,
                )
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
                }
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(res_payload, indent=2)}],
                    "structuredContent": res_payload,
                    "isError": False,
                },
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps({"error": str(exc)})}],
                    "isError": True,
                },
            }

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Method not supported: {method}"},
    }


def run_opencommons_stdio_server(store_path: Path | str | None = None) -> int:
    """Run the 2-tool `opencommons-mcp` JSON-RPC 2.0 server over stdin/stdout."""
    resolved_path = store_path or os.environ.get("OPENCOMMONS_STORE_PATH") or DEFAULT_STORE_PATH
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        resp = handle_opencommons_mcp_request(req, store_path=resolved_path)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
    return 0


def main() -> None:
    raise SystemExit(run_opencommons_stdio_server())


if __name__ == "__main__":
    main()
