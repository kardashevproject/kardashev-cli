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
import threading
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


def synthesize_ast_guided_probes(source_code: str, n_probes: int = 16) -> list[tuple[Any, ...]]:
    """Agentic AST-Guided Probe Synthesizer.

    When a domain-specific function has custom preconditions (e.g., `if p <= 0 or p >= 1:` or
    `if len(matrix) == 0:` or unannotated parameters) that cause generic probes to fail, this
    synthesizer inspects the AST branch guards, default parameter values, and parameter names to
    autonomously construct 16 valid domain probes without human intervention.
    """
    tree = ast.parse(source_code)
    fn_nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if not fn_nodes:
        return []
    fn = fn_nodes[0]

    # Collect numeric constants appearing in comparisons inside the function body
    numeric_constants: list[float] = []
    has_index_or_len = False
    has_keys_or_items = False
    for sub in ast.walk(fn):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, (int, float)) and not isinstance(sub.value, bool):
            val = float(sub.value)
            if math.isfinite(val) and abs(val) < 1e9:
                numeric_constants.append(val)
        if isinstance(sub, ast.Subscript):
            has_index_or_len = True
        if isinstance(sub, ast.Attribute) and sub.attr in ("keys", "values", "items", "get"):
            has_keys_or_items = True

    # Build candidate pools per parameter using name heuristics + AST constants + default values
    num_args = len(fn.args.args)
    defaults_offset = num_args - len(fn.args.defaults)
    per_param_candidates: list[list[Any]] = []

    for idx, arg in enumerate(fn.args.args):
        pname = arg.arg.lower()
        ann = ast.unparse(arg.annotation).strip().lower() if arg.annotation is not None else "any"

        default_val: Any = None
        if idx >= defaults_offset:
            with contextlib.suppress(Exception):
                default_val = ast.literal_eval(fn.args.defaults[idx - defaults_offset])

        if any(k in pname for k in ("prob", "rate", "alpha", "beta", "gamma", "eps", "tol", "ratio", "weight")) or (
            isinstance(default_val, float) and 0.0 < default_val < 1.0
        ):
            cands: list[Any] = [0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95, 0.15, 0.35, 0.85]
        elif any(k in pname for k in ("pos", "count", "size", "len", "dim", "degree", "steps", "iter", "k", "n", "m")) or (
            ann == "int"
        ):
            base_ints = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 20, 32, 64]
            for c in numeric_constants:
                ic = round(c)
                if ic > 0:
                    base_ints.extend([max(1, ic - 1), ic, ic + 1, ic + 2])
            cands = base_ints[:24]
        elif "matrix" in pname or "grid" in pname or "list[list" in ann:
            cands = [
                [[1.0, 0.0], [0.0, 1.0]],
                [[2.0, 1.0], [1.0, 2.0]],
                [[1.0, 2.0], [3.0, 4.0]],
                [[4.0, 2.0], [2.0, 3.0]],
                [[1.0, 1.0], [1.0, 2.0]],
                [[3.0, 0.0], [0.0, 3.0]],
                [[5.0, 1.0], [1.0, 5.0]],
                [[2.0, 0.5], [0.5, 2.0]],
            ] * 2
        elif has_keys_or_items or "dict" in ann or "map" in pname:
            cands = _build_probe_values_for_type("dict")
        elif has_index_or_len or any(k in pname for k in ("arr", "vec", "seq", "items", "vals", "data", "xs", "ys")):
            cands = [
                [1.0, 2.0, 3.0, 4.0],
                [2.0, 4.0, 6.0, 8.0],
                [0.5, 1.5, 2.5, 3.5],
                [10.0, 20.0, 30.0],
                [1.0, 3.0, 2.0, 5.0],
                [0.2, 0.3, 0.5],
                [5.0, 10.0, 15.0, 20.0],
                [1.0, 4.0, 9.0, 16.0],
            ] * 2
        else:
            # Strictly positive floats + AST constants shifted away from singularities
            cands = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0, 12.5, 16.0, 25.0, 50.0, 0.25, 0.75]
            for c in numeric_constants:
                cands.extend([c + 0.5, c + 1.0, max(0.1, abs(c) + 0.25)])

        if default_val is not None:
            cands.insert(0, default_val)
        per_param_candidates.append(cands)

    probes: list[tuple[Any, ...]] = []
    for i in range(n_probes):
        row = []
        for p_idx, plist in enumerate(per_param_candidates):
            row.append(copy.deepcopy(plist[(i + p_idx * 2) % len(plist)]))
        probes.append(tuple(row))
    return probes


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

    def _run_probe_suite(probe_list: list[tuple[Any, ...]]) -> tuple[list[str], list[str], bool]:
        out_tokens: list[str] = []
        val_outputs: list[str] = []
        mutated = False
        for probe_args in probe_list:
            before_serialized = [_canonical_serialize(a) for a in probe_args]
            working_args = [copy.deepcopy(a) for a in probe_args]
            try:
                res = fn(*working_args)
                after_serialized = [_canonical_serialize(a) for a in working_args]
                if after_serialized != before_serialized:
                    mutated = True
                res_str = _canonical_serialize(res)
                state_str = "|".join(after_serialized) if mutated else ""
                token = f"OK:{res_str}:M({state_str})"
                out_tokens.append(token)
                val_outputs.append(res_str)
            except Exception as exc:  # noqa: BLE001
                out_tokens.append(f"ERR:{type(exc).__name__}")
        return out_tokens, val_outputs, mutated

    probes = generate_canonical_probes(param_types, n_probes=n_probes)
    outcome_tokens, valid_outputs, mutates_args = _run_probe_suite(probes)

    # Agentic Zero-Bottleneck Fallback: if standard type-based probes yielded < 4 valid outputs
    # (e.g. function enforces strict domain bounds like `0 < p < 1` or positive matrices),
    # autonomously synthesize AST-guided domain probes!
    if len(valid_outputs) < 4:
        ast_probes = synthesize_ast_guided_probes(source_code, n_probes=n_probes)
        if ast_probes:
            ast_tokens, ast_valid, ast_mut = _run_probe_suite(ast_probes)
            if len(ast_valid) > len(valid_outputs):
                probes = ast_probes
                outcome_tokens = ast_tokens
                valid_outputs = ast_valid
                mutates_args = ast_mut

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
    {
        "domain": "numerical_analysis",
        "summary": "Kahan-Babushka-Neumaier compensated summation eliminating O(N * eps) floating-point drift",
        "pseudocode": "1. Track running sum s and compensation c; 2. Handle |x| > |s| branch symmetrically; 3. return s + c",
        "invariants": ("error <= 2 * eps * sum(|x_i|)", "order-insensitive on ill-conditioned alternating series"),
        "teleological_intent": "Preserve low-order mantissa bits when summing large and small magnitudes",
        "dimensional_units": "xs: list[U] -> U",
        "pitfalls": (
            PitfallWarning(
                pitfall_id="standard_kahan_large_term",
                naive_pattern="y = x - c; t = s + y; c = (t - s) - y",
                failure_mechanism="Classic Kahan fails when the next term |x| is larger than the running sum |s|",
                counterexample_input="[1.0, 1e100, 1.0, -1e100]",
                verified_fix="Branch on abs(s) >= abs(x) (Neumaier improvement) before updating compensation c",
            ),
        ),
        "source_code": """
def neumaier_compensated_sum(xs: list[float]) -> float:
    total = 0.0
    comp = 0.0
    for x in xs:
        t = total + x
        if abs(total) >= abs(x):
            comp += (total - t) + x
        else:
            comp += (x - t) + total
        total = t
    return total + comp
""",
    },
    {
        "domain": "numerical_analysis",
        "summary": "Numerically stable binary cross-entropy loss from raw logits without exp() overflow or log(0) NaN",
        "pseudocode": "1. max(z, 0) - z * y + log1p(exp(-abs(z)))",
        "invariants": ("BCE(z, y) >= 0", "finite for all float64 z in [-1e300, 1e300]"),
        "teleological_intent": "Exact Kullback-Leibler divergence on Bernoulli logits without sigmoid saturation",
        "dimensional_units": "logit: nats, target: probability -> nats",
        "pitfalls": (
            PitfallWarning(
                pitfall_id="naive_sigmoid_bce",
                naive_pattern="-(y * math.log(1 / (1 + math.exp(-z))) + (1 - y) * math.log(1 - 1 / (1 + math.exp(-z))))",
                failure_mechanism="Raises ValueError (math domain error) or OverflowError when |z| > 37",
                counterexample_input="(100.0, 0.0)",
                verified_fix="Use max(z, 0.0) - z * y + math.log1p(math.exp(-abs(z)))",
            ),
        ),
        "source_code": """
def stable_binary_cross_entropy_with_logits(logit: float, target: float) -> float:
    max_val = logit if logit > 0.0 else 0.0
    return max_val - logit * target + math.log1p(math.exp(-abs(logit)))
""",
    },
    {
        "domain": "numerical_analysis",
        "summary": "Branch-split numerically stable sigmoid 1 / (1 + exp(-x)) preventing exp(-x) overflow for x <= -710",
        "pseudocode": "1. if x >= 0: return 1 / (1 + exp(-x)); 2. else: e = exp(x); return e / (1 + e)",
        "invariants": ("0.0 <= sigmoid(x) <= 1.0", "abs(sigmoid(x) + sigmoid(-x) - 1.0) <= 1e-15"),
        "teleological_intent": "Bounded logistic probability mapping across the full IEEE-754 float64 range",
        "dimensional_units": "x: nats -> probability",
        "pitfalls": (
            PitfallWarning(
                pitfall_id="sigmoid_neg_overflow",
                naive_pattern="1.0 / (1.0 + math.exp(-x))",
                failure_mechanism="Raises OverflowError when x <= -710.0 because -x >= 710.0",
                counterexample_input="-1000.0",
                verified_fix="Branch on x >= 0 and evaluate exp(x) / (1 + exp(x)) for negative inputs",
            ),
        ),
        "source_code": """
def stable_sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-min(x, 709.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(x, -709.0))
    return z / (1.0 + z)
""",
    },
    {
        "domain": "numerical_analysis",
        "summary": "Numerically stable Softplus log(1 + exp(x)) with linear asymptotic bypass above threshold",
        "pseudocode": "1. max(x, 0) + log1p(exp(-abs(x)))",
        "invariants": ("softplus(x) >= max(0.0, x)", "softplus(x) - softplus(-x) == x"),
        "teleological_intent": "Smooth convex rectifier primitive free of exponential overflow",
        "dimensional_units": "x: dimensionless -> dimensionless",
        "pitfalls": (
            PitfallWarning(
                pitfall_id="naive_softplus_overflow",
                naive_pattern="math.log(1.0 + math.exp(x))",
                failure_mechanism="OverflowError for x >= 710.0",
                counterexample_input="800.0",
                verified_fix="Use max(x, 0.0) + math.log1p(math.exp(-abs(x)))",
            ),
        ),
        "source_code": """
def stable_softplus(x: float) -> float:
    return (x if x > 0.0 else 0.0) + math.log1p(math.exp(-abs(x)))
""",
    },
    {
        "domain": "machine_learning",
        "summary": "FlashAttention / Milakov-Gimelshein single-pass online softmax rescaling update step",
        "pseudocode": "1. m_new = max(m_old, x); 2. d_new = d_old * exp(m_old - m_new) + exp(x - m_new)",
        "invariants": ("m_new >= m_old", "d_new >= 1.0"),
        "teleological_intent": "Streaming softmax partition normalization in SRAM without materializing N x N attention matrix",
        "dimensional_units": "m_old: nats, d_old: dimensionless, x: nats -> tuple[nats, dimensionless]",
        "pitfalls": (
            PitfallWarning(
                pitfall_id="two_pass_softmax_materialization",
                naive_pattern="Pass 1: find max; Pass 2: sum exp(x - max)",
                failure_mechanism="Requires 2x HBM memory reads in tiled attention kernels",
                counterexample_input="sequence_length >= 32768",
                verified_fix="Rescale running denominator by exp(m_old - m_new) in a single streaming pass",
            ),
        ),
        "source_code": """
def online_softmax_update(m_old: float, d_old: float, x: float) -> tuple[float, float]:
    m_new = x if x > m_old else m_old
    d_new = d_old * math.exp(max(-700.0, m_old - m_new)) + math.exp(max(-700.0, x - m_new))
    return (m_new, d_new)
""",
    },
    {
        "domain": "machine_learning",
        "summary": "Rotary Position Embedding (RoPE) 2D complex plane rotation preserving L2 vector norm",
        "pseudocode": "1. (x0*cos(theta) - x1*sin(theta), x0*sin(theta) + x1*cos(theta))",
        "invariants": ("x0_rot^2 + x1_rot^2 == x0^2 + x1^2", "orthogonal SO(2) isometry"),
        "teleological_intent": "Encode relative token distance via inner-product invariance <R_m q, R_n k> = g(q, k, m-n)",
        "dimensional_units": "x0: float, x1: float, pos: float, freq: rad -> tuple[float, float]",
        "pitfalls": (
            PitfallWarning(
                pitfall_id="rope_sign_inversion",
                naive_pattern="(x0 * cos + x1 * sin, x0 * sin - x1 * cos)",
                failure_mechanism="Improper reflection (det = -1) breaks relative position inner-product translation invariance",
                counterexample_input="(1.0, 2.0, 3.0, 0.5)",
                verified_fix="Use proper SO(2) rotation matrix [[cos, -sin], [sin, cos]] with det == +1",
            ),
        ),
        "source_code": """
def rope_rotate_pair(x0: float, x1: float, pos: float, freq: float) -> tuple[float, float]:
    angle = pos * freq
    c = math.cos(angle)
    s = math.sin(angle)
    return (x0 * c - x1 * s, x0 * s + x1 * c)
""",
    },
    {
        "domain": "machine_learning",
        "summary": "RMSNorm root-mean-square layer normalization scaling factor with epsilon guard",
        "pseudocode": "1. rms = sqrt(mean(x^2) + eps); 2. return [x / rms for x in xs]",
        "invariants": ("mean(y^2) <= 1.0", "scale-invariant: RMSNorm(alpha * x) == RMSNorm(x) for alpha > 0"),
        "teleological_intent": "Stabilize transformer hidden-state activation norm without mean-centering overhead",
        "dimensional_units": "xs: list[float], eps: float -> list[float]",
        "pitfalls": (
            PitfallWarning(
                pitfall_id="rmsnorm_eps_outside_sqrt",
                naive_pattern="math.sqrt(mean_sq) + eps",
                failure_mechanism="Placing eps outside sqrt leaves non-differentiable derivative singularity at x = 0",
                counterexample_input="([0.0, 0.0, 0.0], 1e-6)",
                verified_fix="Place abs(eps) inside math.sqrt(mean_sq + max(1e-12, abs(eps)))",
            ),
        ),
        "source_code": """
def rmsnorm_vector(xs: list[float], eps: float) -> list[float]:
    if not xs:
        return []
    safe_eps = max(1e-12, abs(eps))
    mean_sq = sum(x * x for x in xs) / len(xs)
    inv_rms = 1.0 / math.sqrt(mean_sq + safe_eps)
    return [x * inv_rms for x in xs]
""",
    },
    {
        "domain": "machine_learning",
        "summary": "Cosine annealing learning rate schedule with linear warmup and clamped progress ratio",
        "pseudocode": "1. if step < warmup: linear ramp; 2. else: min_lr + 0.5*(max_lr - min_lr)*(1 + cos(pi * progress))",
        "invariants": ("min_lr <= lr <= max_lr", "continuous C^0 transition at step == warmup_steps"),
        "teleological_intent": "Smooth gradient-descent kinetic energy decay avoiding early divergence and late overshoot",
        "dimensional_units": "step: int, warmup: int, total: int, max_lr: float, min_lr: float -> float",
        "pitfalls": (
            PitfallWarning(
                pitfall_id="cosine_div_by_zero_or_overshoot",
                naive_pattern="math.cos(math.pi * (step - warmup) / (total - warmup))",
                failure_mechanism="ZeroDivisionError when total == warmup, or LR rebounds upward when step > total",
                counterexample_input="(1200, 100, 1000, 1e-3, 1e-5)",
                verified_fix="Clamp progress ratio to [0.0, 1.0] and guard denom = max(1, total - warmup)",
            ),
        ),
        "source_code": """
def cosine_warmup_lr(step: int, warmup_steps: int, total_steps: int, max_lr: float, min_lr: float) -> float:
    s = max(0, step)
    w = max(0, warmup_steps)
    if w > 0 and s < w:
        return max_lr * (s / w)
    denom = max(1, total_steps - w)
    progress = min(1.0, max(0.0, (s - w) / denom))
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
""",
    },
    {
        "domain": "machine_learning",
        "summary": "Decoupled AdamW single-parameter update with bias-corrected 1st/2nd moments",
        "pseudocode": "1. w_decayed = w * (1 - lr * wd); 2. m_hat = m / (1 - beta1^t); 3. v_hat = v / (1 - beta2^t); 4. return w_decayed - lr * m_hat / (sqrt(v_hat) + eps)",
        "invariants": ("weight decay decoupled from adaptive gradient denominator", "finite for t >= 1"),
        "teleological_intent": "Orthogonalize L2 regularization from adaptive preconditioner scaling (Loshchilov & Hutter)",
        "dimensional_units": "w: float, g: float, m: float, v: float, lr: float, wd: float, step: int -> float",
        "pitfalls": (
            PitfallWarning(
                pitfall_id="coupled_l2_in_adam",
                naive_pattern="g = g + wd * w before moment accumulation",
                failure_mechanism="Couples weight decay with 1/sqrt(v), shrinking regularization on parameters with large historical gradients",
                counterexample_input="(1.0, 10.0, 0.0, 100.0, 0.01, 0.1, 10)",
                verified_fix="Apply w * (1 - lr * wd) directly to the weight outside the adaptive moment ratio",
            ),
        ),
        "source_code": """
def adamw_decoupled_step(w: float, g: float, m: float, v: float, lr: float, wd: float, step: int) -> float:
    t = max(1, abs(step))
    b1 = 0.9
    b2 = 0.999
    m_new = b1 * m + (1.0 - b1) * g
    v_new = b2 * v + (1.0 - b2) * (g * g)
    m_hat = m_new / (1.0 - (b1**t))
    v_hat = max(0.0, v_new / (1.0 - (b2**t)))
    w_decayed = w * (1.0 - lr * wd)
    return w_decayed - lr * (m_hat / (math.sqrt(v_hat) + 1e-8))
""",
    },
    {
        "domain": "numerical_analysis",
        "summary": "4th-order Runge-Kutta (RK4) step for linear damping ODE dy/dt = -k * y",
        "pseudocode": "1. k1 = -k*y; k2 = -k*(y + 0.5*dt*k1); k3 = -k*(y + 0.5*dt*k2); k4 = -k*(y + dt*k3); 2. y + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)",
        "invariants": ("local truncation error O(dt^5)", "matches Taylor series of exp(-k*dt) through 4th order"),
        "teleological_intent": "Fourth-order accurate autonomous relaxation integration",
        "dimensional_units": "y: U, k: 1/s, dt: s -> U",
        "pitfalls": (
            PitfallWarning(
                pitfall_id="rk4_midpoint_weight_error",
                naive_pattern="y + (dt / 4.0) * (k1 + k2 + k3 + k4)",
                failure_mechanism="Equal 1/4 weights reduce Simpson quadrature to 2nd order instead of 4th order",
                counterexample_input="(1.0, 1.0, 0.5)",
                verified_fix="Use Simpson weights (k1 + 2*k2 + 2*k3 + k4) / 6.0",
            ),
        ),
        "source_code": """
def rk4_damped_step(y: float, k: float, dt: float) -> float:
    k1 = -k * y
    k2 = -k * (y + 0.5 * dt * k1)
    k3 = -k * (y + 0.5 * dt * k2)
    k4 = -k * (y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
""",
    },
    {
        "domain": "cryptography_and_framing",
        "summary": "IEEE 802.3 32-bit Cyclic Redundancy Check (CRC-32) with reflected polynomial 0xEDB88320",
        "pseudocode": "1. crc = 0xFFFFFFFF; 2. for byte: xor & 8-bit shift with 0xEDB88320; 3. return (~crc) & 0xFFFFFFFF",
        "invariants": ("0 <= crc <= 0xFFFFFFFF", "crc32(b'123456789') == 0xCBF43926"),
        "teleological_intent": "Bitwise Galois field GF(2) polynomial division for burst-error detection",
        "dimensional_units": "data: list[int] -> uint32",
        "pitfalls": (
            PitfallWarning(
                pitfall_id="crc32_missing_final_inversion",
                naive_pattern="return crc & 0xFFFFFFFF",
                failure_mechanism="Omitting pre/post bitwise inversion (~0xFFFFFFFF) makes trailing zero bytes undetectable",
                counterexample_input="[49, 50, 51, 52, 53, 54, 55, 56, 57]",
                verified_fix="Initialize with 0xFFFFFFFF and XOR final state with 0xFFFFFFFF",
            ),
        ),
        "source_code": """
def crc32_ieee_ints(data: list[int]) -> int:
    crc = 0xFFFFFFFF
    for b in data:
        crc ^= b & 0xFF
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xEDB88320
            else:
                crc >>= 1
    return (crc ^ 0xFFFFFFFF) & 0xFFFFFFFF
""",
    },
    {
        "domain": "cryptography_and_framing",
        "summary": "Unsigned LEB128 variable-length integer encoding into 7-bit continuation byte array",
        "pseudocode": "1. Extract 7 bits (val & 0x7F); 2. Set MSB 0x80 if more non-zero bits remain; 3. Shift right by 7",
        "invariants": ("1 <= len(out) <= 10 for uint64", "last byte has MSB == 0"),
        "teleological_intent": "Prefix-free compact wire framing for WebAssembly, DWARF, and Protobuf varints",
        "dimensional_units": "value: int -> list[int]",
        "pitfalls": (
            PitfallWarning(
                pitfall_id="leb128_negative_infinite_loop",
                naive_pattern="while value > 0: ... value >>= 7",
                failure_mechanism="Returns empty list [] for value == 0 or loops infinitely on negative signed ints",
                counterexample_input="0",
                verified_fix="Mask non-negative value and use a do-while loop emitting at least one byte for 0",
            ),
        ),
        "source_code": """
def uleb128_encode(value: int) -> list[int]:
    v = abs(value)
    out: list[int] = []
    while True:
        byte = v & 0x7F
        v >>= 7
        if v != 0:
            byte |= 0x80
        out.append(byte)
        if v == 0:
            break
    return out
""",
    },
    {
        "domain": "cryptography_and_framing",
        "summary": "Unsigned LEB128 variable-length byte array decoder with shift overflow protection",
        "pseudocode": "1. Accumulate (byte & 0x7F) << shift; 2. Break when (byte & 0x80) == 0",
        "invariants": ("uleb128_decode(uleb128_encode(x)) == abs(x)", "bounded shift <= 63"),
        "teleological_intent": "Deterministic inverse decoding of prefix-free varint streams",
        "dimensional_units": "encoded: list[int] -> int",
        "pitfalls": (
            PitfallWarning(
                pitfall_id="uleb128_unbounded_shift",
                naive_pattern="shift += 7 without upper bound check",
                failure_mechanism="Malformed all-0x80 stream causes arbitrary-precision integer memory exhaustion",
                counterexample_input="[128] * 10000",
                verified_fix="Break or cap shift at 63 bits (10 bytes max for uint64)",
            ),
        ),
        "source_code": """
def uleb128_decode(encoded: list[int]) -> int:
    result = 0
    shift = 0
    for b in encoded:
        result |= (b & 0x7F) << shift
        if (b & 0x80) == 0 or shift >= 63:
            break
        shift += 7
    return result
""",
    },
    {
        "domain": "distributed_systems",
        "summary": "Token-bucket rate limiter state transition with monotonic clock clamping",
        "pseudocode": "1. elapsed = max(0, now - last); 2. tokens = min(cap, current + elapsed * rate); 3. if tokens >= cost: deduct & allow",
        "invariants": ("0.0 <= new_tokens <= capacity", "clock skew (now < last) never subtracts tokens"),
        "teleological_intent": "Enforce average throughput rate R with bounded burst size C",
        "dimensional_units": "tokens: float, cap: float, rate: 1/s, elapsed: s, cost: float -> tuple[bool, float]",
        "pitfalls": (
            PitfallWarning(
                pitfall_id="negative_elapsed_clock_skew",
                naive_pattern="tokens = min(capacity, tokens + (now - last) * rate)",
                failure_mechanism="NTP backward clock step produces negative elapsed time, draining valid tokens",
                counterexample_input="(5.0, 10.0, 2.0, -3.0, 1.0)",
                verified_fix="Clamp elapsed = max(0.0, elapsed) and capacity = max(0.0, capacity)",
            ),
        ),
        "source_code": """
def token_bucket_step(tokens: float, capacity: float, refill_rate: float, elapsed: float, cost: float) -> tuple[bool, float]:
    cap = max(1.0, abs(capacity))
    rate = max(0.0, refill_rate)
    dt = max(0.0, elapsed)
    req = max(0.0, cost)
    available = min(cap, max(0.0, tokens) + dt * rate)
    if available >= req:
        return (True, available - req)
    return (False, available)
""",
    },
    {
        "domain": "distributed_systems",
        "summary": "Exponential backoff delay with cap and deterministic jitter factor",
        "pseudocode": "1. raw = base * (2 ** min(attempt, 30)); 2. capped = min(max_delay, raw); 3. return capped * (0.5 + 0.5 * jitter)",
        "invariants": ("0.0 <= delay <= max_delay", "no integer shift overflow for large attempt counts"),
        "teleological_intent": "Desynchronize thundering-herd client retries under contention",
        "dimensional_units": "attempt: int, base_delay: s, max_delay: s, jitter_unit: float -> s",
        "pitfalls": (
            PitfallWarning(
                pitfall_id="backoff_exponent_overflow",
                naive_pattern="min(max_delay, base_delay * (2.0 ** attempt))",
                failure_mechanism="Raises OverflowError in float exponentiation when attempt >= 1024",
                counterexample_input="(2000, 0.1, 60.0, 0.5)",
                verified_fix="Clamp exponent min(max(0, attempt), 30) before computing 2 ** exp",
            ),
        ),
        "source_code": """
def exponential_backoff_delay(attempt: int, base_delay: float, max_delay: float, jitter_unit: float) -> float:
    exp = min(max(0, attempt), 30)
    cap = max(0.001, abs(max_delay))
    base = max(0.0001, abs(base_delay))
    capped = min(cap, base * (2.0**exp))
    j = min(1.0, max(0.0, abs(jitter_unit)))
    return capped * (0.5 + 0.5 * j)
""",
    },
    {
        "domain": "distributed_systems",
        "summary": "Rendezvous (Highest Random Weight / HRW) consistent hashing node selection",
        "pseudocode": "1. For each node_id, mix(key, node_id) via 32-bit FNV-1a; 2. Return node_id maximizing weight",
        "invariants": ("deterministic node assignment", "only 1/N keys remap when a node is added or removed"),
        "teleological_intent": "Zero-ring-state uniform shard placement with minimal disruption on topology changes",
        "dimensional_units": "key: int, nodes: list[int] -> int",
        "pitfalls": (
            PitfallWarning(
                pitfall_id="modulo_hash_remapping",
                naive_pattern="nodes[hash(key) % len(nodes)]",
                failure_mechanism="Adding 1 node to N nodes remaps (N-1)/N (~90%+) of all cached keys simultaneously",
                counterexample_input="(42, [1, 2, 3, 4]) -> (42, [1, 2, 3, 4, 5])",
                verified_fix="Compute pairwise HRW hash score(key, node) and pick argmax",
            ),
        ),
        "source_code": """
def rendezvous_hrw_select(key: int, nodes: list[int]) -> int:
    if not nodes:
        return -1
    best_node = nodes[0]
    best_score = -1
    for n in nodes:
        h = 0x811C9DC5
        h = ((h ^ (key & 0xFFFF)) * 0x01000193) & 0xFFFFFFFF
        h = ((h ^ (n & 0xFFFF)) * 0x01000193) & 0xFFFFFFFF
        if h > best_score:
            best_score = h
            best_node = n
    return best_node
""",
    },
    {
        "domain": "geodesy_and_spatial",
        "summary": "Haversine great-circle geodesic distance on a sphere with antipodal asin clamping",
        "pseudocode": "1. a = sin(dlat/2)^2 + cos(lat1)*cos(lat2)*sin(dlon/2)^2; 2. c = 2 * asin(min(1.0, sqrt(a))); 3. R * c",
        "invariants": ("0.0 <= d <= pi * R", "d(p1, p2) == d(p2, p1)"),
        "teleological_intent": "Well-conditioned spherical distance for both sub-meter and antipodal coordinates",
        "dimensional_units": "lat1_deg: deg, lon1_deg: deg, lat2_deg: deg, lon2_deg: deg -> km",
        "pitfalls": (
            PitfallWarning(
                pitfall_id="spherical_law_of_cosines_nan",
                naive_pattern="math.acos(sin(lat1)*sin(lat2) + cos(lat1)*cos(lat2)*cos(dlon))",
                failure_mechanism="Floating-point rounding produces argument 1.0000000000000002 > 1.0, raising ValueError in math.acos",
                counterexample_input="(37.7749, -122.4194, 37.7749, -122.4194)",
                verified_fix="Use Haversine sin^2(d/2) formulation and clamp min(1.0, math.sqrt(max(0.0, a))) before math.asin",
            ),
        ),
        "source_code": """
def haversine_distance_km(lat1_deg: float, lon1_deg: float, lat2_deg: float, lon2_deg: float) -> float:
    r_earth_km = 6371.0088
    lat1 = math.radians(lat1_deg)
    lon1 = math.radians(lon1_deg)
    lat2 = math.radians(lat2_deg)
    lon2 = math.radians(lon2_deg)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (math.sin(dlat * 0.5) ** 2) + math.cos(lat1) * math.cos(lat2) * (math.sin(dlon * 0.5) ** 2)
    c = 2.0 * math.asin(min(1.0, math.sqrt(max(0.0, a))))
    return r_earth_km * c
""",
    },
    {
        "domain": "statistics_and_information",
        "summary": "Shannon discrete entropy H(p) in nats with automatic L1 normalization and 0*log(0) == 0 limit",
        "pseudocode": "1. Normalize positive weights p_i = w_i / sum(w); 2. return -sum(p_i * log(p_i) for p_i > 0)",
        "invariants": ("0.0 <= H(p) <= log(N)", "scale-invariant: H(c * w) == H(w) for c > 0"),
        "teleological_intent": "Expected information content without log(0) domain singularities",
        "dimensional_units": "weights: list[float] -> nats",
        "pitfalls": (
            PitfallWarning(
                pitfall_id="entropy_zero_probability_nan",
                naive_pattern="-sum(p * math.log(p) for p in probs)",
                failure_mechanism="Raises ValueError (math domain error) when any category has probability 0.0",
                counterexample_input="[0.5, 0.5, 0.0]",
                verified_fix="Skip zero/non-positive probabilities using lim_{p->0+} p*log(p) = 0",
            ),
        ),
        "source_code": """
def shannon_entropy_nats(weights: list[float]) -> float:
    pos = [abs(w) for w in weights if abs(w) > 0.0]
    total = sum(pos)
    if total <= 0.0:
        return 0.0
    ent = 0.0
    for w in pos:
        p = w / total
        if p > 0.0:
            ent -= p * math.log(p)
    return max(0.0, ent)
""",
    },
    {
        "domain": "statistics_and_information",
        "summary": "Cosine similarity between two vectors with zero-norm denominator guard and [-1, 1] clamping",
        "pseudocode": "1. dot(u, v) / (||u|| * ||v||); 2. Clamp to [-1.0, 1.0]",
        "invariants": ("-1.0 <= sim(u, v) <= 1.0", "sim(u, u) == 1.0 for u != 0"),
        "teleological_intent": "Scale-invariant angular similarity in Hilbert space",
        "dimensional_units": "u: list[float], v: list[float] -> dimensionless",
        "pitfalls": (
            PitfallWarning(
                pitfall_id="cosine_sim_zero_norm_and_fp_overshoot",
                naive_pattern="sum(a*b) / (math.sqrt(sum(a*a)) * math.sqrt(sum(b*b)))",
                failure_mechanism="ZeroDivisionError on zero vectors and FP drift produces 1.0000000000000002 (breaking math.acos downstream)",
                counterexample_input="([0.0, 0.0], [1.0, 2.0])",
                verified_fix="Return 0.0 when norm product == 0 and clamp result to [-1.0, 1.0]",
            ),
        ),
        "source_code": """
def safe_cosine_similarity(u: list[float], v: list[float]) -> float:
    n = min(len(u), len(v))
    if n == 0:
        return 0.0
    dot = sum(u[i] * v[i] for i in range(n))
    nu = math.sqrt(sum(u[i] * u[i] for i in range(n)))
    nv = math.sqrt(sum(v[i] * v[i] for i in range(n)))
    denom = nu * nv
    if denom <= 1e-30:
        return 0.0
    return min(1.0, max(-1.0, dot / denom))
""",
    },
    {
        "domain": "quantitative_finance",
        "summary": "Black-Scholes log-moneyness d1 term with volatility and maturity guards",
        "pseudocode": "1. (log(S / K) + (r + 0.5 * sigma^2) * T) / (sigma * sqrt(T))",
        "invariants": ("antisymmetric under S <-> K when r == -0.5*sigma^2", "finite for all S, K, sigma, T != 0"),
        "teleological_intent": "Standardized log-normal drift-adjusted moneyness under risk-neutral measure",
        "dimensional_units": "spot: USD, strike: USD, rate: 1/yr, vol: 1/sqrt(yr), maturity: yr -> z_score",
        "pitfalls": (
            PitfallWarning(
                pitfall_id="bs_d1_missing_ito_half_variance",
                naive_pattern="(math.log(spot / strike) + rate * maturity) / (vol * math.sqrt(maturity))",
                failure_mechanism="Omits Ito lemma +0.5 * vol^2 convexity adjustment, violating martingale put-call parity",
                counterexample_input="(100.0, 100.0, 0.05, 0.2, 1.0)",
                verified_fix="Include + 0.5 * sigma * sigma * T in the drift numerator",
            ),
        ),
        "source_code": """
def black_scholes_d1(spot: float, strike: float, rate: float, vol: float, maturity: float) -> float:
    s = max(1e-8, abs(spot))
    k = max(1e-8, abs(strike))
    sigma = max(1e-6, abs(vol))
    t = max(1e-6, abs(maturity))
    return (math.log(s / k) + (rate + 0.5 * sigma * sigma) * t) / (sigma * math.sqrt(t))
""",
    },
    {
        "domain": "number_theory_and_algebra",
        "summary": "Extended Euclidean Algorithm computing (gcd(a, b), x, y) satisfying Bezout identity a*x + b*y == gcd(a, b)",
        "pseudocode": "1. Iterative quotient r_old, r = r, r_old - q*r; 2. Track Bezout coefficients s, t",
        "invariants": ("a * x + b * y == g", "g >= 0"),
        "teleological_intent": "Constructive modular inverse and Diophantine linear combination",
        "dimensional_units": "a: int, b: int -> tuple[int, int, int]",
        "pitfalls": (
            PitfallWarning(
                pitfall_id="recursive_egcd_recursion_limit",
                naive_pattern="Recursive egcd(b, a % b) without sign normalization",
                failure_mechanism="Can return negative GCD when inputs are negative or hit RecursionError on large integers",
                counterexample_input="(-240, -46)",
                verified_fix="Use iterative loop and normalize final GCD sign so g >= 0",
            ),
        ),
        "source_code": """
def extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    old_r, r = a, b
    old_s, s = 1, 0
    old_t, t = 0, 1
    while r != 0:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t
    if old_r < 0:
        return (-old_r, -old_s, -old_t)
    return (old_r, old_s, old_t)
""",
    },
    {
        "domain": "signal_processing",
        "summary": "Horner's method polynomial evaluation P(x) = sum_{i} c_i x^{N-1-i} in O(N) fused multiply-adds",
        "pseudocode": "1. acc = 0.0; 2. for c in coeffs: acc = acc * x + c; 3. return acc",
        "invariants": ("exact for degree 0/1", "N multiplications and N additions instead of O(N^2)"),
        "teleological_intent": "Minimize rounding error and operation count in polynomial approximation",
        "dimensional_units": "coeffs: list[float], x: float -> float",
        "pitfalls": (
            PitfallWarning(
                pitfall_id="naive_power_polynomial_eval",
                naive_pattern="sum(c * (x ** i) for i, c in enumerate(reversed(coeffs)))",
                failure_mechanism="Repeated x**i exponentiation amplifies relative error and overflows for large degree",
                counterexample_input="([1.0, -2.0, 1.0], 1e10)",
                verified_fix="Use nested Horner accumulation acc = acc * x + c",
            ),
        ),
        "source_code": """
def horner_polynomial_eval(coeffs: list[float], x: float) -> float:
    acc = 0.0
    for c in coeffs:
        acc = acc * x + c
    return acc
""",
    },
    {
        "domain": "numerical_analysis",
        "summary": "Numerically stable quadratic roots (r1 <= r2) avoiding b^2 ~= 4ac catastrophic cancellation",
        "pseudocode": "1. disc = max(0, b^2 - 4ac); 2. q = -0.5 * (b + copysign(sqrt(disc), b)); 3. roots = (q/a, c/q)",
        "invariants": ("r1 <= r2", "preserves root product r1 * r2 == c / a to machine precision when 4ac << b^2"),
        "teleological_intent": "Citardauq / Viete formula avoiding subtraction of nearly equal numbers",
        "dimensional_units": "a: float, b: float, c: float -> tuple[float, float]",
        "pitfalls": (
            PitfallWarning(
                pitfall_id="quadratic_formula_cancellation",
                naive_pattern="(-b + math.sqrt(b*b - 4*a*c)) / (2*a)",
                failure_mechanism="Catastrophic loss of significant digits when 4*a*c << b*b",
                counterexample_input="(1.0, 1e8, 1.0)",
                verified_fix="Compute q = -0.5 * (b + copysign(sqrt(disc), b)) and obtain smaller root via c / q",
            ),
        ),
        "source_code": """
def stable_real_quadratic_roots(a: float, b: float, c: float) -> tuple[float, float]:
    safe_a = a if abs(a) > 1e-15 else 1.0
    disc = max(0.0, b * b - 4.0 * safe_a * c)
    sqrt_d = math.sqrt(disc)
    q = -0.5 * (b + (sqrt_d if b >= 0.0 else -sqrt_d))
    if abs(q) <= 1e-30:
        return (0.0, 0.0)
    r1 = q / safe_a
    r2 = c / q
    return (r1, r2) if r1 <= r2 else (r2, r1)
""",
    },
    {
        "domain": "machine_learning",
        "summary": "Exact Gaussian Error Linear Unit (GELU) x * Phi(x) via math.erf(x / sqrt(2))",
        "pseudocode": "1. 0.5 * x * (1.0 + erf(x / sqrt(2)))",
        "invariants": ("gelu(0.0) == 0.0", "lim_{x -> -inf} gelu(x) == 0.0"),
        "teleological_intent": "Stochastic regularizer weighting input by Gaussian cumulative distribution",
        "dimensional_units": "x: float -> float",
        "pitfalls": (
            PitfallWarning(
                pitfall_id="gelu_missing_sqrt2_divisor",
                naive_pattern="0.5 * x * (1.0 + math.erf(x))",
                failure_mechanism="Omitting 1/sqrt(2) scales standard deviation by 1/sqrt(2), distorting transformer activations",
                counterexample_input="1.0",
                verified_fix="Divide x by math.sqrt(2.0) (0.7071067811865476) inside math.erf",
            ),
        ),
        "source_code": """
def exact_gelu(x: float) -> float:
    return 0.5 * x * (1.0 + math.erf(x * 0.7071067811865476))
""",
    },
    {
        "domain": "statistics_and_information",
        "summary": "Logit-log-odds transformation log(p / (1 - p)) with symmetric boundary clamping",
        "pseudocode": "1. Clamp p to [eps, 1 - eps]; 2. return log(p) - log1p(-p)",
        "invariants": ("logit(0.5) == 0.0", "logit(1 - p) == -logit(p)"),
        "teleological_intent": "Bijective mapping from probability simplex (0, 1) to real line R",
        "dimensional_units": "p: probability -> nats",
        "pitfalls": (
            PitfallWarning(
                pitfall_id="logit_boundary_div_zero",
                naive_pattern="math.log(p / (1.0 - p))",
                failure_mechanism="Raises ZeroDivisionError or ValueError at p <= 0.0 or p >= 1.0, and loses precision near p -> 1",
                counterexample_input="1.0",
                verified_fix="Clamp p into [1e-15, 1.0 - 1e-15] and use math.log(p) - math.log1p(-p)",
            ),
        ),
        "source_code": """
def safe_logit(p: float) -> float:
    clamped = min(1.0 - 1e-15, max(1e-15, abs(p) if abs(p) <= 1.0 else (1.0 / (1.0 + abs(p)))))
    return math.log(clamped) - math.log1p(-clamped)
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


def run_autonomous_referee_cycle(
    store: OpenCommonsStore,
    new_block: ConceptBlock,
) -> dict[str, Any]:
    """Agentic Differential Referee & Counterexample Hunter.

    When multiple implementations map to the same Concept Equivalence Class (`cid`), the Referee
    Agent pits `new_block` against the existing canonical implementation across 32 extended boundary
    & numerical-stability stress probes:
    - If `new_block` survives extreme boundary probes where the old canonical implementation failed
      (or if both pass and `new_block` has a cleaner, more concise AST), `new_block` is autonomously
      promoted to canonical winner for `cid`.
    - Any failing probe on the losing implementation is automatically recorded as a `PitfallWarning`.
    """
    cid = new_block.cid
    existing_canon_id = store.canonical_block_by_cid.get(cid)
    if not existing_canon_id or existing_canon_id == new_block.block_id:
        store.canonical_block_by_cid[cid] = new_block.block_id
        return {"action": "crowned_initial_canonical", "winner": new_block.block_id, "cid": cid}

    existing_block = store.blocks_by_id.get(existing_canon_id)
    if existing_block is None or existing_canon_id in store.quarantined_ids:
        store.canonical_block_by_cid[cid] = new_block.block_id
        return {"action": "replaced_quarantined_canonical", "winner": new_block.block_id, "cid": cid}

    fn_old = store._get_compiled_fn(existing_canon_id)
    fn_new = store._get_compiled_fn(new_block.block_id)
    if fn_new is None:
        return {"action": "kept_existing", "winner": existing_canon_id, "cid": cid}
    if fn_old is None:
        store.canonical_block_by_cid[cid] = new_block.block_id
        return {"action": "promoted_new_compiled", "winner": new_block.block_id, "cid": cid}

    stress_probes = generate_canonical_probes(new_block.param_types, n_probes=24) + synthesize_ast_guided_probes(
        new_block.source_code, n_probes=8
    )

    old_score = 0
    new_score = 0
    old_fail_probe: tuple[Any, ...] | None = None
    new_fail_probe: tuple[Any, ...] | None = None

    for probe in stress_probes:
        try:
            res_old = fn_old(*copy.deepcopy(probe))
            if isinstance(res_old, float) and (math.isnan(res_old) or math.isinf(res_old)):
                if old_fail_probe is None:
                    old_fail_probe = probe
            else:
                old_score += 1
        except Exception:  # noqa: BLE001
            if old_fail_probe is None:
                old_fail_probe = probe

        try:
            res_new = fn_new(*copy.deepcopy(probe))
            if isinstance(res_new, float) and (math.isnan(res_new) or math.isinf(res_new)):
                if new_fail_probe is None:
                    new_fail_probe = probe
            else:
                new_score += 1
        except Exception:  # noqa: BLE001
            if new_fail_probe is None:
                new_fail_probe = probe

    old_ast_nodes = sum(1 for _ in ast.walk(ast.parse(existing_block.source_code)))
    new_ast_nodes = sum(1 for _ in ast.walk(ast.parse(new_block.source_code)))

    if new_score > old_score or (new_score == old_score and new_ast_nodes < old_ast_nodes):
        store.canonical_block_by_cid[cid] = new_block.block_id
        if old_fail_probe is not None and new_score > old_score:
            pf = PitfallWarning(
                pitfall_id=f"referee_{len(new_block.pitfalls) + 1}",
                naive_pattern=existing_block.name,
                failure_mechanism="Failed extended referee boundary/stability probe",
                counterexample_input=repr(old_fail_probe),
                verified_fix=f"Use canonical implementation {new_block.name}",
            )
            store.blocks_by_id[new_block.block_id] = replace(new_block, pitfalls=(*new_block.pitfalls, pf))
        return {
            "action": "promoted_superior_implementation",
            "winner": new_block.block_id,
            "previous": existing_canon_id,
            "cid": cid,
            "new_score": new_score,
            "old_score": old_score,
        }

    return {
        "action": "retained_canonical_winner",
        "winner": existing_canon_id,
        "challenger": new_block.block_id,
        "cid": cid,
        "new_score": new_score,
        "old_score": old_score,
    }


_LAST_BG_GOSSIP_TS: float = 0.0
_BG_GOSSIP_LOCK = threading.Lock()


def trigger_background_gossip_sync(
    *,
    mode: Literal["pull", "push", "bidirectional"] = "pull",
    hub_url: str | None = None,
    store_path: Path | str | None = None,
    min_interval_seconds: float = 60.0,
) -> bool:
    """Fires a non-blocking daemon thread to sync OpenCommons over the network with zero TUI latency."""
    global _LAST_BG_GOSSIP_TS
    now = time.time()
    with _BG_GOSSIP_LOCK:
        if mode == "pull" and (now - _LAST_BG_GOSSIP_TS) < min_interval_seconds:
            return False
        _LAST_BG_GOSSIP_TS = now

    def _worker() -> None:
        with contextlib.suppress(Exception):
            target = hub_url or os.environ.get("OPENCOMMONS_HUB_URL") or DEFAULT_HUB_URL
            if mode in ("push", "bidirectional") and not target.endswith(".json"):
                push_to_remote_hub(hub_url=target, store_path=store_path, timeout=4.0)
            if mode in ("pull", "bidirectional"):
                pull_from_remote_hub(hub_url=target, store_path=store_path, timeout=4.0)

    t = threading.Thread(target=_worker, name="opencommons-gossip-daemon", daemon=True)
    t.start()
    return True


def process_hub_sync_request(
    req_payload: dict[str, Any],
    *,
    store_path: Path | str | None = None,
) -> dict[str, Any]:
    """Server-side handler for `/v1/sync-delta` that audits incoming blocks through the AST security gate
    + 5-gate adversarial peer review + Autonomous Differential Referee before merging into the Hub's CRDT
    store, and returns only the delta blocks missing on the client.
    """
    hub_store = load_persistent_store(store_path)
    hub_manifest = hub_store.compute_merkle_manifest()
    client_root = str(req_payload.get("merkle_root", ""))
    client_known_ids = set(req_payload.get("known_block_ids", []))
    mode = str(req_payload.get("mode", "bidirectional"))

    accepted_count = 0
    rejected_count = 0
    referee_promotions = 0

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
                    ref_res = run_autonomous_referee_cycle(hub_store, blk)
                    if ref_res.get("action") == "promoted_superior_implementation":
                        referee_promotions += 1
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
            "referee_promotions": referee_promotions,
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
        "referee_promotions": referee_promotions,
        "blocks": delta_blocks,
        "quarantined_ids": sorted(hub_store.quarantined_ids),
    }


def render_hub_dashboard_html(store: OpenCommonsStore) -> str:
    """Renders a live dark-mode HTML dashboard for HuggingFace Spaces / Railway browser visitors."""
    manifest = store.compute_merkle_manifest()
    rows_html = []
    for _cid, bid in sorted(store.canonical_block_by_cid.items()):
        b = store.blocks_by_id[bid]
        rows_html.append(
            f"<tr>"
            f"<td><code>{b.block_id}</code></td>"
            f"<td><code>{b.cid}</code></td>"
            f"<td><span class='badge'>{b.domain}</span></td>"
            f"<td>{b.summary}</td>"
            f"<td><code>({', '.join(b.param_types)}) -> {b.return_type}</code></td>"
            f"</tr>"
        )
    table_body = "\n".join(rows_html)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Kardashev OpenCommons CRDT Hub</title>
  <style>
    body {{ background: #0d1117; color: #e6edf3; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", monospace; margin: 2rem auto; max-width: 1100px; padding: 0 1rem; }}
    h1 {{ color: #58a6ff; margin-bottom: 0.2rem; }}
    .sub {{ color: #8b949e; margin-bottom: 1.5rem; }}
    .stats {{ display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }}
    .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 1rem 1.25rem; flex: 1; min-width: 180px; }}
    .card .val {{ font-size: 1.5rem; font-weight: bold; color: #3fb950; }}
    .card .lbl {{ font-size: 0.85rem; color: #8b949e; }}
    table {{ width: 100%; border-collapse: collapse; background: #161b22; border: 1px solid #30363d; border-radius: 8px; overflow: hidden; }}
    th, td {{ text-align: left; padding: 0.75rem 1rem; border-bottom: 1px solid #21262d; font-size: 0.9rem; }}
    th {{ background: #21262d; color: #58a6ff; }}
    code {{ color: #79c0ff; }}
    .badge {{ background: #1f6feb33; color: #58a6ff; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.8rem; }}
  </style>
</head>
<body>
  <h1>✦ Kardashev OpenCommons — Autonomous Agentic CRDT Hub</h1>
  <div class="sub">Zero-Trust AST Sandbox · 16-Probe Behavioral Equivalence (CID) · 5-Gate Adversarial Referee · HTTP Gossip Sync</div>
  <div class="stats">
    <div class="card"><div class="val">{len(store.canonical_block_by_cid)}</div><div class="lbl">Active Equivalence Classes (CID)</div></div>
    <div class="card"><div class="val">{len(store.blocks_by_id)}</div><div class="lbl">Total Verified Blocks</div></div>
    <div class="card"><div class="val">{len(store.quarantined_ids)}</div><div class="lbl">Quarantined Counterexamples</div></div>
    <div class="card"><div class="val"><code>{manifest['merkle_root'][:16]}</code></div><div class="lbl">16-Bucket Merkle Root (ETag)</div></div>
  </div>
  <table>
    <thead><tr><th>Canonical Block</th><th>Concept ID (CID)</th><th>Domain</th><th>Summary</th><th>Signature</th></tr></thead>
    <tbody>{table_body}</tbody>
  </table>
  <p class="sub" style="margin-top:1.25rem;">
    API Endpoints: <code>GET /v1/manifest</code> · <code>GET /v1/store</code> · <code>POST /v1/sync-delta</code> · <code>POST /mcp</code> (Streamable HTTP MCP) · <code>GET /healthz</code>
  </p>
</body>
</html>"""


def create_opencommons_hub_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    store_path: Path | str | None = None,
) -> http.server.ThreadingHTTPServer:
    """Create a multi-threaded HTTP OpenCommons CRDT Gossip Hub & Streamable HTTP MCP server
    (`GET /`, `GET /healthz`, `GET /v1/manifest`, `GET /v1/store`, `POST /v1/sync-delta`, `POST /mcp`).
    """

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
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, If-None-Match, Authorization")
            if use_gzip:
                self.send_header("Content-Encoding", "gzip")
            if etag:
                self.send_header("ETag", f'"{etag}"')
            self.send_header("Cache-Control", "public, max-age=15, must-revalidate")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, status_code: int, html_text: str) -> None:
            body = html_text.encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, If-None-Match, Authorization")
            self.end_headers()

        def do_GET(self) -> None:
            if self.path == "/healthz":
                self._send_json(200, {"status": "ok", "service": "kardashev-opencommons-hub"})
                return

            store = load_persistent_store(store_path)
            if self.path in ("/", "/index.html"):
                self._send_html(200, render_hub_dashboard_html(store))
                return

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
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length) if length > 0 else b"{}"
            if self.path in ("/mcp", "/v1/mcp"):
                try:
                    req_data = _decode_http_body(raw_body, self.headers.get("Content-Encoding"))
                    mcp_resp = handle_opencommons_mcp_request(req_data, store_path=store_path)
                    if mcp_resp is None:
                        self.send_response(202)
                        self.end_headers()
                        return
                    self._send_json(200, mcp_resp)
                except (ValueError, KeyError, OSError) as exc:
                    self._send_json(
                        400,
                        {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(exc)}},
                    )
                return

            if not self.path.startswith("/v1/sync"):
                self._send_json(404, {"error": f"Unknown endpoint: {self.path}"})
                return
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
                "sync_to_hub": {
                    "type": "boolean",
                    "default": True,
                    "description": "If true and the block passes 5-gate peer review, automatically gossip-sync with the remote OpenCommons Hub.",
                },
            },
        },
    },
    {
        "name": "sync",
        "description": (
            "Synchronize the local OpenCommons CRDT store with the remote OpenCommons Gossip Hub "
            "(pull latest verified concepts, push local peer-reviewed discoveries, or run bidirectional "
            "16-bucket Merkle delta sync)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["bidirectional", "pull", "push"],
                    "default": "bidirectional",
                    "description": "Sync direction: 'bidirectional' (pull + push delta), 'pull' (fetch remote concepts), or 'push' (publish local concepts to hub).",
                },
                "hub_url": {
                    "type": "string",
                    "description": "Optional remote OpenCommons Hub URL (defaults to OPENCOMMONS_HUB_URL or the global Kardashev OpenCommons CDN/Hub).",
                },
                "force": {
                    "type": "boolean",
                    "default": False,
                    "description": "If true, bypass ETag cache check on pull.",
                },
            },
        },
    },
    {
        "name": "status",
        "description": (
            "Return live OpenCommons CRDT store statistics, 16-bucket Merkle root hash, active Concept "
            "Equivalence Classes (CIDs), quarantined counterexamples, and pending proposals."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
]


def execute_commons_sync(
    arguments: dict[str, Any],
    store_path: Path | str | None = None,
) -> dict[str, Any]:
    """Execute the `sync` (or `commons_sync`) MCP tool for any external agent harness."""
    mode = str(arguments.get("mode", "bidirectional")).lower()
    hub_url = str(arguments["hub_url"]) if arguments.get("hub_url") else None
    force = bool(arguments.get("force", False))

    if mode == "pull":
        return pull_from_remote_hub(hub_url=hub_url, store_path=store_path, force=force)
    if mode == "push":
        return push_to_remote_hub(hub_url=hub_url, store_path=store_path)

    target = (hub_url or os.environ.get("OPENCOMMONS_HUB_URL") or DEFAULT_HUB_URL).strip()
    if not target.endswith(".json"):
        push_res = push_to_remote_hub(hub_url=target, store_path=store_path)
        if push_res.get("ok"):
            return push_res
    return pull_from_remote_hub(hub_url=target, store_path=store_path, force=force)


def execute_commons_status(
    store_path: Path | str | None = None,
    pending_path: Path | str | None = None,
) -> dict[str, Any]:
    """Execute the `status` (or `commons_status`) MCP tool."""
    store = load_persistent_store(store_path)
    manifest = store.compute_merkle_manifest()
    pending = load_pending_proposals(pending_path)
    return {
        "merkle_root": manifest["merkle_root"],
        "active_concepts_cid": len(store.canonical_block_by_cid),
        "total_blocks": len(store.blocks_by_id),
        "quarantined_blocks": len(store.quarantined_ids),
        "pending_proposals": len(pending),
        "concepts": [
            {
                "block_id": store.blocks_by_id[bid].block_id,
                "cid": cid,
                "name": store.blocks_by_id[bid].name,
                "domain": store.blocks_by_id[bid].domain,
                "signature": f"({', '.join(store.blocks_by_id[bid].param_types)}) -> {store.blocks_by_id[bid].return_type}",
                "summary": store.blocks_by_id[bid].summary,
            }
            for cid, bid in sorted(store.canonical_block_by_cid.items())
            if bid in store.blocks_by_id
        ],
    }


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
                "serverInfo": {"name": "opencommons-mcp", "version": "0.2.0"},
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
                if res_payload.get("status") == "accepted" and arguments.get("sync_to_hub", True):
                    trigger_background_gossip_sync(mode="bidirectional", store_path=store_path)
            elif tool_name in ("sync", "commons_sync"):
                res_payload = execute_commons_sync(arguments, store_path=store_path)
            elif tool_name in ("status", "commons_status"):
                res_payload = execute_commons_status(store_path=store_path, pending_path=pending_path)
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
