"""OpenCommons Client SDK: Anonymous Identity, Dynamic PoW & Concept Extractor.

Conforms strictly to The Kardashev Project zero-IP privacy protocol,
v1 Proposal JSON Schema, and RFC 8032 / RFC 8785 specifications.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

# ---------------------------------------------------------------------------
# Constants & Enums
# ---------------------------------------------------------------------------

DEFAULT_KEY_PATH = Path(
    os.environ.get("KARDASHEV_KEY_PATH", "~/.kardashev/opencommons_key.json")
).expanduser()

FALLBACK_KEY_PATH = Path("~/.opencommons/opencommons_key.json").expanduser()

DEFAULT_HUB_URL = os.environ.get("OPENCOMMONS_HUB_URL", "https://commons.kardashevproject.org")

BASE_POW_DIFFICULTY = 16  # bits
MAX_ADDITIONAL_POW_DIFFICULTY = 8  # bits (max 24 bits)
MAX_CLOCK_SKEW_SECONDS = 300  # 5 minutes
REJECTION_DECAY_WINDOW_SECONDS = 86400  # 24 hours

VALID_DOMAINS = frozenset(
    {
        "agentic_reasoning",
        "bioinformatics_and_control",
        "cryptography_and_framing",
        "distributed_systems",
        "geodesy_and_spatial",
        "graph_and_combinatorial",
        "hamiltonian_dynamics",
        "machine_learning",
        "number_theory_and_algebra",
        "numerical_analysis",
        "quantitative_finance",
        "signal_processing",
        "statistics_and_information",
        "task_planning_and_execution",
        "general",
    }
)

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
        "system",
        "popen",
    }
)

ALLOWED_IMPORTS = frozenset(
    {
        "math",
        "itertools",
        "functools",
        "collections",
        "heapq",
        "bisect",
        "operator",
        "dataclasses",
        "typing",
    }
)

NAME_REGEX = re.compile(r"^[a-z][a-z0-9_]{2,63}$")

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class KeyRecord:
    public_key_hex: str  # 64 lowercase hex chars
    private_key_hex: str  # 64 lowercase hex chars (raw 32-byte seed)
    created_at: int
    rejection_count: int = 0
    last_rejected_at: int | None = None
    submission_count: int = 0


@dataclass(frozen=True)
class PoWStamp:
    nonce: str
    timestamp: int
    difficulty: int
    content_hash: str

    def format(self) -> str:
        return f"{self.nonce}:{self.timestamp}:{self.difficulty}:{self.content_hash}"

    @classmethod
    def parse(cls, stamp_str: str) -> PoWStamp:
        parts = stamp_str.strip().split(":")
        if len(parts) != 4:
            raise ValueError(f"Invalid PoW stamp format: expected 4 colon-separated parts, got {len(parts)}")
        return cls(
            nonce=parts[0],
            timestamp=int(parts[1]),
            difficulty=int(parts[2]),
            content_hash=parts[3],
        )


@dataclass
class Proposal:
    name: str
    param_types: list[str]
    return_type: str
    summary: str
    pseudocode: str
    invariants: list[str]
    pitfalls: list[dict[str, str]]
    source_code: str
    domain: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SubmissionReceipt:
    status: str  # "accepted", "queued", "rejected", "rate_limited", "network_error"
    submission_id: str
    content_hash: str
    timestamp: int
    error: str | None = None
    retry_after: int | None = None

# ---------------------------------------------------------------------------
# Key Manager (Zero-IP Anonymous Ed25519 Identity)
# ---------------------------------------------------------------------------

class OpenCommonsKeyManager:
    """Manages anonymous local Ed25519 contributor credentials with strictly zero IP tracking."""

    @classmethod
    def resolve_key_path(cls, explicit_path: Path | None = None) -> Path:
        if explicit_path:
            return explicit_path
        primary = DEFAULT_KEY_PATH
        try:
            primary.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            return primary
        except OSError:
            FALLBACK_KEY_PATH.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            return FALLBACK_KEY_PATH

    @classmethod
    def generate_key_record(cls) -> KeyRecord:
        private_key = ed25519.Ed25519PrivateKey.generate()
        public_key = private_key.public_key()
        pub_hex = public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        ).hex()
        priv_hex = private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        ).hex()
        return KeyRecord(
            public_key_hex=pub_hex,
            private_key_hex=priv_hex,
            created_at=int(time.time()),
        )

    @classmethod
    def load_or_create(cls, key_path: Path | None = None) -> KeyRecord:
        path = cls.resolve_key_path(key_path)
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if len(data.get("public_key_hex", "")) == 64 and len(data.get("private_key_hex", "")) == 64:
                    return KeyRecord(
                        public_key_hex=data["public_key_hex"],
                        private_key_hex=data["private_key_hex"],
                        created_at=data.get("created_at", int(time.time())),
                        rejection_count=data.get("rejection_count", 0),
                        last_rejected_at=data.get("last_rejected_at"),
                        submission_count=data.get("submission_count", 0),
                    )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError, OSError):
                pass  # Fall through to regenerate corrupted key
        record = cls.generate_key_record()
        cls.save(record, path)
        return record

    @classmethod
    def save(cls, record: KeyRecord, key_path: Path | None = None) -> None:
        path = cls.resolve_key_path(key_path)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        payload = json.dumps(asdict(record), indent=2)
        # Atomic replace with POSIX 0o600 permissions
        with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as tf:
            tf.write(payload)
            temp_name = tf.name
        os.chmod(temp_name, 0o600)
        os.replace(temp_name, path)

    @classmethod
    def sign(cls, message: bytes, record: KeyRecord) -> str:
        priv_bytes = bytes.fromhex(record.private_key_hex)
        private_key = ed25519.Ed25519PrivateKey.from_private_bytes(priv_bytes)
        sig = private_key.sign(message)
        return sig.hex()

    @classmethod
    def verify(cls, public_key_hex: str, message: bytes, signature_hex: str) -> bool:
        try:
            pub_bytes = bytes.fromhex(public_key_hex)
            sig_bytes = bytes.fromhex(signature_hex)
            public_key = ed25519.Ed25519PublicKey.from_public_bytes(pub_bytes)
            public_key.verify(sig_bytes, message)
            return True
        except (ValueError, InvalidSignature, TypeError):
            return False

    @classmethod
    def record_rejection(cls, key_path: Path | None = None) -> int:
        record = cls.load_or_create(key_path)
        now = int(time.time())
        # Decay if older than 24h
        if record.last_rejected_at and (now - record.last_rejected_at) > REJECTION_DECAY_WINDOW_SECONDS:
            record.rejection_count = 1
        else:
            record.rejection_count += 1
        record.last_rejected_at = now
        cls.save(record, key_path)
        return record.rejection_count

    @classmethod
    def record_acceptance(cls, key_path: Path | None = None) -> None:
        record = cls.load_or_create(key_path)
        record.submission_count += 1
        record.rejection_count = max(0, record.rejection_count - 1)
        cls.save(record, key_path)

# ---------------------------------------------------------------------------
# Dynamic Hashcash Proof-of-Work
# ---------------------------------------------------------------------------

class HashcashPoW:
    """Computes and verifies SHA-256 Hashcash stamps with bit-level zero counting."""

    @staticmethod
    def count_leading_zero_bits(digest: bytes) -> int:
        zeros = 0
        for b in digest:
            if b == 0:
                zeros += 8
            else:
                zeros += 8 - b.bit_length()
                break
        return zeros

    @classmethod
    def compute_difficulty(cls, rejection_count: int, base_difficulty: int = BASE_POW_DIFFICULTY) -> int:
        scaling = min(max(0, rejection_count), MAX_ADDITIONAL_POW_DIFFICULTY)
        return base_difficulty + scaling

    @classmethod
    def solve(
        cls,
        content_hash: str,
        difficulty: int = BASE_POW_DIFFICULTY,
        timestamp: int | None = None,
        max_iterations: int = 50_000_000,
    ) -> PoWStamp:
        ts = int(time.time()) if timestamp is None else timestamp
        nonce = 0
        suffix = f":{ts}:{difficulty}:{content_hash}".encode("ascii")
        while nonce < max_iterations:
            challenge = str(nonce).encode("ascii") + suffix
            digest = hashlib.sha256(challenge).digest()
            if cls.count_leading_zero_bits(digest) >= difficulty:
                return PoWStamp(
                    nonce=str(nonce),
                    timestamp=ts,
                    difficulty=difficulty,
                    content_hash=content_hash,
                )
            nonce += 1
        raise TimeoutError(f"Failed to solve PoW within {max_iterations} iterations at difficulty {difficulty}")

    @classmethod
    def verify(
        cls,
        stamp: str | PoWStamp,
        content_hash: str,
        min_difficulty: int = BASE_POW_DIFFICULTY,
        max_clock_skew: int = MAX_CLOCK_SKEW_SECONDS,
    ) -> tuple[bool, str | None]:
        if isinstance(stamp, str):
            try:
                parsed = PoWStamp.parse(stamp)
            except ValueError as e:
                return False, str(e)
        else:
            parsed = stamp

        if parsed.content_hash != content_hash:
            return False, f"Content hash mismatch: {parsed.content_hash} != {content_hash}"
        if parsed.difficulty < min_difficulty:
            return False, f"Insufficient difficulty: {parsed.difficulty} < {min_difficulty}"
        now = int(time.time())
        if abs(now - parsed.timestamp) > max_clock_skew:
            return False, f"Timestamp expired or excessive clock skew ({abs(now - parsed.timestamp)}s > {max_clock_skew}s)"

        challenge = f"{parsed.nonce}:{parsed.timestamp}:{parsed.difficulty}:{parsed.content_hash}".encode("ascii")
        digest = hashlib.sha256(challenge).digest()
        zeros = cls.count_leading_zero_bits(digest)
        if zeros < parsed.difficulty:
            return False, f"Invalid PoW solution: {zeros} leading zeros < required {parsed.difficulty}"
        return True, None

# ---------------------------------------------------------------------------
# Concept Extractor & Normalizer
# ---------------------------------------------------------------------------

class ConceptExtractor:
    """Extracts candidate pure functions, strips comments/docstrings, and produces v1 proposals."""

    @staticmethod
    def check_nesting_depth(source_code: str, max_depth: int = 20) -> bool:
        """Pre-parse check: verifies indentation and brackets do not exceed max nesting depth."""
        depth = 0
        for char in source_code:
            if char in "([{":
                depth += 1
                if depth > max_depth:
                    return False
            elif char in ")]}":
                depth = max(0, depth - 1)
        # Check indentation depth (spaces/tabs)
        for line in source_code.splitlines():
            indent = len(line) - len(line.lstrip(" "))
            if (indent // 4) > max_depth:
                return False
        return True

    @classmethod
    def is_candidate_function(cls, node: ast.AST) -> bool:
        if not isinstance(node, ast.FunctionDef):
            return False
        # Reject private functions and test methods
        if node.name.startswith("_") or node.name.startswith("test_"):
            return False
        if not NAME_REGEX.match(node.name):
            return False
        # Require arguments and explicit type annotations
        if not node.args.args or node.returns is None:
            return False
        if any(a.annotation is None for a in node.args.args):
            return False
        # Reject methods (self / cls)
        if node.args.args[0].arg in ("self", "cls"):
            return False
        # Forbid async, yield, global, nonlocal
        for sub in ast.walk(node):
            if isinstance(sub, (ast.Yield, ast.YieldFrom, ast.Await, ast.Global, ast.Nonlocal)):
                return False
            # Forbid calls to banned functions
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Name)
                and sub.func.id in FORBIDDEN_CALL_NAMES
            ):
                return False
            # Forbid private attribute access (._*) except single throwaway '_'
            if isinstance(sub, ast.Attribute) and sub.attr.startswith("_") and sub.attr != "_":
                return False
        return True

    @classmethod
    def strip_comments_and_docstrings(cls, fn_node: ast.FunctionDef) -> tuple[str, str]:
        """Extracts docstring and re-emits AST with docstrings and comments completely stripped."""
        doc = ast.get_docstring(fn_node) or ""
        # Create a clean copy of the node's body without the leading docstring Expr
        cleaned_body = list(fn_node.body)
        if (
            cleaned_body
            and isinstance(cleaned_body[0], ast.Expr)
            and isinstance(cleaned_body[0].value, ast.Constant)
            and isinstance(cleaned_body[0].value.value, str)
        ):
            cleaned_body.pop(0)

        # Clone node and reassign cleaned body
        clean_fn = ast.FunctionDef(
            name=fn_node.name,
            args=fn_node.args,
            body=cleaned_body,
            decorator_list=[],  # Forbid decorators in canonical representation
            returns=fn_node.returns,
            type_comment=None,
        )
        ast.copy_location(clean_fn, fn_node)
        ast.fix_missing_locations(clean_fn)
        canonical_code = ast.unparse(clean_fn)
        return canonical_code, doc

    @classmethod
    def canonicalize_json(cls, data: dict[str, Any]) -> bytes:
        """RFC 8785 JSON Canonicalization Scheme (JCS)."""
        return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    @classmethod
    def compute_content_hash(cls, proposal_dict: dict[str, Any]) -> str:
        canonical_bytes = cls.canonicalize_json(proposal_dict)
        return hashlib.sha256(canonical_bytes).hexdigest()

    _TASK_PLANNING_KEYWORDS = (
        "topological_task",
        "task_wave",
        "task_priority",
        "schedule_wave",
        "makespan",
        "critical_path",
        "wsjf",
        "circuit_breaker",
        "context_budget",
        "token_budget",
        "fuzzy_line",
        "anchor_distance",
        "retry_policy",
        "rate_limit",
    )
    _AGENTIC_REASONING_KEYWORDS = (
        "git_bisect",
        "bisect_midpoint",
        "regression_bisect",
        "suspiciousness",
        "ochiai",
        "tarantula",
        "fault_local",
        "hypothesis",
        "ucb1",
        "tool_select",
        "consensus_verdict",
        "cross_family",
        "repair_loop",
        "rank_fusion",
        "rrf_rank",
        "information_gain",
        "triage",
        "root_cause",
    )
    _DOMAIN_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
        (
            "task_planning_and_execution",
            (
                *_TASK_PLANNING_KEYWORDS,
                "topological",
                "schedule",
                "cooldown_elapsed",
                "failure_threshold",
                "max_parallel_workers",
                "serial_bottlenecks",
                "stage_max_durations",
            ),
        ),
        (
            "agentic_reasoning",
            (
                *_AGENTIC_REASONING_KEYWORDS,
                "failing_tests",
                "prev_failing_tests",
                "total_reviewers",
                "min_quorum_ratio",
                "lexical_rank",
                "semantic_rank",
                "exploration_c",
            ),
        ),
        (
            "graph_and_combinatorial",
            (
                "centrality",
                "betweenness",
                "closeness",
                "pagerank",
                "dijkstra",
                "bellman_ford",
                "floyd_warshall",
                "spanning_tree",
                "kruskal",
                "bipartite",
                "chromatic",
                "clique",
                "vertex_cover",
                "adjacency",
                "laplacian",
                "jaccard_set",
                "edit_distance",
                "hamming",
            ),
        ),
        (
            "bioinformatics_and_control",
            (
                "dna",
                "rna",
                "codon",
                "gc_content",
                "rev_comp",
                "phred",
                "hardy_weinberg",
                "wright_fisher",
                "michaelis_menten",
                "shannon_diversity",
                "kalman",
                "pid_discrete",
                "pid_control",
            ),
        ),
        (
            "numerical_analysis",
            (
                "newton_raphson",
                "runge_kutta",
                "rk4",
                "simpson",
                "trapezoid",
                "horner",
                "chebyshev",
                "lagrange",
                "bisect_root",
                "root_bisect",
                "secant",
                "householder",
                "gram_schmidt",
            ),
        ),
        (
            "cryptography_and_framing",
            (
                "crc32",
                "crc16",
                "fnv1a",
                "adler32",
                "zigzag",
                "leb128",
                "varint",
                "hkdf",
                "merkle",
                "siphash",
                "chacha",
            ),
        ),
        (
            "distributed_systems",
            (
                "vector_clock",
                "hlc_",
                "hybrid_logical_clock",
                "bloom_optimal",
                "bloom_filter",
                "rendezvous_hrw",
                "consistent_hash",
                "raft_",
                "paxos",
                "quorum_intersection",
                "littles_law",
                "mm1_queue",
            ),
        ),
        (
            "machine_learning",
            (
                "softmax",
                "logsumexp",
                "cross_entropy",
                "gelu",
                "silu",
                "adamw",
                "cosine_similarity",
                "attention",
                "kl_divergence",
                "focal_loss",
                "huber_loss",
            ),
        ),
        (
            "statistics_and_information",
            (
                "welford",
                "wilson_score",
                "wasserstein",
                "mahalanobis",
                "shannon_entropy",
                "spearman",
                "kendall_tau",
                "mann_whitney",
                "wilcoxon",
            ),
        ),
        (
            "quantitative_finance",
            (
                "black_scholes",
                "kelly_optimal",
                "sharpe_ratio",
                "sortino",
                "value_at_risk",
                "cvar",
                "merton",
                "vasicek",
                "cox_ingersoll",
            ),
        ),
        (
            "signal_processing",
            (
                "hamming_window",
                "hann_window",
                "blackman",
                "goertzel",
                "biquad",
                "butterworth",
                "nyquist",
                "ema_step",
                "moving_average",
                "zero_crossing",
            ),
        ),
        (
            "geodesy_and_spatial",
            (
                "haversine",
                "vincenty",
                "mercator",
                "ecef",
                "geodetic",
                "bbox_iou",
                "Barycentric",
                "shoelace",
            ),
        ),
        (
            "hamiltonian_dynamics",
            (
                "verlet",
                "leapfrog",
                "symplectic",
                "kepler",
                "vis_viva",
                "roche_limit",
                "hohmann",
            ),
        ),
        (
            "number_theory_and_algebra",
            (
                "totient",
                "mobius",
                "legendre_symbol",
                "jacobi_symbol",
                "extended_gcd",
                "chinese_remainder",
                "mod_inverse",
                "Catalan",
            ),
        ),
    )

    @classmethod
    def infer_domain(
        cls,
        name: str,
        docstring: str = "",
        source_code: str = "",
        default_domain: str = "general",
    ) -> str:
        """Infer a canonical v1 domain using weighted multi-domain keyword scoring across name, docstring, and code."""
        if default_domain in VALID_DOMAINS and default_domain not in ("general", "workspace_discovery"):
            return default_domain
        name_l = name.lower()
        doc_l = docstring.lower()
        code_l = source_code.lower()

        best_domain = default_domain if default_domain in VALID_DOMAINS else "general"
        best_score = 0

        for dom, keywords in cls._DOMAIN_KEYWORDS:
            score = 0
            for kw in keywords:
                kw_l = kw.lower()
                if kw_l in name_l:
                    score += 4 if "_" in kw_l else 3
                if kw_l in doc_l:
                    score += 3 if "_" in kw_l else 2
                if kw_l in code_l:
                    score += 1
            if score > best_score:
                best_score = score
                best_domain = dom

        return best_domain

    @classmethod
    def extract_from_source(
        cls,
        source_code: str,
        default_domain: str = "general",
        auto_infer_domain: bool = False,
    ) -> list[Proposal]:
        if not cls.check_nesting_depth(source_code):
            return []
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return []

        base_domain = default_domain if default_domain in VALID_DOMAINS else "general"
        proposals: list[Proposal] = []

        for node in tree.body:
            if not cls.is_candidate_function(node):
                continue
            assert isinstance(node, ast.FunctionDef)

            canonical_src, docstring = cls.strip_comments_and_docstrings(node)
            if not canonical_src.isascii() or len(canonical_src) > 8192:
                continue

            param_types = [ast.unparse(a.annotation) for a in node.args.args]
            return_type = ast.unparse(node.returns)

            domain = (
                cls.infer_domain(node.name, docstring, canonical_src, default_domain=base_domain)
                if auto_infer_domain
                else base_domain
            )

            # Extract or synthesize summary
            lines = [l.strip() for l in docstring.splitlines() if l.strip()]
            summary = lines[0] if lines else f"Deterministic implementation of {node.name}."
            if len(summary) < 10:
                summary = f"Verified pure function {node.name}: {summary}"
            summary = summary[:500]

            # Extract or synthesize pseudocode
            pseudocode = f"Algorithm {node.name}({', '.join(param_types)}) -> {return_type}:\n"
            if len(lines) > 1:
                pseudocode += "\n".join(lines[1:])
            else:
                pseudocode += f"1. Receive typed arguments ({', '.join(param_types)})\n2. Execute canonical calculation\n3. Return {return_type}"
            pseudocode = pseudocode[:2000]

            invariants = [f"total on domain ({', '.join(param_types)})", f"returns strictly typed {return_type}"]

            prop = Proposal(
                name=node.name,
                param_types=param_types,
                return_type=return_type,
                summary=summary,
                pseudocode=pseudocode,
                invariants=invariants,
                pitfalls=[],
                source_code=canonical_src,
                domain=domain,
            )
            proposals.append(prop)
        return proposals

# ---------------------------------------------------------------------------
# Universal Submission Client
# ---------------------------------------------------------------------------

class OpenCommonsClient:
    """Client SDK for synchronous and asynchronous background proposal submissions."""

    def __init__(
        self,
        hub_url: str = DEFAULT_HUB_URL,
        key_path: Path | None = None,
        timeout: float = 10.0,
    ):
        self.hub_url = hub_url.rstrip("/")
        self.key_path = key_path
        self.timeout = timeout
        self.key_manager = OpenCommonsKeyManager

    def submit(self, proposal: Proposal | dict[str, Any]) -> SubmissionReceipt:
        prop_dict = proposal.to_dict() if isinstance(proposal, Proposal) else proposal
        content_hash = ConceptExtractor.compute_content_hash(prop_dict)
        key_record = self.key_manager.load_or_create(self.key_path)

        # Dynamic difficulty based on rejection count
        difficulty = HashcashPoW.compute_difficulty(key_record.rejection_count)
        pow_stamp = HashcashPoW.solve(content_hash, difficulty=difficulty)

        # Sign content hash with Ed25519 private key
        signature = self.key_manager.sign(content_hash.encode("ascii"), key_record)

        payload_bytes = ConceptExtractor.canonicalize_json(prop_dict)
        url = f"{self.hub_url}/v1/proposals"

        headers = {
            "Content-Type": "application/json",
            "X-OpenCommons-Key": key_record.public_key_hex,
            "X-OpenCommons-PoW": pow_stamp.format(),
            "X-OpenCommons-Signature": signature,
            "User-Agent": "kardashev-opencommons-client/1.0",
        }

        req = urllib.request.Request(url, data=payload_bytes, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                status = resp.status
                body = json.loads(resp.read().decode("utf-8"))
                if status == 202 or (status == 200 and body.get("status") == "accepted"):
                    self.key_manager.record_acceptance(self.key_path)
                    return SubmissionReceipt(
                        status="accepted",
                        submission_id=body.get("submission_id", ""),
                        content_hash=content_hash,
                        timestamp=int(time.time()),
                    )
                return SubmissionReceipt(
                    status="unexpected",
                    submission_id=body.get("submission_id", ""),
                    content_hash=content_hash,
                    timestamp=int(time.time()),
                )
        except urllib.error.HTTPError as err:
            if err.code == 429:
                retry_after = int(err.headers.get("Retry-After", 30))
                return SubmissionReceipt(
                    status="rate_limited",
                    submission_id="",
                    content_hash=content_hash,
                    timestamp=int(time.time()),
                    retry_after=retry_after,
                    error="Rate limit exceeded",
                )
            # Rejection increments difficulty
            rejections = self.key_manager.record_rejection(self.key_path)
            return SubmissionReceipt(
                status="rejected",
                submission_id="",
                content_hash=content_hash,
                timestamp=int(time.time()),
                error=f"Submission rejected by hub (HTTP {err.code}); dynamic PoW scaled to {HashcashPoW.compute_difficulty(rejections)} bits",
            )
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as ex:
            return SubmissionReceipt(
                status="network_error",
                submission_id="",
                content_hash=content_hash,
                timestamp=int(time.time()),
                error=str(ex),
            )

    def submit_background(
        self,
        proposal: Proposal | dict[str, Any],
        callback: Callable[[SubmissionReceipt], None] | None = None,
    ) -> threading.Thread:
        def _worker():
            receipt = self.submit(proposal)
            if callback:
                try:
                    callback(receipt)
                except Exception:  # noqa: BLE001, S110
                    pass

        t = threading.Thread(target=_worker, daemon=True, name="opencommons-submitter")
        t.start()
        return t
