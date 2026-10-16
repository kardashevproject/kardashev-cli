"""OpenCommons Production Storage Engine (Milestone 4).

Implements:
- SQLite WAL mode storage engine (PRAGMA journal_mode=WAL, PRAGMA synchronous=NORMAL,
  PRAGMA busy_timeout=5000) with ACID guarantees and thread-safe connection pooling.
- ConceptStorage Protocol interface.
- Core schema: `concepts` table, FTS5 virtual table `concepts_fts`, monotonic sequencer `log_index`.
- Append-only transparency log (C2SP tlog-tiles and signed-note checkpoints):
  * Hash tiles: `tile/<L>/<N>`
  * Data tiles: `entries/<N>`
  * Delta packs: `packs/<from>-<to>.zst` (or zlib compressed)
  * Signed short-expiry Ed25519 checkpoints
  * Append-only revocations
- Two-tier storage support: local lightweight catalog (~250B/concept) + lazy body fetch by content hash.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import re
import sqlite3
import threading
import time
import zlib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

# Try importing zstandard for high-performance delta packs; fallback to zlib
try:
    import zstandard as zstd  # type: ignore[import-untyped]
except ImportError:
    zstd = None


# ---------------------------------------------------------------------------
# Constants & Enums
# ---------------------------------------------------------------------------

DEFAULT_ORIGIN: str = "opencommons.kardashevproject.org/tlog/v1"
TILE_SIZE: int = 256  # 2^8 entries per tile (W=8)
DEFAULT_BUSY_TIMEOUT_MS: int = 5000
CHECKPOINT_EXPIRY_SECONDS: int = 86400  # 24 hours


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConceptSummary:
    """Lightweight concept catalog representation (~250 bytes)."""

    content_hash: str
    name: str
    domain: str
    summary: str
    log_index: int = 0
    status: str = "accepted"
    created_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "content_hash": self.content_hash,
            "name": self.name,
            "domain": self.domain,
            "summary": self.summary,
            "log_index": self.log_index,
            "status": self.status,
            "created_at": self.created_at,
        }


@dataclass
class ConceptRecord:
    """Full concept record including verified source code, invariants, and pitfalls."""

    content_hash: str
    name: str
    domain: str
    json_body: dict[str, Any] | str
    log_index: int = 0
    created_at: float = field(default_factory=time.time)
    status: str = "accepted"  # "accepted", "revoked", "quarantined", "probation"

    @property
    def body_dict(self) -> dict[str, Any]:
        if isinstance(self.json_body, dict):
            return self.json_body
        try:
            return json.loads(self.json_body)
        except (json.JSONDecodeError, TypeError, ValueError):
            return {}

    @property
    def body_json(self) -> str:
        if isinstance(self.json_body, str):
            return self.json_body
        return json.dumps(self.json_body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def to_summary(self) -> ConceptSummary:
        b = self.body_dict
        summary_text = str(b.get("summary", ""))
        return ConceptSummary(
            content_hash=self.content_hash,
            name=self.name,
            domain=self.domain,
            summary=summary_text,
            log_index=self.log_index,
            status=self.status,
            created_at=self.created_at,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "content_hash": self.content_hash,
            "name": self.name,
            "domain": self.domain,
            "json_body": self.body_dict,
            "log_index": self.log_index,
            "created_at": self.created_at,
            "status": self.status,
        }


@dataclass(frozen=True)
class LogEntry:
    """An immutable entry within the append-only transparency log."""

    log_index: int
    entry_type: str  # "concept" or "revocation"
    content_hash: str
    leaf_hash: str  # Hex-encoded SHA-256 leaf hash
    data: dict[str, Any]
    created_at: float

    def to_hint(self) -> dict[str, Any]:
        """Compact hint payload for live SSE stream broadcast."""
        return {
            "log_index": self.log_index,
            "entry_type": self.entry_type,
            "content_hash": self.content_hash,
            "domain": self.data.get("domain", ""),
            "name": self.data.get("name", ""),
            "created_at": self.created_at,
        }

    def canonical_bytes(self) -> bytes:
        payload = {
            "log_index": self.log_index,
            "entry_type": self.entry_type,
            "content_hash": self.content_hash,
            "data": self.data,
            "created_at": self.created_at,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    def to_dict(self) -> dict[str, Any]:
        return {
            "log_index": self.log_index,
            "entry_type": self.entry_type,
            "content_hash": self.content_hash,
            "leaf_hash": self.leaf_hash,
            "data": self.data,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class SignedCheckpoint:
    """C2SP tlog-checkpoint and signed-note structure."""

    origin: str
    size: int
    root_hash: str  # Base64-encoded Merkle tree root hash
    timestamp: int  # Unix epoch timestamp in seconds
    signature: str  # Base64-encoded Ed25519 signature
    key_hint: str = ""  # Base64-encoded 4-byte key identifier hint
    raw_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "origin": self.origin,
            "size": self.size,
            "root_hash": self.root_hash,
            "timestamp": self.timestamp,
            "signature": self.signature,
            "key_hint": self.key_hint,
            "raw_text": self.raw_text,
        }

    def format_note(self) -> str:
        header = f"{self.origin}\n{self.size}\n{self.root_hash}\nTimestamp: {self.timestamp}\n"
        sig_line = f"— {self.origin} {self.key_hint} {self.signature}\n"
        return f"{header}\n{sig_line}"

    @classmethod
    def from_note(cls, text: str) -> SignedCheckpoint:
        stripped = text.strip()
        parts = stripped.split("\n\n")
        if len(parts) < 2:
            raise ValueError("Invalid signed-note format: missing signature block delimiter")

        header = parts[0].strip()
        sig_block = parts[1].strip()

        lines = header.splitlines()
        if len(lines) < 3:
            raise ValueError("Invalid checkpoint header: expected at least origin, size, root_hash")

        origin = lines[0].strip()
        size = int(lines[1].strip())
        root_hash = lines[2].strip()

        timestamp = 0
        for extra in lines[3:]:
            if extra.lower().startswith("timestamp:"):
                ts_str = extra.split(":", 1)[1].strip()
                timestamp = int(ts_str) if ts_str.isdigit() else 0

        # Parse signature line: "— <name> <hint> <signature>"
        sig_lines = sig_block.splitlines()
        first_sig = sig_lines[0].lstrip("—").strip()
        sig_tokens = first_sig.split()
        if len(sig_tokens) < 3:
            raise ValueError("Malformed signature line in signed-note")

        key_hint = sig_tokens[1]
        signature = sig_tokens[2]

        return cls(
            origin=origin,
            size=size,
            root_hash=root_hash,
            timestamp=timestamp,
            signature=signature,
            key_hint=key_hint,
            raw_text=stripped,
        )


@dataclass(frozen=True)
class RevocationRecord:
    """An append-only record indicating a revoked concept."""

    content_hash: str
    reason: str
    log_index: int
    created_at: float


# ---------------------------------------------------------------------------
# ConceptStorage Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ConceptStorage(Protocol):
    """Universal storage interface for OpenCommons hubs and client nodes."""

    def get_concept(self, content_hash: str) -> ConceptRecord | None: ...

    def find_concepts(
        self, query: str = "", domain: str | None = None, limit: int = 50, offset: int = 0
    ) -> list[ConceptSummary]: ...

    def append_accepted_concept(self, record: ConceptRecord) -> int: ...

    def append_revocation(self, content_hash: str, reason: str) -> int: ...

    def get_latest_checkpoint(self) -> SignedCheckpoint: ...

    def get_entries_since(self, since_index: int, limit: int = 100) -> list[LogEntry]: ...


# ---------------------------------------------------------------------------
# Merkle Tree & Transparency Log Cryptographic Primitives
# ---------------------------------------------------------------------------


def leaf_hash(data: bytes) -> bytes:
    """Computes leaf hash according to RFC 6962 / RFC 9162 (SHA256(0x00 || data))."""
    return hashlib.sha256(b"\x00" + data).digest()


def node_hash(left: bytes, right: bytes) -> bytes:
    """Computes interior node hash (SHA256(0x01 || left || right))."""
    return hashlib.sha256(b"\x01" + left + right).digest()


def compute_merkle_root(leaf_hashes: list[bytes]) -> bytes:
    """Computes RFC 6962 Merkle Tree Head (MTH) for an arbitrary list of leaf hashes."""
    n = len(leaf_hashes)
    if n == 0:
        return hashlib.sha256(b"").digest()
    if n == 1:
        return leaf_hashes[0]

    # Split at largest power of 2 strictly less than n
    k = 1 << ((n - 1).bit_length() - 1)
    left_root = compute_merkle_root(leaf_hashes[:k])
    right_root = compute_merkle_root(leaf_hashes[k:])
    return node_hash(left_root, right_root)


def compute_inclusion_proof(leaf_hashes: list[bytes], target_idx: int) -> list[bytes]:
    """Generates an RFC 6962 inclusion audit path for leaf at target_idx."""
    n = len(leaf_hashes)
    if target_idx < 0 or target_idx >= n:
        raise IndexError(f"Target index {target_idx} out of range for tree size {n}")

    def _path(m: int, leaves: list[bytes]) -> list[bytes]:
        k_len = len(leaves)
        if k_len <= 1:
            return []
        k = 1 << ((k_len - 1).bit_length() - 1)
        if m < k:
            return _path(m, leaves[:k]) + [compute_merkle_root(leaves[k:])]
        else:
            return _path(m - k, leaves[k:]) + [compute_merkle_root(leaves[:k])]

    return _path(target_idx, leaf_hashes)


def verify_inclusion_proof(leaf_h: bytes, index: int, tree_size: int, proof: list[bytes], root: bytes) -> bool:
    """Verifies an RFC 6962 inclusion proof against a known root."""
    if tree_size == 1:
        return leaf_h == root if not proof else False

    curr = leaf_h
    idx = index
    size = tree_size
    proof_iter = iter(proof)

    try:
        def _verify_step(m: int, n: int, h: bytes) -> bytes:
            if n <= 1:
                return h
            k = 1 << ((n - 1).bit_length() - 1)
            if m < k:
                sub = _verify_step(m, k, h)
                sibling = next(proof_iter)
                return node_hash(sub, sibling)
            else:
                sub = _verify_step(m - k, n - k, h)
                sibling = next(proof_iter)
                return node_hash(sibling, sub)

        computed_root = _verify_step(idx, size, curr)
        return computed_root == root
    except (StopIteration, IndexError, ValueError):
        return False


def sign_checkpoint(
    origin: str,
    size: int,
    root_hash_bytes: bytes,
    private_key: ed25519.Ed25519PrivateKey,
    timestamp: int | None = None,
) -> SignedCheckpoint:
    """Produces a C2SP signed-note checkpoint signed with an Ed25519 private key."""
    ts = timestamp if timestamp is not None else int(time.time())
    root_b64 = base64.b64encode(root_hash_bytes).decode("ascii")

    # Header to sign: origin\nsize\nroot_hash\nTimestamp: ts\n
    header = f"{origin}\n{size}\n{root_b64}\nTimestamp: {ts}\n"
    sig_bytes = private_key.sign(header.encode("utf-8"))
    sig_b64 = base64.b64encode(sig_bytes).decode("ascii")

    pub_bytes = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
    )
    hint_digest = hashlib.sha256(origin.encode("utf-8") + b"\n" + pub_bytes).digest()[:4]
    hint_b64 = base64.b64encode(hint_digest).decode("ascii")

    raw_note = f"{header}\n— {origin} {hint_b64} {sig_b64}\n"
    return SignedCheckpoint(
        origin=origin,
        size=size,
        root_hash=root_b64,
        timestamp=ts,
        signature=sig_b64,
        key_hint=hint_b64,
        raw_text=raw_note,
    )


def verify_checkpoint_signature(
    checkpoint: SignedCheckpoint,
    public_key: ed25519.Ed25519PublicKey,
) -> bool:
    """Verifies the Ed25519 signature on a SignedCheckpoint."""
    header = f"{checkpoint.origin}\n{checkpoint.size}\n{checkpoint.root_hash}\nTimestamp: {checkpoint.timestamp}\n"
    try:
        sig_bytes = base64.b64decode(checkpoint.signature)
        public_key.verify(sig_bytes, header.encode("utf-8"))
        return True
    except (InvalidSignature, ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# SQLite Production Storage Engine (SQLiteConceptStorage)
# ---------------------------------------------------------------------------


class SQLiteConceptStorage:
    """High-concurrency SQLite storage engine in WAL mode with FTS5 and C2SP tlog-tiles."""

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        origin: str = DEFAULT_ORIGIN,
        signing_key: ed25519.Ed25519PrivateKey | None = None,
        timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
    ) -> None:
        self.db_path = str(db_path)
        self.origin = origin
        self.timeout_ms = timeout_ms
        self._lock = threading.RLock()
        self._local = threading.local()

        # Checkpoint signing key (online signing key)
        if signing_key is None:
            self._signing_key = ed25519.Ed25519PrivateKey.generate()
        else:
            self._signing_key = signing_key
        self._public_key = self._signing_key.public_key()

        # Ensure directory exists if not memory
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialize schema and verify WAL mode
        self._init_db()

    @property
    def public_key(self) -> ed25519.Ed25519PublicKey:
        return self._public_key

    @property
    def public_key_hex(self) -> str:
        raw = self._public_key.public_bytes(
            encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
        )
        return raw.hex()

    def _get_connection(self) -> sqlite3.Connection:
        """Retrieves or creates a thread-local SQLite connection with WAL mode enabled."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(
                self.db_path,
                timeout=self.timeout_ms / 1000.0,
                check_same_thread=False,
                isolation_level=None,  # Autocommit mode; manual transactions via BEGIN
            )
            conn.row_factory = sqlite3.Row
            # Enforce production PRAGMAs
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute(f"PRAGMA busy_timeout={self.timeout_ms};")
            conn.execute("PRAGMA foreign_keys=ON;")
            conn.execute("PRAGMA cache_size=-64000;")  # 64 MB cache
            conn.execute("PRAGMA mmap_size=268435456;")  # 256 MB mmap
            self._local.conn = conn
        return self._local.conn

    def _execute_write_transaction(self, callback: Callable[[sqlite3.Connection], Any]) -> Any:
        """Executes callback inside a BEGIN IMMEDIATE transaction with lock-retry semantics."""
        max_retries = 30
        for attempt in range(max_retries):
            try:
                with self._lock:
                    conn = self._get_connection()
                    conn.execute("BEGIN IMMEDIATE;")
                    try:
                        res = callback(conn)
                        conn.execute("COMMIT;")
                        return res
                    except Exception:
                        with contextlib.suppress(sqlite3.Error):
                            conn.execute("ROLLBACK;")
                        raise
            except sqlite3.OperationalError as exc:
                if ("locked" in str(exc).lower() or "busy" in str(exc).lower()) and attempt < max_retries - 1:
                    time.sleep(0.01 * (1.1**attempt))
                    continue
                raise

    def _init_db(self) -> None:
        """Initializes tables, FTS5 virtual index, and transparency structures."""
        def _do_init(conn: sqlite3.Connection) -> None:
            # 1. Main concepts table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS concepts (
                    content_hash TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    json_body TEXT,
                    log_index INTEGER UNIQUE,
                    created_at REAL NOT NULL,
                    status TEXT NOT NULL DEFAULT 'accepted'
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_concepts_name ON concepts(name);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_concepts_domain ON concepts(domain);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_concepts_status ON concepts(status);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_concepts_log_index ON concepts(log_index);")

            # 2. Full-Text Search 5 (FTS5) virtual table
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS concepts_fts USING fts5(
                    content_hash UNINDEXED,
                    name,
                    summary,
                    pseudocode,
                    invariants,
                    domain UNINDEXED,
                    tokenize='porter unicode61'
                );
                """
            )

            # 3. Append-only log entries table (sequenced)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS log_entries (
                    log_index INTEGER PRIMARY KEY,
                    entry_type TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    leaf_hash TEXT NOT NULL,
                    data TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_log_content_hash ON log_entries(content_hash);")

            # 4. Revocations table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS revocations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content_hash TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    log_index INTEGER NOT NULL,
                    created_at REAL NOT NULL
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_revocations_hash ON revocations(content_hash);")

            # 5. Checkpoints table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS checkpoints (
                    size INTEGER PRIMARY KEY,
                    origin TEXT NOT NULL,
                    root_hash TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    signature TEXT NOT NULL,
                    key_hint TEXT NOT NULL,
                    raw_text TEXT NOT NULL
                );
                """
            )

            # 6. Submission job queue state table (for hub ingestion durable jobs)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS submission_jobs (
                    submission_id TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    failure_rule_id TEXT,
                    failure_detail TEXT
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON submission_jobs(status);")

        self._execute_write_transaction(_do_init)

    def close(self) -> None:
        """Closes all active database connections."""
        with self._lock:
            if hasattr(self._local, "conn") and self._local.conn is not None:
                with contextlib.suppress(sqlite3.Error):
                    self._local.conn.close()
                self._local.conn = None

    # -----------------------------------------------------------------------
    # Concept Storage Protocol Implementation
    # -----------------------------------------------------------------------

    def get_concept(self, content_hash: str) -> ConceptRecord | None:
        """Retrieves a full concept record by its RFC 8785 content hash."""
        conn = self._get_connection()
        cur = conn.execute(
            """
            SELECT content_hash, name, domain, json_body, log_index, created_at, status
            FROM concepts
            WHERE content_hash = ?;
            """,
            (content_hash,),
        )
        row = cur.fetchone()
        if not row:
            return None

        body_val = row["json_body"]
        body_dict: dict[str, Any]
        if body_val:
            try:
                body_dict = json.loads(body_val)
            except (json.JSONDecodeError, TypeError, ValueError):
                body_dict = {}
        else:
            body_dict = {}

        return ConceptRecord(
            content_hash=row["content_hash"],
            name=row["name"],
            domain=row["domain"],
            json_body=body_dict,
            log_index=row["log_index"] or 0,
            created_at=row["created_at"],
            status=row["status"],
        )

    def find_concepts(
        self, query: str = "", domain: str | None = None, limit: int = 50, offset: int = 0
    ) -> list[ConceptSummary]:
        """Searches concepts using FTS5 and optional domain filtering."""
        conn = self._get_connection()
        query_str = query.strip()
        results: list[ConceptSummary] = []

        if query_str:
            # Sanitize FTS5 query terms (escaping non-alphanumeric punctuation)
            clean_tokens = [re.sub(r"[^\w*]", "", token) for token in query_str.split()]
            clean_tokens = [tok for tok in clean_tokens if tok]
            fts_query = " ".join(clean_tokens) if clean_tokens else ""

            if fts_query:
                sql = """
                    SELECT c.content_hash, c.name, c.domain, c.json_body, c.log_index, c.status, c.created_at
                    FROM concepts c
                    JOIN concepts_fts f ON c.content_hash = f.content_hash
                    WHERE concepts_fts MATCH ?
                """
                params: list[Any] = [fts_query]
                if domain:
                    sql += " AND c.domain = ?"
                    params.append(domain)
                sql += " AND c.status = 'accepted' ORDER BY rank LIMIT ? OFFSET ?;"
                params.extend([limit, offset])

                try:
                    cur = conn.execute(sql, params)
                    rows = cur.fetchall()
                    for r in rows:
                        body = json.loads(r["json_body"]) if r["json_body"] else {}
                        results.append(
                            ConceptSummary(
                                content_hash=r["content_hash"],
                                name=r["name"],
                                domain=r["domain"],
                                summary=str(body.get("summary", "")),
                                log_index=r["log_index"] or 0,
                                status=r["status"],
                                created_at=r["created_at"],
                            )
                        )
                    return results
                except sqlite3.OperationalError:
                    # If FTS MATCH fails syntax (e.g. invalid syntax), fall back to LIKE search below
                    pass

        # Fallback query without FTS or with general LIKE
        sql = """
            SELECT content_hash, name, domain, json_body, log_index, status, created_at
            FROM concepts
            WHERE status = 'accepted'
        """
        params = []
        if domain:
            sql += " AND domain = ?"
            params.append(domain)
        if query_str:
            sql += " AND (name LIKE ? OR json_body LIKE ?)"
            params.extend([f"%{query_str}%", f"%{query_str}%"])

        sql += " ORDER BY log_index DESC LIMIT ? OFFSET ?;"
        params.extend([limit, offset])

        cur = conn.execute(sql, params)
        for r in cur.fetchall():
            body = json.loads(r["json_body"]) if r["json_body"] else {}
            results.append(
                ConceptSummary(
                    content_hash=r["content_hash"],
                    name=r["name"],
                    domain=r["domain"],
                    summary=str(body.get("summary", "")),
                    log_index=r["log_index"] or 0,
                    status=r["status"],
                    created_at=r["created_at"],
                )
            )
        return results

    def append_accepted_concept(self, record: ConceptRecord) -> int:
        """Appends an accepted concept into the database and transparency log."""
        def _do_append(conn: sqlite3.Connection) -> int:
            # 1. Monotonic sequencer: assign next log_index
            cur = conn.execute("SELECT COALESCE(MAX(log_index), 0) + 1 AS next_idx FROM log_entries;")
            next_idx = int(cur.fetchone()["next_idx"])

            body_dict = record.body_dict
            body_json = record.body_json
            now = time.time()

            # 2. Insert or update in concepts table
            conn.execute(
                """
                INSERT INTO concepts (content_hash, name, domain, json_body, log_index, created_at, status)
                VALUES (?, ?, ?, ?, ?, ?, 'accepted')
                ON CONFLICT(content_hash) DO UPDATE SET
                    name=excluded.name,
                    domain=excluded.domain,
                    json_body=excluded.json_body,
                    log_index=excluded.log_index,
                    status='accepted';
                """,
                (record.content_hash, record.name, record.domain, body_json, next_idx, now),
            )

            # 3. Update FTS5 virtual table
            conn.execute("DELETE FROM concepts_fts WHERE content_hash = ?;", (record.content_hash,))
            summary_text = str(body_dict.get("summary", ""))
            pseudocode_text = str(body_dict.get("pseudocode", ""))
            invariants_text = " ".join(body_dict.get("invariants", []))
            conn.execute(
                """
                INSERT INTO concepts_fts (content_hash, name, summary, pseudocode, invariants, domain)
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                (
                    record.content_hash,
                    record.name,
                    summary_text,
                    pseudocode_text,
                    invariants_text,
                    record.domain,
                ),
            )

            # 4. Create and insert LogEntry
            entry_data = {
                "content_hash": record.content_hash,
                "name": record.name,
                "domain": record.domain,
                "summary": summary_text,
            }
            entry = LogEntry(
                log_index=next_idx,
                entry_type="concept",
                content_hash=record.content_hash,
                leaf_hash="",
                data=entry_data,
                created_at=now,
            )
            entry_leaf_hash = leaf_hash(entry.canonical_bytes()).hex()

            conn.execute(
                """
                INSERT INTO log_entries (log_index, entry_type, content_hash, leaf_hash, data, created_at)
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                (
                    next_idx,
                    "concept",
                    record.content_hash,
                    entry_leaf_hash,
                    json.dumps(entry_data),
                    now,
                ),
            )

            # 5. Recompute Merkle root and sign new checkpoint
            cur = conn.execute("SELECT leaf_hash FROM log_entries ORDER BY log_index ASC;")
            all_leaf_hashes = [bytes.fromhex(r["leaf_hash"]) for r in cur.fetchall()]
            root_bytes = compute_merkle_root(all_leaf_hashes)

            checkpoint = sign_checkpoint(
                origin=self.origin,
                size=len(all_leaf_hashes),
                root_hash_bytes=root_bytes,
                private_key=self._signing_key,
                timestamp=int(now),
            )

            conn.execute(
                """
                INSERT INTO checkpoints (size, origin, root_hash, timestamp, signature, key_hint, raw_text)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(size) DO UPDATE SET
                    origin=excluded.origin,
                    root_hash=excluded.root_hash,
                    timestamp=excluded.timestamp,
                    signature=excluded.signature,
                    key_hint=excluded.key_hint,
                    raw_text=excluded.raw_text;
                """,
                (
                    checkpoint.size,
                    checkpoint.origin,
                    checkpoint.root_hash,
                    checkpoint.timestamp,
                    checkpoint.signature,
                    checkpoint.key_hint,
                    checkpoint.raw_text,
                ),
            )

            return next_idx

        return int(self._execute_write_transaction(_do_append))

    def append_revocation(self, content_hash: str, reason: str) -> int:
        """Appends a revocation record to the transparency log and updates concept status."""
        def _do_revoke(conn: sqlite3.Connection) -> int:
            # 1. Monotonic sequencer: assign next log_index
            cur = conn.execute("SELECT COALESCE(MAX(log_index), 0) + 1 AS next_idx FROM log_entries;")
            next_idx = int(cur.fetchone()["next_idx"])
            now = time.time()

            # 2. Mark concept as revoked in concepts table if present
            conn.execute(
                """
                UPDATE concepts
                SET status = 'revoked'
                WHERE content_hash = ?;
                """,
                (content_hash,),
            )
            conn.execute("DELETE FROM concepts_fts WHERE content_hash = ?;", (content_hash,))

            # 3. Insert into revocations table
            conn.execute(
                """
                INSERT INTO revocations (content_hash, reason, log_index, created_at)
                VALUES (?, ?, ?, ?);
                """,
                (content_hash, reason, next_idx, now),
            )

            # 4. Insert into log_entries
            rev_data = {
                "content_hash": content_hash,
                "reason": reason,
            }
            entry = LogEntry(
                log_index=next_idx,
                entry_type="revocation",
                content_hash=content_hash,
                leaf_hash="",
                data=rev_data,
                created_at=now,
            )
            entry_leaf_hash = leaf_hash(entry.canonical_bytes()).hex()

            conn.execute(
                """
                INSERT INTO log_entries (log_index, entry_type, content_hash, leaf_hash, data, created_at)
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                (
                    next_idx,
                    "revocation",
                    content_hash,
                    entry_leaf_hash,
                    json.dumps(rev_data),
                    now,
                ),
            )

            # 5. Update Merkle root & signed checkpoint
            cur = conn.execute("SELECT leaf_hash FROM log_entries ORDER BY log_index ASC;")
            all_leaf_hashes = [bytes.fromhex(r["leaf_hash"]) for r in cur.fetchall()]
            root_bytes = compute_merkle_root(all_leaf_hashes)

            checkpoint = sign_checkpoint(
                origin=self.origin,
                size=len(all_leaf_hashes),
                root_hash_bytes=root_bytes,
                private_key=self._signing_key,
                timestamp=int(now),
            )

            conn.execute(
                """
                INSERT INTO checkpoints (size, origin, root_hash, timestamp, signature, key_hint, raw_text)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(size) DO UPDATE SET
                    origin=excluded.origin,
                    root_hash=excluded.root_hash,
                    timestamp=excluded.timestamp,
                    signature=excluded.signature,
                    key_hint=excluded.key_hint,
                    raw_text=excluded.raw_text;
                """,
                (
                    checkpoint.size,
                    checkpoint.origin,
                    checkpoint.root_hash,
                    checkpoint.timestamp,
                    checkpoint.signature,
                    checkpoint.key_hint,
                    checkpoint.raw_text,
                ),
            )

            return next_idx

        return int(self._execute_write_transaction(_do_revoke))

    def get_latest_checkpoint(self) -> SignedCheckpoint:
        """Retrieves the latest signed checkpoint from the database."""
        conn = self._get_connection()
        cur = conn.execute(
            """
            SELECT origin, size, root_hash, timestamp, signature, key_hint, raw_text
            FROM checkpoints
            ORDER BY size DESC
            LIMIT 1;
            """
        )
        row = cur.fetchone()
        if row:
            return SignedCheckpoint(
                origin=self.origin,
                size=row["size"],
                root_hash=row["root_hash"],
                timestamp=row["timestamp"],
                signature=row["signature"],
                key_hint=row["key_hint"],
                raw_text=row["raw_text"],
            )

        # If empty log, produce empty checkpoint (size 0)
        empty_root = compute_merkle_root([])
        return sign_checkpoint(
            origin=self.origin,
            size=0,
            root_hash_bytes=empty_root,
            private_key=self._signing_key,
        )

    def get_entries_since(self, since_index: int, limit: int = 100) -> list[LogEntry]:
        """Retrieves entries with log_index > since_index in strictly increasing order."""
        conn = self._get_connection()
        cur = conn.execute(
            """
            SELECT log_index, entry_type, content_hash, leaf_hash, data, created_at
            FROM log_entries
            WHERE log_index > ?
            ORDER BY log_index ASC
            LIMIT ?;
            """,
            (since_index, limit),
        )
        entries: list[LogEntry] = []
        for r in cur.fetchall():
            try:
                data_dict = json.loads(r["data"])
            except (json.JSONDecodeError, TypeError, ValueError):
                data_dict = {}
            entries.append(
                LogEntry(
                    log_index=r["log_index"],
                    entry_type=r["entry_type"],
                    content_hash=r["content_hash"],
                    leaf_hash=r["leaf_hash"],
                    data=data_dict,
                    created_at=r["created_at"],
                )
            )
        return entries

    def get_entry_by_index(self, log_index: int) -> LogEntry | None:
        """Retrieves an individual log entry by its 1-based log_index."""
        conn = self._get_connection()
        cur = conn.execute(
            """
            SELECT log_index, entry_type, content_hash, leaf_hash, data, created_at
            FROM log_entries
            WHERE log_index = ?;
            """,
            (log_index,),
        )
        r = cur.fetchone()
        if not r:
            return None
        try:
            data_dict = json.loads(r["data"])
        except (json.JSONDecodeError, TypeError, ValueError):
            data_dict = {}
        return LogEntry(
            log_index=r["log_index"],
            entry_type=r["entry_type"],
            content_hash=r["content_hash"],
            leaf_hash=r["leaf_hash"],
            data=data_dict,
            created_at=r["created_at"],
        )

    # -----------------------------------------------------------------------
    # C2SP tlog-tiles Serving: Hash Tiles, Data Tiles & Delta Packs
    # -----------------------------------------------------------------------

    def get_tile(self, level: int, tile_index: int, partial_w: int | None = None) -> bytes:
        """Generates a C2SP hash tile `tile/<L>/<N>` (concatenated 32-byte hashes)."""
        conn = self._get_connection()
        span_per_node = 1 << (8 * level)  # for L=0: 1 leaf per node; L=1: 256 leaves per node
        nodes_per_tile = TILE_SIZE

        start_entry_idx = tile_index * nodes_per_tile * span_per_node
        end_entry_idx = start_entry_idx + nodes_per_tile * span_per_node

        cur = conn.execute(
            """
            SELECT leaf_hash FROM log_entries
            WHERE log_index > ? AND log_index <= ?
            ORDER BY log_index ASC;
            """,
            (start_entry_idx, end_entry_idx),
        )
        leaves = [bytes.fromhex(r["leaf_hash"]) for r in cur.fetchall()]

        if not leaves:
            return b""

        if level == 0:
            # Level 0: leaf hashes directly
            hashes = leaves
        else:
            # Higher levels: roots of subtrees of size span_per_node
            hashes = []
            for i in range(0, len(leaves), span_per_node):
                sub_leaves = leaves[i : i + span_per_node]
                hashes.append(compute_merkle_root(sub_leaves))

        if partial_w is not None and partial_w < len(hashes):
            hashes = hashes[:partial_w]

        return b"".join(hashes)

    def get_data_tile(self, tile_index: int) -> bytes:
        """Serves a C2SP data tile `entries/<N>` containing up to 256 JSON log entries."""
        start_idx = tile_index * TILE_SIZE
        end_idx = start_idx + TILE_SIZE
        entries = self.get_entries_since(start_idx, limit=TILE_SIZE)
        # Filter entries within the range
        entries = [e for e in entries if e.log_index <= end_idx]
        lines = [json.dumps(e.to_dict(), separators=(",", ":")) for e in entries]
        return ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")

    def get_delta_pack(self, from_index: int, to_index: int) -> bytes:
        """Packs and compresses log entries `packs/<from>-<to>` using zstandard or zlib."""
        conn = self._get_connection()
        cur = conn.execute(
            """
            SELECT log_index, entry_type, content_hash, leaf_hash, data, created_at
            FROM log_entries
            WHERE log_index >= ? AND log_index <= ?
            ORDER BY log_index ASC;
            """,
            (from_index, to_index),
        )
        entries = []
        for r in cur.fetchall():
            try:
                data_dict = json.loads(r["data"])
            except (json.JSONDecodeError, TypeError, ValueError):
                data_dict = {}
            entries.append(
                {
                    "log_index": r["log_index"],
                    "entry_type": r["entry_type"],
                    "content_hash": r["content_hash"],
                    "leaf_hash": r["leaf_hash"],
                    "data": data_dict,
                    "created_at": r["created_at"],
                }
            )

        raw_payload = json.dumps(
            {"from": from_index, "to": to_index, "entries": entries},
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")

        if zstd is not None:
            cctx = zstd.ZstdCompressor(level=3)
            return cctx.compress(raw_payload)
        return zlib.compress(raw_payload)

    def get_inclusion_proof_for_index(self, log_index: int) -> list[str]:
        """Returns inclusion proof (hex hashes) for the entry at 1-based log_index."""
        conn = self._get_connection()
        cur = conn.execute("SELECT leaf_hash FROM log_entries ORDER BY log_index ASC;")
        leaves = [bytes.fromhex(r["leaf_hash"]) for r in cur.fetchall()]
        if not (1 <= log_index <= len(leaves)):
            raise ValueError(f"log_index {log_index} out of range [1, {len(leaves)}]")

        proof_bytes = compute_inclusion_proof(leaves, log_index - 1)
        return [p.hex() for p in proof_bytes]


# ---------------------------------------------------------------------------
# Two-Tier Client Storage Support (TwoTierClientStorage)
# ---------------------------------------------------------------------------


class TwoTierClientStorage:
    """Two-tier client storage: local lightweight catalog (~250B/concept) + lazy body fetch."""

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        fetch_body_fn: Callable[[str], dict[str, Any] | None] | None = None,
        max_cached_bodies: int = 1000,
        timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
    ) -> None:
        self.db_path = str(db_path)
        self.fetch_body_fn = fetch_body_fn
        self.max_cached_bodies = max_cached_bodies
        self.timeout_ms = timeout_ms
        self._lock = threading.RLock()
        self._local = threading.local()

        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(
                self.db_path,
                timeout=self.timeout_ms / 1000.0,
                check_same_thread=False,
                isolation_level=None,
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute(f"PRAGMA busy_timeout={self.timeout_ms};")
            self._local.conn = conn
        return self._local.conn

    def _execute_write_transaction(self, callback: Callable[[sqlite3.Connection], Any]) -> Any:
        """Executes callback inside a BEGIN IMMEDIATE transaction with lock-retry semantics."""
        max_retries = 30
        for attempt in range(max_retries):
            try:
                with self._lock:
                    conn = self._get_connection()
                    conn.execute("BEGIN IMMEDIATE;")
                    try:
                        res = callback(conn)
                        conn.execute("COMMIT;")
                        return res
                    except Exception:
                        with contextlib.suppress(sqlite3.Error):
                            conn.execute("ROLLBACK;")
                        raise
            except sqlite3.OperationalError as exc:
                if ("locked" in str(exc).lower() or "busy" in str(exc).lower()) and attempt < max_retries - 1:
                    time.sleep(0.01 * (1.1**attempt))
                    continue
                raise

    def _init_db(self) -> None:
        def _do_init(conn: sqlite3.Connection) -> None:
            # Lightweight catalog table: full body is NULL until lazy fetched
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS catalog (
                    content_hash TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    json_body TEXT,
                    log_index INTEGER,
                    status TEXT NOT NULL DEFAULT 'accepted',
                    created_at REAL NOT NULL,
                    cached_at REAL
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cat_name ON catalog(name);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cat_domain ON catalog(domain);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cat_log_index ON catalog(log_index);")

            # FTS5 search index over catalog
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS catalog_fts USING fts5(
                    content_hash UNINDEXED,
                    name,
                    summary,
                    domain UNINDEXED,
                    tokenize='porter unicode61'
                );
                """
            )

        self._execute_write_transaction(_do_init)

    def close(self) -> None:
        with self._lock:
            if hasattr(self._local, "conn") and self._local.conn is not None:
                with contextlib.suppress(sqlite3.Error):
                    self._local.conn.close()
                self._local.conn = None

    def upsert_catalog_item(self, summary: ConceptSummary) -> None:
        """Upserts a lightweight concept summary (~250 bytes) into the local catalog."""
        def _do_upsert(conn: sqlite3.Connection) -> None:
            conn.execute(
                """
                INSERT INTO catalog (content_hash, name, domain, summary, log_index, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(content_hash) DO UPDATE SET
                    name=excluded.name,
                    domain=excluded.domain,
                    summary=excluded.summary,
                    log_index=excluded.log_index,
                    status=excluded.status;
                """,
                (
                    summary.content_hash,
                    summary.name,
                    summary.domain,
                    summary.summary,
                    summary.log_index,
                    summary.status,
                    summary.created_at or time.time(),
                ),
            )

            conn.execute("DELETE FROM catalog_fts WHERE content_hash = ?;", (summary.content_hash,))
            conn.execute(
                """
                INSERT INTO catalog_fts (content_hash, name, summary, domain)
                VALUES (?, ?, ?, ?);
                """,
                (summary.content_hash, summary.name, summary.summary, summary.domain),
            )

        self._execute_write_transaction(_do_upsert)

    def find_concepts(
        self, query: str = "", domain: str | None = None, limit: int = 50, offset: int = 0
    ) -> list[ConceptSummary]:
        """Searches the local lightweight catalog with FTS5."""
        conn = self._get_connection()
        query_str = query.strip()
        results: list[ConceptSummary] = []

        if query_str:
            clean_tokens = [re.sub(r"[^\w*]", "", token) for token in query_str.split()]
            clean_tokens = [tok for tok in clean_tokens if tok]
            fts_query = " ".join(clean_tokens) if clean_tokens else ""

            if fts_query:
                sql = """
                    SELECT c.content_hash, c.name, c.domain, c.summary, c.log_index, c.status, c.created_at
                    FROM catalog c
                    JOIN catalog_fts f ON c.content_hash = f.content_hash
                    WHERE catalog_fts MATCH ?
                """
                params: list[Any] = [fts_query]
                if domain:
                    sql += " AND c.domain = ?"
                    params.append(domain)
                sql += " AND c.status = 'accepted' ORDER BY rank LIMIT ? OFFSET ?;"
                params.extend([limit, offset])

                try:
                    cur = conn.execute(sql, params)
                    for r in cur.fetchall():
                        results.append(
                            ConceptSummary(
                                content_hash=r["content_hash"],
                                name=r["name"],
                                domain=r["domain"],
                                summary=r["summary"],
                                log_index=r["log_index"] or 0,
                                status=r["status"],
                                created_at=r["created_at"],
                            )
                        )
                    return results
                except sqlite3.OperationalError:
                    pass

        sql = """
            SELECT content_hash, name, domain, summary, log_index, status, created_at
            FROM catalog
            WHERE status = 'accepted'
        """
        params = []
        if domain:
            sql += " AND domain = ?"
            params.append(domain)
        if query_str:
            sql += " AND (name LIKE ? OR summary LIKE ?)"
            params.extend([f"%{query_str}%", f"%{query_str}%"])

        sql += " ORDER BY log_index DESC LIMIT ? OFFSET ?;"
        params.extend([limit, offset])

        cur = conn.execute(sql, params)
        for r in cur.fetchall():
            results.append(
                ConceptSummary(
                    content_hash=r["content_hash"],
                    name=r["name"],
                    domain=r["domain"],
                    summary=r["summary"],
                    log_index=r["log_index"] or 0,
                    status=r["status"],
                    created_at=r["created_at"],
                )
            )
        return results

    def get_concept(self, content_hash: str) -> ConceptRecord | None:
        """Retrieves concept body from local cache, or lazily fetches it on demand."""
        conn = self._get_connection()
        cur = conn.execute(
            """
            SELECT content_hash, name, domain, summary, json_body, log_index, created_at, status
            FROM catalog
            WHERE content_hash = ?;
            """,
            (content_hash,),
        )
        row = cur.fetchone()
        if not row:
            # If not in catalog, try fetching directly via fetch_body_fn
            if self.fetch_body_fn:
                fetched_body = self.fetch_body_fn(content_hash)
                if fetched_body:
                    rec = ConceptRecord(
                        content_hash=content_hash,
                        name=str(fetched_body.get("name", "")),
                        domain=str(fetched_body.get("domain", "")),
                        json_body=fetched_body,
                    )
                    self.cache_concept_body(rec)
                    return rec
            return None

        # If body is already cached locally, return it
        if row["json_body"]:
            try:
                body_dict = json.loads(row["json_body"])
            except (json.JSONDecodeError, TypeError, ValueError):
                body_dict = {}
            return ConceptRecord(
                content_hash=row["content_hash"],
                name=row["name"],
                domain=row["domain"],
                json_body=body_dict,
                log_index=row["log_index"] or 0,
                created_at=row["created_at"],
                status=row["status"],
            )

        # Lazy body fetch from remote hub
        if self.fetch_body_fn:
            fetched_body = self.fetch_body_fn(content_hash)
            if fetched_body:
                rec = ConceptRecord(
                    content_hash=content_hash,
                    name=row["name"],
                    domain=row["domain"],
                    json_body=fetched_body,
                    log_index=row["log_index"] or 0,
                    created_at=row["created_at"],
                    status=row["status"],
                )
                self.cache_concept_body(rec)
                return rec

        return None

    def cache_concept_body(self, record: ConceptRecord) -> None:
        """Stores the full concept body in local cache and manages cache eviction."""
        def _do_cache(conn: sqlite3.Connection) -> None:
            body_json = record.body_json
            now = time.time()
            conn.execute(
                """
                INSERT INTO catalog (content_hash, name, domain, summary, json_body, log_index, created_at, cached_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(content_hash) DO UPDATE SET
                    json_body=excluded.json_body,
                    cached_at=excluded.cached_at;
                """,
                (
                    record.content_hash,
                    record.name,
                    record.domain,
                    str(record.body_dict.get("summary", "")),
                    body_json,
                    record.log_index,
                    record.created_at,
                    now,
                ),
            )

            # LRU eviction if cached bodies exceed max_cached_bodies
            cur = conn.execute("SELECT COUNT(*) AS cnt FROM catalog WHERE json_body IS NOT NULL;")
            count = cur.fetchone()["cnt"]
            if count > self.max_cached_bodies:
                excess = count - self.max_cached_bodies
                conn.execute(
                    """
                    UPDATE catalog
                    SET json_body = NULL, cached_at = NULL
                    WHERE content_hash IN (
                        SELECT content_hash FROM catalog
                        WHERE json_body IS NOT NULL
                        ORDER BY cached_at ASC
                        LIMIT ?
                    );
                    """,
                    (excess,),
                )

        self._execute_write_transaction(_do_cache)
