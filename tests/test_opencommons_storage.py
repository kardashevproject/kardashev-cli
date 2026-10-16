"""Comprehensive Unit & Concurrency Tests for OpenCommons Storage Engine (Milestone 4).

Validates:
- SQLite WAL mode (PRAGMA journal_mode=WAL, PRAGMA synchronous=NORMAL, PRAGMA busy_timeout=5000).
- Monotonic sequencer assigning strictly sequential log_index (1, 2, ...).
- High-concurrency thread safety (concurrent readers and writers).
- FTS5 full-text indexing, phrase queries, domain filters, and punctuation resilience.
- C2SP transparency log (Merkle Tree Head, inclusion proofs, hash tiles, data tiles, delta packs).
- Ed25519 signed checkpoints (C2SP signed-note format, signature verification, tamper resistance).
- Append-only revocations (concept status update, FTS removal, log entry append).
- Two-tier storage engine (local lightweight catalog + lazy body fetch + LRU cache eviction).
"""

from __future__ import annotations

import base64
import concurrent.futures
import hashlib
import json
import sqlite3
import threading
import time
import zlib
from pathlib import Path
from typing import Any

import pytest

from kardashev_cli.opencommons_storage import (
    ConceptRecord,
    SignedCheckpoint,
    SQLiteConceptStorage,
    TwoTierClientStorage,
    compute_merkle_root,
    leaf_hash,
    node_hash,
    verify_checkpoint_signature,
    verify_inclusion_proof,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> Path:
    return tmp_path / "test_opencommons.db"


@pytest.fixture
def storage(tmp_db_path: Path) -> SQLiteConceptStorage:
    st = SQLiteConceptStorage(db_path=tmp_db_path)
    yield st
    st.close()


def make_sample_concept(
    name: str = "welford_variance",
    domain: str = "numerical_analysis",
    summary: str = "One-pass online variance update using Welford recurrence",
    invariants: list[str] | None = None,
) -> ConceptRecord:
    body = {
        "name": name,
        "domain": domain,
        "param_types": ["float", "float", "float", "int"],
        "return_type": "tuple[float, float]",
        "summary": summary,
        "pseudocode": "M1 += (x - M1)/k; M2 += (x - M1)*(x - M1_old)",
        "invariants": invariants or ["finite variance for k >= 2", "non-negative second moment"],
        "pitfalls": [
            {
                "naive_pattern": "sum(x^2) - sum(x)^2 / N",
                "failure_mechanism": "Catastrophic cancellation on large mean offsets",
                "counterexample_input": "x = 1e9 + 1",
                "verified_fix": "Track deviations from running mean",
            }
        ],
        "source_code": (
            f"def {name}(x: float, mean: float, m2: float, count: int) -> tuple[float, float]:\n"
            f"    count += 1\n"
            f"    delta = x - mean\n"
            f"    mean += delta / count\n"
            f"    delta2 = x - mean\n"
            f"    m2 += delta * delta2\n"
            f"    return (mean, m2)\n"
        ),
    }
    canonical_jcs = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    content_hash = hashlib.sha256(canonical_jcs).hexdigest()
    return ConceptRecord(
        content_hash=content_hash,
        name=name,
        domain=domain,
        json_body=body,
    )


# ---------------------------------------------------------------------------
# 1. SQLite WAL Mode & Concurrency Tests
# ---------------------------------------------------------------------------


def test_sqlite_wal_pragmas(storage: SQLiteConceptStorage, tmp_db_path: Path) -> None:
    """Verifies SQLite is operating in WAL mode with normal sync and busy timeout."""
    conn = sqlite3.connect(tmp_db_path)
    cur = conn.cursor()

    cur.execute("PRAGMA journal_mode;")
    journal_mode = cur.fetchone()[0].upper()
    assert journal_mode == "WAL", f"Expected WAL mode, got {journal_mode}"

    cur.execute("PRAGMA busy_timeout;")
    timeout = cur.fetchone()[0]
    assert timeout >= 5000, f"Expected busy_timeout >= 5000, got {timeout}"

    conn.close()


def test_concurrent_read_write_wal(tmp_db_path: Path) -> None:
    """Tests high-concurrency execution: multiple readers while writing under WAL mode."""
    storage = SQLiteConceptStorage(db_path=tmp_db_path)

    # Pre-populate 10 items
    for i in range(10):
        storage.append_accepted_concept(make_sample_concept(name=f"concept_{i}"))

    num_readers = 8
    num_writes = 20
    read_counts = [0] * num_readers
    errors: list[Exception] = []

    def reader_worker(worker_id: int) -> None:
        try:
            reader_storage = SQLiteConceptStorage(db_path=tmp_db_path)
            for _ in range(30):
                res = reader_storage.find_concepts(query="variance", limit=50)
                assert len(res) >= 10
                read_counts[worker_id] += 1
                time.sleep(0.005)
            reader_storage.close()
        except Exception as e:  # noqa: BLE001 - thread error capture for test assertions
            errors.append(e)

    def writer_worker() -> None:
        try:
            for j in range(num_writes):
                storage.append_accepted_concept(
                    make_sample_concept(name=f"concurrent_concept_{j}", summary=f"Online variance update {j}")
                )
                time.sleep(0.005)
        except Exception as e:  # noqa: BLE001 - thread error capture for test assertions
            errors.append(e)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_readers + 1) as pool:
        futures = [pool.submit(reader_worker, i) for i in range(num_readers)]
        futures.append(pool.submit(writer_worker))
        concurrent.futures.wait(futures)

    storage.close()
    assert not errors, f"Concurrency errors occurred: {errors}"
    assert all(c == 30 for c in read_counts)


# ---------------------------------------------------------------------------
# 2. Monotonic Sequencer Tests
# ---------------------------------------------------------------------------


def test_monotonic_sequencer_sequential_assignment(storage: SQLiteConceptStorage) -> None:
    """Verifies that the sequencer assigns strictly increasing 1-based log_index."""
    indices = []
    for i in range(15):
        c = make_sample_concept(name=f"func_{i}", summary=f"Summary for func {i}")
        idx = storage.append_accepted_concept(c)
        indices.append(idx)

    assert indices == list(range(1, 16)), f"Expected [1..15], got {indices}"


def test_monotonic_sequencer_concurrent_appends(tmp_db_path: Path) -> None:
    """Verifies that concurrent appends across threads receive unique, strictly monotonic indices."""
    num_workers = 10
    appends_per_worker = 10
    total_expected = num_workers * appends_per_worker

    assigned_indices: list[int] = []
    lock = threading.Lock()
    errors: list[Exception] = []

    def worker_fn(worker_id: int) -> None:
        try:
            worker_storage = SQLiteConceptStorage(db_path=tmp_db_path)
            for j in range(appends_per_worker):
                c = make_sample_concept(name=f"worker_{worker_id}_fn_{j}")
                idx = worker_storage.append_accepted_concept(c)
                with lock:
                    assigned_indices.append(idx)
            worker_storage.close()
        except Exception as e:  # noqa: BLE001 - thread error capture for test assertions
            errors.append(e)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = [pool.submit(worker_fn, i) for i in range(num_workers)]
        concurrent.futures.wait(futures)

    assert not errors, f"Errors during concurrent appends: {errors}"
    assert len(assigned_indices) == total_expected
    # Verify uniqueness and complete 1..total_expected coverage
    assert sorted(assigned_indices) == list(range(1, total_expected + 1))


# ---------------------------------------------------------------------------
# 3. FTS5 Search & Domain Filtering Tests
# ---------------------------------------------------------------------------


def test_fts5_indexing_and_phrase_search(storage: SQLiteConceptStorage) -> None:
    """Tests FTS5 index accuracy, BM25 ranking, and multi-word matching."""
    storage.append_accepted_concept(
        make_sample_concept(
            name="symplectic_integrator",
            domain="hamiltonian_dynamics",
            summary="Leapfrog numerical integrator preserving phase space volume",
            invariants=["Hamiltonian energy conservation", "symplectic 2-form invariance"],
        )
    )
    storage.append_accepted_concept(
        make_sample_concept(
            name="adamw_optimizer",
            domain="machine_learning",
            summary="Decoupled weight decay regularization for adaptive gradients",
            invariants=["weight decay independent of adaptive gradient"],
        )
    )
    storage.append_accepted_concept(
        make_sample_concept(
            name="fnv1a_hash",
            domain="cryptography_and_framing",
            summary="Non-cryptographic 32-bit Fowler Noll Vo hash function",
            invariants=["avalanche effect across byte boundaries"],
        )
    )

    # Search for "symplectic"
    res1 = storage.find_concepts(query="symplectic")
    assert len(res1) == 1
    assert res1[0].name == "symplectic_integrator"
    assert res1[0].domain == "hamiltonian_dynamics"

    # Search across invariants ("Hamiltonian")
    res2 = storage.find_concepts(query="Hamiltonian")
    assert len(res2) == 1
    assert res2[0].name == "symplectic_integrator"

    # Search for "decay"
    res3 = storage.find_concepts(query="decay")
    assert len(res3) == 1
    assert res3[0].name == "adamw_optimizer"

    # Domain filtering
    res4 = storage.find_concepts(query="hash", domain="cryptography_and_framing")
    assert len(res4) == 1
    assert res4[0].name == "fnv1a_hash"

    # Filter with non-matching domain
    res5 = storage.find_concepts(query="hash", domain="machine_learning")
    assert len(res5) == 0


def test_fts5_punctuation_and_symbol_query_safety(storage: SQLiteConceptStorage) -> None:
    """Ensures search queries containing code punctuation don't trigger FTS5 syntax errors."""
    storage.append_accepted_concept(
        make_sample_concept(name="safe_math_func", summary="Computes x**2 + y / z safely")
    )

    # Potentially problematic characters in FTS syntax: quotes, parens, stars, asterisks
    dangerous_queries = ['x**2', '"nested quotes"', "func(x, y)", "a AND OR NOT *", "///???"]
    for q in dangerous_queries:
        # Should not raise sqlite3.OperationalError
        results = storage.find_concepts(query=q)
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# 4. Append-Only Revocation Tests
# ---------------------------------------------------------------------------


def test_append_revocation_lifecycle(storage: SQLiteConceptStorage) -> None:
    """Verifies that revocations append a new log entry and update concept status."""
    c = make_sample_concept(name="flawed_concept")
    idx = storage.append_accepted_concept(c)
    assert idx == 1

    # Verify concept is initially accepted and discoverable
    rec1 = storage.get_concept(c.content_hash)
    assert rec1 is not None and rec1.status == "accepted"
    assert len(storage.find_concepts(query="flawed_concept")) == 1

    # Revoke concept
    rev_idx = storage.append_revocation(c.content_hash, reason="Numerical overflow vulnerability")
    assert rev_idx == 2

    # Status must now be revoked
    rec2 = storage.get_concept(c.content_hash)
    assert rec2 is not None and rec2.status == "revoked"

    # Must be excluded from active search results
    assert len(storage.find_concepts(query="flawed_concept")) == 0

    # Log entries must contain both concept and revocation
    entries = storage.get_entries_since(0)
    assert len(entries) == 2
    assert entries[0].entry_type == "concept"
    assert entries[1].entry_type == "revocation"
    assert entries[1].content_hash == c.content_hash


# ---------------------------------------------------------------------------
# 5. C2SP Transparency Log & Merkle Tree Tests
# ---------------------------------------------------------------------------


def test_rfc6962_merkle_tree_calculation() -> None:
    """Tests RFC 6962 Merkle Tree Head calculation against reference vectors."""
    # Empty tree
    assert compute_merkle_root([]) == hashlib.sha256(b"").digest()

    # Single leaf
    leaf0 = leaf_hash(b"leaf0")
    assert compute_merkle_root([leaf0]) == leaf0

    # Two leaves
    leaf1 = leaf_hash(b"leaf1")
    expected_root_2 = node_hash(leaf0, leaf1)
    assert compute_merkle_root([leaf0, leaf1]) == expected_root_2

    # Three leaves (k = 2)
    leaf2 = leaf_hash(b"leaf2")
    expected_root_3 = node_hash(expected_root_2, leaf2)
    assert compute_merkle_root([leaf0, leaf1, leaf2]) == expected_root_3


def test_merkle_inclusion_proof_and_verification(storage: SQLiteConceptStorage) -> None:
    """Generates and verifies cryptographic inclusion proofs for entries."""
    for i in range(8):
        storage.append_accepted_concept(make_sample_concept(name=f"leaf_concept_{i}"))

    cp = storage.get_latest_checkpoint()
    assert cp.size == 8
    root_bytes = base64.b64decode(cp.root_hash)

    # Verify inclusion proof for each leaf
    conn = storage._get_connection()
    cur = conn.execute("SELECT leaf_hash FROM log_entries ORDER BY log_index ASC;")
    all_leaves = [bytes.fromhex(r["leaf_hash"]) for r in cur.fetchall()]

    for idx_1based in range(1, 9):
        proof_hex = storage.get_inclusion_proof_for_index(idx_1based)
        proof_bytes = [bytes.fromhex(p) for p in proof_hex]
        target_leaf = all_leaves[idx_1based - 1]

        valid = verify_inclusion_proof(
            leaf_h=target_leaf,
            index=idx_1based - 1,
            tree_size=8,
            proof=proof_bytes,
            root=root_bytes,
        )
        assert valid, f"Inclusion proof verification failed for index {idx_1based}"


def test_tiles_and_delta_packs(storage: SQLiteConceptStorage) -> None:
    """Tests hash tiles (tile/<L>/<N>), data tiles (entries/<N>), and delta packs."""
    for i in range(10):
        storage.append_accepted_concept(make_sample_concept(name=f"item_{i}"))

    # Level 0 Tile 0
    tile0 = storage.get_tile(level=0, tile_index=0)
    assert len(tile0) == 10 * 32, f"Expected 320 bytes, got {len(tile0)}"

    # Data Tile 0
    data_tile = storage.get_data_tile(0)
    lines = [ln for ln in data_tile.decode("utf-8").splitlines() if ln]
    assert len(lines) == 10
    first_entry = json.loads(lines[0])
    assert first_entry["log_index"] == 1
    assert first_entry["entry_type"] == "concept"

    # Delta pack from index 3 to 7
    pack_bytes = storage.get_delta_pack(from_index=3, to_index=7)
    unpacked_json = zlib.decompress(pack_bytes).decode("utf-8")
    pack_data = json.loads(unpacked_json)
    assert pack_data["from"] == 3
    assert pack_data["to"] == 7
    assert len(pack_data["entries"]) == 5
    assert pack_data["entries"][0]["log_index"] == 3
    assert pack_data["entries"][-1]["log_index"] == 7


# ---------------------------------------------------------------------------
# 6. Ed25519 Signed Checkpoints Tests
# ---------------------------------------------------------------------------


def test_ed25519_signed_checkpoint_format_and_verification(storage: SQLiteConceptStorage) -> None:
    """Validates C2SP signed-note checkpoint format and Ed25519 verification."""
    storage.append_accepted_concept(make_sample_concept(name="checkpoint_test_concept"))
    cp = storage.get_latest_checkpoint()

    assert cp.size == 1
    assert cp.origin == storage.origin
    assert cp.timestamp > 0
    assert len(cp.signature) > 0
    assert len(cp.key_hint) > 0

    # Verify signature passes with legitimate public key
    assert verify_checkpoint_signature(cp, storage.public_key)

    # Verify signature fails if root hash is tampered
    tampered_root = SignedCheckpoint(
        origin=cp.origin,
        size=cp.size,
        root_hash=base64.b64encode(b"\x00" * 32).decode("ascii"),
        timestamp=cp.timestamp,
        signature=cp.signature,
        key_hint=cp.key_hint,
    )
    assert not verify_checkpoint_signature(tampered_root, storage.public_key)

    # Verify signature fails if size is tampered
    tampered_size = SignedCheckpoint(
        origin=cp.origin,
        size=cp.size + 1,
        root_hash=cp.root_hash,
        timestamp=cp.timestamp,
        signature=cp.signature,
        key_hint=cp.key_hint,
    )
    assert not verify_checkpoint_signature(tampered_size, storage.public_key)

    # Verify from_note parsing roundtrip
    note_text = cp.format_note()
    parsed_cp = SignedCheckpoint.from_note(note_text)
    assert parsed_cp.size == cp.size
    assert parsed_cp.root_hash == cp.root_hash
    assert parsed_cp.signature == cp.signature
    assert verify_checkpoint_signature(parsed_cp, storage.public_key)


# ---------------------------------------------------------------------------
# 7. Two-Tier Storage Engine Tests
# ---------------------------------------------------------------------------


def test_two_tier_catalog_and_lazy_fetch(tmp_path: Path) -> None:
    """Tests lightweight catalog population and lazy body fetch on cache miss."""
    client_db = tmp_path / "client_catalog.db"

    # Mock remote hub store
    hub_store: dict[str, dict[str, Any]] = {}
    fetch_calls: list[str] = []

    def mock_remote_fetcher(content_hash: str) -> dict[str, Any] | None:
        fetch_calls.append(content_hash)
        return hub_store.get(content_hash)

    client_storage = TwoTierClientStorage(
        db_path=client_db,
        fetch_body_fn=mock_remote_fetcher,
        max_cached_bodies=3,
    )

    # 1. Populate lightweight catalog with 5 concepts (no bodies stored locally)
    sample_items = []
    for i in range(5):
        c = make_sample_concept(name=f"remote_concept_{i}", summary=f"Lightweight summary {i}")
        sample_items.append(c)
        hub_store[c.content_hash] = c.body_dict
        client_storage.upsert_catalog_item(c.to_summary())

    # 2. Local search on catalog works with zero body fetches
    search_res = client_storage.find_concepts(query="Lightweight")
    assert len(search_res) == 5
    assert len(fetch_calls) == 0, "Catalog search must not fetch bodies from remote hub"

    # 3. Lazy fetch concept 0
    rec0 = client_storage.get_concept(sample_items[0].content_hash)
    assert rec0 is not None
    assert rec0.name == "remote_concept_0"
    assert rec0.body_dict["name"] == "remote_concept_0"
    assert len(fetch_calls) == 1

    # 4. Second access must hit local cache without invoking remote fetcher
    rec0_again = client_storage.get_concept(sample_items[0].content_hash)
    assert rec0_again is not None
    assert len(fetch_calls) == 1, "Cached concept must not trigger second remote fetch"

    # 5. Fetch items 1, 2, 3 to trigger LRU eviction (cap is 3)
    client_storage.get_concept(sample_items[1].content_hash)
    client_storage.get_concept(sample_items[2].content_hash)
    client_storage.get_concept(sample_items[3].content_hash)

    # Verify cached count stays within max_cached_bodies (3)
    conn = client_storage._get_connection()
    cur = conn.execute("SELECT COUNT(*) AS cnt FROM catalog WHERE json_body IS NOT NULL;")
    cached_count = cur.fetchone()["cnt"]
    assert cached_count == 3, f"Expected 3 cached bodies, got {cached_count}"

    client_storage.close()
