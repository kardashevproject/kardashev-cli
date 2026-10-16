"""Comprehensive Production Async Hub Server Tests (Milestone 4).

Validates:
- POST /v1/proposals:
  * 32KB payload cap (413 Payload Too Large)
  * Safe decompression limit (64KB cap & 5x ratio -> 400 Bad Request)
  * Anonymous Ed25519 key authentication (401 Unauthorized on missing/invalid)
  * Dynamic Proof-of-Work verification (400 Bad Request on missing/invalid)
  * Ingestion queue backpressure (429 Too Many Requests + Retry-After)
  * Idempotency for existing concepts
  * Successful submission returning 202 Accepted with opaque status query URL
- GET /v1/proposals/{id}: Opaque status inquiry (pending / accepted / rejected / 404)
- GET /v1/concepts: FTS5 search, domain filtering, and pagination
- GET /v1/concepts/{content_hash}: Full concept body retrieval (200 / 404)
- GET /v1/checkpoint: Signed C2SP checkpoint in signed-note text and JSON formats
- GET /v1/tiles/...: Hash tiles (tile/0/0), data tiles (entries/0), and delta packs (packs/1-2)
- GET /v1/stream?since=N: Live Server-Sent Events (SSE) streaming and hint broadcast
- POST /v1/sync & GET /v1/sync: Immediate cutover (410 Gone)
- GET /healthz: Container liveness check
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import time
import zlib
from pathlib import Path
from typing import Any

from aiohttp.test_utils import TestClient, TestServer

from kardashev_cli.opencommons_client import HashcashPoW, OpenCommonsKeyManager
from kardashev_cli.opencommons_hub import HUB_APP_KEY, create_hub_app
from kardashev_cli.opencommons_storage import (
    ConceptRecord,
    SQLiteConceptStorage,
    verify_checkpoint_signature,
)

# ---------------------------------------------------------------------------
# Test Helpers & Fixtures
# ---------------------------------------------------------------------------


def make_valid_proposal(name: str = "fast_inv_sqrt") -> dict[str, Any]:
    return {
        "name": name,
        "domain": "numerical_analysis",
        "param_types": ["float"],
        "return_type": "float",
        "summary": "Fast inverse square root approximation using Newton-Raphson iteration",
        "pseudocode": "xhalf = 0.5 * x; i = *(long*)&x; i = 0x5f3759df - (i >> 1); x = *(float*)&i; x = x*(1.5 - xhalf*x*x)",
        "invariants": ["converges monotonically within 1 iteration", "finite for x > 0"],
        "pitfalls": [
            {
                "naive_pattern": "1.0 / sqrt(x)",
                "failure_mechanism": "High latency on older FPU hardware",
                "counterexample_input": "0.0",
                "verified_fix": "Use magic constant approximation followed by single Newton step",
            }
        ],
        "source_code": (
            f"def {name}(x: float) -> float:\n"
            f"    return 1.0 / (x ** 0.5)\n"
        ),
    }


def compute_hash(proposal: dict[str, Any]) -> str:
    jcs = json.dumps(proposal, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(jcs).hexdigest()


def generate_auth_headers(
    proposal: dict[str, Any],
    key_record: Any | None = None,
    difficulty: int = 16,
) -> dict[str, str]:
    if key_record is None:
        key_record = OpenCommonsKeyManager.generate_key_record()
    content_hash = compute_hash(proposal)
    pow_stamp = HashcashPoW.solve(content_hash, difficulty=difficulty)
    return {
        "Content-Type": "application/json",
        "X-OpenCommons-Key": key_record.public_key_hex,
        "X-OpenCommons-PoW": pow_stamp.format(),
    }


# ---------------------------------------------------------------------------
# 1. Edge Ingestion Caps & Auth Tests (POST /v1/proposals)
# ---------------------------------------------------------------------------


def test_post_proposal_body_cap_enforcement(tmp_path: Path) -> None:
    """Verifies that payloads exceeding the 32KB cap are rejected with 413 Payload Too Large."""
    storage = SQLiteConceptStorage(db_path=tmp_path / "test.db")
    app = create_hub_app(storage=storage)

    async def _test() -> None:
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            # Create a 35 KB payload
            large_body = b"{" + b'"data": "' + (b"A" * 36_000) + b'"}'
            resp = await client.post(
                "/v1/proposals",
                data=large_body,
                headers={"Content-Type": "application/json"},
            )
            assert resp.status == 413
            data = await resp.json()
            assert "exceeds" in data.get("error", "").lower()
        finally:
            await client.close()
            storage.close()

    asyncio.run(_test())


def test_post_proposal_safe_decompression_guard(tmp_path: Path) -> None:
    """Verifies that compressed payloads violating safe decompression limits are rejected (400)."""
    storage = SQLiteConceptStorage(db_path=tmp_path / "test.db")
    app = create_hub_app(storage=storage)

    async def _test() -> None:
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            # 1. Create a compression bomb (> 64 KB decompressed)
            bomb_payload = b'{"name": "test", "blob": "' + (b"0" * 70_000) + b'"}'
            gzipped_bomb = gzip.compress(bomb_payload)

            resp1 = await client.post(
                "/v1/proposals",
                data=gzipped_bomb,
                headers={
                    "Content-Type": "application/json",
                    "Content-Encoding": "gzip",
                    "X-OpenCommons-Key": "a" * 64,
                    "X-OpenCommons-PoW": "dummy",
                },
            )
            assert resp1.status == 400
            data1 = await resp1.json()
            assert "decompressed payload exceeds" in data1.get("error", "").lower()

            # 2. Corrupted gzip stream
            resp2 = await client.post(
                "/v1/proposals",
                data=b"not_really_gzip_data",
                headers={
                    "Content-Type": "application/json",
                    "Content-Encoding": "gzip",
                },
            )
            assert resp2.status == 400
        finally:
            await client.close()
            storage.close()

    asyncio.run(_test())


def test_post_proposal_auth_and_pow_verification(tmp_path: Path) -> None:
    """Verifies that missing/invalid keys (401) and PoW stamps (400) are strictly rejected."""
    storage = SQLiteConceptStorage(db_path=tmp_path / "test.db")
    app = create_hub_app(storage=storage)
    proposal = make_valid_proposal()
    raw_json = json.dumps(proposal).encode("utf-8")

    async def _test() -> None:
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            # 1. Missing X-OpenCommons-Key header -> 401
            resp1 = await client.post(
                "/v1/proposals",
                data=raw_json,
                headers={"Content-Type": "application/json"},
            )
            assert resp1.status == 401

            # 2. Invalid key length -> 401
            resp2 = await client.post(
                "/v1/proposals",
                data=raw_json,
                headers={
                    "Content-Type": "application/json",
                    "X-OpenCommons-Key": "short_key",
                },
            )
            assert resp2.status == 401

            # 3. Missing X-OpenCommons-PoW header -> 400
            key_record = OpenCommonsKeyManager.generate_key_record()
            resp3 = await client.post(
                "/v1/proposals",
                data=raw_json,
                headers={
                    "Content-Type": "application/json",
                    "X-OpenCommons-Key": key_record.public_key_hex,
                },
            )
            assert resp3.status == 400
            data3 = await resp3.json()
            assert "pow" in data3.get("error", "").lower()

            # 4. Forged/invalid PoW nonce -> 400
            forged_pow = f"1:16:{int(time.time())}:{compute_hash(proposal)}::bad_nonce"
            resp4 = await client.post(
                "/v1/proposals",
                data=raw_json,
                headers={
                    "Content-Type": "application/json",
                    "X-OpenCommons-Key": key_record.public_key_hex,
                    "X-OpenCommons-PoW": forged_pow,
                },
            )
            assert resp4.status == 400
        finally:
            await client.close()
            storage.close()

    asyncio.run(_test())


# ---------------------------------------------------------------------------
# 2. Ingestion Queue, Backpressure & 202 Accepted Tests
# ---------------------------------------------------------------------------


def test_post_proposal_success_and_opaque_status(tmp_path: Path) -> None:
    """Verifies 202 Accepted response, status URL query, and worker background acceptance."""
    storage = SQLiteConceptStorage(db_path=tmp_path / "test.db")
    app = create_hub_app(storage=storage)
    proposal = make_valid_proposal(name="euler_totient")
    headers = generate_auth_headers(proposal)
    raw_json = json.dumps(proposal).encode("utf-8")

    async def _test() -> None:
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            # 1. Submit proposal -> 202 Accepted
            resp = await client.post("/v1/proposals", data=raw_json, headers=headers)
            assert resp.status == 202
            data = await resp.json()
            assert data["status"] == "accepted"
            sub_id = data["submission_id"]
            check_url = data["check_url"]
            assert check_url == f"/v1/proposals/{sub_id}"

            # 2. Wait for worker to finish evaluation (max 5s)
            status_data: dict[str, Any] = {}
            for _ in range(50):
                status_resp = await client.get(check_url)
                assert status_resp.status == 200
                status_data = await status_resp.json()
                if status_data.get("status") in ("accepted", "rejected"):
                    break
                await asyncio.sleep(0.1)

            assert status_data["status"] == "accepted"
            assert status_data["submission_id"] == sub_id

            # 3. Verify concept is stored in SQLite WAL database
            rec = storage.get_concept(data["content_hash"])
            assert rec is not None
            assert rec.name == "euler_totient"
            assert rec.status == "accepted"
            assert rec.log_index >= 1

            # 4. Unknown submission ID returns 404
            bad_status_resp = await client.get("/v1/proposals/non-existent-uuid")
            assert bad_status_resp.status == 404
        finally:
            await client.close()
            storage.close()

    asyncio.run(_test())


def test_post_proposal_queue_backpressure_429(tmp_path: Path) -> None:
    """Tests 429 Too Many Requests backpressure with Retry-After when queue is full."""
    storage = SQLiteConceptStorage(db_path=tmp_path / "test.db")
    # Small queue of size 2, with long retry-after
    app = create_hub_app(storage=storage, max_queue_size=2, retry_after_seconds=10)
    hub = app[HUB_APP_KEY]

    async def _test() -> None:
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            # Pause worker task so items remain queued to test backpressure
            await hub.stop_workers()
            hub.job_queue.put_nowait(("dummy1", "h1", {}, "k1"))
            hub.job_queue.put_nowait(("dummy2", "h2", {}, "k2"))
            assert hub.job_queue.full()

            proposal = make_valid_proposal(name="overflow_func")
            headers = generate_auth_headers(proposal)
            raw_json = json.dumps(proposal).encode("utf-8")

            resp = await client.post("/v1/proposals", data=raw_json, headers=headers)
            assert resp.status == 429
            assert resp.headers.get("Retry-After") == "10"
            data = await resp.json()
            assert "overloaded" in data.get("error", "").lower()
            assert data.get("retry_after") == 10
        finally:
            await client.close()
            storage.close()

    asyncio.run(_test())


def test_post_proposal_idempotency(tmp_path: Path) -> None:
    """Verifies that re-submitting an already accepted concept returns immediate acknowledgement."""
    storage = SQLiteConceptStorage(db_path=tmp_path / "test.db")
    app = create_hub_app(storage=storage)
    proposal = make_valid_proposal(name="idempotent_func")
    headers = generate_auth_headers(proposal)
    raw_json = json.dumps(proposal).encode("utf-8")
    c_hash = compute_hash(proposal)

    # Pre-insert concept into storage as accepted
    storage.append_accepted_concept(
        ConceptRecord(
            content_hash=c_hash,
            name="idempotent_func",
            domain="numerical_analysis",
            json_body=proposal,
            status="accepted",
        )
    )

    async def _test() -> None:
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            resp = await client.post("/v1/proposals", data=raw_json, headers=headers)
            # Idempotency returns 200 OK with existing submission info
            assert resp.status == 200
            data = await resp.json()
            assert data["status"] == "accepted"
            assert data["content_hash"] == c_hash
        finally:
            await client.close()
            storage.close()

    asyncio.run(_test())


# ---------------------------------------------------------------------------
# 3. Concept Query & Retrieval Endpoints (GET /v1/concepts)
# ---------------------------------------------------------------------------


def test_get_concepts_search_and_pagination(tmp_path: Path) -> None:
    """Tests GET /v1/concepts with FTS5 search, domain filtering, and pagination."""
    storage = SQLiteConceptStorage(db_path=tmp_path / "test.db")
    app = create_hub_app(storage=storage)

    # Pre-seed 3 concepts
    storage.append_accepted_concept(
        ConceptRecord(
            content_hash="h1",
            name="cholesky_decomposition",
            domain="numerical_analysis",
            json_body={"name": "cholesky_decomposition", "summary": "Hermitian positive-definite matrix factorization", "domain": "numerical_analysis"},
        )
    )
    storage.append_accepted_concept(
        ConceptRecord(
            content_hash="h2",
            name="qr_factorization",
            domain="numerical_analysis",
            json_body={"name": "qr_factorization", "summary": "Orthogonal triangular matrix decomposition using Householder reflections", "domain": "numerical_analysis"},
        )
    )
    storage.append_accepted_concept(
        ConceptRecord(
            content_hash="h3",
            name="transformer_attention",
            domain="machine_learning",
            json_body={"name": "transformer_attention", "summary": "Scaled dot-product multi-head attention mechanism", "domain": "machine_learning"},
        )
    )

    async def _test() -> None:
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            # 1. Search all concepts
            resp1 = await client.get("/v1/concepts")
            assert resp1.status == 200
            d1 = await resp1.json()
            assert d1["total"] == 3

            # 2. Search query for "factorization"
            resp2 = await client.get("/v1/concepts?q=factorization")
            assert resp2.status == 200
            d2 = await resp2.json()
            assert d2["total"] >= 1
            names = [c["name"] for c in d2["concepts"]]
            assert "qr_factorization" in names or "cholesky_decomposition" in names

            # 3. Domain filtering
            resp3 = await client.get("/v1/concepts?domain=machine_learning")
            assert resp3.status == 200
            d3 = await resp3.json()
            assert d3["total"] == 1
            assert d3["concepts"][0]["name"] == "transformer_attention"

            # 4. Pagination (limit=1, offset=1)
            resp4 = await client.get("/v1/concepts?limit=1&offset=1")
            assert resp4.status == 200
            d4 = await resp4.json()
            assert len(d4["concepts"]) == 1
            assert d4["limit"] == 1
            assert d4["offset"] == 1
        finally:
            await client.close()
            storage.close()

    asyncio.run(_test())


def test_get_concept_body_by_hash(tmp_path: Path) -> None:
    """Tests GET /v1/concepts/{content_hash} for full verified concept body and 404 handling."""
    storage = SQLiteConceptStorage(db_path=tmp_path / "test.db")
    app = create_hub_app(storage=storage)
    proposal = make_valid_proposal(name="body_test_func")
    c_hash = compute_hash(proposal)

    storage.append_accepted_concept(
        ConceptRecord(
            content_hash=c_hash,
            name="body_test_func",
            domain="numerical_analysis",
            json_body=proposal,
        )
    )

    async def _test() -> None:
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            # 1. Existing concept
            resp1 = await client.get(f"/v1/concepts/{c_hash}")
            assert resp1.status == 200
            assert resp1.headers.get("ETag") == f'"{c_hash}"'
            d1 = await resp1.json()
            assert d1["name"] == "body_test_func"
            assert d1["content_hash"] == c_hash

            # 2. Non-existent hash
            resp2 = await client.get("/v1/concepts/non_existent_hash_404")
            assert resp2.status == 404
            d2 = await resp2.json()
            assert "not found" in d2.get("error", "").lower()
        finally:
            await client.close()
            storage.close()

    asyncio.run(_test())


# ---------------------------------------------------------------------------
# 4. Transparency Log & Checkpoint Endpoints
# ---------------------------------------------------------------------------


def test_get_checkpoint_formats(tmp_path: Path) -> None:
    """Tests GET /v1/checkpoint in signed-note plain text and JSON formats."""
    storage = SQLiteConceptStorage(db_path=tmp_path / "test.db")
    app = create_hub_app(storage=storage)

    # Append 1 concept so checkpoint size is 1
    storage.append_accepted_concept(
        ConceptRecord(
            content_hash="h1",
            name="cp_func",
            domain="general",
            json_body={"name": "cp_func", "domain": "general"},
        )
    )

    async def _test() -> None:
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            # 1. Default plain text signed-note
            resp1 = await client.get("/v1/checkpoint")
            assert resp1.status == 200
            assert "text/plain" in resp1.headers.get("Content-Type", "")
            assert "max-age=60" in resp1.headers.get("Cache-Control", "")
            text_body = await resp1.text()
            cp = storage.get_latest_checkpoint()
            assert cp.raw_text.strip() == text_body.strip()
            assert verify_checkpoint_signature(cp, storage.public_key)

            # 2. JSON format via query param
            resp2 = await client.get("/v1/checkpoint?format=json")
            assert resp2.status == 200
            d2 = await resp2.json()
            assert d2["size"] == 1
            assert d2["root_hash"] == cp.root_hash
            assert d2["origin"] == storage.origin

            # 3. JSON format via Accept header
            resp3 = await client.get("/v1/checkpoint", headers={"Accept": "application/json"})
            assert resp3.status == 200
            d3 = await resp3.json()
            assert d3["size"] == 1
        finally:
            await client.close()
            storage.close()

    asyncio.run(_test())


def test_get_tiles_and_packs(tmp_path: Path) -> None:
    """Tests GET /v1/tiles/tile/0/0, entries/0, and packs/1-2."""
    storage = SQLiteConceptStorage(db_path=tmp_path / "test.db")
    app = create_hub_app(storage=storage)

    # Insert 2 concepts
    storage.append_accepted_concept(
        ConceptRecord(
            content_hash="h1",
            name="func_a",
            domain="numerical_analysis",
            json_body={"name": "func_a", "domain": "numerical_analysis"},
        )
    )
    storage.append_accepted_concept(
        ConceptRecord(
            content_hash="h2",
            name="func_b",
            domain="numerical_analysis",
            json_body={"name": "func_b", "domain": "numerical_analysis"},
        )
    )

    async def _test() -> None:
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            # 1. Hash tile level 0, index 0
            resp1 = await client.get("/v1/tiles/tile/0/0")
            assert resp1.status == 200
            assert "immutable" in resp1.headers.get("Cache-Control", "")
            tile_bytes = await resp1.read()
            assert len(tile_bytes) == 2 * 32

            # 2. Data tile index 0
            resp2 = await client.get("/v1/tiles/entries/0")
            assert resp2.status == 200
            assert "immutable" in resp2.headers.get("Cache-Control", "")
            data_bytes = await resp2.read()
            lines = [ln for ln in data_bytes.decode("utf-8").splitlines() if ln]
            assert len(lines) == 2

            # 3. Delta pack from 1 to 2
            resp3 = await client.get("/v1/tiles/packs/1-2")
            assert resp3.status == 200
            pack_bytes = await resp3.read()
            unpacked = json.loads(zlib.decompress(pack_bytes).decode("utf-8"))
            assert unpacked["from"] == 1
            assert unpacked["to"] == 2
            assert len(unpacked["entries"]) == 2

            # 4. Unknown tile path
            resp4 = await client.get("/v1/tiles/invalid/path/404")
            assert resp4.status == 404
        finally:
            await client.close()
            storage.close()

    asyncio.run(_test())


# ---------------------------------------------------------------------------
# 5. Live Server-Sent Events (SSE) Stream Tests
# ---------------------------------------------------------------------------


def test_sse_stream_replay_and_live_broadcast(tmp_path: Path) -> None:
    """Tests GET /v1/stream?since=N replaying past entries and receiving live hints."""
    storage = SQLiteConceptStorage(db_path=tmp_path / "test.db")
    app = create_hub_app(storage=storage)
    hub = app[HUB_APP_KEY]

    # Pre-insert concept 1
    storage.append_accepted_concept(
        ConceptRecord(
            content_hash="h1",
            name="first_concept",
            domain="machine_learning",
            json_body={"name": "first_concept", "domain": "machine_learning"},
        )
    )

    async def _test() -> None:
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            # Connect to stream since 0
            resp = await client.get("/v1/stream?since=0")
            assert resp.status == 200
            assert "text/event-stream" in resp.headers.get("Content-Type", "")

            # Read first replayed chunk
            chunk = await asyncio.wait_for(resp.content.readline(), timeout=3.0)
            assert b"id: 1" in chunk

            # Trigger live broadcast of a new concept
            live_hint = {
                "log_index": 2,
                "entry_type": "concept",
                "content_hash": "h2",
                "domain": "machine_learning",
                "name": "live_concept",
                "created_at": time.time(),
            }
            hub.broadcaster.broadcast(live_hint)

            # Read live broadcast chunk from SSE stream
            event_line = b""
            for _ in range(10):
                line = await asyncio.wait_for(resp.content.readline(), timeout=3.0)
                if b"id: 2" in line:
                    event_line = line
                    break
            assert b"id: 2" in event_line
        finally:
            await client.close()
            storage.close()

    asyncio.run(_test())


# ---------------------------------------------------------------------------
# 6. Legacy Sync Cutover Tests (410 Gone) & Health Check
# ---------------------------------------------------------------------------


def test_legacy_sync_cutover_410_gone(tmp_path: Path) -> None:
    """Verifies that retired legacy sync endpoints return 410 Gone with migration guide."""
    storage = SQLiteConceptStorage(db_path=tmp_path / "test.db")
    app = create_hub_app(storage=storage)

    async def _test() -> None:
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            # POST /v1/sync
            resp1 = await client.post("/v1/sync", data=b"{}")
            assert resp1.status == 410
            d1 = await resp1.json()
            assert "retired" in d1.get("error", "").lower()
            assert "migration" in d1

            # GET /v1/sync
            resp2 = await client.get("/v1/sync")
            assert resp2.status == 410

            # POST /v1/sync-delta
            resp3 = await client.post("/v1/sync-delta", data=b"{}")
            assert resp3.status == 410
        finally:
            await client.close()
            storage.close()

    asyncio.run(_test())


def test_healthz_and_root_dashboard(tmp_path: Path) -> None:
    """Verifies GET /healthz liveness probe and GET / dashboard."""
    storage = SQLiteConceptStorage(db_path=tmp_path / "test.db")
    app = create_hub_app(storage=storage)

    async def _test() -> None:
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            # 1. GET /healthz
            resp1 = await client.get("/healthz")
            assert resp1.status == 200
            d1 = await resp1.json()
            assert d1["status"] == "ok"
            assert d1["service"] == "opencommons-hub"
            assert "checkpoint_size" in d1

            # 2. GET /v1/health
            resp_v1h = await client.get("/v1/health")
            assert resp_v1h.status == 200
            d_v1h = await resp_v1h.json()
            assert d_v1h["status"] == "ok"

            # 3. POST /mcp (Streamable HTTP MCP JSON-RPC 2.0)
            resp_mcp = await client.post(
                "/mcp",
                data=json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            assert resp_mcp.status == 200
            d_mcp = await resp_mcp.json()
            assert d_mcp.get("jsonrpc") == "2.0"
            assert d_mcp.get("id") == 1

            # 3b. POST /mcp with non-object JSON (null, [], number) -> 400 Invalid Request (-32600)
            for non_obj in (b"null", b"[]", b"123", b""):
                resp_bad = await client.post(
                    "/mcp",
                    data=non_obj,
                    headers={"Content-Type": "application/json"},
                )
                assert resp_bad.status == 400
                d_bad = await resp_bad.json()
                assert d_bad["error"]["code"] == -32600

            # 4. GET / (landing page)
            resp2 = await client.get("/")
            assert resp2.status == 200
            html = await resp2.text()
            assert "OpenCommons Production Hub" in html
        finally:
            await client.close()
            storage.close()

    asyncio.run(_test())
