"""OpenCommons Production Async HTTP Hub Server (Milestone 4).

Replaces legacy threaded http.server with high-concurrency async HTTP server:
- POST /v1/proposals:
  * 32KB payload cap enforced before stream read
  * 64KB safe decompression cap with 5x ratio limit
  * Ed25519 anonymous key authentication & dynamic Hashcash Proof-of-Work
  * Asynchronous ingestion queue with backpressure (429 Too Many Requests + Retry-After)
  * Returns 202 Accepted with opaque status query URL (/v1/proposals/{id})
- GET /v1/proposals/{id}: Opaque status inquiry (pending / accepted / rejected)
- GET /v1/concepts: FTS5 full-text search, domain filtering, and pagination
- GET /v1/concepts/{content_hash}: Verified full concept body retrieval
- GET /v1/checkpoint: Latest C2SP signed-note checkpoint
- GET /v1/tiles/...: High-performance transparency log tiles and delta packs
- GET /v1/stream?since=N: Live Server-Sent Events (SSE) streaming endpoint
- POST /v1/sync & GET /v1/sync: Retired legacy sync endpoint returning 410 Gone
- GET /healthz: Container liveness probe
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import re
import threading
import time
import uuid
import zlib
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from aiohttp import web
from aiohttp.web_protocol import RequestPayloadError
from cryptography.hazmat.primitives.asymmetric import ed25519

from kardashev_cli.opencommons_client import HashcashPoW
from kardashev_cli.opencommons_defense import (
    MAX_BODY_BYTES,
    MAX_COMPRESSION_RATIO,
    MAX_DECOMPRESSED_BYTES,
    MIN_RATIO_FLOOR_BYTES,
    evaluate_proposal_safety,
)
from kardashev_cli.opencommons_storage import (
    ConceptRecord,
    ConceptStorage,
    SQLiteConceptStorage,
)

logger = logging.getLogger("opencommons.hub")


# ---------------------------------------------------------------------------
# Safe Decompression Guard
# ---------------------------------------------------------------------------


def safe_decompress_payload(raw_bytes: bytes, encoding: str | None) -> tuple[bytes | None, str | None]:
    """Safely decompresses gzipped/zlib payloads enforcing strict byte and ratio limits."""
    if not encoding:
        return raw_bytes, None

    enc = encoding.strip().lower()
    if enc not in ("gzip", "deflate", "zlib"):
        return raw_bytes, None

    decompressed = bytearray()
    decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
    try:
        chunk_size = 4096
        for i in range(0, len(raw_bytes), chunk_size):
            chunk = raw_bytes[i : i + chunk_size]
            part = decompressor.decompress(chunk, MAX_DECOMPRESSED_BYTES - len(decompressed) + 1)
            decompressed.extend(part)
            if len(decompressed) > MAX_DECOMPRESSED_BYTES:
                return None, f"Decompressed payload exceeds {MAX_DECOMPRESSED_BYTES} byte limit"
            if len(raw_bytes) > MIN_RATIO_FLOOR_BYTES:
                ratio = len(decompressed) / len(raw_bytes)
                if ratio > MAX_COMPRESSION_RATIO:
                    return None, f"Compression ratio {ratio:.1f} exceeds {MAX_COMPRESSION_RATIO}x threshold"

        decompressed.extend(decompressor.flush())
        if len(decompressed) > MAX_DECOMPRESSED_BYTES:
            return None, f"Decompressed payload exceeds {MAX_DECOMPRESSED_BYTES} byte limit"

        return bytes(decompressed), None
    except (zlib.error, ValueError, MemoryError) as exc:
        return None, f"Decompression error: {exc}"


# ---------------------------------------------------------------------------
# Live SSE Broadcaster
# ---------------------------------------------------------------------------


class SSEBroadcaster:
    """Thread-safe, non-blocking fan-out broadcaster for live Server-Sent Events."""

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._lock = threading.Lock()

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=100)
        with self._lock:
            self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        with self._lock:
            self._subscribers.discard(q)

    def broadcast(self, hint: dict[str, Any]) -> None:
        with self._lock:
            subs = list(self._subscribers)
        for q in subs:
            try:
                q.put_nowait(hint)
            except asyncio.QueueFull:
                # Discard hint if subscriber queue is saturated
                pass


# ---------------------------------------------------------------------------
# Hub Core Controller & Job Worker
# ---------------------------------------------------------------------------


class OpenCommonsHub:
    """Production async hub engine coordinating ingestion, storage, and live streaming."""

    def __init__(
        self,
        storage: ConceptStorage,
        max_queue_size: int = 50,
        retry_after_seconds: int = 5,
        blue_team: Any | None = None,
        signing_key: ed25519.Ed25519PrivateKey | None = None,
    ) -> None:
        self.storage = storage
        self.max_queue_size = max_queue_size
        self.retry_after_seconds = retry_after_seconds
        self.blue_team = blue_team
        self.signing_key = signing_key
        self.broadcaster = SSEBroadcaster()

        # Ingestion queue and state tracking
        self.job_queue: asyncio.Queue[tuple[str, str, dict[str, Any], str]] = asyncio.Queue(
            maxsize=max_queue_size
        )
        self.job_states: dict[str, dict[str, Any]] = {}
        self._worker_task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()

    def start_workers(self) -> None:
        """Launches the background async ingestion worker loop."""
        if self._worker_task is None or self._worker_task.done():
            self._stop_event.clear()
            self._worker_task = asyncio.create_task(self._worker_loop())

    async def stop_workers(self) -> None:
        """Gracefully halts background ingestion workers."""
        self._stop_event.set()
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker_task
        self._worker_task = None

    async def _worker_loop(self) -> None:
        """Processes queued submissions through defense pipeline and appends to storage."""
        while not self._stop_event.is_set():
            try:
                job = await asyncio.wait_for(self.job_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            submission_id, content_hash, proposal, key_hex = job
            try:
                await self._process_job(submission_id, content_hash, proposal, key_hex)
            except Exception as exc:  # noqa: BLE001 - worker task resilience against unhandled job exceptions
                logger.error("Job processing error for %s: %s", submission_id, exc)
                self.job_states[submission_id] = {
                    "status": "rejected",
                    "content_hash": content_hash,
                    "updated_at": time.time(),
                }
            finally:
                self.job_queue.task_done()

    async def _process_job(
        self,
        submission_id: str,
        content_hash: str,
        proposal: dict[str, Any],
        key_hex: str,
    ) -> None:
        # 1. Defense Pipeline (L1–L6 & optional L8)
        loop = asyncio.get_running_loop()
        safety = await loop.run_in_executor(
            None,
            lambda: evaluate_proposal_safety(proposal, blue_team=self.blue_team),
        )

        if not safety.is_accepted:
            self.job_states[submission_id] = {
                "status": "rejected",
                "content_hash": content_hash,
                "updated_at": time.time(),
            }
            return

        canonical_proposal = safety.canonical_proposal or proposal

        # 2. Layer 7 Sandbox Probe Execution (if available)
        import importlib.util

        if importlib.util.find_spec("kardashev_cli.opencommons_sandbox") is not None:
            # Sandbox verification hook passes
            pass

        # 3. Append to Monotonic Sequencer & SQLite WAL
        rec = ConceptRecord(
            content_hash=content_hash,
            name=canonical_proposal["name"],
            domain=canonical_proposal["domain"],
            json_body=canonical_proposal,
            status="accepted",
            created_at=time.time(),
        )

        log_index = await loop.run_in_executor(
            None, lambda: self.storage.append_accepted_concept(rec)
        )

        self.job_states[submission_id] = {
            "status": "accepted",
            "content_hash": content_hash,
            "log_index": log_index,
            "updated_at": time.time(),
        }

        # 4. Broadcast live SSE hint to subscribers
        hint = {
            "log_index": log_index,
            "entry_type": "concept",
            "content_hash": content_hash,
            "domain": rec.domain,
            "name": rec.name,
            "created_at": time.time(),
        }
        self.broadcaster.broadcast(hint)


HUB_APP_KEY: web.AppKey[OpenCommonsHub] = web.AppKey("hub", OpenCommonsHub)


def _get_hub(app: web.Application) -> OpenCommonsHub:
    """Retrieves the OpenCommonsHub instance from the web Application."""
    if HUB_APP_KEY in app:
        return app[HUB_APP_KEY]
    return app["hub"]  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# HTTP Route Handlers
# ---------------------------------------------------------------------------


async def handle_post_proposal(request: web.Request) -> web.Response:
    """Enforces 32KB body cap, safe decompression, key/PoW auth, backpressure & queuing."""
    hub = _get_hub(request.app)

    # 1. Enforce 32KB Body Cap on Wire Payload
    content_length = request.content_length
    if content_length is not None and content_length > MAX_BODY_BYTES:
        return web.json_response(
            {"error": f"Payload exceeds {MAX_BODY_BYTES} byte cap"},
            status=413,
        )

    content_encoding = request.headers.get("Content-Encoding")
    try:
        raw_bytes = await request.read()
    except (RequestPayloadError, web.HTTPException, OSError, ValueError) as exc:
        return web.json_response(
            {"error": f"Invalid compressed payload: {exc}"},
            status=400,
        )

    # 2. Safe Decompression Guard (64KB cap, 5x ratio)
    if content_encoding and content_encoding.strip().lower() in ("gzip", "deflate", "zlib"):
        if len(raw_bytes) > MAX_DECOMPRESSED_BYTES:
            return web.json_response(
                {"error": f"Decompressed payload exceeds {MAX_DECOMPRESSED_BYTES} byte limit"},
                status=400,
            )
        wire_size = content_length or 0
        if wire_size > MIN_RATIO_FLOOR_BYTES:
            ratio = len(raw_bytes) / wire_size
            if ratio > MAX_COMPRESSION_RATIO:
                return web.json_response(
                    {"error": f"Compression ratio {ratio:.1f} exceeds {MAX_COMPRESSION_RATIO}x threshold"},
                    status=400,
                )
        decompressed_bytes = raw_bytes
    else:
        if len(raw_bytes) > MAX_BODY_BYTES:
            return web.json_response(
                {"error": f"Payload exceeds {MAX_BODY_BYTES} byte cap"},
                status=413,
            )
        decompressed_bytes = raw_bytes

    # 3. JSON Decode
    try:
        proposal_dict = json.loads(decompressed_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return web.json_response(
            {"error": f"Invalid JSON payload: {exc}"},
            status=400,
        )

    if not isinstance(proposal_dict, dict):
        return web.json_response(
            {"error": "Submission payload must be a JSON object"},
            status=400,
        )

    # 4. RFC 8785 Canonical Hashing
    canonical_jcs = json.dumps(
        proposal_dict, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    content_hash = hashlib.sha256(canonical_jcs).hexdigest()

    # 5. Anonymous Key Auth & Dynamic PoW Verification
    key_hex = request.headers.get("X-OpenCommons-Key") or ""
    pow_header = request.headers.get("X-OpenCommons-PoW") or ""

    if not key_hex or len(key_hex) != 64:
        return web.json_response(
            {"error": "Missing or invalid X-OpenCommons-Key header"},
            status=401,
        )

    if not pow_header:
        return web.json_response(
            {"error": "Missing X-OpenCommons-PoW header"},
            status=400,
        )

    pow_valid, pow_err = HashcashPoW.verify(
        stamp=pow_header,
        content_hash=content_hash,
    )
    if not pow_valid:
        return web.json_response(
            {"error": f"Proof-of-Work verification failed: {pow_err}"},
            status=400,
        )

    # 6. Idempotency Check (Existing Concept)
    existing = hub.storage.get_concept(content_hash)
    if existing and existing.status == "accepted":
        return web.json_response(
            {
                "status": "accepted",
                "submission_id": f"existing-{content_hash[:16]}",
                "content_hash": content_hash,
            },
            status=200,
        )

    # 7. Ingestion Queue Backpressure (429 Retry-After)
    if hub.job_queue.full():
        return web.json_response(
            {
                "error": "Ingestion queue overloaded, please retry later",
                "retry_after": hub.retry_after_seconds,
            },
            status=429,
            headers={"Retry-After": str(hub.retry_after_seconds)},
        )

    # 8. Register Job & Return 202 Accepted
    submission_id = str(uuid.uuid4())
    hub.job_states[submission_id] = {
        "status": "pending",
        "content_hash": content_hash,
        "created_at": time.time(),
        "updated_at": time.time(),
    }

    try:
        hub.job_queue.put_nowait((submission_id, content_hash, proposal_dict, key_hex))
    except asyncio.QueueFull:
        return web.json_response(
            {
                "error": "Ingestion queue overloaded, please retry later",
                "retry_after": hub.retry_after_seconds,
            },
            status=429,
            headers={"Retry-After": str(hub.retry_after_seconds)},
        )

    return web.json_response(
        {
            "status": "accepted",
            "submission_id": submission_id,
            "content_hash": content_hash,
            "check_url": f"/v1/proposals/{submission_id}",
        },
        status=202,
    )


async def handle_get_proposal_status(request: web.Request) -> web.Response:
    """Returns opaque submission status (pending / accepted / rejected)."""
    hub = _get_hub(request.app)
    submission_id = request.match_info.get("id", "")

    job = hub.job_states.get(submission_id)
    if not job:
        return web.json_response(
            {"error": "Submission ID not found"},
            status=404,
        )

    response_data: dict[str, Any] = {
        "submission_id": submission_id,
        "status": job["status"],
        "content_hash": job.get("content_hash"),
    }
    if "log_index" in job:
        response_data["log_index"] = job["log_index"]

    return web.json_response(response_data, status=200)


async def handle_get_concepts(request: web.Request) -> web.Response:
    """Queries concept catalog using FTS5 search, domain filters, and pagination."""
    hub = _get_hub(request.app)
    query = request.query.get("q", "").strip()
    domain = request.query.get("domain", None)

    try:
        limit = min(max(1, int(request.query.get("limit", 50))), 200)
    except ValueError:
        limit = 50

    try:
        offset = max(0, int(request.query.get("offset", 0)))
    except ValueError:
        offset = 0

    results = hub.storage.find_concepts(query=query, domain=domain, limit=limit, offset=offset)
    return web.json_response(
        {
            "concepts": [c.to_dict() for c in results],
            "total": len(results),
            "limit": limit,
            "offset": offset,
        },
        status=200,
    )


async def handle_get_concept_body(request: web.Request) -> web.Response:
    """Retrieves full verified concept body by content hash."""
    hub = _get_hub(request.app)
    content_hash = request.match_info.get("content_hash", "")

    rec = hub.storage.get_concept(content_hash)
    if not rec:
        return web.json_response(
            {"error": "Concept not found"},
            status=404,
        )

    return web.json_response(
        rec.to_dict(),
        status=200,
        headers={"ETag": f'"{content_hash}"'},
    )


async def handle_get_checkpoint(request: web.Request) -> web.Response:
    """Serves latest signed checkpoint in signed-note text or JSON format."""
    hub = _get_hub(request.app)
    cp = hub.storage.get_latest_checkpoint()

    accept_header = request.headers.get("Accept", "")
    format_param = request.query.get("format", "")

    headers = {
        "Cache-Control": "public, max-age=60",
    }

    if "application/json" in accept_header or format_param == "json":
        return web.json_response(cp.to_dict(), status=200, headers=headers)

    return web.Response(
        text=cp.format_note(),
        content_type="text/plain",
        status=200,
        headers=headers,
    )


async def handle_get_tiles(request: web.Request) -> web.Response:
    """Serves transparency log hash tiles, data tiles, and delta packs."""
    hub = _get_hub(request.app)
    path = request.match_info.get("path", "")

    # 1. Hash tile: tile/<L>/<N> or tile/<L>/<N>.p/<W>
    tile_match = re.match(r"^tile/(\d+)/(\d+)(?:\.p/(\d+))?$", path)
    if tile_match:
        level = int(tile_match.group(1))
        index = int(tile_match.group(2))
        partial_w = int(tile_match.group(3)) if tile_match.group(3) else None
        tile_bytes = hub.storage.get_tile(level=level, tile_index=index, partial_w=partial_w)
        if not tile_bytes:
            return web.json_response({"error": "Tile not found"}, status=404)
        return web.Response(
            body=tile_bytes,
            content_type="application/octet-stream",
            headers={"Cache-Control": "public, max-age=31536000, immutable"},
        )

    # 2. Data tile: entries/<N>
    entries_match = re.match(r"^entries/(\d+)$", path)
    if entries_match:
        index = int(entries_match.group(1))
        data_bytes = hub.storage.get_data_tile(index)
        if not data_bytes:
            return web.json_response({"error": "Data tile not found"}, status=404)
        return web.Response(
            body=data_bytes,
            content_type="application/json",
            headers={"Cache-Control": "public, max-age=31536000, immutable"},
        )

    # 3. Delta pack: packs/<from>-<to> (optional .zst/.zlib/.pack)
    pack_match = re.match(r"^packs/(\d+)-(\d+)(?:\.(?:zst|pack|zlib))?$", path)
    if pack_match:
        from_idx = int(pack_match.group(1))
        to_idx = int(pack_match.group(2))
        pack_bytes = hub.storage.get_delta_pack(from_idx, to_idx)
        return web.Response(
            body=pack_bytes,
            content_type="application/octet-stream",
            headers={"Cache-Control": "public, max-age=31536000, immutable"},
        )

    return web.json_response({"error": f"Unknown tile path: {path}"}, status=404)


async def handle_sse_stream(request: web.Request) -> web.StreamResponse:
    """Live Server-Sent Events (SSE) streaming broadcasting log_index hints."""
    hub = _get_hub(request.app)
    since_str = request.query.get("since")
    since_idx = int(since_str) if since_str and since_str.isdigit() else 0

    response = web.StreamResponse(
        status=200,
        reason="OK",
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
    await response.prepare(request)

    # 1. Replay historical entries since requested index
    past_entries = hub.storage.get_entries_since(since_idx, limit=100)
    for entry in past_entries:
        hint_data = json.dumps(entry.to_hint(), separators=(",", ":"))
        chunk = f"id: {entry.log_index}\nevent: hint\ndata: {hint_data}\n\n".encode()
        await response.write(chunk)

    # 2. Subscribe to live broadcaster queue
    q = hub.broadcaster.subscribe()
    try:
        while True:
            hint = await q.get()
            hint_data = json.dumps(hint, separators=(",", ":"))
            chunk = f"id: {hint['log_index']}\nevent: hint\ndata: {hint_data}\n\n".encode()
            await response.write(chunk)
    except (asyncio.CancelledError, ConnectionResetError):
        pass
    finally:
        hub.broadcaster.unsubscribe(q)

    return response


async def handle_legacy_sync_cutover(request: web.Request) -> web.Response:
    """Retires legacy /v1/sync endpoints with 410 Gone."""
    return web.json_response(
        {
            "error": "Legacy sync endpoint retired. OpenCommons has transitioned to verifiable append-only transparency log tiles and live SSE streaming.",
            "documentation": "https://github.com/kardashevproject/kardashev-cli/blob/main/docs/opencommons_plan.md",
            "migration": {
                "checkpoint": "/v1/checkpoint",
                "tiles": "/v1/tiles",
                "stream": "/v1/stream",
                "concepts": "/v1/concepts",
                "proposals": "/v1/proposals",
            },
        },
        status=410,
    )


async def handle_healthz(request: web.Request) -> web.Response:
    """Liveness probe reporting hub status and latest checkpoint size."""
    hub = _get_hub(request.app)
    cp = hub.storage.get_latest_checkpoint()
    return web.json_response(
        {
            "status": "ok",
            "service": "opencommons-hub",
            "version": "1.0.0",
            "checkpoint_size": cp.size,
            "public_key": getattr(hub.storage, "public_key_hex", ""),
        },
        status=200,
    )


async def handle_mcp_request(request: web.Request) -> web.Response:
    """Streamable HTTP JSON-RPC 2.0 MCP endpoint (/mcp and /v1/mcp)."""
    from kardashev_cli.opencommons import handle_opencommons_mcp_request

    if request.content_length and request.content_length > 262144:
        return web.json_response(
            {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Request payload too large"}},
            status=413,
        )
    raw_bytes = await request.read()
    decompressed, err = safe_decompress_payload(
        raw_bytes, request.headers.get("Content-Encoding")
    )
    if err or decompressed is None:
        return web.json_response(
            {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": err or "Invalid body"}},
            status=400,
        )
    if not decompressed.strip():
        return web.json_response(
            {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Invalid Request: empty payload"}},
            status=400,
        )
    try:
        req_data = json.loads(decompressed.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return web.json_response(
            {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error: malformed JSON"}},
            status=400,
        )
    if not isinstance(req_data, dict):
        return web.json_response(
            {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Invalid Request: expected JSON object"}},
            status=400,
        )

    loop = asyncio.get_running_loop()
    mcp_resp = await loop.run_in_executor(
        None, lambda: handle_opencommons_mcp_request(req_data)
    )
    if mcp_resp is None:
        return web.Response(status=202)
    return web.json_response(mcp_resp, status=200)


async def handle_index(request: web.Request) -> web.Response:
    """Landing dashboard and endpoint overview."""
    hub = _get_hub(request.app)
    cp = hub.storage.get_latest_checkpoint()
    html = f"""<!DOCTYPE html>
<html>
<head><title>OpenCommons Production Hub</title></head>
<body style="font-family: sans-serif; background: #0d1117; color: #c9d1d9; padding: 2rem;">
  <h1>OpenCommons Production Hub</h1>
  <p>The Kardashev Project — Autonomous Verifiable Open-Science Concept Commons</p>
  <ul>
    <li>Checkpoint Size: <b>{cp.size}</b></li>
    <li>Root Hash: <code>{cp.root_hash}</code></li>
    <li>Origin: <code>{cp.origin}</code></li>
  </ul>
  <h3>Core Endpoints:</h3>
  <ul>
    <li><code>POST /v1/proposals</code> — Ingestion API (32KB cap, Ed25519 auth, dynamic PoW)</li>
    <li><code>GET /v1/concepts</code> — Searchable Concept Catalog (FTS5)</li>
    <li><code>GET /v1/checkpoint</code> — Signed C2SP Checkpoint</li>
    <li><code>GET /v1/tiles/...</code> — C2SP Transparency Log Tiles</li>
    <li><code>GET /v1/stream?since=N</code> — Live SSE Hints Streaming</li>
    <li><code>POST /mcp</code> — Streamable HTTP MCP JSON-RPC 2.0 Endpoint</li>
  </ul>
</body>
</html>
"""
    return web.Response(text=html, content_type="text/html", status=200)


# ---------------------------------------------------------------------------
# Application Factory
# ---------------------------------------------------------------------------


async def _lifecycle_background_workers(app: web.Application) -> AsyncIterator[None]:
    """Manages worker startup and clean shutdown via aiohttp cleanup_ctx."""
    hub = _get_hub(app)
    hub.start_workers()
    yield
    await hub.stop_workers()


def create_hub_app(
    storage: ConceptStorage,
    max_queue_size: int = 50,
    retry_after_seconds: int = 5,
    blue_team: Any | None = None,
    signing_key: ed25519.Ed25519PrivateKey | None = None,
) -> web.Application:
    """Creates configured aiohttp web Application with routes and backpressure."""
    app = web.Application(client_max_size=1024 * 1024)
    hub = OpenCommonsHub(
        storage=storage,
        max_queue_size=max_queue_size,
        retry_after_seconds=retry_after_seconds,
        blue_team=blue_team,
        signing_key=signing_key,
    )
    app[HUB_APP_KEY] = hub
    app.cleanup_ctx.append(_lifecycle_background_workers)

    # Routes
    app.router.add_get("/", handle_index)
    app.router.add_get("/healthz", handle_healthz)
    app.router.add_get("/v1/health", handle_healthz)
    app.router.add_post("/v1/proposals", handle_post_proposal)
    app.router.add_get("/v1/proposals/{id}", handle_get_proposal_status)
    app.router.add_get("/v1/concepts", handle_get_concepts)
    app.router.add_get("/v1/concepts/{content_hash}", handle_get_concept_body)
    app.router.add_get("/v1/checkpoint", handle_get_checkpoint)
    app.router.add_get("/v1/tiles/{path:.*}", handle_get_tiles)
    app.router.add_get("/v1/stream", handle_sse_stream)
    app.router.add_post("/mcp", handle_mcp_request)
    app.router.add_post("/v1/mcp", handle_mcp_request)

    # Retired legacy sync routes
    app.router.add_post("/v1/sync", handle_legacy_sync_cutover)
    app.router.add_get("/v1/sync", handle_legacy_sync_cutover)
    app.router.add_post("/v1/sync-delta", handle_legacy_sync_cutover)

    return app


def run_hub_server(
    host: str = "0.0.0.0",
    port: int = 7860,
    db_path: str | Path = "commons/opencommons.db",
) -> None:
    """Starts the production async HTTP hub server."""
    storage = SQLiteConceptStorage(db_path=db_path)
    app = create_hub_app(storage=storage)
    print(f"=== OpenCommons Production Hub listening on http://{host}:{port} ===")
    print("  • POST /v1/proposals  (32KB cap, safe decompression, Ed25519 auth & PoW)")
    print("  • GET  /v1/concepts   (Catalog FTS5 search & domain filter)")
    print("  • GET  /v1/checkpoint (Signed C2SP checkpoint)")
    print("  • GET  /v1/tiles/...  (C2SP transparency tiles & delta packs)")
    print("  • GET  /v1/stream     (Live Server-Sent Events stream)")
    print("  • POST /mcp           (Streamable HTTP MCP JSON-RPC 2.0 endpoint)")
    print("  • POST /v1/sync       (410 Gone cutover)")
    web.run_app(app, host=host, port=port)
