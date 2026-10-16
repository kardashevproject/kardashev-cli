---
title: Kardashev OpenCommons Production Hub
emoji: 🧬
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: true
license: apache-2.0
---

# Kardashev OpenCommons — Production Transparency Log & Ingestion Hub

This directory contains deployment assets for the **OpenCommons Production Transparency Log & Ingestion Hub** (`kardashev commons serve-hub`).

OpenCommons is The Kardashev Project's decentralized, privacy-preserving, and tamper-resistant open-science concept commons.

## Architecture

The production hub runs an asynchronous HTTP server powered by SQLite in Write-Ahead Logging (WAL) mode, implementing:
- **C2SP Append-Only Transparency Log (tlog-tiles)**: immutable Merkle hash tiles, data tiles, and signed checkpoints.
- **Fail-Closed Async Ingestion**: 32KB payload cap, safe decompression guard (64KB, 5x ratio), Ed25519 key authentication, Hashcash Proof-of-Work, and bounded job queue with `429 Retry-After` backpressure.
- **Live Streaming**: Server-Sent Events (SSE) broadcasting sequential `log_index` hints to connected client agents.
- **Immediate Cutover**: Retired legacy sync (`/v1/sync`) returns `410 Gone`.

## Exposed Endpoints (Port `7860`)

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | `GET` | Landing dashboard showing checkpoint size, root hash, and status. |
| `/healthz` | `GET` | Liveness & health probe (`{"status": "ok", "service": "opencommons-hub"}`). |
| `/v1/proposals` | `POST` | Asynchronous submission ingestion (enforces 32KB body cap, safe decompression, Ed25519 auth, PoW, and queue backpressure). Returns `202 Accepted`. |
| `/v1/proposals/{id}` | `GET` | Opaque submission status query (`pending`, `accepted`, `rejected`). |
| `/v1/concepts` | `GET` | Searchable concept catalog using FTS5 full-text indexing, domain filtering, and pagination. |
| `/v1/concepts/{content_hash}` | `GET` | Full verified concept body by RFC 8785 content hash. |
| `/v1/checkpoint` | `GET` | Latest signed C2SP checkpoint (`text/plain` signed-note or JSON). |
| `/v1/tiles/{path:.*}` | `GET` | Static transparency log hash tiles (`tile/<L>/<N>`), data tiles (`entries/<N>`), and delta packs (`packs/<from>-<to>`). |
| `/v1/stream?since=N` | `GET` | Live Server-Sent Events (SSE) streaming broadcasting log hints. |
| `/v1/sync` | `POST`/`GET` | Retired legacy sync endpoint (returns `410 Gone`). |

## Deployment Options

### 1. Systemd Service (Linux Server / VM)

1. Create application user and data directory:
   ```bash
   sudo useradd -r -s /bin/false opencommons
   sudo mkdir -p /var/lib/opencommons
   sudo chown -R opencommons:opencommons /var/lib/opencommons
   ```
2. Copy systemd unit file:
   ```bash
   sudo cp deploy/opencommons-hub/opencommons-hub.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable --now opencommons-hub
   ```
3. Check status:
   ```bash
   sudo systemctl status opencommons-hub
   ```

### 2. Docker Container

```bash
docker build -f deploy/opencommons-hub/Dockerfile -t opencommons-hub .
docker run -d -p 7860:7860 -v opencommons_data:/data/opencommons --name opencommons-hub opencommons-hub
```

### 3. Hugging Face Spaces / Container Cloud

1. Create a new **Docker** Space at `https://huggingface.co/new-space`.
2. Push repository using `deploy/opencommons-hub/Dockerfile` as root Dockerfile.
3. Hub automatically initializes and serves on port `7860`.
