# OpenCommons Production Hub Deployment (`opencommons-hub`)

The OpenCommons Production Hub serves the append-only transparency log, FTS5-indexed concept catalog, C2SP Merkle tiles, SSE live hint stream, and Streamable HTTP MCP endpoint for coding agents and developers worldwide.

- **Public Cloud Endpoint**: `https://commons.kardashevproject.org`
- **Transparency Log Origin**: `opencommons.kardashevproject.org/tlog/v1`
- **Public Verification Key (Ed25519)**: `1171c0c7070ed279dad0cab8be637614a537af9381260dc8f7028c36d416ad73`

## 1. Public HTTP & MCP Endpoints

| Endpoint | Method | Description |
| --- | --- | --- |
| `/v1/health` (or `/healthz`) | `GET` | Hub liveness, Ed25519 public key, checkpoint size, and Merkle root hash |
| `/v1/checkpoint` | `GET` | C2SP signed-note checkpoint (`text/plain` or `?format=json`) |
| `/v1/concepts` | `GET` | FTS5 BM25 search (`?q=...&domain=...&limit=50&offset=0`) |
| `/v1/concepts/{content_hash}` | `GET` | Full verified concept record (`ETag` cached) |
| `/v1/tiles/tile/{L}/{N}` | `GET` | C2SP 32-byte Merkle tree hash tiles (`application/octet-stream`) |
| `/v1/tiles/entries/{N}` | `GET` | C2SP transparency log entry bundle (256 entries per bundle) |
| `/v1/tiles/packs/{from}-{to}` | `GET` | Compressed delta pack between log indices |
| `/v1/proposals` | `POST` | Submit a verified concept (`X-OpenCommons-Key` + `X-OpenCommons-PoW`) |
| `/v1/proposals/{id}` | `GET` | Poll submission status (`accepted` / `pending` / `rejected`) |
| `/v1/stream?since=N` | `GET` | Live Server-Sent Events (SSE) stream of newly sequenced concepts |
| `/mcp` (or `/v1/mcp`) | `POST` | Streamable HTTP JSON-RPC 2.0 MCP server (`find`, `record`, `status`) |

## 2. Cloudflare Worker + D1 Edge Deployment

`worker.js` and `wrangler.toml` deploy the global edge Hub backed by Cloudflare D1 (`opencommons-db`):

```bash
npx wrangler deploy
```

## 3. Container / Self-Hosted Deployment

Build and run the self-contained Python `aiohttp` + SQLite WAL container from the repository root:

```bash
docker build -f deploy/opencommons-hub/Dockerfile -t opencommons-hub:latest .
docker run -d -p 7860:7860 -v opencommons-data:/data opencommons-hub:latest
```

On first startup, `run_hub_server()` automatically persists the Ed25519 checkpoint signing key in `hub_metadata` and seeds all 601 verified concepts if the database is empty.
