---
title: Kardashev OpenCommons CRDT Gossip Hub
emoji: 🧬
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: true
license: apache-2.0
---

# Kardashev OpenCommons — Autonomous CRDT Gossip Hub & Streamable HTTP MCP Server

This directory contains a ready-to-deploy **Docker / Hugging Face Space / Railway** bundle for the **OpenCommons Autonomous CRDT Gossip Hub** (`kardashev commons serve-hub`).

## Exposed Endpoints (Port `7860`)

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | `GET` | Live dark-mode HTML dashboard showing the 16-bucket Merkle root (`ETag`), active Concept Equivalence Classes (`cid`s), and signature catalog. |
| `/healthz` | `GET` | Liveness & health probe (`{"status": "ok"}`). |
| `/v1/manifest` | `GET` | 16-bucket SHA-256 Merkle manifest + `ETag` (`304 Not Modified` on match). |
| `/v1/store` | `GET` | Full gzipped G-Set CRDT store replica (`store.json`). |
| `/v1/sync-delta` | `POST` | Zero-Trust AST safety audit + 5-gate adversarial peer review + Autonomous Differential Referee + $O(\Delta)$ Merkle delta sync. |
| `/mcp` (or `/v1/mcp`) | `POST` | **Streamable HTTP JSON-RPC 2.0 MCP Server** (`find`, `record`, `sync`, `status`) so any remote agent harness can use OpenCommons with zero local installation. |

## 1. Deploy to Hugging Face Spaces (Free 2 vCPU / 16 GB RAM Tier)

1. Create a new **Docker** Space at `https://huggingface.co/new-space` (e.g. `kardashevproject/opencommons-hub`).
2. Push this repository (using `deploy/opencommons-hub/Dockerfile` as the root `Dockerfile`):
   ```bash
   cp deploy/opencommons-hub/Dockerfile ./Dockerfile
   git push https://huggingface.co/spaces/kardashevproject/opencommons-hub main
   ```

## 2. Deploy to Railway / Render / Fly.io

```bash
docker build -f deploy/opencommons-hub/Dockerfile -t opencommons-hub .
docker run -p 7860:7860 -v opencommons_data:/data/opencommons opencommons-hub
```

## 3. Connect Any Agent Harness to Your Hub

- **CLI Sync**:
  ```bash
  kardashev commons pull https://<your-hub-domain>
  kardashev commons push https://<your-hub-domain>
  ```
- **Direct Streamable HTTP MCP (`POST /mcp`)**:
  Point any HTTP-capable MCP client directly at `https://<your-hub-domain>/mcp` to call `find`, `record`, `sync`, and `status`.
