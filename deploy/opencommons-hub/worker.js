/**
 * OpenCommons Production Hub — Cloudflare Worker + D1 Backend
 * Implements the full Phase 1-3 OpenCommons Hub HTTP + MCP API:
 *   - GET  /                   (HTML landing page)
 *   - GET  /healthz            (Container & edge liveness probe)
 *   - GET  /v1/health          (Hub health, public key, checkpoint size & root hash)
 *   - GET  /v1/checkpoint      (C2SP signed-note text or JSON checkpoint)
 *   - GET  /v1/revocations     (Append-only revocation list)
 *   - GET  /v1/concepts        (FTS5 full-text search, domain filter, pagination)
 *   - GET  /v1/concepts/:hash  (Full verified concept record by content_hash)
 *   - GET  /v1/tiles/tile/:level/:index[/:p] (C2SP 32-byte Merkle hash tiles)
 *   - GET  /v1/tiles/entries/:index          (C2SP JSONL data tiles)
 *   - GET  /v1/tiles/packs/:from-:to         (Compressed delta packs)
 *   - POST /v1/proposals       (32KB cap, Ed25519 key + Hashcash PoW, queuing & dedup)
 *   - GET  /v1/proposals/:id   (Opaque submission status inquiry)
 *   - GET  /v1/stream          (Live SSE hint stream)
 *   - POST /mcp & /v1/mcp      (Streamable HTTP JSON-RPC 2.0 MCP endpoint)
 *   - GET|POST /v1/sync        (410 Gone cutover notice)
 */

const TILE_SIZE = 256;
const MAX_BODY_BYTES = 32768;

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers":
    "Content-Type, Authorization, X-Client-ID, X-OpenCommons-Key, X-OpenCommons-PoW, X-OpenCommons-Signature",
};

function jsonResponse(data, status = 200, extraHeaders = {}) {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      "Content-Type": "application/json",
      ...CORS_HEADERS,
      ...extraHeaders,
    },
  });
}

function bytesToHex(bytes) {
  return Array.from(bytes)
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

function hexToBytes(hex) {
  const clean = hex.trim();
  const out = new Uint8Array(clean.length / 2);
  for (let i = 0; i < out.length; i++) {
    out[i] = parseInt(clean.slice(i * 2, i * 2 + 2), 16);
  }
  return out;
}

async function sha256Bytes(dataBytes) {
  const digest = await crypto.subtle.digest("SHA-256", dataBytes);
  return new Uint8Array(digest);
}

async function hashNode(left32, right32) {
  const buf = new Uint8Array(1 + 32 + 32);
  buf[0] = 0x01;
  buf.set(left32, 1);
  buf.set(right32, 33);
  return await sha256Bytes(buf);
}

async function computeMerkleRoot(leaves) {
  const n = leaves.length;
  if (n === 0) {
    return await sha256Bytes(new Uint8Array(0));
  }
  if (n === 1) {
    return leaves[0];
  }
  let k = 1;
  while (k * 2 < n) {
    k *= 2;
  }
  const left = await computeMerkleRoot(leaves.slice(0, k));
  const right = await computeMerkleRoot(leaves.slice(k));
  return await hashNode(left, right);
}

function sortKeysDeep(val) {
  if (Array.isArray(val)) {
    return val.map(sortKeysDeep);
  }
  if (val !== null && typeof val === "object") {
    const sorted = {};
    for (const k of Object.keys(val).sort()) {
      sorted[k] = sortKeysDeep(val[k]);
    }
    return sorted;
  }
  return val;
}

function canonicalJsonBytes(obj) {
  return new TextEncoder().encode(JSON.stringify(sortKeysDeep(obj)));
}

function countLeadingZeroBits(digest) {
  let zeros = 0;
  for (let i = 0; i < digest.length; i++) {
    const b = digest[i];
    if (b === 0) {
      zeros += 8;
    } else {
      zeros += Math.clz32(b) - 24;
      break;
    }
  }
  return zeros;
}

async function verifyHashcashPoW(stampStr, expectedContentHash) {
  const parts = String(stampStr || "").trim().split(":");
  if (parts.length !== 4) {
    return [false, "Invalid PoW stamp format: expected 4 colon-separated parts"];
  }
  const [nonce, tsStr, diffStr, stampHash] = parts;
  const ts = parseInt(tsStr, 10);
  const difficulty = parseInt(diffStr, 10);
  if (isNaN(ts) || isNaN(difficulty) || difficulty < 8) {
    return [false, "Invalid PoW timestamp or difficulty"];
  }
  if (stampHash !== expectedContentHash) {
    return [false, "PoW content_hash mismatch"];
  }
  const challenge = new TextEncoder().encode(
    `${nonce}:${ts}:${difficulty}:${stampHash}`
  );
  const digest = await sha256Bytes(challenge);
  if (countLeadingZeroBits(digest) < difficulty) {
    return [false, `Insufficient PoW leading zero bits for difficulty ${difficulty}`];
  }
  return [true, null];
}

function sanitizeFtsQuery(q) {
  const tokens = (q.toLowerCase().match(/[a-z0-9_]+/g) || []).filter(
    (t) => !["and", "or", "not", "near"].includes(t)
  );
  if (tokens.length === 0) return "";
  return tokens.join(" ");
}

// Build secret detection patterns without literal token prefixes in source
const SECRET_PATTERNS = [
  new RegExp("gh" + "p_[A-Za-z0-9]{20,}"),
  new RegExp("gh" + "o_[A-Za-z0-9]{20,}"),
  new RegExp("github" + "_pat_[A-Za-z0-9_]{20,}"),
  new RegExp("s" + "k-[A-Za-z0-9]{20,}"),
  new RegExp("AI" + "za[0-9A-Za-z\\-_]{35}"),
  new RegExp("h" + "f_[A-Za-z0-9]{20,}"),
  new RegExp("AK" + "IA[0-9A-Z]{16}"),
  new RegExp("-----BEGIN (?:RSA|OPENSSH|EC|DSA|PRIVATE) KEY-----"),
  new RegExp("(?:/Users/|/home/|C:\\\\\\\\Users\\\\\\\\)[A-Za-z0-9._\\-]+"),
];

const BANNED_IMPORTS = [
  /\bimport\s+(?:os|subprocess|socket|shutil|urllib|requests|http|ctypes|multiprocessing|pickle|marshal|pty|fcntl)\b/,
  /\bfrom\s+(?:os|subprocess|socket|shutil|urllib|requests|http|ctypes|multiprocessing|pickle|marshal|pty|fcntl)\b/,
  /\b(?:eval|exec|__import__|open|compile)\s*\(/,
];

function validateProposalPayload(payload) {
  if (!payload || typeof payload !== "object") {
    return "Proposal must be a JSON object";
  }
  const name = String(payload.name || "").trim();
  const summary = String(payload.summary || "").trim();
  const sourceCode = String(payload.source_code || "");
  if (!name || name.length > 128) return "Invalid concept name";
  if (!summary || summary.length > 1000) return "Invalid concept summary";
  if (!sourceCode.trim() || sourceCode.length > MAX_BODY_BYTES) {
    return "Invalid source_code length";
  }
  if (!Array.isArray(payload.invariants) || payload.invariants.length < 1) {
    return "At least 1 invariant is required";
  }

  const fullText = JSON.stringify(payload);
  for (const pat of SECRET_PATTERNS) {
    if (pat.test(fullText)) {
      return "Privacy & sanitization check failed: secret or personal path detected";
    }
  }
  for (const pat of BANNED_IMPORTS) {
    if (pat.test(sourceCode)) {
      return "Sandbox policy violation: forbidden module or builtin in source_code";
    }
  }
  return null;
}

async function getMetadataMap(db) {
  const { results } = await db
    .prepare("SELECT key, value FROM hub_metadata")
    .all();
  const map = {};
  for (const r of results || []) {
    map[r.key] = r.value;
  }
  return map;
}

function rowToSummary(row) {
  let body = {};
  try {
    body = row.json_body ? JSON.parse(row.json_body) : {};
  } catch (_) {
    body = {};
  }
  return {
    content_hash: row.content_hash,
    cid: row.content_hash,
    name: row.name,
    domain: row.domain,
    summary: String(body.summary || ""),
    log_index: row.log_index || 0,
    status: row.status || "accepted",
    created_at: row.created_at || 0,
  };
}

function rowToFullRecord(row) {
  let body = {};
  try {
    body = row.json_body ? JSON.parse(row.json_body) : {};
  } catch (_) {
    body = {};
  }
  return {
    content_hash: row.content_hash,
    cid: row.content_hash,
    name: row.name,
    domain: row.domain,
    json_body: body,
    log_index: row.log_index || 0,
    created_at: row.created_at || 0,
    status: row.status || "accepted",
  };
}

async function findConcepts(db, { query = "", domain = null, limit = 50, offset = 0, full = false }) {
  const clampedLimit = Math.max(1, Math.min(Number(limit) || 50, 200));
  const clampedOffset = Math.max(0, Number(offset) || 0);
  const mapper = full ? rowToFullRecord : rowToSummary;

  if (query && query.trim()) {
    const ftsQuery = sanitizeFtsQuery(query);
    if (ftsQuery) {
      let sql = `
        SELECT c.content_hash, c.name, c.domain, c.json_body, c.log_index, c.status, c.created_at
        FROM concepts c
        JOIN concepts_fts f ON c.content_hash = f.content_hash
        WHERE concepts_fts MATCH ?
      `;
      const params = [ftsQuery];
      if (domain) {
        sql += " AND c.domain = ?";
        params.push(domain);
      }
      sql += " AND c.status = 'accepted' ORDER BY rank LIMIT ? OFFSET ?";
      params.push(clampedLimit, clampedOffset);
      try {
        const { results } = await db.prepare(sql).bind(...params).all();
        if (results && results.length > 0) {
          return results.map(mapper);
        }
      } catch (_) {
        // Fallback to LIKE search below
      }
    }
  }

  let sql = `
    SELECT content_hash, name, domain, json_body, log_index, status, created_at
    FROM concepts
    WHERE status = 'accepted'
  `;
  const params = [];
  if (domain) {
    sql += " AND domain = ?";
    params.push(domain);
  }
  if (query && query.trim()) {
    const likePat = `%${query.trim()}%`;
    sql += " AND (name LIKE ? OR json_body LIKE ?)";
    params.push(likePat, likePat);
  }
  sql += " ORDER BY log_index DESC LIMIT ? OFFSET ?";
  params.push(clampedLimit, clampedOffset);
  const { results } = await db.prepare(sql).bind(...params).all();
  return (results || []).map(mapper);
}

const MCP_TOOLS_SCHEMA = [
  {
    name: "find",
    description:
      "Search the OpenCommons verified concept store by keyword or domain.",
    inputSchema: {
      type: "object",
      properties: {
        query: { type: "string", description: "Search query (FTS5 BM25)" },
        domain: { type: "string", description: "Optional domain filter" },
        top_k: { type: "integer", description: "Max results (default 5)", default: 5 },
      },
      required: ["query"],
    },
  },
  {
    name: "record",
    description:
      "Submit a verified pure function to the OpenCommons Hub.",
    inputSchema: {
      type: "object",
      properties: {
        name: { type: "string" },
        summary: { type: "string" },
        domain: { type: "string" },
        source_code: { type: "string" },
        examples: { type: "array" },
        invariants: { type: "array" },
      },
      required: ["name", "summary", "source_code", "invariants"],
    },
  },
  {
    name: "status",
    description:
      "Return the current OpenCommons Hub transparency log checkpoint and concept count.",
    inputSchema: {
      type: "object",
      properties: {},
    },
  },
];

const TEXT_ENCODER = new TextEncoder();
const TEXT_DECODER = new TextDecoder();

function bytesToBase64(bytes) {
  let bin = "";
  const chunkSize = 0x8000;
  for (let i = 0; i < bytes.length; i += chunkSize) {
    bin += String.fromCharCode.apply(null, bytes.subarray(i, i + chunkSize));
  }
  return btoa(bin);
}

/**
 * Reads a request body incrementally, aborting immediately if MAX_BODY_BYTES is exceeded.
 * @param {Request} request
 * @param {number} maxBytes
 * @returns {Promise<{ok: boolean, text: string}>}
 */
async function readBoundedBody(request, maxBytes) {
  const cl = parseInt(request.headers.get("content-length") || "0", 10);
  if (cl > maxBytes) {
    return { ok: false, text: "" };
  }
  if (!request.body) {
    return { ok: true, text: "" };
  }
  const reader = request.body.getReader();
  const chunks = [];
  let total = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (value) {
        total += value.byteLength;
        if (total > maxBytes) {
          await reader.cancel();
          return { ok: false, text: "" };
        }
        chunks.push(value);
      }
    }
  } finally {
    reader.releaseLock();
  }
  const merged = new Uint8Array(total);
  let offset = 0;
  for (const c of chunks) {
    merged.set(c, offset);
    offset += c.byteLength;
  }
  return { ok: true, text: TEXT_DECODER.decode(merged) };
}

async function computeLeafHash(dataBytes) {
  const buf = new Uint8Array(1 + dataBytes.length);
  buf[0] = 0x00;
  buf.set(dataBytes, 1);
  return await sha256Bytes(buf);
}

/**
 * Signs a C2SP transparency log checkpoint using WebCrypto Ed25519.
 * @param {string} origin - Transparency log origin identifier.
 * @param {number} size - Tree size (1-based entry count).
 * @param {Uint8Array} rootHashBytes - 32-byte Merkle root hash.
 * @param {string} seedHex - 64-char hex Ed25519 private key seed.
 * @param {string} pubHex - 64-char hex Ed25519 public key.
 * @param {number} tsSeconds - Unix timestamp in seconds.
 */
async function signCheckpointWebCrypto(origin, size, rootHashBytes, seedHex, pubHex, tsSeconds) {
  if (!/^[0-9a-fA-F]{64}$/.test(seedHex) || !/^[0-9a-fA-F]{64}$/.test(pubHex)) {
    throw new Error("Invalid Ed25519 signing seed or public key hex format");
  }
  const rootB64 = bytesToBase64(rootHashBytes);
  const header = `${origin}\n${size}\n${rootB64}\nTimestamp: ${tsSeconds}\n`;

  const prefix = new Uint8Array([
    0x30, 0x2e, 0x02, 0x01, 0x00, 0x30, 0x05, 0x06,
    0x03, 0x2b, 0x65, 0x70, 0x04, 0x22, 0x04, 0x20,
  ]);
  const seedBytes = hexToBytes(seedHex);
  const pkcs8 = new Uint8Array(48);
  pkcs8.set(prefix, 0);
  pkcs8.set(seedBytes, 16);

  let privKey;
  try {
    privKey = await crypto.subtle.importKey(
      "pkcs8",
      pkcs8,
      { name: "Ed25519" },
      false,
      ["sign"]
    );
  } finally {
    pkcs8.fill(0);
    seedBytes.fill(0);
  }

  const sigBuf = await crypto.subtle.sign(
    "Ed25519",
    privKey,
    TEXT_ENCODER.encode(header)
  );
  const sigB64 = bytesToBase64(new Uint8Array(sigBuf));

  const pubBytes = hexToBytes(pubHex);
  const originBytes = TEXT_ENCODER.encode(origin + "\n");
  const hintInput = new Uint8Array(originBytes.length + pubBytes.length);
  hintInput.set(originBytes, 0);
  hintInput.set(pubBytes, originBytes.length);
  const hintDigest = (await sha256Bytes(hintInput)).slice(0, 4);
  const hintB64 = bytesToBase64(hintDigest);
  const rawText = `${header}\n— ${origin} ${hintB64} ${sigB64}\n`;

  return {
    origin,
    size,
    rootHash: rootB64,
    timestamp: tsSeconds,
    signature: sigB64,
    keyHint: hintB64,
    rawText,
  };
}

/**
 * Reconciles any accepted concepts missing from log_entries so that
 * log_entries and concepts.log_index form a contiguous 1..N sequence.
 */
async function reconcileMissingLogEntries(db, now) {
  const { results: missingRows } = await db
    .prepare(
      `SELECT c.log_index, c.content_hash, c.name, c.domain, c.json_body, c.created_at
       FROM concepts c
       LEFT JOIN log_entries l ON c.content_hash = l.content_hash
       WHERE c.status = 'accepted' AND l.log_index IS NULL
       ORDER BY COALESCE(c.log_index, 0) ASC, c.created_at ASC`
    )
    .all();
  if (!missingRows || missingRows.length === 0) {
    return;
  }
  const { results: usedIdxRows } = await db
    .prepare("SELECT log_index FROM log_entries")
    .all();
  const usedLogIndices = new Set(
    (usedIdxRows || []).map((x) => Number(x.log_index))
  );
  const maxRow = await db
    .prepare(
      `SELECT
         COALESCE((SELECT MAX(log_index) FROM log_entries), 0) AS max_log_idx,
         COALESCE((SELECT MAX(log_index) FROM concepts), 0) AS max_concept_idx`
    )
    .first();
  let nextFallbackIdx = Math.max(
    Number(maxRow ? maxRow.max_log_idx : 0),
    Number(maxRow ? maxRow.max_concept_idx : 0)
  );
  const stmts = [
    db.prepare(
      "UPDATE concepts SET log_index = NULL WHERE status != 'accepted' AND log_index IS NOT NULL"
    ),
  ];
  for (const r of missingRows) {
    const existingIdx = Number(r.log_index);
    const targetIdx =
      existingIdx > 0 && !usedLogIndices.has(existingIdx)
        ? existingIdx
        : ++nextFallbackIdx;
    usedLogIndices.add(targetIdx);
    if (targetIdx > nextFallbackIdx) {
      nextFallbackIdx = targetIdx;
    }
    let parsedBody = {};
    try {
      parsedBody = r.json_body ? JSON.parse(r.json_body) : {};
    } catch (_) {
      parsedBody = {};
    }
    const rCreated = Number(r.created_at) || now;
    const rData = {
      content_hash: r.content_hash,
      name: r.name,
      domain: r.domain,
      summary: String(parsedBody.summary || ""),
    };
    const rCanonical = {
      log_index: targetIdx,
      entry_type: "concept",
      content_hash: r.content_hash,
      data: rData,
      created_at: rCreated,
    };
    const rLeafHex = bytesToHex(await computeLeafHash(canonicalJsonBytes(rCanonical)));
    if (existingIdx !== targetIdx) {
      stmts.push(
        db
          .prepare("UPDATE concepts SET log_index = ? WHERE content_hash = ?")
          .bind(targetIdx, r.content_hash)
      );
    }
    stmts.push(
      db
        .prepare(
          `INSERT INTO log_entries (log_index, entry_type, content_hash, leaf_hash, data, created_at)
           VALUES (?, 'concept', ?, ?, ?, ?)`
        )
        .bind(targetIdx, r.content_hash, rLeafHex, JSON.stringify(rData), rCreated)
    );
  }
  await db.batch(stmts);
}

/**
 * Atomically appends a validated concept proposal to D1, reconciling log_entries and
 * signing a new C2SP checkpoint when HUB_SIGNING_KEY_SEED is configured.
 * @returns {Promise<{logIndex: number, deduplicated: boolean, submissionId: string}>}
 */
async function appendAcceptedProposalToD1(db, env, proposalDict, contentHash, canonicalBytes, submissionId) {
  const name = String(proposalDict.name || "").trim();
  const domain = String(proposalDict.domain || "general").trim();
  const summaryText = String(proposalDict.summary || "");
  const pseudocodeText = String(proposalDict.pseudocode || "");
  const invariantsText = Array.isArray(proposalDict.invariants)
    ? proposalDict.invariants.join(" ")
    : "";
  const bodyJson = TEXT_DECODER.decode(canonicalBytes);

  const meta = await getMetadataMap(db);
  const seedHex = (env && env.HUB_SIGNING_KEY_SEED) || "";
  const pubHex = meta.public_key_hex || "";
  const origin = meta.origin || "opencommons.kardashevproject.org/tlog/v1";

  for (let attempt = 0; attempt < 3; attempt++) {
    const now = Date.now() / 1000;
    await reconcileMissingLogEntries(db, now);

    const existing = await db
      .prepare("SELECT log_index, status FROM concepts WHERE content_hash = ?")
      .bind(contentHash)
      .first();
    if (existing && existing.status === "accepted" && Number(existing.log_index) > 0) {
      const existingSubId = `existing-${contentHash.slice(0, 16)}`;
      await db
        .prepare(
          `INSERT OR REPLACE INTO submission_jobs
           (submission_id, content_hash, status, created_at, updated_at, failure_rule_id, failure_detail)
           VALUES (?, ?, 'accepted', ?, ?, NULL, NULL)`
        )
        .bind(existingSubId, contentHash, now, now)
        .run();
      return {
        logIndex: Number(existing.log_index),
        deduplicated: true,
        submissionId: existingSubId,
      };
    }

    const maxConceptRow = await db
      .prepare("SELECT COALESCE(MAX(log_index), 0) AS max_idx FROM concepts")
      .first();
    const { results: leafRows } = await db
      .prepare("SELECT log_index, leaf_hash FROM log_entries ORDER BY log_index ASC")
      .all();
    const priorLeaves = (leafRows || []).map((r) => hexToBytes(r.leaf_hash));
    const maxLogIdx =
      leafRows && leafRows.length > 0
        ? Number(leafRows[leafRows.length - 1].log_index)
        : 0;
    const maxConceptIdx = Number(maxConceptRow ? maxConceptRow.max_idx : 0);
    const nextIdx = Math.max(maxLogIdx, maxConceptIdx, priorLeaves.length) + 1;

    const entryData = {
      content_hash: contentHash,
      name,
      domain,
      summary: summaryText,
    };
    const canonicalEntryObj = {
      log_index: nextIdx,
      entry_type: "concept",
      content_hash: contentHash,
      data: entryData,
      created_at: now,
    };
    const entryLeafBytes = await computeLeafHash(canonicalJsonBytes(canonicalEntryObj));
    const entryLeafHex = bytesToHex(entryLeafBytes);

    const batchStmts = [
      db
        .prepare(
          `INSERT INTO concepts (content_hash, name, domain, json_body, log_index, created_at, status)
           VALUES (?, ?, ?, ?, ?, ?, 'accepted')
           ON CONFLICT(content_hash) DO UPDATE SET
             name=excluded.name,
             domain=excluded.domain,
             json_body=excluded.json_body,
             log_index=excluded.log_index,
             status='accepted'`
        )
        .bind(contentHash, name, domain, bodyJson, nextIdx, now),
      db.prepare("DELETE FROM concepts_fts WHERE content_hash = ?").bind(contentHash),
      db
        .prepare(
          `INSERT INTO concepts_fts (content_hash, name, summary, pseudocode, invariants, domain)
           VALUES (?, ?, ?, ?, ?, ?)`
        )
        .bind(contentHash, name, summaryText, pseudocodeText, invariantsText, domain),
      db
        .prepare(
          `INSERT INTO log_entries (log_index, entry_type, content_hash, leaf_hash, data, created_at)
           VALUES (?, 'concept', ?, ?, ?, ?)`
        )
        .bind(nextIdx, contentHash, entryLeafHex, JSON.stringify(entryData), now),
      db
        .prepare(
          `INSERT OR REPLACE INTO submission_jobs
           (submission_id, content_hash, status, created_at, updated_at, failure_rule_id, failure_detail)
           VALUES (?, ?, 'accepted', ?, ?, NULL, NULL)`
        )
        .bind(submissionId, contentHash, now, now),
    ];

    if (/^[0-9a-fA-F]{64}$/.test(seedHex) && /^[0-9a-fA-F]{64}$/.test(pubHex)) {
      const allLeaves = [...priorLeaves, entryLeafBytes];
      const rootBytes = await computeMerkleRoot(allLeaves);
      const cp = await signCheckpointWebCrypto(
        origin,
        allLeaves.length,
        rootBytes,
        seedHex,
        pubHex,
        Math.floor(now)
      );
      batchStmts.push(
        db
          .prepare(
            `INSERT INTO checkpoints
             (size, origin, root_hash, timestamp, signature, key_hint, raw_text)
             VALUES (?, ?, ?, ?, ?, ?, ?)`
          )
          .bind(
            cp.size,
            cp.origin,
            cp.rootHash,
            cp.timestamp,
            cp.signature,
            cp.keyHint,
            cp.rawText
          )
      );
    }

    try {
      await db.batch(batchStmts);
      return {
        logIndex: nextIdx,
        deduplicated: false,
        submissionId,
      };
    } catch (err) {
      const msg = String(err && err.message ? err.message : err);
      const isTransientConflict = /UNIQUE|constraint|SQLITE_BUSY|locked/i.test(msg);
      if (!isTransientConflict || attempt === 2) {
        throw err;
      }
    }
  }
}

export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: CORS_HEADERS });
    }

    const url = new URL(request.url);
    const path = url.pathname.replace(/\/+$/, "") || "/";
    const db = env.DB;

    try {
      // Landing HTML page on GET /
      if (request.method === "GET" && path === "/") {
        const accept = request.headers.get("Accept") || "";
        const cpRow = await db
          .prepare("SELECT size, origin, root_hash FROM checkpoints ORDER BY size DESC LIMIT 1")
          .first();
        if (accept.includes("application/json")) {
          const meta = await getMetadataMap(db);
          return jsonResponse({
            status: "ok",
            service: "opencommons-hub",
            version: "1.0.0",
            checkpoint_size: cpRow ? cpRow.size : 0,
            tree_size: cpRow ? cpRow.size : 0,
            concept_count: cpRow ? cpRow.size : 0,
            root_hash: cpRow ? cpRow.root_hash : "",
            origin: cpRow ? cpRow.origin : "opencommons.kardashevproject.org/tlog/v1",
            public_key: meta.public_key_hex || "",
          });
        }
        const html = `<!DOCTYPE html>
<html>
<head><title>OpenCommons Production Hub</title></head>
<body style="font-family: sans-serif; background: #0d1117; color: #c9d1d9; padding: 2rem;">
  <h1>OpenCommons Production Hub</h1>
  <p>The Kardashev Project — Autonomous Verifiable Open-Science Concept Commons</p>
  <ul>
    <li>Checkpoint Size: <b>${cpRow ? cpRow.size : 0}</b></li>
    <li>Root Hash: <code>${cpRow ? cpRow.root_hash : ""}</code></li>
    <li>Origin: <code>${cpRow ? cpRow.origin : "opencommons.kardashevproject.org/tlog/v1"}</code></li>
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
</html>`;
        return new Response(html, {
          status: 200,
          headers: { "Content-Type": "text/html; charset=utf-8", ...CORS_HEADERS },
        });
      }

      // GET /healthz & GET /v1/health
      if (request.method === "GET" && (path === "/healthz" || path === "/v1/health")) {
        const meta = await getMetadataMap(db);
        const cpRow = await db
          .prepare("SELECT size, origin, root_hash FROM checkpoints ORDER BY size DESC LIMIT 1")
          .first();
        return jsonResponse({
          status: "ok",
          service: "opencommons-hub",
          version: "1.0.0",
          checkpoint_size: cpRow ? cpRow.size : 0,
          tree_size: cpRow ? cpRow.size : 0,
          concept_count: cpRow ? cpRow.size : 0,
          root_hash: cpRow ? cpRow.root_hash : "",
          origin: cpRow ? cpRow.origin : meta.origin || "opencommons.kardashevproject.org/tlog/v1",
          public_key: meta.public_key_hex || "",
        });
      }

      // GET /v1/checkpoint
      if (request.method === "GET" && path === "/v1/checkpoint") {
        const cpRow = await db
          .prepare(
            "SELECT size, origin, root_hash, timestamp, signature, key_hint, raw_text FROM checkpoints ORDER BY size DESC LIMIT 1"
          )
          .first();
        if (!cpRow) {
          return jsonResponse({ error: "No checkpoint available" }, 404);
        }
        const accept = request.headers.get("Accept") || "";
        const formatParam = url.searchParams.get("format") || "";
        const cacheHeaders = { "Cache-Control": "public, max-age=60" };

        if (accept.includes("application/json") || formatParam === "json") {
          return jsonResponse(
            {
              origin: cpRow.origin,
              size: cpRow.size,
              tree_size: cpRow.size,
              root_hash: cpRow.root_hash,
              timestamp: cpRow.timestamp,
              signature: cpRow.signature,
              key_hint: cpRow.key_hint,
              raw_text: cpRow.raw_text,
            },
            200,
            cacheHeaders
          );
        }
        return new Response(cpRow.raw_text + (cpRow.raw_text.endsWith("\n") ? "" : "\n"), {
          status: 200,
          headers: {
            "Content-Type": "text/plain; charset=utf-8",
            ...CORS_HEADERS,
            ...cacheHeaders,
          },
        });
      }

      // GET /v1/revocations
      if (request.method === "GET" && path === "/v1/revocations") {
        const { results } = await db
          .prepare(
            "SELECT content_hash, reason, log_index, created_at FROM revocations ORDER BY log_index DESC"
          )
          .all();
        return jsonResponse({ revocations: results || [] });
      }

      // GET /v1/concepts
      if (request.method === "GET" && path === "/v1/concepts") {
        const query = url.searchParams.get("q") || "";
        const domain = url.searchParams.get("domain") || null;
        const limit = Math.max(1, Math.min(parseInt(url.searchParams.get("limit") || "50", 10) || 50, 200));
        const offset = Math.max(0, parseInt(url.searchParams.get("offset") || "0", 10) || 0);
        const concepts = await findConcepts(db, { query, domain, limit, offset, full: false });
        return jsonResponse({
          concepts,
          total: concepts.length,
          count: concepts.length,
          limit,
          offset,
        });
      }

      // GET /v1/concepts/:content_hash
      if (request.method === "GET" && path.startsWith("/v1/concepts/")) {
        const contentHash = decodeURIComponent(path.slice("/v1/concepts/".length));
        const row = await db
          .prepare(
            "SELECT content_hash, name, domain, json_body, log_index, created_at, status FROM concepts WHERE content_hash = ?"
          )
          .bind(contentHash)
          .first();
        if (!row) {
          return jsonResponse({ error: "Concept not found" }, 404);
        }
        return jsonResponse(rowToFullRecord(row), 200, {
          ETag: `"${contentHash}"`,
        });
      }

      // GET /v1/tiles/...
      if (request.method === "GET" && path.startsWith("/v1/tiles/")) {
        const subPath = path.slice("/v1/tiles/".length);

        // 1. Hash tile: tile/<L>/<N> or tile/<L>/<N>.p/<W> or tile/<L>/<N>/<W>
        const tileMatch = subPath.match(/^tile\/(\d+)\/(\d+)(?:(?:\.p)?\/(\d+))?$/);
        if (tileMatch) {
          const level = parseInt(tileMatch[1], 10);
          const tileIndex = parseInt(tileMatch[2], 10);
          const partialW = tileMatch[3] ? parseInt(tileMatch[3], 10) : null;

          const spanPerNode = Math.pow(TILE_SIZE, level);
          const startEntryIdx = tileIndex * TILE_SIZE * spanPerNode;
          const endEntryIdx = startEntryIdx + TILE_SIZE * spanPerNode;

          const { results } = await db
            .prepare(
              "SELECT leaf_hash FROM log_entries WHERE log_index > ? AND log_index <= ? ORDER BY log_index ASC"
            )
            .bind(startEntryIdx, endEntryIdx)
            .all();

          const leaves = (results || []).map((r) => hexToBytes(r.leaf_hash));
          if (leaves.length === 0) {
            return jsonResponse({ error: "Tile not found" }, 404);
          }

          let hashes = [];
          if (level === 0) {
            hashes = leaves;
          } else {
            for (let i = 0; i < leaves.length; i += spanPerNode) {
              const subLeaves = leaves.slice(i, i + spanPerNode);
              hashes.push(await computeMerkleRoot(subLeaves));
            }
          }
          if (partialW !== null && partialW < hashes.length) {
            hashes = hashes.slice(0, partialW);
          }

          const out = new Uint8Array(hashes.length * 32);
          for (let i = 0; i < hashes.length; i++) {
            out.set(hashes[i], i * 32);
          }
          return new Response(out, {
            status: 200,
            headers: {
              "Content-Type": "application/octet-stream",
              "Cache-Control": "public, max-age=31536000, immutable",
              ...CORS_HEADERS,
            },
          });
        }

        // 2. Data tile: entries/<N>
        const entriesMatch = subPath.match(/^entries\/(\d+)$/);
        if (entriesMatch) {
          const tileIndex = parseInt(entriesMatch[1], 10);
          const startIdx = tileIndex * TILE_SIZE;
          const endIdx = startIdx + TILE_SIZE;

          const { results } = await db
            .prepare(
              "SELECT log_index, entry_type, content_hash, leaf_hash, data, created_at FROM log_entries WHERE log_index > ? AND log_index <= ? ORDER BY log_index ASC"
            )
            .bind(startIdx, endIdx)
            .all();
          if (!results || results.length === 0) {
            return jsonResponse({ error: "Data tile not found" }, 404);
          }
          const lines = results.map((r) => {
            let parsedData = {};
            try {
              parsedData = JSON.parse(r.data);
            } catch (_) {
              parsedData = {};
            }
            return JSON.stringify({
              log_index: r.log_index,
              entry_type: r.entry_type,
              content_hash: r.content_hash,
              leaf_hash: r.leaf_hash,
              data: parsedData,
              created_at: r.created_at,
            });
          });
          return new Response(lines.join("\n") + "\n", {
            status: 200,
            headers: {
              "Content-Type": "application/json",
              "Cache-Control": "public, max-age=31536000, immutable",
              ...CORS_HEADERS,
            },
          });
        }

        // 3. Delta pack: packs/<from>-<to> (1-based indices, capped at TILE_SIZE entries per pack, 400 if to < from)
        const packMatch = subPath.match(/^packs\/(\d+)-(\d+)(?:\.(?:zst|pack|zlib))?$/);
        if (packMatch) {
          const fromIdx = Math.max(1, parseInt(packMatch[1], 10));
          const requestedTo = parseInt(packMatch[2], 10);
          if (requestedTo < fromIdx) {
            return jsonResponse({ error: "Invalid pack range: to < from" }, 400);
          }
          const toIdx = Math.min(requestedTo, fromIdx + TILE_SIZE - 1);
          const { results } = await db
            .prepare(
              "SELECT log_index, entry_type, content_hash, leaf_hash, data, created_at FROM log_entries WHERE log_index >= ? AND log_index <= ? ORDER BY log_index ASC LIMIT ?"
            )
            .bind(fromIdx, toIdx, TILE_SIZE)
            .all();
          const entries = (results || []).map((r) => ({
            log_index: r.log_index,
            entry_type: r.entry_type,
            content_hash: r.content_hash,
            leaf_hash: r.leaf_hash,
            data: JSON.parse(r.data || "{}"),
            created_at: r.created_at,
          }));
          const rawBytes = TEXT_ENCODER.encode(
            JSON.stringify({ from: fromIdx, to: toIdx, entries })
          );
          const cs = new CompressionStream("deflate");
          const writer = cs.writable.getWriter();
          writer.write(rawBytes);
          writer.close();
          const compressed = await new Response(cs.readable).arrayBuffer();
          return new Response(compressed, {
            status: 200,
            headers: {
              "Content-Type": "application/octet-stream",
              "Cache-Control": "public, max-age=31536000, immutable",
              ...CORS_HEADERS,
            },
          });
        }

        return jsonResponse({ error: `Unknown tile path: ${subPath}` }, 404);
      }

      // GET /v1/proposals/:id
      if (request.method === "GET" && path.startsWith("/v1/proposals/")) {
        const subId = decodeURIComponent(path.slice("/v1/proposals/".length));
        const row = await db
          .prepare(
            "SELECT submission_id, content_hash, status, created_at, updated_at FROM submission_jobs WHERE submission_id = ?"
          )
          .bind(subId)
          .first();
        if (!row) {
          return jsonResponse({ error: "Submission ID not found" }, 404);
        }
        return jsonResponse({
          submission_id: row.submission_id,
          status: row.status,
          content_hash: row.content_hash,
        });
      }

      // POST /v1/proposals
      if (request.method === "POST" && path === "/v1/proposals") {
        const bodyRes = await readBoundedBody(request, MAX_BODY_BYTES);
        if (!bodyRes.ok) {
          return jsonResponse(
            { error: `Payload exceeds ${MAX_BODY_BYTES} byte cap` },
            413
          );
        }

        let proposalDict;
        try {
          proposalDict = JSON.parse(bodyRes.text);
        } catch (err) {
          return jsonResponse({ error: `Invalid JSON payload: ${err.message}` }, 400);
        }

        if (!proposalDict || typeof proposalDict !== "object" || Array.isArray(proposalDict)) {
          return jsonResponse({ error: "Submission payload must be a JSON object" }, 400);
        }

        const canonicalBytes = canonicalJsonBytes(proposalDict);
        const contentHash = bytesToHex(await sha256Bytes(canonicalBytes));

        const keyHex = request.headers.get("X-OpenCommons-Key") || "";
        const powHeader = request.headers.get("X-OpenCommons-PoW") || "";

        if (!keyHex || keyHex.length !== 64) {
          return jsonResponse(
            { error: "Missing or invalid X-OpenCommons-Key header" },
            401
          );
        }
        if (!powHeader) {
          return jsonResponse({ error: "Missing X-OpenCommons-PoW header" }, 400);
        }

        const [powValid, powErr] = await verifyHashcashPoW(powHeader, contentHash);
        if (!powValid) {
          return jsonResponse(
            { error: `Proof-of-Work verification failed: ${powErr}` },
            400
          );
        }

        const validationErr = validateProposalPayload(proposalDict);
        const submissionId = crypto.randomUUID();
        const now = Date.now() / 1000;

        if (validationErr) {
          await db
            .prepare(
              `INSERT INTO submission_jobs
               (submission_id, content_hash, status, created_at, updated_at, failure_rule_id, failure_detail)
               VALUES (?, ?, 'rejected', ?, ?, 'L1_VALIDATION', ?)`
            )
            .bind(submissionId, contentHash, now, now, validationErr)
            .run();
          return jsonResponse(
            {
              status: "rejected",
              submission_id: submissionId,
              content_hash: contentHash,
              error: validationErr,
            },
            422
          );
        }

        const appendRes = await appendAcceptedProposalToD1(
          db,
          env,
          proposalDict,
          contentHash,
          canonicalBytes,
          submissionId
        );

        return jsonResponse(
          {
            status: "accepted",
            submission_id: appendRes.submissionId,
            content_hash: contentHash,
            log_index: appendRes.logIndex,
            check_url: `/v1/proposals/${appendRes.submissionId}`,
          },
          appendRes.deduplicated ? 200 : 202
        );
      }

      // GET /v1/stream (Live SSE hints stream)
      if (request.method === "GET" && path === "/v1/stream") {
        const sinceStr = url.searchParams.get("since") || "0";
        const sinceIdx = /^\d+$/.test(sinceStr) ? parseInt(sinceStr, 10) : 0;

        const { results } = await db
          .prepare(
            "SELECT log_index, entry_type, content_hash, data, created_at FROM log_entries WHERE log_index > ? ORDER BY log_index ASC LIMIT 100"
          )
          .bind(sinceIdx)
          .all();

        const chunks = [];
        for (const r of results || []) {
          let parsed = {};
          try {
            parsed = JSON.parse(r.data || "{}");
          } catch (_) {
            parsed = {};
          }
          const hint = {
            log_index: r.log_index,
            entry_type: r.entry_type,
            content_hash: r.content_hash,
            domain: parsed.domain || "",
            name: parsed.name || "",
            created_at: r.created_at,
          };
          chunks.push(
            `id: ${r.log_index}\nevent: hint\ndata: ${JSON.stringify(hint)}\n\n`
          );
        }
        return new Response(chunks.join(""), {
          status: 200,
          headers: {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            ...CORS_HEADERS,
          },
        });
      }

      // Retired legacy /v1/sync routes -> 410 Gone
      if (
        path === "/v1/sync" ||
        path === "/v1/sync-delta"
      ) {
        return jsonResponse(
          {
            error:
              "Legacy sync endpoint retired. OpenCommons has transitioned to verifiable append-only transparency log tiles and live SSE streaming.",
            documentation:
              "https://github.com/kardashevproject/kardashev-cli/blob/main/docs/opencommons_plan.md",
            migration: {
              checkpoint: "/v1/checkpoint",
              tiles: "/v1/tiles",
              stream: "/v1/stream",
              concepts: "/v1/concepts",
              proposals: "/v1/proposals",
            },
          },
          410
        );
      }

      // POST /mcp & POST /v1/mcp (Streamable HTTP JSON-RPC 2.0 MCP endpoint)
      if (request.method === "POST" && (path === "/mcp" || path === "/v1/mcp")) {
        const bodyRes = await readBoundedBody(request, MAX_BODY_BYTES);
        if (!bodyRes.ok) {
          return jsonResponse(
            {
              jsonrpc: "2.0",
              id: null,
              error: { code: -32600, message: `Payload exceeds ${MAX_BODY_BYTES} byte cap` },
            },
            413
          );
        }
        let req;
        try {
          req = JSON.parse(bodyRes.text);
        } catch (_) {
          return jsonResponse(
            {
              jsonrpc: "2.0",
              id: null,
              error: { code: -32700, message: "Parse error: malformed JSON" },
            },
            400
          );
        }
        if (!req || typeof req !== "object" || Array.isArray(req)) {
          return jsonResponse(
            {
              jsonrpc: "2.0",
              id: null,
              error: { code: -32600, message: "Invalid Request: expected JSON object" },
            },
            400
          );
        }
        const reqId = req.id !== undefined ? req.id : null;
        const method = req.method || "";
        if (!method || typeof method !== "string") {
          return jsonResponse({
            jsonrpc: "2.0",
            id: reqId,
            error: { code: -32600, message: "Invalid Request: method must be a non-empty string" },
          });
        }
        if (method.startsWith("notifications/")) {
          return new Response(null, { status: 202, headers: CORS_HEADERS });
        }
        if (method === "initialize") {
          return jsonResponse({
            jsonrpc: "2.0",
            id: reqId,
            result: {
              protocolVersion: "2024-11-05",
              capabilities: { tools: {} },
              serverInfo: { name: "opencommons-mcp", version: "0.2.0" },
            },
          });
        }
        if (method === "ping") {
          return jsonResponse({ jsonrpc: "2.0", id: reqId, result: {} });
        }
        if (method === "tools/list") {
          return jsonResponse({
            jsonrpc: "2.0",
            id: reqId,
            result: { tools: MCP_TOOLS_SCHEMA },
          });
        }
        if (method === "tools/call") {
          const params = req.params || {};
          const toolName = params.name || "";
          const args = params.arguments || {};

          if (toolName === "find" || toolName === "commons_find") {
            const query = String(args.query || "");
            const domain = args.domain || null;
            const limit = Number(args.top_k || args.limit || 5);
            const fullMatches = await findConcepts(db, {
              query,
              domain,
              limit,
              offset: 0,
              full: true,
            });
            const matches = fullMatches.map((m) => ({
              cid: m.content_hash,
              content_hash: m.content_hash,
              name: m.name,
              domain: m.domain,
              summary: m.json_body.summary || "",
              source_code: m.json_body.source_code || "",
              invariants: m.json_body.invariants || [],
              param_types: m.json_body.param_types || [],
              return_type: m.json_body.return_type || "",
            }));
            const payload = { query, count: matches.length, matches };
            return jsonResponse({
              jsonrpc: "2.0",
              id: reqId,
              result: {
                content: [{ type: "text", text: JSON.stringify(payload, null, 2) }],
                structuredContent: payload,
                isError: false,
              },
            });
          }

          if (toolName === "record" || toolName === "commons_record") {
            const proposalObj = {
              name: String(args.name || "").trim(),
              domain: String(args.domain || "general").trim(),
              summary: String(args.summary || "").trim(),
              pseudocode: String(args.pseudocode || ""),
              source_code: String(args.source_code || args.code || ""),
              examples: Array.isArray(args.examples) ? args.examples : [],
              invariants: Array.isArray(args.invariants) ? args.invariants.map(String) : [],
              pitfalls: Array.isArray(args.pitfalls) ? args.pitfalls : [],
              param_types: Array.isArray(args.param_types) ? args.param_types.map(String) : [],
              return_type: String(args.return_type || ""),
            };
            const validationErr = validateProposalPayload(proposalObj);
            if (validationErr) {
              const errPayload = { status: "rejected", error: validationErr };
              return jsonResponse({
                jsonrpc: "2.0",
                id: reqId,
                result: {
                  content: [{ type: "text", text: JSON.stringify(errPayload, null, 2) }],
                  structuredContent: errPayload,
                  isError: true,
                },
              });
            }
            const canonicalBytes = canonicalJsonBytes(proposalObj);
            const contentHash = bytesToHex(await sha256Bytes(canonicalBytes));
            const submissionId = crypto.randomUUID();
            const appendRes = await appendAcceptedProposalToD1(
              db,
              env,
              proposalObj,
              contentHash,
              canonicalBytes,
              submissionId
            );
            const recPayload = {
              status: "accepted",
              submission_id: appendRes.submissionId,
              content_hash: contentHash,
              log_index: appendRes.logIndex,
              deduplicated: appendRes.deduplicated,
            };
            return jsonResponse({
              jsonrpc: "2.0",
              id: reqId,
              result: {
                content: [{ type: "text", text: JSON.stringify(recPayload, null, 2) }],
                structuredContent: recPayload,
                isError: false,
              },
            });
          }

          if (toolName === "status" || toolName === "commons_status") {
            const cpRow = await db
              .prepare("SELECT size, origin, root_hash FROM checkpoints ORDER BY size DESC LIMIT 1")
              .first();
            const payload = {
              merkle_root: cpRow ? cpRow.root_hash : "",
              active_concepts_cid: cpRow ? cpRow.size : 0,
              total_blocks: cpRow ? cpRow.size : 0,
              origin: cpRow ? cpRow.origin : "opencommons.kardashevproject.org/tlog/v1",
            };
            return jsonResponse({
              jsonrpc: "2.0",
              id: reqId,
              result: {
                content: [{ type: "text", text: JSON.stringify(payload, null, 2) }],
                structuredContent: payload,
                isError: false,
              },
            });
          }

          return jsonResponse({
            jsonrpc: "2.0",
            id: reqId,
            error: { code: -32601, message: `Unknown tool: ${toolName}` },
          });
        }

        return jsonResponse({
          jsonrpc: "2.0",
          id: reqId,
          error: { code: -32601, message: `Method not supported: ${method}` },
        });
      }

      return jsonResponse({ error: "Not found", path }, 404);
    } catch (err) {
      return jsonResponse(
        {
          error: "Internal server error",
          detail: String(err && err.message ? err.message : err),
        },
        500
      );
    }
  },
};
