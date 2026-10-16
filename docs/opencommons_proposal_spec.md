# OpenCommons Concept Proposal Specification (v1)

> Normative specification for OpenCommons concept proposals across HTTP API, MCP tools, and Kardashev harness hooks.

## 1. Overview & Scope

OpenCommons is a decentralized, privacy-preserving concept repository for scientific, mathematical, and systems computing algorithms.
This specification defines the canonical proposal payload structure (`v1`), deterministic canonicalization rules, anonymous contributor authentication, and proof-of-work validation.

Every proposal submitted via:
- HTTP submission API (`POST /v1/proposals`),
- Model Context Protocol (MCP) tool interfaces (`commons_record`, `record`), or
- Automated agent harness hooks (`kardashev` background submission)

MUST strictly conform to this normative specification and the companion JSON Schema at `src/kardashev_cli/opencommons_schema.json`.

---

## 2. Proposal Data Model

A proposal is represented as a JSON object with **`additionalProperties: false`**. Exactly nine top-level fields are permitted. Any additional keys will cause immediate rejection at Layer 1 (L1).

| Field | Type | Bounds | Description |
|---|---|---|---|
| `name` | `string` | 3–64 chars | Identifier matching regex `^[a-z][a-z0-9_]{2,63}$`. Must match the top-level Python function name. |
| `param_types` | `array[string]` | 0–32 items | Declared type annotations for each parameter. Each string is 1–128 chars. |
| `return_type` | `string` | 1–128 chars | Declared return type annotation. |
| `summary` | `string` | 10–500 chars | High-level mathematical or algorithmic summary of the function. |
| `pseudocode` | `string` | 10–2000 chars | Step-by-step algorithmic description or pseudocode. |
| `invariants` | `array[string]` | 0–8 items | Mathematical, numerical, or systems invariants that hold true. Each string is 1–200 chars. |
| `pitfalls` | `array[object]` | 0–8 items | Known numerical failure modes, edge cases, and refuted naive implementations. |
| `source_code` | `string` | 1–8192 chars | Single pure Python function definition in 100% printable ASCII (0x20–0x7E plus newline `\n`). |
| `domain` | `string` | enum (11) | Classification domain string (see Section 2.2). |

### 2.1 Pitfall Object Schema
Each entry in the `pitfalls` array is an object with **`additionalProperties: false`** containing exactly four mandatory fields:

1. `naive_pattern` (`string`, 1–500 chars): Description of the flawed naive implementation or common bug.
2. `failure_mechanism` (`string`, 1–500 chars): Underlying cause of the failure (e.g. numerical underflow, float roundoff, deadlocks).
3. `counterexample_input` (`string`, 1–500 chars): Concrete input vector that demonstrates the failure.
4. `verified_fix` (`string`, 1–500 chars): Explanation of the mathematically verified solution.

### 2.2 Domain Taxonomy
The `domain` field must be one of the following 11 canonical domains:
- `cryptography_and_framing`
- `distributed_systems`
- `geodesy_and_spatial`
- `hamiltonian_dynamics`
- `machine_learning`
- `number_theory_and_algebra`
- `numerical_analysis`
- `quantitative_finance`
- `signal_processing`
- `statistics_and_information`
- `general`

---

## 3. Source Code Policy & Canonicalization

Source code submissions must adhere to strict structural constraints:

1. **Printable ASCII Only**: The `source_code` string must consist exclusively of printable ASCII characters (`0x20` through `0x7E`) and newline (`\n`, `0x0A`). All Unicode homoglyphs, invisible characters, zero-width spaces, and bidi override tags are strictly forbidden.
2. **Single Pure Function**: The AST must contain exactly one top-level statement, which must be an `ast.FunctionDef`. The function name must match `name`.
3. **No Classes, Async, or State Mutation**: Classes (`ClassDef`), asynchronous functions (`AsyncFunctionDef`), generators (`Yield`, `YieldFrom`), and scopes modifying globals/nonlocals (`Global`, `Nonlocal`) are prohibited.
4. **Import Allowlist**: No top-level imports are allowed in stored source code. Standard mathematical libraries (`math`, `itertools`, `functools`, `collections`, `heapq`, `bisect`, `operator`, `dataclasses`, `typing`) are injected into the sandboxed execution environment.
5. **Forbidden Calls and Identifiers**: Calls to `eval`, `exec`, `compile`, `open`, `__import__`, `getattr`, `setattr`, `globals`, `locals`, `vars`, `breakpoint`, `exit`, etc. are forbidden. Identifiers or attribute accesses starting with `_` (except single throwaway loop targets `_`) are rejected.
6. **AST Canonicalization**: Before storage or transmission, code must be canonicalized via `ast.unparse(clean_ast)` with docstrings and comments completely stripped.

---

## 4. Deterministic Canonicalization & Content Addressing

Proposals are content-addressed using SHA-256 over canonical JSON according to **RFC 8785 (JSON Canonicalization Scheme - JCS)**:

```python
canonical_bytes = json.dumps(
    proposal,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False
).encode("utf-8")

content_hash = hashlib.sha256(canonical_bytes).hexdigest()
```

The resulting 64-character lowercase hexadecimal string serves as the unique, immutable content address for the proposal across all nodes, transparency logs, and client caches.

---

## 5. Contributor Identity & Privacy Protocol

OpenCommons enforces a strict zero-identifier privacy guarantee:

1. **Anonymous Ed25519 Keypair**: Contributors are identified solely by an RFC 8032 Ed25519 public key generated locally.
2. **Zero IP Tracking**: Hubs and nodes NEVER record, store, log, or track client IP addresses. Rate limits and Sybil defenses are enforced exclusively via cryptographic keys and Proof-of-Work.
3. **No Free-Text Attribution**: Free-text contributor names (e.g. `contributor_agent`) are removed from all proposals.
4. **Key Storage**: Local client keys are stored in `~/.kardashev/opencommons_key.json` with POSIX `0o600` permissions (`-rw-------`).

---

## 6. Dynamic Hashcash Proof-of-Work

To protect the ingestion pipeline against DoS and Sybil spam without tracking IP addresses, submissions require a dynamic Hashcash Proof-of-Work stamp:

### 6.1 Stamp Format
```
<nonce>:<timestamp>:<difficulty>:<content_hash>
```
- `nonce`: Numeric string counter (e.g. `"182405"`).
- `timestamp`: Unix epoch seconds (must be within 300 seconds of server time).
- `difficulty`: Number of leading zero bits required in `sha256(stamp)`.
- `content_hash`: The SHA-256 content address of the proposal.

### 6.2 Dynamic Scaling Ratchet
- **Base Difficulty**: 16 leading zero bits (~65,536 iterations, <100ms CPU).
- **Dynamic Ratchet**: Each consecutive proposal rejection increments the required difficulty by 1 bit:
  $$\text{difficulty} = \text{base} + \min(\text{rejection\_count}, 8)$$
  (Capped at 24 bits, ~16.7M iterations, ~5s CPU).
- **Decay**: Rejection penalties decay after 24 hours of inactivity or on successful proposal acceptance.

---

## 7. Submission Protocol & Wire Format

Submissions are sent via HTTP `POST /v1/proposals`:

### Request Headers
```http
POST /v1/proposals HTTP/1.1
Host: commons.kardashevproject.org
Content-Type: application/json
Content-Length: <length>
X-OpenCommons-Key: <ed25519_public_key_hex>
X-OpenCommons-PoW: <nonce>:<timestamp>:<difficulty>:<content_hash>
X-OpenCommons-Signature: <ed25519_signature_hex>
```

### Response Statuses
- `202 Accepted`: Submission passed L0 edge checks and is queued for verification.
  ```json
  {"status": "accepted", "submission_id": "c1f7a08b-...", "content_hash": "..."}
  ```
- `413 Payload Too Large`: Payload exceeds the 32 KB edge cap.
- `429 Too Many Requests`: Per-key rate limit exceeded. Includes `Retry-After: <seconds>`.
- `400 Bad Request`: Schema validation failure (L1).
- `403 Forbidden`: Invalid signature or blacklisted key.
