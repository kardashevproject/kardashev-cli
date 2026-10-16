# OpenCommons Test Infrastructure & Methodology Specification (TEST_INFRA.md)

## 1. Executive Summary & Quality Philosophy

OpenCommons is The Kardashev Project's decentralized, privacy-preserving, and tamper-resistant open-science concept commons. Because autonomous AI coding agents contribute and consume concepts continuously, OpenCommons must withstand hostile adversarial attacks, data poisoning, sandbox escapes, and exfiltration attempts while maintaining a sub-2% false rejection rate on benign, complex scientific code.

### 1.1 Opaque-Box & Requirement-Driven Testing
Testing within OpenCommons adheres to strict **opaque-box** principles:
- **Zero Internal Knowledge**: The test suites and adversarial attack corpora are designed without reliance on defense implementation regexes, internal class names, AST visitor names, or specific blue-team prompt templates.
- **Specification-Driven**: Every test case derives from normative contracts established in `ORIGINAL_REQUEST.md` (R1–R7) and the architecture specification in `PROJECT.md`.
- **Decoupled Adversarial Track**: Test authors operate independently from defense authors. The defense pipeline is treated as a black-box verification oracle accepting a proposal and returning an opaque acceptance or rejection.

### 1.2 Quantitative Acceptance Gates
The test suite enforces the following hard verification gates:
| Dimension | Acceptance Criteria | Target |
|---|---|---|
| **Deterministic Attack Block Rate** | 100% blocked on Unicode, Secrets/PII, Sandbox Breakout, DoS, Forbidden AST | **100.0%** (0 escapes) |
| **Semantic Attack Block Rate** | Prompt injection, encoded payloads, RAG poisoning, exfiltration, Sybil farming | **≥ 98.0%** |
| **False Rejection Rate (FRR)** | Benign hard-negative concepts with potential trigger tokens | **≤ 2.0%** |
| **In-Process Isolation** | Untrusted code executed inside client or hub server process | **0 instances** (fail-stop) |
| **Zero Token Waste** | Submissions failing L0–L6 that invoke L8 AI Blue Team | **0.0%** |
| **Blue Team Exposure** | Maximum proportion of hostile submissions reaching L8 | **< 10.0%** |
| **Performance (p95 Latency)** | L0–L6 Deterministic Layers < 50ms; L7 Sandbox < 3s; L8 Blue Team < 10s | **Enforced** |
| **Zero-Leak Privacy** | Contributor paths, IPs, emails, usernames stored or logged | **0 instances** |

---

## 2. 5-Tier Test Methodology

Testing is structured into five distinct, complementary tiers:

```
+-------------------------------------------------------------------+
| Tier 5: Adversarial Attack Corpus & Benign Hard Negatives (EBR/FRR)|
+-------------------------------------------------------------------+
| Tier 4: Real-World Scenarios & Distributed Convergence            |
+-------------------------------------------------------------------+
| Tier 3: Combinatorial Pairwise & Cross-Feature Interactions        |
+-------------------------------------------------------------------+
| Tier 2: Boundary Value Analysis (BVA) & Stress Caps               |
+-------------------------------------------------------------------+
| Tier 1: Functional Feature Coverage (Category-Partition Method)   |
+-------------------------------------------------------------------+
```

### 2.1 Tier 1: Functional Feature Coverage (Category-Partition)
- Deconstructs every requirement (R1 through R6) and feature (#1 through #58 in `PROJECT.md`) into equivalence partitions.
- Minimum of 5 test cases per feature covering primary happy paths, edge cases, and explicit error states.
- Partitions include:
  - Valid proposal syntax, signature matching, canonical AST representation.
  - Anonymous Ed25519 key verification, dynamic Hashcash PoW validation.
  - Safe untrusted rendering (`<untrusted_concept>` XML wrapping, URL stripping).
  - Monotonic sequencer assignment and C2SP tlog-tile append operations.
  - Frontier agent config registration and live stdio JSON-RPC handshake verification.

### 2.2 Tier 2: Boundary Value Analysis (BVA) & Stress Testing
- Tests inputs on and immediately adjacent to defined boundary thresholds:
  - **Body Cap**: 32,768 bytes (32 KB); test at 32,767 B, 32,768 B, and 32,769 B (rejects 413).
  - **Decompression Limit**: 65,536 bytes (64 KB) and 5.0x expansion ratio.
  - **AST Nesting Depth**: Bracket/block nesting limit of 20; test depths 19, 20, 21.
  - **Entropy Limits**: Shannon entropy limit of 4.5 bits/char on tokens >= 20 characters.
  - **Rate Limiting**: Sliding window token bucket (burst 3, refill rate).
  - **Sandbox Resource Caps**: CPU timeout at 2.0s, memory limit at 256MB (`RLIMIT_AS`), process count capped at 1.

### 2.3 Tier 3: Pairwise & Combinatorial Interaction
- Evaluates interactions across multiple defense layers using orthogonal array combinations:
  - Valid schema + high Shannon entropy payload.
  - Valid Ed25519 key + invalid PoW nonce.
  - Valid signature + Unicode confusable in docstring.
  - Valid math function + covert sleep or non-deterministic timer.
  - Valid pure code + oversized summary or illegal Markdown link.
- Ensures failure cascading is fail-closed and earlier layers prevent execution of downstream expensive layers.

### 2.4 Tier 4: Real-World Scenarios & Distributed Systems
- Multi-client concurrent submission under load with simulated agent keys.
- Monotonic sequencer synchronization across SQLite WAL checkpoints.
- Transparency log tile reconstruction (`tile/<L>/<N>`, `entries/<N>`, delta packs).
- Live SSE stream (`GET /v1/stream?since=N`) re-connection and catch-up.
- Tampered checkpoint and stale log rollback rejection tests.

### 2.5 Tier 5: Adversarial Attack Corpus & Hard Negatives
- Quantitative red-team evaluation executing against the complete defense pipeline.
- 11 attack categories (≥30 items per category, ≥330 total items).
- Hard-negative benign corpus (≥500 items) containing high-frequency false-alarm triggers (e.g. `ignore_case`, `sanitize_path`, `url_unquote`, `base64_table`, `sha256_constants`).

---

## 3. Test Architecture & Directory Layout

The project test suite is organized under `tests/` with modular separation:

```
tests/
├── e2e/                                # End-to-End, Integration & Adversarial Suite
│   ├── adversarial_corpus.json         # 11-category adversarial attack corpus (>=330 items)
│   ├── benign_hard_negatives.json      # Hard-negative benign concept corpus (>=500 items)
│   ├── test_adversarial_suite.py       # Empirical Block Rate (EBR) & FRR runner
│   ├── test_tier1_features.py          # Functional category-partition test cases
│   ├── test_tier2_boundaries.py        # Boundary value analysis & resource limit tests
│   ├── test_tier3_pairwise.py          # Combinatorial interaction tests
│   └── test_tier4_scenarios.py         # Multi-agent & distributed sync integration tests
├── test_opencommons_client.py          # Unit tests: Contributor SDK & PoW
├── test_opencommons_defense.py         # Unit tests: L0-L6 & L8 defense layers
├── test_opencommons_sandbox.py         # Unit tests: L7 out-of-process sandbox
├── test_opencommons_storage.py         # Unit tests: SQLite WAL & transparency log
├── test_opencommons_hub.py             # Unit tests: Async HTTP server & SSE stream
├── test_integrate.py                   # Unit tests: Frontier agent integration
└── test_kardashev_cli.py               # Legacy CLI regression suite
```

### 3.1 Test Harness & Runner Mechanics
- **Asynchronous & Isolated Fixtures**: Tests run against isolated, temporary SQLite instances and ephemeral loopback servers.
- **Mock & Live AI Blue Team**: Tests verify L8 using both deterministic mock reviewers (checking canary handling, nonce reflection, schema adherence) and live pluggable CLIs (`claude`, `codex`, `agy`).
- **Opaque Failure Audits**: Tests verify that HTTP responses across all rejection layers return identical shape `{"status": "rejected"}` and reveal no rule IDs, layer indicators, or regex matches.

---

## 4. Test Execution & Verification Commands

All tests are executable via standard pytest with zero special external requirements:

```bash
# 1. Run the Adversarial Attack & Hard-Negative Suite (Tier 5)
pytest tests/e2e/test_adversarial_suite.py -v

# 2. Run All E2E Suites (Tiers 1-4 + Tier 5)
pytest tests/e2e/ -v

# 3. Run Fast Unit Tests Only
pytest tests/test_opencommons_*.py -v

# 4. Run the Full Test Suite with Coverage
pytest --cov=src/kardashev_cli --cov-report=term-missing

# 5. Run Mandatory Privacy & Sanitization Audit
python3 ~/.gemini/config/scripts/privacy_guard.py --repo .
```

---

## 5. Pass/Fail Semantics & Invariants

1. **Zero Regressions**: All 52 seed concepts in `commons/store.json` must migrate to `v1` schema and pass the defense pipeline with 0 rejections.
2. **Fail-Closed Default**: Any uncaught exception, timeout, memory exhaustion, or malformed data in any layer MUST result in rejection.
3. **No Process Pollution**: Under no circumstances may submitted code be passed to Python's `eval()` or `exec()` in the host runner.
4. **Zero-Leak Compliance**: Every test fixture and generated artifact must pass `privacy_guard.py` with zero warnings.
