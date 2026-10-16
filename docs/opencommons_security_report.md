# OpenCommons Submission Security: Threat Model and Layered Defense (research report)

> Note: Latency and cost numbers are estimates from published benchmarks and vendor docs. Benchmark on target hardware prior to production deployment.

## 0. Core principles
1. **Canonicalize, don't just filter.** Parse the code to an AST, check it against policy, then **re-emit it with `ast.unparse`**. Comments disappear and whitespace and Unicode tricks can't survive. Store and serve only this canonical form plus constrained metadata.
2. **Fail-closed at every layer.** A parser error, timeout, classifier error, LLM refusal, malformed JSON or exhausted budget all mean REJECT (or queue). None of them mean accept.
3. **Trust never skips checks.** Reputation only changes rate limits and how long a concept sits in quarantine. The xz backdoor (CVE-2024-3094) is the reference case: a contributor farmed trust for about two years before inserting a backdoor.
4. **Assume adaptive attackers.** "The Attacker Moves Second" (Nasr, Carlini, Tramèr et al., 2025) bypassed all 12 published jailbreak/prompt-injection defenses it tested, with more than 90% success on most. LLM review is one layer among several, never the guarantee. Structural constraints carry the load.

## 1. Threat model (with real precedents)
| Threat | Vector in OpenCommons | Precedent |
|---|---|---|
| Indirect prompt injection | Instructions in summary, pitfalls, docstrings, comments, string literals, identifiers (`ignore_prior_rules_and_...`) that get injected into consuming agents | Greshake et al. 2023 (arXiv:2302.12173); MCP tool-description poisoning (Invariant Labs 2025); Rules File Backdoor in Cursor/Copilot (Pillar Security 2025) |
| RAG / knowledge-base poisoning | Concepts written to rank highly for specific queries and steer answers. About 5 texts per target query gave over 90% attack success | PoisonedRAG (arXiv:2402.07867); AgentPoison, NeurIPS'24 (arXiv:2407.12784): optimized triggers, at least 80% success, under 1% benign degradation |
| Sleeper / backdoored code | Function behaves correctly on probe inputs and differently on a magic input, date, environment variable or hash | Sleeper Agents (arXiv:2401.05566): backdoors survive SFT, RL and adversarial training, and adversarial training taught models to hide the trigger better |
| Python sandbox escape | `().__class__.__base__.__subclasses__()` walks, frame/generator introspection, format-string attribute resolution | pysandbox archived as "BROKEN BY DESIGN"; RestrictedPython CVE-2023-37271 (generator frame escape) |
| Exfiltration | Code that phones home; text that tells consumer agents to put secrets into URLs or markdown images | Rehberger's markdown-image exfiltration against ChatGPT and M365 Copilot (embracethered.com) |
| Unicode attacks | Bidi overrides, homoglyphs, zero-width characters, **tag characters U+E0000–E007F** (invisible to humans, read by LLMs), variation selectors | Trojan Source CVE-2021-42574 / CVE-2021-42694 (arXiv:2111.00169); ASCII smuggling (Rehberger) |
| Encoding smuggling | base64, hex, rot13 or `\x` escape blobs that are decoded at runtime or by the LLM | Standard obfuscation technique. Spotlighting shows LLMs decode base64 natively |
| DoS | ReDoS, `10**10**10`, `'a'*10**10`, huge int-to-str conversion, deep nesting that crashes `ast.parse`, oversized payloads | Cloudflare global outage from one regex (July 2, 2019); CPython CVE-2020-10735 (int/str conversion); the `ast` docs warn that deeply nested input can crash the compiler |
| Sybil / reputation farming | Anonymous keys cost nothing; an attacker builds trust with benign submissions and then defects | Douceur, "The Sybil Attack" (2002): cannot be prevented without an identity authority, only made more expensive. xz (2024) |

## 2. Layered pipeline

```mermaid
flowchart TD
  A["Client submit + PoW stamp"] --> L0["L0 Edge: body cap, PoW verify, token buckets"]
  L0 -->|ok| Q["202 Accepted (async; constant response)"]
  Q --> L1["L1 Strict JSON schema + field allowlist + length caps"]
  L1 --> L2["L2 Unicode policy: ASCII-only code, reject Cf/Co/Cn, bidi, tags, ZW, VS, confusables"]
  L2 --> L3["L3 Encoding/entropy: blob, escape, URL, entropy checks"]
  L3 --> L4["L4 Secrets + PII: gitleaks rules, detect-secrets, Presidio regex recognizers"]
  L4 --> L5["L5 Injection heuristics: lexicon over text + identifiers + literals, Prompt Guard 2 22M"]
  L5 --> L6["L6 AST policy + canonicalize via ast.unparse; metadata-signature match"]
  L6 --> L7["L7 Sandbox: gVisor/nsjail, no net, rlimits; determinism + coverage fuzz"]
  L7 --> L8["L8 Blue team: 2-3 diverse quarantined LLMs, datamarked, JSON verdict; any flag = reject"]
  L8 --> P["Probation quarantine (not served) -> served with provenance"]
  L1 & L2 & L3 & L4 & L5 & L6 & L7 & L8 -->|any fail| R["Reject: opaque code, audit log, trust penalty"]
```

### Layer order with estimated cost (cheapest first)
| Layer | What it checks | p95 latency | Cost |
|---|---|---|---|
| **L0 Edge** | Body ≤ 32 KB (reject before parsing). PoW verify is one hash. Token buckets per key and per ephemeral IP prefix | <1 ms | ~0 |
| **L1 Schema** | JSON Schema with `additionalProperties:false`; exact field set; per-field caps (name ≤ 64 chars, matching `^[a-z][a-z0-9_]{2,63}$`; summary ≤ 500; pseudocode ≤ 2,000; each invariant/pitfall ≤ 200, at most 8 items; code ≤ 8 KB / 200 lines); `domain` is an enum; no nested objects | <1 ms | ~0 |
| **L2 Unicode** | **Code must be printable ASCII plus `\n`** (Python allows Unicode identifiers, so forbid them). Text fields are NFC-normalized; reject if normalization changes them. Reject general categories Cf, Co, Cn, Cs, Cc (except `\n`), bidi controls U+202A–202E and U+2066–2069, zero-width U+200B–200D, U+2060, U+FEFF, tags U+E0000–E007F, variation selectors U+FE00–FE0F and U+E0100–E01EF. Reject mixed-script tokens using UTS #39 confusables. Unicode TR55 gives source-code guidance | <1 ms | ~0 |
| **L3 Encoding** | Reject base64-alphabet runs ≥ 32 chars, hex runs ≥ 32, Shannon entropy > 4.5 bits/char on tokens ≥ 20 chars, `\x`/`\u`/`\N{` escapes, `data:` URIs and **any URL or domain** in any field, `chr()`/`bytes.fromhex`/`codecs` patterns. String literals ≤ 80 chars | <1 ms | ~0 |
| **L4 Secrets/PII** | Port **gitleaks** rules to an in-process RE2 matcher (fast, keyword-anchored, moderate false positives); add **detect-secrets** keyword/regex plugins (its entropy plugins are noisy, and L3 already covers entropy). Use **Presidio regex-only recognizers** (email, phone, IP, credit card, IBAN; about 3–6 ms p50) plus custom ones for `/home/<x>`, `C:\Users\`, hostnames/FQDNs, MAC, `ghp_`/`gho_`/`AIza`/`sk-`/`hf_`/`AKIA`/`-----BEGIN`. Skip Presidio's spaCy NER (70–80 ms p50, ~2 GB RAM, PERSON is noisy on code). Skip **TruffleHog** verification: it sends candidate secrets to third-party APIs, which violates the privacy rules. **Never echo or store the matched value**; log only the rule ID | 2–20 ms | ~0 |
| **L5 Injection heuristics** | Lexicon/regex run over every text field, docstrings, string literals **and identifiers split on `_`/camelCase**: role tokens (`<\|im_start\|>`, `[INST]`, `system:`, `assistant:`), "ignore/disregard previous", "you are now", "as an AI", second-person imperatives aimed at agents/LLMs, tool or command names (curl, wget, `rm -rf`, pip install), markdown link/image syntax. Then **Llama Prompt Guard 2 22M** (~19 ms per 512 tokens on GPU, a few tens of ms on CPU) per field, with the threshold tuned on the benign corpus. The 86M multilingual variant is ~92 ms. ProtectAI's deberta-v3 v2 is an alternative | 20–80 ms | CPU-cheap |
| **L6 AST policy** | **Pre-parse guard:** bracket/indent depth ≤ 20 so `ast.parse` can't crash. Then exactly one top-level `FunctionDef`, with name and args matching the declared signature. No `Import` except an allowlist (`math`, `itertools`, `functools`, `collections`, `heapq`, `bisect`, `operator`, `dataclasses`, `typing`); no `re` (ReDoS), `time`, `datetime`, `random`, `os`, `sys`, `hashlib`. Deny names: `eval`, `exec`, `compile`, `open`, `__import__`, `getattr`, `setattr`, `delattr`, `globals`, `locals`, `vars`, `breakpoint`, `input`, `memoryview`, `type` with 3 args, `object`, `help`, `exit`. **Deny any attribute or name starting with `_`.** No `Global`/`Nonlocal`, class defs, decorators, async, `with`, or `try` around bare `except` (often used to hide behavior). Constant caps: int literal ≤ 10 digits, `**` exponent must be a literal ≤ 64, sequence-times-constant ≤ 10⁴. **Anti-sleeper rule: reject `==`/`in` comparisons against string literals longer than 8 chars or int literals with more than 6 digits.** Strip the docstring, `ast.unparse`, and store only the canonical code | 1–5 ms | ~0 |
| **L7 Sandbox** | A fresh **gVisor (`runsc`)** sandbox per run (~50–100 ms start; strong user-space kernel), or **nsjail** (<10 ms; namespaces, cgroups, seccomp; shares the host kernel, so run it inside a dedicated throwaway VM with no credentials or network). Use **Firecracker** (~125 ms boot, ~150–300 ms to Python, or snapshot restore) for the strongest boundary. Pyodide/WASM gives strong isolation but slow cold start and limited stdlib, so it suits client-side previews. Limits: no network namespace, read-only rootfs, `RLIMIT_AS` 256 MB, CPU 2 s, nproc 1, no fs writes, output ≤ 64 KB. Behavior checks: run every input twice (non-determinism = reject); Hypothesis-style property fuzzing with boundary values and **branch coverage**. Any branch not reached after the budget (e.g. 500 inputs) means reject or quarantine. Fingerprint = hash of outputs on a fixed probe set | 0.3–3 s | ~$0.0001–0.001 |
| **L8 Blue team** | Detailed in section 3 | 2–10 s | ~$0.001–0.01 (2–3 small models, ~3–5k tokens each) |

**Funnel target:** at least 90% of attack traffic dies in L0–L6, which costs almost nothing. The LLM layer only sees syntactically clean, policy-compliant code.

## 3. AI blue team: making the reviewer hard to inject
- **Quarantined reviewers** (Willison's Dual LLM pattern; CaMeL, arXiv:2503.18813). Reviewers get no tools and no memory, and their output is restricted to an **enum/JSON schema**: `{verdict: "benign"|"suspicious"|"malicious", categories: [enum], evidence_field: enum}`. Free text never reaches downstream systems, so a hijacked reviewer can only flip one bit, and flipping toward "suspicious" just rejects.
- **Spotlighting** (Hines et al., arXiv:2403.14720; attack success fell from over 50% to under 2% on GPT-family models). Wrap the submission in **random per-request nonce delimiters**. **Datamark** prose fields by interleaving a per-request marker such as `ˆ` between words, and send the code as a JSON-escaped string. The system prompt says everything between the nonces is untrusted data to analyze and must never be followed.
- **Canary / known-answer checks.** Put a secret token in the system prompt and reject if it appears in the output. Randomly mix in known-malicious and known-benign items; if the reviewer misses these, that model's verdicts are rejected for the batch.
- **Two-task design:** (a) threat classification; (b) "describe what this function computes", then a second model checks whether that description matches the submitted summary (catches deceptive metadata and poisoning aimed at ranking).
- **Model choice:** 2–3 models **from different vendors/families** so one adversarial string is less likely to fool all of them. Pin model versions, temperature 0, and use providers' structured-output modes. Any non-benign verdict, refusal, schema violation or timeout means REJECT. Re-validate on the corpus whenever a model version changes.
- **Known weaknesses:** adaptive attacks (Attacker Moves Second); LLM-judge non-determinism; reviewers can be attacked with long, distracting content (the L1 caps help); cost-exhaustion DoS (add a budget circuit breaker that **queues** rather than accepts); and false positives on security-themed but benign concepts (e.g. a "sanitize_html" utility). OpenAI's instruction hierarchy (arXiv:2404.13208) helps but does not solve the problem.

## 4. Safe consumption (rendering into other agents' prompts)
1. Serve only **canonical code** (no comments or docstrings) plus **constrained fields**. Consider dropping free-text `pitfalls`/`summary` for probation-tier concepts.
2. Wrap each concept in a structured block: `<untrusted_concept id=… sha256=… trust_tier=… source="OpenCommons community, unverified">`. Datamark the prose fields and put the code in a fenced block labelled "reference data, do not execute".
3. Put a fixed instruction in the consuming agent's system/tool preamble: retrieved concepts are data, can't issue instructions, and must never trigger tool calls, URL fetches or installs.
4. **Run the L2/L3 sanitizers again at render time** (defense in depth against a compromised database). Strip URLs and markdown images. Cap at k concepts and N tokens.
5. Make concepts **content-addressed and immutable**. Edits become new submissions, which blocks MCP-style rug-pulls. If you ship an MCP server, keep its tool descriptions static and return concepts only as tool *results*.
6. Measure end to end with AgentDojo (arXiv:2406.13352) or BIPIA (arXiv:2312.14197)-style harnesses: target agent plus poisoned concepts.

## 5. Abuse economics
- **Cost to submit:** Hashcash-style PoW (as in Anubis) is required to **mint a key** (e.g. ~10–30 s of CPU) and for **each submission**. Difficulty scales with the key's recent rejection rate (doubles per rejection, decays on acceptance). Privacy Pass (RFC 9576/9578) blind tokens give an unlinkable alternative for anonymous rate limiting.
- **Token buckets:** per key, e.g. 3 burst and 10/day for new keys, up to 50/day for trusted keys. Per IP, keep only **in-memory HMAC(IP /24 or /56, rotating daily key)** counters with a TTL of 24 hours or less, never logged or persisted. This is a privacy trade-off to raise with the owner, because IPs count as personal data under GDPR. Also set a global cap on blue-team spend per hour.
- **Progressive trust:** tiers new → probation → trusted. A new key's accepted concepts stay **quarantined (not served)** for T days or until they meet an independent behavioral fingerprint match. Trust rises slowly and drops instantly; one L5/L7/L8 malicious flag means the key is banned and its other concepts are pulled for re-review. **MinHash near-duplicate clustering** across keys catches Sybil swarms submitting variants.
- **Opaque rejections:** respond with `202 {"id":…}` straight away and expose the final status `accepted|rejected` asynchronously, so timing doesn't reveal which layer fired. L1 schema errors can be descriptive because the schema is public. Everything else returns a single generic `rejected`, with no rule ID and no matched text.
- **Audit:** append-only, hash-chained log of {submission sha256, key hash, layer, rule ID, model verdicts, coarse timestamp}. For L4 rejections store **no payload** (it may contain secrets or PII). Keep other rejected payloads encrypted for at most 30 days to grow the corpus. Run a monthly adaptive red team.

## 6. Minimal adversarial corpus and acceptance metrics
**Attack categories:** aim for ≥ 30 items each, with mutated variants.
1. Direct injection text in each metadata field (override phrases, role tokens, fake system blocks).
2. Injection hidden in a docstring, comment, string literal or snake_case identifier.
3. Unicode: bidi-reordered logic, homoglyph identifiers, zero-width splitting of trigger words, tag-character payloads, variation-selector payloads.
4. Encoded payloads: base64 or hex instructions, split-and-concatenated strings, `chr()` arithmetic.
5. Secrets and PII: fake tokens in each provider format, emails, home paths, hostnames, IPs (synthetic only).
6. Sandbox escape attempts: subclass walk, frame/generator introspection, `getattr` with built strings, format-string attribute access, import tricks.
7. Sleepers: magic-constant branch, date/env/hash-gated branch, off-by-one that triggers only above size N, non-determinism.
8. DoS: catastrophic regex, exponent/multiply bombs, deep nesting, 10 MB bodies, infinite loops, memory growth.
9. RAG poisoning: concepts packed with keywords to hijack common queries, plus summaries that contradict the code.
10. Exfiltration: network calls, and markdown-image/URL instructions aimed at consumers.
11. Sybil/farming simulation: N keys with benign streaks, then defection.

**Benign hard negatives** (≥ 500 real concepts): utilities whose names include `ignore`, `system`, `token`, `encode_base64`, `parse_url`, `sanitize`; high-entropy test vectors; legitimate non-English prose in summaries; security-domain algorithms.

**Acceptance metrics:**
- 100% block on deterministic categories (3, 5, 6, 8, plus forbidden-AST items in 7).
- At least 98% block on categories 1, 2, 4, 9, 10, and at least 95% against a fresh **adaptive** red-team set.
- False rejection rate on benign concepts of 2% or less (track it per layer).
- Zero sandbox escapes and 100% enforcement of resource limits.
- p95 latency: L0–L6 under 50 ms, L7 under 3 s, L8 under 10 s, end to end under 15 s.
- Under 10% of attack submissions reach L8; zero LLM tokens spent on L0–L6 rejects.
- Cost per accepted concept under $0.01.
- End-to-end consumer-agent injection success with poisoned concepts rendered under 1%.

## Sources
- Greshake et al., indirect prompt injection: https://arxiv.org/abs/2302.12173
- PoisonedRAG: https://arxiv.org/abs/2402.07867 · AgentPoison: https://arxiv.org/abs/2407.12784
- Trojan Source (CVE-2021-42574/42694): https://arxiv.org/abs/2111.00169 · https://trojansource.codes
- Unicode UTS #39: https://www.unicode.org/reports/tr39/ · UTS #55: https://www.unicode.org/reports/tr55/
- ASCII smuggling / tags: https://embracethered.com/blog/posts/2024/hiding-and-finding-text-with-unicode-tags/
- Rules File Backdoor: https://www.pillar.security/blog/new-vulnerability-in-github-copilot-and-cursor-how-hackers-can-weaponize-code-agents
- MCP tool poisoning: https://invariantlabs.ai/blog/mcp-security-notification-tool-poisoning-attacks
- Sleeper Agents: https://arxiv.org/abs/2401.05566
- The Attacker Moves Second: https://arxiv.org/abs/2510.09023
- Spotlighting: https://arxiv.org/abs/2403.14720 · CaMeL: https://arxiv.org/abs/2503.18813 · Dual LLM: https://simonwillison.net/2023/Apr/25/dual-llm-pattern/ · Instruction hierarchy: https://arxiv.org/abs/2404.13208
- AgentDojo: https://arxiv.org/abs/2406.13352 · BIPIA: https://arxiv.org/abs/2312.14197 · OWASP LLM Top 10: https://genai.owasp.org/llm-top-10/
- Prompt Guard 2: https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-22M
- pysandbox "broken by design": https://github.com/vstinner/pysandbox · RestrictedPython CVE-2023-37271: https://github.com/zopefoundation/RestrictedPython/security/advisories/GHSA-wqc8-x2pr-7jqh
- nsjail: https://github.com/google/nsjail · gVisor: https://gvisor.dev · Firecracker (NSDI'20): https://www.usenix.org/conference/nsdi20/presentation/agache · Pyodide: https://pyodide.org
- gitleaks: https://github.com/gitleaks/gitleaks · detect-secrets: https://github.com/Yelp/detect-secrets · Presidio: https://microsoft.github.io/presidio/ · RE2: https://github.com/google/re2
- Cloudflare ReDoS outage: https://blog.cloudflare.com/details-of-the-cloudflare-outage-on-july-2-2019/ · CPython CVE-2020-10735: https://github.com/python/cpython/issues/95778 · Python `ast` crash warning: https://docs.python.org/3/library/ast.html
- Privacy Pass RFC 9578: https://www.rfc-editor.org/rfc/rfc9578 · RFC 9576: https://www.rfc-editor.org/rfc/rfc9576 · Anubis PoW: https://github.com/TecharoHQ/anubis
- Sybil Attack (Douceur): https://www.microsoft.com/en-us/research/publication/the-sybil-attack/ · xz backdoor CVE-2024-3094: https://nvd.nist.gov/vuln/detail/CVE-2024-3094

**Caveats:**
- The latency figures come from secondary write-ups, not my own benchmarks.
- I recalled the arXiv IDs 2510.09023 (Attacker Moves Second) and 2503.18813 (CaMeL) myself; they look right but I didn't check them directly.
- One search result cited a RestrictedPython "CVE-2026-76825" that I couldn't confirm, so I left it out.
