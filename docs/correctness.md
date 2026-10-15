# Correctness verification

Every concept contributed to OpenCommons goes through a fixed set of
deterministic checks before it can be trusted. This explains what the
checks do, what the three statuses mean, and how to get your concept to
`verified`.

Nothing here is a human reviewer or a vote. Every check is deterministic:
run it twice on the same concept and you get the same answer.

## Statuses

- **refuted** — at least one check found a concrete disagreement: a
  submitted example that doesn't hold, an invariant that fails, a reference
  implementation that disagrees, a stability claim that doesn't survive
  extreme inputs, or output that changes between two identical calls.
  Refuted concepts are never ranked and never marked peer-reviewed.
- **verified** — either (a) the function's name matches a known concept
  (mean, sigmoid, gcd, haversine, ...) and it agrees with a trusted
  reference implementation on a range of generated inputs, or (b) it has at
  least 3 passing contract examples *and* at least one executable invariant
  that held on 20+ generated inputs, with nothing refuted.
- **unverifiable** — nothing was refuted, but there also wasn't enough
  evidence for `verified`. This is the default for a genuinely novel
  function (no reference implementation exists for it) with thin evidence.
  It's a candidate, not a trusted result — it is never ranked as peer
  reviewed.

A contract crafted to match a bug (the code and the examples agree with
each other, but both are wrong) does not reach `verified`: if a reference
implementation exists, it still gets compared against that, independent of
what the submitter's own examples say.

## What gets checked

1. **Contract.** The submitter's own I/O examples are run and compared to
   the function's actual output (float tolerance: relative 1e-9, absolute
   1e-12; NaN is treated as equal to NaN; lists/tuples compared
   elementwise).
2. **Executable invariants.** An invariant like `"out >= 0"` or
   `"not (p > 0) or math.isclose(f(p), 1 - f(-p))"` is parsed and checked
   against a narrow allowlist (no imports, no dunders, no attribute access
   outside `math`, calls only to `f`, `math.*`, or a handful of safe
   builtins). A valid invariant is then evaluated on ~30 generated inputs
   (0, ±1, tiny, huge, empty list, singleton, long list, plus pseudo-random
   values). An invariant that doesn't parse as a safe boolean expression is
   recorded as **unchecked** — free text is never counted as a passing
   check.
3. **Reference oracle.** If the function's name (after stripping words like
   `stable`/`safe`/`fast`/`naive`/`robust`) matches a known concept —
   sigmoid, logsumexp, softplus, mean, variance, std, hypot, haversine,
   gcd, modpow, is_prime, crc32, entropy, cosine similarity, softmax,
   quadratic roots, sinc, binomial coefficient, factorial, fibonacci,
   clamp, lerp, and others — its output is compared against a trusted
   reference implementation on generated inputs. Matching is exact on the
   stripped name; a summary is only used to pick between declared variants
   (population vs. sample variance, nats vs. bits entropy), never to guess
   a match the name doesn't support.
4. **Claimed robustness.** If the name or summary claims stability or
   overflow-safety, the function is run on extreme-magnitude inputs
   (±1e3, ±1e300, ±1e-300) and must still return the correct, finite
   answer wherever the true answer is finite. Without such a claim, the
   same failures don't refute the concept — they're recorded as a pitfall
   so later readers know where the edges are.
5. **Determinism.** The same inputs are run twice; a different answer the
   second time is a refutation.

Submitted code never runs inside the verifier's process. Every check that
needs to execute it goes through a resource-limited subprocess (`python
-I`, CPU/memory/wall-clock limits, no inherited environment), or through
whatever sandboxed executor the surrounding pipeline provides instead.

## Getting to `verified`

- If your concept's name matches a well-known function, just make sure it
  agrees with the standard definition — that alone is enough.
- Otherwise, add at least 3 contract examples and at least one executable
  invariant that actually holds (watch for invariants that are only true
  within a domain — e.g. a `clamp(x, lo, hi)` invariant should account for
  `lo > hi`, and a probability-based invariant should only be asserted
  over inputs that are actually probabilities).
- A free-text invariant ("should feel reasonable") is never checked and
  never counts toward `verified`.

## Reporting a wrong result

If you believe a `verified` concept is actually wrong, report a concrete
counterexample: an input and the output you expect. It's only accepted as
a real refutation if the input satisfies the concept's declared parameter
types (a string where `list[float]` is declared doesn't count) and the
function returns a genuinely different value than expected — an exception
on an out-of-contract input is not, by itself, a refutation.
