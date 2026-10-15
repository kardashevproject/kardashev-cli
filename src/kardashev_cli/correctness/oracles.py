"""Trusted reference implementations, keyed by canonical concept name.

These functions run in the *verifier's* process (never submitted code).
They are deliberately written in plain stdlib Python with numerically
careful formulations (the same "stable" tricks a correct submission should
use), so disagreement with a submitted concept at generated inputs is a real
correctness signal rather than a precision artifact.

Matching is name-only: the submitted function name is stripped of qualifier
words (stable/safe/fast/...) and compared *exactly* against each oracle's
alias set. Summaries are only used to disambiguate declared variants
(population vs sample variance, nats vs bits entropy, normalized vs
unnormalized sinc) -- never to guess a match that the name doesn't support.
"""

from __future__ import annotations

import math
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

QUALIFIER_WORDS = {
    "stable",
    "safe",
    "fast",
    "exact",
    "naive",
    "robust",
    "compute",
    "numerically",
    "vectorized",
    "simple",
    "basic",
    "safe2",
    "v2",
    "impl",
    "my",
}


def canonicalize_name(name: str) -> str:
    tokens = [t for t in name.strip().lower().split("_") if t]
    tokens = [t for t in tokens if t not in QUALIFIER_WORDS]
    return "_".join(tokens)


@dataclass
class OracleMatch:
    key: str
    reference: Callable[..., Any]
    note: str = ""


# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _logsumexp(xs: list[float]) -> float:
    if not xs:
        raise ValueError("logsumexp of empty sequence")
    m = max(xs)
    if math.isinf(m) and m > 0:
        return m
    return m + math.log(math.fsum(math.exp(v - m) for v in xs))


def _softplus(x: float) -> float:
    if x > 0:
        return x + math.log1p(math.exp(-x))
    return math.log1p(math.exp(x))


def _logit(p: float) -> float:
    return math.log(p) - math.log1p(-p)


def _gelu(x: float) -> float:
    return 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))


def _silu(x: float) -> float:
    return x * _sigmoid(x)


def _mean(xs: list[float]) -> float:
    if not xs:
        raise ValueError("mean of empty sequence")
    return math.fsum(xs) / len(xs)


def _variance(xs: list[float], *, sample: bool) -> float:
    n = len(xs)
    denom = (n - 1) if sample else n
    if denom <= 0:
        raise ValueError("variance undefined for this length")
    m = math.fsum(xs) / n
    return math.fsum((x - m) ** 2 for x in xs) / denom


def _std(xs: list[float], *, sample: bool) -> float:
    return math.sqrt(_variance(xs, sample=sample))


def _hypot(a: float, b: float) -> float:
    return math.hypot(a, b)


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float, *, radius: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def _gcd(a: int, b: int) -> int:
    return math.gcd(a, b)


def _extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    old_r, r = a, b
    old_s, s = 1, 0
    old_t, t = 0, 1
    while r != 0:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t
    return old_r, old_s, old_t


def _modpow(a: int, b: int, m: int) -> int:
    return pow(a, b, m)


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def _crc32(data) -> int:
    if isinstance(data, str):
        data = data.encode("utf-8")
    elif isinstance(data, (list, tuple)):
        data = bytes(int(b) & 0xFF for b in data)
    return zlib.crc32(bytes(data)) & 0xFFFFFFFF


def _fnv1a_32(data) -> int:
    if isinstance(data, str):
        data = data.encode("utf-8")
    elif isinstance(data, (list, tuple)):
        data = bytes(int(b) & 0xFF for b in data)
    h = 0x811C9DC5
    for byte in data:
        h ^= byte
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def _entropy(probs: list[float], *, base2: bool) -> float:
    total = 0.0
    for p in probs:
        if p <= 0:
            continue
        total -= p * math.log(p, 2 if base2 else math.e)
    return total


def _kl_divergence(p: list[float], q: list[float], *, base2: bool) -> float:
    total = 0.0
    for pi, qi in zip(p, q):
        if pi <= 0:
            continue
        total += pi * math.log(pi / qi, 2 if base2 else math.e)
    return total


def _cosine_similarity(u: list[float], v: list[float]) -> float:
    dot = math.fsum(a * b for a, b in zip(u, v))
    nu = math.sqrt(math.fsum(a * a for a in u))
    nv = math.sqrt(math.fsum(b * b for b in v))
    if nu == 0 or nv == 0:
        raise ValueError("cosine similarity undefined for a zero vector")
    return dot / (nu * nv)


def _softmax(xs: list[float]) -> list[float]:
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = math.fsum(exps)
    return [e / s for e in exps]


def _quadratic_roots(a: float, b: float, c: float) -> tuple:
    if a == 0:
        if b == 0:
            raise ValueError("not a quadratic")
        r = -c / b
        return (r, r)
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    sq = math.sqrt(disc)
    if b >= 0:
        q = -0.5 * (b + sq)
    else:
        q = -0.5 * (b - sq)
    if q == 0:
        return (0.0, 0.0)
    r1 = q / a
    r2 = c / q
    return tuple(sorted((r1, r2)))


def _sinc(x: float, *, normalized: bool) -> float:
    if x == 0:
        return 1.0
    arg = math.pi * x if normalized else x
    return math.sin(arg) / arg


def _horner(coeffs: list[float], x: float) -> float:
    result = 0.0
    for c in reversed(coeffs):
        result = result * x + c
    return result


def _binomial(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


def _factorial(n: int) -> int:
    return math.factorial(n)


def _fibonacci(n: int) -> int:
    if n < 0:
        raise ValueError("fibonacci undefined for negative n")
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def _clamp(x: float, lo: float, hi: float) -> float:
    return min(max(x, lo), hi)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _sum_fsum(xs: list[float]) -> float:
    return math.fsum(xs)


# ---------------------------------------------------------------------------
# Variant-aware oracle wrappers
#
# Some references have a declared variant (population/sample variance, nats
# /bits entropy, normalized/unnormalized sinc, haversine's radius). Each
# wrapper takes the concept's summary text and returns (reference_fn, note),
# or a list of (reference_fn, note) candidates to try when the variant is
# undeclared (per spec: match either -> verified-with-note; match neither ->
# refute).
# ---------------------------------------------------------------------------


def _declares(summary: str, *words: str) -> bool:
    low = summary.lower()
    return any(w in low for w in words)


def _variance_candidates(summary: str):
    wants_sample = _declares(summary, "sample")
    wants_pop = _declares(summary, "population")
    if wants_sample and not wants_pop:
        return [(lambda xs: _variance(xs, sample=True), "sample variance")]
    if wants_pop and not wants_sample:
        return [(lambda xs: _variance(xs, sample=False), "population variance")]
    return [
        (lambda xs: _variance(xs, sample=False), "population variance (undeclared, matched)"),
        (lambda xs: _variance(xs, sample=True), "sample variance (undeclared, matched)"),
    ]


def _std_candidates(summary: str):
    wants_sample = _declares(summary, "sample")
    wants_pop = _declares(summary, "population")
    if wants_sample and not wants_pop:
        return [(lambda xs: _std(xs, sample=True), "sample std")]
    if wants_pop and not wants_sample:
        return [(lambda xs: _std(xs, sample=False), "population std")]
    return [
        (lambda xs: _std(xs, sample=False), "population std (undeclared, matched)"),
        (lambda xs: _std(xs, sample=True), "sample std (undeclared, matched)"),
    ]


def _entropy_candidates(summary: str):
    wants_bits = _declares(summary, "bit", "bits", "log2")
    wants_nats = _declares(summary, "nat", "nats", "ln ")
    if wants_bits and not wants_nats:
        return [(lambda p: _entropy(p, base2=True), "entropy in bits")]
    if wants_nats and not wants_bits:
        return [(lambda p: _entropy(p, base2=False), "entropy in nats")]
    return [
        (lambda p: _entropy(p, base2=False), "entropy in nats (undeclared, matched)"),
        (lambda p: _entropy(p, base2=True), "entropy in bits (undeclared, matched)"),
    ]


def _kl_candidates(summary: str):
    wants_bits = _declares(summary, "bit", "bits", "log2")
    if wants_bits:
        return [(lambda p, q: _kl_divergence(p, q, base2=True), "kl divergence in bits")]
    return [
        (lambda p, q: _kl_divergence(p, q, base2=False), "kl divergence in nats (undeclared, matched)"),
        (lambda p, q: _kl_divergence(p, q, base2=True), "kl divergence in bits (undeclared, matched)"),
    ]


def _sinc_candidates(summary: str):
    wants_unnorm = _declares(summary, "unnormalized", "unnormalised")
    if wants_unnorm:
        return [(lambda x: _sinc(x, normalized=False), "unnormalized sinc")]
    return [(lambda x: _sinc(x, normalized=True), "normalized sinc (default convention)")]


def _haversine_candidates(summary: str):
    return [
        (lambda lat1, lon1, lat2, lon2: _haversine(lat1, lon1, lat2, lon2, radius=6371.0088), "haversine, R=6371.0088 km"),
        (lambda lat1, lon1, lat2, lon2: _haversine(lat1, lon1, lat2, lon2, radius=6371.0), "haversine, R=6371 km"),
    ]


# key -> list of (alias names stripped of qualifiers) that map to it, a
# candidate-producing function (summary -> list[(ref_fn, note)]) and arity.
_SIMPLE = lambda fn, note="": (lambda _summary: [(fn, note)])

REGISTRY: dict[str, Callable[[str], list[tuple[Callable, str]]]] = {}
ALIASES: dict[str, str] = {}


def _register(canonical_key: str, aliases: list[str], candidates_fn) -> None:
    REGISTRY[canonical_key] = candidates_fn
    for alias in aliases + [canonical_key]:
        ALIASES[alias] = canonical_key


_register("sigmoid", ["logistic", "expit"], _SIMPLE(_sigmoid))
_register("logsumexp", [], _SIMPLE(_logsumexp))
_register("softplus", [], _SIMPLE(_softplus))
_register("logit", [], _SIMPLE(_logit))
_register("gelu", [], _SIMPLE(_gelu))
_register("silu", ["swish"], _SIMPLE(_silu))
_register("mean", ["average"], _SIMPLE(_mean))
_register("variance", ["var"], _variance_candidates)
_register("std", ["stddev", "standard_deviation"], _std_candidates)
_register("hypot", [], _SIMPLE(_hypot))
_register("haversine", ["haversine_distance"], _haversine_candidates)
_register("gcd", ["greatest_common_divisor"], _SIMPLE(_gcd))
_register("extended_gcd", ["ext_gcd", "egcd"], _SIMPLE(_extended_gcd))
_register("modpow", ["mod_pow", "powmod", "modular_pow", "modular_exponentiation"], _SIMPLE(_modpow))
_register("is_prime", ["isprime", "primality", "primality_test"], _SIMPLE(_is_prime))
_register("crc32", [], _SIMPLE(_crc32))
_register("fnv1a_32", ["fnv1a", "fnv_hash", "fnv1a32"], _SIMPLE(_fnv1a_32))
_register("entropy", ["shannon_entropy"], _entropy_candidates)
_register("kl_divergence", ["kldivergence", "kl"], _kl_candidates)
_register("cosine_similarity", ["cosine_sim", "cossim"], _SIMPLE(_cosine_similarity))
_register("softmax", [], _SIMPLE(_softmax))
_register("quadratic_roots", ["solve_quadratic", "quadratic_formula"], _SIMPLE(_quadratic_roots))
_register("sinc", [], _sinc_candidates)
_register("horner", ["polyval", "poly_eval", "polynomial_eval"], _SIMPLE(_horner))
_register("binomial_coefficient", ["binomial", "choose", "comb", "n_choose_k"], _SIMPLE(_binomial))
_register("factorial", [], _SIMPLE(_factorial))
_register("fibonacci", ["fib"], _SIMPLE(_fibonacci))
_register("clamp", ["clip"], _SIMPLE(_clamp))
_register("lerp", ["linear_interpolation", "linterp"], _SIMPLE(_lerp))
_register("sum", ["total", "fsum"], _SIMPLE(_sum_fsum))


def lookup_oracle(func_name: str, summary: str) -> OracleMatch | None:
    """Find the oracle matching ``func_name`` exactly (after stripping qualifiers)."""
    canonical = canonicalize_name(func_name)
    key = ALIASES.get(canonical)
    if key is None:
        return None
    candidates = REGISTRY[key](summary or "")
    if len(candidates) == 1:
        ref, note = candidates[0]
        return OracleMatch(key=key, reference=ref, note=note)
    # Multiple undeclared-variant candidates: caller tries each in turn.
    return OracleMatch(key=key, reference=candidates, note="multi-candidate")


def candidates_for(func_name: str, summary: str) -> list[tuple[Callable, str]] | None:
    """Return the list of (reference_fn, note) candidates for a name, or None if unmatched."""
    canonical = canonicalize_name(func_name)
    key = ALIASES.get(canonical)
    if key is None:
        return None
    return REGISTRY[key](summary or "")


DEFAULT_ORACLES = REGISTRY
