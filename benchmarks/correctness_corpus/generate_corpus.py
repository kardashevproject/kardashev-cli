"""Generates benchmarks/correctness_corpus/corpus.jsonl.

Run with: python -m benchmarks.correctness_corpus.generate_corpus

The corpus is checked in (corpus.jsonl), so this script only needs to be
re-run when items are added or changed; the runner and pytest both read the
checked-in file directly.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

Item = dict[str, Any]

ITEMS: list[Item] = []


def add(
    item_id: str,
    source_code: str,
    summary: str,
    invariants: list[str],
    contract: list[list],
    label: str,
    category: str,
    param_types: list[str] | None = None,
) -> None:
    ITEMS.append(
        {
            "id": item_id,
            "source_code": source_code,
            "summary": summary,
            "invariants": invariants,
            "contract": contract,
            "label": label,
            "category": category,
            "param_types": param_types,  # only needed when contract is empty
        }
    )


# ---------------------------------------------------------------------------
# The three real failures from the bug report.
# ---------------------------------------------------------------------------

add(
    "real_001_stable_logsumexp_returns_max",
    "def stable_logsumexp(xs):\n    return max(xs)\n",
    "Numerically stable log-sum-exp of a list of floats.",
    ["out >= max(xs)"],
    [[[[1.0, 2.0, 3.0]], 3.408605001]],
    "wrong",
    "wrong_formula",
)

add(
    "real_002_sign_flipped_sigmoid",
    "import math\ndef stable_sigmoid(x):\n    return 1.0 / (1.0 + math.exp(x))\n",
    "Numerically stable sigmoid / logistic function.",
    ["0.0 <= out <= 1.0"],
    [[[2.0], 0.8807970779778823]],
    "wrong",
    "sign_error",
)

add(
    "real_003_mean_divides_by_len_plus_one",
    "def mean(xs):\n    return sum(xs) / (len(xs) + 1)\n",
    "Arithmetic mean of a list of floats.",
    ["min(xs) <= out <= max(xs)"],
    [[[[2.0, 4.0]], 3.0]],
    "wrong",
    "off_by_one",
)

# ---------------------------------------------------------------------------
# Correct implementations (including unusual-but-valid hard negatives).
# ---------------------------------------------------------------------------

add(
    "correct_001_mean_fsum",
    "import math\ndef mean(xs):\n    return math.fsum(xs) / len(xs)\n",
    "Arithmetic mean of a list of floats.",
    ["min(xs) <= out <= max(xs)"],
    [[[[2.0, 4.0]], 3.0], [[[1.0, 1.0, 1.0]], 1.0], [[[0.0]], 0.0]],
    "correct",
    "correct",
)

add(
    "correct_002_mean_running_average_iterative",
    (
        "def mean(xs):\n"
        "    total = 0.0\n"
        "    for i, x in enumerate(xs, start=1):\n"
        "        total += (x - total) / i\n"
        "    return total\n"
    ),
    "Arithmetic mean computed as an online running average (Welford style, no sum).",
    ["min(xs) <= out <= max(xs)"],
    [[[[2.0, 4.0]], 3.0], [[[1.0, 1.0, 1.0]], 1.0], [[[0.0]], 0.0]],
    "correct",
    "correct",
)

add(
    "correct_003_stable_sigmoid",
    (
        "import math\n"
        "def stable_sigmoid(x):\n"
        "    if x >= 0:\n"
        "        z = math.exp(-x)\n"
        "        return 1.0 / (1.0 + z)\n"
        "    z = math.exp(x)\n"
        "    return z / (1.0 + z)\n"
    ),
    "Numerically stable sigmoid, avoids overflow for very negative/positive inputs.",
    ["0.0 <= out <= 1.0"],
    [[[0.0], 0.5]],
    "correct",
    "correct",
)

add(
    "correct_004_logsumexp_stable",
    (
        "import math\n"
        "def stable_logsumexp(xs):\n"
        "    if not xs:\n"
        "        return float('-inf')\n"
        "    m = max(xs)\n"
        "    return m + math.log(sum(math.exp(v - m) for v in xs))\n"
    ),
    "Numerically stable log-sum-exp of a list of floats.",
    ["out >= max(xs)"],
    [[[[1.0, 2.0, 3.0]], 3.4076059644443806]],
    "correct",
    "correct",
)

add(
    "correct_005_gcd_euclid_iterative",
    "def gcd(a, b):\n    a, b = abs(a), abs(b)\n    while b:\n        a, b = b, a % b\n    return a\n",
    "Greatest common divisor via iterative Euclidean algorithm.",
    ["out >= 0"],
    [[[48, 18], 6], [[0, 5], 5]],
    "correct",
    "correct",
)

add(
    "correct_006_factorial_recursive",
    "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)\n",
    "Factorial of a non-negative integer, recursive implementation.",
    ["out >= 1"],
    [[[0], 1], [[5], 120]],
    "correct",
    "correct",
)

add(
    "correct_007_fibonacci_iterative",
    "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n",
    "The n-th Fibonacci number, iterative.",
    ["out >= 0"],
    [[[0], 0], [[1], 1], [[10], 55]],
    "correct",
    "correct",
)

add(
    "correct_008_clamp",
    "def clamp(x, lo, hi):\n    return min(max(x, lo), hi)\n",
    "Clamps x into the closed interval [lo, hi].",
    ["lo > hi or lo <= out <= hi"],
    [[[5.0, 0.0, 10.0], 5.0], [[-5.0, 0.0, 10.0], 0.0], [[15.0, 0.0, 10.0], 10.0]],
    "correct",
    "correct",
)

add(
    "correct_009_lerp",
    "def lerp(a, b, t):\n    return a + (b - a) * t\n",
    "Linear interpolation between a and b at parameter t.",
    [],
    [[[0.0, 10.0, 0.5], 5.0], [[0.0, 10.0, 0.0], 0.0], [[0.0, 10.0, 1.0], 10.0]],
    "correct",
    "correct",
)

add(
    "correct_010_softmax_stable",
    (
        "import math\n"
        "def softmax(xs):\n"
        "    m = max(xs)\n"
        "    exps = [math.exp(x - m) for x in xs]\n"
        "    s = sum(exps)\n"
        "    return [e / s for e in exps]\n"
    ),
    "Numerically stable softmax of a list of floats.",
    ["math.isclose(sum(out), 1.0, rel_tol=1e-6)"],
    [[[[0.0, 0.0]], [0.5, 0.5]]],
    "correct",
    "correct",
)

add(
    "correct_011_variance_population",
    (
        "def variance(xs):\n"
        "    n = len(xs)\n"
        "    m = sum(xs) / n\n"
        "    return sum((x - m) ** 2 for x in xs) / n\n"
    ),
    "Population variance of a list of floats.",
    ["out >= 0.0"],
    [[[[1.0, 2.0, 3.0, 4.0]], 1.25]],
    "correct",
    "correct",
)

add(
    "correct_012_variance_sample",
    (
        "def sample_variance(xs):\n"
        "    n = len(xs)\n"
        "    m = sum(xs) / n\n"
        "    return sum((x - m) ** 2 for x in xs) / (n - 1)\n"
    ),
    "Sample variance of a list of floats (Bessel-corrected).",
    ["out >= 0.0"],
    [[[[1.0, 2.0, 3.0, 4.0]], 1.6666666666666667]],
    "correct",
    "correct",
)

add(
    "correct_013_haversine_6371",
    (
        "import math\n"
        "def haversine(lat1, lon1, lat2, lon2):\n"
        "    R = 6371.0\n"
        "    p1, p2 = math.radians(lat1), math.radians(lat2)\n"
        "    dphi = math.radians(lat2 - lat1)\n"
        "    dlmb = math.radians(lon2 - lon1)\n"
        "    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2\n"
        "    return 2 * R * math.asin(math.sqrt(a))\n"
    ),
    "Great-circle distance between two lat/lon points in degrees, result in km.",
    ["out >= 0.0"],
    [[[0.0, 0.0, 0.0, 0.0], 0.0]],
    "correct",
    "correct",
)

add(
    "correct_014_is_prime_trial_division",
    (
        "def is_prime(n):\n"
        "    if n < 2:\n"
        "        return False\n"
        "    if n in (2, 3):\n"
        "        return True\n"
        "    if n % 2 == 0:\n"
        "        return False\n"
        "    i = 3\n"
        "    while i * i <= n:\n"
        "        if n % i == 0:\n"
        "            return False\n"
        "        i += 2\n"
        "    return True\n"
    ),
    "Primality test via trial division up to sqrt(n).",
    [],
    [[[2], True], [[4], False], [[17], True], [[1], False]],
    "correct",
    "correct",
)

add(
    "correct_015_binomial_coefficient_pascal",
    (
        "def binomial_coefficient(n, k):\n"
        "    if k < 0 or k > n:\n"
        "        return 0\n"
        "    row = [0] * (k + 1)\n"
        "    row[0] = 1\n"
        "    for i in range(1, n + 1):\n"
        "        for j in range(min(i, k), 0, -1):\n"
        "            row[j] += row[j - 1]\n"
        "    return row[k]\n"
    ),
    "Binomial coefficient n choose k, Pascal's-triangle row recurrence.",
    ["out >= 0"],
    [[[5, 2], 10], [[6, 0], 1]],
    "correct",
    "correct",
)

add(
    "correct_016_cosine_similarity",
    (
        "import math\n"
        "def cosine_similarity(u, v):\n"
        "    dot = sum(a * b for a, b in zip(u, v))\n"
        "    nu = math.sqrt(sum(a * a for a in u))\n"
        "    nv = math.sqrt(sum(b * b for b in v))\n"
        "    return dot / (nu * nv)\n"
    ),
    "Cosine similarity between two equal-length vectors.",
    ["-1.0 - 1e-9 <= out <= 1.0 + 1e-9"],
    [[[[1.0, 0.0], [1.0, 0.0]], 1.0], [[[1.0, 0.0], [0.0, 1.0]], 0.0]],
    "correct",
    "correct",
)

add(
    "correct_017_entropy_nats",
    "import math\ndef shannon_entropy(probs):\n    return -sum(p * math.log(p) for p in probs if p > 0)\n",
    "Shannon entropy of a probability distribution, in nats.",
    ["not all(0 <= p <= 1 for p in probs) or out >= 0.0"],
    [[[[0.5, 0.5]], 0.6931471805599453]],
    "correct",
    "correct",
)

add(
    "correct_018_entropy_bits",
    "import math\ndef entropy_bits(probs):\n    return -sum(p * math.log2(p) for p in probs if p > 0)\n",
    "Shannon entropy of a probability distribution, in bits.",
    ["not all(0 <= p <= 1 for p in probs) or out >= 0.0"],
    [[[[0.5, 0.5]], 1.0]],
    "correct",
    "correct",
)

add(
    "correct_019_horner_polyval",
    "def horner(coeffs, x):\n    result = 0.0\n    for c in reversed(coeffs):\n        result = result * x + c\n    return result\n",
    "Evaluates a polynomial given low-to-high coefficients at x, via Horner's method.",
    [],
    [[[[1.0, 0.0, 1.0], 2.0], 5.0]],
    "correct",
    "correct",
)

add(
    "correct_020_quadratic_roots_stable",
    (
        "import math\n"
        "def quadratic_roots(a, b, c):\n"
        "    disc = b * b - 4 * a * c\n"
        "    if disc < 0:\n"
        "        return None\n"
        "    sq = math.sqrt(disc)\n"
        "    q = -0.5 * (b + sq) if b >= 0 else -0.5 * (b - sq)\n"
        "    if q == 0:\n"
        "        return (0.0, 0.0)\n"
        "    return tuple(sorted((q / a, c / q)))\n"
    ),
    "Roots of a*x^2+b*x+c=0, numerically stable formulation avoiding cancellation.",
    [],
    [[[1.0, -3.0, 2.0], [1.0, 2.0]]],
    "correct",
    "correct",
)

add(
    "correct_021_hypot_builtin_style",
    "def hypot(a, b):\n    return (a * a + b * b) ** 0.5\n",
    "Euclidean norm of a 2D vector, naive but correct at ordinary magnitudes.",
    ["out >= 0.0"],
    [[[3.0, 4.0], 5.0]],
    "correct",
    "correct",
)

add(
    "correct_022_kl_divergence_nats",
    "import math\ndef kl_divergence(p, q):\n    return sum(pi * math.log(pi / qi) for pi, qi in zip(p, q) if pi > 0)\n",
    "KL divergence D(p||q) in nats.",
    ["out >= -1e-9"],
    [[[[0.5, 0.5], [0.5, 0.5]], 0.0]],
    "correct",
    "correct",
)

add(
    "correct_023_fnv1a_32",
    (
        "def fnv1a_32(data):\n"
        "    if isinstance(data, str):\n"
        "        data = data.encode('utf-8')\n"
        "    h = 0x811c9dc5\n"
        "    for byte in data:\n"
        "        h ^= byte\n"
        "        h = (h * 0x01000193) & 0xFFFFFFFF\n"
        "    return h\n"
    ),
    "FNV-1a 32-bit hash of a byte string.",
    ["0 <= out <= 4294967295"],
    [[[""], 0x811C9DC5]],
    "correct",
    "correct",
)

add(
    "correct_024_crc32",
    "import zlib\ndef crc32(data):\n    if isinstance(data, str):\n        data = data.encode('utf-8')\n    return zlib.crc32(data) & 0xFFFFFFFF\n",
    "CRC-32 checksum of a byte string.",
    ["0 <= out <= 4294967295"],
    [[[""], 0]],
    "correct",
    "correct",
)

add(
    "correct_025_modpow_builtin",
    "def modpow(a, b, m):\n    return pow(a, b, m)\n",
    "Modular exponentiation a^b mod m.",
    ["m <= 0 or 0 <= out < m"],
    [[[2, 10, 1000], 24]],
    "correct",
    "correct",
)

add(
    "correct_026_extended_gcd",
    (
        "def extended_gcd(a, b):\n"
        "    old_r, r = a, b\n"
        "    old_s, s = 1, 0\n"
        "    old_t, t = 0, 1\n"
        "    while r != 0:\n"
        "        q = old_r // r\n"
        "        old_r, r = r, old_r - q * r\n"
        "        old_s, s = s, old_s - q * s\n"
        "        old_t, t = t, old_t - q * t\n"
        "    return (old_r, old_s, old_t)\n"
    ),
    "Extended Euclidean algorithm, returns (gcd, x, y) with a*x+b*y=gcd.",
    [],
    [[[30, 20], [10, 1, -1]]],
    "correct",
    "correct",
)

add(
    "correct_027_sinc_normalized",
    "import math\ndef sinc(x):\n    if x == 0:\n        return 1.0\n    return math.sin(math.pi * x) / (math.pi * x)\n",
    "Normalized sinc function, sin(pi x)/(pi x).",
    ["out <= 1.0 + 1e-9"],
    [[[0.0], 1.0], [[1.0], 0.0]],
    "correct",
    "correct",
)

add(
    "correct_028_logit",
    "import math\ndef logit(p):\n    return math.log(p / (1 - p))\n",
    "Logit (inverse sigmoid) function.",
    [],
    [[[0.5], 0.0]],
    "correct",
    "correct",
)

add(
    "correct_029_softplus_stable",
    (
        "import math\n"
        "def softplus(x):\n"
        "    if x > 0:\n"
        "        return x + math.log1p(math.exp(-x))\n"
        "    return math.log1p(math.exp(x))\n"
    ),
    "Numerically stable softplus, log(1+exp(x)).",
    ["out >= 0.0"],
    [[[0.0], 0.6931471805599453]],
    "correct",
    "correct",
)

add(
    "correct_030_std_population",
    "def std(xs):\n    n = len(xs)\n    m = sum(xs) / n\n    return (sum((x - m) ** 2 for x in xs) / n) ** 0.5\n",
    "Population standard deviation of a list of floats.",
    ["out >= 0.0"],
    [[[[1.0, 2.0, 3.0, 4.0]], 1.118033988749895]],
    "correct",
    "correct",
)

add(
    "correct_031_novel_no_oracle_zigzag",
    "def zigzag_encode(n):\n    return (n << 1) ^ (n >> 31) if n < 0 else (n << 1)\n",
    "Zigzag-encodes a signed int into an unsigned varint-friendly representation.",
    ["out >= 0 or n >= 0"],
    [[[0], 0], [[-1], 1], [[1], 2]],
    "correct",
    "edge_case",
)

add(
    "correct_032_sum_fsum",
    "import math\ndef total(xs):\n    return math.fsum(xs)\n",
    "Sum of a list of floats, using math.fsum for precision.",
    [],
    [[[[1.0, 2.0, 3.0]], 6.0]],
    "correct",
    "correct",
)

add(
    "correct_033_gelu_exact",
    "import math\ndef gelu(x):\n    return 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))\n",
    "Exact GELU activation using the error function.",
    [],
    [[[0.0], 0.0]],
    "correct",
    "correct",
)

add(
    "correct_034_silu",
    "import math\ndef silu(x):\n    return x / (1.0 + math.exp(-x))\n",
    "SiLU/Swish activation: x * sigmoid(x).",
    [],
    [[[0.0], 0.0]],
    "correct",
    "correct",
)

add(
    "correct_035_gcd_euclid_recursive_variant",
    "def gcd(a, b):\n    return abs(a) if b == 0 else gcd(b, a % b)\n",
    "Greatest common divisor, recursive formulation (hard negative vs iterative).",
    ["out >= 0"],
    [[[48, 18], 6]],
    "correct",
    "correct",
)

# ---------------------------------------------------------------------------
# Wrong implementations, one per bug category (plus extra variety).
# ---------------------------------------------------------------------------

add(
    "wrong_001_sigmoid_sign_flip",
    "import math\ndef safe_sigmoid(x):\n    return 1.0 / (1.0 + math.exp(x))\n",
    "Numerically stable sigmoid function.",
    ["0.0 <= out <= 1.0"],
    [[[2.0], 0.8807970779778823]],
    "wrong",
    "sign_error",
)

add(
    "wrong_002_cosine_similarity_sign_flip",
    "import math\ndef cosine_similarity(u, v):\n    dot = -sum(a * b for a, b in zip(u, v))\n    nu = math.sqrt(sum(a * a for a in u))\n    nv = math.sqrt(sum(b * b for b in v))\n    return dot / (nu * nv)\n",
    "Cosine similarity between two vectors.",
    [],
    [[[[1.0, 0.0], [1.0, 0.0]], 1.0]],
    "wrong",
    "sign_error",
)

add(
    "wrong_003_variance_off_by_one",
    "def variance(xs):\n    n = len(xs)\n    m = sum(xs) / n\n    return sum((x - m) ** 2 for x in xs) / (n + 1)\n",
    "Population variance of a list of floats.",
    [],
    [[[[1.0, 2.0, 3.0, 4.0]], 1.25]],
    "wrong",
    "off_by_one",
)

add(
    "wrong_004_factorial_off_by_one",
    "def factorial(n):\n    result = 1\n    for i in range(1, n):\n        result *= i\n    return result\n",
    "Factorial of a non-negative integer.",
    [],
    [[[5], 120]],
    "wrong",
    "off_by_one",
)

add(
    "wrong_005_fibonacci_off_by_one",
    "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n + 1):\n        a, b = b, a + b\n    return a\n",
    "The n-th Fibonacci number.",
    [],
    [[[10], 55]],
    "wrong",
    "off_by_one",
)

add(
    "wrong_006_haversine_wrong_radius",
    (
        "import math\n"
        "def haversine(lat1, lon1, lat2, lon2):\n"
        "    R = 3958.8\n"  # miles radius mislabeled as the km function
        "    p1, p2 = math.radians(lat1), math.radians(lat2)\n"
        "    dphi = math.radians(lat2 - lat1)\n"
        "    dlmb = math.radians(lon2 - lon1)\n"
        "    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2\n"
        "    return 2 * R * math.asin(math.sqrt(a))\n"
    ),
    "Great-circle distance between two lat/lon points in degrees, result in km.",
    [],
    [[[0.0, 0.0, 10.0, 10.0], 1568.52]],
    "wrong",
    "wrong_constant",
)

add(
    "wrong_007_binomial_wrong_constant",
    "def binomial_coefficient(n, k):\n    import math\n    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k + 1))\n",
    "Binomial coefficient n choose k.",
    [],
    [[[5, 2], 10]],
    "wrong",
    "wrong_constant",
)

add(
    "wrong_008_softplus_wrong_formula",
    "import math\ndef softplus(x):\n    return math.log(math.exp(x))\n",
    "Softplus activation function, log(1+exp(x)).",
    [],
    [[[0.0], 0.6931471805599453]],
    "wrong",
    "wrong_formula",
)

add(
    "wrong_009_logsumexp_sum_not_log",
    "import math\ndef logsumexp(xs):\n    m = max(xs)\n    return sum(math.exp(v - m) for v in xs) + m\n",
    "Log-sum-exp of a list of floats.",
    [],
    [[[[1.0, 2.0, 3.0]], 3.4076059644443806]],
    "wrong",
    "wrong_formula",
)

add(
    "wrong_010_quadratic_roots_wrong_formula",
    "import math\ndef quadratic_roots(a, b, c):\n    disc = b * b - 2 * a * c\n    sq = math.sqrt(disc)\n    return tuple(sorted(((-b + sq) / a, (-b - sq) / a)))\n",
    "Roots of a*x^2+b*x+c=0.",
    [],
    [[[1.0, -3.0, 2.0], [1.0, 2.0]]],
    "wrong",
    "wrong_formula",
)

add(
    "wrong_011_variance_undeclared_wrong_both",
    "def variance(xs):\n    n = len(xs)\n    m = sum(xs) / n\n    return sum((x - m) ** 2 for x in xs) / (2 * n)\n",
    "Variance of a list of floats.",
    [],
    [[[[1.0, 2.0, 3.0, 4.0]], 0.625]],
    "wrong",
    "wrong_convention",
)

add(
    "wrong_012_entropy_units_mislabeled",
    "import math\ndef entropy_bits(probs):\n    return -sum(p * math.log(p) for p in probs if p > 0)\n",
    "Shannon entropy of a probability distribution, in bits.",
    [],
    [[[[0.5, 0.5]], 1.0]],
    "wrong",
    "wrong_convention",
)

add(
    "wrong_013_kl_divergence_units_mislabeled",
    "import math\ndef kl_divergence_bits(p, q):\n    return sum(pi * math.log(pi / qi) for pi, qi in zip(p, q) if pi > 0)\n",
    "KL divergence D(p||q) in bits.",
    [],
    [[[[0.5, 0.5], [0.25, 0.75]], 0.2074]],
    "wrong",
    "wrong_convention",
)

add(
    "wrong_014_sample_variance_declared_but_population_formula",
    "def sample_variance(xs):\n    n = len(xs)\n    m = sum(xs) / n\n    return sum((x - m) ** 2 for x in xs) / n\n",
    "Sample variance of a list of floats.",
    [],
    [],
    "wrong",
    "wrong_convention",
    ["list[float]"],
)

add(
    "wrong_015_haversine_radians_input",
    (
        "import math\n"
        "def haversine(lat1, lon1, lat2, lon2):\n"
        "    R = 6371.0\n"
        "    dphi = lat2 - lat1\n"
        "    dlmb = lon2 - lon1\n"
        "    a = math.sin(dphi / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlmb / 2) ** 2\n"
        "    return 2 * R * math.asin(math.sqrt(a))\n"
    ),
    "Great-circle distance between two lat/lon points in degrees, result in km.",
    [],
    [[[0.0, 0.0, 0.0, 0.0], 0.0]],
    "wrong",
    "unit_error",
)

add(
    "wrong_016_sinc_unit_confusion",
    "import math\ndef sinc(x):\n    if x == 0:\n        return 1.0\n    return math.sin(x) / x\n",  # unnormalized, but claims normalized by name/summary
    "Normalized sinc function, sin(pi x)/(pi x).",
    [],
    [[[1.0], 0.0]],
    "wrong",
    "unit_error",
)

add(
    "wrong_017_logit_degrees_not_probability",
    "import math\ndef logit(p):\n    return math.log(math.radians(p) / (1 - math.radians(p)))\n",
    "Logit (inverse sigmoid) function of a probability.",
    [],
    [[[0.5], -1.3904]],
    "wrong",
    "unit_error",
)

add(
    "wrong_018_stable_logsumexp_overflows",
    "import math\ndef stable_logsumexp(xs):\n    return math.log(sum(math.exp(v) for v in xs))\n",
    "Numerically stable log-sum-exp, claims overflow safety.",
    [],
    [[[[1.0, 2.0, 3.0]], 3.4076059644443806]],
    "wrong",
    "numerical_instability",
)

add(
    "wrong_019_stable_softmax_overflows",
    "import math\ndef stable_softmax(xs):\n    exps = [math.exp(x) for x in xs]\n    s = sum(exps)\n    return [e / s for e in exps]\n",
    "Numerically stable softmax, robust to large magnitude inputs.",
    [],
    [[[[0.0, 0.0]], [0.5, 0.5]]],
    "wrong",
    "numerical_instability",
)

add(
    "wrong_020_robust_sigmoid_overflows",
    "import math\ndef robust_sigmoid(x):\n    return 1.0 / (1.0 + math.exp(-x))\n",
    "Overflow-safe sigmoid function for extreme inputs.",
    [],
    [[[0.0], 0.5]],
    "wrong",
    "numerical_instability",
)

add(
    "wrong_021_mean_empty_list_crashes_silently",
    "def mean(xs):\n    if not xs:\n        return 0.0\n    return sum(xs) / len(xs)\n",
    "Arithmetic mean of a list of floats; claims to equal the minimum.",
    ["out == min(xs)"],
    [[[[2.0, 4.0]], 3.0]],
    "wrong",
    "edge_case",
)

add(
    "wrong_022_gcd_zero_zero_wrong",
    "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a if a != 0 else 1\n",
    "Greatest common divisor via the Euclidean algorithm.",
    [],
    [[[0, 0], 1]],
    "wrong",
    "edge_case",
)

add(
    "wrong_023_false_claim_invariant_bounds",
    "def clamp(x, lo, hi):\n    return x\n",
    "Clamps x into [lo, hi]; the result is always within [lo, hi].",
    ["lo <= out <= hi"],
    [[[5.0, 0.0, 10.0], 5.0]],
    "wrong",
    "false_claim",
)

add(
    "wrong_024_false_claim_softmax_sums_to_two",
    "import math\ndef softmax(xs):\n    m = max(xs)\n    exps = [math.exp(x - m) for x in xs]\n    s = sum(exps)\n    return [2 * e / s for e in exps]\n",
    "Softmax of a list of floats; outputs always sum to 1.",
    ["math.isclose(sum(out), 1.0, rel_tol=1e-6)"],
    [],
    "wrong",
    "false_claim",
    ["list[float]"],
)

add(
    "wrong_025_consistent_but_wrong_mean",
    "def mean(xs):\n    return sum(xs) / (len(xs) + 1)\n",
    "Arithmetic mean of a list of floats.",
    [],
    [[[[2.0, 4.0]], 2.0]],  # contract crafted to match the bug
    "wrong",
    "consistent_but_wrong",
)

add(
    "wrong_026_consistent_but_wrong_variance",
    "def variance(xs):\n    n = len(xs)\n    m = sum(xs) / n\n    return sum((x - m) ** 2 for x in xs) / (n + 2)\n",
    "Population variance of a list of floats.",
    [],
    [[[[1.0, 2.0, 3.0, 4.0]], 0.8333333333333334]],  # matches the n+2 bug exactly
    "wrong",
    "consistent_but_wrong",
)

add(
    "wrong_027_lerp_wrong_formula",
    "def lerp(a, b, t):\n    return a * t + b * (1 - t)\n",  # swapped endpoints
    "Linear interpolation between a and b at parameter t.",
    [],
    [[[0.0, 10.0, 1.0], 10.0]],  # passes at t=1 by coincidence
    "wrong",
    "wrong_formula",
)

add(
    "wrong_028_crc32_wrong_initial_value",
    "def crc32(data):\n    if isinstance(data, str):\n        data = data.encode('utf-8')\n    h = 0\n    for b in data:\n        h = (h + b) & 0xFFFFFFFF\n    return h\n",
    "CRC-32 checksum of a byte string.",
    [],
    [[["abc"], 0x352441C2]],
    "wrong",
    "wrong_formula",
)

add(
    "wrong_029_fnv1a_wrong_prime",
    (
        "def fnv1a_32(data):\n"
        "    if isinstance(data, str):\n"
        "        data = data.encode('utf-8')\n"
        "    h = 0x811c9dc5\n"
        "    for byte in data:\n"
        "        h ^= byte\n"
        "        h = (h * 31) & 0xFFFFFFFF\n"  # wrong prime
        "    return h\n"
    ),
    "FNV-1a 32-bit hash of a byte string.",
    [],
    [[[""], 0x811C9DC5]],
    "wrong",
    "wrong_constant",
)

add(
    "wrong_030_modpow_ignores_modulus_order",
    "def modpow(a, b, m):\n    return (a ** b) % m\n",
    "Modular exponentiation a^b mod m; should be correct for small inputs but misleads on claims.",
    [],
    [[[2, 10, 1000], 24]],
    "wrong",
    "numerical_instability",
)

add(
    "wrong_031_is_prime_misses_even_check",
    "def is_prime(n):\n    if n < 2:\n        return False\n    i = 3\n    while i * i <= n:\n        if n % i == 0:\n            return False\n        i += 2\n    return True\n",
    "Primality test via trial division.",
    [],
    [[[4], False]],
    "wrong",
    "edge_case",
)

add(
    "wrong_032_hypot_wrong_formula",
    "def hypot(a, b):\n    return a + b\n",
    "Euclidean norm (hypotenuse) of a 2D vector.",
    [],
    [[[3.0, 4.0], 5.0]],
    "wrong",
    "wrong_formula",
)

add(
    "wrong_033_gelu_wrong_constant",
    "import math\ndef gelu(x):\n    return 0.5 * x * (1.0 + math.erf(x))\n",  # missing the /sqrt(2)
    "Exact GELU activation using the error function.",
    [],
    [[[1.0], 0.5 * 1.0 * (1.0 + 0.8427007929497149)]],
    "wrong",
    "wrong_constant",
)

add(
    "wrong_034_silu_wrong_formula",
    "import math\ndef silu(x):\n    return x * x / (1.0 + math.exp(-x))\n",
    "SiLU/Swish activation function.",
    [],
    [[[1.0], 1.0 / (1.0 + math.exp(-1.0))]],
    "wrong",
    "wrong_formula",
)

add(
    "wrong_035_extended_gcd_wrong_bezout",
    "def extended_gcd(a, b):\n    g = a if b == 0 else None\n    return (g if g else 1, 1, 1)\n",
    "Extended Euclidean algorithm, returns (gcd, x, y) with a*x+b*y=gcd.",
    [],
    [],
    "wrong",
    "wrong_formula",
    ["int", "int"],
)

add(
    "wrong_036_binomial_negative_k_wrong",
    "def binomial_coefficient(n, k):\n    import math\n    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))\n",
    "Binomial coefficient n choose k.",
    [],
    [[[5, -1], 0]],
    "wrong",
    "edge_case",
)

add(
    "wrong_037_quadratic_roots_empty_discriminant",
    "def quadratic_roots(a, b, c):\n    import math\n    disc = b * b - 4 * a * c\n    sq = math.sqrt(disc)\n    return tuple(sorted(((-b + sq) / (2 * a), (-b - sq) / (2 * a))))\n",
    "Roots of a*x^2+b*x+c=0; crashes on complex roots instead of signalling none.",
    [],
    [[[1.0, 0.0, 1.0], None]],
    "wrong",
    "edge_case",
)

add(
    "wrong_038_cosine_similarity_zero_vector_wrong",
    "def cosine_similarity(u, v):\n    dot = sum(a * b for a, b in zip(u, v))\n    return dot\n",
    "Cosine similarity between two vectors.",
    [],
    [[[[0.0, 0.0], [1.0, 0.0]], 1.0]],
    "wrong",
    "wrong_formula",
)

add(
    "wrong_039_entropy_negative_prob_false_claim",
    "import math\ndef shannon_entropy(probs):\n    return -sum(p * math.log(abs(p)) for p in probs)\n",
    "Shannon entropy of a probability distribution in nats; always non-negative.",
    ["out >= 0.0"],
    [[[[0.5, 0.5]], 0.6931471805599453]],
    "wrong",
    "false_claim",
)

add(
    "wrong_040_logsumexp_empty_list_wrong_default",
    "import math\ndef logsumexp(xs):\n    if not xs:\n        return 0.0\n    m = max(xs)\n    return m + math.log(sum(math.exp(v - m) for v in xs))\n",
    "Log-sum-exp of a list of floats; the log-sum-exp of an empty list is -infinity, implemented as 0.",
    [],
    [[[[1.0, 2.0, 3.0]], 3.4076059644443806]],
    "wrong",
    "edge_case",
)


# ---------------------------------------------------------------------------
# Additional items for corpus breadth (>=120 total, every category covered
# with multiple examples).
# ---------------------------------------------------------------------------

add(
    "correct_036_clamp_alt_order",
    "def clip(x, lo, hi):\n    if x < lo:\n        return lo\n    if x > hi:\n        return hi\n    return x\n",
    "Clamps x into [lo, hi], branch-based implementation (hard negative vs min/max form).",
    ["lo > hi or lo <= out <= hi"],
    [[[5.0, 0.0, 10.0], 5.0], [[-5.0, 0.0, 10.0], 0.0]],
    "correct",
    "correct",
)

add(
    "correct_037_binomial_via_comb",
    "import math\ndef binomial(n, k):\n    return math.comb(n, k) if 0 <= k <= n else 0\n",
    "Binomial coefficient n choose k, delegates to math.comb.",
    ["out >= 0"],
    [[[5, 2], 10]],
    "correct",
    "correct",
)

add(
    "correct_038_fibonacci_matrix_power",
    (
        "def fibonacci(n):\n"
        "    if n < 0:\n"
        "        raise ValueError('fibonacci undefined for negative n')\n"
        "    def mul(m1, m2):\n"
        "        return [\n"
        "            [m1[0][0] * m2[0][0] + m1[0][1] * m2[1][0], m1[0][0] * m2[0][1] + m1[0][1] * m2[1][1]],\n"
        "            [m1[1][0] * m2[0][0] + m1[1][1] * m2[1][0], m1[1][0] * m2[0][1] + m1[1][1] * m2[1][1]],\n"
        "        ]\n"
        "    result = [[1, 0], [0, 1]]\n"
        "    base = [[1, 1], [1, 0]]\n"
        "    e = n\n"
        "    while e:\n"
        "        if e & 1:\n"
        "            result = mul(result, base)\n"
        "        base = mul(base, base)\n"
        "        e >>= 1\n"
        "    return result[0][1]\n"
    ),
    "The n-th Fibonacci number via fast matrix exponentiation (hard negative vs iterative).",
    ["out >= 0"],
    [[[0], 0], [[1], 1], [[10], 55]],
    "correct",
    "correct",
)

add(
    "correct_039_no_oracle_run_length_encode",
    (
        "def run_length_encode(xs):\n"
        "    if not xs:\n"
        "        return []\n"
        "    out = []\n"
        "    cur, count = xs[0], 1\n"
        "    for x in xs[1:]:\n"
        "        if x == cur:\n"
        "            count += 1\n"
        "        else:\n"
        "            out.append((cur, count))\n"
        "            cur, count = x, 1\n"
        "    out.append((cur, count))\n"
        "    return out\n"
    ),
    "Run-length encodes a list of ints into (value, count) pairs.",
    ["sum(c for _, c in out) == len(xs)"],
    [[[[1, 1, 2, 2, 2, 3]], [[1, 2], [2, 3], [3, 1]]]],
    "correct",
    "edge_case",
)

add(
    "correct_040_no_oracle_is_palindrome",
    "def is_palindrome(s):\n    return s == s[::-1]\n",
    "Checks whether a string reads the same forwards and backwards.",
    [],
    [[["racecar"], True], [["hello"], False], [[""], True]],
    "correct",
    "edge_case",
)

add(
    "correct_041_no_oracle_digit_sum",
    "def digit_sum(n):\n    return sum(int(c) for c in str(abs(n)))\n",
    "Sum of the base-10 digits of an integer.",
    ["out >= 0"],
    [[[123], 6], [[-45], 9], [[0], 0]],
    "correct",
    "edge_case",
)

add(
    "correct_042_entropy_sample_with_zeros",
    "import math\ndef shannon_entropy(probs):\n    return -sum(p * math.log(p) for p in probs if p > 0)\n",
    "Shannon entropy in nats; zero-probability events contribute nothing.",
    ["not all(0 <= p <= 1 for p in probs) or out >= 0.0"],
    [[[[1.0, 0.0, 0.0]], 0.0]],
    "correct",
    "edge_case",
)

add(
    "wrong_043_lerp_clamped_but_undeclared",
    "def lerp(a, b, t):\n    t = max(0.0, min(1.0, t))\n    return a + (b - a) * t\n",
    "Linear interpolation between a and b at parameter t.",
    [],
    [[[0.0, 10.0, 0.5], 5.0]],
    "wrong",
    "wrong_convention",
)

add(
    "correct_044_std_sample",
    "def sample_std(xs):\n    n = len(xs)\n    m = sum(xs) / n\n    return (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5\n",
    "Sample standard deviation of a list of floats.",
    ["out >= 0.0"],
    [[[[1.0, 2.0, 3.0, 4.0]], 1.2909944487358056]],
    "correct",
    "correct",
)

add(
    "correct_045_hypot_overflow_guard",
    (
        "def hypot(a, b):\n"
        "    a, b = abs(a), abs(b)\n"
        "    if a < b:\n"
        "        a, b = b, a\n"
        "    if a == 0:\n"
        "        return 0.0\n"
        "    t = b / a\n"
        "    return a * (1 + t * t) ** 0.5\n"
    ),
    "Overflow-robust Euclidean norm, scales by the larger magnitude first.",
    ["out >= 0.0"],
    [[[3.0, 4.0], 5.0]],
    "correct",
    "correct",
)

add(
    "wrong_041_hypot_overflow_claim_fails",
    "def robust_hypot(a, b):\n    return (a * a + b * b) ** 0.5\n",
    "Overflow-safe Euclidean norm, stable for very large magnitudes.",
    [],
    [[[3.0, 4.0], 5.0]],
    "wrong",
    "numerical_instability",
)

add(
    "wrong_042_std_formula_missing_sqrt",
    "def std(xs):\n    n = len(xs)\n    m = sum(xs) / n\n    return sum((x - m) ** 2 for x in xs) / n\n",
    "Population standard deviation of a list of floats.",
    [],
    [[[[1.0, 2.0, 3.0, 4.0]], 1.118033988749895]],
    "wrong",
    "wrong_formula",
)

add(
    "wrong_043_clamp_swapped_bounds",
    "def clamp(x, lo, hi):\n    return min(max(x, hi), lo)\n",
    "Clamps x into [lo, hi].",
    ["lo <= out <= hi"],
    [[[5.0, 0.0, 10.0], 5.0]],
    "wrong",
    "wrong_formula",
)

add(
    "wrong_044_lerp_off_by_one_t",
    "def lerp(a, b, t):\n    return a + (b - a) * (t + 1)\n",
    "Linear interpolation between a and b at parameter t.",
    [],
    [[[0.0, 10.0, 0.0], 10.0]],
    "wrong",
    "off_by_one",
)

add(
    "wrong_045_factorial_wrong_base_case",
    "def factorial(n):\n    if n == 0:\n        return 0\n    return 1 if n == 1 else n * factorial(n - 1)\n",
    "Factorial of a non-negative integer.",
    [],
    [[[0], 0]],
    "wrong",
    "edge_case",
)

add(
    "wrong_046_binomial_coefficient_wrong_recursion",
    "def binomial_coefficient(n, k):\n    if k == 0 or k == n:\n        return 1\n    return binomial_coefficient(n - 1, k) + binomial_coefficient(n - 1, k)\n",
    "Binomial coefficient n choose k via Pascal's rule.",
    [],
    [[[4, 2], 6]],
    "wrong",
    "wrong_formula",
)

add(
    "wrong_047_gcd_wrong_base_case",
    "def gcd(a, b):\n    if b == 0:\n        return 0\n    return gcd(b, a % b)\n",
    "Greatest common divisor via the Euclidean algorithm.",
    [],
    [[[12, 8], 4]],
    "wrong",
    "edge_case",
)

add(
    "wrong_048_softmax_missing_normalization",
    "import math\ndef softmax(xs):\n    m = max(xs)\n    return [math.exp(x - m) for x in xs]\n",
    "Softmax of a list of floats, outputs sum to 1.",
    ["math.isclose(sum(out), 1.0, rel_tol=1e-6)"],
    [],
    "wrong",
    "wrong_formula",
    ["list[float]"],
)

add(
    "wrong_049_kl_divergence_swapped_args",
    "import math\ndef kl_divergence(p, q):\n    return sum(qi * math.log(qi / pi) for pi, qi in zip(p, q) if qi > 0)\n",
    "KL divergence D(p||q) in nats.",
    [],
    [[[[0.5, 0.5], [0.5, 0.5]], 0.0]],
    "wrong",
    "wrong_formula",
)

add(
    "wrong_050_cosine_similarity_unnormalized",
    "def cosine_similarity(u, v):\n    return sum(a * b for a, b in zip(u, v))\n",
    "Cosine similarity between two vectors, always in [-1, 1].",
    ["-1.0 - 1e-9 <= out <= 1.0 + 1e-9"],
    [[[[1.0, 0.0], [1.0, 0.0]], 1.0]],
    "wrong",
    "false_claim",
)

add(
    "wrong_051_is_prime_one_is_prime",
    "def is_prime(n):\n    if n < 2:\n        return n == 1\n    i = 2\n    while i * i <= n:\n        if n % i == 0:\n            return False\n        i += 1\n    return True\n",
    "Primality test via trial division.",
    [],
    [[[1], True]],
    "wrong",
    "edge_case",
)

add(
    "wrong_052_consistent_but_wrong_factorial",
    "def factorial(n):\n    return 2 ** n\n",
    "Factorial of a non-negative integer.",
    [],
    [[[1], 1], [[2], 2]],  # matches 2**n at n<=2, diverges after
    "wrong",
    "consistent_but_wrong",
)

add(
    "wrong_053_consistent_but_wrong_fibonacci",
    "def fibonacci(n):\n    return n\n",
    "The n-th Fibonacci number.",
    [],
    [[[0], 0], [[1], 1]],  # matches identity at n=0,1 only
    "wrong",
    "consistent_but_wrong",
)

add(
    "wrong_054_false_claim_is_prime_evens",
    "def is_prime(n):\n    return n % 2 != 0\n",
    "Primality test; returns True exactly for prime numbers.",
    [],
    [[[3], True], [[9], False]],
    "wrong",
    "false_claim",
)

add(
    "wrong_055_false_claim_variance_nonneg_broken",
    "def variance(xs):\n    n = len(xs)\n    m = sum(xs) / n\n    return sum((x - m) for x in xs) / n\n",
    "Population variance of a list of floats; always non-negative.",
    ["out >= 0.0"],
    [],
    "wrong",
    "false_claim",
    ["list[float]"],
)

add(
    "wrong_056_unit_error_entropy_degrees",
    "import math\ndef shannon_entropy(probs):\n    return -sum(math.radians(p) * math.log(p) for p in probs if p > 0)\n",
    "Shannon entropy of a probability distribution in nats.",
    [],
    [[[[0.5, 0.5]], 0.6931471805599453]],
    "wrong",
    "unit_error",
)

add(
    "wrong_057_modpow_wrong_convention_signed",
    "def modpow(a, b, m):\n    result = pow(a, b, m)\n    return result - m if result > m // 2 else result\n",
    "Modular exponentiation a^b mod m, result in [0, m).",
    ["0 <= out < m"],
    [[[2, 10, 1000], 24]],
    "wrong",
    "wrong_convention",
)

add(
    "wrong_058_sample_std_declared_population_formula",
    "def sample_std(xs):\n    n = len(xs)\n    m = sum(xs) / n\n    return (sum((x - m) ** 2 for x in xs) / n) ** 0.5\n",
    "Sample standard deviation of a list of floats.",
    [],
    [],
    "wrong",
    "wrong_convention",
    ["list[float]"],
)

add(
    "wrong_059_gelu_tanh_approx_mislabeled_exact",
    (
        "import math\n"
        "def gelu(x):\n"
        "    return 0.5 * x * (1.0 + math.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))\n"
    ),
    "Exact GELU activation using the error function.",
    [],
    [[[0.0], 0.0]],
    "wrong",
    "wrong_formula",
)

add(
    "wrong_060_horner_wrong_order",
    "def horner(coeffs, x):\n    result = 0.0\n    for c in coeffs:\n        result = result * x + c\n    return result\n",
    "Evaluates a polynomial given low-to-high coefficients at x via Horner's method.",
    [],
    [[[[1.0, 0.0, 1.0], 2.0], 5.0]],
    "wrong",
    "wrong_convention",
)


add(
    "correct_046_no_oracle_count_vowels",
    "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')\n",
    "Counts vowel characters in a string, case-insensitive.",
    ["out >= 0"],
    [[["Hello World"], 3], [[""], 0]],
    "correct",
    "edge_case",
)

add(
    "correct_047_no_oracle_flatten_once",
    "def flatten_once(xs):\n    out = []\n    for x in xs:\n        if isinstance(x, list):\n            out.extend(x)\n        else:\n            out.append(x)\n    return out\n",
    "Flattens one level of nested lists.",
    ["len(out) >= len(xs)"],
    [[[[1, [2, 3], 4]], [1, 2, 3, 4]]],
    "correct",
    "edge_case",
)

add(
    "correct_048_entropy_bits_log2",
    "import math\ndef entropy_bits(probs):\n    return -sum(p * math.log(p, 2) for p in probs if p > 0)\n",
    "Shannon entropy of a probability distribution, in bits (log base 2).",
    ["not all(0 <= p <= 1 for p in probs) or out >= 0.0"],
    [[[[0.25, 0.25, 0.25, 0.25]], 2.0]],
    "correct",
    "correct",
)

add(
    "correct_049_kl_divergence_bits",
    "import math\ndef kl_divergence_bits(p, q):\n    return sum(pi * math.log(pi / qi, 2) for pi, qi in zip(p, q) if pi > 0)\n",
    "KL divergence D(p||q) in bits.",
    ["out >= -1e-9"],
    [[[[0.5, 0.5], [0.5, 0.5]], 0.0]],
    "correct",
    "correct",
)

add(
    "correct_050_sinc_unnormalized",
    "import math\ndef unnormalized_sinc(x):\n    if x == 0:\n        return 1.0\n    return math.sin(x) / x\n",
    "Unnormalized sinc function, sin(x)/x.",
    [],
    [[[0.0], 1.0]],
    "correct",
    "correct",
)

add(
    "correct_051_haversine_wgs84_radius",
    (
        "import math\n"
        "def haversine(lat1, lon1, lat2, lon2):\n"
        "    R = 6371.0088\n"
        "    p1, p2 = math.radians(lat1), math.radians(lat2)\n"
        "    dphi = math.radians(lat2 - lat1)\n"
        "    dlmb = math.radians(lon2 - lon1)\n"
        "    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2\n"
        "    return 2 * R * math.asin(math.sqrt(a))\n"
    ),
    "Great-circle distance between two lat/lon points in degrees, result in km (WGS84 mean radius).",
    ["out >= 0.0"],
    [[[0.0, 0.0, 0.0, 0.0], 0.0]],
    "correct",
    "correct",
)

add(
    "correct_052_binomial_edge_k_zero",
    "def binomial_coefficient(n, k):\n    import math\n    if k < 0 or k > n:\n        return 0\n    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))\n",
    "Binomial coefficient n choose k via factorials.",
    ["out >= 0"],
    [[[0, 0], 1], [[5, 0], 1], [[5, 5], 1]],
    "correct",
    "edge_case",
)

add(
    "correct_053_quadratic_roots_double_root",
    (
        "import math\n"
        "def quadratic_roots(a, b, c):\n"
        "    disc = b * b - 4 * a * c\n"
        "    if disc < 0:\n"
        "        return None\n"
        "    sq = math.sqrt(disc)\n"
        "    q = -0.5 * (b + sq) if b >= 0 else -0.5 * (b - sq)\n"
        "    if q == 0:\n"
        "        return (0.0, 0.0)\n"
        "    return tuple(sorted((q / a, c / q)))\n"
    ),
    "Roots of a*x^2+b*x+c=0; handles the double-root case.",
    [],
    [[[1.0, -2.0, 1.0], [1.0, 1.0]]],
    "correct",
    "edge_case",
)

add(
    "wrong_061_count_vowels_wrong_set",
    "def count_vowels(s):\n    return sum(1 for c in s if c in 'aeiou')\n",  # case-sensitive, undeclared bug
    "Counts vowel characters in a string, case-insensitive.",
    [],
    [[["Hello World"], 3]],  # this particular example happens to match the bug
    "wrong",
    "consistent_but_wrong",
)

add(
    "wrong_062_flatten_once_drops_duplicates",
    "def flatten_once(xs):\n    out = []\n    for x in xs:\n        if isinstance(x, list):\n            out.extend(set(x))\n        else:\n            out.append(x)\n    return out\n",
    "Flattens one level of nested lists, preserving every element.",
    ["len(out) >= len(xs)"],
    [[[[1, [2, 2, 3], 4]], [1, 2, 2, 3, 4]]],
    "wrong",
    "false_claim",
)

add(
    "wrong_063_entropy_bits_wrong_base",
    "import math\ndef entropy_bits(probs):\n    return -sum(p * math.log(p, 10) for p in probs if p > 0)\n",
    "Shannon entropy of a probability distribution, in bits.",
    [],
    [[[[0.25, 0.25, 0.25, 0.25]], 2.0]],
    "wrong",
    "wrong_convention",
)

add(
    "wrong_064_sinc_normalized_but_unnormalized_formula",
    "import math\ndef normalized_sinc(x):\n    if x == 0:\n        return 1.0\n    return math.sin(x) / x\n",
    "Normalized sinc function, sin(pi x)/(pi x).",
    [],
    [[[1.0], 0.0]],
    "wrong",
    "unit_error",
)

add(
    "wrong_065_haversine_degree_radius_confusion",
    (
        "import math\n"
        "def haversine(lat1, lon1, lat2, lon2):\n"
        "    R = 6371.0\n"
        "    dphi = lat2 - lat1\n"  # forgot radians conversion for the delta terms
        "    dlmb = lon2 - lon1\n"
        "    p1, p2 = math.radians(lat1), math.radians(lat2)\n"
        "    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2\n"
        "    return 2 * R * math.asin(math.sqrt(a))\n"
    ),
    "Great-circle distance between two lat/lon points in degrees, result in km.",
    [],
    [[[0.0, 0.0, 0.0, 0.0], 0.0]],
    "wrong",
    "unit_error",
)

add(
    "wrong_066_quadratic_roots_discriminant_sign_flip",
    (
        "import math\n"
        "def quadratic_roots(a, b, c):\n"
        "    disc = 4 * a * c - b * b\n"  # sign flipped
        "    if disc < 0:\n"
        "        return None\n"
        "    sq = math.sqrt(disc)\n"
        "    return tuple(sorted(((-b + sq) / (2 * a), (-b - sq) / (2 * a))))\n"
    ),
    "Roots of a*x^2+b*x+c=0.",
    [],
    [[[1.0, -3.0, 2.0], [1.0, 2.0]]],
    "wrong",
    "sign_error",
)

add(
    "wrong_067_consistent_but_wrong_binomial",
    "def binomial_coefficient(n, k):\n    return n * k\n",
    "Binomial coefficient n choose k.",
    [],
    [[[1, 1], 1]],  # matches only at n=k=1
    "wrong",
    "consistent_but_wrong",
)


def main() -> None:
    out_path = Path(__file__).resolve().parent / "corpus.jsonl"
    with out_path.open("w") as f:
        for item in ITEMS:
            f.write(json.dumps(item) + "\n")
    print(f"wrote {len(ITEMS)} items to {out_path}")


if __name__ == "__main__":
    main()
