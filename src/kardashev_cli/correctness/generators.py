"""Deterministic, property-based input generators, keyed by declared param type.

Every generator mixes explicit edge cases (0, +/-1, tiny, huge, empty,
singleton, long) with a bounded amount of pseudo-random sampling from a
fixed seed, so repeated runs (and the determinism check) see the same
inputs.
"""

from __future__ import annotations

import random

_EDGE_FLOATS = [0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 1e-9, -1e-9, 1e3, -1e3, 1e6, -1e6]
_EDGE_INTS = [0, 1, -1, 2, -2, 10, -10, 100]


def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def gen_float(n: int, seed: int = 0) -> list[float]:
    rng = _rng(seed)
    out = list(_EDGE_FLOATS)
    while len(out) < n:
        out.append(rng.uniform(-50.0, 50.0))
    return out[:n]


def gen_int(n: int, seed: int = 0) -> list[int]:
    rng = _rng(seed)
    out = list(_EDGE_INTS)
    while len(out) < n:
        out.append(rng.randint(-200, 200))
    return out[:n]


def gen_bool(n: int, seed: int = 0) -> list[bool]:
    vals = [True, False]
    return [vals[i % 2] for i in range(n)]


def gen_str(n: int, seed: int = 0) -> list[str]:
    rng = _rng(seed)
    out = ["", "a", "ab", "hello world", "x" * 32]
    alphabet = "abcdefghijklmnopqrstuvwxyz "
    while len(out) < n:
        out.append("".join(rng.choice(alphabet) for _ in range(rng.randint(0, 12))))
    return out[:n]


def gen_list_float(n: int, seed: int = 0) -> list[list[float]]:
    rng = _rng(seed)
    out: list[list[float]] = [[], [0.0], [1.0], [1.0, 2.0, 3.0], [-1.0, 1.0]]
    while len(out) < n:
        size = rng.randint(1, 20)
        out.append([rng.uniform(-30.0, 30.0) for _ in range(size)])
    return out[:n]


def gen_list_int(n: int, seed: int = 0) -> list[list[int]]:
    rng = _rng(seed)
    out: list[list[int]] = [[], [0], [1], [1, 2, 3], [-1, 1]]
    while len(out) < n:
        size = rng.randint(1, 20)
        out.append([rng.randint(-50, 50) for _ in range(size)])
    return out[:n]


_GENERATORS = {
    "float": gen_float,
    "int": gen_int,
    "bool": gen_bool,
    "str": gen_str,
    "list[float]": gen_list_float,
    "list[int]": gen_list_int,
}


def values_for_type(type_str: str, n: int, seed: int = 0) -> list:
    """Return ``n`` generated values for a single declared type string."""
    gen = _GENERATORS.get(type_str)
    if gen is None:
        # Unknown/compound type: fall back to float, which is harmless and
        # keeps generation total rather than raising on an unseen annotation.
        gen = gen_float
    return gen(n, seed=seed)


def generate_inputs(param_types: tuple[str, ...], n: int = 25, seed: int = 0) -> list[tuple]:
    """Generate ``n`` argument tuples across all declared parameter types.

    Each parameter is sampled independently with its own stream (distinct
    seeds), then zipped positionally, so edge cases for different
    parameters still get combined across the generated set.
    """
    if not param_types:
        return [()] * n
    per_param = [values_for_type(t, n, seed=seed + i) for i, t in enumerate(param_types)]
    return [tuple(col[i] for col in per_param) for i in range(n)]


def extreme_inputs(param_types: tuple[str, ...], n: int = 10) -> list[tuple]:
    """Generate inputs biased toward extreme magnitudes, for the robustness check."""
    extreme_scalars = [1e3, -1e3, 1e300, -1e300, 1e-300, -1e-300]
    out = []
    for i in range(n):
        row = []
        for t in param_types:
            if t == "float":
                row.append(extreme_scalars[i % len(extreme_scalars)])
            elif t == "int":
                row.append(int(extreme_scalars[i % len(extreme_scalars)]) if abs(extreme_scalars[i % len(extreme_scalars)]) < 1e15 else 10**15)
            elif t == "list[float]":
                row.append([extreme_scalars[i % len(extreme_scalars)]] * 5)
            elif t == "list[int]":
                row.append([1, 2, 3])
            else:
                row.append(values_for_type(t, 1, seed=i)[0])
        out.append(tuple(row))
    return out
