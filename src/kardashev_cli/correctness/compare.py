"""Tolerant value comparison shared by every check."""

from __future__ import annotations

import math

REL_TOL = 1e-9
ABS_TOL = 1e-12


def approx_eq(a, b, rel_tol: float = REL_TOL, abs_tol: float = ABS_TOL) -> bool:
    """Elementwise, tolerant equality. NaN == NaN here (unlike IEEE 754)."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        fa, fb = float(a), float(b)
        if math.isnan(fa) and math.isnan(fb):
            return True
        if math.isinf(fa) or math.isinf(fb):
            return fa == fb
        return math.isclose(fa, fb, rel_tol=rel_tol, abs_tol=abs_tol)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(approx_eq(x, y, rel_tol, abs_tol) for x, y in zip(a, b))
    return a == b
