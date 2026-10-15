"""Unit tests for kardashev_cli.correctness.

Covers the three real false-positives from the bug report (stable_logsumexp
returning max(xs), a sign-flipped stable_sigmoid, mean dividing by len+1),
plus the sandbox boundary, the AST allowlist, and the counterexample
validator.
"""

from __future__ import annotations

import math

import pytest

from kardashev_cli.correctness import Concept, validate_counterexample, verify
from kardashev_cli.correctness.ast_safety import InvariantSyntaxError, validate_invariant_ast
from kardashev_cli.correctness.oracles import canonicalize_name


def test_wrong_logsumexp_is_refuted():
    concept = Concept(
        source_code="def stable_logsumexp(xs):\n    return max(xs)\n",
        func_name="stable_logsumexp",
        param_types=("list[float]",),
        return_type="float",
        summary="Numerically stable log-sum-exp.",
        invariants=["out >= max(xs)"],
        contract=[(([1.0, 2.0, 3.0],), 3.408605001)],
    )
    verdict = verify(concept)
    assert verdict.status == "refuted"
    assert verdict.counterexample is not None


def test_sign_flipped_sigmoid_is_refuted():
    concept = Concept(
        source_code="import math\ndef stable_sigmoid(x):\n    return 1.0 / (1.0 + math.exp(x))\n",
        func_name="stable_sigmoid",
        param_types=("float",),
        return_type="float",
        summary="Numerically stable sigmoid / logistic function.",
        invariants=["0.0 <= out <= 1.0"],
        contract=[((2.0,), 1 / (1 + math.exp(-2.0)))],
    )
    verdict = verify(concept)
    assert verdict.status == "refuted"


def test_mean_off_by_one_is_refuted():
    concept = Concept(
        source_code="def mean(xs):\n    return sum(xs) / (len(xs) + 1)\n",
        func_name="mean",
        param_types=("list[float]",),
        return_type="float",
        summary="Arithmetic mean of a list of floats.",
        invariants=["min(xs) <= out <= max(xs)"],
        contract=[(([2.0, 4.0],), 3.0)],
    )
    verdict = verify(concept)
    assert verdict.status == "refuted"


def test_correct_mean_is_verified():
    concept = Concept(
        source_code="import math\ndef mean(xs):\n    return math.fsum(xs) / len(xs)\n",
        func_name="mean",
        param_types=("list[float]",),
        return_type="float",
        summary="Arithmetic mean of a list of floats.",
        invariants=["min(xs) <= out <= max(xs)"],
        contract=[(([2.0, 4.0],), 3.0), (([1.0, 1.0, 1.0],), 1.0), (([0.0],), 0.0)],
    )
    verdict = verify(concept)
    assert verdict.status == "verified"


def test_consistent_but_wrong_contract_never_verifies():
    """A contract crafted to match a bug must not be enough to verify against an oracle."""
    concept = Concept(
        source_code="def mean(xs):\n    return sum(xs) / (len(xs) + 1)\n",
        func_name="mean",
        param_types=("list[float]",),
        return_type="float",
        summary="Arithmetic mean of a list of floats.",
        invariants=[],
        contract=[(([2.0, 4.0],), 2.0)],
    )
    verdict = verify(concept)
    assert verdict.status != "verified"


def test_novel_function_without_oracle_and_thin_evidence_is_unverifiable():
    concept = Concept(
        source_code="def zigzag_pack(n):\n    return (n << 1) ^ (n >> 31) if n < 0 else (n << 1)\n",
        func_name="zigzag_pack",
        param_types=("int",),
        return_type="int",
        summary="Zigzag-encodes a signed int into an unsigned representation.",
        invariants=[],
        contract=[((3,), 6)],
    )
    verdict = verify(concept)
    assert verdict.status == "unverifiable"


def test_undeclared_variance_matches_either_variant():
    concept = Concept(
        source_code=(
            "def variance(xs):\n"
            "    n = len(xs)\n"
            "    m = sum(xs) / n\n"
            "    return sum((x - m) ** 2 for x in xs) / (n - 1)\n"
        ),
        func_name="variance",
        param_types=("list[float]",),
        return_type="float",
        summary="Variance of a list of floats.",  # does not say sample or population
        invariants=[],
        contract=[(([1.0, 2.0, 3.0, 4.0],), 5 / 3)],
    )
    verdict = verify(concept)
    assert verdict.status == "verified"


def test_declared_sample_variance_rejects_population_formula():
    concept = Concept(
        source_code=(
            "def sample_variance(xs):\n"
            "    n = len(xs)\n"
            "    m = sum(xs) / n\n"
            "    return sum((x - m) ** 2 for x in xs) / n\n"  # uses population denominator
        ),
        func_name="sample_variance",
        param_types=("list[float]",),
        return_type="float",
        summary="Sample variance of a list of floats.",
        invariants=[],
        contract=[],
    )
    verdict = verify(concept)
    assert verdict.status == "refuted"


def test_free_text_invariant_is_recorded_as_unchecked_not_passed():
    concept = Concept(
        source_code="def clamp(x, lo, hi):\n    return min(max(x, lo), hi)\n",
        func_name="clamp",
        param_types=("float", "float", "float"),
        return_type="float",
        summary="Clamps x into [lo, hi].",
        invariants=["result should feel reasonable to a human reviewer"],
        contract=[],
    )
    verdict = verify(concept)
    unchecked = [c for c in verdict.checks if c.name.startswith("invariant:") and "unchecked" in c.detail]
    assert len(unchecked) == 1
    assert unchecked[0].passed is True  # recorded, but never counts as a passing check


def test_submitted_code_never_runs_in_calling_process():
    """A concept that pollutes module globals must not affect this process."""
    sentinel = "__opencommons_should_not_exist__"
    concept = Concept(
        source_code=(
            f"import builtins\nbuiltins.{sentinel} = True\n\ndef identity(x):\n    return x\n"
        ),
        func_name="identity",
        param_types=("float",),
        return_type="float",
        summary="Returns its argument unchanged.",
        invariants=[],
        contract=[((1.0,), 1.0)],
    )
    verify(concept)
    import builtins

    assert not hasattr(builtins, sentinel)


def test_ast_allowlist_rejects_dunder_and_imports():
    with pytest.raises(InvariantSyntaxError):
        validate_invariant_ast("out.__class__", {"out"})
    with pytest.raises(InvariantSyntaxError):
        validate_invariant_ast("__import__('os').system('echo hi')", {"out"})
    with pytest.raises(InvariantSyntaxError):
        validate_invariant_ast("os.system('echo hi')", {"out", "os"})


def test_ast_allowlist_accepts_math_and_safe_builtins():
    validate_invariant_ast("math.isclose(out, 1.0) or abs(out) < 1e-9", {"out", "math"})


def test_canonicalize_name_strips_qualifiers():
    assert canonicalize_name("stable_logsumexp") == "logsumexp"
    assert canonicalize_name("safe_sigmoid") == "sigmoid"
    assert canonicalize_name("numerically_stable_softmax") == "softmax"
    assert canonicalize_name("mean") == "mean"


def test_validate_counterexample_rejects_type_invalid_args():
    concept = Concept(
        source_code="def mean(xs):\n    return sum(xs) / len(xs)\n",
        func_name="mean",
        param_types=("list[float]",),
        return_type="float",
        summary="Mean of a list.",
        invariants=[],
        contract=[],
    )
    valid, reason = validate_counterexample(concept, ("not a list",), 1.0)
    assert valid is False
    assert "type" in reason


def test_validate_counterexample_rejects_mere_exception():
    concept = Concept(
        source_code="def mean(xs):\n    return sum(xs) / len(xs)\n",
        func_name="mean",
        param_types=("list[float]",),
        return_type="float",
        summary="Mean of a list.",
        invariants=[],
        contract=[],
    )
    valid, reason = validate_counterexample(concept, ([],), 0.0)
    assert valid is False
    assert "raised" in reason


def test_validate_counterexample_accepts_real_wrong_output():
    concept = Concept(
        source_code="def mean(xs):\n    return sum(xs) / (len(xs) + 1)\n",
        func_name="mean",
        param_types=("list[float]",),
        return_type="float",
        summary="Mean of a list.",
        invariants=[],
        contract=[],
    )
    valid, _reason = validate_counterexample(concept, ([2.0, 4.0],), 3.0)
    assert valid is True


def test_determinism_check_catches_nondeterministic_concept():
    concept = Concept(
        source_code=(
            "import random\n"
            "def noisy_identity(x):\n"
            "    return x + random.random() * 1e-3\n"
        ),
        func_name="noisy_identity",
        param_types=("float",),
        return_type="float",
        summary="Returns its argument.",
        invariants=[],
        contract=[],
    )
    verdict = verify(concept)
    assert verdict.status == "refuted"
    determinism = next(c for c in verdict.checks if c.name == "determinism")
    assert determinism.passed is False
