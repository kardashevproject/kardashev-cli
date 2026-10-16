"""Orchestrates the deterministic checks into a single :class:`Verdict`."""

from __future__ import annotations

from .checks import (
    MIN_CONTRACT_FOR_VERIFIED,
    MIN_GENERATED_FOR_VERIFIED,
    check_contract,
    check_determinism,
    check_invariants,
    check_oracle,
    check_robustness,
)
from .executor import subprocess_executor
from .oracles import DEFAULT_ORACLES
from .types import CheckResult, Concept, Executor, Verdict

__all__ = ["DEFAULT_ORACLES", "validate_counterexample", "verify"]

_DEFAULT_TIMEOUT_S = 5.0


def verify(concept: Concept, executor: Executor | None = None, oracles=DEFAULT_ORACLES) -> Verdict:
    """Run every deterministic check against ``concept`` and return a :class:`Verdict`.

    ``executor`` defaults to the sandboxed subprocess runner; a caller with
    its own sandbox (e.g. the OpenCommons L0-L8 pipeline) may pass any
    callable matching the :data:`kardashev_cli.correctness.types.Executor`
    signature instead. ``oracles`` is accepted for interface compatibility
    with alternate registries; the module-level registry in
    :mod:`kardashev_cli.correctness.oracles` is used for lookups either way
    unless a caller has monkeypatched it.
    """
    run_executor = executor or subprocess_executor
    checks: list[CheckResult] = []
    pitfalls: list[str] = []
    counterexample: str | None = None
    refuted = False

    contract_result = check_contract(concept, run_executor, _DEFAULT_TIMEOUT_S)
    checks.append(contract_result)
    contract_examples_passed = 0
    if concept.contract:
        if contract_result.passed:
            contract_examples_passed = len(concept.contract)
        else:
            refuted = True
            counterexample = contract_result.detail

    invariant_results, invariant_outcomes = check_invariants(concept, run_executor, _DEFAULT_TIMEOUT_S)
    checks.extend(invariant_results)
    any_invariant_failed = any(not r.passed for r in invariant_results)
    if any_invariant_failed and not refuted:
        refuted = True
        failing = next(r for r in invariant_results if not r.passed)
        counterexample = failing.detail
    executable_invariant_passed_enough = any(
        o.checked and o.failed_count == 0 and o.passed_count >= MIN_GENERATED_FOR_VERIFIED for o in invariant_outcomes
    )

    oracle_result = check_oracle(concept, run_executor, _DEFAULT_TIMEOUT_S)
    oracle_matched_and_agreed = False
    if oracle_result is not None:
        checks.append(oracle_result)
        if oracle_result.passed:
            oracle_matched_and_agreed = True
        elif not refuted:
            refuted = True
            counterexample = oracle_result.detail

    robustness_result, robustness_pitfalls = check_robustness(concept, run_executor, _DEFAULT_TIMEOUT_S)
    checks.append(robustness_result)
    pitfalls.extend(robustness_pitfalls)
    if not robustness_result.passed and not refuted:
        refuted = True
        counterexample = robustness_result.detail

    determinism_result = check_determinism(concept, run_executor, _DEFAULT_TIMEOUT_S)
    checks.append(determinism_result)
    if not determinism_result.passed and not refuted:
        refuted = True
        counterexample = determinism_result.detail

    if refuted:
        status = "refuted"
    elif oracle_matched_and_agreed or contract_examples_passed >= MIN_CONTRACT_FOR_VERIFIED and executable_invariant_passed_enough:
        status = "verified"
    else:
        status = "unverifiable"

    return Verdict(status=status, checks=checks, counterexample=counterexample, pitfalls=pitfalls)


def validate_counterexample(concept: Concept, args: tuple, expected, executor: Executor | None = None) -> tuple[bool, str]:
    """Check that a reported counterexample is a real refutation, not an invalid-input artifact.

    Returns ``(valid, reason)``. A counterexample is only valid when:
    1. ``args`` satisfies the concept's declared ``param_types`` (a string
       passed where ``list[float]`` is declared is a type error, not a
       refutation).
    2. The submitted function actually returns a *wrong* value compared to
       ``expected`` -- an exception on an out-of-contract input does not
       count; only a concrete wrong output does.
    """
    from .compare import approx_eq

    if len(args) != len(concept.param_types):
        return False, "argument count does not match the declared signature"

    for value, declared_type in zip(args, concept.param_types):
        if not _matches_type(value, declared_type):
            return False, f"argument {value!r} does not satisfy declared type '{declared_type}'"

    run_executor = executor or subprocess_executor
    outcomes = run_executor(concept.source_code, concept.func_name, [args], _DEFAULT_TIMEOUT_S)
    outcome = outcomes[0]
    if not outcome.ok:
        return False, f"function raised ({outcome.error_type}) rather than returning a wrong value"
    if approx_eq(outcome.value, expected):
        return False, "function actually agrees with the expected value; not a refutation"
    return True, f"{concept.func_name}{args} -> {outcome.value!r}, expected {expected!r}"


def _matches_type(value, declared_type: str) -> bool:
    if declared_type == "float":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if declared_type == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if declared_type == "bool":
        return isinstance(value, bool)
    if declared_type == "str":
        return isinstance(value, str)
    if declared_type == "list":
        return isinstance(value, list)
    if declared_type == "list[float]":
        return isinstance(value, list) and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)
    if declared_type == "list[int]":
        return isinstance(value, list) and all(isinstance(v, int) and not isinstance(v, bool) for v in value)
    return True  # unknown declared type: do not block on it
