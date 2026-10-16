"""The individual deterministic checks that make up a verification run.

Every check that needs to run submitted code goes through an ``Executor``
(see :mod:`kardashev_cli.correctness.types`); none of them import or exec
``concept.source_code`` directly in this process.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import oracles as oracle_registry
from .ast_safety import InvariantSyntaxError, extract_param_names, validate_invariant_ast
from .compare import approx_eq
from .generators import extreme_inputs, generate_inputs
from .types import CheckResult, Concept, Executor

ROBUSTNESS_CLAIM_WORDS = ("stable", "safe", "overflow", "robust")

MIN_CONTRACT_FOR_VERIFIED = 3
MIN_GENERATED_FOR_VERIFIED = 20
INVARIANT_SAMPLE_SIZE = 30


@dataclass
class InvariantOutcome:
    expr: str
    checked: bool  # False => free-text / unparseable, recorded as "unchecked"
    passed_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    failure_detail: str | None = None


def check_contract(concept: Concept, executor: Executor, timeout_s: float) -> CheckResult:
    if not concept.contract:
        return CheckResult("contract", True, "no contract examples submitted")

    args_list = [args for args, _expected in concept.contract]
    outcomes = executor(concept.source_code, concept.func_name, args_list, timeout_s)

    failures = []
    passed = 0
    for (args, expected), outcome in zip(concept.contract, outcomes):
        if outcome.ok and approx_eq(outcome.value, expected):
            passed += 1
        else:
            failures.append((args, expected, outcome))

    if failures:
        args, expected, outcome = failures[0]
        detail = (
            f"{concept.func_name}{args} -> {outcome.value!r} "
            f"(error={outcome.error_type}), expected {expected!r}"
        )
        return CheckResult("contract", False, f"{len(failures)}/{len(concept.contract)} examples failed: {detail}")

    return CheckResult("contract", True, f"{passed}/{len(concept.contract)} examples passed")


def _invariant_harness_source(concept: Concept, param_names: list[str], expr: str) -> str:
    params_sig = ", ".join(param_names) if param_names else ""
    bind_lines = "\n".join(f"    __ns__['{p}'] = {p}" for p in param_names)
    return (
        concept.source_code.rstrip()
        + "\n\n\n"
        + "def __invariant_check__("
        + params_sig
        + "):\n"
        + "    import math as __math__\n"
        + "    __ns__ = {}\n"
        + (bind_lines if bind_lines else "    pass")
        + "\n"
        + "    try:\n"
        + f"        out = {concept.func_name}({params_sig})\n"
        + "    except Exception:\n"
        + "        return None\n"
        + "    __ns__['out'] = out\n"
        + f"    __ns__['f'] = {concept.func_name}\n"
        + "    __ns__['math'] = __math__\n"
        + "    __ns__['abs'] = abs\n"
        + "    __ns__['min'] = min\n"
        + "    __ns__['max'] = max\n"
        + "    __ns__['len'] = len\n"
        + "    __ns__['sum'] = sum\n"
        + "    __ns__['all'] = all\n"
        + "    __ns__['any'] = any\n"
        + "    __ns__['isclose'] = __math__.isclose\n"
        + "    __ns__['range'] = range\n"
        + "    __ns__['round'] = round\n"
        + "    try:\n"
        + f"        return bool(eval(compile({expr!r}, '<invariant>', 'eval'), {{'__builtins__': {{}}}}, __ns__))\n"
        + "    except Exception:\n"
        + "        return None\n"  # the expression itself is ill-defined at this input: skip, not a violation
        + "\n"
    )


def check_one_invariant(
    concept: Concept, expr: str, executor: Executor, timeout_s: float, *, n: int = INVARIANT_SAMPLE_SIZE
) -> InvariantOutcome:
    try:
        param_names = extract_param_names(concept.source_code, concept.func_name)
    except InvariantSyntaxError:
        param_names = [f"_p{i}" for i in range(len(concept.param_types))]

    allowed = set(param_names) | {"f", "out", "math"}
    try:
        validate_invariant_ast(expr, allowed)
    except InvariantSyntaxError:
        return InvariantOutcome(expr=expr, checked=False)

    harness_source = _invariant_harness_source(concept, param_names, expr)
    args_list = generate_inputs(concept.param_types, n=n)
    outcomes = executor(harness_source, "__invariant_check__", args_list, timeout_s)

    passed = failed = skipped = 0
    failure_args = None
    for args, outcome in zip(args_list, outcomes):
        if not outcome.ok:
            skipped += 1
            continue
        if outcome.value is None:
            skipped += 1
        elif outcome.value is True:
            passed += 1
        else:
            failed += 1
            if failure_args is None:
                failure_args = args

    detail = f"invariant failed on input {failure_args!r}" if failure_args is not None else None
    return InvariantOutcome(expr=expr, checked=True, passed_count=passed, failed_count=failed, skipped_count=skipped, failure_detail=detail)


def check_invariants(concept: Concept, executor: Executor, timeout_s: float) -> tuple[list[CheckResult], list[InvariantOutcome]]:
    results = []
    outcomes = []
    for expr in concept.invariants:
        outcome = check_one_invariant(concept, expr, executor, timeout_s)
        outcomes.append(outcome)
        if not outcome.checked:
            results.append(CheckResult(f"invariant:{expr}", True, "unchecked (free text)"))
            continue
        if outcome.failed_count > 0:
            results.append(CheckResult(f"invariant:{expr}", False, outcome.failure_detail or "invariant failed"))
        else:
            results.append(
                CheckResult(
                    f"invariant:{expr}",
                    True,
                    f"passed on {outcome.passed_count}/{outcome.passed_count + outcome.skipped_count} evaluated inputs",
                )
            )
    return results, outcomes


def check_oracle(concept: Concept, executor: Executor, timeout_s: float, *, n: int = 30) -> CheckResult | None:
    candidates = oracle_registry.candidates_for(concept.func_name, concept.summary)
    if candidates is None:
        return None

    args_list = generate_inputs(concept.param_types, n=n)
    outcomes = executor(concept.source_code, concept.func_name, args_list, timeout_s)

    best_mismatch = None
    for reference, note in candidates:
        all_match = True
        mismatch = None
        for args, outcome in zip(args_list, outcomes):
            try:
                expected = reference(*args)
            except Exception:  # noqa: BLE001, S112 - reference domain errors are expected and skipped
                continue
            if not outcome.ok or not approx_eq(outcome.value, expected):
                all_match = False
                mismatch = (args, expected, outcome)
                break
        if all_match:
            return CheckResult("oracle", True, f"matched reference: {note}")
        if best_mismatch is None:
            best_mismatch = (note, mismatch)

    note, mismatch = best_mismatch if best_mismatch else ("<none>", None)
    if mismatch:
        args, expected, outcome = mismatch
        detail = (
            f"disagrees with reference ({note}): {concept.func_name}{args} -> "
            f"{outcome.value!r} (error={outcome.error_type}), reference gives {expected!r}"
        )
    else:
        detail = f"disagrees with reference ({note})"
    return CheckResult("oracle", False, detail)


def claims_robustness(concept: Concept) -> bool:
    text = f"{concept.func_name} {concept.summary}".lower()
    return any(word in text for word in ROBUSTNESS_CLAIM_WORDS)


def check_robustness(concept: Concept, executor: Executor, timeout_s: float) -> tuple[CheckResult, list[str]]:
    candidates = oracle_registry.candidates_for(concept.func_name, concept.summary)
    args_list = extreme_inputs(concept.param_types, n=8)
    outcomes = executor(concept.source_code, concept.func_name, args_list, timeout_s)
    claims = claims_robustness(concept)

    pitfalls: list[str] = []
    failures = []
    for args, outcome in zip(args_list, outcomes):
        if not outcome.ok:
            failures.append((args, outcome, "raised"))
            continue
        value = outcome.value
        if _has_nonfinite(value):
            if candidates:
                # only a real failure if some candidate reference is itself finite here
                for reference, _note in candidates:
                    try:
                        expected = reference(*args)
                    except Exception:  # noqa: BLE001, S112 - reference may be undefined at extreme inputs
                        continue
                    if not _has_nonfinite(expected):
                        failures.append((args, outcome, "nonfinite vs finite reference"))
                        break
            else:
                failures.append((args, outcome, "nonfinite output"))
            continue
        if candidates:
            for reference, _note in candidates:
                try:
                    expected = reference(*args)
                except Exception:  # noqa: BLE001, S112 - reference may be undefined at extreme inputs
                    continue
                if not _has_nonfinite(expected) and not approx_eq(value, expected):
                    failures.append((args, outcome, f"mismatch vs reference ({expected!r})"))
                break

    if not failures:
        return CheckResult("robustness", True, "stable on extreme-magnitude inputs"), pitfalls

    args, outcome, reason = failures[0]
    detail = f"{concept.func_name}{args} -> {outcome.value!r} ({reason})"
    if claims:
        return CheckResult("robustness", False, f"claims stability but fails at extreme inputs: {detail}"), pitfalls
    pitfalls.append(f"fails at extreme magnitudes without claiming stability: {detail}")
    return CheckResult("robustness", True, "no stability claim; extreme-input pitfall recorded"), pitfalls


def _has_nonfinite(value) -> bool:
    if isinstance(value, float):
        return math.isnan(value) or math.isinf(value)
    if isinstance(value, (list, tuple)):
        return any(_has_nonfinite(v) for v in value)
    return False


def check_determinism(concept: Concept, executor: Executor, timeout_s: float, *, n: int = 6) -> CheckResult:
    sample_args = generate_inputs(concept.param_types, n=n)
    doubled = sample_args + sample_args
    outcomes = executor(concept.source_code, concept.func_name, doubled, timeout_s)
    first_half, second_half = outcomes[:n], outcomes[n:]
    for args, o1, o2 in zip(sample_args, first_half, second_half):
        if o1.ok != o2.ok or (o1.ok and not approx_eq(o1.value, o2.value)):
            return CheckResult("determinism", False, f"{concept.func_name}{args} returned different results on repeat calls")
    return CheckResult("determinism", True, "stable across repeated calls")
