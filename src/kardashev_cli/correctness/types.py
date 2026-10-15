"""Core data types for the OpenCommons correctness-verification engine.

Nothing in this module executes submitted concept code. It only defines the
shapes that flow between the verifier, the executor, and the oracle registry.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

Status = Literal["verified", "refuted", "unverifiable"]


@dataclass(frozen=True)
class Outcome:
    """The result of one sandboxed call: either a value or an error label."""

    ok: bool
    value: Any = None
    error_type: str | None = None


@dataclass
class Concept:
    """A single contributed concept: code plus the metadata needed to check it."""

    source_code: str
    func_name: str
    param_types: tuple[str, ...]
    return_type: str
    summary: str
    invariants: list[str] = field(default_factory=list)
    # Submitter's own I/O examples: list of (args_tuple, expected_output).
    contract: list[tuple[tuple, object]] = field(default_factory=list)


@dataclass
class CheckResult:
    """Outcome of a single named check."""

    name: str
    passed: bool
    detail: str = ""


@dataclass
class Verdict:
    status: Status
    checks: list[CheckResult] = field(default_factory=list)
    counterexample: str | None = None
    pitfalls: list[str] = field(default_factory=list)


# (source, func_name, args_list, timeout_s) -> outcomes, one per args tuple.
Executor = Callable[[str, str, list[tuple], float], list[Outcome]]
