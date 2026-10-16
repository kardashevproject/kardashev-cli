"""OpenCommons correctness-verification engine.

Self-contained module meant to sit between the OpenCommons sandbox (L7) and
the blue-team review stage (L8): ``verify(concept)`` runs a fixed set of
deterministic checks -- submitted I/O contract, executable invariants,
reference-oracle agreement, claimed-robustness-under-extremes, and
determinism -- and returns a :class:`Verdict` of ``verified``, ``refuted``,
or ``unverifiable``. Submitted code is never executed in this process; every
check that runs it goes through an ``Executor`` callable, which defaults to
a resource-limited subprocess and can be swapped for another sandbox.
"""

from .executor import subprocess_executor
from .oracles import DEFAULT_ORACLES, canonicalize_name
from .types import CheckResult, Concept, Executor, Outcome, Verdict
from .verify import validate_counterexample, verify

__all__ = [
    "DEFAULT_ORACLES",
    "CheckResult",
    "Concept",
    "Executor",
    "Outcome",
    "Verdict",
    "canonicalize_name",
    "subprocess_executor",
    "validate_counterexample",
    "verify",
]
