"""Enforces the correctness-corpus confusion-matrix gates.

wrong -> verified must be exactly 0% (a wrong concept must never be trusted)
and correct -> refuted must be at most 2% (the gate should rarely punish a
correct contribution). See benchmarks/correctness_corpus/ for the corpus and
``python -m benchmarks.correctness_corpus`` for the full report.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.correctness_corpus.runner import run_corpus, summarize


def test_corpus_has_at_least_120_items():
    assert len(run_corpus()) >= 120


def test_wrong_items_are_never_verified():
    results = run_corpus()
    stats = summarize(results)
    assert stats["wrong_to_verified_rate"] == 0.0


def test_correct_items_are_rarely_refuted():
    results = run_corpus()
    stats = summarize(results)
    assert stats["correct_to_refuted_rate"] <= 0.02
