"""Loads corpus.jsonl, runs verify() on every item, and reports a confusion matrix."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from kardashev_cli.correctness import Concept, verify

CORPUS_PATH = Path(__file__).resolve().parent / "corpus.jsonl"


@dataclass
class ItemResult:
    item_id: str
    category: str
    label: str
    status: str


def _to_tuple_args(args) -> tuple:
    return tuple(tuple(a) if isinstance(a, list) else a for a in args)


def load_corpus(path: Path = CORPUS_PATH) -> list[dict]:
    items = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def item_to_concept(item: dict) -> Concept:
    contract = [(_to_tuple_args(args), expected) for args, expected in item["contract"]]
    declared = item.get("param_types")
    param_types = tuple(declared) if declared else _infer_param_types(contract)
    return Concept(
        source_code=item["source_code"],
        func_name=_func_name_from_source(item["source_code"]),
        param_types=param_types,
        return_type="",
        summary=item["summary"],
        invariants=item["invariants"],
        contract=contract,
    )


def _func_name_from_source(source_code: str) -> str:
    import ast

    tree = ast.parse(source_code)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            return node.name
    raise ValueError("no function definition found in source_code")


def _infer_param_types(contract: list[tuple[tuple, object]]) -> tuple[str, ...]:
    if not contract:
        return ()
    args, _expected = contract[0]
    types = []
    for value in args:
        if isinstance(value, bool):
            types.append("bool")
        elif isinstance(value, int):
            types.append("int")
        elif isinstance(value, float):
            types.append("float")
        elif isinstance(value, str):
            types.append("str")
        elif isinstance(value, (list, tuple)):
            inner = "float"
            if value and all(isinstance(v, int) and not isinstance(v, bool) for v in value):
                inner = "int"
            types.append(f"list[{inner}]")
        else:
            types.append("float")
    return tuple(types)


def run_corpus(path: Path = CORPUS_PATH) -> list[ItemResult]:
    results = []
    for item in load_corpus(path):
        concept = item_to_concept(item)
        verdict = verify(concept)
        results.append(ItemResult(item_id=item["id"], category=item["category"], label=item["label"], status=verdict.status))
    return results


def confusion_matrix(results: list[ItemResult]) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in results:
        matrix[r.category][f"{r.label}->{r.status}"] += 1
    return {k: dict(v) for k, v in matrix.items()}


def summarize(results: list[ItemResult]) -> dict[str, float]:
    wrong = [r for r in results if r.label == "wrong"]
    correct = [r for r in results if r.label == "correct"]
    wrong_to_verified = sum(1 for r in wrong if r.status == "verified")
    wrong_to_refuted = sum(1 for r in wrong if r.status == "refuted")
    correct_to_refuted = sum(1 for r in correct if r.status == "refuted")
    correct_to_verified = sum(1 for r in correct if r.status == "verified")
    correct_to_unverifiable = sum(1 for r in correct if r.status == "unverifiable")
    return {
        "n_wrong": len(wrong),
        "n_correct": len(correct),
        "wrong_to_verified_rate": wrong_to_verified / len(wrong) if wrong else 0.0,
        "wrong_to_refuted_rate": wrong_to_refuted / len(wrong) if wrong else 0.0,
        "correct_to_refuted_rate": correct_to_refuted / len(correct) if correct else 0.0,
        "correct_to_verified_rate": correct_to_verified / len(correct) if correct else 0.0,
        "correct_to_unverifiable_rate": correct_to_unverifiable / len(correct) if correct else 0.0,
    }


def print_report(results: list[ItemResult]) -> None:
    matrix = confusion_matrix(results)
    print(f"{len(results)} items verified\n")
    print(f"{'category':<28} outcomes")
    print("-" * 72)
    for category in sorted(matrix):
        parts = ", ".join(f"{k}={v}" for k, v in sorted(matrix[category].items()))
        print(f"{category:<28} {parts}")

    stats = summarize(results)
    print("\nSummary")
    print("-" * 72)
    print(f"wrong items:    {stats['n_wrong']}")
    print(f"  -> refuted:   {stats['wrong_to_refuted_rate']:.1%}")
    print(f"  -> verified:  {stats['wrong_to_verified_rate']:.1%}  (target: 0%)")
    print(f"correct items:  {stats['n_correct']}")
    print(f"  -> refuted:     {stats['correct_to_refuted_rate']:.1%}  (target: <=2%)")
    print(f"  -> verified:    {stats['correct_to_verified_rate']:.1%}")
    print(f"  -> unverifiable:{stats['correct_to_unverifiable_rate']:.1%}")
