"""Tests for OpenCommons task-reasoning, fault-localization, and planning primitives."""

from __future__ import annotations

import json
from pathlib import Path

from kardashev_cli.harness_hooks import SUPER_INTELLIGENCE_COT_PROMPT
from kardashev_cli.opencommons import execute_commons_find, execute_commons_record
from kardashev_cli.opencommons_client import VALID_DOMAINS, ConceptExtractor
from kardashev_cli.opencommons_defense import evaluate_proposal_safety
from kardashev_cli.opencommons_sandbox import run_sandboxed_probes

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED_TASK_REASONING_PATH = REPO_ROOT / "commons" / "seed_task_reasoning_v1.json"


def test_task_reasoning_domains_registered() -> None:
    assert "agentic_reasoning" in VALID_DOMAINS
    assert "task_planning_and_execution" in VALID_DOMAINS
    assert "bioinformatics_and_control" in VALID_DOMAINS
    assert "graph_and_combinatorial" in VALID_DOMAINS


def test_cot_prompt_includes_task_reasoning_domains() -> None:
    assert "agentic_reasoning" in SUPER_INTELLIGENCE_COT_PROMPT
    assert "task_planning_and_execution" in SUPER_INTELLIGENCE_COT_PROMPT


def test_concept_extractor_infer_domain() -> None:
    src_reasoning = '''def ochiai_suspiciousness(ef: int, nf: int, ep: int) -> float:
    """Compute fault localization suspiciousness score for debugging."""
    return float(ef)
'''
    proposals_r = ConceptExtractor.extract_from_source(src_reasoning, auto_infer_domain=True)
    assert len(proposals_r) == 1
    assert proposals_r[0].domain == "agentic_reasoning"

    src_planning = '''def critical_path_schedule_wave(num_tasks: int, workers: int) -> int:
    """Compute critical path wave schedule for topological task DAG."""
    return num_tasks + workers
'''
    proposals_p = ConceptExtractor.extract_from_source(src_planning, auto_infer_domain=True)
    assert len(proposals_p) == 1
    assert proposals_p[0].domain == "task_planning_and_execution"


def test_seed_task_reasoning_catalog_passes_defense_and_sandbox() -> None:
    assert SEED_TASK_REASONING_PATH.exists()
    catalog = json.loads(SEED_TASK_REASONING_PATH.read_text(encoding="utf-8"))
    assert len(catalog) == 15

    domains_seen = set()
    for item in catalog:
        domains_seen.add(item["domain"])
        verdict = evaluate_proposal_safety(item)
        assert verdict.is_accepted, f"{item['name']} failed defense: {verdict}"

        # Execute in isolated sandbox with default zero/empty args matching param_types
        default_args = []
        for pt in item["param_types"]:
            if pt.startswith("list"):
                default_args.append([])
            elif pt == "bool":
                default_args.append(False)
            elif pt == "int":
                default_args.append(1)
            elif pt == "float":
                default_args.append(0.5)
            else:
                default_args.append("")

        sb = run_sandboxed_probes(
            item["source_code"],
            item["name"],
            [tuple(default_args)],
        )
        assert sb.success is True, f"{item['name']} failed sandbox execution: {sb.error}"

    assert domains_seen == {"agentic_reasoning", "task_planning_and_execution"}


def test_execute_commons_record_and_find_task_reasoning(tmp_path: Path) -> None:
    store_file = tmp_path / "store.json"

    res = execute_commons_record(
        {
            "name": "git_bisect_midpoint_step",
            "domain": "agentic_reasoning",
            "summary": "Compute the next binary-search commit index to test during automated regression bisection.",
            "source_code": """def git_bisect_midpoint_step(good_idx: int, bad_idx: int) -> int:
    lo = min(int(good_idx), int(bad_idx))
    hi = max(int(good_idx), int(bad_idx))
    if hi - lo <= 1:
        return hi
    return lo + (hi - lo) // 2
""",
            "contract": [([0, 10], 5), ([0, 1], 1), ([4, 8], 6)],
            "invariants": ["min(good_idx, bad_idx) <= out <= max(good_idx, bad_idx)", "out == f(bad_idx, good_idx)"],
        },
        store_path=store_file,
    )
    assert res["status"] == "accepted"
    assert res["verification"]["status"] == "verified"

    found = execute_commons_find({"query": "bisect regression commit", "top_k": 5}, store_path=store_file)
    assert any(h["block_id"].startswith("git_bisect_midpoint_step") for h in found["hits"])
