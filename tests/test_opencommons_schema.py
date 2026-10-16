"""Unit tests for OpenCommons v1 Proposal JSON Schema (Draft 2020-12).

Tests:
1. Schema integrity & Draft 2020-12 conformance.
2. Positive validation for valid proposal structures.
3. Negative validation for missing required fields.
4. Strict enforcement of additionalProperties: false.
5. Name pattern and length boundaries.
6. Summary, pseudocode, invariants, and source code length constraints.
7. Pitfalls object schema and prohibition of pitfall_id.
8. Domain enum allowlist.
9. Validation of all 52 canonical seed concepts against the schema.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = REPO_ROOT / "src" / "kardashev_cli" / "opencommons_schema.json"
SEED_PATH = REPO_ROOT / "commons" / "seed_concepts_v1.json"


@pytest.fixture(scope="module")
def schema_v1() -> dict:
    with SCHEMA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def validator_v1(schema_v1) -> jsonschema.Draft202012Validator:
    jsonschema.Draft202012Validator.check_schema(schema_v1)
    return jsonschema.Draft202012Validator(schema_v1)


@pytest.fixture
def valid_proposal() -> dict:
    return {
        "name": "binary_search",
        "param_types": ["list[int]", "int"],
        "return_type": "int",
        "summary": "Efficiently locates target integer in sorted list in O(log n) time.",
        "pseudocode": "1. Initialize left=0, right=len-1\n2. Loop while left <= right\n3. Return index or -1.",
        "invariants": ["Input array must be sorted ascending", "Time complexity is bounded by O(log n)"],
        "pitfalls": [
            {
                "naive_pattern": "mid = (left + right) // 2",
                "failure_mechanism": "Integer overflow in fixed-width systems",
                "counterexample_input": "left=2**30, right=2**30",
                "verified_fix": "mid = left + (right - left) // 2",
            }
        ],
        "source_code": "def binary_search(arr: list[int], target: int) -> int:\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = left + (right - left) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1\n",
        "domain": "numerical_analysis",
    }


def test_schema_file_exists_and_is_valid_draft_2020_12(schema_v1):
    assert schema_v1["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema_v1["title"] == "OpenCommonsProposalV1"
    jsonschema.Draft202012Validator.check_schema(schema_v1)


def test_valid_proposal_passes(validator_v1, valid_proposal):
    validator_v1.validate(valid_proposal)


@pytest.mark.parametrize(
    "missing_field",
    [
        "name",
        "param_types",
        "return_type",
        "summary",
        "pseudocode",
        "invariants",
        "pitfalls",
        "source_code",
        "domain",
    ],
)
def test_missing_required_fields_fails(validator_v1, valid_proposal, missing_field):
    proposal = dict(valid_proposal)
    del proposal[missing_field]
    with pytest.raises(jsonschema.ValidationError):
        validator_v1.validate(proposal)


def test_additional_properties_forbidden_on_proposal(validator_v1, valid_proposal):
    proposal = dict(valid_proposal)
    proposal["block_id"] = "legacy_identifier"
    with pytest.raises(jsonschema.ValidationError):
        validator_v1.validate(proposal)

    proposal = dict(valid_proposal)
    proposal["teleological_intent"] = "some intent"
    with pytest.raises(jsonschema.ValidationError):
        validator_v1.validate(proposal)


@pytest.mark.parametrize(
    "invalid_name",
    [
        "BinarySearch",      # uppercase
        "ab",                # too short (<3)
        "1binary_search",    # starts with digit
        "_binary_search",    # starts with underscore
        "binary-search",     # hyphen forbidden
        "a" * 65,            # too long (>64)
        "func$name",         # special char
    ],
)
def test_name_pattern_rejections(validator_v1, valid_proposal, invalid_name):
    proposal = dict(valid_proposal)
    proposal["name"] = invalid_name
    with pytest.raises(jsonschema.ValidationError):
        validator_v1.validate(proposal)


def test_summary_length_bounds(validator_v1, valid_proposal):
    proposal = dict(valid_proposal)
    proposal["summary"] = "Too short"  # 9 chars (<10)
    with pytest.raises(jsonschema.ValidationError):
        validator_v1.validate(proposal)

    proposal["summary"] = "A" * 501  # >500 chars
    with pytest.raises(jsonschema.ValidationError):
        validator_v1.validate(proposal)


def test_pseudocode_length_bounds(validator_v1, valid_proposal):
    proposal = dict(valid_proposal)
    proposal["pseudocode"] = "Short"  # <10 chars
    with pytest.raises(jsonschema.ValidationError):
        validator_v1.validate(proposal)

    proposal["pseudocode"] = "A" * 2001  # >2000 chars
    with pytest.raises(jsonschema.ValidationError):
        validator_v1.validate(proposal)


def test_invariants_bounds(validator_v1, valid_proposal):
    proposal = dict(valid_proposal)
    # Exceed max items (8)
    proposal["invariants"] = [f"Invariant {i}" for i in range(9)]
    with pytest.raises(jsonschema.ValidationError):
        validator_v1.validate(proposal)

    # Exceed max length of invariant string (200)
    proposal["invariants"] = ["X" * 201]
    with pytest.raises(jsonschema.ValidationError):
        validator_v1.validate(proposal)


def test_pitfalls_schema_and_additional_properties(validator_v1, valid_proposal):
    # Pitfall with extra key pitfall_id
    proposal = dict(valid_proposal)
    proposal["pitfalls"] = [
        {
            "pitfall_id": "pit_123",
            "naive_pattern": "p",
            "failure_mechanism": "m",
            "counterexample_input": "c",
            "verified_fix": "f",
        }
    ]
    with pytest.raises(jsonschema.ValidationError):
        validator_v1.validate(proposal)

    # Pitfall missing verified_fix
    proposal["pitfalls"] = [
        {
            "naive_pattern": "p",
            "failure_mechanism": "m",
            "counterexample_input": "c",
        }
    ]
    with pytest.raises(jsonschema.ValidationError):
        validator_v1.validate(proposal)


@pytest.mark.parametrize(
    "domain",
    [
        "cryptography_and_framing",
        "distributed_systems",
        "geodesy_and_spatial",
        "hamiltonian_dynamics",
        "machine_learning",
        "number_theory_and_algebra",
        "numerical_analysis",
        "quantitative_finance",
        "signal_processing",
        "statistics_and_information",
        "general",
    ],
)
def test_all_11_valid_domains_pass(validator_v1, valid_proposal, domain):
    proposal = dict(valid_proposal)
    proposal["domain"] = domain
    validator_v1.validate(proposal)


def test_invalid_domain_rejected(validator_v1, valid_proposal):
    proposal = dict(valid_proposal)
    proposal["domain"] = "invalid_science_domain"
    with pytest.raises(jsonschema.ValidationError):
        validator_v1.validate(proposal)


def test_seed_concepts_v1_file_validates(validator_v1):
    assert SEED_PATH.is_file(), f"Missing {SEED_PATH}"
    with SEED_PATH.open("r", encoding="utf-8") as f:
        proposals = json.load(f)

    assert len(proposals) == 52
    for p in proposals:
        validator_v1.validate(p)
