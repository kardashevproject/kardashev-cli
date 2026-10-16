"""Empirical Test Suite for OpenCommons Seed Concept Migration (Milestone 1).

Verifies that all 52 seed concepts in commons/store.json migrate cleanly into v1 proposals,
satisfy all L1 schema rules, satisfy all L6 AST policy rules (single pure FunctionDef, import allowlist),
and exhibit deterministic execution and counterexample behavior.
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import jsonschema
import pytest

from kardashev_cli.opencommons_migration import (
    canonicalize_source_code,
    compute_proposal_content_hash,
    migrate_seed_store,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
STORE_PATH = REPO_ROOT / "commons" / "store.json"
SCHEMA_PATH = REPO_ROOT / "src" / "kardashev_cli" / "opencommons_schema.json"
SEED_V1_PATH = REPO_ROOT / "commons" / "seed_concepts_v1.json"


@pytest.fixture(scope="module")
def schema_v1() -> dict:
    with SCHEMA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def validator_v1(schema_v1) -> jsonschema.Draft202012Validator:
    return jsonschema.Draft202012Validator(schema_v1)


@pytest.fixture(scope="module")
def migrated_concepts() -> list[dict]:
    """Loads and migrates all 52 seed concepts."""
    return migrate_seed_store(STORE_PATH, canonicalize_ast=True)


def test_seed_store_deduplication_and_count():
    """Verify legacy store has 56 entries collapsing to exactly 52 unique canonical concepts."""
    with STORE_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    raw_blocks = data.get("blocks", [])
    assert len(raw_blocks) == 56, f"Expected 56 legacy blocks, found {len(raw_blocks)}"

    unique_names = {b["name"] for b in raw_blocks}
    assert len(unique_names) == 52, f"Expected 52 unique concept names, found {len(unique_names)}"

    migrated = migrate_seed_store(STORE_PATH)
    assert len(migrated) == 52, f"Expected 52 migrated proposals, found {len(migrated)}"


def test_migrated_concepts_schema_conformance(validator_v1, migrated_concepts):
    """Every migrated concept must validate strictly against the v1 JSON schema."""
    for proposal in migrated_concepts:
        name = proposal["name"]
        errors = list(validator_v1.iter_errors(proposal))
        assert not errors, f"Concept '{name}' failed v1 schema validation: {errors}"


def test_no_forbidden_legacy_keys_leaked(migrated_concepts):
    """Ensure internal legacy fields are completely stripped from proposals."""
    forbidden_keys = {
        "block_id",
        "cid",
        "sid",
        "fp",
        "dependencies",
        "teleological_intent",
        "dimensional_units",
        "trust_tier",
        "contributor_agent",
        "wave",
        "created_at_ms",
    }
    for proposal in migrated_concepts:
        name = proposal["name"]
        for key in forbidden_keys:
            assert key not in proposal, f"Concept '{name}' leaked forbidden legacy key '{key}'"

        for pitfall in proposal["pitfalls"]:
            assert "pitfall_id" not in pitfall, f"Concept '{name}' pitfall leaked 'pitfall_id'"


def test_ast_single_pure_functiondef(migrated_concepts):
    """Verify that every concept contains exactly 1 top-level FunctionDef matching signature."""
    for proposal in migrated_concepts:
        name = proposal["name"]
        code = proposal["source_code"]
        tree = ast.parse(code)

        assert len(tree.body) == 1, f"Concept '{name}' has {len(tree.body)} top-level nodes, expected 1"
        func = tree.body[0]
        assert isinstance(func, ast.FunctionDef), f"Concept '{name}' top node is not FunctionDef: {type(func)}"
        assert func.name == name, f"FunctionDef name '{func.name}' does not match proposal name '{name}'"

        # Parameter types match
        ast_params = [ast.unparse(arg.annotation) if arg.annotation else "Any" for arg in func.args.args]
        assert ast_params == proposal["param_types"], (
            f"Concept '{name}' params mismatch: AST {ast_params} != declared {proposal['param_types']}"
        )

        # Return type matches
        ast_ret = ast.unparse(func.returns) if func.returns else "Any"
        assert ast_ret == proposal["return_type"], (
            f"Concept '{name}' return mismatch: AST {ast_ret} != declared {proposal['return_type']}"
        )


def test_ast_import_and_isolation_policy(migrated_concepts):
    """Verify import allowlist, zero dunders, and bounded AST complexity."""
    for proposal in migrated_concepts:
        name = proposal["name"]
        code = proposal["source_code"]
        tree = ast.parse(code)

        # 1. Zero imports
        imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
        assert not imports, f"Concept '{name}' contains disallowed imports: {imports}"

        # 2. No classes, async, global, nonlocal, with
        for n in ast.walk(tree):
            assert not isinstance(n, (ast.ClassDef, ast.AsyncFunctionDef)), f"Concept '{name}' defines class/async"
            assert not isinstance(n, (ast.Global, ast.Nonlocal)), f"Concept '{name}' mutates global/nonlocal"
            assert not isinstance(n, (ast.With, ast.AsyncWith)), f"Concept '{name}' uses with-statement"

            # 3. No dunder identifiers or attributes
            if isinstance(n, ast.Name):
                assert "__" not in n.id, f"Concept '{name}' contains dunder identifier '{n.id}'"
            if isinstance(n, ast.Attribute):
                assert "__" not in n.attr, f"Concept '{name}' contains dunder attribute '{n.attr}'"

        # 4. AST node count <= 500
        node_count = sum(1 for _ in ast.walk(tree))
        assert node_count <= 500, f"Concept '{name}' exceeds 500 AST nodes ({node_count})"


def test_content_hashes_deterministic_and_unique(migrated_concepts):
    """Verify RFC 8785 canonical JSON produces 52 unique SHA-256 content hashes."""
    hashes = {}
    for proposal in migrated_concepts:
        name = proposal["name"]
        h = compute_proposal_content_hash(proposal)
        assert len(h) == 64, f"Hash for '{name}' is not 64 hex characters"
        assert h not in hashes.values(), f"Hash collision detected between '{name}' and '{hashes.get(h)}'"
        hashes[name] = h

    assert len(hashes) == 52, f"Expected 52 unique content hashes, got {len(hashes)}"


def test_canonical_unparse_idempotent(migrated_concepts):
    """Verify canonicalize_source_code is idempotent across all 52 concepts."""
    for proposal in migrated_concepts:
        name = proposal["name"]
        code = proposal["source_code"]
        re_canonicalized = canonicalize_source_code(code)
        assert code == re_canonicalized, f"Source code for '{name}' was not canonically idempotent"


def test_seed_store_file_export(tmp_path):
    """Verify export to structured target JSON file and round-trip fidelity."""
    target_json = tmp_path / "seed_concepts_v1.json"
    proposals = migrate_seed_store(STORE_PATH, output_path=target_json)
    assert target_json.is_file()

    with target_json.open("r", encoding="utf-8") as f:
        loaded = json.load(f)

    assert len(loaded) == 52
    assert loaded == proposals


def test_seed_concepts_v1_file_on_disk(migrated_concepts):
    """Verify commons/seed_concepts_v1.json matches current migration output exactly."""
    assert SEED_V1_PATH.is_file(), f"Missing {SEED_V1_PATH}"
    with SEED_V1_PATH.open("r", encoding="utf-8") as f:
        on_disk = json.load(f)

    assert len(on_disk) == 52
    assert on_disk == migrated_concepts


def test_functional_execution_samples(migrated_concepts):
    """Verify selected canonical functions execute correctly in an isolated environment."""
    proposals_by_name = {p["name"]: p for p in migrated_concepts}

    # 1. fnv1a_32_ints
    fnv_code = proposals_by_name["fnv1a_32_ints"]["source_code"]
    env: dict[str, object] = {}
    exec(fnv_code, env)  # noqa: S102
    fnv_fn = env["fnv1a_32_ints"]
    assert fnv_fn([]) == 0x811C9DC5
    assert fnv_fn([1, 2, 3]) == 1456420779

    # 2. stable_logsumexp
    lse_code = proposals_by_name["stable_logsumexp"]["source_code"]
    env_lse: dict[str, object] = {"math": math}
    exec(lse_code, env_lse)  # noqa: S102
    lse_fn = env_lse["stable_logsumexp"]
    assert lse_fn([]) == 0.0
    res = lse_fn([1000.0, 1000.0])
    assert math.isclose(res, 1000.0 + math.log(2.0), rel_tol=1e-9)

    # 3. zigzag_encode and decode
    enc_code = proposals_by_name["zigzag_encode_int"]["source_code"]
    dec_code = proposals_by_name["zigzag_decode_int"]["source_code"]
    env_zz: dict[str, object] = {}
    exec(enc_code, env_zz)  # noqa: S102
    exec(dec_code, env_zz)  # noqa: S102
    enc_fn = env_zz["zigzag_encode_int"]
    dec_fn = env_zz["zigzag_decode_int"]
    for val in [0, -1, 1, -2, 2, 1000, -1000]:
        encoded = enc_fn(val)
        decoded = dec_fn(encoded)
        assert decoded == val
