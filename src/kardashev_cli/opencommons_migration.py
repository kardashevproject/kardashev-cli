"""Migration script & utilities for OpenCommons Seed Concepts (Milestone 1).

Converts legacy ConceptBlocks in commons/store.json into valid OpenCommons v1 proposal
specifications adhering to src/kardashev_cli/opencommons_schema.json and L1-L6 defense rules.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any

VALID_DOMAINS: frozenset[str] = frozenset(
    {
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
    }
)


def rfc8785_canonical_json(obj: Any) -> bytes:
    """Deterministic JSON serialization conforming to RFC 8785 (JCS)."""
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def compute_proposal_content_hash(proposal: dict[str, Any]) -> str:
    """Computes SHA-256 digest over RFC 8785 canonical JSON bytes."""
    return hashlib.sha256(rfc8785_canonical_json(proposal)).hexdigest()


def canonicalize_source_code(source_code: str, normalize_throwaways: bool = False) -> str:
    """Parses, verifies, and canonicalizes Python source code via ast.unparse().

    Optionally renames single throwaway loop variables '_' to named loop indices
    if strict AST identifier filtering forbids leading underscores.
    """
    code = source_code.strip()
    tree = ast.parse(code)

    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("Source code must contain exactly one top-level FunctionDef")

    func = tree.body[0]

    # Strip any docstrings
    if (
        func.body
        and isinstance(func.body[0], ast.Expr)
        and isinstance(func.body[0].value, ast.Constant)
        and isinstance(func.body[0].value.value, str)
    ):
        func.body.pop(0)

    if normalize_throwaways:
        class ThrowawayRenamer(ast.NodeTransformer):
            def __init__(self) -> None:
                self.counter = 0

            def visit_For(self, node: ast.For) -> ast.For:
                if isinstance(node.target, ast.Name) and node.target.id == "_":
                    self.counter += 1
                    node.target.id = f"iter_step_{self.counter}"
                self.generic_visit(node)
                return node

        ThrowawayRenamer().visit(tree)

    return ast.unparse(tree)


def migrate_legacy_block(
    block: dict[str, Any],
    canonicalize_ast: bool = True,
    normalize_throwaways: bool = False,
) -> dict[str, Any]:
    """Transforms a legacy ConceptBlock dictionary into a strict v1 proposal dictionary.

    Strips internal/governance fields (block_id, cid, sid, fp, dependencies,
    teleological_intent, dimensional_units, trust_tier, contributor_agent, wave,
    created_at_ms) and strips pitfall_id from each pitfall object.
    """
    name = str(block.get("name", "")).strip()
    if not re.match(r"^[a-z][a-z0-9_]{2,63}$", name):
        raise ValueError(f"Invalid concept name '{name}': does not match ^[a-z][a-z0-9_]{{2,63}}$")

    domain = str(block.get("domain", "general")).strip()
    if domain not in VALID_DOMAINS:
        raise ValueError(f"Invalid domain '{domain}' for concept '{name}'. Must be in {sorted(VALID_DOMAINS)}")

    param_types = [str(t).strip() for t in block.get("param_types", [])]
    return_type = str(block.get("return_type", "")).strip()
    summary = str(block.get("summary", "")).strip()
    pseudocode = str(block.get("pseudocode", "")).strip()
    invariants = [str(inv).strip() for inv in block.get("invariants", [])]

    raw_code = str(block.get("source_code", "")).strip()
    if canonicalize_ast:
        code = canonicalize_source_code(raw_code, normalize_throwaways=normalize_throwaways)
    else:
        code = raw_code

    cleaned_pitfalls = []
    for p in block.get("pitfalls", []):
        cleaned_pitfalls.append(
            {
                "naive_pattern": str(p.get("naive_pattern", "")).strip(),
                "failure_mechanism": str(p.get("failure_mechanism", "")).strip(),
                "counterexample_input": str(p.get("counterexample_input", "")).strip(),
                "verified_fix": str(p.get("verified_fix", "")).strip(),
            }
        )

    return {
        "name": name,
        "param_types": param_types,
        "return_type": return_type,
        "summary": summary,
        "pseudocode": pseudocode,
        "invariants": invariants,
        "pitfalls": cleaned_pitfalls,
        "source_code": code,
        "domain": domain,
    }


def migrate_seed_store(
    legacy_store_path: Path | str,
    output_path: Path | str | None = None,
    canonicalize_ast: bool = True,
    normalize_throwaways: bool = False,
) -> list[dict[str, Any]]:
    """Loads legacy store.json, deduplicates blocks, and converts all 52 seed concepts.

    If output_path is specified, writes out the migrated proposals as formatted JSON.
    """
    store_file = Path(legacy_store_path)
    if not store_file.is_file():
        raise FileNotFoundError(f"Legacy store file not found: {store_file}")

    with store_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    blocks = data.get("blocks", [])
    unique_blocks: dict[str, dict[str, Any]] = {}

    for b in blocks:
        name = b["name"]
        if name not in unique_blocks:
            unique_blocks[name] = b

    if len(unique_blocks) != 52:
        raise ValueError(f"Expected exactly 52 unique seed concepts, found {len(unique_blocks)}")

    migrated_proposals: list[dict[str, Any]] = []
    for name in sorted(unique_blocks.keys()):
        proposal = migrate_legacy_block(
            unique_blocks[name],
            canonicalize_ast=canonicalize_ast,
            normalize_throwaways=normalize_throwaways,
        )
        migrated_proposals.append(proposal)

    if output_path is not None:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write
        temp_path = out_file.with_suffix(".tmp")
        with temp_path.open("w", encoding="utf-8") as f:
            json.dump(migrated_proposals, f, indent=2, ensure_ascii=False)
            f.write("\n")
        temp_path.replace(out_file)

    return migrated_proposals


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent.parent
    legacy_store = repo_root / "commons" / "store.json"
    output_target = repo_root / "commons" / "seed_concepts_v1.json"
    proposals = migrate_seed_store(legacy_store, output_target)
    print(f"Successfully migrated {len(proposals)} seed concepts to {output_target}")


if __name__ == "__main__":
    main()
