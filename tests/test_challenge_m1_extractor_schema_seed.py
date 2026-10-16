"""Empirical Challenge Test Suite for Milestone 1 (M1).

Adversarial stress-testing of ConceptExtractor, RFC 8785 JCS Hashing,
Schema Validation, and 100% execution coverage across all 52 Seed Concepts.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from kardashev_cli.opencommons_client import (
    ConceptExtractor,
)
from kardashev_cli.opencommons_migration import (
    compute_proposal_content_hash,
    rfc8785_canonical_json,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = REPO_ROOT / "src" / "kardashev_cli" / "opencommons_schema.json"
STORE_PATH = REPO_ROOT / "commons" / "store.json"
SEED_V1_PATH = REPO_ROOT / "commons" / "seed_concepts_v1.json"


@pytest.fixture(scope="module")
def schema_v1() -> dict[str, Any]:
    with SCHEMA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def validator_v1(schema_v1: dict[str, Any]) -> jsonschema.Draft202012Validator:
    return jsonschema.Draft202012Validator(schema_v1)


@pytest.fixture(scope="module")
def seed_concepts_v1() -> list[dict[str, Any]]:
    with SEED_V1_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


# ===========================================================================
# 1. ConceptExtractor Adversarial Challenge Suite
# ===========================================================================

class TestConceptExtractorChallenges:
    """Stress tests and boundary checks for ConceptExtractor."""

    def test_nesting_depth_bracket_boundary(self):
        """Test nesting depth bracket checks: 20 is allowed, 21 is rejected."""
        # 20 nested brackets
        nested_20 = "def deep_ok(x: int) -> int:\n    return " + ("(" * 20) + "x" + (")" * 20)
        assert ConceptExtractor.check_nesting_depth(nested_20, max_depth=20) is True

        # 21 nested brackets
        nested_21 = "def deep_bad(x: int) -> int:\n    return " + ("(" * 21) + "x" + (")" * 21)
        assert ConceptExtractor.check_nesting_depth(nested_21, max_depth=20) is False

        # extract_from_source returns empty list for > 20 depth
        props = ConceptExtractor.extract_from_source(nested_21)
        assert props == []

    def test_nesting_depth_indent_boundary(self):
        """Test indentation depth checks: 20 levels allowed, 21 levels rejected."""
        # 20 levels (80 spaces)
        indent_20 = "def indent_ok(x: int) -> int:\n" + (" " * 80) + "return x"
        assert ConceptExtractor.check_nesting_depth(indent_20, max_depth=20) is True

        # 21 levels (84 spaces)
        indent_21 = "def indent_bad(x: int) -> int:\n" + (" " * 84) + "return x"
        assert ConceptExtractor.check_nesting_depth(indent_21, max_depth=20) is False

        props = ConceptExtractor.extract_from_source(indent_21)
        assert props == []

    def test_rejection_of_async_functions(self):
        """ConceptExtractor must strictly reject async functions."""
        async_code = """
async def compute_async(x: int) -> int:
    return x * 2
"""
        props = ConceptExtractor.extract_from_source(async_code)
        assert props == []

    def test_rejection_of_generators(self):
        """ConceptExtractor must strictly reject generator functions (yield and yield from)."""
        gen_yield = """
def stream_items(x: int) -> int:
    yield x
"""
        assert ConceptExtractor.extract_from_source(gen_yield) == []

        gen_yield_from = """
def stream_from(xs: list[int]) -> int:
    yield from xs
"""
        assert ConceptExtractor.extract_from_source(gen_yield_from) == []

    def test_rejection_of_global_and_nonlocal(self):
        """ConceptExtractor must strictly reject functions containing global or nonlocal."""
        code_global = """
g_val = 10
def mutate_global(x: int) -> int:
    global g_val
    g_val += x
    return g_val
"""
        assert ConceptExtractor.extract_from_source(code_global) == []

        code_nonlocal = """
def outer_fn(x: int) -> int:
    y = x
    def inner_fn() -> int:
        nonlocal y
        y += 1
        return y
    return inner_fn()
"""
        assert ConceptExtractor.extract_from_source(code_nonlocal) == []

    def test_rejection_of_methods_with_self_or_cls(self):
        """ConceptExtractor must strictly reject methods containing self or cls as first parameter."""
        code_self = """
def method_on_self(self: object, x: int) -> int:
    return x + 1
"""
        assert ConceptExtractor.extract_from_source(code_self) == []

        code_cls = """
def method_on_cls(cls: object, x: int) -> int:
    return x + 1
"""
        assert ConceptExtractor.extract_from_source(code_cls) == []

        # Contrast with standard function where first arg is not self/cls
        code_valid = """
def pure_adder(a: int, b: int) -> int:
    \"\"\"Pure adder function.\"\"\"
    return a + b
"""
        props = ConceptExtractor.extract_from_source(code_valid)
        assert len(props) == 1
        assert props[0].name == "pure_adder"

    def test_syntax_errors_and_malformed_code_safe_handling(self):
        """Malformed code and syntax errors must be caught safely without crashing."""
        malformed_inputs = [
            "def broken(: int) ->",
            "for i in range(10)",
            "def 123_invalid(x: int) -> int: return x",
            "class (object): pass",
            "(((",
            "",
            "   \n\t\n",
        ]
        for bad in malformed_inputs:
            props = ConceptExtractor.extract_from_source(bad)
            assert props == []

    def test_banned_calls_and_private_attributes(self):
        """ConceptExtractor must strictly reject calls to banned functions and private attributes."""
        banned_calls = [
            "def try_eval(x: str) -> int: return eval(x)",
            "def try_exec(x: str) -> None: exec(x); return None",
            "def try_open(p: str) -> str: return open(p).read()",
            "def try_import(n: str) -> object: return __import__(n)",
            "def try_globals(x: int) -> dict: return globals()",
            "def try_locals(x: int) -> dict: return locals()",
            "def try_getattr(o: object) -> str: return getattr(o, 'name')",
        ]
        for code in banned_calls:
            assert ConceptExtractor.extract_from_source(code) == []

        # Private attribute access
        code_private_attr = """
def get_secret(o: object) -> str:
    return o._secret_field
"""
        assert ConceptExtractor.extract_from_source(code_private_attr) == []

        # Single throwaway '_' loop variable is allowed
        code_throwaway_loop = """
def count_steps(n: int) -> int:
    \"\"\"Loop counter using throwaway.\"\"\"
    total = 0
    for _ in range(n):
        total += 1
    return total
"""
        props = ConceptExtractor.extract_from_source(code_throwaway_loop)
        assert len(props) == 1
        assert props[0].name == "count_steps"

    def test_non_ascii_rejection(self):
        """Non-ASCII in function identifiers, code literals, or bodies must be rejected."""
        # Non-ASCII in function name
        code_unicode_name = """
def calc_α(x: int) -> int:
    return x * 2
"""
        assert ConceptExtractor.extract_from_source(code_unicode_name) == []

        # Non-ASCII in code body (string literal)
        code_unicode_body = """
def greet_user(x: int) -> int:
    msg = "こんにちは世界"
    return x + len(msg)
"""
        assert ConceptExtractor.extract_from_source(code_unicode_body) == []

    def test_docstring_and_comment_stripping_idempotency(self):
        """Docstring and comments must be stripped cleanly and idempotently."""
        source = '''
def calculate_metric(a: float, b: float) -> float:
    """Primary docstring line.

    Detailed multiline explanation with math details.
    """
    # Inline comment before variable declaration
    val = a * 2.0  # inline trailing comment
    # Another block comment
    return val + b
'''
        props1 = ConceptExtractor.extract_from_source(source)
        assert len(props1) == 1
        p1 = props1[0]

        # Verify docstring and comments stripped from source_code
        assert "#" not in p1.source_code
        assert "Primary docstring line" not in p1.source_code
        assert p1.summary.startswith("Primary docstring line")

        # Second extraction pass on canonical source code
        props2 = ConceptExtractor.extract_from_source(p1.source_code)
        assert len(props2) == 1
        p2 = props2[0]

        assert p1.source_code == p2.source_code
        assert p1.param_types == p2.param_types
        assert p1.return_type == p2.return_type


# ===========================================================================
# 2. RFC 8785 JCS Hashing Invariance Challenge Suite
# ===========================================================================

class TestRFC8785JCSHashingChallenges:
    """Adversarial testing of RFC 8785 JCS canonical hashing across arbitrary key ordering."""

    def test_root_and_nested_key_order_invariance(self):
        """Verify identical SHA-256 digest regardless of dictionary key insertion order."""
        dict_a = {
            "name": "sample_function",
            "param_types": ["int", "float"],
            "return_type": "float",
            "summary": "Sample summary text for testing.",
            "pseudocode": "Step 1: compute; Step 2: return.",
            "invariants": ["positive output"],
            "pitfalls": [
                {
                    "naive_pattern": "p1",
                    "failure_mechanism": "m1",
                    "counterexample_input": "c1",
                    "verified_fix": "f1",
                }
            ],
            "source_code": "def sample_function(x: int, y: float) -> float:\n    return x + y",
            "domain": "general",
        }

        hash_original = ConceptExtractor.compute_content_hash(dict_a)

        # Permute keys at root and inside nested pitfalls 100 times
        for _ in range(100):
            keys = list(dict_a.keys())
            random.shuffle(keys)
            shuffled: dict[str, Any] = {}
            for k in keys:
                if k == "pitfalls":
                    shuffled_pitfalls = []
                    for pit in dict_a["pitfalls"]:
                        pkeys = list(pit.keys())
                        random.shuffle(pkeys)
                        shuffled_pitfalls.append({pk: pit[pk] for pk in pkeys})
                    shuffled[k] = shuffled_pitfalls
                else:
                    shuffled[k] = dict_a[k]

            shuffled_hash = ConceptExtractor.compute_content_hash(shuffled)
            assert shuffled_hash == hash_original

    def test_arbitrary_nested_jcs_invariance(self):
        """Stress-test JCS serialization on deep arbitrary nested dictionary topologies."""
        obj = {
            "z": {"gamma": 3, "alpha": 1, "beta": {"b": 2, "a": 1}},
            "a": [10, {"k2": True, "k1": None, "k0": "text"}, 20],
            "m": 42.5,
            "b": False,
        }
        ref_bytes = rfc8785_canonical_json(obj)

        for _ in range(50):
            # Reverse and shuffle keys
            shuffled_obj = {
                "b": False,
                "m": 42.5,
                "a": [10, {"k0": "text", "k2": True, "k1": None}, 20],
                "z": {"beta": {"a": 1, "b": 2}, "gamma": 3, "alpha": 1},
            }
            cand_bytes = rfc8785_canonical_json(shuffled_obj)
            assert cand_bytes == ref_bytes

    def test_seed_concepts_jcs_invariance_under_shuffles(self, seed_concepts_v1):
        """Verify that all 52 seed concepts produce identical digests under key permutations."""
        for concept in seed_concepts_v1:
            canonical_hash = compute_proposal_content_hash(concept)
            for _ in range(10):
                keys = list(concept.keys())
                random.shuffle(keys)
                shuffled = {}
                for k in keys:
                    if k == "pitfalls":
                        shuffled_p = []
                        for pit in concept["pitfalls"]:
                            pkeys = list(pit.keys())
                            random.shuffle(pkeys)
                            shuffled_p.append({pk: pit[pk] for pk in pkeys})
                        shuffled[k] = shuffled_p
                    else:
                        shuffled[k] = concept[k]

                shuffled_hash = compute_proposal_content_hash(shuffled)
                assert shuffled_hash == canonical_hash


# ===========================================================================
# 3. Schema Validation Adversarial Challenge Suite
# ===========================================================================

class TestSchemaValidationChallenges:
    """Stress tests and boundary checks for opencommons_schema.json."""

    @pytest.fixture
    def valid_proposal(self) -> dict[str, Any]:
        return {
            "name": "fast_inverse_sqrt",
            "param_types": ["float"],
            "return_type": "float",
            "summary": "Computes fast approximation of 1/sqrt(x).",
            "pseudocode": "1. Bit level hack.\n2. One Newton-Raphson iteration.",
            "invariants": ["strictly positive input", "returns finite float"],
            "pitfalls": [
                {
                    "naive_pattern": "x ** -0.5",
                    "failure_mechanism": "slower throughput on older hardware",
                    "counterexample_input": "-1.0",
                    "verified_fix": "use specialized approximation",
                }
            ],
            "source_code": "def fast_inverse_sqrt(x: float) -> float:\n    return 1.0 / (x ** 0.5)",
            "domain": "numerical_analysis",
        }

    def test_additional_properties_forbidden(self, validator_v1, valid_proposal):
        """Schema must reject additional properties (governance/telemetry fields)."""
        forbidden_injections = [
            ("block_id", "blk_12345"),
            ("cid", "bafybeic5..."),
            ("trust_tier", "quarantine"),
            ("contributor_agent", "claude_code"),
            ("extra_field", 42),
        ]
        for field, value in forbidden_injections:
            tampered = dict(valid_proposal)
            tampered[field] = value
            errors = list(validator_v1.iter_errors(tampered))
            assert len(errors) > 0, f"Schema failed to reject additional property '{field}'"

    def test_missing_required_fields_rejected(self, validator_v1, valid_proposal):
        """Every required field must be strictly enforced."""
        for req in validator_v1.schema["required"]:
            tampered = dict(valid_proposal)
            del tampered[req]
            errors = list(validator_v1.iter_errors(tampered))
            assert len(errors) > 0, f"Schema allowed missing required field '{req}'"

    def test_name_regex_and_length_boundaries(self, validator_v1, valid_proposal):
        """Name must strictly match ^[a-z][a-z0-9_]{2,63}$."""
        invalid_names = [
            "ab",  # too short (2 chars)
            "a" * 65,  # too long (65 chars)
            "_private_func",  # starts with underscore
            "123_starts_digit",  # starts with digit
            "camelCaseFunc",  # contains uppercase
            "PascalCase",  # contains uppercase
            "hyphen-name",  # contains hyphen
            "name with spaces",  # contains spaces
        ]
        for bad_name in invalid_names:
            tampered = dict(valid_proposal, name=bad_name)
            errors = list(validator_v1.iter_errors(tampered))
            assert len(errors) > 0, f"Schema accepted invalid name: {bad_name}"

    def test_domain_enum_enforcement(self, validator_v1, valid_proposal):
        """Domain must be strictly within the 11 valid domain taxonomy enum."""
        invalid_domains = ["web_dev", "quantum", "chemistry", "random_domain", ""]
        for bad_dom in invalid_domains:
            tampered = dict(valid_proposal, domain=bad_dom)
            errors = list(validator_v1.iter_errors(tampered))
            assert len(errors) > 0, f"Schema accepted invalid domain: {bad_dom}"

    def test_field_length_caps(self, validator_v1, valid_proposal):
        """Verify strict upper bounds on text and array fields."""
        # Summary max 500 chars
        tampered_summary = dict(valid_proposal, summary="s" * 501)
        assert len(list(validator_v1.iter_errors(tampered_summary))) > 0

        # Pseudocode max 2000 chars
        tampered_pseudo = dict(valid_proposal, pseudocode="p" * 2001)
        assert len(list(validator_v1.iter_errors(tampered_pseudo))) > 0

        # Source code max 8192 chars
        tampered_code = dict(valid_proposal, source_code="c" * 8193)
        assert len(list(validator_v1.iter_errors(tampered_code))) > 0

        # Invariants max 8 items
        tampered_inv = dict(valid_proposal, invariants=["inv"] * 9)
        assert len(list(validator_v1.iter_errors(tampered_inv))) > 0

        # Pitfalls max 8 items
        tampered_pit = dict(valid_proposal, pitfalls=[valid_proposal["pitfalls"][0]] * 9)
        assert len(list(validator_v1.iter_errors(tampered_pit))) > 0


# ===========================================================================
# 4. 100% Execution Coverage Challenge Across All 52 Seed Concepts
# ===========================================================================

@pytest.fixture(scope="module")
def execution_env(seed_concepts_v1: list[dict[str, Any]]) -> dict[str, Any]:
    """Compiles all 52 concepts into an isolated environment containing math."""
    env: dict[str, Any] = {"math": math}
    for concept in seed_concepts_v1:
        exec(concept["source_code"], env)  # noqa: S102
    return env


class TestAll52SeedConceptsExecution:
    """Empirical verification that all 52 seed concepts execute accurately without exceptions."""

    def test_all_52_concepts_present(self, execution_env: dict[str, Any], seed_concepts_v1: list[dict[str, Any]]):
        """Verify all 52 concept functions are successfully compiled into the environment."""
        assert len(seed_concepts_v1) == 52
        for c in seed_concepts_v1:
            name = c["name"]
            assert name in execution_env
            assert callable(execution_env[name])

    def test_execute_all_52_representative_inputs(self, execution_env: dict[str, Any]):
        """Execute reference logic for every single one of the 52 concepts on representative inputs."""
        env = execution_env

        # 1. adamw_decoupled_step
        w1 = env["adamw_decoupled_step"](1.0, 0.1, 0.05, 0.01, 1e-3, 0.01, 1)
        assert isinstance(w1, float) and 0.99 < w1 < 1.01

        # 2. bbox_iou_2d
        iou = env["bbox_iou_2d"]([0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0])
        assert math.isclose(iou, 1.0 / 7.0, rel_tol=1e-6)

        # 3. bias_corrected_ema_step
        ema = env["bias_corrected_ema_step"](0.0, 10.0, 0.1, 1)
        assert math.isclose(ema, 10.0, rel_tol=1e-6)

        # 4. black_scholes_d1
        d1 = env["black_scholes_d1"](100.0, 100.0, 0.05, 0.2, 1.0)
        assert math.isclose(d1, 0.35, rel_tol=1e-6)

        # 5. bloom_optimal_k_and_fpr
        k, fpr = env["bloom_optimal_k_and_fpr"](1000, 100)
        assert k == 7 and 0.0 < fpr < 0.02

        # 6. box_muller_normal_pair
        z0, z1 = env["box_muller_normal_pair"](0.5, 0.25)
        assert isinstance(z0, float) and isinstance(z1, float)

        # 7. clip_grad_l2_norm
        clipped = env["clip_grad_l2_norm"]([3.0, 4.0], 2.5)
        assert math.isclose(math.hypot(clipped[0], clipped[1]), 2.5, rel_tol=1e-6)

        # 8. cosine_warmup_lr
        lr_warm = env["cosine_warmup_lr"](50, 100, 1000, 1e-3, 1e-5)
        assert math.isclose(lr_warm, 5e-4, rel_tol=1e-6)
        lr_decay = env["cosine_warmup_lr"](550, 100, 1000, 1e-3, 1e-5)
        assert 1e-5 < lr_decay < 1e-3

        # 9. crc32_ieee_ints
        crc = env["crc32_ieee_ints"]([49, 50, 51, 52, 53, 54, 55, 56, 57])
        assert crc == 3421780262

        # 10. exact_gelu
        gelu = env["exact_gelu"](1.0)
        assert math.isclose(gelu, 0.841344746, rel_tol=1e-6)

        # 11. exponential_backoff_delay
        backoff = env["exponential_backoff_delay"](3, 0.1, 10.0, 0.5)
        assert math.isclose(backoff, 0.6, rel_tol=1e-6)

        # 12. extended_gcd
        g, s, t = env["extended_gcd"](240, 46)
        assert g == 2 and 240 * s + 46 * t == g

        # 13. fnv1a_32_ints
        fnv = env["fnv1a_32_ints"]([1, 2, 3])
        assert fnv == 1456420779

        # 14. hamming_window_weight
        ham = env["hamming_window_weight"](0, 10)
        assert math.isclose(ham, 0.08, rel_tol=1e-6)

        # 15. haversine_distance_km
        dist = env["haversine_distance_km"](40.7128, -74.0060, 51.5074, -0.1278)
        assert 5500 < dist < 5600

        # 16. hlc_local_tick
        l_tick, c_tick = env["hlc_local_tick"](100, 2, 150)
        assert (l_tick, c_tick) == (150, 0)

        # 17. hlc_receive_merge
        l_merge, c_merge = env["hlc_receive_merge"](100, 2, 120, 1, 110)
        assert (l_merge, c_merge) == (120, 2)

        # 18. horner_polynomial_eval
        poly = env["horner_polynomial_eval"]([2.0, 3.0, 4.0], 2.0)
        assert math.isclose(poly, 18.0, rel_tol=1e-6)

        # 19. huber_robust_loss
        huber = env["huber_robust_loss"](1.5, 1.0, 1.0)
        assert math.isclose(huber, 0.125, rel_tol=1e-6)

        # 20. is_prime_miller_rabin_64
        assert env["is_prime_miller_rabin_64"](1000000007) is True
        assert env["is_prime_miller_rabin_64"](561) is False  # Carmichael number

        # 21. jensen_shannon_divergence
        jsd_zero = env["jensen_shannon_divergence"]([0.5, 0.5], [0.5, 0.5])
        assert math.isclose(jsd_zero, 0.0, abs_tol=1e-6)
        jsd_max = env["jensen_shannon_divergence"]([1.0, 0.0], [0.0, 1.0])
        assert math.isclose(jsd_max, math.log(2.0), rel_tol=1e-6)

        # 22. kelly_optimal_fraction
        k_frac = env["kelly_optimal_fraction"](0.6, 1.0, 0.5)
        assert math.isclose(k_frac, 0.2, rel_tol=1e-6)

        # 23. kl_divergence_smoothed
        kl = env["kl_divergence_smoothed"]([0.5, 0.5], [0.5, 0.5])
        assert math.isclose(kl, 0.0, abs_tol=1e-6)

        # 24. lora_adapter_scale
        scale_std = env["lora_adapter_scale"](16.0, 4, False)
        assert math.isclose(scale_std, 4.0, rel_tol=1e-6)
        scale_rs = env["lora_adapter_scale"](16.0, 4, True)
        assert math.isclose(scale_rs, 8.0, rel_tol=1e-6)

        # 25. maechler_log1mexp
        log1m = env["maechler_log1mexp"](2.0)
        assert math.isclose(log1m, math.log(1.0 - math.exp(-2.0)), rel_tol=1e-6)

        # 26. modular_pow_fast
        mod_pow = env["modular_pow_fast"](3, 5, 7)
        assert mod_pow == 5

        # 27. neumaier_compensated_sum
        n_sum = env["neumaier_compensated_sum"]([1.0, 1e100, 1.0, -1e100])
        assert math.isclose(n_sum, 2.0, rel_tol=1e-6)

        # 28. normalized_sinc
        assert math.isclose(env["normalized_sinc"](0.0), 1.0, rel_tol=1e-6)
        assert math.isclose(env["normalized_sinc"](1.0), 0.0, abs_tol=1e-6)

        # 29. online_softmax_update
        m_up, d_up = env["online_softmax_update"](0.0, 1.0, 1.0)
        assert m_up == 1.0 and d_up > 1.0

        # 30. pchip_monotone_slope
        slope = env["pchip_monotone_slope"](1.0, 2.0, 1.0, 1.0)
        assert math.isclose(slope, 4.0 / 3.0, rel_tol=1e-6)

        # 31. rendezvous_hrw_select
        chosen = env["rendezvous_hrw_select"](42, [1, 2, 3, 4])
        assert chosen in [1, 2, 3, 4]

        # 32. rk4_damped_step
        rk4 = env["rk4_damped_step"](1.0, 0.5, 0.1)
        assert 0.9 < rk4 < 1.0

        # 33. rmsnorm_vector
        rms = env["rmsnorm_vector"]([2.0, 2.0], 1e-6)
        assert math.isclose(rms[0], 1.0, rel_tol=1e-3)

        # 34. rope_rotate_pair
        rx, ry = env["rope_rotate_pair"](1.0, 2.0, 3.0, 0.5)
        assert isinstance(rx, float) and isinstance(ry, float)

        # 35. safe_cosine_similarity
        cos_orth = env["safe_cosine_similarity"]([1.0, 0.0], [0.0, 1.0])
        assert math.isclose(cos_orth, 0.0, abs_tol=1e-6)
        cos_par = env["safe_cosine_similarity"]([1.0, 2.0], [2.0, 4.0])
        assert math.isclose(cos_par, 1.0, rel_tol=1e-6)

        # 36. safe_logit
        logit_mid = env["safe_logit"](0.5)
        assert math.isclose(logit_mid, 0.0, abs_tol=1e-6)

        # 37. safe_silu_swish
        silu = env["safe_silu_swish"](2.0, 1.0)
        assert math.isclose(silu, 2.0 / (1.0 + math.exp(-2.0)), rel_tol=1e-6)

        # 38. scaled_hypot_2d
        hypot = env["scaled_hypot_2d"](3.0, 4.0)
        assert math.isclose(hypot, 5.0, rel_tol=1e-6)

        # 39. shannon_entropy_nats
        ent = env["shannon_entropy_nats"]([1.0, 1.0])
        assert math.isclose(ent, math.log(2.0), rel_tol=1e-6)

        # 40. stable_binary_cross_entropy_with_logits
        bce = env["stable_binary_cross_entropy_with_logits"](0.0, 1.0)
        assert math.isclose(bce, math.log(2.0), rel_tol=1e-6)

        # 41. stable_logsumexp
        lse = env["stable_logsumexp"]([1000.0, 1000.0])
        assert math.isclose(lse, 1000.0 + math.log(2.0), rel_tol=1e-6)

        # 42. stable_real_quadratic_roots
        r1, r2 = env["stable_real_quadratic_roots"](1.0, -5.0, 6.0)
        assert math.isclose(r1, 2.0, rel_tol=1e-6) and math.isclose(r2, 3.0, rel_tol=1e-6)

        # 43. stable_sigmoid
        assert math.isclose(env["stable_sigmoid"](0.0), 0.5, rel_tol=1e-6)
        assert math.isclose(env["stable_sigmoid"](100.0), 1.0, rel_tol=1e-6)

        # 44. stable_softplus
        assert math.isclose(env["stable_softplus"](0.0), math.log(2.0), rel_tol=1e-6)

        # 45. symplectic_leapfrog_step
        x_leap, v_leap = env["symplectic_leapfrog_step"](1.0, 0.0, 1.0, 0.1)
        assert isinstance(x_leap, float) and isinstance(v_leap, float)

        # 46. token_bucket_step
        allowed, remaining = env["token_bucket_step"](5.0, 10.0, 1.0, 1.0, 2.0)
        assert allowed is True and math.isclose(remaining, 4.0, rel_tol=1e-6)

        # 47. uleb128_decode
        assert env["uleb128_decode"]([229, 142, 38]) == 624485

        # 48. uleb128_encode
        assert env["uleb128_encode"](624485) == [229, 142, 38]

        # 49. welford_variance
        var = env["welford_variance"]([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        assert math.isclose(var, 4.0, rel_tol=1e-6)

        # 50. wilson_score_interval
        w_low, w_high = env["wilson_score_interval"](50, 100, 1.96)
        assert 0.0 <= w_low < 0.5 < w_high <= 1.0

        # 51. zigzag_decode_int
        assert env["zigzag_decode_int"](2) == 1

        # 52. zigzag_encode_int
        assert env["zigzag_encode_int"](-1) == 1

    def test_dual_run_determinism_all_52(self, execution_env: dict[str, Any]):
        """Verify strict dual-run determinism across all 52 functions."""
        env = execution_env

        sample_calls = [
            ("adamw_decoupled_step", (1.0, 0.1, 0.05, 0.01, 1e-3, 0.01, 1)),
            ("bbox_iou_2d", ([0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0])),
            ("bias_corrected_ema_step", (0.0, 10.0, 0.1, 1)),
            ("black_scholes_d1", (100.0, 100.0, 0.05, 0.2, 1.0)),
            ("bloom_optimal_k_and_fpr", (1000, 100)),
            ("box_muller_normal_pair", (0.5, 0.25)),
            ("clip_grad_l2_norm", ([3.0, 4.0], 2.5)),
            ("cosine_warmup_lr", (50, 100, 1000, 1e-3, 1e-5)),
            ("crc32_ieee_ints", ([49, 50, 51, 52, 53, 54, 55, 56, 57],)),
            ("exact_gelu", (1.0,)),
            ("exponential_backoff_delay", (3, 0.1, 10.0, 0.5)),
            ("extended_gcd", (240, 46)),
            ("fnv1a_32_ints", ([1, 2, 3],)),
            ("hamming_window_weight", (0, 10)),
            ("haversine_distance_km", (40.7128, -74.0060, 51.5074, -0.1278)),
            ("hlc_local_tick", (100, 2, 150)),
            ("hlc_receive_merge", (100, 2, 120, 1, 110)),
            ("horner_polynomial_eval", ([2.0, 3.0, 4.0], 2.0)),
            ("huber_robust_loss", (1.5, 1.0, 1.0)),
            ("is_prime_miller_rabin_64", (1000000007,)),
            ("jensen_shannon_divergence", ([0.5, 0.5], [0.5, 0.5])),
            ("kelly_optimal_fraction", (0.6, 1.0, 0.5)),
            ("kl_divergence_smoothed", ([0.5, 0.5], [0.5, 0.5])),
            ("lora_adapter_scale", (16.0, 4, True)),
            ("maechler_log1mexp", (2.0,)),
            ("modular_pow_fast", (3, 5, 7)),
            ("neumaier_compensated_sum", ([1.0, 1e100, 1.0, -1e100],)),
            ("normalized_sinc", (0.0,)),
            ("online_softmax_update", (0.0, 1.0, 1.0)),
            ("pchip_monotone_slope", (1.0, 2.0, 1.0, 1.0)),
            ("rendezvous_hrw_select", (42, [1, 2, 3, 4])),
            ("rk4_damped_step", (1.0, 0.5, 0.1)),
            ("rmsnorm_vector", ([2.0, 2.0], 1e-6)),
            ("rope_rotate_pair", (1.0, 2.0, 3.0, 0.5)),
            ("safe_cosine_similarity", ([1.0, 0.0], [0.0, 1.0])),
            ("safe_logit", (0.5,)),
            ("safe_silu_swish", (2.0, 1.0)),
            ("scaled_hypot_2d", (3.0, 4.0)),
            ("shannon_entropy_nats", ([1.0, 1.0],)),
            ("stable_binary_cross_entropy_with_logits", (0.0, 1.0)),
            ("stable_logsumexp", ([1000.0, 1000.0],)),
            ("stable_real_quadratic_roots", (1.0, -5.0, 6.0)),
            ("stable_sigmoid", (0.0,)),
            ("stable_softplus", (0.0,)),
            ("symplectic_leapfrog_step", (1.0, 0.0, 1.0, 0.1)),
            ("token_bucket_step", (5.0, 10.0, 1.0, 1.0, 2.0)),
            ("uleb128_decode", ([229, 142, 38],)),
            ("uleb128_encode", (624485,)),
            ("welford_variance", ([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0],)),
            ("wilson_score_interval", (50, 100, 1.96)),
            ("zigzag_decode_int", (2,)),
            ("zigzag_encode_int", (-1,)),
        ]

        assert len(sample_calls) == 52
        for fn_name, args in sample_calls:
            fn = env[fn_name]
            out1 = fn(*args)
            out2 = fn(*args)
            assert out1 == out2, f"Non-deterministic execution in {fn_name}: {out1} != {out2}"
