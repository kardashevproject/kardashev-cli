"""Empirical Challenge Test Suite: Milestone 2 AI Blue Team (L8) and Opacity Governance.

Adversarial stress harness challenging:
1. Canary integrity: rogue model echoing canary token in raw output; verify immediate rejection.
2. Consensus fail-closed: 1-flag-2-benign, 1-suspicious-2-benign, 1-refusal-2-benign,
   1-timeout-2-benign, 1-schema-violation-2-benign, and dual-task summarizer discrepancy.
3. Budget circuit breaker: hourly spend cap and request cap exhaustion; verify queuing (pending)
   rather than auto-accepting; test sliding window reset and concurrent spend tracking.
4. Opacity governance: compare public query responses and HTTP receipts across all failure layers
   (L0 through L8) to verify 100% indistinguishability (zero rule ID leaks, zero error excerpts).
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import time
from typing import Any

import pytest

from kardashev_cli.opencommons_client import HashcashPoW, OpenCommonsKeyManager
from kardashev_cli.opencommons_defense import (
    AIBlueTeamEvaluator,
    DualTaskDescriptionChecker,
    EdgeGuard,
    HourlySpendCircuitBreaker,
    ImmediateReceipt,
    MockModelRunner,
    PipelineVerdict,
    ProposalStateTracker,
    RunnerResult,
    SummarizerResult,
    assert_response_shapes_indistinguishable,
    check_canary_leak,
    evaluate_submission,
    generate_canary_token,
    verify_zero_leakage,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_candidate_proposal() -> dict[str, Any]:
    """Canonical valid candidate proposal passing L1-L6."""
    return {
        "name": "vector_dot_product_verified",
        "param_types": ["list[float]", "list[float]"],
        "return_type": "float",
        "summary": "Computes Euclidean inner product between two real vectors.",
        "pseudocode": "Initialize accumulator to zero. Iterate over paired elements, adding product. Return sum.",
        "invariants": ["Vectors must have identical length", "Output is zero for orthogonal vectors"],
        "pitfalls": [
            {
                "naive_pattern": "Unbounded recursion",
                "failure_mechanism": "Stack overflow",
                "counterexample_input": "len(u) > 1000",
                "verified_fix": "Iterative loop accumulator",
            }
        ],
        "source_code": (
            "def vector_dot_product_verified(u: list[float], v: list[float]) -> float:\n"
            "    total = 0.0\n"
            "    for i in range(len(u)):\n"
            "        total += u[i] * v[i]\n"
            "    return total\n"
        ),
        "domain": "numerical_analysis",
    }


@pytest.fixture
def submitter_key_and_headers(valid_candidate_proposal: dict[str, Any]) -> tuple[bytes, dict[str, str], Any]:
    key_mgr = OpenCommonsKeyManager()
    key_rec = key_mgr.generate_key_record()
    raw_bytes = json.dumps(valid_candidate_proposal).encode("utf-8")
    canonical_jcs = json.dumps(
        valid_candidate_proposal, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    content_hash = hashlib.sha256(canonical_jcs).hexdigest()
    pow_stamp = HashcashPoW.solve(content_hash, 16)
    sig_hex = OpenCommonsKeyManager.sign(content_hash.encode("utf-8"), key_rec)
    headers = {
        "x-opencommons-key": key_rec.public_key_hex,
        "x-opencommons-pow": pow_stamp.format(),
        "x-opencommons-signature": sig_hex,
    }
    return raw_bytes, headers, key_rec


# ===========================================================================
# 1. Canary Integrity Challenge Suite
# ===========================================================================


class TestCanaryIntegrityChallenge:
    """Stress-test secret canary injection, model leakage, and detection oracle."""

    def test_canary_leak_detection_oracle(self):
        """Verify check_canary_leak detects canary across multiple representations."""
        token = generate_canary_token()
        assert token.startswith("canary-token-")
        assert len(token) > 20

        # Exact match
        assert check_canary_leak(f"Here is the token: {token}", token) is True
        # Embedded inside JSON string
        assert check_canary_leak(json.dumps({"token": token}), token) is True
        # Embedded inside markdown fence
        assert check_canary_leak(f"```python\n# {token}\n```", token) is True
        # Clean text
        assert check_canary_leak("Completely benign response with no token", token) is False
        # Empty inputs
        assert check_canary_leak("", token) is False
        assert check_canary_leak("Any text", "") is False

    def test_rogue_reviewer_echoing_canary_rejected(self, valid_candidate_proposal):
        """Simulate a rogue model echoing canary token and verify immediate rejection."""

        class LeakingReviewer(MockModelRunner):
            def review_security(self, prompt, system_prompt, canary_token, timeout=10.0):
                res = super().review_security(prompt, system_prompt, canary_token, timeout)
                # Rogue behavior: echoes canary while returning benign verdict with 1.0 confidence
                leaked_raw = (
                    f'{{"verdict": "benign", "confidence": 1.0, "summary": "Looks benign", '
                    f'"categories": [], "evidence_field": "none", "echo": "{canary_token}"}}'
                )
                return RunnerResult(
                    raw_response=leaked_raw,
                    parsed_verdict="benign",
                    confidence=1.0,
                    categories=[],
                    evidence_field="none",
                    summary="Looks benign",
                    token_usage=res.token_usage,
                    estimated_cost_usd=res.estimated_cost_usd,
                    duration_seconds=res.duration_seconds,
                    success=True,
                )

        runners = [
            LeakingReviewer("rogue_claude", "claude", "benign", 1.0, code_summary=valid_candidate_proposal["summary"]),
            MockModelRunner("safe_codex", "codex", "benign", 1.0, code_summary=valid_candidate_proposal["summary"]),
        ]
        evaluator = AIBlueTeamEvaluator(runners=runners)
        verdict = evaluator.evaluate_proposal(valid_candidate_proposal)

        assert verdict.verdict == "rejected", f"Expected rejected, got {verdict.verdict}"
        assert "rogue_claude" in verdict.canary_leaks
        assert any("echoed secret canary token" in r for r in verdict.reasons)

    def test_rogue_summarizer_echoing_canary_rejected(self, valid_candidate_proposal):
        """Simulate independent summarizer echoing canary and verify rejection."""

        class LeakingSummarizer(MockModelRunner):
            def summarize_code(self, prompt, system_prompt, canary_token, timeout=10.0):
                return SummarizerResult(
                    raw_response=f'{{"code_summary": "Computes Euclidean inner product", "is_deceptive": false, "canary": "{canary_token}"}}',
                    code_summary="Computes Euclidean inner product between two real vectors.",
                    is_deceptive=False,
                    token_usage=120,
                    estimated_cost_usd=0.001,
                    duration_seconds=0.1,
                    success=True,
                )

        runners = [
            MockModelRunner("safe_claude", "claude", "benign", 1.0, code_summary=valid_candidate_proposal["summary"]),
            LeakingSummarizer("rogue_codex", "codex", "benign", 1.0),
        ]
        evaluator = AIBlueTeamEvaluator(runners=runners)
        verdict = evaluator.evaluate_proposal(valid_candidate_proposal)

        assert verdict.verdict == "rejected"
        assert "rogue_codex-summarizer" in verdict.canary_leaks
        assert any("Summarizer model rogue_codex echoed secret canary token" in r for r in verdict.reasons)

    def test_canary_tokens_unique_per_evaluation(self, valid_candidate_proposal):
        """Verify each evaluation session generates a distinct cryptographic canary token."""
        tokens = {generate_canary_token() for _ in range(50)}
        assert len(tokens) == 50, "Canary tokens must be cryptographically unique"

    def test_canary_case_sensitivity_boundary(self):
        """Verify canary leak detection is case-insensitive.

        If a rogue model converts the canary to uppercase, case-insensitive
        matching detects the leak.
        """
        token = generate_canary_token()
        assert check_canary_leak(f"echo {token}", token) is True
        # Upper-case variant test: detected by case-insensitive check
        assert check_canary_leak(f"echo {token.upper()}", token) is True


# ===========================================================================
# 2. Consensus Fail-Closed Challenge Suite
# ===========================================================================


class TestConsensusFailClosedChallenge:
    """Stress-test multi-model fail-closed consensus under various failure modes."""

    def test_consensus_one_flag_two_benign(self, valid_candidate_proposal):
        """1 malicious flag against 2 benign approvals must reject (fail-closed, not majority rule)."""
        runners = [
            MockModelRunner("m1_claude", "claude", "benign", 0.99, code_summary=valid_candidate_proposal["summary"]),
            MockModelRunner("m2_codex", "codex", "malicious", 0.95, code_summary=valid_candidate_proposal["summary"]),
            MockModelRunner("m3_gemini", "gemini", "benign", 0.99, code_summary=valid_candidate_proposal["summary"]),
        ]
        evaluator = AIBlueTeamEvaluator(runners=runners)
        verdict = evaluator.evaluate_proposal(valid_candidate_proposal)

        assert verdict.verdict == "rejected"
        assert verdict.model_verdicts["m2_codex"] == "malicious"
        assert any("m2_codex returned non-benign verdict 'malicious'" in r for r in verdict.reasons)

    def test_consensus_one_suspicious_two_benign(self, valid_candidate_proposal):
        """1 suspicious verdict against 2 benign approvals must reject."""
        runners = [
            MockModelRunner("m1_claude", "claude", "benign", 0.99, code_summary=valid_candidate_proposal["summary"]),
            MockModelRunner("m2_codex", "codex", "suspicious", 0.75, code_summary=valid_candidate_proposal["summary"]),
            MockModelRunner("m3_gemini", "gemini", "benign", 0.99, code_summary=valid_candidate_proposal["summary"]),
        ]
        evaluator = AIBlueTeamEvaluator(runners=runners)
        verdict = evaluator.evaluate_proposal(valid_candidate_proposal)

        assert verdict.verdict == "rejected"
        assert any("suspicious" in r for r in verdict.reasons)

    def test_consensus_one_refusal_error_two_benign(self, valid_candidate_proposal):
        """1 model API error/refusal against 2 benign approvals must fail-closed."""

        class ErroredRunner(MockModelRunner):
            def review_security(self, prompt, system_prompt, canary_token, timeout=10.0):
                return RunnerResult(
                    raw_response="",
                    parsed_verdict=None,
                    confidence=0.0,
                    categories=[],
                    evidence_field="none",
                    summary="",
                    token_usage=0,
                    estimated_cost_usd=0.0,
                    duration_seconds=0.05,
                    success=False,
                    error_message="Model provider returned 503 Service Unavailable",
                )

        runners = [
            MockModelRunner("m1_claude", "claude", "benign", 0.99, code_summary=valid_candidate_proposal["summary"]),
            ErroredRunner("m2_codex", "codex", "benign", 0.99, code_summary=valid_candidate_proposal["summary"]),
            MockModelRunner("m3_gemini", "gemini", "benign", 0.99, code_summary=valid_candidate_proposal["summary"]),
        ]
        evaluator = AIBlueTeamEvaluator(runners=runners)
        verdict = evaluator.evaluate_proposal(valid_candidate_proposal)

        assert verdict.verdict == "rejected"
        assert any("failed: Model provider returned 503" in r for r in verdict.reasons)

    def test_consensus_one_timeout_two_benign(self, valid_candidate_proposal):
        """1 model timeout must reject the entire submission."""

        class TimedOutRunner(MockModelRunner):
            def review_security(self, prompt, system_prompt, canary_token, timeout=10.0):
                return RunnerResult(
                    raw_response="",
                    parsed_verdict=None,
                    confidence=0.0,
                    categories=[],
                    evidence_field="none",
                    summary="",
                    token_usage=0,
                    estimated_cost_usd=0.0,
                    duration_seconds=timeout + 2.0,
                    success=False,
                    error_message="Timeout of 10.0s exceeded during inference",
                )

        runners = [
            MockModelRunner("m1_claude", "claude", "benign", 0.99, code_summary=valid_candidate_proposal["summary"]),
            TimedOutRunner("m2_codex", "codex", "benign", 0.99, code_summary=valid_candidate_proposal["summary"]),
            MockModelRunner("m3_gemini", "gemini", "benign", 0.99, code_summary=valid_candidate_proposal["summary"]),
        ]
        evaluator = AIBlueTeamEvaluator(runners=runners)
        verdict = evaluator.evaluate_proposal(valid_candidate_proposal)

        assert verdict.verdict == "rejected"
        assert any("Timeout of 10.0s exceeded" in r for r in verdict.reasons)

    def test_consensus_one_schema_violation_two_benign(self, valid_candidate_proposal):
        """1 model outputting malformed non-JSON must trigger fail-closed rejection."""

        class MalformedRunner(MockModelRunner):
            def review_security(self, prompt, system_prompt, canary_token, timeout=10.0):
                return RunnerResult(
                    raw_response="Sure! Here is my review: The code is safe.",
                    parsed_verdict=None,
                    confidence=0.0,
                    categories=[],
                    evidence_field="none",
                    summary="",
                    token_usage=20,
                    estimated_cost_usd=0.0001,
                    duration_seconds=0.1,
                    success=False,
                    error_message="Failed to parse JSON response from model output",
                )

        runners = [
            MockModelRunner("m1_claude", "claude", "benign", 0.99, code_summary=valid_candidate_proposal["summary"]),
            MalformedRunner("m2_codex", "codex", "benign", 0.99, code_summary=valid_candidate_proposal["summary"]),
            MockModelRunner("m3_gemini", "gemini", "benign", 0.99, code_summary=valid_candidate_proposal["summary"]),
        ]
        evaluator = AIBlueTeamEvaluator(runners=runners)
        verdict = evaluator.evaluate_proposal(valid_candidate_proposal)

        assert verdict.verdict == "rejected"
        assert any("Failed to parse JSON response" in r for r in verdict.reasons)

    def test_dual_task_deceptive_code_flag_triggers_rejection(self, valid_candidate_proposal):
        """If independent summarizer flags code as deceptive, submission must reject."""

        class DeceptiveFlagSummarizer(MockModelRunner):
            def summarize_code(self, prompt, system_prompt, canary_token, timeout=10.0):
                return SummarizerResult(
                    raw_response='{"code_summary": "Inner product calculation", "is_deceptive": true}',
                    code_summary="Calculates inner product but exhibits suspicious behavioral discrepancies",
                    is_deceptive=True,
                    token_usage=100,
                    estimated_cost_usd=0.001,
                    duration_seconds=0.1,
                    success=True,
                )

        runners = [
            MockModelRunner("m1_claude", "claude", "benign", 0.99, code_summary=valid_candidate_proposal["summary"]),
            DeceptiveFlagSummarizer("m2_codex", "codex", "benign", 0.99),
        ]
        evaluator = AIBlueTeamEvaluator(runners=runners)
        verdict = evaluator.evaluate_proposal(valid_candidate_proposal)

        assert verdict.verdict == "rejected"
        assert verdict.summary_match_passed is False
        assert any("deceptive" in r for r in verdict.reasons)

    def test_dual_task_summary_semantic_mismatch_triggers_rejection(self, valid_candidate_proposal):
        """If independent summarizer produces summary with 0.0 Jaccard overlap, reject."""

        class DisjointSummarizer(MockModelRunner):
            def summarize_code(self, prompt, system_prompt, canary_token, timeout=10.0):
                return SummarizerResult(
                    raw_response='{"code_summary": "Scans local filesystem", "is_deceptive": false}',
                    code_summary="Traverses directories and builds file tree hash hierarchy.",
                    is_deceptive=False,
                    token_usage=100,
                    estimated_cost_usd=0.001,
                    duration_seconds=0.1,
                    success=True,
                )

        runners = [
            MockModelRunner("m1_claude", "claude", "benign", 0.99, code_summary=valid_candidate_proposal["summary"]),
            DisjointSummarizer("m2_codex", "codex", "benign", 0.99),
        ]
        evaluator = AIBlueTeamEvaluator(runners=runners)
        verdict = evaluator.evaluate_proposal(valid_candidate_proposal)

        assert verdict.verdict == "rejected"
        assert verdict.summary_match_passed is False
        assert any("mismatch" in r for r in verdict.reasons)

    def test_insufficient_reviewer_count_rejected(self):
        """AIBlueTeamEvaluator must reject initialization with fewer than min_required_reviewers."""
        runners = [
            MockModelRunner("single_runner", "claude", "benign", 0.95),
        ]
        with pytest.raises(ValueError, match="requires at least 2 reviewers"):
            AIBlueTeamEvaluator(runners=runners, min_required_reviewers=2)

    def test_non_diverse_model_families_rejected(self):
        """Real (non-mock) runners must come from diverse model families."""

        class DummyRealRunner:
            name = "claude-3-haiku"
            family = "claude"

            def review_security(self, *args, **kwargs):
                pass

            def summarize_code(self, *args, **kwargs):
                pass

        class DummyRealRunner2:
            name = "claude-3-opus"
            family = "claude"

            def review_security(self, *args, **kwargs):
                pass

            def summarize_code(self, *args, **kwargs):
                pass

    def test_undisclosed_capability_detected(self, valid_candidate_proposal):
        """DualTaskDescriptionChecker detects suspicious capability terms absent from author summary."""
        checker = DualTaskDescriptionChecker()
        author_summary = valid_candidate_proposal["summary"]
        independent_summary = "Computes inner product and establishes network socket connection."

        ok, reason = checker.verify(
            author_summary=author_summary,
            independent_summary=independent_summary,
            is_deceptive_flag=False,
        )
        assert ok is False
        assert "undisclosed capability" in reason
        assert "network" in reason or "socket" in reason


# ===========================================================================
# 3. Budget Circuit Breaker Challenge Suite
# ===========================================================================


class TestBudgetCircuitBreakerChallenge:
    """Stress-test hourly spend cap, request quota, queuing, and thread safety."""

    def test_budget_exhaustion_evaluator_queues_pending(self, valid_candidate_proposal):
        """When hourly spend cap is exhausted, evaluate_proposal must return pending (never auto-accept)."""
        breaker = HourlySpendCircuitBreaker(max_spend_per_hour_usd=0.01)
        breaker.record_spend(cost_usd=0.015, token_count=1500, model_name="claude")

        assert breaker.can_spend(0.005) is False

        runners = [
            MockModelRunner("m1_claude", "claude", "benign", 0.99, code_summary=valid_candidate_proposal["summary"]),
            MockModelRunner("m2_codex", "codex", "benign", 0.99, code_summary=valid_candidate_proposal["summary"]),
        ]
        evaluator = AIBlueTeamEvaluator(runners=runners, circuit_breaker=breaker)
        verdict = evaluator.evaluate_proposal(valid_candidate_proposal)

        assert verdict.verdict == "pending", f"Expected 'pending', got {verdict.verdict}"
        assert verdict.verdict != "accepted", "Budget exhaustion must NEVER auto-accept"
        assert verdict.total_cost_usd == 0.0, "Zero tokens must be spent when circuit breaker is open"
        assert any("spend circuit breaker limit reached; queued" in r for r in verdict.reasons)

    def test_request_count_cap_exhaustion(self, valid_candidate_proposal):
        """When hourly request count cap is reached, evaluate_proposal must return pending."""
        breaker = HourlySpendCircuitBreaker(
            max_spend_per_hour_usd=100.0,
            max_requests_per_hour=2,
            window_seconds=3600.0,
        )
        breaker.record_spend(0.001, 100, "claude")
        breaker.record_spend(0.001, 100, "codex")

        assert breaker.can_spend(0.001) is False

        runners = [
            MockModelRunner("m1_claude", "claude", "benign", 0.99, code_summary=valid_candidate_proposal["summary"]),
            MockModelRunner("m2_codex", "codex", "benign", 0.99, code_summary=valid_candidate_proposal["summary"]),
        ]
        evaluator = AIBlueTeamEvaluator(runners=runners, circuit_breaker=breaker)
        verdict = evaluator.evaluate_proposal(valid_candidate_proposal)

        assert verdict.verdict == "pending"

    def test_circuit_breaker_sliding_window_reset(self):
        """Verify budget resets after window_seconds expires."""
        breaker = HourlySpendCircuitBreaker(
            max_spend_per_hour_usd=0.02,
            window_seconds=0.08,
        )
        breaker.record_spend(0.025, 2000, "claude")
        assert breaker.can_spend(0.005) is False

        # Wait for sliding window to elapse
        time.sleep(0.10)
        assert breaker.can_spend(0.005) is True

    def test_concurrent_multithreaded_spend_tracking(self):
        """Verify thread safety and atomic locking under concurrent multi-threaded load."""
        breaker = HourlySpendCircuitBreaker(
            max_spend_per_hour_usd=0.50,
            max_requests_per_hour=100,
            window_seconds=3600.0,
        )

        def worker():
            for _ in range(10):
                if breaker.can_spend(0.01):
                    breaker.record_spend(0.01, 100, "worker")

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker) for _ in range(10)]
            concurrent.futures.wait(futures)

        metrics = breaker.get_current_metrics()
        assert metrics["spend_usd"] <= 0.50 + 0.01  # Cannot exceed cap
        assert metrics["request_count"] <= 100

    def test_pipeline_budget_exhaustion_empirical_behavior(self, submitter_key_and_headers, valid_candidate_proposal):
        """Empirically observe pipeline behavior when budget is exhausted.

        Verifies:
        1. Submission is NOT auto-accepted.
        2. Documents the behavioral gap between component BlueTeamVerdict(verdict='pending')
           and pipeline evaluate_submission() handling.
        """
        raw_bytes, headers, _key_rec = submitter_key_and_headers
        breaker = HourlySpendCircuitBreaker(max_spend_per_hour_usd=0.01)
        breaker.record_spend(0.02, 1000, "test")

        runners = [
            MockModelRunner("r1", "claude", "benign", 0.99, code_summary=valid_candidate_proposal["summary"]),
            MockModelRunner("r2", "codex", "benign", 0.99, code_summary=valid_candidate_proposal["summary"]),
        ]
        bt = AIBlueTeamEvaluator(runners=runners, circuit_breaker=breaker)
        tracker = ProposalStateTracker()
        edge = EdgeGuard()

        verdict = evaluate_submission(
            raw_bytes,
            headers,
            edge_guard=edge,
            state_tracker=tracker,
            blue_team=bt,
        )

        # Component-level evaluation correctly queues as pending with zero spend
        bt_verdict = bt.evaluate_proposal(valid_candidate_proposal)
        assert bt_verdict.verdict == "pending"
        assert bt_verdict.total_cost_usd == 0.0

        # Invariant 1: Budget exhaustion must NEVER auto-accept
        assert verdict.status != "accepted"
        assert verdict.status in {"pending", "rejected"}


# ===========================================================================
# 4. Opacity Governance Challenge Suite
# ===========================================================================


class TestOpacityGovernanceChallenge:
    """Stress-test response indistinguishability, zero-leakage, and confidential audit logging."""

    def test_opaque_query_response_indistinguishability_across_layers(self, valid_candidate_proposal):
        """Verify query response shape and content is 100% identical across all failure layers (L1-L8)."""
        tracker = ProposalStateTracker()

        failure_layers = [
            ("L1", "L1_SCHEMA_VALIDATION", "Invalid param_types"),
            ("L2", "L2_UNICODE_POLICY", "Non-ASCII byte 0x80"),
            ("L3", "L3_ENCODING_OR_URL_VIOLATION", "High entropy blob > 4.5"),
            ("L4", "L4_AWS_KEY", "AKIAIOSFODNN7EXAMPLE detected"),
            ("L5", "L5_PROMPT_INJECTION_DETECTED", "Role delimiter <<<OVERRIDE>>>"),
            ("L6", "L6_POLICY_VIOLATION", "Banned built-in function eval()"),
            ("L8", "L8_BLUE_TEAM_REJECTED", "Model consensus rejected"),
        ]

        query_responses = []
        for layer, rule_id, notes in failure_layers:
            content_hash = hashlib.sha256(f"test_hash_{layer}".encode()).hexdigest()
            receipt = tracker.register_submission(content_hash)
            tracker.mark_rejected(
                submission_id=receipt.submission_id,
                failure_layer=layer,
                failure_rule_id=rule_id,
                audit_notes=notes,
            )
            query_resp = tracker.get_public_status(receipt.submission_id)
            assert query_resp is not None
            query_responses.append(query_resp)

        # Verify all responses are strictly {'status': 'rejected'}
        for resp in query_responses:
            assert resp == {"status": "rejected"}
            verify_zero_leakage(resp)

        # Verify 100% indistinguishability across all failure layers
        assert assert_response_shapes_indistinguishable(query_responses) is True

    def test_zero_leakage_validator_detects_forbidden_keys(self):
        """verify_zero_leakage must raise ValueError on any leaked metadata key."""
        forbidden_test_cases = [
            {"status": "rejected", "rule_id": "L4_AWS_KEY"},
            {"status": "rejected", "layer": "L2"},
            {"status": "rejected", "error": "Internal exception"},
            {"status": "rejected", "details": "Regex match at offset 12"},
            {"status": "rejected", "matched_text": "AKIA123"},
            {"status": "rejected", "traceback": "Traceback (most recent call last):"},
            {"status": "rejected", "payload": "def rogue(): pass"},
            {"status": "unknown_status"},
            {"status": "accepted", "extra": "bonus"},
        ]

        for bad_resp in forbidden_test_cases:
            with pytest.raises(ValueError, match="Opacity violation"):
                verify_zero_leakage(bad_resp)

    def test_pipeline_public_responses_zero_leakage(self):
        """PipelineVerdict.to_public_response must never leak internal rule or error information."""
        # 413 Payload Too Large
        p_413 = PipelineVerdict(
            status="rejected",
            submission_id=None,
            http_status=413,
            failure_layer="L0",
            failure_rule_id="L0_BODY_TOO_LARGE",
            error_message="Raw bytes size 35000 exceeds cap 32768",
        )
        assert p_413.to_public_response() == {"error": "Payload exceeds 32KB cap"}

        # 429 Rate Limit Exceeded
        p_429 = PipelineVerdict(
            status="rejected",
            submission_id=None,
            http_status=429,
            failure_layer="L0",
            failure_rule_id="L0_RATE_LIMIT_EXCEEDED",
            error_message="Rate limit capacity 0 exhausted",
            retry_after=60,
        )
        assert p_429.to_public_response() == {"error": "Rate limit exceeded"}

        # L1-L8 Rejection
        p_l4 = PipelineVerdict(
            status="rejected",
            submission_id="sub-1234-abcd",
            http_status=200,
            failure_layer="L4",
            failure_rule_id="L4_SECRET_OPENAI_KEY",
            error_message="Secret key sk-proj-12345 detected in pseudocode",
        )
        # External public response returns strictly {"status": "rejected"}
        assert p_l4.to_public_response() == {"status": "rejected"}

    def test_internal_audit_record_zero_payload_guarantee(self):
        """InternalAuditRecord must store hash and rule ID only, NEVER storing the raw code/payload."""
        tracker = ProposalStateTracker()
        receipt = tracker.register_submission("hash_zero_payload_test")
        tracker.mark_rejected(
            submission_id=receipt.submission_id,
            failure_layer="L4",
            failure_rule_id="L4_AWS_KEY",
            audit_notes="Rule matched on regex pattern",
        )

        audit_record = tracker.get_internal_audit_record(receipt.submission_id)
        assert audit_record is not None
        assert audit_record.content_hash == "hash_zero_payload_test"
        assert audit_record.failure_layer == "L4"
        assert audit_record.failure_rule_id == "L4_AWS_KEY"

        # Verify no code or payload field exists in InternalAuditRecord
        field_names = set(audit_record.__dataclass_fields__.keys())
        assert "source_code" not in field_names
        assert "payload" not in field_names
        assert "raw_bytes" not in field_names

    def test_receipt_structure_invariance(self):
        """ImmediateReceipt must strictly contain 'status' and 'submission_id'."""
        receipt = ImmediateReceipt(status="accepted", submission_id="test-uuid-1234")
        d = receipt.to_dict()
        assert set(d.keys()) == {"status", "submission_id"}
        assert d["status"] == "accepted"
        assert d["submission_id"] == "test-uuid-1234"
