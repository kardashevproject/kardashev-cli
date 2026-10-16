"""Unit and integration test suite for OpenCommons Client SDK.

Covers:
1. Anonymous Ed25519 key management, atomic 0o600 file persistence, signing and verification.
2. Dynamic Hashcash Proof-of-Work solver, bit-level verifier, clock skew, and scaling ratchet.
3. AST concept extraction, nesting depth check, docstring/comment stripping, JCS canonicalization.
4. Universal client submission with zero-IP HTTP headers, mock responses, and background threading.
"""

from __future__ import annotations

import ast
import io
import json
import os
import threading
import time
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kardashev_cli.opencommons_client import (
    ConceptExtractor,
    HashcashPoW,
    OpenCommonsClient,
    OpenCommonsKeyManager,
    PoWStamp,
    Proposal,
)


@pytest.fixture
def tmp_key_file(tmp_path: Path) -> Path:
    return tmp_path / "test_opencommons_key.json"


@pytest.fixture
def sample_python_code() -> str:
    return '''# Module comments that should be stripped
def fast_power(base: float, exponent: int) -> float:
    """Computes base raised to exponent using binary exponentiation.

    Invariants:
        - exponent must be non-negative
    """
    # Inline algorithm comment
    result: float = 1.0
    current: float = base
    p: int = exponent
    while p > 0:
        if p % 2 == 1:
            result *= current
        current *= current
        p //= 2
    return result
'''


# ===========================================================================
# Group 1: Key Management Tests
# ===========================================================================

class TestOpenCommonsKeyManager:
    def test_key_generation_and_storage(self, tmp_key_file):
        record = OpenCommonsKeyManager.load_or_create(tmp_key_file)
        assert len(record.public_key_hex) == 64
        assert len(record.private_key_hex) == 64
        assert tmp_key_file.is_file()

        mode = os.stat(tmp_key_file).st_mode & 0o777
        assert mode == 0o600

    def test_key_persistence(self, tmp_key_file):
        r1 = OpenCommonsKeyManager.load_or_create(tmp_key_file)
        r2 = OpenCommonsKeyManager.load_or_create(tmp_key_file)
        assert r1.public_key_hex == r2.public_key_hex
        assert r1.private_key_hex == r2.private_key_hex

    def test_key_signing_and_verification(self, tmp_key_file):
        record = OpenCommonsKeyManager.load_or_create(tmp_key_file)
        msg = b"canonical_content_hash_sample_bytes"
        sig_hex = OpenCommonsKeyManager.sign(msg, record)
        assert len(sig_hex) == 128  # 64 bytes = 128 hex chars

        valid = OpenCommonsKeyManager.verify(record.public_key_hex, msg, sig_hex)
        assert valid is True

        tampered_msg = b"tampered_payload_bytes"
        assert OpenCommonsKeyManager.verify(record.public_key_hex, tampered_msg, sig_hex) is False

        bad_sig = "00" * 64
        assert OpenCommonsKeyManager.verify(record.public_key_hex, msg, bad_sig) is False

    def test_rejection_tracking_and_decay(self, tmp_key_file):
        r = OpenCommonsKeyManager.load_or_create(tmp_key_file)
        assert r.rejection_count == 0

        c1 = OpenCommonsKeyManager.record_rejection(tmp_key_file)
        assert c1 == 1

        c2 = OpenCommonsKeyManager.record_rejection(tmp_key_file)
        assert c2 == 2

        # Acceptance decreases rejection count
        OpenCommonsKeyManager.record_acceptance(tmp_key_file)
        r = OpenCommonsKeyManager.load_or_create(tmp_key_file)
        assert r.rejection_count == 1
        assert r.submission_count == 1

        # Test decay after > 24 hours
        r.last_rejected_at = int(time.time()) - 90000
        OpenCommonsKeyManager.save(r, tmp_key_file)

        c3 = OpenCommonsKeyManager.record_rejection(tmp_key_file)
        assert c3 == 1  # Reset to 1 after decay


# ===========================================================================
# Group 2: Proof of Work Tests
# ===========================================================================

class TestHashcashPoW:
    def test_leading_zero_bits_counting(self):
        assert HashcashPoW.count_leading_zero_bits(b"\x00\x00\x00") == 24
        assert HashcashPoW.count_leading_zero_bits(b"\x00\x80") == 8
        assert HashcashPoW.count_leading_zero_bits(b"\x00\x40") == 9
        assert HashcashPoW.count_leading_zero_bits(b"\x00\x01") == 15
        assert HashcashPoW.count_leading_zero_bits(b"\x80") == 0
        assert HashcashPoW.count_leading_zero_bits(b"\x01") == 7

    def test_stamp_formatting_and_parsing(self):
        stamp = PoWStamp(nonce="12345", timestamp=1700000000, difficulty=16, content_hash="abcd1234")
        formatted = stamp.format()
        assert formatted == "12345:1700000000:16:abcd1234"

        parsed = PoWStamp.parse(formatted)
        assert parsed == stamp

        with pytest.raises(ValueError):
            PoWStamp.parse("bad:stamp:format")

    def test_solve_and_verify(self):
        content_hash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        diff = 12  # Fast difficulty for deterministic test execution
        stamp = HashcashPoW.solve(content_hash, difficulty=diff)

        assert stamp.difficulty == diff
        assert stamp.content_hash == content_hash

        valid, err = HashcashPoW.verify(stamp, content_hash, min_difficulty=diff)
        assert valid is True
        assert err is None

        # Content hash mismatch
        valid, err = HashcashPoW.verify(stamp, "other_hash", min_difficulty=diff)
        assert valid is False
        assert "Content hash mismatch" in err

        # Insufficient difficulty
        valid, err = HashcashPoW.verify(stamp, content_hash, min_difficulty=diff + 10)
        assert valid is False
        assert "Insufficient difficulty" in err

        # Clock skew expiration (>300s)
        expired_stamp = PoWStamp(
            nonce=stamp.nonce,
            timestamp=stamp.timestamp - 400,
            difficulty=stamp.difficulty,
            content_hash=stamp.content_hash,
        )
        valid, err = HashcashPoW.verify(expired_stamp, content_hash, min_difficulty=diff)
        assert valid is False
        assert "clock skew" in err

    def test_dynamic_difficulty_scaling(self):
        base = 16
        assert HashcashPoW.compute_difficulty(0, base) == 16
        assert HashcashPoW.compute_difficulty(1, base) == 17
        assert HashcashPoW.compute_difficulty(5, base) == 21
        assert HashcashPoW.compute_difficulty(8, base) == 24
        assert HashcashPoW.compute_difficulty(20, base) == 24  # Capped at base + 8


# ===========================================================================
# Group 3: Concept Extractor Tests
# ===========================================================================

class TestConceptExtractor:
    def test_nesting_depth_guard(self):
        shallow = "def foo():\n    if True:\n        return 1\n"
        assert ConceptExtractor.check_nesting_depth(shallow, max_depth=5) is True

        deep_brackets = "((((((1))))))"
        assert ConceptExtractor.check_nesting_depth(deep_brackets, max_depth=5) is False

        deep_indent = "def f():\n" + "    " * 25 + "pass\n"
        assert ConceptExtractor.check_nesting_depth(deep_indent, max_depth=20) is False

    def test_strip_comments_and_docstrings(self, sample_python_code):
        tree = ast.parse(sample_python_code)
        func_node = tree.body[0]
        assert isinstance(func_node, ast.FunctionDef)

        canonical, doc = ConceptExtractor.strip_comments_and_docstrings(func_node)
        assert "Computes base raised to exponent" in doc
        assert "Module comments" not in canonical
        assert "Inline algorithm comment" not in canonical
        assert "Computes base raised to exponent" not in canonical
        assert canonical.startswith("def fast_power(base: float, exponent: int) -> float:")

    def test_candidate_filtering(self):
        # Normal candidate
        assert ConceptExtractor.is_candidate_function(
            ast.parse("def add(a: int, b: int) -> int:\n    return a + b").body[0]
        ) is True

        # Method with self
        assert ConceptExtractor.is_candidate_function(
            ast.parse("def method(self, x: int) -> int:\n    return x").body[0]
        ) is False

        # Generator with yield
        assert ConceptExtractor.is_candidate_function(
            ast.parse("def gen(n: int) -> int:\n    yield n").body[0]
        ) is False

        # Missing return annotation
        assert ConceptExtractor.is_candidate_function(
            ast.parse("def fn(a: int):\n    return a").body[0]
        ) is False

        # Forbidden call
        assert ConceptExtractor.is_candidate_function(
            ast.parse("def evil(s: str) -> None:\n    eval(s)").body[0]
        ) is False

    def test_extract_from_source(self, sample_python_code):
        proposals = ConceptExtractor.extract_from_source(sample_python_code, default_domain="numerical_analysis")
        assert len(proposals) == 1
        prop = proposals[0]

        assert prop.name == "fast_power"
        assert prop.param_types == ["float", "int"]
        assert prop.return_type == "float"
        assert "Computes base raised to exponent" in prop.summary
        assert prop.domain == "numerical_analysis"
        assert prop.source_code.startswith("def fast_power(base: float, exponent: int) -> float:")
        assert "#" not in prop.source_code

    def test_canonicalize_json_and_content_hash(self):
        d1 = {"b": 2, "a": 1}
        d2 = {"a": 1, "b": 2}
        assert ConceptExtractor.canonicalize_json(d1) == b'{"a":1,"b":2}'
        assert ConceptExtractor.compute_content_hash(d1) == ConceptExtractor.compute_content_hash(d2)


# ===========================================================================
# Group 4: Client Submission Tests
# ===========================================================================

class TestOpenCommonsClient:
    def test_client_submit_accepted(self, tmp_key_file):
        client = OpenCommonsClient(
            hub_url="https://mock.commons.org",
            key_path=tmp_key_file,
        )

        proposal = Proposal(
            name="test_func",
            param_types=["int"],
            return_type="int",
            summary="Test function summary for testing client.",
            pseudocode="Return arg unchanged.",
            invariants=["Identity"],
            pitfalls=[],
            source_code="def test_func(x: int) -> int:\n    return x\n",
            domain="general",
        )

        mock_resp = MagicMock()
        mock_resp.status = 202
        mock_resp.read.return_value = json.dumps({"status": "accepted", "submission_id": "sub_123"}).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen, \
             patch("kardashev_cli.opencommons_client.HashcashPoW.solve") as mock_solve:
            mock_solve.return_value = PoWStamp("0", int(time.time()), 16, "hash")

            receipt = client.submit(proposal)

            assert receipt.status == "accepted"
            assert receipt.submission_id == "sub_123"

            # Verify request headers
            req = mock_urlopen.call_args[0][0]
            assert req.get_header("Content-type") == "application/json"
            assert len(req.get_header("X-opencommons-key")) == 64
            assert req.get_header("X-opencommons-pow") is not None
            assert len(req.get_header("X-opencommons-signature")) == 128

    def test_client_submit_rate_limited(self, tmp_key_file):
        client = OpenCommonsClient(
            hub_url="https://mock.commons.org",
            key_path=tmp_key_file,
        )
        proposal = {
            "name": "rate_limited_func",
            "param_types": [],
            "return_type": "None",
            "summary": "Rate limited placeholder summary.",
            "pseudocode": "No-op.",
            "invariants": [],
            "pitfalls": [],
            "source_code": "def rate_limited_func() -> None:\n    pass\n",
            "domain": "general",
        }

        fp = io.BytesIO(b'{"error": "Too Many Requests"}')
        headers = {"Retry-After": "45"}
        err = urllib.error.HTTPError("https://mock.commons.org/v1/proposals", 429, "Rate Limited", headers, fp)

        with patch("urllib.request.urlopen", side_effect=err), \
             patch("kardashev_cli.opencommons_client.HashcashPoW.solve") as mock_solve:
            mock_solve.return_value = PoWStamp("0", int(time.time()), 16, "hash")

            receipt = client.submit(proposal)
            assert receipt.status == "rate_limited"
            assert receipt.retry_after == 45

    def test_client_submit_background(self, tmp_key_file):
        client = OpenCommonsClient(
            hub_url="https://mock.commons.org",
            key_path=tmp_key_file,
        )
        proposal = {
            "name": "bg_func",
            "param_types": [],
            "return_type": "None",
            "summary": "Background test summary.",
            "pseudocode": "No-op.",
            "invariants": [],
            "pitfalls": [],
            "source_code": "def bg_func() -> None:\n    pass\n",
            "domain": "general",
        }

        mock_resp = MagicMock()
        mock_resp.status = 202
        mock_resp.read.return_value = json.dumps({"status": "accepted", "submission_id": "bg_id"}).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp

        callback_event = threading.Event()
        receipt_box = []

        def callback(r):
            receipt_box.append(r)
            callback_event.set()

        with patch("urllib.request.urlopen", return_value=mock_resp), \
             patch("kardashev_cli.opencommons_client.HashcashPoW.solve") as mock_solve:
            mock_solve.return_value = PoWStamp("0", int(time.time()), 16, "hash")

            thread = client.submit_background(proposal, callback=callback)
            thread.join(timeout=2.0)

            assert callback_event.wait(timeout=2.0)
            assert len(receipt_box) == 1
            assert receipt_box[0].status == "accepted"
            assert receipt_box[0].submission_id == "bg_id"
