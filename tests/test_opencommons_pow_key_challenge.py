"""Empirical Challenge Test Suite: Proof-of-Work and Ed25519 Key Management.

Adversarial stress harness challenging:
1. Bit-level leading zero counting against independent reference oracle.
2. Dynamic difficulty scaling formula ($d = 16..24$) and rejection decay ratchet.
3. Proof-of-Work verifier resistance against forged nonces, corrupted content hashes,
   clock drift (>300s), and invalid difficulty claims.
4. Ed25519 100-keypair uniqueness, true-positive validity, and zero false positives
   across 9,900 cross-key pairs, bit-flipped signatures, and malformed inputs.
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import pytest

from kardashev_cli.opencommons_client import (
    BASE_POW_DIFFICULTY,
    MAX_ADDITIONAL_POW_DIFFICULTY,
    HashcashPoW,
    OpenCommonsKeyManager,
    PoWStamp,
)


@pytest.fixture
def tmp_key_file(tmp_path: Path) -> Path:
    return tmp_path / "challenge_test_key.json"


# ---------------------------------------------------------------------------
# Oracle Helper
# ---------------------------------------------------------------------------

def reference_leading_zeros(digest: bytes) -> int:
    """Independent oracle for bit-level leading zero counting via binary string expansion."""
    if not digest:
        return 0
    bits = "".join(f"{b:08b}" for b in digest)
    zeros = 0
    for bit in bits:
        if bit == "0":
            zeros += 1
        else:
            break
    return zeros


# ===========================================================================
# 1. Bit-Level Leading Zero Oracle Challenge
# ===========================================================================

class TestPoWLeadingZeroBitCheck:
    """Empirically challenge HashcashPoW.count_leading_zero_bits against an independent oracle."""

    def test_single_byte_exhaustive_oracle(self):
        """Test all 256 single-byte values against independent bit-level oracle."""
        for b in range(256):
            digest = bytes([b])
            expected = reference_leading_zeros(digest)
            actual = HashcashPoW.count_leading_zero_bits(digest)
            assert actual == expected, f"Mismatch for byte 0x{b:02x}: expected {expected}, got {actual}"

    def test_two_byte_exhaustive_oracle(self):
        """Test all 65,536 2-byte values against independent bit-level oracle."""
        for b1 in range(256):
            for b2 in range(256):
                digest = bytes([b1, b2])
                expected = reference_leading_zeros(digest)
                actual = HashcashPoW.count_leading_zero_bits(digest)
                assert actual == expected, f"Mismatch for 2-byte 0x{b1:02x}{b2:02x}: expected {expected}, got {actual}"

    def test_boundary_zero_cases(self):
        """Verify boundary cases: empty, all zeros, all ones, single set bit."""
        # Empty bytes
        assert HashcashPoW.count_leading_zero_bits(b"") == 0

        # All zeros (32 bytes = 256 bits)
        assert HashcashPoW.count_leading_zero_bits(b"\x00" * 32) == 256

        # All ones (32 bytes = 0 bits)
        assert HashcashPoW.count_leading_zero_bits(b"\xff" * 32) == 0

        # 255 zeros followed by 1 (last bit set)
        assert HashcashPoW.count_leading_zero_bits(b"\x00" * 31 + b"\x01") == 255

        # 120 zeros followed by 0x80 (15 zero bytes + 1000 0000)
        assert HashcashPoW.count_leading_zero_bits(b"\x00" * 15 + b"\x80" + b"\x00" * 16) == 120

    def test_random_digests_oracle(self):
        """Verify 5,000 random 32-byte SHA-256 style digests against oracle."""
        rng = random.Random(42)
        for _ in range(5000):
            digest = rng.randbytes(32)
            expected = reference_leading_zeros(digest)
            actual = HashcashPoW.count_leading_zero_bits(digest)
            assert actual == expected


# ===========================================================================
# 2. Dynamic Difficulty Scaling & Ratchet Challenge
# ===========================================================================

class TestDynamicDifficultyScaling:
    """Empirically challenge the dynamic difficulty formula and ratchet mechanics."""

    def test_difficulty_formula_range(self):
        """Verify dynamic scaling ratchet: d = 16 + min(rejections, 8) in [16..24]."""
        base = BASE_POW_DIFFICULTY
        assert base == 16
        assert MAX_ADDITIONAL_POW_DIFFICULTY == 8

        # Escalation from 16 to 24
        for r in range(9):
            expected = 16 + r
            actual = HashcashPoW.compute_difficulty(r, base)
            assert actual == expected, f"Rejection count {r} produced difficulty {actual}, expected {expected}"

        # Hard cap at 24 bits for rejections >= 8
        for r in [9, 10, 20, 50, 100, 10000]:
            actual = HashcashPoW.compute_difficulty(r, base)
            assert actual == 24, f"Rejection count {r} exceeded cap: got {actual}"

        # Floor at 16 for negative rejections
        for r in [-1, -5, -100]:
            actual = HashcashPoW.compute_difficulty(r, base)
            assert actual == 16, f"Negative rejection count {r} produced {actual}, expected 16"

    def test_custom_base_difficulty(self):
        """Verify dynamic difficulty with non-standard base."""
        assert HashcashPoW.compute_difficulty(rejection_count=3, base_difficulty=20) == 23
        assert HashcashPoW.compute_difficulty(rejection_count=10, base_difficulty=20) == 28

    def test_key_manager_rejection_ratchet_and_decay(self, tmp_key_file):
        """Verify KeyManager tracks rejections, caps difficulty, and decays after 24h."""
        record = OpenCommonsKeyManager.load_or_create(tmp_key_file)
        assert record.rejection_count == 0

        # Increment through full range
        for expected_count in range(1, 10):
            count = OpenCommonsKeyManager.record_rejection(tmp_key_file)
            assert count == expected_count

        # Acceptance decrements count
        OpenCommonsKeyManager.record_acceptance(tmp_key_file)
        updated = OpenCommonsKeyManager.load_or_create(tmp_key_file)
        assert updated.rejection_count == 8
        assert updated.submission_count == 1

        # Inactivity decay (> 24 hours = 86,400s)
        updated.last_rejected_at = int(time.time()) - 90000
        OpenCommonsKeyManager.save(updated, tmp_key_file)

        # Next rejection resets count to 1
        new_count = OpenCommonsKeyManager.record_rejection(tmp_key_file)
        assert new_count == 1


# ===========================================================================
# 3. Proof-of-Work Verification Adversarial Challenges
# ===========================================================================

class TestPoWVerificationAdversarialChallenges:
    """Empirically test forged nonces, corrupted content hashes, clock drift, and invalid parameters."""

    @pytest.fixture(autouse=True)
    def setup_challenge(self):
        self.content_hash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        self.difficulty = 12  # Fast difficulty for deterministic test execution
        self.now = int(time.time())
        self.valid_stamp = HashcashPoW.solve(
            self.content_hash,
            difficulty=self.difficulty,
            timestamp=self.now,
        )

    def test_forged_nonces_rejected(self):
        """Challenge verifier with forged nonces: off-by-one, zero, negative, large, non-digits."""
        valid_nonce_int = int(self.valid_stamp.nonce)
        forged_nonces = [
            str(valid_nonce_int + 1),
            str(valid_nonce_int - 1) if valid_nonce_int > 0 else "999999",
            "0",
            "1",
            "999999999",
            "-1",
            "abc_nonce",
            "00" + self.valid_stamp.nonce,
            self.valid_stamp.nonce + " ",
            "",
        ]

        for forged in forged_nonces:
            bad_stamp = PoWStamp(
                nonce=forged,
                timestamp=self.valid_stamp.timestamp,
                difficulty=self.valid_stamp.difficulty,
                content_hash=self.valid_stamp.content_hash,
            )
            valid, err = HashcashPoW.verify(bad_stamp, self.content_hash, min_difficulty=self.difficulty)
            assert not valid, f"Forged nonce '{forged}' was unexpectedly accepted!"
            assert err is not None
            assert "Invalid PoW solution" in err

    def test_corrupted_content_hashes_rejected(self):
        """Challenge verifier with corrupted content hashes: 1-char flip, truncation, casing, empty."""
        corrupted_hashes = [
            self.content_hash[:-1] + ("0" if self.content_hash[-1] != "0" else "1"),
            self.content_hash[:-1],
            self.content_hash + "a",
            "",
            "f" * 64,
            self.content_hash.upper(),  # Casing divergence
            "0" * 64,
        ]

        for corrupted in corrupted_hashes:
            # Hash mismatch between stamp and passed argument
            valid, err = HashcashPoW.verify(self.valid_stamp, corrupted, min_difficulty=self.difficulty)
            assert not valid, f"Corrupted hash '{corrupted}' was unexpectedly accepted!"
            assert "Content hash mismatch" in err

            # Corrupted stamp content hash
            tampered_stamp = PoWStamp(
                nonce=self.valid_stamp.nonce,
                timestamp=self.valid_stamp.timestamp,
                difficulty=self.valid_stamp.difficulty,
                content_hash=corrupted,
            )
            valid_t, _err_t = HashcashPoW.verify(tampered_stamp, corrupted, min_difficulty=self.difficulty)
            assert not valid_t, "Tampered stamp with corrupted hash passed verification!"

    def test_clock_drift_future_rejected(self):
        """Challenge verifier with timestamps drifting into future (>300s)."""
        future_offsets = [301, 350, 600, 3600, 86400, 1000000]
        for offset in future_offsets:
            current_now = int(time.time())
            future_ts = current_now + offset
            stamp = HashcashPoW.solve(self.content_hash, difficulty=self.difficulty, timestamp=future_ts)
            valid, err = HashcashPoW.verify(stamp, self.content_hash, min_difficulty=self.difficulty)
            assert not valid, f"Future stamp at +{offset}s was accepted!"
            assert "clock skew" in err.lower() or "expired" in err.lower()

    def test_clock_drift_past_rejected(self):
        """Challenge verifier with timestamps drifting into past (>300s)."""
        past_offsets = [301, 350, 600, 3600, 86400]
        for offset in past_offsets:
            current_now = int(time.time())
            past_ts = current_now - offset
            stamp = HashcashPoW.solve(self.content_hash, difficulty=self.difficulty, timestamp=past_ts)
            valid, err = HashcashPoW.verify(stamp, self.content_hash, min_difficulty=self.difficulty)
            assert not valid, f"Past stamp at -{offset}s was accepted!"
            assert "clock skew" in err.lower() or "expired" in err.lower()

    def test_clock_drift_boundaries(self):
        """Verify boundary behavior at exactly +/-300s and +/-299s."""
        # +/- 299s must pass
        for offset in [299, -299]:
            current_now = int(time.time())
            ts = current_now + offset
            stamp = HashcashPoW.solve(self.content_hash, difficulty=self.difficulty, timestamp=ts)
            valid, err = HashcashPoW.verify(stamp, self.content_hash, min_difficulty=self.difficulty)
            assert valid is True, f"Boundary offset {offset}s failed: {err}"

        # Exactly +/- 300s (tolerance is abs(now - ts) > 300, so 300 is allowed)
        for offset in [300, -300]:
            current_now = int(time.time())
            ts = current_now + offset
            stamp = HashcashPoW.solve(self.content_hash, difficulty=self.difficulty, timestamp=ts)
            valid, err = HashcashPoW.verify(stamp, self.content_hash, min_difficulty=self.difficulty)
            assert valid is True, f"Boundary offset {offset}s failed: {err}"

    def test_invalid_difficulty_parameters(self):
        """Challenge verifier with invalid difficulties: lower than min, zero, negative, fake claims."""
        # 1. Lower difficulty than required
        low_stamp = HashcashPoW.solve(self.content_hash, difficulty=10, timestamp=self.now)
        valid, err = HashcashPoW.verify(low_stamp, self.content_hash, min_difficulty=12)
        assert not valid
        assert "Insufficient difficulty: 10 < 12" in err

        # 2. Fake difficulty claim (stamp claims 16, but was solved at 12)
        faked_stamp = PoWStamp(
            nonce=self.valid_stamp.nonce,
            timestamp=self.valid_stamp.timestamp,
            difficulty=16,
            content_hash=self.valid_stamp.content_hash,
        )
        valid, err = HashcashPoW.verify(faked_stamp, self.content_hash, min_difficulty=16)
        assert not valid
        assert "leading zeros < required 16" in err

        # 3. Zero difficulty stamp with standard min_difficulty
        zero_stamp = PoWStamp(
            nonce="0",
            timestamp=self.now,
            difficulty=0,
            content_hash=self.content_hash,
        )
        valid, err = HashcashPoW.verify(zero_stamp, self.content_hash, min_difficulty=16)
        assert not valid
        assert "Insufficient difficulty: 0 < 16" in err

        # 4. Negative difficulty stamp
        neg_stamp = PoWStamp(
            nonce="0",
            timestamp=self.now,
            difficulty=-1,
            content_hash=self.content_hash,
        )
        valid, err = HashcashPoW.verify(neg_stamp, self.content_hash, min_difficulty=16)
        assert not valid
        assert "Insufficient difficulty: -1 < 16" in err

    def test_malformed_stamp_strings(self):
        """Challenge PoW parsing and verification with malformed wire formats."""
        malformed_stamps = [
            "",
            "single_token",
            "two:tokens",
            "three:tokens:16",
            "too:many:tokens:16:extra:hash",
            "nonce:not_an_int:16:hash",
            "nonce:1700000000:not_diff:hash",
            ":::",
            "   ",
        ]
        for bad in malformed_stamps:
            valid, err = HashcashPoW.verify(bad, self.content_hash, min_difficulty=self.difficulty)
            assert not valid
            assert err is not None


# ===========================================================================
# 4. Ed25519 100-Keypair Stress Test & Zero False Positives
# ===========================================================================

class TestEd25519KeyStressAndFalsePositiveResistance:
    """Stress test 100 random Ed25519 keypairs for uniqueness and zero signature false positives."""

    def test_100_keypairs_uniqueness_and_validity(self):
        """Generate 100 keypairs and verify strictly 100 unique public and private keys."""
        keypairs = [OpenCommonsKeyManager.generate_key_record() for _ in range(100)]

        pub_keys = [k.public_key_hex for k in keypairs]
        priv_keys = [k.private_key_hex for k in keypairs]

        # 100 unique public keys
        assert len(set(pub_keys)) == 100, f"Public key collision detected! Unique: {len(set(pub_keys))}"
        # 100 unique private keys
        assert len(set(priv_keys)) == 100, f"Private key collision detected! Unique: {len(set(priv_keys))}"

        # Verify length and hex validity
        for k in keypairs:
            assert len(k.public_key_hex) == 64
            assert len(k.private_key_hex) == 64
            assert bytes.fromhex(k.public_key_hex)
            assert bytes.fromhex(k.private_key_hex)

    def test_100_keypairs_true_positive_signatures(self):
        """Verify all 100 keypairs produce valid verifiable signatures on distinct messages."""
        keypairs = [OpenCommonsKeyManager.generate_key_record() for _ in range(100)]
        for i, k in enumerate(keypairs):
            msg = f"opencommons_proposal_content_hash_{i}_{k.public_key_hex[:8]}".encode("ascii")
            sig = OpenCommonsKeyManager.sign(msg, k)
            assert len(sig) == 128
            valid = OpenCommonsKeyManager.verify(k.public_key_hex, msg, sig)
            assert valid is True, f"Keypair {i} failed to verify its own signature!"

    def test_9900_cross_key_signature_rejection(self):
        """Challenge all 9,900 cross-key pairs (i != j): zero false positive signature verifications."""
        keypairs = [OpenCommonsKeyManager.generate_key_record() for _ in range(100)]
        messages = [f"payload_{i}".encode("ascii") for i in range(100)]
        signatures = [OpenCommonsKeyManager.sign(messages[i], keypairs[i]) for i in range(100)]

        false_positives = 0
        total_cross_checks = 0

        for i in range(100):
            pub_i = keypairs[i].public_key_hex
            for j in range(100):
                if i == j:
                    continue
                # Key i attempts to verify signature j on message j
                if OpenCommonsKeyManager.verify(pub_i, messages[j], signatures[j]):
                    false_positives += 1
                total_cross_checks += 1

        assert total_cross_checks == 9900
        assert false_positives == 0, f"Detected {false_positives} cross-key false positives out of 9900!"

    def test_signature_bit_flip_tampering(self):
        """Challenge verifier by flipping bits in signatures across 100 keypairs: zero false positives."""
        keypairs = [OpenCommonsKeyManager.generate_key_record() for _ in range(100)]
        for i, k in enumerate(keypairs):
            msg = f"tamper_test_payload_{i}".encode("ascii")
            sig_hex = OpenCommonsKeyManager.sign(msg, k)
            sig_bytes = bytearray(bytes.fromhex(sig_hex))

            # Flip bits at first byte, middle byte, and last byte
            for pos in [0, 16, 32, 48, 63]:
                corrupted = bytearray(sig_bytes)
                corrupted[pos] ^= 0x01
                assert OpenCommonsKeyManager.verify(k.public_key_hex, msg, corrupted.hex()) is False

    def test_message_tampering(self):
        """Challenge verifier by altering message bytes across 100 keypairs: zero false positives."""
        keypairs = [OpenCommonsKeyManager.generate_key_record() for _ in range(100)]
        for i, k in enumerate(keypairs):
            msg = f"authentic_payload_{i}".encode("ascii")
            sig_hex = OpenCommonsKeyManager.sign(msg, k)

            tampered_msg = f"authentic_payload_{i}_tampered".encode("ascii")
            assert OpenCommonsKeyManager.verify(k.public_key_hex, tampered_msg, sig_hex) is False

            bit_flipped_msg = bytearray(msg)
            bit_flipped_msg[0] ^= 0x01
            assert OpenCommonsKeyManager.verify(k.public_key_hex, bytes(bit_flipped_msg), sig_hex) is False

    def test_malformed_key_and_signature_robustness(self):
        """Verify robust rejection of malformed, non-hex, odd-length, or wrong-length keys and signatures."""
        record = OpenCommonsKeyManager.generate_key_record()
        valid_msg = b"sample_message"
        valid_sig = OpenCommonsKeyManager.sign(valid_msg, record)

        adversarial_inputs = [
            ("", valid_msg, valid_sig),
            ("not_hex_chars", valid_msg, valid_sig),
            ("123", valid_msg, valid_sig),  # Odd length
            (record.public_key_hex[:-2], valid_msg, valid_sig),  # 31 bytes
            (record.public_key_hex + "00", valid_msg, valid_sig),  # 33 bytes
            ("00" * 32, valid_msg, valid_sig),  # Null bytes public key
            ("ff" * 32, valid_msg, valid_sig),  # Invalid curve point
            (record.public_key_hex, valid_msg, ""),
            (record.public_key_hex, valid_msg, "bad_hex"),
            (record.public_key_hex, valid_msg, valid_sig[:-2]),
            (record.public_key_hex, valid_msg, valid_sig + "00"),
            (record.public_key_hex, valid_msg, "00" * 64),
            (record.public_key_hex, "not_bytes", valid_sig),  # Type safety
            (None, valid_msg, valid_sig),
            (record.public_key_hex, valid_msg, None),
        ]

        for pub, m, s in adversarial_inputs:
            assert OpenCommonsKeyManager.verify(pub, m, s) is False
