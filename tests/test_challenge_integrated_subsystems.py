"""Empirical Adversarial Challenge Suite for Integrated OpenCommons Subsystems.

Conducted by challenger_cons (Empirical Challenger):
1. Sandbox Containment Empirical Attacks:
   - CPU timeout containment (infinite busy loop)
   - Memory limit containment (RLIMIT_AS <= 256MB)
   - Fork bomb containment (RLIMIT_NPROC <= 1)
   - File write bomb containment (RLIMIT_FSIZE <= 0)
   - Network socket creation attempt
   - Non-deterministic return value detection
   - In-place argument mutation detection
   - Stdout tampering / fake JSON injection
2. Storage & Transparency Integrity:
   - SQLite WAL concurrency (multithreaded readers/writers)
   - Monotonic sequencer integrity under rapid concurrent appends
   - C2SP tlog tiles & Merkle root consistency
   - Ed25519 signed checkpoint verification & tamper detection
   - Live SSE stream replay and real-time event broadcast
   - Legacy sync cutover (HTTP 410 Gone on /v1/sync)
3. Frontier Agent Integration Handshake:
   - Live stdio JSON-RPC handshake verification
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from aiohttp import test_utils

from kardashev_cli.integrate import (
    check_integrations,
    perform_mcp_handshake,
)
from kardashev_cli.opencommons_hub import (
    HUB_APP_KEY,
    OpenCommonsHub,
    create_hub_app,
)
from kardashev_cli.opencommons_sandbox import (
    SandboxConfig,
    SandboxRunner,
)
from kardashev_cli.opencommons_storage import (
    ConceptRecord,
    SQLiteConceptStorage,
    verify_checkpoint_signature,
)

# ===========================================================================
# 1. Sandbox Containment Empirical Adversarial Attacks
# ===========================================================================


class TestSandboxContainmentAttacks:
    """Adversarial attacks attempting to violate isolated subprocess boundaries."""

    @pytest.fixture(autouse=True)
    def setup_runner(self):
        self.runner = SandboxRunner(
            SandboxConfig(
                as_limit_bytes=256 * 1024 * 1024,  # 256 MB
                cpu_limit_seconds=1,  # 1s CPU
                wall_timeout_seconds=2.0,  # 2s wall timeout
                fsize_limit_bytes=0,
                nproc_limit=1,
            )
        )

    def test_attack_cpu_timeout_busy_loop(self):
        """Infinite busy loop must be contained within wall timeout and report TIMEOUT."""
        source = """\
def infinite_busy_loop(x: int) -> int:
    while True:
        pass
"""
        t0 = time.perf_counter()
        res = self.runner.run_sandboxed_probes(
            source_code=source,
            func_name="infinite_busy_loop",
            probes=[(42,)],
            timeout_seconds=1.5,
        )
        elapsed = time.perf_counter() - t0

        assert res.success is False
        assert res.error_code == "TIMEOUT"
        assert elapsed < 3.0, f"CPU timeout hung for {elapsed:.2f}s!"

    def test_attack_memory_limit_exhaustion(self):
        """Attempt to allocate 500MB must trigger OOM/CRASH within RLIMIT_AS 256MB."""
        source = """\
def memory_exhaustion_attack(x: int) -> int:
    # Attempt to allocate 500MB in virtual address space
    blob = bytearray(500 * 1024 * 1024)
    return len(blob)
"""
        res = self.runner.run_sandboxed_probes(
            source_code=source,
            func_name="memory_exhaustion_attack",
            probes=[(1,)],
            timeout_seconds=3.0,
        )
        assert res.success is False
        assert res.error_code in ("OOM", "CRASH", "SANDBOX_VIOLATION")

    def test_attack_fork_bomb(self):
        """Attempt to fork child processes must be blocked by RLIMIT_NPROC / sandbox policy."""
        source = """\
def fork_bomb_attack(x: int) -> int:
    import os
    for _ in range(50):
        os.fork()
    return x
"""
        res = self.runner.run_sandboxed_probes(
            source_code=source,
            func_name="fork_bomb_attack",
            probes=[(0,)],
            inject_modules=["os"],
            timeout_seconds=3.0,
        )
        assert res.success is False
        # Child process cannot fork or raises BlockingIOError/OSError
        assert res.error_code in ("CRASH", "SANDBOX_VIOLATION")

    def test_attack_file_write_bomb(self):
        """Attempt to write files must trigger SIGXFSZ or OSError due to RLIMIT_FSIZE=0."""
        source = """\
def file_write_attack(x: int) -> int:
    with open('/tmp/opencommons_adversarial_test.bin', 'wb') as f:
        f.write(b'EVIL' * 10000)
    return x
"""
        res = self.runner.run_sandboxed_probes(
            source_code=source,
            func_name="file_write_attack",
            probes=[(10,)],
            inject_builtins=["open"],
            timeout_seconds=3.0,
        )
        assert res.success is False
        assert res.error_code in ("SANDBOX_VIOLATION", "CRASH")
        # Ensure file was not created or has 0 bytes
        test_file = Path("/tmp/opencommons_adversarial_test.bin")
        if test_file.exists():
            assert test_file.stat().st_size == 0
            test_file.unlink(missing_ok=True)

    def test_attack_network_socket_creation(self):
        """Attempt to open network sockets must fail via socket monkeypatch or namespace."""
        source = """\
def network_socket_attack(x: int) -> int:
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(('1.1.1.1', 80))
    return x
"""
        res = self.runner.run_sandboxed_probes(
            source_code=source,
            func_name="network_socket_attack",
            probes=[(1,)],
            inject_modules=["socket"],
            timeout_seconds=3.0,
        )
        assert res.success is False
        assert res.error_code in ("CRASH", "SANDBOX_VIOLATION")

    def test_attack_nondeterminism_detection(self):
        """Function returning non-deterministic results between runs must be detected and rejected."""
        source = """\
def stateful_flaky(x: int, state: list = [0]) -> int:
    state[0] += 1
    return x + state[0]
"""
        res = self.runner.run_sandboxed_probes(
            source_code=source,
            func_name="stateful_flaky",
            probes=[(10,)],
            timeout_seconds=3.0,
        )
        assert res.success is False
        assert res.is_deterministic is False
        assert res.error_code == "DISCREPANCY"

    def test_attack_argument_mutation_detection(self):
        """Function mutating input arguments in place must be detected."""
        source = """\
def argument_mutator(lst: list) -> int:
    lst.append(999)
    return len(lst)
"""
        res = self.runner.run_sandboxed_probes(
            source_code=source,
            func_name="argument_mutator",
            probes=[([1, 2, 3],)],
            timeout_seconds=3.0,
        )
        assert res.success is True
        assert len(res.probe_results) == 1
        assert res.probe_results[0].mutated_args is True

    def test_attack_stdout_tampering_spoofed_json(self):
        """Code writing fake JSON to stdout must NOT spoof sandbox verification."""
        source = """\
def stdout_tampering_attack(x: int) -> int:
    # Attacker tries to inject spoofed JSON to stdout
    print('{"nonce": "fake_nonce", "success": true, "outputs": [999]}')
    return x * 2
"""
        res = self.runner.run_sandboxed_probes(
            source_code=source,
            func_name="stdout_tampering_attack",
            probes=[(21,)],
            timeout_seconds=3.0,
        )
        # Because stdout is redirected to /dev/null and authentic communication
        # uses private pipe with authentic nonce, the actual computation succeeds
        # without accepting the spoofed output!
        assert res.success is True
        assert res.outputs == [42]

    def test_attack_output_size_cap(self):
        """Attempt to flood parent with massive output (>64KB) must be blocked."""
        source = """\
def output_flood_attack(x: int) -> str:
    return "A" * (128 * 1024)
"""
        res = self.runner.run_sandboxed_probes(
            source_code=source,
            func_name="output_flood_attack",
            probes=[(1,)],
            timeout_seconds=3.0,
        )
        assert res.success is False
        assert res.error_code in ("SANDBOX_VIOLATION", "CRASH")

    def test_attack_recursion_limit_stack_overflow(self):
        """Infinite recursion must be caught as RecursionError and contained without crashing host."""
        source = """\
def recursion_bomb(x: int) -> int:
    return recursion_bomb(x + 1)
"""
        res = self.runner.run_sandboxed_probes(
            source_code=source,
            func_name="recursion_bomb",
            probes=[(0,)],
            timeout_seconds=3.0,
        )
        assert res.success is False
        assert res.error_code == "CRASH"



# ===========================================================================
# 2. Storage & Transparency Integrity Tests
# ===========================================================================


class TestStorageAndTransparencyIntegrity:
    """Stress tests on SQLite WAL concurrency, sequencer, tlog tiles, and checkpoints."""

    @pytest.fixture
    def temp_storage(self, tmp_path: Path) -> SQLiteConceptStorage:
        db_path = tmp_path / "test_commons.db"
        return SQLiteConceptStorage(db_path=db_path)

    def test_sqlite_wal_concurrent_readers_and_writers(self, temp_storage: SQLiteConceptStorage):
        """Verify high concurrency: multiple parallel writers and readers without database locked errors."""
        num_writers = 8
        items_per_writer = 15
        num_readers = 10

        errors: list[Exception] = []

        def writer_task(writer_id: int):
            try:
                for i in range(items_per_writer):
                    c_hash = f"writer_{writer_id}_hash_{i:04d}"
                    body = {
                        "name": f"func_{writer_id}_{i}",
                        "domain": "numerical",
                        "summary": f"Concurrent test concept {writer_id}-{i}",
                    }
                    record = ConceptRecord(
                        content_hash=c_hash,
                        name=f"func_{writer_id}_{i}",
                        domain="numerical",
                        json_body=body,
                    )
                    temp_storage.append_accepted_concept(record)
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        def reader_task():
            try:
                for _ in range(30):
                    temp_storage.find_concepts(query="concept", limit=20)
                    temp_storage.get_latest_checkpoint()
                    time.sleep(0.005)
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_writers + num_readers) as executor:
            futures = []
            for w in range(num_writers):
                futures.append(executor.submit(writer_task, w))
            for _ in range(num_readers):
                futures.append(executor.submit(reader_task))
            concurrent.futures.wait(futures)

        assert not errors, f"Concurrent WAL operations failed with errors: {errors}"

        # Verify all concepts were written
        total_expected = num_writers * items_per_writer
        cp = temp_storage.get_latest_checkpoint()
        assert cp.size == total_expected, f"Expected {total_expected} entries, found {cp.size}"

    def test_monotonic_sequencer_under_rapid_appends(self, temp_storage: SQLiteConceptStorage):
        """Verify that concurrent appends result in strictly sequential 1..N log_index with no gaps."""
        total_records = 100
        indices: list[int] = []
        lock = threading.Lock()

        def append_task(i: int):
            record = ConceptRecord(
                content_hash=f"rapid_seq_{i:04d}",
                name=f"seq_func_{i}",
                domain="cryptography",
                json_body={"name": f"seq_func_{i}", "domain": "cryptography"},
            )
            idx = temp_storage.append_accepted_concept(record)
            with lock:
                indices.append(idx)

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(append_task, i) for i in range(total_records)]
            concurrent.futures.wait(futures)

        assert len(indices) == total_records
        sorted_indices = sorted(indices)
        expected_indices = list(range(1, total_records + 1))
        assert sorted_indices == expected_indices, "Sequencer produced gaps or non-monotonic indices!"

    def test_c2sp_tlog_tiles_and_signature_integrity(self, temp_storage: SQLiteConceptStorage):
        """Test C2SP hash tiles, data tiles, and Ed25519 signature tamper detection."""
        # Append 10 concepts
        for i in range(10):
            record = ConceptRecord(
                content_hash=f"tlog_hash_{i:04d}",
                name=f"tlog_func_{i}",
                domain="astronomy",
                json_body={"name": f"tlog_func_{i}", "summary": f"TLog item {i}"},
            )
            temp_storage.append_accepted_concept(record)

        # 1. Fetch data tile 0
        data_tile = temp_storage.get_data_tile(0)
        assert data_tile is not None
        assert len(data_tile) > 0

        # 2. Fetch latest checkpoint
        checkpoint = temp_storage.get_latest_checkpoint()
        assert checkpoint.size == 10
        assert checkpoint.root_hash != ""

        # 3. Verify valid signature with storage public key
        assert verify_checkpoint_signature(checkpoint, temp_storage.public_key) is True

        # 4. Adversarial tamper detection:
        # A. Tamper with root hash
        tampered_hash_checkpoint = checkpoint.__class__(
            origin=checkpoint.origin,
            size=checkpoint.size,
            root_hash="TamperedRootHash1234567890=",
            timestamp=checkpoint.timestamp,
            signature=checkpoint.signature,
            key_hint=checkpoint.key_hint,
        )
        assert verify_checkpoint_signature(tampered_hash_checkpoint, temp_storage.public_key) is False

        # B. Tamper with size
        tampered_size_checkpoint = checkpoint.__class__(
            origin=checkpoint.origin,
            size=checkpoint.size + 1,
            root_hash=checkpoint.root_hash,
            timestamp=checkpoint.timestamp,
            signature=checkpoint.signature,
            key_hint=checkpoint.key_hint,
        )
        assert verify_checkpoint_signature(tampered_size_checkpoint, temp_storage.public_key) is False

        # C. Tamper with signature
        tampered_sig_checkpoint = checkpoint.__class__(
            origin=checkpoint.origin,
            size=checkpoint.size,
            root_hash=checkpoint.root_hash,
            timestamp=checkpoint.timestamp,
            signature=checkpoint.signature[:-4] + "AAAA",
            key_hint=checkpoint.key_hint,
        )
        assert verify_checkpoint_signature(tampered_sig_checkpoint, temp_storage.public_key) is False

    def test_live_sse_stream_and_replay(self, tmp_path: Path):
        """Test GET /v1/stream?since=N historical replay and real-time push notification."""
        async def _run():
            db_path = tmp_path / "sse_test.db"
            storage = SQLiteConceptStorage(db_path=db_path)

            # Seed 3 concepts
            for i in range(1, 4):
                storage.append_accepted_concept(
                    ConceptRecord(
                        content_hash=f"seed_hash_{i}",
                        name=f"seed_func_{i}",
                        domain="physics",
                        json_body={"name": f"seed_func_{i}"},
                    )
                )

            app = create_hub_app(storage=storage)
            hub: OpenCommonsHub = app[HUB_APP_KEY]
            client = test_utils.TestClient(test_utils.TestServer(app))
            await client.start_server()

            try:
                # 1. Connect to SSE stream requesting entries since=1
                resp = await client.get("/v1/stream?since=1")
                assert resp.status == 200
                assert resp.headers.get("Content-Type") == "text/event-stream"

                # 2. Read replayed chunks (should contain entries 2 and 3)
                chunk1 = await resp.content.readuntil(b"\n\n")
                text1 = chunk1.decode("utf-8")
                assert "id: 2" in text1

                chunk2 = await resp.content.readuntil(b"\n\n")
                text2 = chunk2.decode("utf-8")
                assert "id: 3" in text2

                # 3. Trigger a live broadcast while client is connected
                live_record = ConceptRecord(
                    content_hash="live_push_hash_004",
                    name="live_push_func",
                    domain="physics",
                    json_body={"name": "live_push_func"},
                )
                new_idx = storage.append_accepted_concept(live_record)
                hub.broadcaster.broadcast(
                    {
                        "log_index": new_idx,
                        "entry_type": "concept",
                        "content_hash": "live_push_hash_004",
                        "domain": "physics",
                        "name": "live_push_func",
                        "created_at": time.time(),
                    }
                )

                # Read live pushed chunk
                live_chunk = await resp.content.readuntil(b"\n\n")
                live_text = live_chunk.decode("utf-8")
                assert f"id: {new_idx}" in live_text
                assert "live_push_hash_004" in live_text

                resp.close()
            finally:
                await client.close()

        asyncio.run(_run())

    def test_legacy_sync_cutover_returns_410_gone(self, tmp_path: Path):
        """Verify immediate cutover: GET/POST /v1/sync returns 410 Gone."""
        async def _run():
            db_path = tmp_path / "sync_cutover.db"
            storage = SQLiteConceptStorage(db_path=db_path)
            app = create_hub_app(storage=storage)
            client = test_utils.TestClient(test_utils.TestServer(app))
            await client.start_server()

            try:
                # Test POST /v1/sync
                post_resp = await client.post("/v1/sync", json={"concepts": []})
                assert post_resp.status == 410
                post_body = await post_resp.json()
                assert "retired" in post_body.get("error", "").lower()
                assert "migration" in post_body

                # Test GET /v1/sync
                get_resp = await client.get("/v1/sync")
                assert get_resp.status == 410
                get_body = await get_resp.json()
                assert "retired" in get_body.get("error", "").lower()
            finally:
                await client.close()

        asyncio.run(_run())

    def test_fts5_malformed_query_injection(self, temp_storage: SQLiteConceptStorage):
        """Adversarial FTS queries with unclosed quotes, boolean operators, and wildcards must not crash."""
        temp_storage.append_accepted_concept(
            ConceptRecord(
                content_hash="fts_test_hash",
                name="fts_func",
                domain="numerical",
                json_body={"name": "fts_func", "summary": "numerical optimization using Nelder-Mead"},
            )
        )
        malformed_queries = [
            '"""""',
            'AND OR NOT ""',
            '*',
            'NEAR(a, b, c)',
            "'; DROP TABLE concepts; --",
            "^%$#@!",
        ]
        for q in malformed_queries:
            results = temp_storage.find_concepts(query=q)
            assert isinstance(results, list)

    def test_delta_pack_compression_and_retrieval(self, temp_storage: SQLiteConceptStorage):
        """Verify delta pack generation and integrity across a range of log indices."""
        for i in range(1, 11):
            temp_storage.append_accepted_concept(
                ConceptRecord(
                    content_hash=f"delta_pack_hash_{i:04d}",
                    name=f"delta_func_{i}",
                    domain="physics",
                    json_body={"name": f"delta_func_{i}", "summary": f"Delta item {i}"},
                )
            )
        pack_bytes = temp_storage.get_delta_pack(from_index=1, to_index=10)
        assert pack_bytes is not None
        assert len(pack_bytes) > 0


# ===========================================================================
# 3. Frontier Agent Integration Handshake Tests
# ===========================================================================


class TestFrontierAgentIntegrationHandshake:
    """Live stdio JSON-RPC handshake verification across kardashev-mcp and opencommons-mcp."""

    def test_live_integration_check_stdio_handshake(self):
        """Execute full integrate --check health check and verify all targets and MCP servers."""
        summary = check_integrations()
        assert summary.all_ok is True, f"integrate --check failed: {summary.format_table()}"

        # Verify kardashev-mcp handshake details
        k_hs = summary.kardashev_handshake
        assert k_hs.success is True
        assert k_hs.server_name == "kardashev-mcp"
        assert len(k_hs.tools) >= 8
        assert "scout_codebase" in k_hs.tools
        assert "apply_smart_patch" in k_hs.tools
        assert "verify_and_repair" in k_hs.tools

        # Verify opencommons-mcp handshake details
        oc_hs = summary.opencommons_handshake
        assert oc_hs.success is True
        assert oc_hs.server_name == "opencommons-mcp"
        assert len(oc_hs.tools) == 4
        assert set(oc_hs.tools) == {"find", "record", "sync", "status"}

    def test_mcp_handshake_failure_handling_on_invalid_command(self):
        """Verify perform_mcp_handshake gracefully handles nonexistent commands or timeouts."""
        bad_hs = perform_mcp_handshake(["nonexistent-mcp-binary-9999"], timeout=1.0)
        assert bad_hs.success is False
        assert bad_hs.error is not None

    def test_cli_subprocess_integrate_check(self):
        """Verify executing 'kardashev integrate --check' via CLI returns exit code 0."""
        res = subprocess.run(
            [sys.executable, "-m", "kardashev_cli.cli", "integrate", "--check"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        assert res.returncode == 0, f"CLI exited with {res.returncode}: {res.stderr}"
        assert "ALL SYSTEMS OPERATIONAL" in res.stdout

