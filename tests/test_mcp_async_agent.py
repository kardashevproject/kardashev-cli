"""Tests for asynchronous local-agent task execution over the MCP server.

Regression anchors: run_local_agent_task used to block for the entire agent
run, so long tasks exceeded MCP client call timeouts and their results were
lost (observed 2026-10-15 on a multi-minute GLM task). Now the MCP tool starts
a task and callers poll `local_agent_task_status`.
"""

from __future__ import annotations

import time

import pytest

import kardashev_cli.mcp_server as mcp_server


@pytest.fixture(autouse=True)
def _clear_registry():
    mcp_server._AGENT_TASKS.clear()
    yield
    mcp_server._AGENT_TASKS.clear()


def test_async_start_returns_immediately_and_result_is_pollable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fast_agent(**kwargs):
        return {"output": "done", "tool_calls": []}

    monkeypatch.setattr(mcp_server, "_execute_local_agent_async", fast_agent)

    t0 = time.monotonic()
    started = mcp_server.run_local_agent_task(prompt="do a thing", async_start=True)
    elapsed = time.monotonic() - t0
    assert started["status"] == "started"
    assert started["task_id"].startswith("lat_")
    assert elapsed < 1.0

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        status = mcp_server.agent_task_status(started["task_id"])
        if status["status"] != "running":
            break
        time.sleep(0.05)
    status = mcp_server.agent_task_status(started["task_id"])
    assert status["status"] == "succeeded"
    assert status["result"]["output"] == "done"


def test_async_failure_is_recorded_not_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    async def broken_agent(**kwargs):
        raise RuntimeError("provider exploded")

    monkeypatch.setattr(mcp_server, "_execute_local_agent_async", broken_agent)
    started = mcp_server.run_local_agent_task(prompt="x", async_start=True)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        status = mcp_server.agent_task_status(started["task_id"])
        if status["status"] != "running":
            break
        time.sleep(0.05)
    status = mcp_server.agent_task_status(started["task_id"])
    assert status["status"] == "failed"
    assert "RuntimeError" in status["error"]


def test_mcp_dispatch_defaults_to_async_and_status_tool_works(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fast_agent(**kwargs):
        return {"output": "ok"}

    monkeypatch.setattr(mcp_server, "_execute_local_agent_async", fast_agent)

    resp = mcp_server.handle_mcp_request(
        {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {
                "name": "run_local_agent_task",
                "arguments": {"prompt": "hello"},
            },
        }
    )
    assert resp is not None and resp["result"]["isError"] is False
    payload = resp["result"]["structuredContent"]
    assert payload["status"] == "started"

    task_id = payload["task_id"]
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        poll = mcp_server.handle_mcp_request(
            {
                "jsonrpc": "2.0",
                "id": 8,
                "method": "tools/call",
                "params": {
                    "name": "local_agent_task_status",
                    "arguments": {"task_id": task_id},
                },
            }
        )
        state = poll["result"]["structuredContent"]
        if state["status"] != "running":
            break
        time.sleep(0.05)
    assert state["status"] == "succeeded"
    assert state["result"]["output"] == "ok"


def test_status_of_unknown_task_is_honest() -> None:
    status = mcp_server.agent_task_status("lat_missing")
    assert status["found"] is False
    assert status["task_id"] == "lat_missing"