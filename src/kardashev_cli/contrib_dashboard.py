"""Local dashboard for OpenCommons contribution settings and review queue.

A tiny, self-contained HTTP server (stdlib only, no external dependencies) that:

- shows and edits the settings in `kardashev_cli.contrib_settings`,
- shows the local review queue (pending proposals staged by
  `kardashev_cli.opencommons`, with approve/reject buttons),
- shows a history of what was approved, rejected, or sent (concept fields and
  timestamps only; never file paths).

It binds to 127.0.0.1 only. Every POST request must carry the per-launch CSRF
token embedded in the page, or it is rejected with 403. The server never
serves any file from disk; the only GET responses are the page itself and a
small, fixed set of JSON API endpoints.

Submitting approved proposals to a shared hub is out of scope here. See
`submit_approved()` below, which is a deliberate stub for the hub team to
wire up.
"""

from __future__ import annotations

import contextlib
import json
import os
import secrets
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from kardashev_cli import contrib_settings as settings_mod
from kardashev_cli.contrib_settings import ContribSettingsError


def _pending_path() -> str | None:
    """Reads `OPENCOMMONS_PENDING_PATH` at call time (not import time) so tests can isolate it."""
    return os.environ.get("OPENCOMMONS_PENDING_PATH")

HistoryAction = str  # "approved" | "rejected" | "sent"


def history_path() -> Path:
    """Returns the history log path, sitting beside the settings file."""
    cfg = settings_mod.config_path()
    return cfg.with_name(cfg.stem + "_history.json")


def load_history() -> list[dict[str, Any]]:
    path = history_path()
    if not path.is_file():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return raw if isinstance(raw, list) else []


def _append_history(action: HistoryAction, concept: dict[str, Any]) -> None:
    """Appends one history entry. Stores only concept fields + timestamp, never paths."""
    entry = {
        "action": action,
        "func_name": concept.get("func_name", ""),
        "domain": concept.get("domain", ""),
        "summary": concept.get("summary", ""),
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    entries = load_history()
    entries.append(entry)
    path = history_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")


def submit_approved() -> None:
    """Stub for the hub team: submits already-approved, locally-stored concepts to the shared hub.

    This function intentionally does nothing yet. Approval here only moves a proposal
    from the local pending-review queue into the local OpenCommons store (see
    `kardashev_cli.opencommons.approve_pending_proposal`). Nothing leaves this machine
    until a hub-submission implementation replaces this stub and `sync_to_hub` is enabled.
    """
    raise NotImplementedError("Hub submission is not implemented yet; see contrib_dashboard.submit_approved.")


def _pending_proposals_as_dicts() -> list[dict[str, Any]]:
    from kardashev_cli.opencommons import load_pending_proposals

    out = []
    for p in load_pending_proposals(_pending_path()):
        d = p.to_dict()
        out.append(d)
    return out


@dataclass
class _ServerState:
    csrf_token: str = field(default_factory=lambda: secrets.token_hex(32))


_PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OpenCommons Contribution Settings</title>
<meta name="csrf-token" content="{csrf_token}">
<style>
  :root {{
    color-scheme: light dark;
    --bg: #f7f7f8; --fg: #1a1a1a; --card: #ffffff; --border: #d9d9df;
    --accent: #2b6cb0; --danger: #b03030; --muted: #666;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg: #16161a; --fg: #ececec; --card: #202026; --border: #35353d; --muted: #a0a0a8; }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 16px; background: var(--bg); color: var(--fg);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-size: 15px; line-height: 1.45;
  }}
  h1 {{ font-size: 1.3rem; margin: 0 0 4px; }}
  h2 {{ font-size: 1.05rem; margin: 24px 0 8px; }}
  p.sub {{ color: var(--muted); margin: 0 0 20px; }}
  .card {{
    background: var(--card); border: 1px solid var(--border); border-radius: 10px;
    padding: 16px; margin-bottom: 16px;
  }}
  label {{ display: block; font-weight: 600; margin: 12px 0 4px; }}
  label:first-child {{ margin-top: 0; }}
  input[type=text], input[type=number], select {{
    width: 100%; padding: 8px; border: 1px solid var(--border); border-radius: 6px;
    background: var(--bg); color: var(--fg); font-size: 14px;
  }}
  .row {{ display: flex; align-items: center; gap: 8px; margin: 8px 0; }}
  .row input[type=checkbox] {{ width: 18px; height: 18px; }}
  button {{
    padding: 8px 14px; border-radius: 6px; border: 1px solid var(--accent);
    background: var(--accent); color: #fff; font-size: 14px; cursor: pointer;
  }}
  button.secondary {{ background: transparent; color: var(--accent); }}
  button.danger {{ background: transparent; color: var(--danger); border-color: var(--danger); }}
  .queue-item {{ border-top: 1px solid var(--border); padding: 10px 0; }}
  .queue-item:first-child {{ border-top: none; }}
  .queue-item code {{ display: block; white-space: pre-wrap; background: var(--bg);
    padding: 8px; border-radius: 6px; font-size: 12px; margin: 6px 0; max-height: 200px; overflow: auto; }}
  .muted {{ color: var(--muted); font-size: 13px; }}
  .status {{ font-size: 13px; margin-top: 8px; min-height: 1.2em; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  td, th {{ text-align: left; padding: 4px 6px; border-bottom: 1px solid var(--border); }}
</style>
</head>
<body>
<h1>OpenCommons contribution settings</h1>
<p class="sub">Controls whether and how this install contributes verified concepts.
Everything here is local to this machine.</p>

<div class="card">
  <h2 style="margin-top:0">Settings</h2>
  <label for="mode">Mode</label>
  <select id="mode">
    <option value="off">Off - never contribute</option>
    <option value="review">Review - queue locally, you approve each one</option>
    <option value="auto">Auto - submit after local checks</option>
  </select>

  <label>Sources</label>
  <div class="row"><input type="checkbox" id="src_mcp_record"><label for="src_mcp_record" style="margin:0">mcp_record (agent explicitly records a concept)</label></div>
  <div class="row"><input type="checkbox" id="src_harvest_edits"><label for="src_harvest_edits" style="margin:0">harvest_edits (auto-harvest from edited files)</label></div>

  <label for="domains_allow">Domains allow (comma separated, empty = all)</label>
  <input type="text" id="domains_allow">

  <label for="domains_deny">Domains deny (comma separated)</label>
  <input type="text" id="domains_deny">

  <label for="repo_optout_globs">Repo opt-out globs (comma separated)</label>
  <input type="text" id="repo_optout_globs">

  <label for="daily_cap">Daily cap</label>
  <input type="number" id="daily_cap" min="0">

  <div class="row"><input type="checkbox" id="sync_to_hub"><label for="sync_to_hub" style="margin:0">Sync approved concepts to the shared hub</label></div>

  <p class="muted">Anonymous key: <span id="anon_key"></span></p>

  <button id="save_btn">Save settings</button>
  <div class="status" id="save_status"></div>
</div>

<div class="card">
  <h2 style="margin-top:0">Review queue</h2>
  <div id="queue"></div>
</div>

<div class="card">
  <h2 style="margin-top:0">History</h2>
  <table id="history_table">
    <thead><tr><th>When</th><th>Action</th><th>Function</th><th>Domain</th></tr></thead>
    <tbody></tbody>
  </table>
</div>

<script>
const csrf = document.querySelector('meta[name="csrf-token"]').content;

function escapeHtml(s) {{
  return String(s).replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}})[c]);
}}

async function api(path, opts) {{
  opts = opts || {{}};
  opts.headers = Object.assign({{'X-CSRF-Token': csrf}}, opts.headers || {{}});
  const res = await fetch(path, opts);
  if (!res.ok) throw new Error('request failed: ' + res.status);
  return res.json();
}}

function fillSettings(s) {{
  document.getElementById('mode').value = s.mode;
  document.getElementById('src_mcp_record').checked = !!s.sources.mcp_record;
  document.getElementById('src_harvest_edits').checked = !!s.sources.harvest_edits;
  document.getElementById('domains_allow').value = s.domains_allow.join(', ');
  document.getElementById('domains_deny').value = s.domains_deny.join(', ');
  document.getElementById('repo_optout_globs').value = s.repo_optout_globs.join(', ');
  document.getElementById('daily_cap').value = s.daily_cap;
  document.getElementById('sync_to_hub').checked = !!s.sync_to_hub;
  document.getElementById('anon_key').textContent = s.anonymous_key;
}}

function renderQueue(items) {{
  const el = document.getElementById('queue');
  if (!items.length) {{
    el.innerHTML = '<p class="muted">Nothing pending.</p>';
    return;
  }}
  el.innerHTML = items.map(p => `
    <div class="queue-item" data-id="${{escapeHtml(p.proposal_id)}}">
      <strong>${{escapeHtml(p.func_name)}}</strong> <span class="muted">(${{escapeHtml(p.domain)}})</span>
      <div class="muted">${{escapeHtml(p.summary)}}</div>
      <code>${{escapeHtml(p.source_code)}}</code>
      <button class="approve-btn">Approve</button>
      <button class="danger reject-btn">Reject</button>
    </div>
  `).join('');
  el.querySelectorAll('.approve-btn').forEach(btn => btn.addEventListener('click', () => act(btn, 'approve')));
  el.querySelectorAll('.reject-btn').forEach(btn => btn.addEventListener('click', () => act(btn, 'reject')));
}}

async function act(btn, action) {{
  const id = btn.closest('.queue-item').dataset.id;
  await api('/api/queue/' + encodeURIComponent(id) + '/' + action, {{method: 'POST'}});
  await refreshQueue();
  await refreshHistory();
}}

function renderHistory(items) {{
  const tbody = document.querySelector('#history_table tbody');
  tbody.innerHTML = items.slice().reverse().map(h => `
    <tr><td>${{escapeHtml(h.at)}}</td><td>${{escapeHtml(h.action)}}</td><td>${{escapeHtml(h.func_name)}}</td><td>${{escapeHtml(h.domain)}}</td></tr>
  `).join('');
}}

async function refreshSettings() {{ fillSettings(await api('/api/settings')); }}
async function refreshQueue() {{ renderQueue(await api('/api/queue')); }}
async function refreshHistory() {{ renderHistory(await api('/api/history')); }}

document.getElementById('save_btn').addEventListener('click', async () => {{
  const status = document.getElementById('save_status');
  status.textContent = 'Saving...';
  try {{
    const payload = {{
      mode: document.getElementById('mode').value,
      sources: {{
        mcp_record: document.getElementById('src_mcp_record').checked,
        harvest_edits: document.getElementById('src_harvest_edits').checked,
      }},
      domains_allow: document.getElementById('domains_allow').value,
      domains_deny: document.getElementById('domains_deny').value,
      repo_optout_globs: document.getElementById('repo_optout_globs').value,
      daily_cap: parseInt(document.getElementById('daily_cap').value, 10),
      sync_to_hub: document.getElementById('sync_to_hub').checked,
    }};
    const s = await api('/api/settings', {{method: 'POST', headers: {{'Content-Type': 'application/json'}}, body: JSON.stringify(payload)}});
    fillSettings(s);
    status.textContent = 'Saved.';
  }} catch (e) {{
    status.textContent = 'Failed to save: ' + e.message;
  }}
}});

refreshSettings();
refreshQueue();
refreshHistory();
</script>
</body>
</html>
"""


class _Handler(BaseHTTPRequestHandler):
    server_version = "KardashevContribDashboard/1"

    def log_message(self, fmt: str, *args: Any) -> None:  # silence default stderr logging
        pass

    def _send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, status: int, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _check_csrf(self) -> bool:
        token = self.headers.get("X-CSRF-Token") or ""
        return secrets.compare_digest(token, self.server.state.csrf_token)  # type: ignore[attr-defined]

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/":
            html = _PAGE_TEMPLATE.format(csrf_token=self.server.state.csrf_token)  # type: ignore[attr-defined]
            self._send_html(200, html)
            return
        if path == "/api/settings":
            self._send_json(200, settings_mod.load_settings().to_dict())
            return
        if path == "/api/queue":
            self._send_json(200, _pending_proposals_as_dicts())
            return
        if path == "/api/history":
            self._send_json(200, load_history())
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if not self._check_csrf():
            self._send_json(403, {"error": "missing or invalid CSRF token"})
            return
        parsed = urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw_body = self.rfile.read(length) if length else b""

        if path == "/api/settings":
            try:
                payload = json.loads(raw_body.decode("utf-8")) if raw_body else {}
                if not isinstance(payload, dict):
                    raise ContribSettingsError("body must be a JSON object")
                current = settings_mod.load_settings()
                for key, value in payload.items():
                    if key == "sources" and isinstance(value, dict):
                        for src, enabled in value.items():
                            current = settings_mod.apply_update(current, f"sources.{src}", bool(enabled))
                    else:
                        current = settings_mod.apply_update(current, key, value)
                settings_mod.save_settings(current)
                self._send_json(200, current.to_dict())
            except (ContribSettingsError, json.JSONDecodeError, UnicodeDecodeError) as exc:
                self._send_json(400, {"error": str(exc)})
            return

        if path.startswith("/api/queue/"):
            parts = path.split("/")
            # /api/queue/<id>/approve|reject
            if len(parts) == 5 and parts[4] in ("approve", "reject"):
                proposal_id, verb = parts[3], parts[4]
                self._handle_queue_action(proposal_id, verb)
                return

        self._send_json(404, {"error": "not found"})

    def _handle_queue_action(self, proposal_id: str, verb: str) -> None:
        from kardashev_cli.opencommons import (
            approve_pending_proposal,
            load_pending_proposals,
            reject_pending_proposal,
        )

        pending_path = _pending_path()
        matched = next(
            (
                p
                for p in load_pending_proposals(pending_path)
                if p.proposal_id == proposal_id or p.proposal_id.startswith(proposal_id)
            ),
            None,
        )
        concept = matched.to_dict() if matched else {"func_name": proposal_id, "domain": "", "summary": ""}

        if verb == "approve":
            res = approve_pending_proposal(proposal_id, pending_path=pending_path)
            if res.get("approved"):
                _append_history("approved", concept)
            self._send_json(200 if res.get("approved") else 404, res)
        else:
            res = reject_pending_proposal(proposal_id, pending_path=pending_path)
            if res.get("rejected"):
                _append_history("rejected", concept)
            self._send_json(200 if res.get("rejected") else 404, res)


class ContribDashboardServer(ThreadingHTTPServer):
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], state: _ServerState) -> None:
        self.state = state
        super().__init__(address, _Handler)


def start_dashboard(port: int = 0, open_browser: bool = False) -> tuple[ContribDashboardServer, threading.Thread]:
    """Starts the dashboard server bound to 127.0.0.1 in a background thread.

    Returns `(server, thread)`. Call `server.shutdown()` then `thread.join()` to stop it.
    """
    state = _ServerState()
    server = ContribDashboardServer(("127.0.0.1", port), state)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    if open_browser:
        host, actual_port = server.server_address[:2]
        with contextlib.suppress(Exception):
            webbrowser.open(f"http://{host}:{actual_port}/")
    return server, thread


def stop_dashboard(server: ContribDashboardServer, thread: threading.Thread) -> None:
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)
