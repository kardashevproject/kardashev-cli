"""Tests for OpenCommons contribution settings, decide(), and the local dashboard."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from kardashev_cli import contrib_settings as cs
from kardashev_cli.contrib_dashboard import start_dashboard, stop_dashboard


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cfg = tmp_path / "contribution.json"
    monkeypatch.setenv("KARDASHEV_CONTRIB_CONFIG", str(cfg))
    # Isolate the OpenCommons pending-review queue too, so dashboard queue/history
    # tests never touch a real user's local queue.
    monkeypatch.setenv("OPENCOMMONS_PENDING_PATH", str(tmp_path / "pending_queue.json"))
    return cfg


# --- defaults & persistence -------------------------------------------------


def test_defaults_on_fresh_install(_isolated_config: Path) -> None:
    settings = cs.load_settings()
    assert settings.mode == "review"
    assert settings.sources == {"mcp_record": True, "harvest_edits": False}
    assert settings.domains_allow == []
    assert settings.domains_deny == []
    assert settings.repo_optout_globs == []
    assert settings.daily_cap == 20
    assert settings.sync_to_hub is False
    assert len(settings.anonymous_key) == 32  # 128 bits of hex
    assert _isolated_config.is_file()


def test_anonymous_key_stable_across_loads(_isolated_config: Path) -> None:
    key1 = cs.load_settings().anonymous_key
    key2 = cs.load_settings().anonymous_key
    assert key1 == key2


def test_anonymous_key_not_derived_from_identity(_isolated_config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("USER", "jeevesh")
    monkeypatch.setenv("HOSTNAME", "some-host")
    key = cs.load_settings().anonymous_key
    assert "jeevesh" not in key
    assert "host" not in key


def test_persistence_round_trip(_isolated_config: Path) -> None:
    settings = cs.load_settings()
    updated = cs.apply_update(settings, "mode", "auto")
    updated = cs.apply_update(updated, "daily_cap", "7")
    cs.save_settings(updated)

    reloaded = cs.load_settings()
    assert reloaded.mode == "auto"
    assert reloaded.daily_cap == 7
    assert reloaded.anonymous_key == settings.anonymous_key


def test_file_permissions_are_owner_only(_isolated_config: Path) -> None:
    cs.load_settings()
    mode = os.stat(_isolated_config).st_mode & 0o777
    assert mode == 0o600


def test_xdg_config_home_respected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KARDASHEV_CONTRIB_CONFIG", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    expected = tmp_path / "kardashev" / "contribution.json"
    assert cs.config_path() == expected


# --- validation --------------------------------------------------------------


def test_load_rejects_unknown_keys(_isolated_config: Path) -> None:
    _isolated_config.write_text(json.dumps({"mode": "review", "not_a_real_field": 1}))
    with pytest.raises(cs.ContribSettingsError):
        cs.load_settings()


def test_load_rejects_bad_mode(_isolated_config: Path) -> None:
    _isolated_config.write_text(json.dumps({"mode": "yolo"}))
    with pytest.raises(cs.ContribSettingsError):
        cs.load_settings()


def test_apply_update_rejects_unknown_key(_isolated_config: Path) -> None:
    settings = cs.load_settings()
    with pytest.raises(cs.ContribSettingsError):
        cs.apply_update(settings, "bogus", "x")


def test_apply_update_rejects_bad_daily_cap(_isolated_config: Path) -> None:
    settings = cs.load_settings()
    with pytest.raises(cs.ContribSettingsError):
        cs.apply_update(settings, "daily_cap", "not-a-number")


def test_apply_update_rejects_anonymous_key_edit(_isolated_config: Path) -> None:
    settings = cs.load_settings()
    with pytest.raises(cs.ContribSettingsError):
        cs.apply_update(settings, "anonymous_key", "deadbeef")


def test_apply_update_source_flag(_isolated_config: Path) -> None:
    settings = cs.load_settings()
    updated = cs.apply_update(settings, "sources.harvest_edits", True)
    assert updated.sources["harvest_edits"] is True
    assert updated.sources["mcp_record"] is True


def test_apply_update_rejects_unknown_source(_isolated_config: Path) -> None:
    settings = cs.load_settings()
    with pytest.raises(cs.ContribSettingsError):
        cs.apply_update(settings, "sources.not_a_source", True)


def test_apply_update_coerces_comma_separated_lists(_isolated_config: Path) -> None:
    settings = cs.load_settings()
    updated = cs.apply_update(settings, "domains_deny", "finance, medical ,  legal")
    assert updated.domains_deny == ["finance", "medical", "legal"]


# --- decide() branches -------------------------------------------------------


def test_decide_mode_off_drops_everything(_isolated_config: Path) -> None:
    settings = cs.apply_update(cs.load_settings(), "mode", "off")
    cs.save_settings(settings)
    decision = cs.decide({"domain": "general"}, "mcp_record", None)
    assert decision.action == "drop"
    assert decision.reason == "mode_off"


def test_decide_review_mode_queues(_isolated_config: Path) -> None:
    decision = cs.decide({"domain": "general"}, "mcp_record", None)
    assert decision.action == "queue"


def test_decide_auto_mode_submits(_isolated_config: Path) -> None:
    settings = cs.apply_update(cs.load_settings(), "mode", "auto")
    cs.save_settings(settings)
    decision = cs.decide({"domain": "general"}, "mcp_record", None)
    assert decision.action == "submit"


def test_decide_drops_disabled_source(_isolated_config: Path) -> None:
    decision = cs.decide({"domain": "general"}, "harvest_edits", None)
    assert decision.action == "drop"
    assert decision.reason == "source_disabled:harvest_edits"


def test_decide_allows_enabled_source(_isolated_config: Path) -> None:
    settings = cs.apply_update(cs.load_settings(), "sources.harvest_edits", True)
    cs.save_settings(settings)
    decision = cs.decide({"domain": "general"}, "harvest_edits", None)
    assert decision.action == "queue"


def test_decide_drops_denied_domain(_isolated_config: Path) -> None:
    settings = cs.apply_update(cs.load_settings(), "domains_deny", "finance")
    cs.save_settings(settings)
    decision = cs.decide({"domain": "finance"}, "mcp_record", None)
    assert decision.action == "drop"
    assert decision.reason.startswith("domain_denied")


def test_decide_drops_domain_not_in_allowlist(_isolated_config: Path) -> None:
    settings = cs.apply_update(cs.load_settings(), "domains_allow", "math")
    cs.save_settings(settings)
    decision = cs.decide({"domain": "biology"}, "mcp_record", None)
    assert decision.action == "drop"
    assert decision.reason.startswith("domain_not_allowed")


def test_decide_allows_domain_in_allowlist(_isolated_config: Path) -> None:
    settings = cs.apply_update(cs.load_settings(), "domains_allow", "math")
    cs.save_settings(settings)
    decision = cs.decide({"domain": "math"}, "mcp_record", None)
    assert decision.action == "queue"


def test_decide_respects_repo_optout_glob(_isolated_config: Path, tmp_path: Path) -> None:
    repo = tmp_path / "secret-repo"
    repo.mkdir()
    settings = cs.apply_update(cs.load_settings(), "repo_optout_globs", str(tmp_path / "secret-*"))
    cs.save_settings(settings)
    decision = cs.decide({"domain": "general"}, "mcp_record", repo)
    assert decision.action == "drop"
    assert decision.reason == "repo_optout_glob"


def test_decide_respects_optout_marker_file(_isolated_config: Path, tmp_path: Path) -> None:
    repo = tmp_path / "some-repo"
    repo.mkdir()
    (repo / ".opencommons-ignore").write_text("")
    decision = cs.decide({"domain": "general"}, "mcp_record", repo)
    assert decision.action == "drop"
    assert decision.reason == "repo_optout_marker"


def test_decide_optout_marker_found_in_ancestor(_isolated_config: Path, tmp_path: Path) -> None:
    repo = tmp_path / "root-repo"
    nested = repo / "pkg" / "subpkg"
    nested.mkdir(parents=True)
    (repo / ".opencommons-ignore").write_text("")
    decision = cs.decide({"domain": "general"}, "mcp_record", nested)
    assert decision.action == "drop"
    assert decision.reason == "repo_optout_marker"


def test_decide_unknown_source_drops(_isolated_config: Path) -> None:
    decision = cs.decide({"domain": "general"}, "not_a_source", None)
    assert decision.action == "drop"


# --- daily cap & rollover ----------------------------------------------------


def test_daily_cap_enforced(_isolated_config: Path) -> None:
    settings = cs.apply_update(cs.load_settings(), "daily_cap", 2)
    cs.save_settings(settings)

    d1 = cs.decide({"domain": "general"}, "mcp_record", None)
    d2 = cs.decide({"domain": "general"}, "mcp_record", None)
    d3 = cs.decide({"domain": "general"}, "mcp_record", None)

    assert d1.action == "queue"
    assert d2.action == "queue"
    assert d3.action == "drop"
    assert d3.reason == "daily_cap_reached"
    assert cs.daily_count() == 2


def test_daily_cap_rolls_over_on_new_day(_isolated_config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = cs.apply_update(cs.load_settings(), "daily_cap", 1)
    cs.save_settings(settings)

    assert cs.decide({"domain": "general"}, "mcp_record", None).action == "queue"
    assert cs.decide({"domain": "general"}, "mcp_record", None).action == "drop"

    monkeypatch.setattr(cs, "_today", lambda: "2099-01-01")
    decision = cs.decide({"domain": "general"}, "mcp_record", None)
    assert decision.action == "queue"
    assert cs.daily_count() == 1


def test_decide_without_recording_usage_does_not_increment(_isolated_config: Path) -> None:
    cs.decide({"domain": "general"}, "mcp_record", None, record_usage=False)
    cs.decide({"domain": "general"}, "mcp_record", None, record_usage=False)
    assert cs.daily_count() == 0


# --- dashboard server ---------------------------------------------------------


@pytest.fixture
def dashboard():
    server, thread = start_dashboard(port=0, open_browser=False)
    try:
        yield server
    finally:
        stop_dashboard(server, thread)


def _base_url(server) -> str:
    host, port = server.server_address[:2]
    return f"http://{host}:{port}"


def test_dashboard_binds_127_0_0_1(dashboard) -> None:
    assert dashboard.server_address[0] == "127.0.0.1"


def test_dashboard_serves_page(dashboard) -> None:
    resp = urllib.request.urlopen(_base_url(dashboard) + "/")
    assert resp.status == 200
    body = resp.read().decode()
    assert "<title>OpenCommons Contribution Settings</title>" in body
    assert 'name="csrf-token"' in body


def test_dashboard_post_without_csrf_rejected(dashboard) -> None:
    req = urllib.request.Request(_base_url(dashboard) + "/api/settings", data=b"{}", method="POST")
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(req)
    assert excinfo.value.code == 403


def test_dashboard_post_with_wrong_csrf_rejected(dashboard) -> None:
    req = urllib.request.Request(
        _base_url(dashboard) + "/api/settings",
        data=b"{}",
        method="POST",
        headers={"X-CSRF-Token": "wrong-token"},
    )
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(req)
    assert excinfo.value.code == 403


def _extract_csrf(dashboard) -> str:
    page = urllib.request.urlopen(_base_url(dashboard) + "/").read().decode()
    match = re.search(r'name="csrf-token" content="([a-f0-9]+)"', page)
    assert match
    return match.group(1)


def test_dashboard_settings_round_trip(dashboard, _isolated_config: Path) -> None:
    csrf = _extract_csrf(dashboard)
    before = json.loads(urllib.request.urlopen(_base_url(dashboard) + "/api/settings").read())
    assert before["mode"] == "review"

    payload = json.dumps({"mode": "auto", "daily_cap": 3}).encode()
    req = urllib.request.Request(
        _base_url(dashboard) + "/api/settings",
        data=payload,
        method="POST",
        headers={"X-CSRF-Token": csrf, "Content-Type": "application/json"},
    )
    updated = json.loads(urllib.request.urlopen(req).read())
    assert updated["mode"] == "auto"
    assert updated["daily_cap"] == 3

    after = json.loads(urllib.request.urlopen(_base_url(dashboard) + "/api/settings").read())
    assert after["mode"] == "auto"
    assert after["daily_cap"] == 3
    # persisted to disk too
    assert cs.load_settings().mode == "auto"


def test_dashboard_settings_post_rejects_unknown_key(dashboard) -> None:
    csrf = _extract_csrf(dashboard)
    payload = json.dumps({"not_a_field": 1}).encode()
    req = urllib.request.Request(
        _base_url(dashboard) + "/api/settings",
        data=payload,
        method="POST",
        headers={"X-CSRF-Token": csrf, "Content-Type": "application/json"},
    )
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(req)
    assert excinfo.value.code == 400


def test_dashboard_never_serves_arbitrary_files(dashboard) -> None:
    for path in ("/../../etc/passwd", "/etc/passwd", "/some/random/path", "/index.html"):
        req = urllib.request.Request(_base_url(dashboard) + path)
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(req)
        assert excinfo.value.code == 404


def test_dashboard_queue_and_history_empty_by_default(dashboard) -> None:
    queue = json.loads(urllib.request.urlopen(_base_url(dashboard) + "/api/queue").read())
    history = json.loads(urllib.request.urlopen(_base_url(dashboard) + "/api/history").read())
    assert queue == []
    assert history == []
