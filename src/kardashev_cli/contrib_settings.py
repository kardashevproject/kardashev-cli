"""Per-user OpenCommons contribution settings.

Every coding-agent session can harvest small verified concepts (pure functions plus
a summary and invariants) into the OpenCommons concept store. Whether and how that
happens for a given install is controlled here, not by the agent.

Storage: a JSON file at `~/.config/kardashev/contribution.json` (respecting
`XDG_CONFIG_HOME`), overridable for tests with the `KARDASHEV_CONTRIB_CONFIG`
environment variable (a full file path). A small sibling state file tracks the
daily submission count.

Privacy: proposals handled through this module may contain only concept fields
(name, signature/types, summary, pseudocode, invariants, pitfalls, code, domain).
Contributors are identified only by `anonymous_key`, a random 128-bit value
generated on first load. Nothing here is derived from the user's name, machine
hostname, or IP address, and none of that is ever recorded.
"""

from __future__ import annotations

import fnmatch
import json
import os
import secrets
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

Mode = Literal["off", "review", "auto"]
DecisionAction = Literal["drop", "queue", "submit"]

_VALID_MODES = ("off", "review", "auto")
_VALID_SOURCES = ("mcp_record", "harvest_edits")

_OPTOUT_MARKER_FILENAME = ".opencommons-ignore"


class ContribSettingsError(ValueError):
    """Raised for invalid settings data or an invalid key/value update."""


@dataclass
class ContributionSettings:
    """Typed, persisted contribution settings for this install."""

    mode: Mode = "review"
    sources: dict[str, bool] = field(default_factory=lambda: {"mcp_record": True, "harvest_edits": False})
    domains_allow: list[str] = field(default_factory=list)
    domains_deny: list[str] = field(default_factory=list)
    repo_optout_globs: list[str] = field(default_factory=list)
    daily_cap: int = 20
    sync_to_hub: bool = False
    anonymous_key: str = field(default_factory=lambda: secrets.token_hex(16))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Decision:
    """The outcome of `decide()` for a single proposal."""

    action: DecisionAction
    reason: str


def _field_names() -> set[str]:
    return {f.name for f in fields(ContributionSettings)}


def config_path() -> Path:
    """Returns the settings file path, honouring `KARDASHEV_CONTRIB_CONFIG` and `XDG_CONFIG_HOME`."""
    override = os.environ.get("KARDASHEV_CONTRIB_CONFIG")
    if override:
        return Path(override).expanduser()
    base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "kardashev" / "contribution.json"


def state_path() -> Path:
    """Returns the small daily-count state file that sits beside the settings file."""
    cfg = config_path()
    return cfg.with_name(cfg.stem + "_state.json")


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    os.replace(tmp, path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _validate_settings_dict(raw: dict[str, Any]) -> None:
    known = _field_names()
    unknown = set(raw.keys()) - known
    if unknown:
        raise ContribSettingsError(f"Unknown setting(s): {', '.join(sorted(unknown))}")
    if "mode" in raw and raw["mode"] not in _VALID_MODES:
        raise ContribSettingsError(f"Invalid mode '{raw['mode']}'; must be one of {_VALID_MODES}")
    if "sources" in raw:
        if not isinstance(raw["sources"], dict):
            raise ContribSettingsError("'sources' must be an object of {source_name: bool}")
        bad = set(raw["sources"].keys()) - set(_VALID_SOURCES)
        if bad:
            raise ContribSettingsError(f"Unknown source(s): {', '.join(sorted(bad))}")
        for k, v in raw["sources"].items():
            if not isinstance(v, bool):
                raise ContribSettingsError(f"Source '{k}' must be a boolean")
    for list_field in ("domains_allow", "domains_deny", "repo_optout_globs"):
        if list_field in raw and not isinstance(raw[list_field], list):
            raise ContribSettingsError(f"'{list_field}' must be a list")
    if "daily_cap" in raw and (
        not isinstance(raw["daily_cap"], int) or isinstance(raw["daily_cap"], bool) or raw["daily_cap"] < 0
    ):
        raise ContribSettingsError("'daily_cap' must be a non-negative integer")
    if "sync_to_hub" in raw and not isinstance(raw["sync_to_hub"], bool):
        raise ContribSettingsError("'sync_to_hub' must be a boolean")
    if "anonymous_key" in raw and not isinstance(raw["anonymous_key"], str):
        raise ContribSettingsError("'anonymous_key' must be a string")


def load_settings() -> ContributionSettings:
    """Loads settings from disk, creating defaults (with a fresh anonymous key) on first run."""
    path = config_path()
    if not path.is_file():
        settings = ContributionSettings()
        save_settings(settings)
        return settings
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ContribSettingsError("Contribution settings file must contain a JSON object")
    _validate_settings_dict(raw)
    defaults = ContributionSettings()
    merged = {**defaults.to_dict(), **raw}
    return ContributionSettings(**merged)


def save_settings(settings: ContributionSettings) -> Path:
    """Atomically persists settings with owner-only (0600) permissions."""
    path = config_path()
    _atomic_write(path, settings.to_dict())
    return path


def _coerce_value(key: str, raw_value: Any) -> Any:
    """Coerces a CLI/dashboard string value to the type a given field expects."""
    if key in ("domains_allow", "domains_deny", "repo_optout_globs"):
        if isinstance(raw_value, list):
            return raw_value
        if isinstance(raw_value, str):
            return [v.strip() for v in raw_value.split(",") if v.strip()]
        raise ContribSettingsError(f"'{key}' must be a comma-separated string or list")
    if key == "daily_cap":
        try:
            return int(raw_value)
        except (TypeError, ValueError) as exc:
            raise ContribSettingsError("'daily_cap' must be an integer") from exc
    if key == "sync_to_hub":
        if isinstance(raw_value, bool):
            return raw_value
        if isinstance(raw_value, str) and raw_value.lower() in ("true", "false", "1", "0", "yes", "no"):
            return raw_value.lower() in ("true", "1", "yes")
        raise ContribSettingsError("'sync_to_hub' must be a boolean")
    if key == "sources":
        if isinstance(raw_value, dict):
            return raw_value
        raise ContribSettingsError("'sources' must be set per-key, e.g. 'sources.harvest_edits'")
    return raw_value


def apply_update(settings: ContributionSettings, key: str, value: Any) -> ContributionSettings:
    """Returns a new settings object with `key` set to `value` (validated).

    `key` may address a nested source flag as `sources.<name>` (e.g. `sources.harvest_edits`).
    """
    data = settings.to_dict()
    if key.startswith("sources."):
        source_name = key.split(".", 1)[1]
        if source_name not in _VALID_SOURCES:
            raise ContribSettingsError(f"Unknown source '{source_name}'")
        coerced = _coerce_value("sync_to_hub", value) if isinstance(value, str) else bool(value)
        data["sources"] = {**data["sources"], source_name: coerced}
    else:
        if key not in _field_names():
            raise ContribSettingsError(f"Unknown setting '{key}'")
        if key == "anonymous_key":
            raise ContribSettingsError("'anonymous_key' is generated automatically and cannot be set")
        data[key] = _coerce_value(key, value)
    _validate_settings_dict(data)
    return ContributionSettings(**data)


def _today() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())


def _load_state() -> dict[str, Any]:
    path = state_path()
    if not path.is_file():
        return {"date": _today(), "count": 0}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"date": _today(), "count": 0}
    if not isinstance(raw, dict) or "date" not in raw or "count" not in raw:
        return {"date": _today(), "count": 0}
    if raw["date"] != _today():
        return {"date": _today(), "count": 0}
    return raw


def _save_state(state: dict[str, Any]) -> None:
    _atomic_write(state_path(), state)


def daily_count() -> int:
    """Returns how many proposals have been queued or submitted so far today."""
    return int(_load_state().get("count", 0))


def _record_daily_usage() -> None:
    state = _load_state()
    state["count"] = int(state.get("count", 0)) + 1
    _save_state(state)


def _has_optout_marker(repo_path: Path) -> bool:
    current = repo_path if repo_path.is_dir() else repo_path.parent
    home = Path.home()
    seen = set()
    while True:
        if current in seen:
            break
        seen.add(current)
        if (current / _OPTOUT_MARKER_FILENAME).is_file():
            return True
        if current == current.parent or current == home.parent:
            break
        current = current.parent
    return False


def _matches_optout_glob(path_str: str, globs: list[str]) -> bool:
    return any(fnmatch.fnmatch(path_str, pattern) for pattern in globs)


def decide(
    proposal_meta: dict[str, Any],
    source: str,
    repo_path: str | Path | None = None,
    *,
    record_usage: bool = True,
) -> Decision:
    """Decides what to do with a candidate proposal.

    `proposal_meta` must carry at least a `domain` key. `source` is one of
    `mcp_record` or `harvest_edits`. `repo_path` is the repository (or file)
    the proposal came from, used for opt-out checks; pass `None` when there
    is no repo context (e.g. an explicit MCP record call with no file).
    """
    settings = load_settings()

    if settings.mode == "off":
        return Decision("drop", "mode_off")

    if source not in _VALID_SOURCES:
        return Decision("drop", f"unknown_source:{source}")
    if not settings.sources.get(source, False):
        return Decision("drop", f"source_disabled:{source}")

    domain = str(proposal_meta.get("domain", "general"))
    if domain in settings.domains_deny:
        return Decision("drop", f"domain_denied:{domain}")
    if settings.domains_allow and domain not in settings.domains_allow:
        return Decision("drop", f"domain_not_allowed:{domain}")

    if repo_path is not None:
        rp = Path(repo_path).expanduser()
        if settings.repo_optout_globs and _matches_optout_glob(str(rp), settings.repo_optout_globs):
            return Decision("drop", "repo_optout_glob")
        if _has_optout_marker(rp):
            return Decision("drop", "repo_optout_marker")

    if daily_count() >= settings.daily_cap:
        return Decision("drop", "daily_cap_reached")

    action: DecisionAction = "submit" if settings.mode == "auto" else "queue"
    if record_usage:
        _record_daily_usage()
    return Decision(action, f"mode_{settings.mode}")
