"""Self-updater, version checker, user config manager, and shell completion generator for kardashev-cli.

Provides standalone CLI installation & lifecycle management:
- Persistent user defaults in `~/.config/kardashev/config.json` (`default_preset`, `policy_mode`,
  `auto_update`, `ralph_verify_cmd`, `show_thoughts`, `with_chronosml_mcp`, `with_opencommons_mcp`).
- Cached background GitHub release / commit version checking (every 12 hours, zero startup block).
- `kardashev update` (and `/update` inside TUI): detects editable git checkout vs standalone
  `~/.local/share/kardashev` / `uv` / `pip` install, pulls latest commits, upgrades dependencies,
  and refreshes `~/.local/bin/{kardashev,ksh,kardashev-mcp,opencommons-mcp}` symlinks.
- `kardashev completions [bash|zsh|fish]`: generates shell tab-completion scripts.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from kardashev_cli import __version__

GITHUB_REPO = "kardashevproject/kardashev-cli"
GITHUB_API_RELEASES = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
GITHUB_API_MAIN_COMMIT = f"https://api.github.com/repos/{GITHUB_REPO}/commits/main"
UPDATE_CHECK_TTL_SECONDS = 12 * 3600  # 12 hours


@dataclass
class UserConfig:
    """Persistent user configuration stored at `~/.config/kardashev/config.json`."""

    default_preset: str = "gemma4-26b"
    policy_mode: str = "allow-all"
    speculative_decoding: bool = True
    show_thoughts: bool = True
    auto_update_check: bool = True
    auto_apply_updates: bool = False
    with_chronosml_mcp: bool = False
    with_opencommons_mcp: bool = False
    ralph_verify_cmd: str | None = None

    @classmethod
    def config_path(cls) -> Path:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
        return base / "kardashev" / "config.json"

    @classmethod
    def load(cls) -> UserConfig:
        path = cls.config_path()
        if not path.is_file():
            return cls()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            valid_keys = set(cls.__dataclass_fields__.keys())
            filtered = {k: v for k, v in raw.items() if k in valid_keys}
            return cls(**filtered)
        except (OSError, json.JSONDecodeError, TypeError):
            return cls()

    def save(self) -> Path:
        path = self.config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
        return path


@dataclass
class UpdateStatus:
    """Result of checking local installation against upstream GitHub `kardashevproject/kardashev-cli`."""

    current_version: str
    latest_version: str
    local_commit: str | None
    remote_commit: str | None
    update_available: bool
    install_mode: str  # "git-editable" | "managed-venv" | "pip"
    repo_root: str | None
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect_install_layout() -> tuple[str, Path | None]:
    """Detects whether `kardashev-cli` is running from a git checkout, managed venv, or system pip."""
    pkg_dir = Path(__file__).resolve().parent
    candidate_repo = pkg_dir.parent.parent
    if (candidate_repo / ".git").exists():
        return ("git-editable", candidate_repo)

    managed_root = Path.home() / ".local" / "share" / "kardashev" / "repo"
    if (managed_root / ".git").exists():
        return ("managed-venv", managed_root)

    return ("pip", None)


def _local_git_head(repo_root: Path | None) -> str | None:
    if repo_root is None or not (repo_root / ".git").exists():
        return None
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        return proc.stdout.strip() if proc.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _update_cache_path() -> Path:
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    d = base / "kardashev"
    d.mkdir(parents=True, exist_ok=True)
    return d / "update_check.json"


def _is_remote_ahead(repo_root: Path | None, local_sha: str | None, remote_sha: str | None) -> bool:
    """Returns True ONLY if remote_sha differs from local_sha AND local_sha is not already ahead of remote_sha."""
    if not local_sha or not remote_sha or local_sha == remote_sha:
        return False
    if repo_root is not None and (repo_root / ".git").exists():
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            proc = subprocess.run(
                ["git", "-C", str(repo_root), "merge-base", "--is-ancestor", remote_sha, local_sha],
                capture_output=True,
                timeout=1.5,
                check=False,
            )
            if proc.returncode == 0:
                return False
    return True


def _version_tuple(v: str) -> tuple[int, ...]:
    parts: list[int] = []
    for p in str(v).lstrip("v").split("."):
        if p.isdigit():
            parts.append(int(p))
    return tuple(parts)


def check_for_updates(*, force: bool = False, timeout: float = 2.5) -> UpdateStatus:
    """Checks GitHub for newer releases or commits on `main`, caching results for 12 hours."""
    install_mode, repo_root = detect_install_layout()
    local_sha = _local_git_head(repo_root)
    cache_file = _update_cache_path()

    if not force and cache_file.is_file():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if time.time() - float(cached.get("checked_at", 0)) < UPDATE_CHECK_TTL_SECONDS:
                remote_sha = cached.get("remote_commit")
                latest_ver = str(cached.get("latest_version", __version__))
                up_avail = bool(
                    _is_remote_ahead(repo_root, local_sha, remote_sha)
                    or (not local_sha and latest_ver and _version_tuple(latest_ver) > _version_tuple(__version__))
                )
                msg = (
                    f"Update available ({local_sha[:7] if local_sha else __version__} -> "
                    f"{remote_sha[:7] if remote_sha else latest_ver}). Run `kardashev update`."
                    if up_avail
                    else f"kardashev v{__version__} ({local_sha[:7] if local_sha else 'release'}) is up to date."
                )
                return UpdateStatus(
                    current_version=__version__,
                    latest_version=latest_ver,
                    local_commit=local_sha,
                    remote_commit=remote_sha,
                    update_available=up_avail,
                    install_mode=install_mode,
                    repo_root=str(repo_root) if repo_root else None,
                    message=msg,
                )
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    latest_ver = __version__
    remote_sha: str | None = None

    headers = {"Accept": "application/vnd.github+json", "User-Agent": f"kardashev-cli/{__version__}"}
    try:
        req = urllib.request.Request(GITHUB_API_RELEASES, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            tag = str(data.get("tag_name", "")).lstrip("v")
            if tag:
                latest_ver = tag
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        pass

    try:
        req2 = urllib.request.Request(GITHUB_API_MAIN_COMMIT, headers=headers)
        with urllib.request.urlopen(req2, timeout=timeout) as resp2:
            data2 = json.loads(resp2.read().decode("utf-8"))
            sha = str(data2.get("sha", "")).strip()
            if sha:
                remote_sha = sha
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        pass

    try:
        cache_file.write_text(
            json.dumps(
                {
                    "checked_at": time.time(),
                    "latest_version": latest_ver,
                    "remote_commit": remote_sha,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except OSError:
        pass

    up_avail = bool(
        _is_remote_ahead(repo_root, local_sha, remote_sha)
        or (not local_sha and latest_ver and _version_tuple(latest_ver) > _version_tuple(__version__))
    )
    msg = (
        f"Update available ({local_sha[:7] if local_sha else __version__} -> "
        f"{remote_sha[:7] if remote_sha else latest_ver}). Run `kardashev update`."
        if up_avail
        else f"kardashev v{__version__} ({local_sha[:7] if local_sha else 'release'}) is up to date."
    )
    return UpdateStatus(
        current_version=__version__,
        latest_version=latest_ver,
        local_commit=local_sha,
        remote_commit=remote_sha,
        update_available=up_avail,
        install_mode=install_mode,
        repo_root=str(repo_root) if repo_root else None,
        message=msg,
    )


def ensure_bin_symlinks() -> list[str]:
    """Ensures `~/.local/bin/{kardashev,ksh,kardashev-mcp,opencommons-mcp}` point to the active venv scripts."""
    bin_dir = Path(sys.executable).parent
    user_bin = Path.home() / ".local" / "bin"
    user_bin.mkdir(parents=True, exist_ok=True)
    linked: list[str] = []

    primary_cli = bin_dir / "kardashev"
    primary_mcp = bin_dir / "kardashev-mcp"

    targets = {
        "kardashev": primary_cli,
        "ksh": primary_cli,
        "kardashev-mcp": primary_mcp,
        "opencommons-mcp": bin_dir / "opencommons-mcp",
    }
    for name, src in targets.items():
        if not src.exists():
            continue
        dst = user_bin / name
        try:
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(src)
            linked.append(str(dst))
        except OSError:
            pass
    return linked


def perform_self_update(*, include_litert: bool = True) -> tuple[bool, str]:
    """Performs an in-place self-update of `kardashev-cli`."""
    install_mode, repo_root = detect_install_layout()
    pip_cmd = [sys.executable, "-m", "pip", "install", "--upgrade"]
    uv_bin = shutil.which("uv")

    if repo_root is not None and (repo_root / ".git").exists():
        pull = subprocess.run(
            ["git", "-C", str(repo_root), "pull", "--ff-only", "origin", "main"],
            capture_output=True,
            text=True,
            check=False,
        )
        if pull.returncode != 0:
            return (False, f"git pull failed: {(pull.stderr or pull.stdout).strip()}")

        spec = f"{repo_root}[litert]" if include_litert else str(repo_root)
        if uv_bin:
            inst = subprocess.run(
                [uv_bin, "pip", "install", "--python", sys.executable, "-e", spec],
                capture_output=True,
                text=True,
                check=False,
            )
        else:
            inst = subprocess.run(
                [*pip_cmd, "-e", spec],
                capture_output=True,
                text=True,
                check=False,
            )
        if inst.returncode != 0:
            return (False, f"Dependency upgrade failed: {(inst.stderr or inst.stdout).strip()}")

        linked = ensure_bin_symlinks()
        new_sha = _local_git_head(repo_root) or __version__
        check_for_updates(force=True)
        return (
            True,
            f"Updated kardashev-cli ({install_mode}) to {new_sha[:8]} and linked {len(linked)} binaries in ~/.local/bin.",
        )

    git_url = f"git+https://github.com/{GITHUB_REPO}.git"
    if include_litert:
        git_url = f"kardashev-cli[litert] @ git+https://github.com/{GITHUB_REPO}.git"
    inst = subprocess.run(
        [*pip_cmd, git_url],
        capture_output=True,
        text=True,
        check=False,
    )
    if inst.returncode != 0:
        return (False, f"pip upgrade failed: {(inst.stderr or inst.stdout).strip()}")
    linked = ensure_bin_symlinks()
    check_for_updates(force=True)
    return (True, f"Updated kardashev-cli from GitHub main and linked {len(linked)} binaries.")


def generate_shell_completion(shell: str) -> str:
    """Returns shell tab-completion script for `bash`, `zsh`, or `fish`."""
    subcommands = "chat exec serve mcp commons scout integrate presets models doctor import-litert update config completions"
    presets = "gemma4-26b glm4-9b-ollama glm4-vllm glm-zhipu-api qwen3.6-35b-vllm qwen3.8-27b-vllm qwen3-vl-8b-vllm gpt-oss-20b-vllm lm-studio"
    flags = "--preset --backend --model --model-path --base-url --workspace --policy --litert-device --speculative --with-chronosml --with-opencommons --ralph --plain --dry-run --json --help --version"

    if shell == "zsh":
        return f"""#compdef kardashev ksh
_kardashev() {{
  local -a subcmds presets
  subcmds=({subcommands})
  presets=({presets})
  if (( CURRENT == 2 )); then
    _describe 'command' subcmds
    return
  fi
  case "$words[CURRENT-1]" in
    -p|--preset)
      _describe 'preset' presets
      ;;
    --policy)
      _values 'policy' allow-all ask sandbox
      ;;
    *)
      _arguments '*: :({flags})'
      ;;
  esac
}}
compdef _kardashev kardashev ksh
"""

    if shell == "fish":
        lines = [
            "# Fish completions for kardashev & ksh",
            f"complete -c kardashev -f -n '__fish_use_subcommand' -a '{subcommands}'",
            f"complete -c ksh -f -n '__fish_use_subcommand' -a '{subcommands}'",
            f"complete -c kardashev -l preset -s p -a '{presets}'",
            f"complete -c ksh -l preset -s p -a '{presets}'",
            "complete -c kardashev -l policy -a 'allow-all ask sandbox'",
            "complete -c ksh -l policy -a 'allow-all ask sandbox'",
        ]
        return "\n".join(lines) + "\n"

    return f"""# Bash completion for kardashev & ksh
_kardashev_completions() {{
  local cur prev subcmds presets flags
  COMPREPLY=()
  cur="${{COMP_WORDS[COMP_CWORD]}}"
  prev="${{COMP_WORDS[COMP_CWORD-1]}}"
  subcmds="{subcommands}"
  presets="{presets}"
  flags="{flags}"

  if [[ ${{COMP_CWORD}} -eq 1 ]]; then
    COMPREPLY=( $(compgen -W "${{subcmds}} ${{flags}}" -- "${{cur}}") )
    return 0
  fi

  case "${{prev}}" in
    -p|--preset)
      COMPREPLY=( $(compgen -W "${{presets}}" -- "${{cur}}") )
      return 0
      ;;
    --policy)
      COMPREPLY=( $(compgen -W "allow-all ask sandbox" -- "${{cur}}") )
      return 0
      ;;
  esac

  COMPREPLY=( $(compgen -W "${{flags}}" -- "${{cur}}") )
  return 0
}}
complete -F _kardashev_completions kardashev ksh
"""
