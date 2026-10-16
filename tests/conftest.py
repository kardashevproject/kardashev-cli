"""Shared pytest configuration for worktree isolation and SDK resolution.

Ensures that:
1. Running pytest inside any git worktree always imports ``kardashev_cli`` from
   that worktree's own ``src/`` directory rather than an editable install
   pointing to the main checkout.
2. When pytest is invoked with a system Python outside the repo ``.venv``, the
   worktree or parent repository ``.venv`` site-packages is added so real
   ``google.antigravity`` bindings are available.
"""

from __future__ import annotations

import importlib.util
import site
import subprocess
import sys
from pathlib import Path


def _configure_worktree_and_venv() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    worktree_src = str(repo_root / "src")
    if sys.path[:1] != [worktree_src]:
        while worktree_src in sys.path:
            sys.path.remove(worktree_src)
        sys.path.insert(0, worktree_src)

    # Evict any pre-imported kardashev_cli modules from a different checkout.
    for mod_name in list(sys.modules):
        if mod_name == "kardashev_cli" or mod_name.startswith("kardashev_cli."):
            mod = sys.modules.get(mod_name)
            mod_file = getattr(mod, "__file__", None)
            if mod_file and not str(Path(mod_file).resolve()).startswith(worktree_src):
                del sys.modules[mod_name]

    try:
        if importlib.util.find_spec("google.antigravity") is not None:
            return
    except (ModuleNotFoundError, ValueError):
        pass

    candidate_roots = [repo_root]
    git_entry = repo_root / ".git"
    if git_entry.is_file():
        res = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--git-common-dir"],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            common_dir = Path(res.stdout.strip())
            if not common_dir.is_absolute():
                common_dir = (repo_root / common_dir).resolve()
            if common_dir.name == ".git":
                candidate_roots.append(common_dir.parent)

    for root in candidate_roots:
        venv_lib = root / ".venv" / "lib"
        if venv_lib.is_dir():
            for sp in sorted(venv_lib.glob("python*/site-packages")):
                if sp.is_dir():
                    site.addsitedir(str(sp))


_configure_worktree_and_venv()
