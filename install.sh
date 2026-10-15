#!/usr/bin/env bash
# ==============================================================================
# kardashev-cli — One-Line Standalone CLI Installer (Claude Code / Codex style)
# The Kardashev Project (https://github.com/kardashevproject/kardashev-cli)
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/kardashevproject/kardashev-cli/main/install.sh | bash
# ==============================================================================
set -euo pipefail

REPO_URL="${KARDASHEV_REPO_URL:-${AGY_LOCAL_REPO_URL:-https://github.com/kardashevproject/kardashev-cli.git}}"
INSTALL_ROOT="${AGY_LOCAL_HOME:-$HOME/.local/share/agy-local}"
REPO_DIR="$INSTALL_ROOT/repo"
VENV_DIR="$INSTALL_ROOT/venv"
BIN_DIR="${AGY_LOCAL_BIN_DIR:-$HOME/.local/bin}"

printf "\033[1;36m✦ Kardashev CLI & OpenCommons Studio Installer (kardashev / ksh / agy)\033[0m\n"
printf "  Target directory : %s\n" "$INSTALL_ROOT"
printf "  Binary symlinks  : %s/{kardashev,ksh,agy,agy-local,kardashev-mcp,opencommons-mcp}\n\n" "$BIN_DIR"

mkdir -p "$INSTALL_ROOT" "$BIN_DIR"

if [ -d "$REPO_DIR/.git" ]; then
  printf "\033[0;34m→ Updating existing kardashev-cli repository...\033[0m\n"
  git -C "$REPO_DIR" pull --ff-only origin main
else
  printf "\033[0;34m→ Cloning %s...\033[0m\n" "$REPO_URL"
  rm -rf "$REPO_DIR"
  git clone --depth 1 "$REPO_URL" "$REPO_DIR"
fi

if command -v uv >/dev/null 2>&1; then
  printf "\033[0;34m→ Creating managed Python environment with uv...\033[0m\n"
  if [ ! -d "$VENV_DIR" ]; then
    uv venv "$VENV_DIR"
  fi
  uv pip install --python "$VENV_DIR/bin/python" -e "$REPO_DIR[litert]"
else
  printf "\033[0;34m→ Creating managed Python environment with venv + pip...\033[0m\n"
  if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
  fi
  "$VENV_DIR/bin/pip" install --upgrade pip
  "$VENV_DIR/bin/pip" install -e "$REPO_DIR[litert]"
fi

ln -sf "$VENV_DIR/bin/kardashev" "$BIN_DIR/kardashev"
ln -sf "$VENV_DIR/bin/kardashev" "$BIN_DIR/ksh"
ln -sf "$VENV_DIR/bin/agy-local" "$BIN_DIR/agy-local"
ln -sf "$VENV_DIR/bin/agy-local" "$BIN_DIR/agy"
ln -sf "$VENV_DIR/bin/kardashev-mcp" "$BIN_DIR/kardashev-mcp"
ln -sf "$VENV_DIR/bin/agy-local-mcp" "$BIN_DIR/agy-local-mcp"
ln -sf "$VENV_DIR/bin/opencommons-mcp" "$BIN_DIR/opencommons-mcp"

printf "\n\033[1;32m✓ kardashev-cli installed successfully!\033[0m\n"
printf "  Run \033[1;36mkardashev\033[0m (or \033[1;36mksh\033[0m / \033[1;36magy\033[0m) to launch the interactive studio.\n"
printf "  Run \033[1;36mkardashev update\033[0m anytime to self-update.\n"
