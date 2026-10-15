#!/usr/bin/env bash
# ==============================================================================
# agy-local — One-Line Standalone CLI Installer (Claude Code / Codex CLI style)
# The Kardashev Project (https://github.com/kardashevproject/agy-local)
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/kardashevproject/agy-local/main/install.sh | bash
# ==============================================================================
set -euo pipefail

REPO_URL="${AGY_LOCAL_REPO_URL:-https://github.com/kardashevproject/agy-local.git}"
INSTALL_ROOT="${AGY_LOCAL_HOME:-$HOME/.local/share/agy-local}"
REPO_DIR="$INSTALL_ROOT/repo"
VENV_DIR="$INSTALL_ROOT/venv"
BIN_DIR="${AGY_LOCAL_BIN_DIR:-$HOME/.local/bin}"

printf "\033[1;36m✦ Antigravity Local Studio Installer (agy-local)\033[0m\n"
printf "  Target directory : %s\n" "$INSTALL_ROOT"
printf "  Binary symlink   : %s/{agy-local,agy,agy-local-mcp}\n\n" "$BIN_DIR"

mkdir -p "$INSTALL_ROOT" "$BIN_DIR"

if [ -d "$REPO_DIR/.git" ]; then
  printf "\033[0;34m→ Updating existing agy-local repository...\033[0m\n"
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

ln -sf "$VENV_DIR/bin/agy-local" "$BIN_DIR/agy-local"
ln -sf "$VENV_DIR/bin/agy-local" "$BIN_DIR/agy"
ln -sf "$VENV_DIR/bin/agy-local-mcp" "$BIN_DIR/agy-local-mcp"

printf "\n\033[1;32m✓ agy-local installed successfully!\033[0m\n"
printf "  Run \033[1;36magy-local\033[0m (or \033[1;36magy\033[0m) to launch the interactive studio.\n"
printf "  Run \033[1;36magy-local update\033[0m anytime to self-update.\n"
