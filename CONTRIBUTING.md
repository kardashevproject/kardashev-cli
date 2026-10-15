# Contributing to kardashev-cli

Thanks for helping. This file covers the code in this repository. To contribute concepts to OpenCommons, use `opencommons-mcp` or `kardashev commons` instead; you don't need a pull request.

## Setup

```bash
git clone https://github.com/kardashevproject/kardashev-cli.git
cd kardashev-cli
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

Add `[litert]` to the extras if you work on the on-device LiteRT backend.

## Before you open a pull request

```bash
make lint    # ruff check
make test    # pytest
.venv/bin/python benchmarks/evaluate_opencommons.py --store commons/store.json
```

CI runs the same three steps on every pull request.

## Guidelines

- **Keep changes focused.** One fix or feature per pull request, with a test that fails without it.
- **Claims must be true.** Docs, help text, and commit messages describe what the code does today. If a feature is partial, say so.
- **Keep MCP tool schemas small.** Tool descriptions are loaded into every connected agent's context. Keep them short and static, and never put concept content in them.
- **No personal or machine identifiers.** Don't commit home-directory paths, usernames, hostnames, IP addresses, hardware serials, or keys, including in tests and examples. Use `~` or a placeholder.
- **Style.** `ruff check` must pass. Match the surrounding code; use type hints on public functions.

## Commit messages

Use the conventional prefixes (`fix:`, `feat:`, `docs:`, `test:`, `refactor:`, `chore:`), an optional scope (`fix(mcp): ...`), and a plain, factual summary.

## License

By contributing, you agree that your contributions are licensed under the [Apache-2.0 License](LICENSE).
