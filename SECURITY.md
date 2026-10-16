# Security Policy

## Supported versions

| Version | Supported |
| ------- | --------- |
| 0.2.x   | Yes       |
| < 0.2   | No        |

## Reporting a vulnerability

Please report vulnerabilities privately. Don't open a public issue.

- Preferred: open a [private security advisory](https://github.com/kardashevproject/kardashev-cli/security/advisories/new).
- Or email `admin@kardashevproject.org`.

Include the affected version or commit, steps to reproduce, and the impact you observed. We aim to acknowledge reports within 3 working days.

## Scope

Reports we're most interested in:

- Code contributed to OpenCommons that escapes its execution limits or reads or writes outside them.
- Ways to influence another user's local store, trust tiers, or quarantine state remotely.
- Prompt injection that reaches an agent through concept text served by `find` or `commons_find`.
- Leaks of personal or machine identifiers through proposals, sync, or logs.
- MCP tools (`kardashev-mcp`, `opencommons-mcp`) acting outside the workspace they were given.

## Current status

OpenCommons hub hardening is in progress. Until a release says otherwise, don't expose `kardashev commons serve-hub` to untrusted networks.
