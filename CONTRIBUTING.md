# Contributing to ChronosML

Thank you for your interest in contributing to **ChronosML**! We welcome pull requests, bug reports, hardware calibration profiles, and architectural Roofline extensions from the community.

---

## Development Setup

```bash
git clone https://github.com/kardashevproject/chronosml.git
cd chronosml
pip install -e ".[dev]"
```

## Running Verification & Benchmarks

Before submitting a Pull Request, please ensure all tests, hardware calibration checks, and empirical benchmarks pass:

```bash
# 1. Run hardware calibration suite (<2s)
make calibrate

# 2. Run pytest verification suite
make test

# 3. Run empirical PyTorch benchmark suite
make benchmark
```

## Contribution Guidelines

1. **Zero Context Tax for MCP**: Any changes to `chronosml.mcp_server` (`MCP_TOOLS_SCHEMA`) must keep the total serialized tool schema under **300 tokens** (`test_mcp_server_schema_budget_and_jsonrpc_protocol` enforces this automatically).
2. **Empirical Grounding**: New model architecture estimators (e.g., new attention variants, MoE routers, or state-space layers) in `chronosml.roofline` must include unit tests in `tests/test_chronosml.py` verifying exact FLOPs and KV/state cache byte math.
3. **Code Style**: Format and lint code with `ruff check src/ tests/ examples/`.

## License

By contributing to ChronosML, you agree that your contributions will be licensed under the [Apache-2.0 License](LICENSE).
