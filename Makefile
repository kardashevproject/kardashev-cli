.PHONY: test lint format benchmark doctor models chat

VENV_BIN := .venv/bin

test:
	$(VENV_BIN)/pytest

lint:
	$(VENV_BIN)/ruff check src tests benchmarks examples

format:
	$(VENV_BIN)/ruff format src tests benchmarks examples

benchmark:
	$(VENV_BIN)/python benchmarks/evaluate_opencommons.py --store commons/store.json

doctor:
	$(VENV_BIN)/kardashev doctor

models:
	$(VENV_BIN)/kardashev models

chat:
	$(VENV_BIN)/kardashev chat --preset gemma4-26b
