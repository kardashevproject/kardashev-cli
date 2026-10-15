.PHONY: test lint format doctor models chat

VENV_BIN := .venv/bin

test:
	$(VENV_BIN)/pytest

lint:
	$(VENV_BIN)/ruff check src tests examples

format:
	$(VENV_BIN)/ruff format src tests examples

doctor:
	$(VENV_BIN)/agy-local doctor

models:
	$(VENV_BIN)/agy-local models

chat:
	$(VENV_BIN)/agy-local chat --preset gemma4-26b
