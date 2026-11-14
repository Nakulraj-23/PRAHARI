PY ?= python
VENV ?= .venv
BIN := $(VENV)/bin

.PHONY: setup seed train serve web test bench lint
setup:
	test -d $(VENV) || python3.11 -m venv $(VENV)
	$(BIN)/pip install -q -e ".[dev]"

seed:
	@echo "seed: not built yet"
train:
	@echo "train: not built yet"
serve:
	@echo "serve: not built yet"
web:
	@echo "web: not built yet"
bench:
	@echo "bench: not built yet"

test:
	$(BIN)/python -m pytest -q
lint:
	$(BIN)/ruff check .
