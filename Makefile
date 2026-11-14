PY ?= python
VENV ?= .venv
BIN := $(VENV)/bin

.PHONY: setup seed train serve web test bench lint
setup:
	test -d $(VENV) || python3.11 -m venv $(VENV)
	$(BIN)/pip install -q -e ".[dev]"

seed:
	$(BIN)/python -m prahari.sim.generate --days 90
	$(BIN)/python -m prahari.sim.generate --alt --days 7
train:
	@echo "train: not built yet"
serve:
	@echo "serve: not built yet"
web:
	cd web && (test -d node_modules || npm install)
	cd web && VITE_MOCK=$${VITE_MOCK:-1} npm run dev
bench:
	@echo "bench: not built yet"

test:
	$(BIN)/python -m pytest -q
lint:
	$(BIN)/ruff check .
