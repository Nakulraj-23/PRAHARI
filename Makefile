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
	$(BIN)/python -m prahari.models.validator.train
	$(BIN)/python -m prahari.models.anomaly.train
	$(BIN)/python -m prahari.models.rca.train
	$(BIN)/python -m prahari.monitoring.drift build-reference
serve:
	$(BIN)/uvicorn prahari.serving.app:app --host 0.0.0.0 --port 8000
web:
	cd web && (test -d node_modules || npm install)
	cd web && VITE_MOCK=$${VITE_MOCK:-1} npm run dev
bench:
	$(BIN)/python -m prahari.monitoring.bench

test:
	$(BIN)/python -m pytest -q
lint:
	$(BIN)/ruff check .
