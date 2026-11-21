# CI-style entry points. `make check` must be green before every commit.
UV ?= uv
PY := $(UV) run

.PHONY: check lint format type test test-fast smoke web bench-mac hardcode

check: lint type test

lint:
	$(PY) ruff check tinylab viewer/server tests
	$(PY) ruff format --check tinylab viewer/server tests

format:
	$(PY) ruff format tinylab viewer/server tests
	$(PY) ruff check --fix tinylab viewer/server tests

type:
	$(PY) mypy

test:
	$(PY) pytest -q

test-fast:
	$(PY) pytest -q -m "not slow and not distributed"

smoke:
	$(PY) tinylab pipeline configs/pipeline/smoke.yaml

web:
	cd viewer/web && npm ci && npm run build

bench-mac:
	$(PY) tinylab bench configs/pipeline/mac.yaml

hardcode:
	$(PY) pytest -q tests/test_hardcode.py
