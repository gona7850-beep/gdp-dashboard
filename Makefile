# Convenience commands for the Composition Design Platform.
# All targets work from the repo root.

PY ?= python3
PIP ?= $(PY) -m pip
UVICORN_HOST ?= 127.0.0.1
UVICORN_PORT ?= 8000
STREAMLIT_PORT ?= 8501

.DEFAULT_GOAL := help

## help: print this help
.PHONY: help
help:
	@echo "Composition Design Platform — make targets"
	@echo
	@echo "  Verification"
	@echo "    make verify         Run the full end-to-end check (~15s)"
	@echo "    make verify-fast    Skip pytest; use 2 Optuna trials (~7s)"
	@echo "    make test           Run pytest only"
	@echo
	@echo "  Run"
	@echo "    make install        Install all dependencies"
	@echo "    make web            Start FastAPI + web UI at http://$(UVICORN_HOST):$(UVICORN_PORT)"
	@echo "    make streamlit      Start Streamlit at http://$(UVICORN_HOST):$(STREAMLIT_PORT)"
	@echo "    make demo           Run the CLI end-to-end demo on synthetic data"
	@echo "    make docker-up      docker compose up (API:8000 + Streamlit:8501)"
	@echo "    make docker-down    docker compose down"
	@echo
	@echo "  Maintenance"
	@echo "    make clean          Remove __pycache__ and *.pyc"

## install: install all Python deps
.PHONY: install
install:
	$(PIP) install -r requirements.txt

## verify: full end-to-end check
.PHONY: verify
verify:
	$(PY) scripts/verify.py

## verify-fast: skip pytest, use minimal Optuna budget
.PHONY: verify-fast
verify-fast:
	$(PY) scripts/verify.py --fast

## test: pytest only
.PHONY: test
test:
	$(PY) -m pytest tests/ -v

## web: FastAPI + web UI
.PHONY: web
web:
	$(PY) -m uvicorn backend.main:app --reload --host $(UVICORN_HOST) --port $(UVICORN_PORT)

## streamlit: Streamlit research workbench
.PHONY: streamlit
streamlit:
	$(PY) -m streamlit run app/streamlit_app.py --server.address $(UVICORN_HOST) --server.port $(STREAMLIT_PORT)

## demo: CLI end-to-end demo
.PHONY: demo
demo:
	$(PY) examples/alloyforge_demo.py

## docker-up: docker compose up (foreground)
.PHONY: docker-up
docker-up:
	docker compose up --build

## docker-down: docker compose down
.PHONY: docker-down
docker-down:
	docker compose down

## clean: remove caches
.PHONY: clean
clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true
