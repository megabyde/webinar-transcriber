MD_FILES := $(shell git ls-files '*.md')

COLOR_CYAN := \033[36m
COLOR_RESET := \033[0m

.DEFAULT_GOAL := help

UNAME_S := $(shell uname -s)
HAS_NVCC = $(shell command -v nvcc >/dev/null 2>&1 && echo yes)

ifeq ($(UNAME_S),Darwin)
PYWHISPERCPP_BACKEND ?= metal
else ifeq ($(UNAME_S),Linux)
PYWHISPERCPP_BACKEND ?= $(if $(filter yes,$(HAS_NVCC)),cuda,cpu)
else
PYWHISPERCPP_BACKEND ?= cpu
endif

PYWHISPERCPP_SYNC_ENV = $(if $(filter cuda,$(PYWHISPERCPP_BACKEND)),GGML_CUDA=1)

.PHONY: help
help: ## Show this help message
	@awk 'BEGIN {FS = ":.*##"; printf "Usage: make [options] $(COLOR_CYAN)[target] ...$(COLOR_RESET)\n\n"} \
	/^[a-zA-Z_-]+:.*##/ {printf "  $(COLOR_CYAN)%-16s$(COLOR_RESET) %s\n", $$1, $$2}' \
	$(MAKEFILE_LIST)

.PHONY: sync
sync: ## Sync with the detected (or overridden) pywhispercpp backend
	@echo ">>> pywhispercpp backend: $(PYWHISPERCPP_BACKEND)"
	$(PYWHISPERCPP_SYNC_ENV) uv sync --group dev

.PHONY: sync-reinstall
sync-reinstall: ## Rebuild only pywhispercpp with the current backend
	@echo ">>> pywhispercpp backend: $(PYWHISPERCPP_BACKEND)"
	$(PYWHISPERCPP_SYNC_ENV) uv sync --group dev --reinstall-package pywhispercpp

.PHONY: backend-info
backend-info: ## Print detected OS and resolved backend
	@echo "uname -s : $(UNAME_S)"
	@echo "has nvcc : $(HAS_NVCC)"
	@echo "backend  : $(PYWHISPERCPP_BACKEND)"
	@echo "env      : $(PYWHISPERCPP_SYNC_ENV)"

.PHONY: format
format: ## Format Markdown and Python sources
	uv run mdformat $(MD_FILES)
	uv run ruff format .

.PHONY: lint
lint: ## Run Markdown, Ruff, and type checks
	uv run mdformat --check $(MD_FILES)
	uv run pymarkdown scan $(MD_FILES)
	uv run ruff format --check .
	uv run ruff check .
	uv run ty check webinar_transcriber

.PHONY: test
test: ## Run pytest with coverage
	uv run pytest

.PHONY: check
check: lint test ## Run lint and test

.PHONY: clean
clean: ## Remove caches and build artifacts
	rm -rf .coverage .mypy_cache .pytest_cache .ruff_cache .venv coverage.xml dist build
	find webinar_transcriber -type d -name __pycache__ -prune -exec rm -rf {} +
