MD_FILES := $(shell git ls-files '*.md')

COLOR_CYAN := \033[36m
COLOR_RESET := \033[0m

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help message
	@awk 'BEGIN {FS = ":.*##"; printf "Usage: make [options] $(COLOR_CYAN)[target] ...$(COLOR_RESET)\n\n"} \
	/^[a-zA-Z_-]+:.*##/ {printf "  $(COLOR_CYAN)%-16s$(COLOR_RESET) %s\n", $$1, $$2}' \
	$(MAKEFILE_LIST)

.PHONY: sync
sync: ## Sync the development environment
	uv sync

.PHONY: sync-reinstall
sync-reinstall: ## Rebuild pywhispercpp from source for CUDA
	GGML_CUDA=1 uv sync --reinstall-package pywhispercpp --no-binary-package pywhispercpp

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
test: ## Run the fast pytest subset (skips slow tests, no coverage)
	uv run pytest --no-cov -m "not slow"

.PHONY: test-all
test-all: ## Run the full pytest suite with coverage
	uv run pytest

.PHONY: check
check: lint test-all ## Run lint and the full test suite

.PHONY: clean
clean: ## Remove caches and build artifacts
	rm -rf .coverage .mypy_cache .pytest_cache .ruff_cache .venv coverage.xml dist build
	find webinar_transcriber -type d -name __pycache__ -prune -exec rm -rf {} +
