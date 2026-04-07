.PHONY: help sync format lint test check clean

MD_FILES := $(shell git ls-files '*.md')

help: ## Show available targets
	@printf "Available targets:\n"
	@sed -n 's/^\([^:#[:space:]][^:]*\):.*##[[:space:]]*\(.*\)$$/\1\t\2/p' $(MAKEFILE_LIST) | \
	while IFS=$$(printf '\t') read -r target description; do \
		printf "  %-10s %s\n" "$$target" "$$description"; \
	done

sync: ## Install project and dev dependencies with uv
	uv sync --group dev

format: ## Format Markdown and Python sources
	uv run mdformat $(MD_FILES)
	uv run ruff format .

lint: ## Run Markdown, Ruff, and type checks
	uv run mdformat --check $(MD_FILES)
	uv run pymarkdown scan $(MD_FILES)
	uv run ruff format --check .
	uv run ruff check .
	uv run ty check webinar_transcriber

test: ## Run pytest with coverage
	uv run pytest

check: lint test ## Run lint and test

clean: ## Remove caches and build artifacts
	rm -rf .coverage .mypy_cache .pytest_cache .ruff_cache .venv coverage.xml dist build
	find webinar_transcriber -type d -name __pycache__ -prune -exec rm -rf {} +
