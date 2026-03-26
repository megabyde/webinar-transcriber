.PHONY: help sync format lint typecheck test coverage check ci clean

MD_FILES := $(shell git ls-files '*.md')

help:
	@printf "Available targets:\n"
	@printf "  sync       Install project and dev dependencies with uv\n"
	@printf "  format     Format Python sources with Ruff\n"
	@printf "  lint       Run Ruff and markdown lint checks\n"
	@printf "  typecheck  Run ty over the package\n"
	@printf "  test       Run pytest\n"
	@printf "  coverage   Run pytest with coverage gate\n"
	@printf "  check      Run lint, typecheck, test, and coverage\n"
	@printf "  ci         Alias for check\n"
	@printf "  clean      Remove caches and build artifacts\n"

sync:
	uv sync --group dev

format:
	uv run mdformat $(MD_FILES)
	uv run ruff format .

lint:
	uv run mdformat --check $(MD_FILES)
	uv run pymarkdown scan $(MD_FILES)
	uv run ruff check .

typecheck:
	uv run ty check webinar_transcriber

test:
	uv run pytest

coverage:
	uv run pytest --cov=webinar_transcriber --cov-report=term-missing --cov-report=xml

check: lint typecheck test coverage

ci: check

clean:
	rm -rf .coverage .mypy_cache .pytest_cache .ruff_cache .venv coverage.xml dist build
	find webinar_transcriber -type d -name __pycache__ -prune -exec rm -rf {} +
