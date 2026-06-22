# Development

How to set up a checkout, run the CLI without installing it, and pass the quality gate. Coding
conventions, testing notes, and the Definition of Done live in [AGENTS.md](../AGENTS.md).

## Prerequisites

- [Python 3.12+](https://www.python.org/downloads/) and
  [uv](https://docs.astral.sh/uv/getting-started/installation/)
- For CUDA development: [CMake](https://cmake.org/download/), a C/C++ compiler, and a working
  [CUDA toolkit](https://developer.nvidia.com/cuda-downloads) with `nvcc` on `PATH` and `CUDA_HOME`
  set

## Local setup

Sync the checkout environment you need:

- `make sync`: standard development and test dependencies.
- `make sync-llm`: development dependencies plus optional LLM SDKs.
- `make sync-cuda`: development environment with CUDA-built `pywhispercpp`.

Without `make`:

```bash
uv sync
uv sync --extra llm
```

`make sync*` prepares a checkout for development;
[installing as a uv tool](../README.md#install-the-cli-from-this-checkout) registers the built CLI
globally. Use sync while working on the code, install to run the checkout as a tool.

## Running from a checkout

To run the CLI without installing it as a uv tool, use the checkout environment:

```bash
uv run webinar-transcriber --help
uv run webinar-transcriber INPUT
```

## Quality gates

Use the fast test target for iteration:

```bash
make test
```

Before committing, run the full gate:

```bash
make format
make check
```

`make check` runs Markdown checks, Ruff, `ty`, and the full coverage-gated pytest suite. Without
`make`, run the equivalent `uv run ...` commands; the `Makefile` is the source of truth for each
target's exact recipe.

Run `make help` for the full list of targets (including `clean` and `distclean`) with one-line
descriptions. The CLI install targets are documented in the
[README](../README.md#install-the-cli-from-this-checkout).
