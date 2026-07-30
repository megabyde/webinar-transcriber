# Development

Set up a checkout, run the CLI without installing it, and verify changes with the quality gate.
Coding conventions, testing notes, and the Definition of Done live in [AGENTS.md](../AGENTS.md).

The setup is complete when `uv run webinar-transcriber --version` succeeds. A change is ready for
review when `make format` leaves the intended diff and `make check` exits successfully with 100%
coverage.

## Prerequisites

- [Python 3.12+](https://www.python.org/downloads/) and
  [uv](https://docs.astral.sh/uv/getting-started/installation/)
- For CUDA development: [CMake](https://cmake.org/download/), a C/C++ compiler, and a working
  [CUDA toolkit](https://developer.nvidia.com/cuda-downloads) with `nvcc` on `PATH` and `CUDA_HOME`
  set

## Local setup

Choose one environment. `make sync` is the default; the other targets include the standard
development dependencies plus their named runtime support.

- `make sync`: standard development and test dependencies.
- `make sync-llm`: development dependencies plus cloud LLM support.
- `make sync-cuda`: development environment with CUDA-built `pywhispercpp`; confirm that the CUDA
  prerequisites above are installed first.

### Setup without `make`

Run the command that matches the environment you need:

```bash
# Standard development
uv sync

# Development with cloud LLM support
uv sync --extra llm

# Development with NVIDIA CUDA
GGML_CUDA=1 uv sync \
    --reinstall-package pywhispercpp \
    --no-binary-package pywhispercpp
```

`make sync*` prepares the checkout for development. To register the checkout as a global CLI, follow
[Install the CLI from this checkout](../README.md#install-the-cli-from-this-checkout). Use sync
while working on the code; install when you want to run the checkout as a tool.

Verify the selected environment:

```console
$ uv run webinar-transcriber --version
webinar-transcriber, version X.Y.Z
```

Setup is complete when the command prints the project version without an import error.

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

`make test` should exit successfully. It skips slow tests and the coverage gate, so use it only for
the edit-test loop.

Before committing, run the full gate:

```bash
make format
make check
```

`make format` may update Markdown or Python files. Review those changes before continuing. Then run
`make check`, which runs Markdown checks, Ruff, `ty`, and the full coverage-gated pytest suite. If
it fails, fix the first reported problem and rerun `make check`.

The quality gate is complete when `make check` exits successfully and pytest reports 100% coverage.
Without `make`, run the equivalent `uv run ...` commands; the `Makefile` is the source of truth for
each target's exact recipe.

Run `make help` for the full list of targets (including `clean` and `distclean`) with one-line
descriptions. The CLI install commands are documented in the [README](../README.md#install).
