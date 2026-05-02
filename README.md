# Webinar Transcriber

[![CI](https://github.com/megabyde/webinar-transcriber/actions/workflows/ci.yml/badge.svg)](https://github.com/megabyde/webinar-transcriber/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3120/)
[![Coverage 100%](https://img.shields.io/badge/coverage-100%25-brightgreen)](https://github.com/megabyde/webinar-transcriber/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/badge/lint-ruff-D7FF64?logo=ruff&logoColor=1D2B34)](https://docs.astral.sh/ruff/)
[![uv](https://img.shields.io/badge/package-uv-5C5CFF?logo=uv&logoColor=white)](https://docs.astral.sh/uv/)
[![License MIT](https://img.shields.io/badge/license-MIT-2F855A)](LICENSE)

## Overview

`webinar-transcriber` is a local-first CLI for turning webinar recordings into transcripts,
structured notes, diagnostics, and machine-readable report artifacts. It handles both audio-only
input and slide-based video, detects scenes for video runs, and keeps the core pipeline local with
PyAV, Silero VAD, and `whisper.cpp`.

The default flow is deterministic: normalize the media, detect speech regions, transcribe locally,
reconcile overlapping windows, and build report sections with local heuristics. Optional
provider-backed LLM refinement can polish section text and add summary bullets, action items,
section titles, and section TL;DRs on top of that local output, but it does not replace the base
pipeline.

## Install

### Prerequisites

- Python 3.12 and `uv`.
- CUDA source builds also require a C/C++ compiler, `cmake`, and a working CUDA toolkit (`nvcc` on
  `PATH`, `CUDA_HOME` set).

### Install The CLI From This Checkout

From the repository root:

```bash
make install
```

That installs the `webinar-transcriber` command as a uv tool from the local source tree. You do not
need to activate a virtual environment to use the installed CLI. Re-run `make install` after pulling
changes when you want the installed command to use the current checkout.

```bash
webinar-transcriber --help
```

Use the matching Make target for the install you need:

- `make install`: standard local CLI install.
- `make install-llm`: CLI plus optional OpenAI/Anthropic LLM dependencies.
- `make install-cuda`: CLI with `pywhispercpp` rebuilt from source for NVIDIA.
- `make uninstall`: remove the installed `webinar-transcriber` uv tool.

The standard and LLM install targets pull the published `pywhispercpp` wheels from PyPI. On Linux
and Windows, those wheels use the CPU backend. On macOS, the Apple Silicon wheels include Metal
support. The default `large-v3-turbo` model is downloaded on first transcription run, not during
installation. To verify the active backend after a run, inspect `diagnostics.json` →
`asr_pipeline.system_info`. Native whisper.cpp initialization and teardown logs are written to
`whisper-cpp.log` in the run directory.

To inspect all available project commands:

```bash
make help
```

### NVIDIA (CUDA on Linux or Windows)

CUDA is the only supported path that builds `pywhispercpp` from source. Use the CUDA target that
matches your workflow:

- `make install-cuda` for the installed CLI tool
- `make sync-cuda` for the checkout development environment

All usage examples below assume `webinar-transcriber` is available on your `PATH`.

## Usage

### Quick Start

`webinar-transcriber` is a single root command, not a subcommand CLI. By default, it writes a fresh
run directory under `runs/`. Use `--output-dir` to choose a specific location.

```bash
webinar-transcriber INPUT
webinar-transcriber INPUT --keep-audio mp3
webinar-transcriber INPUT --output-dir runs/custom-demo
```

### Cloud LLM

The optional `--llm` flag enables provider-backed report refinement after deterministic sectioning.
It can:

- polish section transcript text with light cleanup and paragraphing
- refine summary bullets, action items, section titles, and section TL;DRs

Supported providers:

- `openai` (default)
- `anthropic`

#### Install the LLM extra

The base install does not include the optional provider SDKs. If you want to use `--llm`, reinstall
the CLI from this checkout with the `llm` extra:

```bash
make install-llm
```

LLM configuration comes only from environment variables.

#### OpenAI

```bash
OPENAI_API_KEY=... \
    OPENAI_MODEL=gpt-5-mini \
    webinar-transcriber INPUT --llm
```

#### Anthropic

```bash
LLM_PROVIDER=anthropic \
    ANTHROPIC_API_KEY=... \
    ANTHROPIC_MODEL=claude-sonnet-4-20250514 \
    webinar-transcriber INPUT --llm
```

### What You Get

Successful default runs write:

- `metadata.json` and `transcript.json`
- `diagnostics.json` with stage timings, counts, warnings, ASR details, and optional LLM metadata
- `report.md`, `report.docx`, and `report.json`
- ASR planning and decode artifacts under `asr/`

Depending on options and input type, successful default runs also write:

- `transcription-audio.wav` or `transcription-audio.mp3` with `--keep-audio FORMAT`
- `scenes.json` and `frames/` for video input

Failed runs also write `diagnostics.json` once the run directory exists, though they may still leave
only partial intermediate artifacts and no final report outputs.

### How It Works

1. Probe the input and prepare deterministic transcription audio. This means a mono, `16 kHz`,
   `16-bit PCM WAV` file that the downstream pipeline can treat as a stable contract instead of
   re-checking format details at every stage.
1. Prepare the local `whisper.cpp` runtime. This loads the selected model identifier or GGML model
   path, resolves the execution backend, and records runtime details for diagnostics and CLI
   progress reporting.
1. Detect speech regions with Silero VAD.
1. Create ASR windows from the detected speech regions.
1. Transcribe the windows locally with `whisper.cpp`.
1. Reconcile adjacent windows into one transcript.
1. Detect scenes and extract representative frames for video input.
1. Build report sections with local heuristics. For audio-only inputs, sectioning is best-effort and
   uses speech gaps; LLM refinement is the preferred path for stronger headings and metadata.
1. Optionally polish the report with an LLM, including summary bullets and action items.
1. Write report, diagnostic, and intermediate artifacts.

## Advanced Usage

### ASR Model

`webinar-transcriber` uses `pywhispercpp` to resolve whisper.cpp models. By default, it uses the
built-in `large-v3-turbo` model identifier for fast, high-quality local transcription, which
`pywhispercpp` downloads into its cache on first use. You can also pass another model identifier or
a local GGML model path with `--asr-model`.

For example:

```bash
webinar-transcriber INPUT --asr-model large-v3-turbo
webinar-transcriber INPUT --asr-model large-v3
webinar-transcriber INPUT --asr-model models/whisper-cpp/ggml-large-v3-turbo.bin
```

To manage the file yourself, download it directly:

```bash
mkdir -p models/whisper-cpp
curl -L \
    -o models/whisper-cpp/ggml-large-v3-turbo.bin \
    https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin
```

Use `--asr-model large-v3` when maximum transcription accuracy is more important than local runtime.

### Speech Detection

Silero VAD is enabled by default and uses the project's selected speech-region defaults. Pass
`--no-vad` only when you need to transcribe the full audio without speech-region planning.

### ASR Controls

The default ASR path uses the selected `whisper.cpp` model, automatic language detection, bounded
prompt carryover, and an automatically selected thread count.

- `--language CODE`: force a Whisper language code such as `en` or `ru`.
- `--threads N`: set the CPU worker count passed to `whisper.cpp`. GPU-enabled builds can offload
  supported model work to the GPU, but `whisper.cpp` still uses CPU threads for non-offloaded work,
  scheduling, and language detection. By default, the transcriber uses the host CPU count; lower
  this when you need to leave CPU capacity for other work.

## Reference

### Output Layout

Successful default runs write:

```text
runs/<timestamp>_<basename>/
├─ asr/
│  ├─ decoded_windows.json
│  └─ speech_regions.json
├─ frames/                 # video only
├─ diagnostics.json
├─ metadata.json
├─ report.docx
├─ report.json
├─ report.md
├─ scenes.json             # video only
├─ transcript.json
├─ transcription-audio.mp3 # optional via --keep-audio mp3
└─ transcription-audio.wav # optional via --keep-audio wav
```

## Development

### Local Setup

1. Install Python 3.12 and `uv`.
1. Sync the checkout environment you need:

- `make sync`: standard development and test dependencies.
- `make sync-llm`: development dependencies plus optional LLM SDKs.
- `make sync-cuda`: development environment with CUDA-built `pywhispercpp`.

### Running from a Checkout

If you are developing inside the repository and do not want to install the CLI as a tool, run it
through the checkout environment:

```bash
uv run webinar-transcriber --help
uv run webinar-transcriber INPUT
```

### Toolchain

- Python 3.12
- `uv`
- `ruff`
- `ty`
- `pytest`
- `pytest-cov`

### Quality Gates

Every logical implementation step is expected to pass the full verification cycle before it is
committed:

```bash
make format
make lint
make check
```

For quick local iteration, `make test` skips tests marked `slow`. Use `make check` to run the full
coverage-gated suite.

The repository keeps tiny committed media fixtures so pipeline tests can run without network fetches
or large binary blobs. Contributor and agent-specific repository guidance lives in
[AGENTS.md](AGENTS.md).
