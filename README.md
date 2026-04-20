# Webinar Transcriber

[![CI](https://github.com/megabyde/webinar-transcriber/actions/workflows/ci.yml/badge.svg)](https://github.com/megabyde/webinar-transcriber/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3120/)
[![Coverage 100%](https://img.shields.io/badge/coverage-100%25-brightgreen)](https://github.com/megabyde/webinar-transcriber/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/badge/lint-ruff-D7FF64?logo=ruff&logoColor=1D2B34)](https://docs.astral.sh/ruff/)
[![uv](https://img.shields.io/badge/package-uv-5C5CFF?logo=uv&logoColor=white)](https://docs.astral.sh/uv/)
[![License MIT](https://img.shields.io/badge/license-MIT-2F855A)](LICENSE)

## Overview

`webinar-transcriber` is a local-first CLI for turning webinar recordings into transcripts,
structured notes, subtitles, diagnostics, and machine-readable report artifacts. It handles both
audio-only input and slide-based video, detects scenes for video runs, and keeps the core pipeline
local with `ffmpeg`, Silero VAD, and `whisper.cpp`.

The default flow is deterministic: normalize the media, detect speech regions, transcribe locally,
reconcile overlapping windows, and build report sections, summary bullets, and action items with
heuristics. Optional provider-backed LLM refinement can polish section text and report metadata on
top of that local output, but it does not replace the base pipeline.

## Install

### Prerequisites

- `ffmpeg` and `ffprobe` (for media handling).
- A C/C++ compiler and `cmake` (needed when `pywhispercpp` builds from source for GPU backends).
- For CUDA: a working CUDA toolkit (`nvcc` on `PATH`, `CUDA_HOME` set).

### Quick Install (CPU Everywhere, Metal by Default on macOS)

```bash
uv tool install .
```

This works on every supported platform. On Linux and Windows, this is the CPU install path. On
macOS, when `pywhispercpp` is source-built, whisper.cpp enables Metal automatically. The default
`large-v3-turbo` model is downloaded on first transcription run, not during `uv tool install .`. To
verify the active backend after a run, look for `Metal = 1` under `diagnostics.json` →
`asr_pipeline.system_info`.

### NVIDIA (CUDA on Linux or Windows)

```bash
GGML_CUDA=1 uv tool install --reinstall .
```

### Switching Backends During Development

If you're working from a repository checkout and want to rebuild only `pywhispercpp` with a
different backend:

```bash
make sync-reinstall PYWHISPERCPP_BACKEND=cuda
```

For contributor/development workflows from a checkout, use `make sync` to create the project
environment. `uv tool install .` is the user-facing CLI install path.

All usage examples below assume `webinar-transcriber` is available on your `PATH`.

## Usage

### Quick Start

`webinar-transcriber` is a single root command, not a subcommand CLI. By default, it writes a fresh
run directory under `runs/`. Use `--output-dir` to choose a specific location.

```bash
webinar-transcriber INPUT
webinar-transcriber INPUT --keep-audio --audio-format mp3
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
uv tool install --reinstall ".[llm]"
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

- `metadata.json`, `transcript.json`, and `transcript.vtt`
- `diagnostics.json` with stage timings, counts, warnings, ASR details, and optional LLM metadata
- `report.md`, `report.docx`, and `report.json`
- ASR planning and decode artifacts under `asr/`

Depending on options and input type, successful default runs also write:

- `transcription-audio.wav` or `transcription-audio.mp3` with `--keep-audio`
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
1. Expand and repair speech regions into ASR windows.
1. Transcribe the windows locally with `whisper.cpp`.
1. Reconcile adjacent windows into one transcript.
1. Detect scenes and extract representative frames for video input.
1. Build sections, summaries, and action items.
1. Optionally polish the report with an LLM.
1. Write report, subtitle, diagnostic, and intermediate artifacts.

## Advanced Usage

### ASR Model

`webinar-transcriber` uses `pywhispercpp` to resolve whisper.cpp models. By default, it uses the
built-in `large-v3-turbo` model identifier, which `pywhispercpp` downloads into its cache on first
use. You can also pass a different model identifier or a local GGML model path with `--asr-model`.

For example:

```bash
webinar-transcriber INPUT --asr-model large-v3-turbo
webinar-transcriber INPUT --asr-model models/whisper-cpp/ggml-large-v3-turbo.bin
```

To manage the file yourself, download it directly:

```bash
mkdir -p models/whisper-cpp
curl -L \
    -o models/whisper-cpp/ggml-large-v3-turbo.bin \
    https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin
```

### VAD Tuning

Silero VAD behavior can be tuned from the CLI:

- `--vad/--no-vad`: enable or disable Silero speech-region detection before ASR planning.
- `--vad-threshold FLOAT`: set the Silero speech detection threshold.
- `--min-speech-ms INT`: require this much speech before a region is kept.
- `--min-silence-ms INT`: require this much silence before adjacent regions are split.
- `--speech-region-pad-ms INT`: add symmetric padding to detected speech regions during ASR
  planning.

### ASR Tuning

`whisper.cpp` decode behavior can be tuned from the CLI:

- `--carryover/--no-carryover`: enable or disable bounded prompt carryover between adjacent
  inference windows.
- `--carryover-max-sentences INT`: cap how many trailing sentences can be reused as carryover.
- `--carryover-max-tokens INT`: cap the approximate token budget for carryover text per inference
  window.
- `--threads INT`: set the number of `whisper.cpp` inference threads.

## Reference

### Output Layout

Successful default runs write:

```text
runs/<timestamp>_<basename>/
├─ asr/
│  ├─ speech_regions.json
│  ├─ expanded_regions.json
│  └─ decoded_windows.json
├─ metadata.json
├─ transcript.json
├─ transcript.vtt
├─ transcription-audio.wav # optional via --keep-audio
├─ transcription-audio.mp3 # optional via --keep-audio --audio-format mp3
├─ scenes.json             # video only
├─ diagnostics.json
├─ report.md
├─ report.docx
├─ report.json
└─ frames/                 # video only
```

## Development

### Local Setup

1. Install Python 3.12 and `uv`.
1. Install project and development dependencies:

```bash
make sync
```

### Running from a Checkout

If you are developing inside the repository and do not want to install the CLI as a tool, use:

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
