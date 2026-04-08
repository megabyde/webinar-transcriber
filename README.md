# Webinar Transcriber

[![CI](https://github.com/megabyde/webinar-transcriber/actions/workflows/ci.yml/badge.svg)](https://github.com/megabyde/webinar-transcriber/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3120/)
[![Coverage 95%+](https://img.shields.io/badge/coverage-95%25%2B-brightgreen)](https://github.com/megabyde/webinar-transcriber/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/badge/lint-ruff-D7FF64?logo=ruff&logoColor=1D2B34)](https://docs.astral.sh/ruff/)
[![uv](https://img.shields.io/badge/package-uv-5C5CFF?logo=uv&logoColor=white)](https://docs.astral.sh/uv/)
[![License MIT](https://img.shields.io/badge/license-MIT-2F855A)](LICENSE)

## Overview

`webinar-transcriber` is a local-first CLI for turning webinar recordings into transcripts,
structured notes, and machine-readable artifacts. It supports audio-only recordings and slide-based
videos, automatically detects language, and writes Markdown, DOCX, JSON, and VTT outputs.

It keeps the core pipeline local with `ffmpeg`, Silero VAD, and `whisper.cpp`, then builds sections,
summaries, and action items with deterministic heuristics. If you want lighter cleanup or report
polish, you can optionally add a provider-backed LLM pass on top of the local output instead of
replacing the default flow.

In practice, that means:

- audio or slide-based video in
- transcripts, reports, diagnostics, and subtitles out
- scene detection and representative frames for video input
- local-first transcription with optional OpenAI or Anthropic report refinement

## Install

### System Dependencies

Install the native media and `whisper.cpp` runtime dependencies before running the CLI locally.

#### macOS

```bash
brew install ffmpeg whisper-cpp
```

Homebrew installs `libwhisper.dylib` in a standard location, so the default setup should usually
work without extra configuration.

#### Linux

On Debian or Ubuntu, a typical setup is:

```bash
sudo apt update
sudo apt install -y ffmpeg libwhisper-cpp-dev
```

If `libwhisper.so` is not in a standard system location, point the app at it with `WHISPER_CPP_LIB`,
for example:

```bash
export WHISPER_CPP_LIB=/path/to/libwhisper.so
```

### Repo-Local Setup

Install the project and development dependencies:

```bash
make sync
```

All examples below assume you are running from this repository with
`uv run webinar-transcriber ...`.

## Usage

### Quick Start

By default, `process` writes a fresh run directory under `runs/`. Use `--output-dir` to pick a
specific location.

```bash
uv run webinar-transcriber process INPUT
uv run webinar-transcriber process INPUT --llm
uv run webinar-transcriber process INPUT --format docx
uv run webinar-transcriber process INPUT --keep-audio --audio-format mp3
uv run webinar-transcriber process INPUT --output-dir runs/custom-demo
uv run webinar-transcriber extract-frames INPUT
```

### Cloud LLM

The optional `--llm` flag enables provider-backed report refinement after deterministic sectioning:

- section transcript polishing for punctuation, spelling, light cleanup, and paragraph breaks
- summary, action items, and section title refinement

Supported providers:

- `openai` (default)
- `anthropic`

LLM configuration comes only from environment variables. There is no repository-specific secrets
integration.

#### OpenAI

```bash
OPENAI_API_KEY=... \
    OPENAI_MODEL=gpt-5-mini \
    uv run webinar-transcriber process INPUT --llm
```

#### Anthropic

```bash
LLM_PROVIDER=anthropic \
    ANTHROPIC_API_KEY=... \
    ANTHROPIC_MODEL=claude-sonnet-4-20250514 \
    uv run webinar-transcriber process INPUT --llm
```

### What You Get

`process` always writes:

- `metadata.json`, `transcript.json`, and `transcript.vtt`
- `diagnostics.json` with ASR backend/model details, stage timings, counts, warnings, and optional
  LLM metadata
- `report.json` regardless of `--format`
- ASR planning/debug artifacts under `asr/`

Depending on options and input type, `process` also writes:

- `report.md` and/or `report.docx` according to `--format`
- `transcription-audio.wav` or `transcription-audio.mp3` with `--keep-audio`
- `scenes.json` and `frames/` for video input

`extract-frames` writes `scenes.json` and `frames/` only.

### How It Works

- probe the input and normalize it into transcription audio
- detect speech regions and plan ASR windows
- transcribe locally with `whisper.cpp`
- reconcile adjacent windows into one transcript
- detect scenes and extract representative frames for video input
- build sections, summaries, and action items
- optionally refine the report with an LLM

## Advanced Usage

### ASR Model

`webinar-transcriber` uses a `whisper.cpp` GGML model file. By default, it resolves
`ggml-large-v3-turbo` from the standard Hugging Face cache and reuses the cached file on later runs.
If you want to pin a different model file or keep models in a repo-local directory, download the
`.bin` yourself and pass its path with `--asr-model`.

For example:

```bash
webinar-transcriber process INPUT --asr-model models/whisper-cpp/ggml-large-v3-turbo.bin
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
- `--speech-region-pad-ms INT`: add symmetric context around each detected speech region before
  decoding.

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

`process` writes:

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
├─ report.md              # written for --format all or --format md
├─ report.docx            # written for --format all or --format docx
├─ report.json
└─ frames/                 # video only
```

`extract-frames` writes:

```text
runs/<timestamp>_<basename>/
├─ scenes.json
└─ frames/
```

## Development

### Local Setup

1. Install Python 3.12 and `uv`.
1. Install the project and development dependencies:

```bash
make sync
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
make test
```

The repository keeps tiny committed media fixtures so pipeline tests can run without network fetches
or large binary blobs.

Contributor and agent-specific repository guidance lives in [AGENTS.md](AGENTS.md).
