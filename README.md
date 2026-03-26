# webinar-transcriber

[![CI][ci-badge]][ci-workflow]

![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)
![Coverage 90%25](https://img.shields.io/badge/coverage-90%25%2B-brightgreen.svg)
![License MIT](https://img.shields.io/badge/license-MIT-green.svg)

`webinar-transcriber` is a local-first CLI for transcribing webinar videos with slides and
audio-only recordings. The tool exports Markdown, DOCX, and JSON and supports automatic language
detection.

## Status

The project now includes a working local pipeline for:

- audio-only inputs
- video inputs with scene detection and representative slide frames
- Markdown, DOCX, and JSON outputs

The implementation is still intentionally conservative: it is local-first, CLI-only, and
heuristic-driven for structuring and summaries.

## Usage

The CLI uses `whisper.cpp` in-process through its C API. The Python layer uses one code path on
macOS and Linux; acceleration depends on how the local `libwhisper` was built and installed. The ASR
model is a local `.bin` file rather than an automatically downloaded Hugging Face model.

Typical runs:

```bash
webinar-transcriber process INPUT
webinar-transcriber process INPUT --llm
webinar-transcriber process INPUT --format docx
webinar-transcriber process INPUT --output-dir runs/custom-demo
webinar-transcriber extract-frames INPUT
```

### ASR Model

The `--asr-model` option can override the `whisper.cpp` model path, for example
`models/whisper-cpp/ggml-large-v3-turbo.bin`.

The default expected model path is:

```text
models/whisper-cpp/ggml-large-v3-turbo.bin
```

You need to download that file yourself before the first run. A direct download example:

```bash
mkdir -p models/whisper-cpp
curl -L \
  -o models/whisper-cpp/ggml-large-v3-turbo.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin
```

You can also use the official `whisper.cpp` helper script:

```bash
git clone https://github.com/ggml-org/whisper.cpp.git
sh whisper.cpp/models/download-ggml-model.sh large-v3-turbo models/whisper-cpp
```

If you store the model somewhere else, point the app at it:

```bash
webinar-transcriber process INPUT --asr-model /path/to/ggml-large-v3-turbo.bin
```

### ASR Tuning

The chunked ASR pipeline can be tuned from the CLI:

```bash
webinar-transcriber process INPUT \
  --vad \
  --chunk-target-sec 20 \
  --chunk-max-sec 30 \
  --chunk-overlap-sec 1.5 \
  --threads 4
```

Silero VAD is included in the project dependencies and is used automatically when installed in the
current environment. If you updated from an older checkout, rerun:

```bash
make sync
```

## Processing Behavior

### Pipeline

1. Probe input media with `ffprobe`.
1. Prepare deterministic transcription audio with `ffmpeg`.
1. Detect speech regions and plan ASR chunks.
1. Transcribe chunks with `whisper.cpp`.
1. Reconcile chunk overlap into one transcript.
1. For video input, detect scenes and extract representative frames.
1. Align transcript content to audio or slide sections.
1. Structure notes and export Markdown, DOCX, and JSON artifacts.

### Runtime Rules

- The tool auto-detects whether `INPUT` is audio or video.
- Every invocation writes a fresh run directory unless `--output-dir` is supplied.
- `process --format md` and `process --format docx` still write `report.json`.
- `diagnostics.json` is written for successful runs and records ASR backend/model, stage timings,
  item counts, warnings, and optional LLM metadata.

### Cloud LLM

The optional `--llm` flag enables OpenAI-backed report refinement after deterministic sectioning:

- section transcript polishing for punctuation, spelling, light cleanup, and paragraph breaks
- summary, action items, and section title refinement

LLM configuration comes only from environment variables:

- `OPENAI_API_KEY`
- `OPENAI_MODEL`

Plain shell usage:

```bash
OPENAI_API_KEY=... \
OPENAI_MODEL=gpt-5-mini \
uv run webinar-transcriber process INPUT --llm
```

With 1Password for local interactive runs:

```bash
OPENAI_API_KEY="$(op read 'op://Private/OpenAI/api key')" \
OPENAI_MODEL='gpt-5-mini' \
uv run webinar-transcriber process INPUT --llm
```

No special 1Password integration is required. The app only reads environment variables.

### Language Support

- Automatic language detection is enabled by default.
- English and Russian are the primary focus languages for fixtures and manual testing.
- The internal pipeline is not hard-coded to only those languages.

## Artifact Contracts

`process` writes:

```text
runs/<timestamp>_<basename>/
├─ metadata.json
├─ transcript.json
├─ scenes.json          # video only
├─ diagnostics.json
├─ report.md
├─ report.docx
├─ report.json
└─ frames/              # video only
```

`extract-frames` writes:

```text
runs/<timestamp>_<basename>/
├─ scenes.json
└─ frames/
```

## Local Setup

1. Install Python 3.12 and `uv`.
1. Install the project:

```bash
make sync
```

### macOS

Install native dependencies with Homebrew:

- `brew install ffmpeg`
- `brew install whisper-cpp`

Homebrew installs `libwhisper.dylib` in a standard location, so the default setup should work
without extra configuration.

### Linux

Install `ffmpeg`, then build or install `libwhisper.so` locally. If it is not in a standard system
location, point the app at it with `WHISPER_CPP_LIB`, for example:

```bash
export WHISPER_CPP_LIB=/path/to/libwhisper.so
```

You also need a local `whisper.cpp` model file as described in [ASR Model](#asr-model).

## Development

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
make typecheck
make test
make coverage
```

The repository keeps tiny committed media fixtures so pipeline tests can run without network fetches
or giant binary blobs.

## Package Shape

The package stays intentionally flat:

- top-level modules own the core flow
- `video/` contains scene detection and frame extraction
- `export/` contains report writers
- tests live alongside the package hierarchy under `webinar_transcriber/tests/` and
  `webinar_transcriber/video/tests/`

Contributor and agent-specific repository guidance lives in [AGENTS.md](AGENTS.md).

[ci-badge]: https://github.com/megabyde/webinar-transcriber/actions/workflows/ci.yml/badge.svg
[ci-workflow]: https://github.com/megabyde/webinar-transcriber/actions/workflows/ci.yml
