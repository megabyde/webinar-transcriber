# webinar-transcriber

[![CI](https://github.com/megabyde/webinar-transcriber/actions/workflows/ci.yml/badge.svg)](https://github.com/megabyde/webinar-transcriber/actions/workflows/ci.yml)
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

On Apple Silicon, the CLI now prefers the `mlx-whisper` backend when it is installed and falls
back to `faster-whisper` everywhere else. The first ASR run will download the configured backend
model if it is not already available locally.

Typical runs:

```bash
webinar-transcriber process INPUT
webinar-transcriber process INPUT --format docx
webinar-transcriber process INPUT --output-dir runs/custom-demo
webinar-transcriber extract-frames INPUT
```

### ASR Backends

- `auto`: prefer MLX on Apple Silicon when `mlx-whisper` is installed, otherwise use
  `faster-whisper`
- `mlx`: Apple Silicon MLX backend
- `faster-whisper`: CTranslate2-based Whisper backend

The `--asr-model` option can override the backend default, for example `small` for
`faster-whisper` or an MLX repository name such as `mlx-community/whisper-large-v3-turbo`.

## Processing Behavior

### Pipeline

1. Probe input media with `ffprobe`.
2. Prepare transcription audio with `ffmpeg`.
3. Transcribe with the selected ASR backend.
4. For video input, detect scenes and extract representative frames.
5. Align transcript content to audio or slide sections.
6. Structure notes and export Markdown, DOCX, and JSON artifacts.

### Runtime Rules

- The tool auto-detects whether `INPUT` is audio or video.
- Every invocation writes a fresh run directory unless `--output-dir` is supplied.
- `process --format md` and `process --format docx` still write `report.json`.
- `diagnostics.json` is written for successful runs and records ASR backend/model, stage timings,
  item counts, and warnings.

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
2. Install native dependencies with Homebrew on macOS:
   - `brew install ffmpeg`
3. Install the project:

```bash
make sync
```

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

The repository keeps tiny committed media fixtures so pipeline tests can run without network
fetches or giant binary blobs.

## Package Shape

The package stays intentionally flat:

- top-level modules own the core flow
- `video/` contains scene detection and frame extraction
- `export/` contains report writers
- tests live alongside the package hierarchy under `webinar_transcriber/tests/` and
  `webinar_transcriber/video/tests/`

Contributor and agent-specific repository guidance lives in [AGENTS.md](AGENTS.md).
