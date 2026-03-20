# webinar-transcriber

![webinar-transcriber banner](docs/assets/banner.svg)

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

## CLI

```bash
webinar-transcriber process INPUT
webinar-transcriber process INPUT --format docx
webinar-transcriber process INPUT --output-dir runs/custom-demo
```

On Apple Silicon, the CLI now prefers the `mlx-whisper` backend when it is installed and falls
back to `faster-whisper` everywhere else. The first ASR run will download the configured backend
model if it is not already available locally.

## Output Layout

Each run writes a fresh directory under `runs/` unless `--output-dir` is supplied:

```text
runs/<timestamp>_<basename>/
├─ audio.wav
├─ metadata.json
├─ transcript.json
├─ scenes.json          # video only
├─ diagnostics.json
├─ report.md
├─ report.docx
├─ report.json
└─ frames/              # video only
```

## Local Setup

1. Install Python 3.12 and `uv`.
2. Install native dependencies with Homebrew on macOS:
   - `brew install ffmpeg`
3. Install the project:

```bash
make sync
```

## Quality Gates

Every logical implementation step is expected to pass the full verification cycle before it is
committed:

```bash
make format
make lint
make typecheck
make test
make coverage
```

## Documentation

- [Architecture](docs/architecture.md)
- [Usage](docs/usage.md)
- [Languages](docs/languages.md)
- [Development](docs/development.md)
