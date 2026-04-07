# Webinar Transcriber

[![CI][ci-badge]][ci-workflow] [![Python 3.12][python-badge]][python-home]
[![Coverage 95%+][coverage-badge]][coverage-home] [![Ruff][ruff-badge]][ruff-home]
[![uv][uv-badge]][uv-home] [![License MIT][license-badge]][license-home]

`webinar-transcriber` is a local-first CLI for turning webinar recordings into transcripts,
structured notes, and machine-readable artifacts. It handles slide-based videos and audio-only
recordings, writes Markdown, DOCX, JSON, and VTT outputs, and supports automatic language detection.

## Capabilities

The tool is built for:

- audio-only recordings
- video recordings with scene detection and representative slide frames
- Markdown, DOCX, JSON, and subtitle outputs
- deterministic local transcription with optional cloud LLM report polish

The default pipeline is intentionally conservative: local-first, CLI-only, and heuristic-driven for
sectioning, summaries, and action items.

## Usage

The CLI uses `whisper.cpp` in-process through its C API. The Python layer uses one code path on
macOS and Linux; acceleration depends on how the local `libwhisper` was built and installed. By
default, the ASR model is resolved from the Hugging Face cache and downloaded there on first use.

Typical runs:

```bash
webinar-transcriber process INPUT
webinar-transcriber process INPUT --llm
webinar-transcriber process INPUT --format docx
webinar-transcriber process INPUT --keep-audio --audio-format mp3
webinar-transcriber process INPUT --output-dir runs/custom-demo
webinar-transcriber extract-frames INPUT
```

### ASR Model

Use `--asr-model` to override the `whisper.cpp` model path, for example:

```bash
webinar-transcriber process INPUT --asr-model models/whisper-cpp/ggml-large-v3-turbo.bin
```

By default, the app downloads the official `whisper.cpp` model into the standard Hugging Face cache
and reuses that cached file on later runs.

If you prefer a manually managed file instead of the cache-backed default, download it yourself. A
direct download example:

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

If you store the model somewhere else, point the app at it explicitly with `--asr-model`.

### Extra Artifacts

`process` always writes subtitle files alongside the report artifacts:

- `transcript.vtt`

To keep the normalized transcription audio, add:

```bash
webinar-transcriber process INPUT --keep-audio
webinar-transcriber process INPUT --keep-audio --audio-format mp3
```

### ASR Tuning

The VAD-aware `whisper.cpp` pipeline can be tuned from the CLI:

```bash
webinar-transcriber process INPUT \
    --vad \
    --vad-threshold 0.5 \
    --min-speech-ms 250 \
    --min-silence-ms 600 \
    --speech-region-pad-ms 200 \
    --carryover \
    --carryover-max-sentences 2 \
    --carryover-max-tokens 64 \
    --threads 4
```

Silero VAD is included in the project dependencies and is used automatically when installed in the
current environment. If you updated from an older checkout, rerun:

```bash
make sync
```

The default ASR design is intentionally simple and deterministic:

- Silero VAD produces coarse speech regions.
- Very short neighboring regions are merged before ASR planning.
- Each repaired region is expanded with explicit symmetric ASR padding.
- Each expanded region becomes one deterministic inference window.
- `whisper.cpp` decodes one window at a time.
- Prompt carryover only reuses a small trusted suffix from the previous window.
- Adjacent decoded windows are reconciled back into one final transcript in Python.

This split keeps `whispercpp.py` focused on the C API, `segmentation.py` focused on speech-region
planning, and `asr.py` focused on prompt carryover and transcript assembly.

## Processing Behavior

### Pipeline

The pipeline runs in this order:

1. Probe input media with `ffprobe`.
1. Prepare deterministic transcription audio with `ffmpeg`.
1. Detect coarse speech regions with Silero VAD.
1. Merge overly short speech regions, then expand speech regions with ASR-specific padding.
1. Plan one inference window per expanded speech region.
1. Decode each window with `whisper.cpp`, using bounded prompt carryover when confidence is good.
1. Reconcile adjacent decoded windows into the final transcript.
1. For video input, detect scenes and extract representative frames.
1. Align transcript content to audio or slide sections.
1. Structure notes and export Markdown, DOCX, and JSON artifacts.

### Runtime Rules

- The tool auto-detects whether `INPUT` is audio or video.
- Every invocation writes a fresh run directory unless `--output-dir` is supplied.
- `process --format md` and `process --format docx` still write `report.json`.
- The ASR planner, carryover builder, and reconciliation path are deterministic and local-first.
- `diagnostics.json` is written for successful runs and records ASR backend/model, stage timings,
  item counts, warnings, and optional LLM metadata.

### Cloud LLM

The optional `--llm` flag enables provider-backed report refinement after deterministic sectioning:

- section transcript polishing for punctuation, spelling, light cleanup, and paragraph breaks
- summary, action items, and section title refinement

Supported providers:

- `openai` (default)
- `anthropic`

LLM configuration comes only from environment variables. There is no repository-specific secrets
integration.

OpenAI:

- `OPENAI_API_KEY`
- `OPENAI_MODEL`

Plain shell usage:

```bash
OPENAI_API_KEY=... \
    OPENAI_MODEL=gpt-5-mini \
    uv run webinar-transcriber process INPUT --llm
```

Anthropic:

```bash
LLM_PROVIDER=anthropic \
    ANTHROPIC_API_KEY=... \
    ANTHROPIC_MODEL=claude-sonnet-4-20250514 \
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
├─ asr/
│  ├─ speech_regions.json
│  ├─ expanded_regions.json
│  └─ decoded_windows.json
├─ metadata.json
├─ transcript.json
├─ transcript.vtt
├─ transcription-audio.wav # optional via --keep-audio
├─ transcription-audio.mp3 # optional via --keep-audio --audio-format mp3
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
1. Install the project and development dependencies:

```bash
make sync
```

### macOS

Install native dependencies with Homebrew:

```bash
brew install ffmpeg whisper-cpp
```

Homebrew installs `libwhisper.dylib` in a standard location, so the default setup should usually
work without extra configuration.

### Linux

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

You also need a local `whisper.cpp` model file, or the default cache-backed model described in
[ASR Model](#asr-model).

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
make test
```

The repository keeps tiny committed media fixtures so pipeline tests can run without network fetches
or large binary blobs.

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
[coverage-badge]: https://img.shields.io/badge/coverage-95%25%2B-brightgreen
[coverage-home]: https://github.com/megabyde/webinar-transcriber/actions/workflows/ci.yml
[license-badge]: https://img.shields.io/badge/license-MIT-2F855A
[license-home]: LICENSE
[python-badge]: https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white
[python-home]: https://www.python.org/downloads/release/python-3120/
[ruff-badge]: https://img.shields.io/badge/lint-ruff-D7FF64?logo=ruff&logoColor=1D2B34
[ruff-home]: https://docs.astral.sh/ruff/
[uv-badge]: https://img.shields.io/badge/package-uv-5C5CFF?logo=uv&logoColor=white
[uv-home]: https://docs.astral.sh/uv/
