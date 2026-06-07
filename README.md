# Webinar Transcriber

[![CI](https://github.com/megabyde/webinar-transcriber/actions/workflows/ci.yml/badge.svg)](https://github.com/megabyde/webinar-transcriber/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Coverage 100%](https://img.shields.io/badge/coverage-100%25-brightgreen)](https://github.com/megabyde/webinar-transcriber/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/badge/lint-ruff-D7FF64?logo=ruff&logoColor=1D2B34)](https://docs.astral.sh/ruff/)
[![uv](https://img.shields.io/badge/package-uv-5C5CFF?logo=uv&logoColor=white)](https://docs.astral.sh/uv/)
[![License MIT](https://img.shields.io/badge/license-MIT-2F855A)](LICENSE)

## Contents

- [Overview](#overview)
- [Install](#install)
- [Usage](#usage)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)
- [Reference](#reference)
- [Development](#development)

## Overview

`webinar-transcriber` is a local-first CLI for turning webinar recordings into transcripts,
structured notes, diagnostics, and machine-readable report artifacts. It handles both audio-only
input and slide-based video, detects scenes for video runs, and keeps the core pipeline local with
PyAV, Silero VAD through `sherpa-onnx`, and `whisper.cpp`.

The default flow is deterministic: normalize the media, detect speech regions, transcribe locally,
reconcile overlapping windows, and build report sections with local heuristics. Optional
provider-backed LLM refinement can polish section text and add summary bullets, action items,
section titles, and section TL;DRs on top of that local output, but it does not replace the base
pipeline.

> [!NOTE]
> The core pipeline runs locally. Cloud access is used only when you explicitly pass `--llm`.

## Install

### Prerequisites

- Python 3.12+ and `uv`.
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

On Windows, or anywhere without `make`, use uv directly:

```bash
uv tool install --reinstall .
```

```bash
webinar-transcriber --help
```

Use the matching Make target for the install you need:

| Command             | Installs                                                |
| ------------------- | ------------------------------------------------------- |
| `make install`      | Standard local CLI install.                             |
| `make install-llm`  | CLI plus optional OpenAI/Anthropic LLM dependencies.    |
| `make install-cuda` | CLI with `pywhispercpp` rebuilt from source for NVIDIA. |
| `make uninstall`    | Removes the installed `webinar-transcriber` uv tool.    |

The standard and LLM install targets pull the published `pywhispercpp` wheels from PyPI. On Linux
and Windows, those wheels use the CPU backend. On macOS, the Apple Silicon wheels include Metal
support. Speech-region detection uses the bundled Silero ONNX model through `sherpa-onnx`, so the
base install does not pull PyTorch or another large deep-learning framework. The default
`large-v3-turbo` model is downloaded on first transcription run, not during installation. To verify
the active backend after a run, inspect `diagnostics.json` → `asr_pipeline.system_info`. Native
whisper.cpp initialization and teardown logs are written to `whisper-cpp.log` in the run directory.

> [!TIP]
> The base install does not pull PyTorch. VAD uses the bundled Silero ONNX model through
> `sherpa-onnx`, and Whisper inference runs through `whisper.cpp`.

To inspect all available project commands:

```bash
make help
```

### NVIDIA (CUDA on Linux or Windows)

> [!CAUTION]
> CUDA installs rebuild `pywhispercpp` locally and depend on your system CUDA toolkit. Use the
> standard install unless you specifically need NVIDIA acceleration.

CUDA is the only supported path that builds `pywhispercpp` from source. Use the CUDA target that
matches your workflow:

- `make install-cuda` for the installed CLI tool
- `make sync-cuda` for the checkout development environment

All usage examples below assume `webinar-transcriber` is available on your `PATH`.

## Usage

### Quick Start

`webinar-transcriber` is a single root command, not a subcommand CLI. By default, it writes a fresh
run directory under `runs/`. Pass multiple inputs to process them sequentially. Use `--output-dir`
with a single input to choose a specific location.

> [!TIP]
> Use a fresh `--output-dir` for reproducible comparisons. Existing output directories are not
> overwritten.

```bash
webinar-transcriber INPUT
webinar-transcriber INPUT1 INPUT2 INPUT3
webinar-transcriber INPUT --keep-audio
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

> [!IMPORTANT]
> `--llm` requires the LLM extra. Install it first with `make install-llm`.

```bash
make install-llm
```

LLM configuration comes only from environment variables. Substitute any model identifier the
provider currently offers; the CLI does not pin a default. Check the provider's documentation for
current model names.

#### OpenAI

```bash
OPENAI_API_KEY=... \
    OPENAI_MODEL=<openai-model> \
    webinar-transcriber INPUT --llm
```

#### Anthropic

```bash
LLM_PROVIDER=anthropic \
    ANTHROPIC_API_KEY=... \
    ANTHROPIC_MODEL=<anthropic-model> \
    webinar-transcriber INPUT --llm
```

### Speaker Diarization

Pass `--diarize` to label transcript segments with anonymous local speaker IDs:

```bash
webinar-transcriber INPUT --diarize
webinar-transcriber INPUT --diarize --diarize-speakers 4
```

> [!WARNING]
> Pass `--diarize-speakers` only when the exact speaker count is known. A wrong count can force poor
> speaker labels; omit the option to let Sherpa estimate the count.

Diarization runs entirely locally through `sherpa-onnx`; it does not use an API key. The first
diarized run downloads the segmentation and speaker-embedding models into
`~/.cache/webinar-transcriber/diarization`.

> [!NOTE]
> Speaker labels are anonymous and stable within a run: `S1`, `S2`, and so on, ordered by first
> appearance in the timeline.

When enabled, reports prefix transcript paragraphs with stable labels such as `S1` and `S2`, ordered
by first appearance. The JSON artifacts include a `speaker` field on transcript segments and a
separate `diarization.json` file with raw speaker turns.

### What You Get

Successful default runs write:

- `metadata.json` and `transcript.json`
- `diagnostics.json` with stage timings, counts, warnings, ASR details, and optional LLM metadata
- `report.md`, `report.docx`, and `report.json`
- ASR planning and decode artifacts under `asr/`

Depending on options and input type, successful default runs also write:

| Artifact                                                   | When written   |
| ---------------------------------------------------------- | -------------- |
| `transcription-audio.mp3`                                  | `--keep-audio` |
| `scenes.json` and `frames/`                                | Video input    |
| `diarization.json` and speaker-labeled transcript segments | `--diarize`    |

Failed runs also write `diagnostics.json` once the run directory exists, though they may still leave
only partial intermediate artifacts and no final report outputs.

### How It Works

1. Create an isolated run directory.
   - Each input gets its own `runs/<timestamp>_<basename>/` directory unless you pass a single
     `--output-dir`.
   - Existing output directories are refused instead of overwritten, so reruns do not silently mix
     old and new artifacts.
1. Probe the media.
   - The probe records whether the input is audio or video, its duration, and the streams available
     to the rest of the pipeline.
   - This data is written early to `metadata.json`, and later stages use the probed duration for
     progress, diagnostics, and report timing.
1. Prepare deterministic transcription audio.
   - PyAV decodes the input into temporary mono, `16 kHz`, `16-bit PCM WAV` audio.
   - Downstream speech detection, ASR, and diarization all consume this normalized audio contract.
   - The temporary WAV normally stays outside the run directory; `--keep-audio` saves a compressed
     `transcription-audio.mp3` copy after transcription.
1. Load the local ASR runtime.
   - `pywhispercpp` loads the selected `whisper.cpp` model identifier or local GGML model path.
   - The backend, model name, thread count, and native whisper.cpp logs are recorded in the run
     directory so performance and failures can be inspected after the run.
1. Detect speech regions.
   - The bundled Silero VAD ONNX model runs locally through `sherpa-onnx`.
   - Speech ranges are padded, overlapping padded ranges are merged, and the result is written to
     `asr/speech_regions.json`.
   - If VAD cannot run on the host, the pipeline records a warning and falls back to treating the
     whole normalized audio file as speech.
1. Plan ASR windows.
   - Speech regions are converted into bounded Whisper decode windows.
   - Long regions are split into windows of about 28 seconds with a small overlap at boundaries, so
     Whisper gets context without decoding the entire file at once.
1. Transcribe windows locally.
   - Each window is decoded by `whisper.cpp` and mapped back onto the original media timeline.
   - The raw per-window result is written to `asr/decoded_windows.json`; this is useful when
     debugging boundary behavior before reconciliation.
1. Reconcile raw windows into a transcript.
   - Overlapping window boundaries are deduplicated by comparing text prefixes and suffixes.
   - Empty or invalid decoded segments are discarded.
   - The reconciled transcript is the source for `transcript.json` and for optional speaker
     assignment.
1. Optionally diarize speakers.
   - `--diarize` runs local `sherpa-onnx` speaker segmentation and embedding models against the same
     normalized audio.
   - Speaker turns are normalized to anonymous labels ordered by first appearance (`S1`, `S2`, and
     so on) and written to `diarization.json`.
   - Transcript segments receive the speaker label with the largest time overlap before transcript
     normalization, so speaker changes can prevent unrelated adjacent segments from being merged.
1. Normalize the transcript for report generation.
   - Adjacent transcript segments are merged only when the time gap is small enough and the speaker
     label allows it.
   - The normalized transcript keeps timestamps and optional speaker labels, but produces cleaner
     paragraph-sized material for sectioning and export.
1. Process video context when the input has video.
   - Scene detection samples the video timeline and writes `scenes.json`.
   - Representative frames are extracted into `frames/`.
   - Transcript segments are aligned to scene ranges so the Markdown, DOCX, and JSON reports can
     associate sections with the right visual frame.
1. Build report sections locally.
   - Video reports are primarily organized by detected scene boundaries.
   - Audio-only reports use transcript timing and gaps to produce a best-effort section structure.
   - This local report is complete even when `--llm` is not used.
1. Optionally polish the report with an LLM.
   - `--llm` runs after the deterministic local report exists.
   - The LLM can polish section transcript text and refine summary bullets, action items, section
     titles, and section TL;DRs.
   - If an LLM step fails, diagnostics and warnings record the fallback and the local report content
     remains available.
1. Write final artifacts and diagnostics.
   - `report.md`, `report.docx`, `report.json`, `transcript.json`, and `diagnostics.json` are
     written at the end of a successful run.
   - Failed runs still try to write `diagnostics.json` once the run directory exists, including the
     failed stage, warnings, timings, and any partial artifacts already produced.

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

### ASR Controls

The default ASR path uses the selected `whisper.cpp` model, automatic language detection, and an
automatically selected thread count.

- `--language CODE`: force a Whisper language code such as `en` or `ru`.
- `--threads N`: set the CPU worker count passed to `whisper.cpp`. GPU-enabled builds can offload
  supported model work to the GPU, but `whisper.cpp` still uses CPU threads for non-offloaded work,
  scheduling, and language detection. The same value is also used by local VAD, local diarization,
  and concurrent LLM section polishing. By default, the CLI uses the host CPU count capped at 8;
  lower this when you need to leave CPU capacity for other work.

## Troubleshooting

### `Missing required LLM environment variables`

`--llm` was passed without the required provider environment variables. Set `OPENAI_API_KEY` and
`OPENAI_MODEL` for the default OpenAI provider, or `LLM_PROVIDER=anthropic` plus `ANTHROPIC_API_KEY`
and `ANTHROPIC_MODEL` for Anthropic. See [Cloud LLM](#cloud-llm) for full examples.

### `requires the 'llm' extra`

The provider SDKs are not installed. Reinstall the CLI with the `llm` extra: `make install-llm` (or
`uv tool install --reinstall ".[llm]"` on Windows). Re-run after the install completes.

### `Unsupported LLM provider`

`LLM_PROVIDER` is set to a value other than `openai` or `anthropic`. Unset it to use the default
OpenAI provider or set it to `anthropic`.

### `Output directory already exists`

The CLI refuses to overwrite existing run directories. Either pass a new `--output-dir`, delete the
existing one, or omit `--output-dir` so the CLI creates a fresh timestamped directory under `runs/`.

### `Could not prepare whisper.cpp model`

`pywhispercpp` could not load the requested model. Check that `--asr-model` matches a known
identifier such as `large-v3-turbo` or `large-v3`, or that any local path points to a valid GGML
file. The native `whisper-cpp.log` inside the run directory has the underlying error.

### Wrong language detected

Whisper sometimes mis-detects the language for short, multilingual, or noisy audio. Pass
`--language CODE` (for example `--language en` or `--language ru`) to force the language hint.

### Poor diarization labels

`--diarize-speakers COUNT` forces an exact speaker count. If the count is wrong the labels degrade.
Omit the flag to let `sherpa-onnx` estimate the count, or pass the correct number.

### CUDA install fails

`make install-cuda` rebuilds `pywhispercpp` from source and needs `nvcc` on `PATH` and `CUDA_HOME`
set. If you do not need NVIDIA acceleration, use `make install` instead — it pulls prebuilt wheels
and skips the C/C++/CUDA toolchain entirely.

## Reference

### Output Layout

Successful runs can write:

```text
runs/<timestamp>_<basename>/
├─ asr/
│  ├─ decoded_windows.json
│  └─ speech_regions.json
├─ diarization.json         # optional via --diarize
├─ frames/                  # video only
├─ diagnostics.json
├─ metadata.json
├─ report.docx
├─ report.json
├─ report.md
├─ scenes.json              # video only
├─ transcript.json
└─ transcription-audio.mp3  # optional via --keep-audio
```

## Development

### Local Setup

1. Install Python 3.12+ and `uv`.

1. Sync the checkout environment you need:

   - `make sync`: standard development and test dependencies.
   - `make sync-llm`: development dependencies plus optional LLM SDKs.
   - `make sync-cuda`: development environment with CUDA-built `pywhispercpp`.

On Windows, use uv directly:

```powershell
uv sync
uv sync --extra llm
```

### Running from a Checkout

If you are developing inside the repository and do not want to install the CLI as a tool, run it
through the checkout environment:

```bash
uv run webinar-transcriber --help
uv run webinar-transcriber INPUT
```

### Toolchain

- Python 3.12+
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

On Windows, run the same checks through uv:

```powershell
uv run mdformat --check AGENTS.md LICENSE-3RDPARTY.md README.md
uv run pymarkdown scan AGENTS.md LICENSE-3RDPARTY.md README.md
uv run ruff format --check .
uv run ruff check .
uv run ty check webinar_transcriber
uv run pytest
```

For quick local iteration, `make test` skips tests marked `slow`. Use `make check` to run the full
coverage-gated suite.

The repository keeps tiny committed media fixtures so pipeline tests can run without network fetches
or large binary blobs. Contributor and agent-specific repository guidance lives in
[AGENTS.md](AGENTS.md).
