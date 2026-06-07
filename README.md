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

`webinar-transcriber` turns webinar recordings into transcripts, structured notes, diagnostics, and
machine-readable report artifacts. It accepts audio-only files and slide-based video. Video runs add
scene detection and representative frames; audio-only runs keep the same transcript and report
contract without visual context.

The useful constraint is locality. The default pipeline normalizes the media, detects speech regions
with the bundled Silero VAD ONNX model, transcribes windows with `whisper.cpp`, reconciles window
overlap, and builds report sections with local heuristics. Optional LLM refinement runs only after
that deterministic report exists. It can polish section text and refine titles, summaries, action
items, and section TL;DRs, but it does not replace the base pipeline.

> [!NOTE]
> The core pipeline runs locally. Cloud access is used only when you pass `--llm`.

## Install

### Prerequisites

- Python 3.12+
- `uv`
- For CUDA source builds: a C/C++ compiler, `cmake`, and a working CUDA toolkit with `nvcc` on
  `PATH` and `CUDA_HOME` set

### Install the CLI from this checkout

From the repository root, run the target that matches the runtime you need:

| Command             | Installs                                                |
| ------------------- | ------------------------------------------------------- |
| `make install`      | Standard local CLI install.                             |
| `make install-llm`  | CLI plus optional OpenAI/Anthropic LLM dependencies.    |
| `make install-cuda` | CLI with `pywhispercpp` rebuilt from source for NVIDIA. |
| `make uninstall`    | Removes the installed `webinar-transcriber` uv tool.    |

`make install` registers `webinar-transcriber` as a uv tool from the local source tree. No virtual
environment activation is needed. Re-run the install target after pulling changes when the installed
command should track the checkout.

Without `make`, use uv directly:

```bash
uv tool install --reinstall .
```

Run `make help` to list every project target.

The standard install uses prebuilt `pywhispercpp` wheels: CPU wheels on Linux and Windows, Metal on
Apple Silicon where the wheel supports it. It does not install PyTorch. Speech-region detection uses
the bundled Silero ONNX model through `sherpa-onnx`.

The default `large-v3-turbo` Whisper model is downloaded by `pywhispercpp` on the first
transcription run, not during installation. Each run records the active ASR backend in
`diagnostics.json` under `asr_pipeline.system_info`; native whisper.cpp logs are written to
`whisper-cpp.log` in the run directory.

### NVIDIA CUDA

> [!CAUTION]
> CUDA installs rebuild `pywhispercpp` locally and depend on the host CUDA toolkit. Use the standard
> install unless you specifically need NVIDIA acceleration.

CUDA is the only supported path that builds `pywhispercpp` from source:

- `make install-cuda` installs the CLI tool with CUDA support.
- `make sync-cuda` prepares the checkout development environment with CUDA support.

If the build fails, see [CUDA install fails](#cuda-install-fails).

All examples below assume `webinar-transcriber` is available on `PATH`.

## Usage

### Quick start

`webinar-transcriber` is a single root command. There are no subcommands. By default, each input
gets a fresh run directory under `runs/`. Multiple inputs are processed sequentially. `--output-dir`
is allowed only with one input.

Any container PyAV can decode is accepted, including common formats such as `.mp4`, `.mkv`, `.mov`,
`.webm`, `.mp3`, `.wav`, and `.m4a`. The first pipeline stage normalizes the audio track to mono
`16 kHz` `16-bit PCM WAV`, independent of the input container.

> [!TIP]
> Use a fresh `--output-dir` for reproducible comparisons. Existing output directories are refused,
> not overwritten.

```bash
webinar-transcriber INPUT
webinar-transcriber INPUT1 INPUT2 INPUT3
webinar-transcriber INPUT --keep-audio
webinar-transcriber INPUT --output-dir runs/custom-demo
```

### Cloud LLM

`--llm` enables provider-backed report refinement after deterministic sectioning. The LLM step can
polish section transcript text with light cleanup and paragraphing, and refine summary bullets,
action items, section titles, and section TL;DRs. Supported providers are `openai` and `anthropic`;
OpenAI is the default.

The base install does not include provider SDKs. Install the LLM extra first:

```bash
make install-llm
```

Configuration comes from environment variables. The CLI does not pin a model name; pass a model
identifier supported by the provider you are using.

```bash
OPENAI_API_KEY=... \
    OPENAI_MODEL=<openai-model> \
    webinar-transcriber INPUT --llm
```

```bash
LLM_PROVIDER=anthropic \
    ANTHROPIC_API_KEY=... \
    ANTHROPIC_MODEL=<anthropic-model> \
    webinar-transcriber INPUT --llm
```

For missing environment variables, missing extras, or unsupported provider names, see
[Troubleshooting](#troubleshooting).

### Speaker diarization

Pass `--diarize` to label transcript segments with anonymous local speaker IDs:

```bash
webinar-transcriber INPUT --diarize
webinar-transcriber INPUT --diarize --diarize-speakers 4
```

> [!WARNING]
> Pass `--diarize-speakers` only when the exact speaker count is known. A wrong count can force poor
> speaker labels. Omit the option to let Sherpa estimate the count.

Diarization runs locally through `sherpa-onnx` and does not use an API key. The first diarized run
downloads the segmentation and speaker-embedding models into
`~/.cache/webinar-transcriber/diarization`.

When diarization is enabled, reports prefix transcript paragraphs with stable anonymous labels
ordered by first appearance in the timeline: `S1`, `S2`, and so on. JSON artifacts include a
`speaker` field on transcript segments and a separate `diarization.json` file with raw speaker
turns. If labels look wrong, see [Poor diarization labels](#poor-diarization-labels).

### How it works

The CLI runs this local pipeline: media probe, normalized transcription audio, Silero VAD, ASR
window planning, `whisper.cpp` transcription, overlap reconciliation, optional speaker diarization,
transcript normalization, optional scene detection and frame extraction, local report assembly,
optional LLM polish, and final artifact and diagnostics write-out.

For per-stage detail and the artifact each stage produces, see [docs/pipeline.md](docs/pipeline.md).

## Advanced Usage

### ASR model

`webinar-transcriber` uses `pywhispercpp` to resolve whisper.cpp models. By default, it uses
`large-v3-turbo`, which `pywhispercpp` downloads into its cache on first use. You can pass another
model identifier or a local GGML model path with `--asr-model`.

```bash
webinar-transcriber INPUT --asr-model large-v3-turbo
webinar-transcriber INPUT --asr-model large-v3
webinar-transcriber INPUT --asr-model models/whisper-cpp/ggml-large-v3-turbo.bin
```

To manage the model file yourself, download it directly:

```bash
mkdir -p models/whisper-cpp
curl -L \
    -o models/whisper-cpp/ggml-large-v3-turbo.bin \
    https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin
```

Use `--asr-model large-v3` when transcription accuracy matters more than local runtime.

### ASR controls

The default ASR path uses the selected `whisper.cpp` model, automatic language detection, and an
automatically selected thread count.

- `--language CODE`: force a Whisper language code such as `en` or `ru`.
- `--threads N`: set the CPU worker count passed to `whisper.cpp`.

GPU-enabled builds can offload supported model work to the GPU, but `whisper.cpp` still uses CPU
threads for scheduling, language detection, and non-offloaded work. The same `--threads` value is
also used by local VAD, local diarization, and concurrent LLM section polishing. By default, the CLI
uses the host CPU count capped at 8. Lower it when the machine needs CPU capacity for other work.

## Troubleshooting

### `Missing required LLM environment variables`

`--llm` was passed without the required provider environment variables. Set `OPENAI_API_KEY` and
`OPENAI_MODEL` for OpenAI, or set `LLM_PROVIDER=anthropic` plus `ANTHROPIC_API_KEY` and
`ANTHROPIC_MODEL` for Anthropic.

### `requires the 'llm' extra`

The provider SDKs are not installed. Reinstall the CLI with the LLM extra:

```bash
make install-llm
```

Without `make`:

```bash
uv tool install --reinstall ".[llm]"
```

### `Unsupported LLM provider`

`LLM_PROVIDER` is set to a value other than `openai` or `anthropic`. Unset it to use OpenAI, or set
it to `anthropic`.

### `Output directory already exists`

The CLI refuses to overwrite existing run directories. Pass a new `--output-dir`, remove the
existing directory, or omit `--output-dir` so the CLI creates a fresh timestamped directory under
`runs/`.

### `Could not prepare whisper.cpp model`

`pywhispercpp` could not load the requested model. Check that `--asr-model` is a known identifier
such as `large-v3-turbo` or `large-v3`, or that a local path points to a valid GGML file. The native
`whisper-cpp.log` inside the run directory has the underlying error.

### Wrong language detected

Whisper can mis-detect language for short, multilingual, or noisy audio. Pass `--language CODE`, for
example `--language en` or `--language ru`, to force the language hint.

### Poor diarization labels

`--diarize-speakers COUNT` forces an exact speaker count. If the count is wrong, labels degrade.
Omit the flag to let `sherpa-onnx` estimate the count, or pass the correct count.

### CUDA install fails

`make install-cuda` rebuilds `pywhispercpp` from source and needs `nvcc` on `PATH` and `CUDA_HOME`
set. If you do not need NVIDIA acceleration, use `make install`; it pulls prebuilt wheels and skips
the C/C++/CUDA toolchain.

## Reference

### Output layout

Successful runs can write:

```text
runs/<timestamp>_<basename>/
├─ metadata.json             # probed media type, duration, streams
├─ transcript.json           # reconciled transcript with timestamps and optional speakers
├─ report.md
├─ report.docx
├─ report.json               # final report in markdown, docx, and json
├─ diagnostics.json          # stage timings, counts, warnings, ASR and optional LLM info
├─ asr/
│  ├─ speech_regions.json    # VAD ranges
│  └─ decoded_windows.json   # per-window decode output
├─ diarization.json          # anonymous speaker turns; --diarize only
├─ scenes.json               # scene boundaries; video only
├─ frames/                   # representative frames; video only
└─ transcription-audio.mp3   # normalized audio copy; --keep-audio only
```

Failed runs still write `diagnostics.json` with the failed stage and any partial intermediate
artifacts already produced, as long as the run directory exists.

## Development

### Local setup

Install Python 3.12+ and `uv`, then sync the checkout environment you need:

- `make sync`: standard development and test dependencies.
- `make sync-llm`: development dependencies plus optional LLM SDKs.
- `make sync-cuda`: development environment with CUDA-built `pywhispercpp`.

Without `make`:

```bash
uv sync
uv sync --extra llm
```

### Running from a checkout

If you are developing in the repository and do not want to install the CLI as a uv tool, run through
the checkout environment:

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

### Quality gates

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

Contributor and agent-specific guidance, including coding conventions, testing notes, and the
Definition of Done, lives in [AGENTS.md](AGENTS.md).
