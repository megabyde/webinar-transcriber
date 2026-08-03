# Troubleshooting

Common errors and how to resolve them. Each heading matches the message the CLI prints. Once a run
directory exists, failures write `diagnostics.json` with the failed stage when the error occurred
inside a named stage, and transcription failures leave the native `whisper-cpp.log` in the run
directory.

Find the heading that matches the error, apply its fix, then run the corrected
`webinar-transcriber INPUT ...` command. Recovery is complete when the command reaches report export
and writes `report.md`, `report.docx`, and `report.json`. If it fails again after creating a run
directory, inspect `diagnostics.json` first and `whisper-cpp.log` for transcription failures.

## `Could not open … with PyAV` / `No audio stream found`

The input could not be decoded, or it has no audio stream to transcribe. Confirm that the file plays
in a media player and uses a container PyAV can decode, such as `.mp4`, `.mkv`, `.mov`, `.webm`,
`.mp3`, `.wav`, or `.m4a`. Re-mux or re-encode a corrupt, video-only, or unsupported container with
`ffmpeg` before transcribing.

Verify the replacement file by running `webinar-transcriber INPUT`. The fix worked when media
probing succeeds and the run advances to transcription.

## `Missing required LLM environment variables`

`--llm` was passed without the required provider environment variables. Set `OPENAI_API_KEY` and
`OPENAI_MODEL` for OpenAI, or set `LLM_PROVIDER=anthropic` plus `ANTHROPIC_API_KEY` and
`ANTHROPIC_MODEL` for Anthropic.

Re-run `webinar-transcriber INPUT --llm` with all required variables set in the same shell. The fix
worked when processing starts instead of stopping before the first input.

## `requires the 'llm' extra`

The provider SDKs are not installed. Reinstall the CLI with the `llm` extra:

```bash
uv tool install --reinstall "webinar-transcriber[llm]"
```

From a checkout, use `uv tool install --reinstall ".[llm]"` instead.

With the provider environment variables set, run `webinar-transcriber INPUT --llm`. The fix worked
when the command starts without the missing-extra error.

## `Unsupported LLM provider`

`LLM_PROVIDER` is set to a value other than `openai` or `anthropic`. Unset it to use OpenAI, or set
it to `anthropic`.

Re-run `webinar-transcriber INPUT --llm` after correcting `LLM_PROVIDER`. The fix worked when
provider validation passes and processing starts.

## `Older LLM-polished runs cannot be rerun`

The source run used an LLM but predates `report.local.json`, so its `report.json` already contains
LLM output. Rerunning from that artifact would polish generated text a second time. Keep the
existing reports, or transcribe the original media again with `--llm` to create a reusable local
snapshot.

The replacement run is reusable when `report.local.json` exists. Pass that run directory to
`webinar-transcriber --rerun-llm RUN_DIR`; the new report variant appears under `RUN_DIR/llm/`.

## `The selected run directory is an LLM rerun variant`

The selected directory is already under a source run's `llm/` directory. Its report contains LLM
output, so using it as another source would polish generated text twice. Pass the original run
directory shown by `llm_rerun.source_run` in the variant's `diagnostics.json`.

## `LLM rerun requires a successfully completed source run`

The selected directory has a failed or incomplete `diagnostics.json`. LLM-only reruns do not resume
partial media processing. Fix the original failure and complete a new transcription run, then pass
that successful run directory to `--rerun-llm`.

The source is ready when its `diagnostics.json` has `"status": "succeeded"` and its final report
artifacts exist.

## `Output directory already exists`

The CLI refuses to overwrite existing run directories.

> [!CAUTION]
> Do not remove the existing directory until any artifacts you need are copied elsewhere.

The safe default is a new path:

```bash
webinar-transcriber INPUT --output-dir runs/new-run
```

Alternatively, omit `--output-dir` so the CLI creates a fresh timestamped directory under `runs/`.
The fix worked when the command creates the new directory and begins media processing.

## `Could not prepare whisper.cpp model`

`pywhispercpp` could not load the requested model. Check that `--asr-model` is a known identifier
such as `large-v3-turbo` or `large-v3`, or that a local path points to a valid GGML file. The native
`whisper-cpp.log` inside the run directory has the underlying error.

Re-run `webinar-transcriber INPUT --asr-model large-v3-turbo`, or pass the corrected local GGML
path. The fix worked when model preparation completes and transcription starts.

## Wrong language detected

Whisper can detect the wrong language for short, multilingual, or noisy audio. Pass
`--language CODE`, for example `--language en` or `--language ru`, to force the language hint.

Run `webinar-transcriber INPUT --language CODE` with the intended language code. Check
`transcript.json` in the new run directory; the fix worked when the transcript uses the intended
language.

## Poor diarization labels

`--diarize-speakers COUNT` forces an exact speaker count. If the count is wrong, labels degrade.
Omit the flag to let `sherpa-onnx` estimate the count, or pass the correct count.

Run `webinar-transcriber INPUT --diarize` without a count first. Check `diarization.json` and the
speaker fields in `transcript.json`; keep an explicit count only when it improves those labels.

## CUDA install fails

The CUDA install rebuilds `pywhispercpp` from source and needs `nvcc` on `PATH` and `CUDA_HOME` set.
If you do not need NVIDIA acceleration, use the standard `uv tool install --reinstall .`; it pulls
prebuilt wheels and skips the C/C++/CUDA toolchain.

If CUDA is required, verify both prerequisites before retrying the install:

```bash
nvcc --version
printf '%s\n' "$CUDA_HOME"
```

The first command must print the CUDA compiler version; the second must print the toolkit path. Only
then rerun the CUDA install command from the README.
