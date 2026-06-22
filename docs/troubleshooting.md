# Troubleshooting

Common errors and how to resolve them. Each heading matches the message the CLI prints. Every run
also writes `diagnostics.json` with the failed stage, and transcription failures leave the native
`whisper-cpp.log` in the run directory.

## `Could not open … with PyAV` / `No audio or video stream found`

The input could not be decoded, or it carries neither an audio nor a video stream. Confirm the file
plays in a normal media player and is a container PyAV can decode (for example `.mp4`, `.mkv`,
`.mov`, `.webm`, `.mp3`, `.wav`, `.m4a`). Re-mux or re-encode a corrupt or exotic container with
`ffmpeg` before transcribing.

## `Missing required LLM environment variables`

`--llm` was passed without the required provider environment variables. Set `OPENAI_API_KEY` and
`OPENAI_MODEL` for OpenAI, or set `LLM_PROVIDER=anthropic` plus `ANTHROPIC_API_KEY` and
`ANTHROPIC_MODEL` for Anthropic.

## `requires the 'llm' extra`

The provider SDKs are not installed. Reinstall the CLI with the `llm` extra:

```bash
uv tool install --reinstall "webinar-transcriber[llm]"
```

From a checkout, use `'.[llm]'` instead.

## `Unsupported LLM provider`

`LLM_PROVIDER` is set to a value other than `openai` or `anthropic`. Unset it to use OpenAI, or set
it to `anthropic`.

## `Output directory already exists`

The CLI refuses to overwrite existing run directories. Pass a new `--output-dir`, remove the
existing directory, or omit `--output-dir` so the CLI creates a fresh timestamped directory under
`runs/`.

## `Could not prepare whisper.cpp model`

`pywhispercpp` could not load the requested model. Check that `--asr-model` is a known identifier
such as `large-v3-turbo` or `large-v3`, or that a local path points to a valid GGML file. The native
`whisper-cpp.log` inside the run directory has the underlying error.

## Wrong language detected

Whisper can mis-detect language for short, multilingual, or noisy audio. Pass `--language CODE`, for
example `--language en` or `--language ru`, to force the language hint.

## Poor diarization labels

`--diarize-speakers COUNT` forces an exact speaker count. If the count is wrong, labels degrade.
Omit the flag to let `sherpa-onnx` estimate the count, or pass the correct count.

## CUDA install fails

The CUDA install rebuilds `pywhispercpp` from source and needs `nvcc` on `PATH` and `CUDA_HOME` set.
If you do not need NVIDIA acceleration, use the standard `uv tool install --reinstall .`; it pulls
prebuilt wheels and skips the C/C++/CUDA toolchain.
