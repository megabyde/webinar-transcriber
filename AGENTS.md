# AGENTS

This file is the contributor and coding-agent guide for this repository.

## Scope

Keep repository documentation consolidated in:

- `README.md` for user-facing usage, behavior, and artifact contracts
- `AGENTS.md` for contributor and implementation guidance

Do not add new standalone documentation files under `docs/` unless there is a strong reason. If
repository assets are needed, keep them under `docs/assets/`.

## Stack

- Python 3.12
- `uv` for dependency management and command execution
- Hatchling build backend with PEP 621 metadata in `pyproject.toml`
- `ruff` for formatting and linting
- `ty` for type checking
- `pytest` and `pytest-cov` for tests and coverage

## Native Dependencies

For local macOS runtime and development, use Homebrew:

```bash
brew install ffmpeg whisper-cpp
```

For Linux runtime, install `ffmpeg` plus a `libwhisper` package such as `libwhisper-cpp-dev`.

CI and most tests only require `ffmpeg`. A full local CLI run also needs the `whisper.cpp` runtime
library to be available.

## Verification

Run the full gate before each logical commit:

```bash
make format
make lint
make test
```

In sandboxed Codex runs, prefer `UV_CACHE_DIR=/tmp/uv-cache` for `uv` and `make` commands to avoid
cache-permission failures.

## Package Layout

The package intentionally avoids deep nesting.

- `webinar_transcriber/cli.py`: Click CLI entrypoints
- `webinar_transcriber/processor.py` and `webinar_transcriber/processor_*`: high-level orchestration
  and processor support helpers
- `webinar_transcriber/asr.py`: ASR backend selection and normalization
- `webinar_transcriber/llm/`: optional cloud LLM integrations
- `webinar_transcriber/media.py`: probing and shared ffmpeg/ffprobe command execution
- `webinar_transcriber/transcription_audio.py`: deterministic transcription-audio preparation
- `webinar_transcriber/structure.py` and `webinar_transcriber/structure_*`: report heuristics
- `webinar_transcriber/reporter.py`: reporter protocol and no-op implementation
- `webinar_transcriber/ui.py`: Rich progress reporting
- `webinar_transcriber/video/`: scene detection and frame extraction
- `webinar_transcriber/export/`: Markdown, DOCX, and JSON writers

## Runtime Contracts

- Successful `process` runs write the report artifact set described in `README.md`.
- `extract-frames` writes `scenes.json` and `frames/`.
- `diagnostics.json` is currently success-only; early failures can leave partial intermediate
  artifacts without diagnostics or final report outputs.
- Temporary audio extracted for transcription should stay outside the run directory.

## Testing Notes

- Keep tests colocated with the package hierarchy.
- Prefer tiny deterministic fixtures committed to the repo.
- Prefer `pytest` `monkeypatch` for simple state replacement and `unittest.mock.patch` when a test
  needs mock semantics such as `side_effect`, `return_value`, or call assertions.
- Group related tests by behavior using `class Test...` when it improves scanability; keep helpers
  and fakes scoped to the narrowest test class that uses them, and leave them module-level only when
  they are intentionally shared across multiple test classes or fixtures.
- CLI tests should generally monkeypatch the reporter and heavy runtime seams instead of invoking
  real media-processing work.
- Use `pytest.mark.parametrize` when cases share the same setup and assertions and the test
  naturally becomes a clear input/output table; avoid it when cases need materially different
  fixtures, monkeypatching, or fake-object wiring.
- Prefer asserting current observable behavior over asserting that recently removed options, fields,
  or artifacts are absent. Deletion-based assertions are usually shallow and brittle.
- For predicate-style tests, prefer `assert expr` / `assert not expr`. Reserve `is True` /
  `is False` for cases where exact boolean identity matters.

## Implementation Notes

- When an earlier pipeline stage already guarantees an invariant, prefer enforcing that invariant
  directly instead of carrying defensive conversion logic downstream. For example, transcription
  audio is normalized to `16000 Hz`, so the Silero integration should assert that contract rather
  than silently resample.

## Style Notes

- Use the walrus operator selectively when it removes repeated work without making the code harder
  to read.
- Prefer short loop and comprehension variable names when the surrounding context already makes the
  meaning obvious.
- In ASR code and artifacts, use `speech region` for VAD/planning inputs and `window` for Whisper
  decode units; avoid `chunk` for ASR concepts.

## Open Work

The main remaining product items are:

- improve scene detection quality without making it expensive
- tighten the remaining artifact and behavior edges
- consider later-stage concurrency in the processor if profiling shows it is worth the complexity
