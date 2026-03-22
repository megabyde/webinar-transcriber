# AGENTS

This file is the contributor and coding-agent guide for this repository.

## Scope

Keep repository documentation consolidated in:

- `README.md` for user-facing usage, behavior, and artifact contracts
- `AGENTS.md` for contributor and implementation guidance

Do not add new standalone documentation files under `docs/` unless there is a strong reason.
`docs/assets/` is still used for repository assets such as the banner.

## Stack

- Python 3.12
- `uv` for dependency management and command execution
- Hatchling build backend with PEP 621 metadata in `pyproject.toml`
- `ruff` for formatting and linting
- `ty` for type checking
- `pytest` and `pytest-cov` for tests and coverage

## Native Dependencies

For local macOS development, use Homebrew:

```bash
brew install ffmpeg
```

The current pipeline does not require additional non-Python dependencies beyond `ffmpeg`.

## Verification

Run the full gate before each logical commit:

```bash
make format
make lint
make typecheck
make test
make coverage
```

## Package Layout

The package intentionally avoids deep nesting.

- `webinar_transcriber/cli.py`: Click CLI entrypoints
- `webinar_transcriber/processor.py`: high-level orchestration
- `webinar_transcriber/asr.py`: ASR backend selection and normalization
- `webinar_transcriber/media.py`: probing and transcription-audio preparation
- `webinar_transcriber/structure.py`: report heuristics
- `webinar_transcriber/ui.py`: Rich progress reporting
- `webinar_transcriber/video/`: scene detection and frame extraction
- `webinar_transcriber/export/`: Markdown, DOCX, and JSON writers

## Runtime Contracts

- `process` writes the full report artifact set.
- `extract-frames` writes `scenes.json` and `frames/`.
- `diagnostics.json` is currently success-only.
- Temporary audio extracted for transcription should stay outside the run directory.

## Testing Notes

- Keep tests colocated with the package hierarchy.
- Prefer tiny deterministic fixtures committed to the repo.
- CLI tests should generally monkeypatch the reporter and heavy runtime seams instead of invoking
  real media-processing work.

## Open Work

The main remaining product items are:

- improve scene detection quality without making it expensive
- tighten the remaining artifact and behavior edges
- consider later-stage concurrency in the processor if profiling shows it is worth the complexity
