# AGENTS

This file is the contributor and coding-agent guide for this repository.

## Scope

Keep repository documentation consolidated in:

- `README.md` for user-facing usage, behavior, and artifact contracts
- `AGENTS.md` for contributor and implementation guidance
- `docs/` for reference-style internals that would bloat `README.md` without helping a typical user
  (currently: `docs/pipeline.md` — per-stage pipeline detail)

Prefer extending `README.md` first. Add a new file under `docs/` only when a section is long enough
to obscure the user-facing flow and self-contained enough to read on its own. Repository assets stay
under `docs/assets/`.

## Tooling

README owns the user-facing install and development setup, including native dependencies and the
standard verification commands. Keep AGENTS focused on contributor-specific guidance that README
does not need.

Keep README and Makefile setup targets in sync. If you add, rename, or remove a `make sync*` or
`make install*` target, update README in the same change so local checkout setup remains
discoverable.

Use `make test` for quick iteration (fast subset, skips slow tests, no coverage gate). Run
`make format` then `make check` before committing — `make check` is the full coverage-gated gate.

GitHub Actions runs `make lint` and `make test-all` so CI mirrors the local quality gate. CI runs on
`ubuntu-latest` and `macos-latest`; Windows is not part of the CI matrix and is treated as
best-effort.

In sandboxed Codex runs, prefer `UV_CACHE_DIR=/tmp/uv-cache` for `uv` and `make` commands to avoid
cache-permission failures.

## Definition of Done

A change is ready to merge when all of the following hold:

- **Quality gates pass.** `make format && make check` runs cleanly — the full coverage-gated test
  suite, `ruff`, `ty`, and the markdown linters.
- **100% coverage stays.** Add `# pragma: no cover` to genuinely unreachable defensive branches
  rather than lowering the threshold.
- **Docs stay in sync.** If the change affects CLI flags, output artifacts, install targets, package
  layout, or runtime contracts, update `README.md` and `AGENTS.md` in the same PR.
- **No dead code.** Remove unused imports, helpers with no callers, fields that are written or
  serialized but never read back, and `# TODO` comments that will not be addressed in this PR.
- **PR has a summary and a test plan.** Each pull request description includes a brief summary and
  an explicit test-plan checklist.
- **Commits follow Conventional Commits.** Prefix commit messages with `refactor:`, `docs:`, `ci:`,
  `style:`, `feat:`, `fix:`, `test:`, `build:`, or `chore:`. Use an optional scope when it adds
  clarity, for example `build(deps):`.

## Releasing

- **Versioning is tag-driven.** `hatch-vcs` derives the version from a `vX.Y.Z` git tag into
  `webinar_transcriber/_version.py`; never hand-edit a version.
- **Choose the bump by semver judgment.** The JSON artifacts (`report.json`, `scenes.json`,
  `diagnostics.json`) are local CLI outputs, not a stable API — a changed or removed artifact key is
  a minor bump, not major, unless deliberately treated otherwise. Reserve major for CLI-flag or
  install-contract breaks.
- **Cut a release by pushing the tag.** With `main` green, push an annotated `vX.Y.Z` tag; the
  `release.yml` workflow validates (lint and `make test-all` on Ubuntu and macOS plus CLI/whisper
  smoke), builds the wheel and sdist, and creates a GitHub Release with those assets. There is no
  PyPI publish.
- **Curate the release notes.** After the release exists, replace the auto-generated "What's
  Changed" list (`gh release edit --notes-file`) with: a `## Highlights` section that cites PRs
  inline as `#NNN` (GitHub auto-links them); an `## Output changes` heads-up whenever the work
  touched `report.json`/`scenes.json`/`diagnostics.json`, CLI flags, or generated artifacts; and the
  **Full Changelog** compare link. Keep highlights scannable — themes, not a per-PR dump.

## Package Layout

The package intentionally avoids deep nesting.

```text
webinar_transcriber/
├── cli.py                  Click CLI entrypoints
├── processor.py            pipeline orchestration and RunContext dataclasses
├── asr/                    window planning, carryover policy, pywhispercpp wrapper
├── diarization/            optional local speaker diarization via sherpa-onnx
├── llm/                    optional cloud LLM integrations
├── transcript/             normalization and window reconciliation
├── video/                  scene detection and frame extraction
├── export/                 Markdown, DOCX, and JSON writers
├── diagnostics.py          run-diagnostics assembly and persistence
├── paths.py                RunLayout — deterministic run-directory construction
├── media.py                probing helpers and shared media error types
├── models.py               shared dataclasses and domain types
├── normalized_audio.py     deterministic transcription-audio preparation
├── segmentation.py         speech-region detection and duration helpers
├── structure.py            transcript-scene alignment and report heuristics
├── text_utils.py           paragraph splitting and sentence terminator helpers
├── ui.py                   StageReporter — Rich-backed progress (track() context manager)
└── tests/                  flat test suite; committed fixtures under tests/fixtures/
```

## Runtime Contracts

- Successful default CLI runs write the report artifact set described in `README.md`.
- `--diarize` runs locally and adds `diarization.json` plus speaker fields on transcript segments.
- Successful runs write `diagnostics.json`; failed runs also write it once the run directory exists,
  though early failures can still leave only partial intermediate artifacts and no final report
  outputs.
- Temporary audio extracted for transcription should stay outside the run directory.

## Testing Notes

- Keep tests under `webinar_transcriber/tests/`, mirroring the package hierarchy in filenames and
  grouping where it helps scanability. Split a test file into a sibling when it crosses roughly 800
  lines or when the production-side module it covers has already split — for example, `llm/utils.py`
  and `llm/processor.py` map to `test_llm_utils.py` and `test_llm.py`.
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
  dependencies, import paths, or artifacts are absent. Deletion-based assertions are usually shallow
  and brittle.
- Avoid tests that only prove one function forwards fields into another helper unless that wiring is
  itself a meaningful behavioral contract.
- For predicate-style tests, prefer `assert expr` / `assert not expr`. Reserve `is True` /
  `is False` for cases where exact boolean identity matters.

## Implementation Notes

- When an earlier pipeline stage already guarantees an invariant, prefer enforcing that invariant
  directly instead of carrying defensive conversion logic downstream. For example, transcription
  audio is normalized to `16000 Hz`, so the `sherpa-onnx` Silero VAD integration should assert that
  contract rather than silently resample.
- Let genuine errors propagate. Reserve warnings for recoverable, user-actionable conditions; do not
  downgrade a real failure (a failed image write, a missing required stream) into a warning that the
  run then has to thread around.
- The Silero VAD ONNX model is vendored under `webinar_transcriber/assets/` so default speech-region
  detection does not require PyTorch or a first-run network download.
- Speaker diarization uses `sherpa-onnx` with downloaded ONNX models cached under
  `~/.cache/webinar-transcriber/diarization`; keep those model artifacts out of the wheel.

## Style Notes

- `cast()` is a runtime no-op and a smell when used purely to satisfy the type checker. Prefer
  `isinstance()` checks for narrowing. Reserve `cast()` for genuine third-party stub gaps and always
  add a comment explaining which stub is missing and why.
- `# type: ignore` is the same smell. Only use it for genuine stub gaps, bare (no bracket error
  codes — `ty` does not use mypy's `[attr-defined]` syntax), with a comment on the same or preceding
  line.
- When adapting third-party types that don't stub well, define a private `Protocol`. If `Any` is
  unavoidable in a Protocol method signature due to stub-level invariance constraints, suppress ruff
  with `# noqa: ANN401` and add a comment.
- Bind caught exceptions as `ex` — `except SomeError as ex:` — never `as error:` or `as exc:`. The
  same applies to `pytest.raises(...) as ex` in tests.
- Replace `assert x is not None` guards in production code with explicit `if x is None: raise`.
  `assert` is stripped by optimised builds and banned by ruff S101.
- Use the walrus operator selectively when it removes repeated work without making the code harder
  to read.
- Prefer short loop and comprehension variable names when the surrounding context already makes the
  meaning obvious.
- In ASR code and artifacts, use `speech region` for VAD/planning inputs and `window` for Whisper
  decode units; avoid `chunk` for ASR concepts.
- Treat acronyms as words in PascalCase identifiers: `LlmProcessor`, `AsrPipelineDiagnostics`,
  `WhisperCppTranscriber`. Constants stay SCREAMING_SNAKE: `ASR_BACKEND_NAME`, `LLM_PROVIDER_ENV`.
