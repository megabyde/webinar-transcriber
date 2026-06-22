# AGENTS

This file is the contributor and coding-agent guide for this repository.

## Scope

Documentation is organized by [Diátaxis](https://diataxis.fr) mode. Decide where content goes by
what it serves, not by which file is nearest:

- `README.md`: the entry point. A how-to quick start, user-facing reference (CLI flags, environment
  variables, output layout), and a brief explanation of what the tool is.
- `docs/pipeline.md`: explanation. How and why the stages work; no step-by-step instructions and no
  flag tables.
- `docs/troubleshooting.md`: how-to. Error-indexed fixes.
- `docs/development.md`: how-to and reference. Checkout setup, running from source, and the
  make-target and quality-gate reference.
- `docs/releasing.md`: how-to. The release process.
- `AGENTS.md`: contributor and coding-agent process the user docs do not need.

When unsure, place content by Diátaxis axis: action for someone working is how-to, information for
someone working is reference, understanding is explanation. Keep modes separate within a file: do
not fold explanation into a how-to, or a flag table into the pipeline explanation.

Prefer extending `README.md` first. Add a new file under `docs/` only when a section is long enough
to obscure the user-facing flow and self-contained enough to read on its own.

Repository-visible assets stay under `docs/assets/`. The README preview uses GitHub's theme
suffixes: `social-preview.png#gh-light-mode-only` and `social-preview-dark.png#gh-dark-mode-only`.
Keep `social-preview-dark.png` suitable for the repository social-preview setting.

## Tooling

Keep setup docs and Makefile targets in sync. Installation is documented in the README as direct uv
commands (`uv tool install`); update it there when the install flow changes. If you add, rename, or
remove a `make sync*` target or quality-gate target, update `docs/development.md` in the same
change.

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

## Package Layout

The package is deliberately flat: shallow subpackages by domain (`asr`, `diarization`, `llm`,
`transcript`, `video`, `export`) with leaf modules at the root. Read the tree directly rather than
maintaining a copy here, and avoid adding deep nesting.

## Runtime Contracts

- Successful default CLI runs write the report artifact set described in `README.md`.
- `--diarize` runs locally and adds `diarization.json` plus speaker fields on transcript segments.
- Successful runs write `diagnostics.json`; failed runs also write it once the run directory exists,
  though early failures can still leave only partial intermediate artifacts and no final report
  outputs.
- Temporary WAV audio extracted for transcription should stay outside the run directory;
  `--keep-audio` may write the compressed `transcription-audio.mp3` artifact described in
  `README.md`.

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

## Simplification and Refactoring Notes

This repo has improved repeatedly by making complexity justify itself, not the reverse. In review
the maintainer keeps pushing past a first "this is load-bearing" reading and is usually right. Carry
that prior: an abstraction must earn its keep, the default answer to "can this be simpler" is yes
until the code proves otherwise, and a challenge to a conservative answer is a request for evidence.
Grep the real readers and callers, then produce a verified cleanup or give the precise structural
reason it can't be done and act rather than re-explain. A small flag-free helper is fine when a
linter genuinely forces extraction; a state flag or multi-helper split to dodge one is
over-engineering.

Recurring smells to check for:

- **Test-only knobs** — a param/field whose only non-default caller is a test; delete it and let
  tests monkeypatch the constant or exercise the real contract.
- **1:1 wrappers / single-caller helpers** that only rename one shape or forward one call; inline at
  the call site.
- **Write-only fields** serialized or assigned but never read back; delete unless an external
  artifact contract consumes them.
- **Duplicated diagnostics or serialization** — one value under two keys or two artifacts; pick the
  single owner.
- **Wrapper / result-object chains** threading one value through re-keying passes, or a plan/spec
  wrapping a `min()`/`len()`; collapse to one shape or a function.
- **Re-export / lazy-import surfaces with no production consumer**; import from the defining leaf
  module, keeping a package re-export only to isolate an optional extra or a real public boundary.
- **Protocols for a single in-tree implementation**; type the collaborator concretely and let tests
  use duck-typed fakes. A Protocol earns its place only at a third-party or polymorphic seam.
- **Fakes re-implementing production logic**; replace with a Mock-based double that runs the real
  policy or object, so the test proves production behavior.
- **Defensive branches guarding states an earlier stage forbids** (a guard on a positive constant, a
  `getattr` default masking a missing attribute); assert the invariant or delete the branch.
- **Errors downgraded to warnings** the run then threads around; let them propagate, reserving
  warnings for recoverable, user-actionable conditions.
- **Config bags between exactly two callers**; inline as kwargs or fields.
- **Stale suppressions** — ruff ignores with zero hits, `type: ignore` / `cast()` that narrow away
  with `isinstance`; re-enable or narrow, keeping only documented stub gaps.

When acting: put ownership at the boundary that already knows the value (the CLI builds
collaborators, stages record their own diagnostics, ASR owns window policy, export/report models
carry no diagnostics-only data); prefer the narrowest cohesive cleanup with a clear before/after
story; verify with a byte-identical fixture-artifact diff where possible over tests that only prove
one helper forwarded to another; and make any user-visible contract change (CLI flags, artifact
paths, JSON shapes, diagnostics or stage keys) explicit in the PR with `README.md` and `AGENTS.md`
updated in the same change.

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
