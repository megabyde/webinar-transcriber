# Releasing

How to cut a release of `webinar-transcriber`.

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
