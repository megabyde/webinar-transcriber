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
  smoke), builds the wheel and sdist, and creates a GitHub Release with those assets. It then waits
  for approval on the `pypi` environment and publishes to PyPI (see Publishing to PyPI below).
- **Curate the release notes.** After the release exists, replace the auto-generated "What's
  Changed" list (`gh release edit --notes-file`) with: a `## Highlights` section that cites PRs
  inline as `#NNN` (GitHub auto-links them); an `## Output changes` heads-up whenever the work
  touched `report.json`/`scenes.json`/`diagnostics.json`, CLI flags, or generated artifacts; and the
  **Full Changelog** compare link. Keep highlights scannable — themes, not a per-PR dump.

## Publishing to PyPI

Publishing uses [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC), so
there are no API tokens to store. On a tag push the `release.yml` `publish-pypi` job runs under the
protected `pypi` GitHub Environment and uploads the built wheel and sdist after a maintainer
approves the deployment.

**Dry-run before a real release.** PyPI versions are immutable, so validate first against TestPyPI.
Run the `Release` workflow manually (`workflow_dispatch`) selecting the latest tag as the ref; the
`publish-testpypi` job builds at that version and uploads to TestPyPI without cutting a GitHub
Release. Confirm the install resolves (dependencies still come from real PyPI):

```bash
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ webinar-transcriber
```

**One-time Trusted Publishing setup.** Before the first publish to each index, register a pending
publisher (PyPI/TestPyPI → Account → Publishing) matching this repo exactly: owner `megabyde`,
repository `webinar-transcriber`, workflow `release.yml`, and environment `pypi` (or `testpypi`).
Create both GitHub Environments under repo Settings; protect `pypi` with a required reviewer so the
tag-triggered publish pauses for approval.
