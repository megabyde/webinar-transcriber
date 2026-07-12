# Releasing

How to cut and publish a release of `webinar-transcriber`.

## Choose the version

Versioning is tag-driven. `hatch-vcs` derives the version from a `vX.Y.Z` git tag and writes it to
`webinar_transcriber/_version.py`; never edit that file by hand.

Choose the bump using semver judgment. JSON artifacts such as `report.json`, `scenes.json`, and
`diagnostics.json` are local CLI outputs, not a stable API. Changing or removing an artifact key is
a minor bump unless the project deliberately treats it otherwise. Reserve a major bump for breaks to
the CLI or installation contract.

## Cut the release

Once `main` is green, push an annotated `vX.Y.Z` tag. The `release.yml` workflow then:

1. Runs lint and `make test-all` on Ubuntu and macOS, plus the CLI and Whisper smoke tests.
1. Builds the wheel and source distribution.
1. Creates a GitHub Release with both distributions attached.
1. Waits for approval on the protected `pypi` environment, then publishes to PyPI.

## Curate the release notes

After the release exists, replace the generated "What's Changed" list with
`gh release edit --notes-file`. Include:

- A `## Highlights` section organized by theme, with PR references written as `#NNN` so GitHub
  auto-links them.
- A `## Output changes` section when the release changes `report.json`, `scenes.json`,
  `diagnostics.json`, CLI flags, or generated artifacts.
- The **Full Changelog** comparison link.

Keep the highlights scannable; do not turn them into a per-PR dump.

## Publishing to PyPI

Publishing uses [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC), so
there are no API tokens to store. On a tag push, the `release.yml` `publish-pypi` job runs under the
protected `pypi` GitHub Environment. It uploads the wheel and source distribution after a maintainer
approves the deployment.

### Dry-run with TestPyPI

PyPI versions are immutable, so validate the release against TestPyPI first. Run the `Release`
workflow manually (`workflow_dispatch`) and select the latest tag as the ref. The `publish-testpypi`
job builds that version and uploads it without creating a GitHub Release.

Confirm that installation resolves. Dependencies still come from the main PyPI index:

```bash
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ webinar-transcriber
```

### Configure Trusted Publishing

Before the first publish to each index, register a pending publisher under PyPI or TestPyPI →
Account → Publishing. Match the repository exactly:

- Owner: `megabyde`
- Repository: `webinar-transcriber`
- Workflow: `release.yml`
- Environment: `pypi` or `testpypi`

Create both environments in the repository settings. Protect `pypi` with a required reviewer so a
tag-triggered publish pauses for approval.
