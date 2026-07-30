# Releasing

A release is complete when the tagged GitHub Release contains the wheel and source distribution, the
same version is available from PyPI, and a clean install reports the expected version.

## Prerequisites

- Write access for tags and GitHub Releases.
- Approval access to the protected `pypi` environment.
- [GitHub CLI](https://cli.github.com/) for editing release notes.
- PyPI and TestPyPI Trusted Publishing configured as described in
  [One-time publishing setup](#one-time-publishing-setup).

## 1. Choose the version

Versioning is tag-driven. `hatch-vcs` derives the version from a `vX.Y.Z` git tag and writes it to
`webinar_transcriber/_version.py`; never edit that file by hand.

Choose the bump using semver judgment. JSON artifacts such as `report.json`, `scenes.json`, and
`diagnostics.json` are local CLI outputs, not a stable API. Changing or removing an artifact key is
a minor bump unless the project deliberately treats it otherwise. Reserve a major bump for breaks to
the CLI or installation contract.

## 2. Validate `main`

Update the checkout and run the same quality gate used by CI:

```bash
git switch main
git pull --ff-only
make format
make check
git status --short
```

`make format` may change files. If it does, stop and send those changes through the normal review
workflow before releasing. Preflight is complete when `make check` exits successfully with 100%
coverage and `git status --short` prints nothing.

## 3. Push the release tag

> [!CAUTION]
> A `vX.Y.Z` tag starts the release workflow, and PyPI versions are immutable. Confirm the version
> and preflight result before pushing the tag.

Create and push an annotated tag, replacing `vX.Y.Z` in both commands with the selected version:

```bash
git tag -a vX.Y.Z -m "vX.Y.Z"
git push origin vX.Y.Z
```

The `release.yml` workflow then:

1. Runs `make check` on Ubuntu and macOS, plus the CLI and Whisper smoke tests.
1. Builds the wheel and source distribution.
1. Creates a GitHub Release with both distributions attached.
1. Waits for approval on the protected `pypi` environment before publishing to PyPI.

Wait for validation and the GitHub Release to complete. Do not approve the `pypi` deployment yet.

## 4. Validate with TestPyPI

Before approving the production publish, run the `Release` workflow manually from GitHub Actions.
Select the new `vX.Y.Z` tag as the ref. The `publish-testpypi` job builds that version and uploads
it without creating another GitHub Release.

Install the exact version from TestPyPI in a disposable environment. Dependencies still come from
the main PyPI index:

```bash
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ webinar-transcriber==X.Y.Z
webinar-transcriber --version
```

TestPyPI validation is complete when installation succeeds and the CLI prints `X.Y.Z`.

## 5. Publish to PyPI

Approve the waiting `publish-pypi` deployment in the protected `pypi` GitHub Environment. Trusted
Publishing uses OIDC, so there is no API token to provide.

After the job succeeds, verify a clean install from PyPI:

```console
$ uvx --from "webinar-transcriber==X.Y.Z" webinar-transcriber --version
webinar-transcriber, version X.Y.Z
```

Production publishing is complete when the command prints `X.Y.Z`.

## 6. Curate the release notes

Replace the generated "What's Changed" list with reviewed notes in `release-notes.md`:

```bash
gh release edit vX.Y.Z --notes-file release-notes.md
```

Include:

- A `## Highlights` section organized by theme, with PR references written as `#NNN` so GitHub
  auto-links them.
- A `## Output changes` section when the release changes `report.json`, `scenes.json`,
  `diagnostics.json`, CLI flags, or generated artifacts.
- The **Full Changelog** comparison link.

Keep the highlights scannable; do not turn them into a per-PR dump. The release is complete when the
GitHub Release has the reviewed notes and attached distributions, PyPI has `X.Y.Z`, and the clean
install check passes.

## One-time publishing setup

Before the first publish to each index, register a pending publisher under PyPI or TestPyPI →
Account → Publishing. Match the repository exactly:

- Owner: `megabyde`
- Repository: `webinar-transcriber`
- Workflow: `release.yml`
- Environment: `pypi` or `testpypi`

Create both environments in the repository settings. Protect `pypi` with a required reviewer so a
tag-triggered publish pauses for approval.
