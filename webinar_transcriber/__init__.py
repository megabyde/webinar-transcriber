"""Project metadata for webinar-transcriber."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

__all__ = ["__version__", "get_version"]

try:
    from webinar_transcriber._version import __version__ as _source_version
except ModuleNotFoundError:  # pragma: no cover
    _source_version = "0.0.0"


def get_version() -> str:
    """Return the installed package version or generated fallback version."""
    try:
        return version("webinar-transcriber")
    except PackageNotFoundError:
        return _source_version


__version__ = get_version()
