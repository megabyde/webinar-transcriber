"""Project metadata for webinar-transcriber."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

__all__ = ["__version__"]

try:
    from webinar_transcriber._version import __version__ as _source_version
except ModuleNotFoundError:
    _source_version = "0.0.0"


def _resolve_version() -> str:
    try:
        return version("webinar-transcriber")
    except PackageNotFoundError:
        return _source_version


__version__ = _resolve_version()
