"""Project metadata for webinar-transcriber."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

__all__ = ["__version__", "get_version"]

try:
    from webinar_transcriber import _version
except ModuleNotFoundError:  # pragma: no cover
    _source_version: str | None = None
else:
    _source_version = _version.__version__


def get_version() -> str:
    """Return the generated source version or installed package version."""
    if _source_version is not None:
        return _source_version
    try:
        return version("webinar-transcriber")
    except PackageNotFoundError:
        return "0.0.0"


__version__ = get_version()
