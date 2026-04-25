"""Project metadata for webinar-transcriber."""

from __future__ import annotations

from importlib import import_module

__all__ = ["__version__"]


def _resolve_version() -> str:
    try:
        return import_module("webinar_transcriber._version").__version__
    except ModuleNotFoundError:
        return "0.0.0"


__version__ = _resolve_version()
