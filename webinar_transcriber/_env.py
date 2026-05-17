"""Typed environment-variable accessors."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

LLM_PROVIDER_ENV = "LLM_PROVIDER"
TQDM_DISABLE_ENV = "TQDM_DISABLE"
WEBINAR_DIARIZATION_CACHE_DIR_ENV = "WEBINAR_DIARIZATION_CACHE_DIR"


def llm_provider_name() -> str:
    """Return the configured LLM provider name."""
    return os.environ.get(LLM_PROVIDER_ENV, "openai").strip().casefold()


def diarization_cache_dir() -> Path | None:
    """Return the configured diarization cache directory, if any."""
    value = os.environ.get(WEBINAR_DIARIZATION_CACHE_DIR_ENV)
    return Path(value).expanduser() if value else None


@contextmanager
def temporary_environment_variable(name: str, value: str) -> Iterator[None]:
    """Temporarily set one environment variable."""
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous
