"""Environment-variable accessors and third-party module loaders."""

from __future__ import annotations

import importlib
import os
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator
    from types import ModuleType

LLM_PROVIDER_ENV = "LLM_PROVIDER"
TQDM_DISABLE_ENV = "TQDM_DISABLE"
WEBINAR_DIARIZATION_CACHE_DIR_ENV = "WEBINAR_DIARIZATION_CACHE_DIR"


def load_sherpa_onnx() -> ModuleType | None:
    """Return the imported sherpa-onnx module, or None when its wheel is missing on this host.

    sherpa-onnx is a required dependency, so None means the wheel did not install rather than that
    the user opted out. Callers decide what that costs them: speech-region detection warns and
    falls back to one whole-file region, while speaker diarization has no degraded form and fails.

    Returns:
        ModuleType | None: The sherpa-onnx module, or None if it is not importable.
    """
    try:
        return importlib.import_module("sherpa_onnx")
    except ImportError:  # pragma: no cover - depends on host wheel availability
        return None


def llm_provider_name() -> str:
    """Return the configured LLM provider name."""
    return os.environ.get(LLM_PROVIDER_ENV, "openai").strip().casefold()


def diarization_cache_dir() -> Path | None:
    """Return the configured diarization cache directory, if any."""
    value = os.environ.get(WEBINAR_DIARIZATION_CACHE_DIR_ENV)
    return Path(value).expanduser() if value else None


@contextmanager
def temporary_environment_variable(name: str, value: str) -> Generator[None, None, None]:
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
