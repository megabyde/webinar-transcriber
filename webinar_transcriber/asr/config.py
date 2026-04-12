"""ASR configuration, defaults, and host-capability helpers."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

ASR_BACKEND_NAME = "whisper.cpp"
DEFAULT_WHISPER_CPP_MODEL_REPO = "ggerganov/whisper.cpp"
DEFAULT_WHISPER_CPP_MODEL_FILENAME = "ggml-large-v3-turbo.bin"
DEFAULT_WHISPER_CPP_MODEL_EXAMPLE = Path("models/whisper-cpp/ggml-large-v3-turbo.bin")
DEFAULT_CARRYOVER_MAX_SENTENCES = 2
DEFAULT_CARRYOVER_MAX_TOKENS = 64
SYSCTL_TIMEOUT_SEC = 1.0


def _read_sysctl_int(name: str) -> int | None:
    try:
        result = subprocess.run(
            ["sysctl", "-n", name],
            check=True,
            capture_output=True,
            text=True,
            timeout=SYSCTL_TIMEOUT_SEC,
        )
    except (
        FileNotFoundError,
        PermissionError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ):
        return None

    value = result.stdout.strip()
    if not value.isdigit():
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


def default_asr_threads() -> int:
    return (
        _read_sysctl_int("hw.perflevel0.physicalcpu")
        or _read_sysctl_int("hw.physicalcpu")
        or os.cpu_count()
        or 4
    )


DEFAULT_ASR_THREADS = default_asr_threads()


@dataclass(frozen=True)
class PromptCarryoverSettings:
    """Configuration for bounded prompt carryover between adjacent windows."""

    enabled: bool = True
    max_sentences: int = DEFAULT_CARRYOVER_MAX_SENTENCES
    max_tokens: int = DEFAULT_CARRYOVER_MAX_TOKENS


@dataclass(frozen=True)
class WhisperDecodeSettings:
    """Decode-time inference settings kept above the low-level whisper.cpp wrapper."""

    carryover: PromptCarryoverSettings = PromptCarryoverSettings()
