"""ASR configuration, defaults, and host-capability helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass

ASR_BACKEND_NAME = "whisper.cpp"
DEFAULT_WHISPER_CPP_MODEL_FILENAME = "large-v3-turbo"
DEFAULT_WHISPER_CPP_MODEL_EXAMPLE = "models/whisper-cpp/ggml-large-v3-turbo.bin"
DEFAULT_CARRYOVER_MAX_TOKENS = 96
DEFAULT_WHISPER_ENTROPY_THOLD = 2.4
DEFAULT_WHISPER_LOGPROB_THOLD = -1.0
DEFAULT_WHISPER_NO_SPEECH_THOLD = 0.6
DEFAULT_WHISPER_TEMPERATURE_INC = 0.2
DEFAULT_MAX_ASR_THREADS = 8


def default_asr_threads() -> int:
    """Return the preferred default whisper.cpp thread count for this host."""
    return min(max(1, os.cpu_count() or 4), DEFAULT_MAX_ASR_THREADS)


@dataclass(frozen=True)
class PromptCarryoverSettings:
    """Configuration for bounded prompt carryover between adjacent windows."""

    enabled: bool = True
    max_tokens: int = DEFAULT_CARRYOVER_MAX_TOKENS
