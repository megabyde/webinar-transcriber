"""ASR configuration, defaults, and host-capability helpers."""

from __future__ import annotations

import os

ASR_BACKEND_NAME = "whisper.cpp"
WHISPER_CPP_MODEL_FILENAME = "large-v3-turbo"
WHISPER_CPP_MODEL_EXAMPLE = "models/whisper-cpp/ggml-large-v3-turbo.bin"
CARRYOVER_MAX_CHARS = 300
WHISPER_ENTROPY_THOLD = 2.4
WHISPER_LOGPROB_THOLD = -1.0
WHISPER_NO_SPEECH_THOLD = 0.6
WHISPER_TEMPERATURE_INC = 0.2
DEFAULT_MAX_ASR_THREADS = 8


def default_asr_threads() -> int:
    """Return the preferred default whisper.cpp thread count for this host."""
    return min(max(1, os.cpu_count() or 4), DEFAULT_MAX_ASR_THREADS)
