"""ASR adapter built on top of the whisper.cpp C API."""

from __future__ import annotations

from .carryover import build_prompt_carryover
from .config import (
    ASR_BACKEND_NAME,
    WHISPER_CPP_MODEL_EXAMPLE,
    WHISPER_CPP_MODEL_FILENAME,
    default_asr_threads,
)
from .transcriber import ASRProcessingError, WhisperCppTranscriber, device_name_from_system_info

__all__ = [
    "ASR_BACKEND_NAME",
    "WHISPER_CPP_MODEL_EXAMPLE",
    "WHISPER_CPP_MODEL_FILENAME",
    "ASRProcessingError",
    "WhisperCppTranscriber",
    "build_prompt_carryover",
    "default_asr_threads",
    "device_name_from_system_info",
]
