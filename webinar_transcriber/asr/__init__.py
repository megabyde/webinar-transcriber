"""ASR adapter built on top of the whisper.cpp C API."""

from __future__ import annotations

from .config import (
    ASR_BACKEND_NAME,
    WHISPER_CPP_MODEL_EXAMPLE,
    WHISPER_CPP_MODEL_FILENAME,
    default_asr_threads,
)
from .transcriber import AsrProcessingError, WhisperCppTranscriber
from .windows import plan_inference_windows

__all__ = [
    "ASR_BACKEND_NAME",
    "WHISPER_CPP_MODEL_EXAMPLE",
    "WHISPER_CPP_MODEL_FILENAME",
    "AsrProcessingError",
    "WhisperCppTranscriber",
    "default_asr_threads",
    "plan_inference_windows",
]
