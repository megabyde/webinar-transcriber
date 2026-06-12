"""ASR adapter built on top of the whisper.cpp C API."""

from __future__ import annotations

from webinar_transcriber.asr.config import (
    ASR_BACKEND_NAME,
    WHISPER_CPP_MODEL_EXAMPLE,
    WHISPER_CPP_MODEL_FILENAME,
    default_asr_threads,
)
from webinar_transcriber.asr.transcriber import AsrProcessingError, WhisperCppTranscriber
from webinar_transcriber.asr.windows import plan_inference_windows

__all__ = [
    "ASR_BACKEND_NAME",
    "WHISPER_CPP_MODEL_EXAMPLE",
    "WHISPER_CPP_MODEL_FILENAME",
    "AsrProcessingError",
    "WhisperCppTranscriber",
    "default_asr_threads",
    "plan_inference_windows",
]
