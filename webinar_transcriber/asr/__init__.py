"""ASR adapter built on top of the whisper.cpp C API."""

from __future__ import annotations

from .carryover import build_prompt_carryover
from .config import (
    ASR_BACKEND_NAME,
    DEFAULT_CARRYOVER_MAX_SENTENCES,
    DEFAULT_CARRYOVER_MAX_TOKENS,
    DEFAULT_WHISPER_CPP_MODEL_EXAMPLE,
    DEFAULT_WHISPER_CPP_MODEL_FILENAME,
    PromptCarryoverSettings,
    WhisperDecodeSettings,
    default_asr_threads,
)
from .transcriber import ASRProcessingError, WhisperCppTranscriber

__all__ = [
    "ASR_BACKEND_NAME",
    "DEFAULT_CARRYOVER_MAX_SENTENCES",
    "DEFAULT_CARRYOVER_MAX_TOKENS",
    "DEFAULT_WHISPER_CPP_MODEL_EXAMPLE",
    "DEFAULT_WHISPER_CPP_MODEL_FILENAME",
    "ASRProcessingError",
    "PromptCarryoverSettings",
    "WhisperCppTranscriber",
    "WhisperDecodeSettings",
    "build_prompt_carryover",
    "default_asr_threads",
]
