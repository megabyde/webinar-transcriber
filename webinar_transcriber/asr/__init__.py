"""ASR adapter built on top of the whisper.cpp C API."""

from .carryover import build_prompt_carryover
from .config import (
    ASR_BACKEND_NAME,
    DEFAULT_CARRYOVER_MAX_SENTENCES,
    DEFAULT_CARRYOVER_MAX_TOKENS,
    DEFAULT_WHISPER_CPP_MODEL_EXAMPLE,
    DEFAULT_WHISPER_CPP_MODEL_FILENAME,
    DEFAULT_WHISPER_CPP_MODEL_REPO,
    PromptCarryoverSettings,
    WhisperDecodeSettings,
    default_asr_threads,
)
from .transcriber import WhisperCppTranscriber

__all__ = [
    "ASR_BACKEND_NAME",
    "DEFAULT_CARRYOVER_MAX_SENTENCES",
    "DEFAULT_CARRYOVER_MAX_TOKENS",
    "DEFAULT_WHISPER_CPP_MODEL_EXAMPLE",
    "DEFAULT_WHISPER_CPP_MODEL_FILENAME",
    "DEFAULT_WHISPER_CPP_MODEL_REPO",
    "PromptCarryoverSettings",
    "WhisperCppTranscriber",
    "WhisperDecodeSettings",
    "build_prompt_carryover",
    "default_asr_threads",
]
