"""ASR adapter built on top of the whisper.cpp C API."""

from .carryover import (
    _carryover_drop_reason,
    _sanitize_prompt,
    build_prompt_carryover,
)
from .config import (
    ASR_BACKEND_NAME,
    DEFAULT_ASR_THREADS,
    DEFAULT_CARRYOVER_MAX_SENTENCES,
    DEFAULT_CARRYOVER_MAX_TOKENS,
    DEFAULT_WHISPER_CPP_MODEL_EXAMPLE,
    DEFAULT_WHISPER_CPP_MODEL_FILENAME,
    DEFAULT_WHISPER_CPP_MODEL_REPO,
    PromptCarryoverSettings,
    WhisperDecodeSettings,
    _read_sysctl_int,
    default_asr_threads,
)
from .transcriber import WhisperCppTranscriber, _device_name_from_system_info

__all__ = [
    "ASR_BACKEND_NAME",
    "DEFAULT_ASR_THREADS",
    "DEFAULT_CARRYOVER_MAX_SENTENCES",
    "DEFAULT_CARRYOVER_MAX_TOKENS",
    "DEFAULT_WHISPER_CPP_MODEL_EXAMPLE",
    "DEFAULT_WHISPER_CPP_MODEL_FILENAME",
    "DEFAULT_WHISPER_CPP_MODEL_REPO",
    "PromptCarryoverSettings",
    "WhisperCppTranscriber",
    "WhisperDecodeSettings",
    "_carryover_drop_reason",
    "_device_name_from_system_info",
    "_read_sysctl_int",
    "_sanitize_prompt",
    "build_prompt_carryover",
    "default_asr_threads",
]
