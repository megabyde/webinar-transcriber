"""ASR adapter built on top of the whisper.cpp C API."""

from __future__ import annotations

import importlib
import os
import re
import subprocess
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from webinar_transcriber.whispercpp import (
    GPU_BACKEND_PATTERN,
    WhisperCppLibrary,
    WhisperCppRuntimeDetails,
    WhisperCppSession,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from webinar_transcriber.models import DecodedWindow, InferenceWindow


ASR_BACKEND_NAME = "whisper.cpp"
DEFAULT_WHISPER_CPP_MODEL_REPO = "ggerganov/whisper.cpp"
DEFAULT_WHISPER_CPP_MODEL_FILENAME = "ggml-large-v3-turbo.bin"
DEFAULT_WHISPER_CPP_MODEL_EXAMPLE = Path("models/whisper-cpp/ggml-large-v3-turbo.bin")
_CARRYOVER_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_CARRYOVER_WHITESPACE = re.compile(r"\s+")
_CARRYOVER_TRAILING_NOISE = re.compile(r"[\(\[\{<\"'`]+$|[\s\-:,;]+$")
_HALLUCINATION_PHRASE_PATTERN = re.compile(r"(?i)thank you for watching|subscribe|like and share")
_CARRYOVER_WORD_RE = re.compile(r"[\w']+")


def _read_sysctl_int(name: str) -> int | None:
    try:
        result = subprocess.run(
            ["sysctl", "-n", name],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, PermissionError, subprocess.CalledProcessError):
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
DEFAULT_CARRYOVER_MAX_SENTENCES = 2
DEFAULT_CARRYOVER_MAX_TOKENS = 64


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


def _missing_model_error_message(model_path: Path) -> str:
    example_path = DEFAULT_WHISPER_CPP_MODEL_EXAMPLE
    location_hint = f"Download a whisper.cpp model there, or use --asr-model {example_path}."
    return (
        f"whisper.cpp model file does not exist: {model_path}. "
        f"{location_hint} See README.md for model download instructions."
    )


def _default_model_download_error_message() -> str:
    return (
        "Could not resolve the default whisper.cpp model from the Hugging Face cache. "
        "The app uses Hugging Face cache defaults for the built-in model. "
        f"To use a manual file instead, pass --asr-model {DEFAULT_WHISPER_CPP_MODEL_EXAMPLE}. "
        "See README.md for model download instructions."
    )


class WhisperCppTranscriber:
    """ASR implementation using the in-process whisper.cpp C API."""

    def __init__(
        self,
        model_name: str | None = None,
        *,
        threads: int = 4,
        carryover_settings: PromptCarryoverSettings | None = None,
        library_path: Path | None = None,
        log_path: Path | None = None,
    ) -> None:
        self._configured_model_path = Path(model_name) if model_name else None
        self._uses_default_model_path = model_name is None
        self._model_path: Path | None = self._configured_model_path
        self._threads = max(1, threads)
        self._decode_settings = WhisperDecodeSettings(
            carryover=carryover_settings or PromptCarryoverSettings()
        )
        self._library_path = library_path
        self._log_path = log_path
        self._runtime_details: WhisperCppRuntimeDetails | None = None
        self._runtime: WhisperCppLibrary | None = None
        self._session: WhisperCppSession | None = None

    @property
    def model_name(self) -> str:
        if self._model_path is not None:
            return str(self._model_path)
        return f"{DEFAULT_WHISPER_CPP_MODEL_REPO}/{DEFAULT_WHISPER_CPP_MODEL_FILENAME}"

    @property
    def device_name(self) -> str:
        if self._runtime_details is None:
            return "auto"
        return _device_name_from_system_info(self._runtime_details.system_info)

    @property
    def threads(self) -> int:
        return self._threads

    @property
    def system_info(self) -> str | None:
        if self._runtime_details is None:
            return None
        return self._runtime_details.system_info

    @property
    def library_path(self) -> str | None:
        if self._runtime_details is None:
            return None
        return self._runtime_details.library_path

    def set_log_path(self, log_path: Path) -> None:
        self._log_path = log_path

    def prepare_model(self) -> None:
        self._model_path = self._resolve_model_path()
        self.close()
        assert self._model_path is not None
        runtime = WhisperCppLibrary(self._library_path, log_path=self._log_path)
        session = runtime.create_session(self._model_path)
        self._runtime = runtime
        self._session = session
        self._runtime_details = session.runtime_details

    def transcribe_inference_windows(
        self,
        audio_samples,
        windows: list[InferenceWindow],
        *,
        progress_callback: Callable[[float, int], None] | None = None,
    ) -> list[DecodedWindow]:
        session = self._ensure_session()
        ordered_windows = sorted(windows)
        language_hint: str | None = None
        carryover_prompt: str | None = None
        decoded_windows: list[DecodedWindow] = []
        decoded_segment_count = 0

        for window in ordered_windows:
            decoded_window = session.decode_window(
                audio_samples,
                window,
                threads=self._threads,
                prompt=carryover_prompt,
                language_hint=language_hint,
            )
            decoded_windows.append(
                decoded_window.model_copy(update={"input_prompt": carryover_prompt})
            )
            decoded_segment_count += len(decoded_window.segments)
            next_carryover = build_prompt_carryover(
                decoded_window,
                settings=self._decode_settings.carryover,
            )
            language_hint = language_hint or decoded_window.language
            carryover_prompt = next_carryover
            if progress_callback is not None:
                progress_callback(window.end_sec, decoded_segment_count)
        return decoded_windows

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None
        self._runtime = None

    def _ensure_session(self) -> WhisperCppSession:
        if self._session is None:
            self.prepare_model()
        assert self._session is not None
        return self._session

    def _resolve_model_path(self) -> Path:
        if not self._uses_default_model_path:
            assert self._configured_model_path is not None
            if not self._configured_model_path.exists():
                raise RuntimeError(_missing_model_error_message(self._configured_model_path))
            return self._configured_model_path

        try:
            return _download_default_whisper_cpp_model()
        except RuntimeError as error:
            raise RuntimeError(f"{_default_model_download_error_message()} {error}") from error

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown cleanup
        with suppress(Exception):
            self.close()


def _device_name_from_system_info(system_info: str) -> str:
    match = GPU_BACKEND_PATTERN.search(system_info)
    if match is not None:
        backend_name = match.group(1).lower()
        return "metal" if backend_name == "mtl" else backend_name
    return "cpu"


def _download_default_whisper_cpp_model() -> Path:
    try:
        huggingface_hub = importlib.import_module("huggingface_hub")
    except ModuleNotFoundError as error:  # pragma: no cover - dependency wiring only
        raise RuntimeError(
            "huggingface_hub is not installed, so the default whisper.cpp model cannot be "
            "downloaded automatically."
        ) from error

    try:
        downloaded_path = huggingface_hub.hf_hub_download(
            repo_id=DEFAULT_WHISPER_CPP_MODEL_REPO,
            filename=DEFAULT_WHISPER_CPP_MODEL_FILENAME,
        )
    except Exception as error:  # pragma: no cover - network/backend specific
        raise RuntimeError(
            "Automatic download of the default whisper.cpp model from Hugging Face failed."
        ) from error

    return Path(downloaded_path)


def build_prompt_carryover(
    decoded_window: DecodedWindow,
    *,
    settings: PromptCarryoverSettings,
) -> str | None:
    """Return a bounded prompt suffix for the next window, or `None` when confidence is weak."""
    if not settings.enabled or not _window_is_confident(decoded_window, settings=settings):
        return None

    sentences = [
        stripped
        for part in _CARRYOVER_SENTENCE_SPLIT.split(decoded_window.text)
        if (stripped := part.strip())
    ]
    carryover = " ".join(sentences[-max(1, settings.max_sentences) :])
    carryover = _sanitize_prompt(carryover, max_tokens=settings.max_tokens)
    return carryover or None


def _window_is_confident(
    decoded_window: DecodedWindow,
    *,
    settings: PromptCarryoverSettings,
) -> bool:
    return _carryover_drop_reason(decoded_window, settings=settings) is None


def _carryover_drop_reason(
    decoded_window: DecodedWindow,
    *,
    settings: PromptCarryoverSettings,
) -> str | None:
    if not settings.enabled:
        return "carryover_disabled"
    if decoded_window.fallback_used:
        return "fallback_used"
    if not decoded_window.text.strip():
        return "empty_text"
    if _looks_like_hallucination(decoded_window.text):
        return "hallucination_detected"
    return None


def _looks_like_hallucination(text: str) -> bool:
    if _HALLUCINATION_PHRASE_PATTERN.search(text):
        return True

    words = _CARRYOVER_WORD_RE.findall(text.casefold())
    if not words:
        return False

    window_size = min(len(words), 10)
    for start in range(len(words) - window_size + 1):
        window = words[start : start + window_size]
        repeated_ratio = (len(window) - len(set(window))) / len(window)
        if repeated_ratio > 0.5:
            return True
    return False


def _sanitize_prompt(prompt: str | None, *, max_tokens: int) -> str:
    if not prompt:
        return ""

    cleaned = _CARRYOVER_WHITESPACE.sub(" ", prompt.strip())
    cleaned = _CARRYOVER_TRAILING_NOISE.sub("", cleaned).strip()
    if not cleaned:
        return ""

    tokens = cleaned.split(" ")
    token_limit = min(len(tokens), max(0, max_tokens))
    tokens = tokens[-token_limit:] if token_limit else []
    return " ".join(tokens)
