"""ASR adapter built on top of the whisper.cpp C API."""

from __future__ import annotations

import importlib
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Self

from webinar_transcriber.asr.carryover import build_prompt_carryover
from webinar_transcriber.asr.config import (
    DEFAULT_WHISPER_CPP_MODEL_EXAMPLE,
    DEFAULT_WHISPER_CPP_MODEL_FILENAME,
    DEFAULT_WHISPER_CPP_MODEL_REPO,
    PromptCarryoverSettings,
    WhisperDecodeSettings,
)
from webinar_transcriber.whispercpp import (
    GPU_BACKEND_PATTERN,
    WhisperCppLibrary,
    WhisperCppRuntimeDetails,
    WhisperCppSession,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import TracebackType

    from webinar_transcriber.models import DecodedWindow, InferenceWindow


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

    def __enter__(self) -> Self:
        return self

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

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        self.close()

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
