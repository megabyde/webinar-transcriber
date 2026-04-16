"""ASR adapter built on top of the whisper.cpp C API."""

from __future__ import annotations

import importlib
import re
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

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import TracebackType

    import numpy as np

    from webinar_transcriber.models import DecodedWindow, InferenceWindow
    from webinar_transcriber.whispercpp import (
        WhisperCppLibrary,
        WhisperCppRuntimeDetails,
        WhisperCppSession,
    )

GPU_BACKEND_PATTERN = re.compile(
    r"(?i)\b(metal|mtl|cuda|vulkan|coreml)\b[^|]*?(?:=|:)\s*(?:1|true)"
)


def _missing_model_error_message(model_path: Path | None) -> str:
    if model_path is not None:
        model_path_text = str(model_path)
    else:
        model_path_text = "--asr-model path was not initialized for the explicit whisper.cpp model"
    example_path = DEFAULT_WHISPER_CPP_MODEL_EXAMPLE
    location_hint = f"Download a whisper.cpp model there, or use --asr-model {example_path}."
    return (
        f"whisper.cpp model file does not exist: {model_path_text}. "
        f"{location_hint} See README.md for model download instructions."
    )


def _default_model_download_error_message() -> str:
    return (
        "Could not resolve the default whisper.cpp model from the Hugging Face cache. "
        "The app uses Hugging Face cache defaults for the built-in model. "
        f"To use a manual file instead, pass --asr-model {DEFAULT_WHISPER_CPP_MODEL_EXAMPLE}. "
        "See README.md for model download instructions."
    )


class ASRProcessingError(RuntimeError):
    """Raised when the whisper.cpp ASR adapter cannot prepare or run."""


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
        """Initialize the whisper.cpp transcriber wrapper."""
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
        """Return the resolved model path or default Hugging Face model reference."""
        if self._model_path is not None:
            return str(self._model_path)
        return f"{DEFAULT_WHISPER_CPP_MODEL_REPO}/{DEFAULT_WHISPER_CPP_MODEL_FILENAME}"

    @property
    def device_name(self) -> str:
        """Return the detected runtime backend name."""
        if self._runtime_details is None:
            return "auto"
        return _device_name_from_system_info(self._runtime_details.system_info)

    @property
    def threads(self) -> int:
        """Return the configured whisper.cpp thread count."""
        return self._threads

    @property
    def system_info(self) -> str | None:
        """Return the whisper.cpp runtime system info string when available."""
        if self._runtime_details is None:
            return None
        return self._runtime_details.system_info

    @property
    def library_path(self) -> str | None:
        """Return the loaded whisper.cpp shared library path when available."""
        if self._runtime_details is None:
            return None
        return self._runtime_details.library_path

    def set_log_path(self, log_path: Path) -> None:
        """Set the whisper.cpp native log sink path for future sessions."""
        self._log_path = log_path

    def __enter__(self) -> Self:
        """Return the transcriber as a context manager."""
        return self

    def prepare_model(self) -> None:
        """Resolve the model and initialize a fresh whisper.cpp session."""
        self._model_path = self._resolve_model_path()
        self.close()
        if self._model_path is None:
            raise ASRProcessingError("Model path resolution returned no whisper.cpp model path.")
        whispercpp_library_cls = importlib.import_module(
            "webinar_transcriber.whispercpp"
        ).WhisperCppLibrary
        runtime = whispercpp_library_cls(self._library_path, log_path=self._log_path)
        session = runtime.create_session(self._model_path)
        self._runtime = runtime
        self._session = session
        self._runtime_details = session.runtime_details

    def transcribe_inference_windows(
        self,
        audio_samples: np.ndarray,
        windows: list[InferenceWindow],
        *,
        progress_callback: Callable[[float, int], None] | None = None,
    ) -> list[DecodedWindow]:
        """Decode ordered inference windows into transcript segments."""
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
                decoded_window, settings=self._decode_settings.carryover
            )
            language_hint = language_hint or decoded_window.language
            carryover_prompt = next_carryover
            if progress_callback is not None:
                progress_callback(window.end_sec, decoded_segment_count)
        return decoded_windows

    def close(self) -> None:
        """Release any active whisper.cpp session resources."""
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
        """Close the transcriber when leaving a context manager block."""
        self.close()

    def _ensure_session(self) -> WhisperCppSession:
        if self._session is None:
            self.prepare_model()
        if self._session is None:
            raise ASRProcessingError(
                "whisper.cpp session was not initialized during model preparation."
            )
        return self._session

    def _resolve_model_path(self) -> Path:
        if not self._uses_default_model_path:
            configured_model_path = self._configured_model_path
            if configured_model_path is None or not configured_model_path.exists():
                raise ASRProcessingError(_missing_model_error_message(configured_model_path))
            return configured_model_path

        try:
            return _download_default_whisper_cpp_model()
        except ASRProcessingError as ex:
            raise ASRProcessingError(f"{_default_model_download_error_message()} {ex}") from ex

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown cleanup
        """Best-effort cleanup during interpreter shutdown."""
        with suppress(Exception):
            self.close()


def _device_name_from_system_info(system_info: str) -> str:
    match = GPU_BACKEND_PATTERN.search(system_info)
    if match is not None:
        backend_name = match.group(1).lower()
        return "metal" if backend_name == "mtl" else backend_name
    return "cpu"


def _download_default_whisper_cpp_model() -> Path:
    hf_hub_download = importlib.import_module("huggingface_hub").hf_hub_download
    try:
        downloaded_path = hf_hub_download(
            repo_id=DEFAULT_WHISPER_CPP_MODEL_REPO, filename=DEFAULT_WHISPER_CPP_MODEL_FILENAME
        )
    except Exception as error:  # pragma: no cover - network/backend specific
        raise ASRProcessingError(
            "Automatic download of the default whisper.cpp model from Hugging Face failed."
        ) from error

    return Path(downloaded_path)
