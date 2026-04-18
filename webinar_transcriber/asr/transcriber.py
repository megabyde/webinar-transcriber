"""ASR adapter built on top of pywhispercpp."""

from __future__ import annotations

import importlib
import re
from contextlib import contextmanager, suppress
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, Self, cast

import numpy as np

from webinar_transcriber.asr.carryover import build_prompt_carryover
from webinar_transcriber.asr.config import (
    DEFAULT_WHISPER_CPP_MODEL_EXAMPLE,
    DEFAULT_WHISPER_CPP_MODEL_FILENAME,
    PromptCarryoverSettings,
    WhisperDecodeSettings,
)
from webinar_transcriber.models import DecodedWindow, TranscriptSegment
from webinar_transcriber.normalized_audio import sample_index_for_time

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from types import TracebackType

    from webinar_transcriber.models import InferenceWindow

GPU_BACKEND_PATTERN = re.compile(r"(?i)\b(metal|mtl|cuda)\b[^|]*?(?:=|:)\s*(?:1|true)")
_WHISPER_TICKS_PER_SECOND = 100.0


class _PyWhisperSegment(Protocol):
    t0: int | float
    t1: int | float
    text: str


class _PyWhisperModel(Protocol):
    def system_info(self) -> str: ...

    def transcribe(
        self, audio_samples: np.ndarray, **kwargs: str | None
    ) -> list[_PyWhisperSegment]: ...

    def auto_detect_language(
        self, audio_samples: np.ndarray, *, n_threads: int
    ) -> tuple[tuple[str, float], dict[str, float]]: ...


def _missing_model_error_message(model_path: Path | None) -> str:
    if model_path is not None:
        model_path_text = str(model_path)
    else:
        model_path_text = "--asr-model path was not initialized for the explicit whisper.cpp model"
    example_path = DEFAULT_WHISPER_CPP_MODEL_EXAMPLE
    location_hint = f"Download a whisper.cpp model there, or use --asr-model {example_path}."
    return (
        f"whisper.cpp model file does not exist: {model_path_text}. "
        f"{location_hint} See README.md for model setup details."
    )


def _model_prepare_error_message(model_name: str) -> str:
    return (
        f"Could not prepare whisper.cpp model '{model_name}'. "
        "pywhispercpp accepts model identifiers such as 'large-v3-turbo' or local ggml model "
        f"paths such as {DEFAULT_WHISPER_CPP_MODEL_EXAMPLE}."
    )


def _model_cls() -> type[_PyWhisperModel]:
    return importlib.import_module("pywhispercpp.model").Model


def _looks_like_model_path(model_name: str) -> bool:
    model_path = Path(model_name).expanduser()
    return model_path.suffix == ".bin" or model_path.parent != Path()


@contextmanager
def _suppress_pywhispercpp_download_progress() -> Iterator[None]:
    pywhispercpp_utils = cast("Any", importlib.import_module("pywhispercpp.utils"))
    original_tqdm = pywhispercpp_utils.tqdm
    pywhispercpp_utils.tqdm = lambda *args, **kwargs: original_tqdm(*args, disable=True, **kwargs)
    try:
        yield
    finally:
        pywhispercpp_utils.tqdm = original_tqdm


class ASRProcessingError(RuntimeError):
    """Raised when the whisper.cpp ASR adapter cannot prepare or run."""


class WhisperCppTranscriber:
    """ASR implementation using pywhispercpp."""

    def __init__(
        self,
        model_name: str | None = None,
        *,
        threads: int = 4,
        carryover_settings: PromptCarryoverSettings | None = None,
        log_path: Path | None = None,
    ) -> None:
        """Initialize the whisper.cpp transcriber wrapper."""
        self._configured_model_path = Path(model_name) if model_name else None
        self._uses_default_model_name = model_name is None
        self._model_name = model_name or DEFAULT_WHISPER_CPP_MODEL_FILENAME
        self._threads = max(1, threads)
        self._decode_settings = WhisperDecodeSettings(
            carryover=carryover_settings or PromptCarryoverSettings()
        )
        self._log_path = log_path
        self._model: _PyWhisperModel | None = None

    @property
    def model_name(self) -> str:
        """Return the configured model identifier or explicit local path."""
        return self._model_name

    @property
    def device_name(self) -> str:
        """Return the detected runtime backend name."""
        if (system_info := self.system_info) is None:
            return "auto"
        return _device_name_from_system_info(system_info)

    @property
    def threads(self) -> int:
        """Return the configured whisper.cpp thread count."""
        return self._threads

    @property
    def system_info(self) -> str | None:
        """Return the whisper.cpp runtime system info string when available."""
        if self._model is None:
            return None
        return self._model.system_info()

    @property
    def library_path(self) -> str | None:
        """Return the loaded whisper.cpp shared library path when available."""
        return None

    def __enter__(self) -> Self:
        """Return the transcriber as a context manager."""
        return self

    def set_log_path(self, log_path: Path) -> None:
        """Set the whisper.cpp native log destination for future model loads."""
        self._log_path = log_path

    def prepare_model(self) -> None:
        """Resolve the model and initialize one pywhispercpp model instance.

        Raises:
            ASRProcessingError: If model resolution or model creation fails.
        """
        self._model_name = self._resolve_model_name()
        self.close()
        model_kwargs: dict[str, object] = {
            "n_threads": self._threads,
            "print_realtime": False,
            "print_progress": False,
            "no_context": True,
            "split_on_word": True,
            "entropy_thold": self._decode_settings.entropy_thold,
            "logprob_thold": self._decode_settings.logprob_thold,
            "no_speech_thold": self._decode_settings.no_speech_thold,
        }
        if self._log_path is not None:
            model_kwargs["redirect_whispercpp_logs_to"] = str(self._log_path)
        try:
            with _suppress_pywhispercpp_download_progress():
                model = _model_cls()(self._model_name, **model_kwargs)
        except Exception as error:
            raise ASRProcessingError(_model_prepare_error_message(self._model_name)) from error
        if getattr(model, "_ctx", True) is None:
            raise ASRProcessingError(_model_prepare_error_message(self._model_name))
        self._model = model

    def transcribe_inference_windows(
        self,
        audio_samples: np.ndarray,
        windows: list[InferenceWindow],
        *,
        progress_callback: Callable[[float, int], None] | None = None,
    ) -> list[DecodedWindow]:
        """Decode ordered inference windows into transcript segments.

        Returns:
            list[DecodedWindow]: The decoded windows in deterministic order.
        """
        model = self._ensure_model()
        ordered_windows = sorted(
            windows,
            key=lambda item: (item.start_sec, item.end_sec, item.region_index, item.window_id),
        )
        language_hint: str | None = None
        carryover_prompt: str | None = None
        decoded_windows: list[DecodedWindow] = []
        decoded_segment_count = 0

        for window in ordered_windows:
            decoded_window = self._transcribe_window(
                model,
                audio_samples,
                window,
                prompt=carryover_prompt,
                language_hint=language_hint,
            )
            decoded_windows.append(replace(decoded_window, input_prompt=carryover_prompt))
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
        """Release any active pywhispercpp model resources."""
        self._model = None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the transcriber when leaving a context manager block."""
        self.close()

    def _ensure_model(self) -> _PyWhisperModel:
        if self._model is None:
            self.prepare_model()
        if self._model is None:
            raise ASRProcessingError(
                "pywhispercpp model was not initialized during model preparation."
            )
        return self._model

    def _resolve_model_name(self) -> str:
        if not self._uses_default_model_name:
            if not _looks_like_model_path(self._model_name):
                return self._model_name
            configured_model_path = None
            if self._configured_model_path is not None:
                configured_model_path = self._configured_model_path.expanduser()
            if configured_model_path is None or not configured_model_path.exists():
                raise ASRProcessingError(_missing_model_error_message(configured_model_path))
            return str(configured_model_path)
        return DEFAULT_WHISPER_CPP_MODEL_FILENAME

    def _transcribe_window(
        self,
        model: _PyWhisperModel,
        audio_samples: np.ndarray,
        window: InferenceWindow,
        *,
        prompt: str | None,
        language_hint: str | None,
    ) -> DecodedWindow:
        start_index = sample_index_for_time(window.start_sec)
        end_index = min(len(audio_samples), sample_index_for_time(window.end_sec))
        window_samples = np.ascontiguousarray(
            audio_samples[start_index:end_index], dtype=np.float32
        )
        if window_samples.size == 0:
            return DecodedWindow(window=window, language=language_hint)

        transcribe_kwargs: dict[str, str] = {}
        if prompt is not None:
            transcribe_kwargs["initial_prompt"] = prompt
        if language_hint is not None:
            transcribe_kwargs["language"] = language_hint
        try:
            raw_segments = model.transcribe(window_samples, **transcribe_kwargs)
        except Exception as error:
            raise ASRProcessingError(
                f"whisper.cpp inference failed for {window.window_id}."
            ) from error

        detected_language = language_hint
        if detected_language is None:
            with suppress(Exception):
                detected_language = model.auto_detect_language(
                    window_samples, n_threads=self._threads
                )[0][0]

        segments = [
            TranscriptSegment(
                id=f"{window.window_id}-segment-{segment_index + 1}",
                text=str(raw_segment.text).strip(),
                start_sec=max(
                    window.start_sec,
                    window.start_sec + (float(raw_segment.t0) / _WHISPER_TICKS_PER_SECOND),
                ),
                end_sec=min(
                    window.end_sec,
                    max(
                        window.start_sec + (float(raw_segment.t0) / _WHISPER_TICKS_PER_SECOND),
                        window.start_sec + (float(raw_segment.t1) / _WHISPER_TICKS_PER_SECOND),
                    ),
                ),
            )
            for segment_index, raw_segment in enumerate(raw_segments)
        ]
        return DecodedWindow(
            window=window,
            text=" ".join(segment.text for segment in segments if segment.text).strip(),
            segments=segments,
            fallback_used=False,
            language=detected_language,
        )

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
