"""ASR adapters for faster-whisper and MLX Whisper."""

from __future__ import annotations

import importlib
import importlib.util
import platform
from typing import TYPE_CHECKING, Any, Protocol

from faster_whisper import WhisperModel

from webinar_transcriber.models import TranscriptionResult, TranscriptSegment, TranscriptWord

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

DEFAULT_FASTER_WHISPER_MODEL = "small"
DEFAULT_MLX_WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"


class Transcriber(Protocol):
    """Protocol for components that convert audio to transcript text."""

    @property
    def supports_live_progress(self) -> bool:
        """Whether transcription emits incremental progress during inference."""

    @property
    def uses_native_progress(self) -> bool:
        """Whether the backend renders its own user-visible progress output."""

    def prepare_model(self) -> None:
        """Warm or download model assets before transcription starts."""

    def transcribe(
        self,
        audio_path: Path,
        *,
        progress_callback: Callable[[float], None] | None = None,
    ) -> TranscriptionResult:
        """Return a normalized transcription for the provided audio file."""


class FasterWhisperTranscriber:
    """ASR implementation using faster-whisper."""

    def __init__(
        self,
        model_name: str = DEFAULT_FASTER_WHISPER_MODEL,
        *,
        device: str = "auto",
        compute_type: str | None = None,
    ) -> None:
        resolved_compute_type = compute_type or _default_compute_type(device)
        self._model = WhisperModel(
            model_name,
            device=device,
            compute_type=resolved_compute_type,
        )

    def transcribe(
        self,
        audio_path: Path,
        *,
        progress_callback: Callable[[float], None] | None = None,
    ) -> TranscriptionResult:
        segments, info = self._model.transcribe(
            str(audio_path),
            beam_size=5,
            vad_filter=True,
        )
        normalized_segments: list[TranscriptSegment] = []

        for index, segment in enumerate(segments, start=1):
            normalized_segments.append(
                TranscriptSegment(
                    id=f"segment-{index}",
                    text=segment.text.strip(),
                    start_sec=float(segment.start),
                    end_sec=float(segment.end),
                    words=[
                        TranscriptWord(
                            text=word.word.strip(),
                            start_sec=float(word.start),
                            end_sec=float(word.end),
                            confidence=float(word.probability),
                        )
                        for word in (segment.words or [])
                    ],
                )
            )
            if progress_callback is not None:
                progress_callback(float(segment.end))

        return TranscriptionResult(
            detected_language=getattr(info, "language", None),
            segments=normalized_segments,
        )

    @property
    def supports_live_progress(self) -> bool:
        """faster-whisper yields segments incrementally during inference."""
        return True

    @property
    def uses_native_progress(self) -> bool:
        """faster-whisper relies on the caller to render progress."""
        return False

    def prepare_model(self) -> None:
        """No-op because the model is prepared during construction."""


class MlxWhisperTranscriber:
    """ASR implementation using MLX Whisper on Apple Silicon."""

    def __init__(self, model_name: str = DEFAULT_MLX_WHISPER_MODEL) -> None:
        self._model_name = model_name

    def transcribe(
        self,
        audio_path: Path,
        *,
        progress_callback: Callable[[float], None] | None = None,
    ) -> TranscriptionResult:
        mlx_whisper = _import_mlx_whisper()
        result = mlx_whisper.transcribe(
            str(audio_path),
            path_or_hf_repo=self._model_name,
            verbose=False,
            word_timestamps=True,
        )
        normalized_segments: list[TranscriptSegment] = []

        for index, segment in enumerate(result.get("segments", []), start=1):
            end_sec = float(segment.get("end", 0.0))
            normalized_segments.append(
                TranscriptSegment(
                    id=f"segment-{index}",
                    text=str(segment.get("text", "")).strip(),
                    start_sec=float(segment.get("start", 0.0)),
                    end_sec=end_sec,
                    words=_normalize_mlx_words(segment.get("words", [])),
                )
            )

        return TranscriptionResult(
            detected_language=_mlx_detected_language(result),
            segments=normalized_segments,
        )

    @property
    def supports_live_progress(self) -> bool:
        """MLX returns the full transcription result at the end."""
        return False

    @property
    def uses_native_progress(self) -> bool:
        """mlx-whisper renders its own tqdm progress bar when verbose is False."""
        return True

    def prepare_model(self) -> None:
        """Preload model weights so downloads happen outside the transcript stage."""
        _preload_mlx_model(self._model_name)


class WhisperTranscriber:
    """Default ASR wrapper that selects the best local backend."""

    def __init__(
        self,
        model_name: str | None = None,
        *,
        backend: str = "auto",
        device: str = "auto",
        compute_type: str | None = None,
    ) -> None:
        self.backend = _resolve_backend_name(backend)
        if self.backend == "mlx":
            self._delegate: Transcriber = MlxWhisperTranscriber(
                model_name=model_name or DEFAULT_MLX_WHISPER_MODEL
            )
        else:
            self._delegate = FasterWhisperTranscriber(
                model_name=model_name or DEFAULT_FASTER_WHISPER_MODEL,
                device=device,
                compute_type=compute_type,
            )

    def transcribe(
        self,
        audio_path: Path,
        *,
        progress_callback: Callable[[float], None] | None = None,
    ) -> TranscriptionResult:
        return self._delegate.transcribe(audio_path, progress_callback=progress_callback)

    @property
    def supports_live_progress(self) -> bool:
        """Expose whether the selected backend can stream inference progress."""
        return self._delegate.supports_live_progress

    @property
    def uses_native_progress(self) -> bool:
        """Expose whether the selected backend renders its own progress output."""
        return self._delegate.uses_native_progress

    def prepare_model(self) -> None:
        """Prepare the selected backend before transcription starts."""
        self._delegate.prepare_model()


def _default_compute_type(device: str) -> str:
    """Choose a less noisy default compute type for the current device."""
    return "int8" if device in {"auto", "cpu"} else "default"


def _resolve_backend_name(backend: str) -> str:
    normalized_backend = backend.lower()
    if normalized_backend not in {"auto", "faster-whisper", "mlx"}:
        raise ValueError(f"Unsupported ASR backend: {backend}")
    if normalized_backend == "auto":
        return "mlx" if _mlx_backend_available() else "faster-whisper"
    if normalized_backend == "mlx" and not _mlx_backend_available():
        raise RuntimeError("MLX Whisper requires macOS arm64 with mlx-whisper installed.")
    return normalized_backend


def _mlx_backend_available() -> bool:
    return _is_apple_silicon_mac() and importlib.util.find_spec("mlx_whisper") is not None


def _is_apple_silicon_mac() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _import_mlx_whisper() -> Any:
    return importlib.import_module("mlx_whisper")


def _preload_mlx_model(model_name: str) -> None:
    transcribe_module = importlib.import_module("mlx_whisper.transcribe")
    mlx_core = importlib.import_module("mlx.core")
    transcribe_module.ModelHolder.get_model(model_name, mlx_core.float16)


def _mlx_detected_language(result: dict[str, Any]) -> str | None:
    detected_language = result.get("language")
    if isinstance(detected_language, str):
        return detected_language
    return None


def _normalize_mlx_words(words: list[dict[str, Any]]) -> list[TranscriptWord]:
    normalized_words: list[TranscriptWord] = []
    for word in words:
        raw_text = str(word.get("word", word.get("text", ""))).strip()
        if not raw_text:
            continue
        confidence = word.get("probability", word.get("confidence"))
        normalized_words.append(
            TranscriptWord(
                text=raw_text,
                start_sec=float(word.get("start", 0.0)),
                end_sec=float(word.get("end", 0.0)),
                confidence=float(confidence) if confidence is not None else None,
            )
        )
    return normalized_words
