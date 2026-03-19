"""ASR adapter built around faster-whisper."""

from pathlib import Path
from typing import Protocol

from faster_whisper import WhisperModel

from webinar_transcriber.models import TranscriptionResult, TranscriptSegment, TranscriptWord


class Transcriber(Protocol):
    """Protocol for components that convert audio to transcript text."""

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        """Return a normalized transcription for the provided audio file."""


class WhisperTranscriber:
    """Default ASR implementation using faster-whisper."""

    def __init__(
        self,
        model_name: str = "tiny",
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

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
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

        return TranscriptionResult(
            detected_language=getattr(info, "language", None),
            segments=normalized_segments,
        )


def _default_compute_type(device: str) -> str:
    """Choose a less noisy default compute type for the current device."""
    return "float32" if device in {"auto", "cpu"} else "default"
