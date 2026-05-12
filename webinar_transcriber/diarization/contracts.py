"""Speaker-diarization contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

    from webinar_transcriber.models import SpeakerTurn


class DiarizationProcessingError(RuntimeError):
    """Raised when local speaker diarization cannot complete."""


class Diarizer(Protocol):
    """Protocol implemented by local speaker-diarization backends."""

    @property
    def model_name(self) -> str:
        """Return a human-facing model name."""

    @property
    def system_info(self) -> str | None:
        """Return runtime system information when available."""

    def diarize(
        self,
        samples: np.ndarray,
        sample_rate: int,
        *,
        speaker_count: int | None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[SpeakerTurn]:
        """Return speaker turns for normalized audio samples."""
