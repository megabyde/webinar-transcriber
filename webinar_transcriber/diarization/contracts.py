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
    def system_info(self) -> str | None:
        """Return runtime system information when available."""

    def prepare(self, *, speaker_count: int | None) -> None:
        """Prepare models for one diarization run."""

    def diarize(
        self, samples: np.ndarray, *, progress_callback: Callable[[int, int], None] | None = None
    ) -> list[SpeakerTurn]:
        """Return speaker turns for normalized audio samples."""
