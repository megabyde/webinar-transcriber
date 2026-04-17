"""Reporter interfaces shared by interactive and no-op implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from webinar_transcriber.processor import ProcessArtifacts


@dataclass
class StageEvent:
    """Structured representation of a stage transition."""

    stage_key: str
    label: str
    started_at: float


class NullStageReporter:
    """No-op reporter used by tests and non-interactive code paths."""

    def begin_run(self, input_path: Path) -> None:
        """Ignore the start of a processing run."""

    def stage_started(self, stage_key: str, label: str) -> None:
        """Ignore a stage-start event."""

    def progress_started(
        self,
        stage_key: str,
        label: str,
        *,
        total: float,
        count_label: str | None = None,
        count_multiplier: float = 1.0,
        rate_label: str | None = None,
        rate_multiplier: float = 1.0,
        detail: str | None = None,
    ) -> None:
        """Ignore a determinate stage-start event."""

    def progress_advanced(
        self, stage_key: str, *, advance: float = 1.0, detail: str | None = None
    ) -> None:
        """Ignore a determinate stage-progress update."""

    def stage_finished(self, stage_key: str, label: str, *, detail: str | None = None) -> None:
        """Ignore a stage-finished event."""

    def warn(self, message: str) -> None:
        """Ignore a warning message."""

    def interrupted(self) -> None:
        """Ignore an interrupted-run event."""

    def reset_active_display(self) -> None:
        """Ignore a request to clear active terminal output."""

    def complete_run(self, artifacts: ProcessArtifacts) -> None:
        """Ignore run completion."""
