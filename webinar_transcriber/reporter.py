"""Reporter interfaces shared by interactive and no-op implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pathlib import Path

    from webinar_transcriber.processor import ProcessArtifacts


@dataclass
class StageEvent:
    """Structured representation of a stage transition."""

    stage_key: str
    label: str
    started_at: float


class StageReporter(Protocol):
    """Reporter protocol shared by interactive and no-op implementations."""

    def begin_run(self, input_path: Path) -> None:
        """Record the start of a processing run."""

    def stage_started(self, stage_key: str, label: str) -> None:
        """Record that a stage has started."""

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
        """Record that a determinate stage has started."""

    def progress_advanced(
        self, stage_key: str, *, advance: float = 1.0, detail: str | None = None
    ) -> None:
        """Record that a determinate stage has advanced."""

    def stage_finished(self, stage_key: str, label: str, *, detail: str | None = None) -> None:
        """Record that a stage has finished."""

    def warn(self, message: str) -> None:
        """Record a warning."""

    def interrupted(self) -> None:
        """Record an interrupted run."""

    def reset_active_display(self) -> None:
        """Clear any in-progress spinner or progress bar without printing output."""

    def complete_run(self, artifacts: ProcessArtifacts) -> None:
        """Record run completion."""


class NullStageReporter:
    """No-op reporter used by tests and non-interactive code paths."""

    def begin_run(self, input_path: Path) -> None:
        """Record the start of a processing run."""

    def stage_started(self, stage_key: str, label: str) -> None:
        """Record that a stage has started."""

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
        """Record that a determinate stage has started."""

    def progress_advanced(
        self, stage_key: str, *, advance: float = 1.0, detail: str | None = None
    ) -> None:
        """Record that a determinate stage has advanced."""

    def stage_finished(self, stage_key: str, label: str, *, detail: str | None = None) -> None:
        """Record that a stage has finished."""

    def warn(self, message: str) -> None:
        """Record a warning."""

    def interrupted(self) -> None:
        """Record an interrupted run."""

    def reset_active_display(self) -> None:
        """Clear any in-progress spinner or progress bar without printing output."""

    def complete_run(self, artifacts: ProcessArtifacts) -> None:
        """Record run completion."""
