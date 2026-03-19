"""Rich-based terminal progress reporting."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from pathlib import Path

    from rich.status import Status

    from webinar_transcriber.processor import ProcessArtifacts


@dataclass
class StageEvent:
    """Structured representation of a stage transition."""

    stage_key: str
    label: str
    started_at: float


class NullStageReporter:
    """No-op reporter used by tests and non-interactive code paths."""

    def begin_run(self, input_path: Path, *, ocr_enabled: bool, output_format: str) -> None:
        """Record the start of a processing run."""

    def stage_started(self, stage_key: str, label: str) -> None:
        """Record that a stage has started."""

    def stage_finished(self, stage_key: str, label: str, *, detail: str | None = None) -> None:
        """Record that a stage has finished."""

    def warn(self, message: str) -> None:
        """Record a warning."""

    def complete_run(self, artifacts: ProcessArtifacts) -> None:
        """Record run completion."""


class RichStageReporter(NullStageReporter):
    """Terminal reporter using Rich status spinners and summaries."""

    def __init__(self, console: Console | None = None) -> None:
        self._console = console or Console(stderr=True)
        self._active_status: Status | None = None
        self._active_event: StageEvent | None = None
        self._warnings: list[str] = []
        self._stage_count = 0

    def begin_run(self, input_path: Path, *, ocr_enabled: bool, output_format: str) -> None:
        mode_label = "with OCR" if ocr_enabled else "without OCR"
        self._console.print(
            f"[bold cyan]Starting[/] {input_path.name} ({mode_label}, format={output_format})"
        )

    def stage_started(self, stage_key: str, label: str) -> None:
        self._stage_count += 1
        self._active_event = StageEvent(
            stage_key=stage_key,
            label=label,
            started_at=perf_counter(),
        )
        self._active_status = self._console.status(
            f"[bold blue][{self._stage_count}][/bold blue] {label}",
            spinner="dots",
        )
        self._active_status.start()

    def stage_finished(self, stage_key: str, label: str, *, detail: str | None = None) -> None:
        if self._active_status is not None:
            self._active_status.stop()
        elapsed = 0.0
        if self._active_event is not None and self._active_event.stage_key == stage_key:
            elapsed = perf_counter() - self._active_event.started_at

        detail_suffix = f" - {detail}" if detail else ""
        self._console.print(f"[green]\u2713[/] {label}{detail_suffix} [dim]({elapsed:.2f}s)[/]")
        self._active_status = None
        self._active_event = None

    def warn(self, message: str) -> None:
        if self._active_status is not None:
            self._active_status.stop()
            self._active_status = None
            self._active_event = None
        self._warnings.append(message)
        self._console.print(f"[yellow]![/] {message}")

    def complete_run(self, artifacts: ProcessArtifacts) -> None:
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_row("Run directory", str(artifacts.layout.run_dir))
        table.add_row("Language", artifacts.report.detected_language or "unknown")
        table.add_row("Sections", str(len(artifacts.report.sections)))
        table.add_row(
            "Warnings",
            str(len(artifacts.report.warnings)),
        )
        self._console.print()
        self._console.print(table)
