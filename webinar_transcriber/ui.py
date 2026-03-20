"""Rich-based terminal progress reporting."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
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

    def progress_started(self, stage_key: str, label: str, *, total: int) -> None:
        """Record that a determinate stage has started."""

    def progress_advanced(self, stage_key: str, *, advance: int = 1) -> None:
        """Record that a determinate stage has advanced."""

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
        self._active_progress: Progress | None = None
        self._active_task_id: TaskID | None = None
        self._active_event: StageEvent | None = None
        self._warnings: list[str] = []
        self._stage_count = 0

    def begin_run(self, input_path: Path, *, ocr_enabled: bool, output_format: str) -> None:
        mode_label = "with OCR" if ocr_enabled else "without OCR"
        self._console.print(
            f"[bold cyan]Starting[/] {input_path.name} ({mode_label}, format={output_format})"
        )

    def stage_started(self, stage_key: str, label: str) -> None:
        self._stop_active_display()
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

    def progress_started(self, stage_key: str, label: str, *, total: int) -> None:
        self._stop_active_display()
        self._stage_count += 1
        self._active_event = StageEvent(
            stage_key=stage_key,
            label=label,
            started_at=perf_counter(),
        )
        self._active_progress = Progress(
            SpinnerColumn(),
            TextColumn(f"[bold blue][{self._stage_count}][/bold blue] {label}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self._console,
            transient=True,
        )
        self._active_progress.start()
        self._active_task_id = self._active_progress.add_task(label, total=max(total, 1))

    def progress_advanced(self, stage_key: str, *, advance: int = 1) -> None:
        if self._active_event is None or self._active_event.stage_key != stage_key:
            return
        if self._active_progress is None or self._active_task_id is None:
            return
        self._active_progress.update(self._active_task_id, advance=advance)

    def stage_finished(self, stage_key: str, label: str, *, detail: str | None = None) -> None:
        self._stop_active_display()
        elapsed = 0.0
        if self._active_event is not None and self._active_event.stage_key == stage_key:
            elapsed = perf_counter() - self._active_event.started_at

        detail_suffix = f" - {detail}" if detail else ""
        self._console.print(f"[green]\u2713[/] {label}{detail_suffix} [dim]({elapsed:.2f}s)[/]")
        self._active_status = None
        self._active_event = None

    def warn(self, message: str) -> None:
        self._stop_active_display()
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

    def _stop_active_display(self) -> None:
        if self._active_status is not None:
            self._active_status.stop()
            self._active_status = None
        if self._active_progress is not None:
            self._active_progress.stop()
            self._active_progress = None
            self._active_task_id = None
