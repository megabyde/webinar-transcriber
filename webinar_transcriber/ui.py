"""Rich-based terminal progress reporting."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from webinar_transcriber.reporter import BaseStageReporter

if TYPE_CHECKING:
    from pathlib import Path

    from rich.status import Status

    from webinar_transcriber.processor import ProcessArtifacts


@dataclass(frozen=True)
class _ActiveStage:
    key: str
    label: str
    started_at: float


@dataclass(frozen=True)
class _ActiveStatusDisplay:
    status: Status


@dataclass(frozen=True)
class _ActiveProgressDisplay:
    progress: Progress
    task_id: TaskID


class RichStageReporter(BaseStageReporter):
    """Terminal reporter using Rich status spinners and summaries."""

    def __init__(self, console: Console | None = None) -> None:
        """Initialize the Rich-backed terminal reporter."""
        self._console = console or Console()
        self._active_display: _ActiveStatusDisplay | _ActiveProgressDisplay | None = None
        self._active_stage: _ActiveStage | None = None
        self._stage_count = 0

    def begin_run(self, input_path: Path) -> None:
        """Render the start of a processing run."""
        self._console.print(f"[bold cyan]Starting[/] {input_path.name}")

    def stage_started(self, stage_key: str, label: str) -> None:
        """Start an indeterminate stage spinner."""
        self._stop_active_display()
        self._stage_count += 1
        self._set_active_stage(stage_key, label)
        status = self._console.status(
            f"[bold blue][{self._stage_count}][/bold blue] {label}", spinner="dots"
        )
        self._active_display = _ActiveStatusDisplay(status=status)
        status.start()

    def progress_started(
        self,
        stage_key: str,
        label: str,
        *,
        total: float,
        count_label: str | None = None,
        rate_label: str | None = None,
        detail: str | None = None,
    ) -> None:
        """Start a determinate progress display for a stage."""
        self._stop_active_display()
        self._stage_count += 1
        self._set_active_stage(stage_key, label)
        progress = Progress(
            SpinnerColumn(),
            TextColumn(f"[bold blue][{self._stage_count}][/bold blue] {label}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("{task.fields[count_text]}", style="progress.data.speed"),
            TextColumn("{task.fields[rate_text]}", style="progress.data.speed"),
            TextColumn("{task.fields[detail_text]}", style="dim"),
            TextColumn("ETA"),
            TimeRemainingColumn(compact=True),
            TimeElapsedColumn(),
            console=self._console,
            transient=True,
        )
        progress.start()
        task_id = progress.add_task(
            label,
            total=max(total, 1.0),
            count_label=count_label,
            count_text=_count_text(
                completed=0.0,
                total=max(total, 1.0),
                count_label=count_label,
            ),
            rate_label=rate_label,
            rate_text="",
            detail_text=detail or "",
        )
        self._active_display = _ActiveProgressDisplay(progress=progress, task_id=task_id)

    def progress_advanced(
        self, stage_key: str, *, advance: float = 1.0, detail: str | None = None
    ) -> None:
        """Advance the active determinate progress display."""
        active_stage = self._active_stage
        if active_stage is None or active_stage.key != stage_key:
            return
        active_display = self._active_display
        if not isinstance(active_display, _ActiveProgressDisplay):
            return
        active_display.progress.update(
            active_display.task_id, advance=advance, detail_text=detail or ""
        )
        task = active_display.progress.tasks[active_display.task_id]
        active_display.progress.update(
            active_display.task_id,
            count_text=_count_text(
                completed=task.completed,
                total=task.total,
                count_label=task.fields.get("count_label"),
            ),
            rate_text=_rate_text(
                completed=task.completed,
                elapsed_sec=perf_counter() - active_stage.started_at,
                rate_label=task.fields.get("rate_label"),
            ),
        )

    def stage_finished(self, stage_key: str, label: str, *, detail: str | None = None) -> None:
        """Stop the active display and render a finished-stage line."""
        self._stop_active_display()
        elapsed = 0.0
        if self._active_stage is not None and self._active_stage.key == stage_key:
            elapsed = perf_counter() - self._active_stage.started_at

        detail_suffix = f" - {detail}" if detail else ""
        self._console.print(f"[green]\u2713[/] {label}{detail_suffix} [dim]({elapsed:.2f}s)[/]")
        self._clear_active_stage()

    def warn(self, message: str) -> None:
        """Render a warning message."""
        self._stop_active_display()
        self._console.print(f"[yellow]![/] {message}")

    def interrupted(self) -> None:
        """Render an interrupted-run message."""
        stage_suffix = ""
        if self._active_stage is not None:
            stage_suffix = f" during {self._active_stage.label.lower()}"
        self._clear_active_display()
        self._console.print(f"[red]\u2717[/] Interrupted{stage_suffix}.")

    def reset_active_display(self) -> None:
        """Clear any active spinner or progress bar."""
        self._clear_active_display()

    def complete_run(self, artifacts: ProcessArtifacts) -> None:
        """Render the completion summary panel."""
        table = Table.grid(padding=(0, 2))
        table.add_column(style="dim")
        table.add_column()
        warning_count = len(artifacts.report.warnings)
        warning_text = Text(str(warning_count), style="yellow" if warning_count else "green")
        table.add_row("Run directory", Text(str(artifacts.layout.run_dir), style="cyan"))
        table.add_row("Diagnostics", Text(str(artifacts.layout.diagnostics_path), style="cyan"))
        table.add_row("Language", Text(artifacts.report.detected_language or "unknown"))
        table.add_row("Sections", Text(str(len(artifacts.report.sections))))
        if processing_detail := _processing_detail(artifacts):
            table.add_row("Processing", Text(processing_detail))
        table.add_row("Warnings", warning_text)
        self._console.print()
        self._console.print(
            Panel.fit(table, title="[bold green]Completed[/]", border_style="green")
        )

    def _stop_active_display(self) -> None:
        active_display = self._active_display
        if isinstance(active_display, _ActiveStatusDisplay):
            active_display.status.stop()
        elif isinstance(active_display, _ActiveProgressDisplay):
            active_display.progress.stop()
        self._active_display = None

    def _clear_active_display(self) -> None:
        self._stop_active_display()
        self._clear_active_stage()

    def _set_active_stage(self, stage_key: str, label: str) -> None:
        self._active_stage = _ActiveStage(key=stage_key, label=label, started_at=perf_counter())

    def _clear_active_stage(self) -> None:
        self._active_stage = None


def _count_text(*, completed: float, total: float | None, count_label: object) -> str:
    if total is None or not count_label:
        return ""

    completed_count = int(completed)
    total_count = int(total)
    sep = "" if count_label in {"s", "ms", "%"} else " "
    return f"{completed_count}/{total_count}{sep}{count_label}"


def _rate_text(*, completed: float, elapsed_sec: float, rate_label: object) -> str:
    if not rate_label or completed <= 0:
        return ""
    if elapsed_sec <= 0:
        return ""

    rate = completed / elapsed_sec
    display_rate = f"{rate:.0f}" if rate >= 100 else f"{rate:.1f}"
    return f"{display_rate} {rate_label}"


def _processing_detail(artifacts: ProcessArtifacts) -> str | None:
    total_sec = sum(artifacts.diagnostics.stage_durations_sec.values())
    if total_sec <= 0:  # pragma: no cover - defensive fallback for malformed diagnostics
        return None

    media_duration_sec = artifacts.media_asset.duration_sec
    if media_duration_sec <= 0:  # pragma: no cover - media probing guarantees duration
        return _duration_text(total_sec)

    rtf = format(round(media_duration_sec / total_sec, 2), "g")
    return f"{_duration_text(total_sec)} | RTF {rtf}x"


def _duration_text(duration_sec: float) -> str:
    total_seconds = round(duration_sec)
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"
