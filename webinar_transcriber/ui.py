"""Terminal progress reporting."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from webinar_transcriber.export.formatting import format_duration

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from webinar_transcriber.processor import ProcessArtifacts


@dataclass
class StageHandle:
    """State for one in-progress stage."""

    label: str
    started_at: float
    detail: str | None = None
    completed: float = 0.0
    _progress: Progress | None = field(default=None, repr=False)
    _task_id: TaskID | None = field(default=None, repr=False)

    def elapsed_sec(self) -> float:
        """Return seconds since the stage started."""
        return perf_counter() - self.started_at

    def update(
        self,
        *,
        advance: float | None = None,
        completed: float | None = None,
        detail: str | None = None,
    ) -> None:
        """Advance the stage and/or update its detail string."""
        if completed is not None:
            advance = max(0.0, completed - self.completed)
        applied_advance = advance if advance is not None and advance > 0 else 0.0
        if applied_advance > 0:
            self.completed += applied_advance
        if detail is not None:
            self.detail = detail
        if self._progress is not None and self._task_id is not None:
            description = f"{self.label} - {self.detail}" if self.detail else self.label
            self._progress.update(self._task_id, advance=applied_advance, description=description)


class StageReporter:
    """Terminal reporter using Rich progress displays."""

    def __init__(self, console: Console | None = None) -> None:
        """Initialize the reporter with an optional Rich console."""
        self._console = console or Console()
        self._active_progress: Progress | None = None
        self._active_handle: StageHandle | None = None

    def begin_run(self, input_path: Path) -> None:
        """Render the start of a processing run."""
        self._console.print(f"[bold cyan]Starting[/] {input_path.name}")

    @contextmanager
    def track(
        self, key: str, label: str, *, total: float | None = None, detail: str | None = None
    ) -> Iterator[StageHandle]:
        """Open a stage display and yield a handle for progress updates."""
        del key  # part of the reporter protocol; recording test reporters consume stage keys
        progress = Progress(
            *self._columns(determinate=total is not None), console=self._console, transient=True
        )
        description = f"{label} - {detail}" if detail else label
        progress.start()
        task_id = progress.add_task(description, total=total)
        handle = StageHandle(
            label=label,
            started_at=perf_counter(),
            detail=detail,
            _progress=progress,
            _task_id=task_id,
        )
        self._active_progress = progress
        self._active_handle = handle
        completed_normally = False
        try:
            yield handle
            completed_normally = True
        finally:
            self._stop_active_progress()
            if completed_normally:
                elapsed = handle.elapsed_sec()
                detail_suffix = f" - {handle.detail}" if handle.detail else ""
                self._console.print(
                    f"[green]✓[/] {handle.label}{detail_suffix} [dim]({elapsed:.2f}s)[/]"
                )
                self._clear_active()

    def warn(self, message: str) -> None:
        """Render a warning message."""
        self._console.print(f"[yellow]![/] {message}")

    def interrupted(self) -> None:
        """Render an interrupted-run message referencing the active stage if any."""
        suffix = ""
        if self._active_handle is not None:
            suffix = f" during {self._active_handle.label.lower()}"
        self.reset_active_display()
        self._console.print(f"[red]✗[/] Interrupted{suffix}.")

    def reset_active_display(self) -> None:
        """Stop any active progress display and clear active-stage state."""
        self._stop_active_progress()
        self._clear_active()

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
        table.add_row("Processing", Text(_processing_detail(artifacts)))
        table.add_row("Warnings", warning_text)
        self._console.print()
        self._console.print(
            Panel.fit(table, title="[bold green]Completed[/]", border_style="green")
        )

    def _stop_active_progress(self) -> None:
        if self._active_progress is not None:
            self._active_progress.stop()
        self._active_progress = None

    def _clear_active(self) -> None:
        self._active_handle = None

    @staticmethod
    def _columns(*, determinate: bool) -> tuple[ProgressColumn, ...]:
        middle: tuple[ProgressColumn, ...] = (
            (
                BarColumn(),
                TaskProgressColumn(),
                TextColumn("ETA"),
                TimeRemainingColumn(compact=True),
            )
            if determinate
            else ()
        )
        return (SpinnerColumn(), TextColumn("{task.description}"), *middle, TimeElapsedColumn())


def _processing_detail(artifacts: ProcessArtifacts) -> str:
    total_sec = sum(artifacts.diagnostics.stage_durations_sec.values())
    media_duration_sec = artifacts.media_asset.duration_sec
    rtf = format(round(media_duration_sec / total_sec, 2), "g")
    return f"{format_duration(total_sec)} | RTF {rtf}x"
