"""Rich-based terminal progress reporting."""

from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from webinar_transcriber.reporter import NullStageReporter, StageEvent

if TYPE_CHECKING:
    from pathlib import Path

    from rich.status import Status

    from webinar_transcriber.processor import ProcessArtifacts


class RichStageReporter(NullStageReporter):
    """Terminal reporter using Rich status spinners and summaries."""

    def __init__(self, console: Console | None = None) -> None:
        self._console = console or Console()
        self._active_status: Status | None = None
        self._active_progress: Progress | None = None
        self._active_task_id: TaskID | None = None
        self._active_event: StageEvent | None = None
        self._warnings: list[str] = []
        self._stage_count = 0

    def begin_run(self, input_path: Path, *, output_format: str) -> None:
        self._console.print(f"[bold cyan]Starting[/] {input_path.name} (format={output_format})")

    def stage_started(self, stage_key: str, label: str) -> None:
        self._stop_active_display()
        self._stage_count += 1
        self._active_event = self._new_stage_event(stage_key, label)
        self._active_status = self._console.status(
            f"[bold blue][{self._stage_count}][/bold blue] {label}",
            spinner="dots",
        )
        self._active_status.start()

    def stage_timing_started(self, stage_key: str, label: str) -> None:
        self._stop_active_display()
        self._stage_count += 1
        self._active_event = self._new_stage_event(stage_key, label)

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
        self._stop_active_display()
        self._stage_count += 1
        self._active_event = self._new_stage_event(stage_key, label)
        self._active_progress = Progress(
            SpinnerColumn(),
            TextColumn(f"[bold blue][{self._stage_count}][/bold blue] {label}"),
            BarColumn(),
            TaskProgressColumn(),
            CountColumn(),
            RateColumn(),
            TextColumn("{task.fields[detail_text]}", style="dim"),
            TextColumn("ETA"),
            TimeRemainingColumn(compact=True),
            TimeElapsedColumn(),
            console=self._console,
            transient=True,
        )
        self._active_progress.start()
        self._active_task_id = self._active_progress.add_task(
            label,
            total=max(total, 1.0),
            count_label=count_label,
            count_multiplier=count_multiplier,
            rate_label=rate_label,
            rate_multiplier=rate_multiplier,
            rate_text="",
            detail_text=detail or "",
        )

    def progress_advanced(
        self,
        stage_key: str,
        *,
        advance: float = 1.0,
        detail: str | None = None,
    ) -> None:
        if self._active_event is None or self._active_event.stage_key != stage_key:
            return
        if self._active_progress is None or self._active_task_id is None:
            return
        self._active_progress.update(
            self._active_task_id,
            advance=advance,
            detail_text=detail or "",
        )
        task = self._active_progress.tasks[self._active_task_id]
        now = perf_counter()
        self._active_progress.update(
            self._active_task_id,
            rate_text=_rate_text_for_update(
                completed=task.completed,
                now=now,
                started_at=self._active_event.started_at,
                rate_label=task.fields.get("rate_label"),
                rate_multiplier=float(task.fields.get("rate_multiplier", 1.0)),
            ),
        )

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

    def interrupted(self) -> None:
        self._stop_active_display()
        stage_suffix = ""
        if self._active_event is not None:
            stage_suffix = f" during {self._active_event.label.lower()}"
        self._console.print(f"[red]\u2717[/] Interrupted{stage_suffix}.")
        self._active_event = None

    def reset_active_display(self) -> None:
        self._stop_active_display()
        self._active_event = None

    def complete_run(self, artifacts: ProcessArtifacts) -> None:
        table = Table.grid(padding=(0, 2))
        table.add_column(style="dim")
        table.add_column()
        warning_count = len(artifacts.report.warnings)
        warning_text = Text(
            str(warning_count),
            style="yellow" if warning_count else "green",
        )
        table.add_row("Run directory", Text(str(artifacts.layout.run_dir), style="cyan"))
        table.add_row("Language", Text(artifacts.report.detected_language or "unknown"))
        table.add_row("Sections", Text(str(len(artifacts.report.sections))))
        table.add_row("Warnings", warning_text)
        self._console.print()
        self._console.print(
            Panel.fit(
                table,
                title="[bold green]Completed[/]",
                border_style="green",
            )
        )

    def _stop_active_display(self) -> None:
        if self._active_status is not None:
            self._active_status.stop()
            self._active_status = None
        if self._active_progress is not None:
            self._active_progress.stop()
            self._active_progress = None
            self._active_task_id = None

    def _new_stage_event(self, stage_key: str, label: str) -> StageEvent:
        return StageEvent(
            stage_key=stage_key,
            label=label,
            started_at=perf_counter(),
        )


class RateColumn(ProgressColumn):
    """Render an optional rate derived from task completion over time."""

    def render(self, task: Task) -> Text:
        rate_text = str(task.fields.get("rate_text", ""))
        return Text(rate_text, style="progress.data.speed")


class CountColumn(ProgressColumn):
    """Render a done/total counter with optional unit scaling."""

    def render(self, task: Task) -> Text:
        count_text = _format_count(
            completed=task.completed,
            total=task.total,
            count_label=task.fields.get("count_label"),
            count_multiplier=float(task.fields.get("count_multiplier", 1.0)),
            has_rate=bool(task.fields.get("rate_text")),
        )
        return Text(count_text, style="progress.data.speed")


def _format_count(
    *,
    completed: float,
    total: float | None,
    count_label: str | None,
    count_multiplier: float,
    has_rate: bool = False,
) -> str:
    if total is None:
        return ""

    completed_count = int(completed * count_multiplier)
    total_count = int(total * count_multiplier)
    if count_label:
        sep = "" if count_label in {"s", "ms", "%"} else " "
        unit_suffix = f"{sep}{count_label}"
    else:
        unit_suffix = ""
    comma_suffix = "," if has_rate else ""
    return f"{completed_count}/{total_count}{unit_suffix}{comma_suffix}"


def _rate_text_for_update(
    *,
    completed: float,
    now: float,
    started_at: float,
    rate_label: str | None,
    rate_multiplier: float,
) -> str:
    if not rate_label or completed <= 0:
        return ""

    elapsed = now - started_at
    if elapsed <= 0:
        return ""

    rate = completed * rate_multiplier / elapsed
    display_rate = f"{rate:.0f}" if rate >= 100 else f"{rate:.1f}"
    return f"{display_rate} {rate_label}"
