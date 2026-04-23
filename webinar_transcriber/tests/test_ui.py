"""Tests for terminal progress helpers."""

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import Mock

import pytest
from rich.console import Console
from rich.panel import Panel

from webinar_transcriber.reporter import BaseStageReporter
from webinar_transcriber.ui import (
    RichStageReporter,
    _count_text,
    _rate_text,
)

if TYPE_CHECKING:
    from webinar_transcriber.processor import ProcessArtifacts


class TestFormatHelpers:
    def test_count_text_renders_frame_counter(self) -> None:
        count_text = _count_text(
            completed=74.0, total=100.0, count_label="frames", count_multiplier=100.0
        )

        assert count_text == "7400/10000 frames"

    def test_count_text_renders_compact_seconds_suffix(self) -> None:
        count_text = _count_text(completed=74.0, total=100.0, count_label="s", count_multiplier=1.0)

        assert count_text == "74/100s"

    def test_rate_text_renders_frames_per_second(self) -> None:
        rate_text = _rate_text(
            completed=74.0, elapsed_sec=1.0, rate_label="frames/s", rate_multiplier=100.0
        )

        assert rate_text == "7400 frames/s"

    def test_rate_text_hides_empty_values(self) -> None:
        rate_text = _rate_text(
            completed=0.0, elapsed_sec=1.0, rate_label="frames/s", rate_multiplier=100.0
        )

        assert rate_text == ""

    def test_count_and_rate_helpers_hide_empty_values(self) -> None:
        formatted_count = _count_text(
            completed=3.0, total=None, count_label="frames", count_multiplier=1.0
        )
        rate_text = _rate_text(
            completed=3.0, elapsed_sec=0.0, rate_label="frames/s", rate_multiplier=1.0
        )

        assert formatted_count == ""
        assert rate_text == ""


class TestRichStageReporter:
    def test_begin_run_prints_input_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        console = Console(width=100)
        print_mock = Mock()
        monkeypatch.setattr(console, "print", print_mock)
        reporter = RichStageReporter(console=console)

        reporter.begin_run(Path("demo.wav"))

        print_mock.assert_called_once_with("[bold cyan]Starting[/] demo.wav")

    def test_complete_run_renders_completion_panel(self, monkeypatch: pytest.MonkeyPatch) -> None:
        console = Console(width=100)
        print_mock = Mock()
        monkeypatch.setattr(console, "print", print_mock)
        reporter = RichStageReporter(console=console)
        artifacts = cast(
            "ProcessArtifacts",
            SimpleNamespace(
                layout=SimpleNamespace(run_dir="runs/example"),
                report=SimpleNamespace(
                    detected_language="ru", sections=[object(), object()], warnings=["warning one"]
                ),
            ),
        )

        reporter.complete_run(artifacts)

        assert print_mock.call_count == 2
        assert print_mock.call_args_list[0].args == ()
        panel = print_mock.call_args_list[1].args[0]
        assert isinstance(panel, Panel)
        assert panel.title == "[bold green]Completed[/]"

    def test_stage_started_records_elapsed_time(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeStatus:
            def __init__(self) -> None:
                self.started = False
                self.stopped = False

            def start(self) -> None:
                self.started = True

            def stop(self) -> None:
                self.stopped = True

        console = Console(width=100)
        print_mock = Mock()
        fake_status = FakeStatus()
        status_mock = Mock(return_value=fake_status)
        monkeypatch.setattr(console, "print", print_mock)
        monkeypatch.setattr(console, "status", status_mock)
        reporter = RichStageReporter(console=console)
        perf_values = iter([10.0, 13.5])
        monkeypatch.setattr("webinar_transcriber.ui.perf_counter", lambda: next(perf_values))

        reporter.stage_started("llm_report", "Polishing report")
        reporter.stage_finished("llm_report", "Polishing report")

        status_mock.assert_called_once_with(
            "[bold blue][1][/bold blue] Polishing report", spinner="dots"
        )
        assert fake_status.started
        assert fake_status.stopped
        print_mock.assert_called_once_with("[green]\u2713[/] Polishing report [dim](3.50s)[/]")

    def test_progress_started_initializes_progress_display(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakeProgress:
            def __init__(self, *_args, **_kwargs) -> None:
                self.started = False
                self.added_task: tuple[str, float, dict[str, object]] | None = None

            def start(self) -> None:
                self.started = True

            def add_task(self, label: str, *, total: float, **fields: object) -> int:
                self.added_task = (label, total, fields)
                return 7

        def fake_column(*_args: object, **_kwargs: object) -> object:
            return object()

        monkeypatch.setattr("webinar_transcriber.ui.Progress", FakeProgress)
        monkeypatch.setattr("webinar_transcriber.ui.SpinnerColumn", fake_column)
        monkeypatch.setattr("webinar_transcriber.ui.TextColumn", fake_column)
        monkeypatch.setattr("webinar_transcriber.ui.BarColumn", fake_column)
        monkeypatch.setattr("webinar_transcriber.ui.TaskProgressColumn", fake_column)
        monkeypatch.setattr("webinar_transcriber.ui.TimeRemainingColumn", fake_column)
        monkeypatch.setattr("webinar_transcriber.ui.TimeElapsedColumn", fake_column)
        monkeypatch.setattr("webinar_transcriber.ui.perf_counter", lambda: 10.0)
        reporter = RichStageReporter(console=Console(width=100))

        reporter.progress_started(
            "extract_frames",
            "Extracting frames",
            total=0.0,
            count_label="frames",
            rate_label="frames/s",
            detail="scene-1",
        )

        assert reporter._active_stage_key == "extract_frames"
        assert reporter._active_stage_label == "Extracting frames"
        assert reporter._active_stage_started_at == 10.0
        assert reporter._active_progress is not None
        active_progress = cast("Any", reporter._active_progress)
        assert active_progress.started
        assert active_progress.added_task == (
            "Extracting frames",
            1.0,
            {
                "count_label": "frames",
                "count_multiplier": 1.0,
                "count_text": "0/1 frames",
                "rate_label": "frames/s",
                "rate_multiplier": 1.0,
                "rate_text": "",
                "detail_text": "scene-1",
            },
        )
        assert reporter._active_task_id == 7

    def test_progress_updates_compute_rate_and_preserve_detail(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakeTask:
            def __init__(self, total: float, fields: dict[str, object]) -> None:
                self.total = total
                self.completed = 0.0
                self.fields = fields

        class FakeProgress:
            def __init__(self, *_args, **_kwargs) -> None:
                self.tasks: list[FakeTask] = []
                self.started = False
                self.stopped = False

            def start(self) -> None:
                self.started = True

            def stop(self) -> None:
                self.stopped = True

            def add_task(self, _label: str, *, total: float, **fields: object) -> int:
                self.tasks.append(FakeTask(total, fields))
                return len(self.tasks) - 1

            def update(self, task_id: int, *, advance: float = 0.0, **fields: object) -> None:
                task = self.tasks[task_id]
                task.completed += advance
                task.fields.update(fields)

        console = Console(width=100)
        print_mock = Mock()
        monkeypatch.setattr(console, "print", print_mock)
        reporter = RichStageReporter(console=console)
        perf_values = iter([22.0, 23.0])
        monkeypatch.setattr("webinar_transcriber.ui.perf_counter", lambda: next(perf_values))

        any_reporter = cast("Any", reporter)
        any_reporter._active_stage_key = "extract_frames"
        any_reporter._active_stage_label = "Extracting frames"
        any_reporter._active_stage_started_at = 20.0
        any_reporter._active_progress = FakeProgress()
        any_reporter._active_progress.start()
        any_reporter._active_task_id = any_reporter._active_progress.add_task(
            "Extracting frames",
            total=4.0,
            count_label="frames",
            count_multiplier=1.0,
            count_text="0/4 frames",
            rate_label="frames/s",
            rate_multiplier=1.0,
            rate_text="",
            detail_text="",
        )
        reporter.progress_advanced("extract_frames", advance=2.0, detail="scene-2")

        assert reporter._active_progress is not None
        assert reporter._active_task_id is not None

        task = reporter._active_progress.tasks[reporter._active_task_id]
        assert task.fields["detail_text"] == "scene-2"
        assert task.fields["count_text"] == "2/4 frames"
        assert task.fields["rate_text"] == "1.0 frames/s"

        reporter.stage_finished("extract_frames", "Extracting frames", detail="done")

        assert reporter._active_progress is None
        print_mock.assert_called_once_with(
            "[green]\u2713[/] Extracting frames - done [dim](3.00s)[/]"
        )

    def test_warn_and_interrupted_render_messages(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeStatus:
            def start(self) -> None:
                pass

            def stop(self) -> None:
                pass

        console = Console(width=100)
        print_mock = Mock()
        status_mock = Mock(return_value=FakeStatus())
        monkeypatch.setattr(console, "print", print_mock)
        monkeypatch.setattr(console, "status", status_mock)
        reporter = RichStageReporter(console=console)

        reporter.stage_started("probe_media", "Probing media")
        reporter.warn("ffprobe returned a warning")
        reporter.stage_started("detect_scenes", "Detecting scenes")
        reporter.interrupted()

        assert print_mock.call_args_list[0].args == ("[yellow]![/] ffprobe returned a warning",)
        assert print_mock.call_args_list[1].args == (
            "[red]\u2717[/] Interrupted during detecting scenes.",
        )

    def test_progress_advanced_ignores_inactive_or_mismatched_stage_keys(self) -> None:
        class FakeTask:
            def __init__(self) -> None:
                self.completed = 0.0

        reporter = RichStageReporter(console=Console(width=100))

        reporter.progress_advanced("extract_frames", advance=1.0)

        any_reporter = cast("Any", reporter)
        any_reporter._active_stage_key = "extract_frames"
        any_reporter._active_stage_label = "Extracting frames"
        any_reporter._active_stage_started_at = 0.0
        update_mock = Mock()
        any_reporter._active_progress = SimpleNamespace(tasks=[FakeTask()], update=update_mock)
        any_reporter._active_task_id = 0
        reporter.progress_advanced("detect_scenes", advance=1.0)

        assert reporter._active_progress is not None
        assert reporter._active_task_id is not None

        task = reporter._active_progress.tasks[reporter._active_task_id]
        assert task.completed == 0
        update_mock.assert_not_called()

    def test_progress_advanced_ignores_active_stage_without_progress_display(self) -> None:
        reporter = RichStageReporter(console=Console(record=True, width=100))
        reporter._active_stage_key = "extract_frames"
        reporter._active_stage_label = "Extracting frames"
        reporter._active_stage_started_at = 0.0

        reporter.progress_advanced("extract_frames", advance=1.0)

        assert reporter._active_progress is None

    def test_stage_finished_without_matching_active_event_reports_zero_elapsed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakeStatus:
            def start(self) -> None:
                pass

            def stop(self) -> None:
                pass

        console = Console(width=100)
        print_mock = Mock()
        status_mock = Mock(return_value=FakeStatus())
        monkeypatch.setattr(console, "print", print_mock)
        monkeypatch.setattr(console, "status", status_mock)
        reporter = RichStageReporter(console=console)

        reporter.stage_started("probe_media", "Probing media")
        reporter.stage_finished("other_stage", "Other stage")

        print_mock.assert_called_once_with("[green]\u2713[/] Other stage [dim](0.00s)[/]")


class TestBaseStageReporter:
    def test_warning_and_interrupt_noops_accept_calls(self) -> None:
        reporter = BaseStageReporter()

        reporter.warn("warning")
        reporter.interrupted()
        reporter.reset_active_display()
