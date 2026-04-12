"""Tests for terminal progress helpers."""

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest
from rich.console import Console

from webinar_transcriber.ui import RichStageReporter, _format_count, _rate_text_for_update

if TYPE_CHECKING:
    from webinar_transcriber.processor import ProcessArtifacts


class TestFormatHelpers:
    def test_format_count_renders_frame_counter(self) -> None:
        count_text = _format_count(
            completed=74.0,
            total=100.0,
            count_label="frames",
            count_multiplier=100.0,
        )

        assert count_text == "7400/10000 frames"

    def test_format_count_renders_compact_seconds_suffix(self) -> None:
        count_text = _format_count(
            completed=74.0,
            total=100.0,
            count_label="s",
            count_multiplier=1.0,
        )

        assert count_text == "74/100s"

    def test_rate_text_for_update_renders_frames_per_second(self) -> None:
        rate_text = _rate_text_for_update(
            completed=74.0,
            now=2.0,
            started_at=1.0,
            rate_label="frames/s",
            rate_multiplier=100.0,
        )

        assert rate_text == "7400 frames/s"

    def test_rate_text_for_update_hides_empty_values(self) -> None:
        rate_text = _rate_text_for_update(
            completed=0.0,
            now=2.0,
            started_at=1.0,
            rate_label="frames/s",
            rate_multiplier=100.0,
        )

        assert rate_text == ""

    def test_count_and_rate_helpers_hide_empty_values(self) -> None:
        formatted_count = _format_count(
            completed=3.0,
            total=None,
            count_label="frames",
            count_multiplier=1.0,
        )
        rate_text = _rate_text_for_update(
            completed=3.0,
            now=2.0,
            started_at=2.0,
            rate_label="frames/s",
            rate_multiplier=1.0,
        )

        assert formatted_count == ""
        assert rate_text == ""


class TestRichStageReporter:
    def test_complete_run_renders_completion_panel(self) -> None:
        console = Console(record=True, width=100)
        reporter = RichStageReporter(console=console)
        artifacts = cast(
            "ProcessArtifacts",
            SimpleNamespace(
                layout=SimpleNamespace(run_dir="runs/example"),
                report=SimpleNamespace(
                    detected_language="ru",
                    sections=[object(), object()],
                    warnings=["warning one"],
                ),
            ),
        )

        reporter.complete_run(artifacts)

        output = console.export_text()
        assert "Completed" in output
        assert "Run directory" in output
        assert "runs/example" in output
        assert "Language" in output
        assert "Sections" in output
        assert "Warnings" in output

    def test_stage_started_records_elapsed_time(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        console = Console(record=True, width=100)
        reporter = RichStageReporter(console=console)
        perf_values = iter([10.0, 13.5])
        monkeypatch.setattr("webinar_transcriber.ui.perf_counter", lambda: next(perf_values))

        reporter.stage_started("llm_report", "Polishing report")
        reporter.stage_finished("llm_report", "Polishing report")

        output = console.export_text()
        assert "Polishing report" in output
        assert "(3.50s)" in output

    def test_progress_updates_compute_rate_and_preserve_detail(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        console = Console(record=True, width=100)
        reporter = RichStageReporter(console=console)
        perf_values = iter([20.0, 22.0, 23.0])
        monkeypatch.setattr("webinar_transcriber.ui.perf_counter", lambda: next(perf_values))

        reporter.progress_started(
            "extract_frames",
            "Extracting frames",
            total=4.0,
            count_label="frames",
            rate_label="frames/s",
        )
        reporter.progress_advanced("extract_frames", advance=2.0, detail="scene-2")

        assert reporter._active_progress is not None
        assert reporter._active_task_id is not None

        task = reporter._active_progress.tasks[reporter._active_task_id]
        assert task.fields["detail_text"] == "scene-2"
        assert task.fields["rate_text"] == "1.0 frames/s"

        reporter.stage_finished("extract_frames", "Extracting frames", detail="done")

        output = console.export_text()
        assert "Extracting frames - done" in output

    def test_warn_and_interrupted_render_messages(self) -> None:
        console = Console(record=True, width=100)
        reporter = RichStageReporter(console=console)

        reporter.stage_started("probe_media", "Probing media")
        reporter.warn("ffprobe returned a warning")
        reporter.stage_started("detect_scenes", "Detecting scenes")
        reporter.interrupted()

        output = console.export_text()
        assert "ffprobe returned a warning" in output
        assert "Interrupted during detecting scenes." in output

    def test_progress_advanced_ignores_inactive_or_mismatched_stage_keys(self) -> None:
        reporter = RichStageReporter(console=Console(record=True, width=100))

        reporter.progress_advanced("extract_frames", advance=1.0)

        reporter.progress_started("extract_frames", "Extracting frames", total=2.0)
        reporter.progress_advanced("detect_scenes", advance=1.0)

        assert reporter._active_progress is not None
        assert reporter._active_task_id is not None

        task = reporter._active_progress.tasks[reporter._active_task_id]
        assert task.completed == 0

    def test_stage_finished_without_matching_active_event_reports_zero_elapsed(self) -> None:
        console = Console(record=True, width=100)
        reporter = RichStageReporter(console=console)

        reporter.stage_started("probe_media", "Probing media")
        reporter.stage_finished("other_stage", "Other stage")

        output = console.export_text()
        assert "Other stage" in output
        assert "(0.00s)" in output
