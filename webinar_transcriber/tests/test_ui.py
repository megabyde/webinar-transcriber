"""Tests for terminal progress helpers."""

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest
from rich.console import Console

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
        count_text = _count_text(completed=74.0, total=100.0, count_label="frames")

        assert count_text == "74/100 frames"

    def test_count_text_renders_compact_seconds_suffix(self) -> None:
        count_text = _count_text(completed=74.0, total=100.0, count_label="s")

        assert count_text == "74/100s"

    def test_rate_text_renders_frames_per_second(self) -> None:
        rate_text = _rate_text(completed=74.0, elapsed_sec=1.0, rate_label="frames/s")

        assert rate_text == "74.0 frames/s"

    def test_rate_text_hides_empty_values(self) -> None:
        rate_text = _rate_text(completed=0.0, elapsed_sec=1.0, rate_label="frames/s")

        assert rate_text is None

    def test_count_and_rate_helpers_hide_empty_values(self) -> None:
        formatted_count = _count_text(completed=3.0, total=None, count_label="frames")
        rate_text = _rate_text(completed=3.0, elapsed_sec=0.0, rate_label="frames/s")

        assert formatted_count is None
        assert rate_text is None


class TestRichStageReporter:
    def test_begin_run_prints_input_name(self) -> None:
        console = Console(record=True, width=100)
        reporter = RichStageReporter(console=console)

        reporter.begin_run(Path("demo.wav"))

        assert console.export_text() == "Starting demo.wav\n"

    def test_complete_run_renders_completion_panel(self) -> None:
        console = Console(record=True, width=100)
        reporter = RichStageReporter(console=console)
        artifacts = cast(
            "ProcessArtifacts",
            SimpleNamespace(
                layout=SimpleNamespace(
                    run_dir="runs/example", diagnostics_path="runs/example/diagnostics.json"
                ),
                report=SimpleNamespace(
                    detected_language="ru", sections=[object(), object()], warnings=["warning one"]
                ),
                diagnostics=SimpleNamespace(
                    stage_durations_sec={"prepare": 1.0, "transcribe": 9.0}
                ),
                media_asset=SimpleNamespace(duration_sec=50.0),
            ),
        )

        reporter.complete_run(artifacts)

        output = console.export_text()
        assert "Completed" in output
        assert "Run directory" in output
        assert "runs/example" in output
        assert "Diagnostics" in output
        assert "runs/example/diagnostics.json" in output
        assert "Language" in output
        assert "ru" in output
        assert "Sections" in output
        assert "2" in output
        assert "Processing" in output
        assert "10s | RTF 5x" in output
        assert "Warnings" in output
        assert "1" in output

    def test_stage_started_records_elapsed_time(self, monkeypatch: pytest.MonkeyPatch) -> None:
        console = Console(record=True, width=100)
        reporter = RichStageReporter(console=console)
        perf_values = iter([10.0, 13.5])
        monkeypatch.setattr("webinar_transcriber.ui.perf_counter", lambda: next(perf_values))

        reporter.stage_started("llm_report", "Polishing report")
        reporter.stage_finished("llm_report", "Polishing report")

        assert console.export_text() == "✓ Polishing report (3.50s)\n"

    def test_progress_started_initializes_progress_display(
        self, monkeypatch: pytest.MonkeyPatch, fake_rich_progress
    ) -> None:
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

        progress = fake_rich_progress[0]
        assert progress.started
        assert progress.added_task == (
            1.0,
            {
                "count_label": "frames",
                "count_text": "0/1 frames",
                "rate_label": "frames/s",
                "rate_text": "",
                "detail_text": "scene-1",
            },
        )

    def test_progress_updates_compute_rate_and_preserve_detail_in_progress_adapter(
        self, monkeypatch: pytest.MonkeyPatch, fake_rich_progress
    ) -> None:
        perf_values = iter([22.0, 23.0, 25.0])
        monkeypatch.setattr("webinar_transcriber.ui.perf_counter", lambda: next(perf_values))

        console = Console(record=True, width=100)
        reporter = RichStageReporter(console=console)
        reporter.progress_started(
            "extract_frames",
            "Extracting frames",
            total=4.0,
            count_label="frames",
            rate_label="frames/s",
        )
        reporter.progress_advanced("extract_frames", advance=2.0, detail="scene-2")

        progress = fake_rich_progress[0]
        task = progress.tasks[0]
        assert task.fields["detail_text"] == "scene-2"
        assert task.fields["count_text"] == "2/4 frames"
        assert task.fields["rate_text"] == "2.0 frames/s"

        reporter.stage_finished("extract_frames", "Extracting frames", detail="done")

        assert progress.stopped
        assert console.export_text() == "✓ Extracting frames - done (3.00s)\n"

    def test_warn_and_interrupted_render_messages(self) -> None:
        console = Console(record=True, width=100)
        reporter = RichStageReporter(console=console)

        reporter.stage_started("probe_media", "Probing media")
        reporter.warn("ffprobe returned a warning")
        reporter.stage_started("detect_scenes", "Detecting scenes")
        reporter.interrupted()

        assert console.export_text() == (
            "! ffprobe returned a warning\n✗ Interrupted during detecting scenes.\n"
        )

    def test_progress_advanced_ignores_inactive_or_mismatched_stage_keys(
        self, fake_rich_progress
    ) -> None:
        reporter = RichStageReporter(console=Console(width=100))

        reporter.progress_advanced("extract_frames", advance=1.0)
        reporter.progress_started("extract_frames", "Extracting frames", total=1.0)
        progress = fake_rich_progress[0]

        reporter.progress_advanced("detect_scenes", advance=1.0)

        assert progress.tasks[0].completed == 0
        progress.update.assert_not_called()

    def test_progress_advanced_ignores_indeterminate_stage(self) -> None:
        console = Console(record=True, width=100)
        reporter = RichStageReporter(console=console)
        reporter.stage_started("extract_frames", "Extracting frames")

        reporter.progress_advanced("extract_frames", advance=1.0)

        reporter.stage_finished("extract_frames", "Extracting frames")
        assert console.export_text() == "✓ Extracting frames (0.00s)\n"

    def test_stage_finished_without_matching_active_event_reports_zero_elapsed(
        self,
    ) -> None:
        console = Console(record=True, width=100)
        reporter = RichStageReporter(console=console)

        reporter.stage_started("probe_media", "Probing media")
        reporter.stage_finished("other_stage", "Other stage")

        assert console.export_text() == "✓ Other stage (0.00s)\n"


class TestBaseStageReporter:
    def test_warning_and_interrupt_noops_accept_calls(self) -> None:
        reporter = BaseStageReporter()

        reporter.warn("warning")
        reporter.interrupted()
        reporter.reset_active_display()
