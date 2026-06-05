"""Tests for terminal progress helpers."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

from webinar_transcriber.ui import StageReporter


class TestStageReporter:
    def test_begin_run_prints_input_name(self) -> None:
        console = Console(record=True, width=100)
        reporter = StageReporter(console=console)

        reporter.begin_run(Path("demo.wav"))

        assert console.export_text() == "Starting demo.wav\n"

    def test_complete_run_renders_completion_panel(self) -> None:
        console = Console(record=True, width=100)
        reporter = StageReporter(console=console)
        artifacts = SimpleNamespace(
            layout=SimpleNamespace(
                run_dir="runs/example", diagnostics_path="runs/example/diagnostics.json"
            ),
            report=SimpleNamespace(
                detected_language="ru", sections=[object(), object()], warnings=["warning one"]
            ),
            diagnostics=SimpleNamespace(stage_durations_sec={"prepare": 1.0, "transcribe": 9.0}),
            media_asset=SimpleNamespace(duration_sec=50.0),
        )

        reporter.complete_run(artifacts)  # type: ignore

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

    def test_track_indeterminate_records_elapsed_time(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        console = Console(record=True, width=100)
        reporter = StageReporter(console=console)
        perf_values = iter([10.0, 13.5])
        monkeypatch.setattr("webinar_transcriber.ui.perf_counter", lambda: next(perf_values))

        with reporter.track("llm_report", "Polishing report"):
            pass

        assert console.export_text().strip() == "✓ Polishing report (3.50s)"

    def test_track_records_detail_on_finish_line(self) -> None:
        console = Console(record=True, width=100)
        reporter = StageReporter(console=console)

        with reporter.track("prepare_asr", "Preparing ASR model") as handle:
            handle.update(detail="large-v3-turbo | metal")

        output = console.export_text()
        assert "Preparing ASR model - large-v3-turbo | metal" in output

    def test_track_determinate_records_progress_advances(self) -> None:
        console = Console(record=True, width=100)
        reporter = StageReporter(console=console)

        with reporter.track("extract_frames", "Extracting frames", total=4.0) as handle:
            handle.update(advance=2.0, detail="scene-2")
            assert handle.completed == 2.0
            assert handle.detail == "scene-2"

        assert "Extracting frames - scene-2" in console.export_text()

    def test_update_with_completed_advances_relative_to_current(self) -> None:
        reporter = StageReporter(console=Console(quiet=True))

        with reporter.track("vad", "Detecting", total=10.0) as handle:
            handle.update(completed=4.0)
            handle.update(completed=4.0)  # no-op advance
            handle.update(completed=7.0)

            assert handle.completed == 7.0

    def test_update_with_zero_advance_keeps_detail_in_sync(self) -> None:
        reporter = StageReporter(console=Console(quiet=True))

        with reporter.track("vad", "Detecting", total=10.0) as handle:
            handle.update(completed=10.0, detail="done")
            handle.update(detail="finalized")

            assert handle.completed == 10.0
            assert handle.detail == "finalized"

    def test_warn_renders_messages(self) -> None:
        console = Console(record=True, width=100)
        reporter = StageReporter(console=console)

        reporter.warn("ffprobe returned a warning")

        assert console.export_text() == "! ffprobe returned a warning\n"

    def test_interrupted_during_active_stage_names_it(self) -> None:
        console = Console(record=True, width=100)
        reporter = StageReporter(console=console)

        try:
            with reporter.track("detect_scenes", "Detecting scenes"):
                raise KeyboardInterrupt
        except KeyboardInterrupt:
            reporter.interrupted()

        output = console.export_text()
        assert "Interrupted during detecting scenes." in output
        # finish line should not be printed on exception
        assert "✓ Detecting scenes" not in output

    def test_interrupted_without_active_stage_omits_suffix(self) -> None:
        console = Console(record=True, width=100)
        reporter = StageReporter(console=console)

        reporter.interrupted()

        assert console.export_text() == "✗ Interrupted.\n"

    def test_reset_active_display_is_safe_when_idle(self) -> None:
        reporter = StageReporter(console=Console(quiet=True))

        reporter.reset_active_display()  # no active display; should not error
