"""Tests for terminal progress helpers."""

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

from rich.console import Console

from webinar_transcriber.ui import RichStageReporter, _format_count, _rate_text_for_update

if TYPE_CHECKING:
    from webinar_transcriber.processor import ProcessArtifacts


def test_format_count_renders_frame_counter() -> None:
    count_text = _format_count(
        completed=74.0,
        total=100.0,
        count_label="frames",
        count_multiplier=100.0,
    )

    assert count_text == "7400/10000 frames"


def test_format_count_renders_compact_seconds_suffix() -> None:
    count_text = _format_count(
        completed=74.0,
        total=100.0,
        count_label="s",
        count_multiplier=1.0,
    )

    assert count_text == "74/100s"


def test_rate_text_for_update_renders_frames_per_second() -> None:
    rate_text = _rate_text_for_update(
        completed=74.0,
        now=2.0,
        started_at=1.0,
        rate_label="frames/s",
        rate_multiplier=100.0,
    )

    assert rate_text == "7400 frames/s"


def test_rate_text_for_update_hides_empty_values() -> None:
    rate_text = _rate_text_for_update(
        completed=0.0,
        now=2.0,
        started_at=1.0,
        rate_label="frames/s",
        rate_multiplier=100.0,
    )

    assert rate_text == ""


def test_complete_run_renders_completion_panel() -> None:
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
