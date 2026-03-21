"""Tests for terminal progress helpers."""

from webinar_transcriber.ui import _format_count, _rate_text_for_update


def test_format_count_renders_frame_counter() -> None:
    assert (
        _format_count(
            completed=74.0,
            total=100.0,
            count_label="frames",
            count_multiplier=100.0,
        )
        == "7400/10000 frames"
    )


def test_rate_text_for_update_renders_frames_per_second() -> None:
    assert (
        _rate_text_for_update(
            completed=74.0,
            now=2.0,
            started_at=1.0,
            rate_label="frames/s",
            rate_multiplier=100.0,
        )
        == ", 7400 frames/s"
    )


def test_rate_text_for_update_hides_empty_values() -> None:
    assert (
        _rate_text_for_update(
            completed=0.0,
            now=2.0,
            started_at=1.0,
            rate_label="frames/s",
            rate_multiplier=100.0,
        )
        == ""
    )
