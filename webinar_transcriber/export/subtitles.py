"""Subtitle export helpers."""

from pathlib import Path

from webinar_transcriber.models import TranscriptionResult


def write_vtt_subtitles(transcription: TranscriptionResult, output_path: Path) -> Path:
    """Write a transcription as a WebVTT subtitle file."""
    lines = ["WEBVTT", ""]
    for segment in transcription.segments:
        lines.extend([
            f"{_vtt_timestamp(segment.start_sec)} --> {_vtt_timestamp(segment.end_sec)}",
            segment.text.strip(),
            "",
        ])

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _vtt_timestamp(total_seconds: float) -> str:
    rounded_milliseconds = max(0, round(total_seconds * 1000))
    seconds, milliseconds = divmod(rounded_milliseconds, 1000)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"
