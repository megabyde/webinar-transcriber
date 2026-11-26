"""Shared exporter formatting helpers."""

from __future__ import annotations

import re
from math import floor
from pathlib import Path

FORMATTED_PARAGRAPH_RE = re.compile(
    r"^(?:"
    r"\*\*(?P<speaker>S\d+):\*\*\s*(?P<speaker_body>.*)"
    r"|[-*]\s+(?P<bullet>.+)"
    r"|\d+[.)]\s+(?P<number>.+)"
    r")$"
)


def format_timecode(total_sec: float) -> str:
    """Format a section timestamp as MM:SS or HH:MM:SS.

    Returns:
        str: The formatted timecode string.
    """
    clamped_sec = max(total_sec, 0.0)
    rounded_sec = floor(clamped_sec)
    hours, remainder = divmod(rounded_sec, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def format_duration(duration_sec: float) -> str:
    """Format an elapsed duration for terminal summaries."""
    total_seconds = round(duration_sec)
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def format_count(n: int, noun: str) -> str:
    """Format a count with a simple plural suffix."""
    return f"{n} {noun}{'' if n == 1 else 's'}"


def format_rtf(audio_sec: float, elapsed_sec: float) -> str:
    """Format a real-time factor detail."""
    return f"RTF {format(round(audio_sec / elapsed_sec, 2), 'g')}x"


def join_detail(*parts: str | None) -> str:
    """Join non-empty stage detail fragments."""
    return " | ".join(part for part in parts if part)


def section_timecode(start_sec: float, end_sec: float) -> str:
    """Format a section start/end range for report headings.

    Returns:
        str: The formatted section range.
    """
    return f"{format_timecode(start_sec)}\N{EN DASH}{format_timecode(end_sec)}"


def resolve_section_image(image_path: str | None, base_dir: Path) -> Path | None:
    """Resolve a report image path against the exported artifact's directory.

    Returns:
        Path | None: The existing image file, or ``None`` when the section has no usable image.
    """
    if not image_path:
        return None
    resolved_path = Path(image_path)
    if not resolved_path.is_absolute():
        resolved_path = base_dir / resolved_path
    # A missing section image does not invalidate the report; exporters omit it and continue
    return resolved_path if resolved_path.exists() else None
