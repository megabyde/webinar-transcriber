"""Shared exporter formatting helpers."""

from __future__ import annotations

from math import floor

EN_DASH = "\N{EN DASH}"


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


def section_timecode(start_sec: float, end_sec: float) -> str:
    """Format a section start/end range for report headings.

    Returns:
        str: The formatted section range.
    """
    return f"{format_timecode(start_sec)}{EN_DASH}{format_timecode(end_sec)}"
