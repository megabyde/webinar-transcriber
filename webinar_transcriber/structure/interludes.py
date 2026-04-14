"""Interlude detection and rendering helpers for report structuring."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .constants import (
    AUDIO_SECTION_BREAK_GAP_SEC,
    INTERLUDE_LOW_UNIQUE_RATIO,
    INTERLUDE_MARKER_PATTERN,
    INTERLUDE_MIN_WORDS,
    INTERLUDE_WORD_RE,
    MIN_INTERLUDE_DURATION_SEC,
)

if TYPE_CHECKING:
    from webinar_transcriber.models import ReportSection, TranscriptSegment


def _render_interlude_sections(
    sections: list[ReportSection], *, detected_language: str | None
) -> list[ReportSection]:
    rendered_sections: list[ReportSection] = []

    for section in sections:
        if not section.is_interlude:
            rendered_sections.append(section)
            continue

        rendered_sections.append(
            section.model_copy(
                update={
                    "title": _interlude_title(detected_language),
                    "transcript_text": _interlude_note(detected_language),
                    "is_interlude": True,
                }
            )
        )

    return rendered_sections


def _segments_excluding_interludes(
    segments: list[TranscriptSegment], sections: list[ReportSection]
) -> list[TranscriptSegment]:
    interlude_ranges = [
        (section.start_sec, section.end_sec) for section in sections if section.is_interlude
    ]
    if not interlude_ranges:
        return segments

    return [
        segment
        for segment in segments
        if not any(
            segment.start_sec < end_sec and segment.end_sec > start_sec
            for start_sec, end_sec in interlude_ranges
        )
    ]


def _detect_interlude_ranges(segments: list[TranscriptSegment]) -> list[tuple[float, float]]:
    meaningful_segments = [segment for segment in segments if segment.text.strip()]
    if not meaningful_segments:
        return []

    ranges: list[tuple[float, float]] = []
    current_run: list[TranscriptSegment] = []

    for segment in meaningful_segments:
        if not _is_likely_interlude_text(segment.text):
            _append_interlude_range(ranges, current_run)
            current_run = []
            continue

        if (
            current_run
            and (segment.start_sec - current_run[-1].end_sec) >= AUDIO_SECTION_BREAK_GAP_SEC
        ):
            _append_interlude_range(ranges, current_run)
            current_run = []
        current_run.append(segment)

    _append_interlude_range(ranges, current_run)
    return ranges


def _append_interlude_range(
    ranges: list[tuple[float, float]], segments: list[TranscriptSegment]
) -> None:
    if not segments:
        return
    if not any(_has_interlude_marker(segment.text) for segment in segments) and len(segments) < 2:
        return

    start_sec = segments[0].start_sec
    end_sec = segments[-1].end_sec
    ranges.append((start_sec, end_sec))


def _renderable_interlude_ranges(ranges: list[tuple[float, float]]) -> list[tuple[float, float]]:
    return [
        (start_sec, end_sec)
        for start_sec, end_sec in ranges
        if (end_sec - start_sec) >= MIN_INTERLUDE_DURATION_SEC
    ]


def _overlaps_interlude_ranges(
    segment: TranscriptSegment, interlude_ranges: list[tuple[float, float]]
) -> bool:
    return any(
        segment.start_sec < end_sec and segment.end_sec > start_sec
        for start_sec, end_sec in interlude_ranges
    )


def _is_likely_interlude_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if _has_interlude_marker(stripped):
        return True

    words = INTERLUDE_WORD_RE.findall(stripped.casefold())
    if len(words) < INTERLUDE_MIN_WORDS:
        return False

    sample_size = min(len(words), 80)
    sampled_words = words[:sample_size]
    unique_ratio = len(set(sampled_words)) / sample_size
    return unique_ratio <= INTERLUDE_LOW_UNIQUE_RATIO


def _has_interlude_marker(text: str) -> bool:
    return bool(INTERLUDE_MARKER_PATTERN.search(text))

def _interlude_title(detected_language: str | None) -> str:
    if detected_language == "ru":
        return "Музыкальная пауза"
    return "Music Interlude"


def _interlude_note(detected_language: str | None) -> str:
    if detected_language == "ru":
        return (
            "Музыкальная вставка или поэтический фрагмент. "
            "Исходная расшифровка сохранена в transcript.json."
        )
    return (
        "Music or spoken-performance interlude. The raw transcript is preserved in transcript.json."
    )
