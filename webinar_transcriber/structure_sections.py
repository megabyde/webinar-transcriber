"""Section-building helpers for deterministic report structuring."""

from __future__ import annotations

from typing import TYPE_CHECKING

from webinar_transcriber.models import ReportSection
from webinar_transcriber.structure_constants import (
    AUDIO_SECTION_BREAK_GAP_SEC,
    MAX_AUDIO_SECTION_CHARS,
    MIN_AUDIO_SECTION_DURATION_SEC,
    TARGET_AUDIO_SECTION_DURATION_SEC,
)
from webinar_transcriber.structure_interludes import _overlaps_interlude_ranges
from webinar_transcriber.structure_scoring import (
    _audio_title_from_segments,
    _title_from_text,
)
from webinar_transcriber.transcript_processing import STRONG_SENTENCE_END_RE

if TYPE_CHECKING:
    from collections.abc import Callable

    from webinar_transcriber.models import AlignmentBlock, TranscriptSegment


def _build_sections_from_blocks(
    blocks: list[AlignmentBlock],
    *,
    transcript_segments: list[TranscriptSegment],
    interlude_ranges: list[tuple[float, float]],
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[ReportSection]:
    segment_by_id = {segment.id: segment for segment in transcript_segments}
    sections: list[ReportSection] = []

    for index, block in enumerate(blocks, start=1):
        block_segments = [
            segment_by_id[segment_id]
            for segment_id in block.transcript_segment_ids
            if segment_id in segment_by_id and segment_by_id[segment_id].text.strip()
        ]
        sections.extend(
            _sections_from_block(
                block,
                block_segments=block_segments,
                interlude_ranges=interlude_ranges,
                next_section_index=len(sections) + 1,
            )
        )
        if progress_callback is not None:
            progress_callback(index, len(sections))

    return sections


def _build_audio_sections(
    segments: list[TranscriptSegment],
    *,
    interlude_ranges: list[tuple[float, float]] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[ReportSection]:
    resolved_interlude_ranges = interlude_ranges or []
    meaningful_segments = [seg for seg in segments if seg.text.strip()]
    if not meaningful_segments:
        return []

    sections: list[ReportSection] = []
    current_speech_segments: list[TranscriptSegment] = []
    current_interlude_segments: list[TranscriptSegment] = []

    for index, segment in enumerate(meaningful_segments, start=1):
        if _overlaps_interlude_ranges(segment, resolved_interlude_ranges):
            if current_speech_segments:
                sections.append(
                    _audio_section_from_segments(
                        current_speech_segments,
                        len(sections) + 1,
                    )
                )
                current_speech_segments = []
            current_interlude_segments.append(segment)
        else:
            if current_interlude_segments:
                sections.append(
                    _interlude_section_from_segments(
                        current_interlude_segments,
                        len(sections) + 1,
                    )
                )
                current_interlude_segments = []
            if _should_start_new_audio_section(current_speech_segments, segment):
                sections.append(
                    _audio_section_from_segments(
                        current_speech_segments,
                        len(sections) + 1,
                    )
                )
                current_speech_segments = []
            current_speech_segments.append(segment)
        if progress_callback is not None:
            progress_callback(index, len(sections) + 1)

    if current_speech_segments:
        sections.append(_audio_section_from_segments(current_speech_segments, len(sections) + 1))
    if current_interlude_segments:
        sections.append(
            _interlude_section_from_segments(
                current_interlude_segments,
                len(sections) + 1,
            )
        )

    return sections


def _should_start_new_audio_section(
    current_segments: list[TranscriptSegment], next_segment: TranscriptSegment
) -> bool:
    if not current_segments:
        return False

    current_start = current_segments[0].start_sec
    current_end = current_segments[-1].end_sec
    current_duration = current_end - current_start
    gap_duration = max(0.0, next_segment.start_sec - current_end)
    projected_duration = max(0.0, next_segment.end_sec - current_start)
    current_chars = sum(len(segment.text.strip()) for segment in current_segments)
    ends_on_sentence_boundary = bool(
        STRONG_SENTENCE_END_RE.search(current_segments[-1].text.strip())
    )
    hard_cap_exceeded = projected_duration > (2 * TARGET_AUDIO_SECTION_DURATION_SEC)

    if gap_duration >= AUDIO_SECTION_BREAK_GAP_SEC:
        return True

    if current_duration < MIN_AUDIO_SECTION_DURATION_SEC:
        return False

    split_budget_reached = (
        projected_duration > TARGET_AUDIO_SECTION_DURATION_SEC
        or current_chars >= MAX_AUDIO_SECTION_CHARS
    )
    if not split_budget_reached:
        return False

    return hard_cap_exceeded or ends_on_sentence_boundary


def _audio_section_from_segments(
    segments: list[TranscriptSegment], section_index: int
) -> ReportSection:
    title = _audio_title_from_segments(segments, fallback=f"Section {section_index}")
    return _section_from_segments(
        segments,
        section_index=section_index,
        title=title,
    )


def _interlude_section_from_segments(
    segments: list[TranscriptSegment],
    section_index: int,
    *,
    frame_id: str | None = None,
) -> ReportSection:
    title = f"Section {section_index}"
    return _section_from_segments(
        segments,
        section_index=section_index,
        title=title,
        frame_id=frame_id,
        is_interlude=True,
    )


def _section_from_segments(
    segments: list[TranscriptSegment],
    *,
    section_index: int,
    title: str,
    frame_id: str | None = None,
    is_interlude: bool = False,
) -> ReportSection:
    transcript_text = "\n\n".join(s for seg in segments if (s := seg.text.strip()))
    return ReportSection(
        id=f"section-{section_index}",
        title=title,
        start_sec=segments[0].start_sec,
        end_sec=segments[-1].end_sec,
        transcript_text=transcript_text,
        frame_id=frame_id,
        is_interlude=is_interlude,
    )


def _sections_from_block(
    block: AlignmentBlock,
    *,
    block_segments: list[TranscriptSegment],
    interlude_ranges: list[tuple[float, float]],
    next_section_index: int,
) -> list[ReportSection]:
    if not block_segments:
        title = _title_from_text(block.transcript_text, fallback=f"Slide {next_section_index}")
        if block.title_hint:
            title = _title_from_text(block.title_hint, fallback=title)
        return [
            ReportSection(
                id=f"section-{next_section_index}",
                title=title,
                start_sec=block.start_sec,
                end_sec=block.end_sec,
                transcript_text=block.transcript_text,
                frame_id=block.frame_id,
            )
        ]

    runs: list[tuple[bool, list[TranscriptSegment]]] = []
    current_run: list[TranscriptSegment] = []
    current_is_interlude = _overlaps_interlude_ranges(block_segments[0], interlude_ranges)

    for segment in block_segments:
        segment_is_interlude = _overlaps_interlude_ranges(segment, interlude_ranges)
        if current_run and segment_is_interlude != current_is_interlude:
            runs.append((current_is_interlude, current_run))
            current_run = []
        current_run.append(segment)
        current_is_interlude = segment_is_interlude

    if current_run:
        runs.append((current_is_interlude, current_run))

    sections: list[ReportSection] = []
    for offset, (is_interlude, run_segments) in enumerate(runs):
        section_index = next_section_index + offset
        if is_interlude:
            sections.append(
                _interlude_section_from_segments(
                    run_segments,
                    section_index,
                    frame_id=block.frame_id,
                )
            )
            continue

        run_text = " ".join(segment.text for segment in run_segments)
        title = _title_from_text(
            run_text,
            fallback=f"Slide {section_index}",
        )
        if block.title_hint:
            title = _title_from_text(block.title_hint, fallback=title)
        sections.append(
            _section_from_segments(
                run_segments,
                section_index=section_index,
                title=title,
                frame_id=block.frame_id,
            )
        )

    return sections
