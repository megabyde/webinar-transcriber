"""Deterministic transcript alignment and report structuring helpers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from webinar_transcriber.models import (
    AlignmentBlock,
    MediaAsset,
    ReportDocument,
    ReportSection,
    Scene,
    SlideFrame,
    TranscriptionResult,
    TranscriptSegment,
)

if TYPE_CHECKING:
    from collections.abc import Callable

AUDIO_SECTION_BREAK_GAP_SEC = 5.0
TITLE_WORD_LIMIT = 6


def align_by_time(
    transcript_segments: list[TranscriptSegment],
    scenes: list[Scene],
    slide_frames: list[SlideFrame],
) -> list[AlignmentBlock]:
    """Assign transcript segments to scenes using midpoint inclusion.

    Returns:
        list[AlignmentBlock]: The scene-aligned transcript blocks.
    """
    frame_by_scene = {frame.scene_id: frame for frame in slide_frames}
    blocks: list[AlignmentBlock] = []

    for index, scene in enumerate(scenes, start=1):
        scene_segments = [
            segment
            for segment in transcript_segments
            if scene.start_sec <= segment.midpoint < scene.end_sec
        ]
        transcript_text = " ".join(segment.text for segment in scene_segments).strip()
        frame = frame_by_scene.get(scene.id)
        blocks.append(
            AlignmentBlock(
                id=f"block-{index}",
                start_sec=scene.start_sec,
                end_sec=scene.end_sec,
                transcript_segment_ids=[segment.id for segment in scene_segments],
                transcript_text=transcript_text,
                scene_id=scene.id,
                frame_id=frame.id if frame else None,
            )
        )

    return blocks


def build_report(
    media_asset: MediaAsset,
    transcription: TranscriptionResult,
    *,
    alignment_blocks: list[AlignmentBlock] | None = None,
    warnings: list[str] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> ReportDocument:
    """Build a report document from media metadata and transcript segments.

    Returns:
        ReportDocument: The structured report document.
    """
    transcript_segments = transcription.segments
    if alignment_blocks is not None:
        sections = _build_sections_from_blocks(
            alignment_blocks,
            transcript_segments=transcript_segments,
            progress_callback=progress_callback,
        )
    else:
        sections = _build_audio_sections(transcript_segments, progress_callback=progress_callback)
    return ReportDocument(
        title=_derive_title(media_asset.path),
        source_file=media_asset.path,
        media_type=media_asset.media_type,
        detected_language=transcription.detected_language,
        summary=[],
        action_items=[],
        sections=sections,
        warnings=warnings or [],
    )


def _build_sections_from_blocks(
    blocks: list[AlignmentBlock],
    *,
    transcript_segments: list[TranscriptSegment],
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
                block, block_segments=block_segments, next_section_index=len(sections) + 1
            )
        )
        if progress_callback is not None:
            progress_callback(index, len(sections))

    return sections


def _build_audio_sections(
    segments: list[TranscriptSegment],
    *,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[ReportSection]:
    meaningful_segments = [seg for seg in segments if seg.text.strip()]
    if not meaningful_segments:
        return []

    sections: list[ReportSection] = []
    current_segments: list[TranscriptSegment] = []

    for index, segment in enumerate(meaningful_segments, start=1):
        if _should_start_new_audio_section(current_segments, segment):
            sections.append(_audio_section_from_segments(current_segments, len(sections) + 1))
            current_segments = []
        current_segments.append(segment)
        if progress_callback is not None:
            progress_callback(index, len(sections) + 1)

    if current_segments:
        sections.append(_audio_section_from_segments(current_segments, len(sections) + 1))

    return sections


def _should_start_new_audio_section(
    current_segments: list[TranscriptSegment], next_segment: TranscriptSegment
) -> bool:
    if not current_segments:
        return False

    current_end = current_segments[-1].end_sec
    gap_duration = max(0.0, next_segment.start_sec - current_end)
    return gap_duration >= AUDIO_SECTION_BREAK_GAP_SEC


def _audio_section_from_segments(
    segments: list[TranscriptSegment], section_index: int
) -> ReportSection:
    title = _first_words_title(segments, fallback=f"Section {section_index}")
    return _section_from_segments(segments, section_index=section_index, title=title)


def _first_words_title(
    segments: list[TranscriptSegment], *, fallback: str, word_limit: int = TITLE_WORD_LIMIT
) -> str:
    text = next((segment.text.strip() for segment in segments if segment.text.strip()), "")
    if not text:
        return fallback
    words = text.split()
    return " ".join(words[:word_limit]) + ("…" if len(words) > word_limit else "")


def _section_from_segments(
    segments: list[TranscriptSegment],
    *,
    section_index: int,
    title: str,
    frame_id: str | None = None,
) -> ReportSection:
    transcript_text = "\n\n".join(s for seg in segments if (s := seg.text.strip()))
    return ReportSection(
        id=f"section-{section_index}",
        title=title,
        start_sec=segments[0].start_sec,
        end_sec=segments[-1].end_sec,
        transcript_text=transcript_text,
        frame_id=frame_id,
    )


def _sections_from_block(
    block: AlignmentBlock, *, block_segments: list[TranscriptSegment], next_section_index: int
) -> list[ReportSection]:
    if not block_segments:
        # Keep slide-backed sections even when no transcript segments aligned, so frame/title
        # context is preserved for scene-only blocks.
        title = _title_from_text(block.transcript_text, fallback=f"Slide {next_section_index}")
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

    title = _title_from_text(block.transcript_text, fallback=f"Slide {next_section_index}")
    return [
        _section_from_segments(
            block_segments, section_index=next_section_index, title=title, frame_id=block.frame_id
        )
    ]


def _title_from_text(text: str, *, fallback: str) -> str:
    cleaned = text.strip().rstrip(".")
    if not cleaned:
        return fallback

    words = cleaned.split()
    return " ".join(words[:TITLE_WORD_LIMIT]) if len(words) > TITLE_WORD_LIMIT else cleaned


def _derive_title(source_path: str) -> str:
    stem = Path(source_path).stem
    return stem.replace("-", " ").replace("_", " ").strip().title() or "Transcription Report"


__all__ = ["align_by_time", "build_report"]
