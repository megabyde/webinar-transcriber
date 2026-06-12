"""Deterministic transcript alignment and report structuring helpers."""

from __future__ import annotations

from dataclasses import replace
from itertools import groupby
from pathlib import Path

from webinar_transcriber.models import (
    MediaAsset,
    ReportDocument,
    ReportSection,
    Scene,
    SceneFrame,
    TranscriptionResult,
    TranscriptSegment,
    VideoAsset,
)

AUDIO_SECTION_BREAK_GAP_SEC = 5.0
TITLE_WORD_LIMIT = 6


def build_report(
    media_asset: MediaAsset,
    transcription: TranscriptionResult,
    *,
    scenes: list[Scene] | None = None,
    scene_frames: list[SceneFrame] | None = None,
    warnings: list[str] | None = None,
) -> ReportDocument:
    """Build a report document from media metadata and transcript segments.

    Returns:
        ReportDocument: The structured report document.
    """
    if isinstance(media_asset, VideoAsset):
        sections = build_video_sections(transcription.segments, scenes or [], scene_frames or [])
    else:
        sections = build_audio_sections(transcription.segments)
    return ReportDocument(
        title=title_from_path(media_asset.path),
        source_file=media_asset.path,
        media_type=media_asset.media_type,
        detected_language=transcription.detected_language,
        summary=[],
        action_items=[],
        sections=sections,
        warnings=warnings or [],
    )


def build_video_sections(
    segments: list[TranscriptSegment],
    scenes: list[Scene],
    scene_frames: list[SceneFrame],
) -> list[ReportSection]:
    """Build report sections by assigning transcript segments to scenes via midpoint inclusion.

    Returns:
        list[ReportSection]: One scene-aligned section per scene, in timeline order.
    """
    frame_by_scene = {frame.scene_id: frame for frame in scene_frames}
    sections: list[ReportSection] = []

    for scene in scenes:
        scene_segments = [
            segment for segment in segments if scene.start_sec <= segment.midpoint < scene.end_sec
        ]
        transcript_text = " ".join(segment.text for segment in scene_segments).strip()
        meaningful_segments = [segment for segment in scene_segments if segment.text.strip()]
        frame = frame_by_scene.get(scene.id)
        title = derive_title(transcript_text, fallback="")
        if meaningful_segments:
            sections.append(
                _draft_section(
                    meaningful_segments,
                    title=title,
                    scene_id=scene.id,
                    frame_id=frame.id if frame else None,
                )
            )
            continue
        # Keep scene-backed sections even when no transcript segments aligned, so frame/title
        # context is preserved for scene-only blocks.
        sections.append(
            ReportSection(
                id="",
                title=title,
                start_sec=scene.start_sec,
                end_sec=scene.end_sec,
                transcript_text=transcript_text,
                scene_id=scene.id,
                frame_id=frame.id if frame else None,
            )
        )

    return _number_sections(sections, fallback_title_prefix="Slide")


def build_audio_sections(segments: list[TranscriptSegment]) -> list[ReportSection]:
    """Build best-effort audio-only report sections from transcript timing gaps."""
    meaningful_segments = [seg for seg in segments if seg.text.strip()]
    if not meaningful_segments:
        return []

    sections: list[ReportSection] = []
    current_segments: list[TranscriptSegment] = []

    for segment in meaningful_segments:
        if should_start_new_audio_section(current_segments, segment):
            sections.append(_audio_draft_section(current_segments))
            current_segments = []
        current_segments.append(segment)

    if current_segments:
        sections.append(_audio_draft_section(current_segments))

    return _number_sections(sections, fallback_title_prefix="Section")


def should_start_new_audio_section(
    current_segments: list[TranscriptSegment], next_segment: TranscriptSegment
) -> bool:
    """Return whether a timing gap should start a new audio-only section."""
    if not current_segments:
        return False

    current_end = current_segments[-1].end_sec
    gap_duration = max(0.0, next_segment.start_sec - current_end)
    return gap_duration >= AUDIO_SECTION_BREAK_GAP_SEC


def _audio_draft_section(segments: list[TranscriptSegment]) -> ReportSection:
    text = next((segment.text for segment in segments if segment.text.strip()), "")
    title = derive_title(text, fallback="", ellipsis=True)
    return _draft_section(segments, title=title)


def _draft_section(
    segments: list[TranscriptSegment],
    *,
    title: str,
    scene_id: str | None = None,
    frame_id: str | None = None,
) -> ReportSection:
    return ReportSection(
        id="",
        title=title,
        start_sec=segments[0].start_sec,
        end_sec=segments[-1].end_sec,
        transcript_text=_transcript_text_from_segments(segments),
        scene_id=scene_id,
        frame_id=frame_id,
    )


def _number_sections(
    sections: list[ReportSection], *, fallback_title_prefix: str
) -> list[ReportSection]:
    """Assign positional ids and fallback titles to draft sections in one pass."""
    return [
        replace(
            section,
            id=f"section-{index}",
            title=section.title or f"{fallback_title_prefix} {index}",
        )
        for index, section in enumerate(sections, start=1)
    ]


def derive_title(
    text: str, *, fallback: str = "Transcription Report", ellipsis: bool = False
) -> str:
    """Return a bounded title from transcript text."""
    cleaned = text.strip().rstrip(".")
    if not cleaned:
        return fallback

    words = cleaned.split()
    title = " ".join(words[:TITLE_WORD_LIMIT])
    return f"{title}…" if ellipsis and len(words) > TITLE_WORD_LIMIT else title


def title_from_path(source_path: str) -> str:
    """Return a report title derived from the source filename."""
    stem = Path(source_path).stem
    return stem.replace("-", " ").replace("_", " ").strip().title() or "Transcription Report"


def _transcript_text_from_segments(segments: list[TranscriptSegment]) -> str:
    meaningful_segments = [segment for segment in segments if segment.text.strip()]
    if not any(segment.speaker for segment in meaningful_segments):
        return "\n\n".join(segment.text.strip() for segment in meaningful_segments)

    grouped_segments = groupby(meaningful_segments, key=lambda segment: segment.speaker)
    return "\n\n".join(
        _speaker_paragraph(speaker, [segment.text.strip() for segment in speaker_segments])
        for speaker, speaker_segments in grouped_segments
    )


def _speaker_paragraph(speaker: str | None, texts: list[str]) -> str:
    text = " ".join(texts).strip()
    return f"**{speaker}:** {text}" if speaker else text


__all__ = [
    "build_audio_sections",
    "build_report",
    "build_video_sections",
    "derive_title",
    "should_start_new_audio_section",
    "title_from_path",
]
