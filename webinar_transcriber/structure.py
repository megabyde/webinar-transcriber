"""Deterministic report structuring helpers."""

import re

from webinar_transcriber.models import (
    AlignmentBlock,
    MediaAsset,
    ReportDocument,
    ReportSection,
    TranscriptionResult,
    TranscriptSegment,
)

ACTION_ITEM_PATTERN = re.compile(
    r"\b(action item|follow up|follow-up|next step|todo|we should|please)\b",
    re.IGNORECASE,
)


def build_report(
    media_asset: MediaAsset,
    transcription: TranscriptionResult,
    *,
    ocr_enabled: bool,
    alignment_blocks: list[AlignmentBlock] | None = None,
    warnings: list[str] | None = None,
) -> ReportDocument:
    """Build a report document from media metadata and transcript segments."""
    sections = (
        _build_sections_from_blocks(alignment_blocks)
        if alignment_blocks is not None
        else _build_sections(transcription.segments)
    )
    summary = _build_summary(transcription.segments)
    action_items = _extract_action_items(transcription.segments)

    return ReportDocument(
        title=_derive_title(media_asset.path),
        source_file=media_asset.path,
        media_type=media_asset.media_type,
        detected_language=transcription.detected_language,
        ocr_enabled=ocr_enabled,
        summary=summary,
        action_items=action_items,
        sections=sections,
        warnings=warnings or [],
    )


def _build_sections_from_blocks(blocks: list[AlignmentBlock]) -> list[ReportSection]:
    sections: list[ReportSection] = []

    for index, block in enumerate(blocks, start=1):
        title = _title_from_text(block.transcript_text, fallback=f"Slide {index}")
        if block.title_hint:
            title = _title_from_text(block.title_hint, fallback=title)
        sections.append(
            ReportSection(
                id=f"section-{index}",
                title=title,
                start_sec=block.start_sec,
                end_sec=block.end_sec,
                transcript_text=block.transcript_text,
                frame_id=block.frame_id,
            )
        )

    return sections


def _build_sections(segments: list[TranscriptSegment]) -> list[ReportSection]:
    sections: list[ReportSection] = []

    for index, segment in enumerate(segments, start=1):
        title = f"Section {index}"
        if segment.text:
            title = _title_from_text(segment.text, fallback=title)

        sections.append(
            ReportSection(
                id=f"section-{index}",
                title=title,
                start_sec=segment.start_sec,
                end_sec=segment.end_sec,
                transcript_text=segment.text,
            )
        )

    return sections


def _build_summary(segments: list[TranscriptSegment]) -> list[str]:
    summary: list[str] = []

    for segment in segments:
        text = segment.text.strip()
        if text and text not in summary:
            summary.append(text)
        if len(summary) == 3:
            break

    return summary


def _extract_action_items(segments: list[TranscriptSegment]) -> list[str]:
    return [
        segment.text.strip() for segment in segments if ACTION_ITEM_PATTERN.search(segment.text)
    ]


def _title_from_text(text: str, *, fallback: str) -> str:
    cleaned = text.strip().rstrip(".")
    if not cleaned:
        return fallback

    words = cleaned.split()
    return " ".join(words[:6]) if len(words) > 6 else cleaned


def _derive_title(source_path: str) -> str:
    stem = source_path.rsplit("/", maxsplit=1)[-1].rsplit(".", maxsplit=1)[0]
    return stem.replace("-", " ").replace("_", " ").strip().title() or "Transcription Report"
