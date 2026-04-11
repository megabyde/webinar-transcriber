"""Deterministic report structuring helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from webinar_transcriber.models import (
    AlignmentBlock,
    MediaAsset,
    ReportDocument,
    TranscriptionResult,
)

from .interludes import (
    _detect_interlude_ranges,
    _is_likely_interlude_text,
    _render_interlude_sections,
    _renderable_interlude_ranges,
    _segments_excluding_interludes,
)
from .scoring import (
    _action_item_score,
    _audio_title_from_segments,
    _audio_title_score,
    _build_summary,
    _derive_title,
    _extract_action_items,
    _fallback_summary,
    _segment_key,
    _summary_filler_penalty,
    _summary_repetition_penalty,
    _summary_start_penalty,
    _title_from_text,
    _title_from_words,
)
from .sections import (
    _build_audio_sections,
    _build_sections_from_blocks,
    _should_start_new_audio_section,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def build_report(
    media_asset: MediaAsset,
    transcription: TranscriptionResult,
    *,
    alignment_blocks: list[AlignmentBlock] | None = None,
    warnings: list[str] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> ReportDocument:
    """Build a report document from media metadata and transcript segments."""
    interlude_candidate_ranges = _detect_interlude_ranges(transcription.segments)
    interlude_ranges = _renderable_interlude_ranges(interlude_candidate_ranges)
    sections = (
        _build_sections_from_blocks(
            alignment_blocks,
            transcript_segments=transcription.segments,
            interlude_ranges=interlude_ranges,
            progress_callback=progress_callback,
        )
        if alignment_blocks is not None
        else _build_audio_sections(
            transcription.segments,
            interlude_ranges=interlude_ranges,
            progress_callback=progress_callback,
        )
    )
    sections = _render_interlude_sections(
        sections,
        detected_language=transcription.detected_language,
    )
    report_segments = _segments_excluding_interludes(transcription.segments, sections)
    summary = _build_summary(report_segments)
    action_items = _extract_action_items(report_segments)

    return ReportDocument(
        title=_derive_title(media_asset.path),
        source_file=media_asset.path,
        media_type=media_asset.media_type,
        detected_language=transcription.detected_language,
        summary=summary,
        action_items=action_items,
        sections=sections,
        warnings=warnings or [],
    )


__all__ = [
    "_action_item_score",
    "_audio_title_from_segments",
    "_audio_title_score",
    "_build_audio_sections",
    "_build_summary",
    "_derive_title",
    "_extract_action_items",
    "_fallback_summary",
    "_is_likely_interlude_text",
    "_segment_key",
    "_should_start_new_audio_section",
    "_summary_filler_penalty",
    "_summary_repetition_penalty",
    "_summary_start_penalty",
    "_title_from_text",
    "_title_from_words",
    "build_report",
]
