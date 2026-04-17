"""Deterministic report structuring helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from webinar_transcriber.models import (
    AlignmentBlock,
    MediaAsset,
    ReportDocument,
    TranscriptionResult,
)

from .scoring import _build_summary, _derive_title, _extract_action_items
from .sections import _build_audio_sections, _build_sections_from_blocks

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
    """Build a report document from media metadata and transcript segments.

    Returns:
        ReportDocument: The structured report document.
    """
    sections = (
        _build_sections_from_blocks(
            alignment_blocks,
            transcript_segments=transcription.segments,
            progress_callback=progress_callback,
        )
        if alignment_blocks is not None
        else _build_audio_sections(
            transcription.segments,
            progress_callback=progress_callback,
        )
    )
    summary = _build_summary(transcription.segments)
    action_items = _extract_action_items(transcription.segments)

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


__all__ = ["build_report"]
