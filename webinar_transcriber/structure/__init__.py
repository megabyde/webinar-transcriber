"""Deterministic report structuring helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from webinar_transcriber.models import (
    AlignmentBlock,
    MediaAsset,
    ReportDocument,
    TranscriptionResult,
)

from .scoring import _derive_title
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


__all__ = ["build_report"]
