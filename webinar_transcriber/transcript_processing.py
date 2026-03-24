"""Helpers for normalizing ASR transcript segments before downstream use."""

from __future__ import annotations

import re

from webinar_transcriber.models import TranscriptionResult, TranscriptSegment

MERGE_GAP_SEC = 0.8
TARGET_SEGMENT_DURATION_SEC = 5.0
MIN_SEGMENT_DURATION_SEC = 3.0
MAX_SEGMENT_DURATION_SEC = 15.0
MAX_SEGMENT_CHARS = 640
STRONG_SENTENCE_END_RE = re.compile(r"[.!?]$")


def normalize_transcription(transcription: TranscriptionResult) -> TranscriptionResult:
    """Drop empty segments and merge adjacent short segments into larger utterances."""
    meaningful_segments = [segment for segment in transcription.segments if segment.text.strip()]
    if not meaningful_segments:
        return TranscriptionResult(detected_language=transcription.detected_language, segments=[])

    merged_segments: list[TranscriptSegment] = []
    current_segments: list[TranscriptSegment] = []

    for segment in meaningful_segments:
        if _should_flush_before_adding(current_segments, segment):
            merged_segments.append(_merge_segment_group(current_segments, len(merged_segments) + 1))
            current_segments = []
        current_segments.append(segment)

    if current_segments:
        merged_segments.append(_merge_segment_group(current_segments, len(merged_segments) + 1))

    return TranscriptionResult(
        detected_language=transcription.detected_language,
        segments=merged_segments,
    )


def _should_flush_before_adding(
    current_segments: list[TranscriptSegment],
    next_segment: TranscriptSegment,
) -> bool:
    if not current_segments:
        return False

    current_start = current_segments[0].start_sec
    current_end = current_segments[-1].end_sec
    current_duration = max(0.0, current_end - current_start)
    next_duration = max(0.0, next_segment.end_sec - current_start)
    gap_duration = max(0.0, next_segment.start_sec - current_end)
    current_text = _merge_text(current_segments)

    if gap_duration > MERGE_GAP_SEC:
        return True
    if current_duration >= MAX_SEGMENT_DURATION_SEC:
        return True
    if len(current_text) >= MAX_SEGMENT_CHARS:
        return True
    return bool(
        current_duration >= MIN_SEGMENT_DURATION_SEC
        and next_duration > TARGET_SEGMENT_DURATION_SEC
        and STRONG_SENTENCE_END_RE.search(current_text)
    )


def _merge_segment_group(
    segments: list[TranscriptSegment],
    merged_index: int,
) -> TranscriptSegment:
    return TranscriptSegment(
        id=f"segment-{merged_index}",
        text=_merge_text(segments),
        start_sec=segments[0].start_sec,
        end_sec=segments[-1].end_sec,
    )


def _merge_text(segments: list[TranscriptSegment]) -> str:
    return " ".join(segment.text.strip() for segment in segments if segment.text.strip())
