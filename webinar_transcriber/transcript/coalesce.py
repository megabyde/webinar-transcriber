"""Coalesce raw ASR segments into readable, speaker-consistent blocks."""

from __future__ import annotations

import re

from webinar_transcriber.models import TranscriptionResult, TranscriptSegment
from webinar_transcriber.text_utils import SENTENCE_TERMINATORS

MERGE_GAP_SEC = 0.6
TARGET_SEGMENT_DURATION_SEC = 4.0
MIN_SEGMENT_DURATION_SEC = 2.5
MAX_SEGMENT_DURATION_SEC = 10.0
MAX_SEGMENT_CHARS = 420
STRONG_SENTENCE_END_RE = re.compile(f"[{re.escape(SENTENCE_TERMINATORS)}]$")


def coalesce_transcript(transcription: TranscriptionResult) -> TranscriptionResult:
    """Merge adjacent segments into readable blocks, splitting on speaker changes.

    Short ASR fragments are merged into utterance-sized blocks; a new block starts on a speaker
    change, a timing gap, or once a block grows past its duration/length/sentence-end bounds.
    Speaker labels only refine the boundaries: unlabeled transcripts are still coalesced into
    readable blocks.

    Returns:
        TranscriptionResult: The coalesced transcription.
    """
    meaningful_segments = [segment for segment in transcription.segments if segment.text.strip()]
    if not meaningful_segments:
        return TranscriptionResult(detected_language=transcription.detected_language, segments=[])

    merged_segments: list[TranscriptSegment] = []
    current_segments: list[TranscriptSegment] = [meaningful_segments[0]]

    for segment in meaningful_segments[1:]:
        if _should_flush_before_adding(current_segments, segment):
            merged_segments.append(_merge_segment_group(current_segments, len(merged_segments) + 1))
            current_segments = []
        current_segments.append(segment)

    merged_segments.append(_merge_segment_group(current_segments, len(merged_segments) + 1))

    return TranscriptionResult(
        detected_language=transcription.detected_language, segments=merged_segments
    )


def _should_flush_before_adding(
    current_segments: list[TranscriptSegment], next_segment: TranscriptSegment
) -> bool:
    current_start = current_segments[0].start_sec
    current_end = current_segments[-1].end_sec
    current_duration = max(0.0, current_end - current_start)
    projected_group_duration = max(0.0, next_segment.end_sec - current_start)
    gap_duration = max(0.0, next_segment.start_sec - current_end)
    current_text = _merge_text(current_segments)

    if current_segments[-1].speaker != next_segment.speaker:
        return True
    if gap_duration > MERGE_GAP_SEC:
        return True
    if current_duration >= MAX_SEGMENT_DURATION_SEC:
        return True
    if len(current_text) >= MAX_SEGMENT_CHARS:
        return True
    return (
        current_duration >= MIN_SEGMENT_DURATION_SEC
        and projected_group_duration > TARGET_SEGMENT_DURATION_SEC
        and STRONG_SENTENCE_END_RE.search(current_text) is not None
    )


def _merge_segment_group(segments: list[TranscriptSegment], merged_index: int) -> TranscriptSegment:
    return TranscriptSegment(
        id=f"segment-{merged_index}",
        text=_merge_text(segments),
        start_sec=segments[0].start_sec,
        end_sec=segments[-1].end_sec,
        speaker=segments[0].speaker,
    )


def _merge_text(segments: list[TranscriptSegment]) -> str:
    return " ".join(s for seg in segments if (s := seg.text.strip()))
