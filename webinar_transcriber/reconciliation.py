"""Reconcile overlapping chunk transcripts into a single global transcript."""

from __future__ import annotations

import re
from dataclasses import dataclass

from webinar_transcriber.models import ChunkTranscription, TranscriptionResult, TranscriptSegment

_WHITESPACE_PATTERN = re.compile(r"\s+")
_NON_WORD_PATTERN = re.compile(r"[^\w]+", re.UNICODE)
_MIN_BOUNDARY_ECHO_SEC = 0.05


@dataclass(frozen=True)
class ReconciliationStats:
    """Observed cleanup performed during chunk transcript reconciliation."""

    duplicate_segments_dropped: int = 0
    boundary_fixes: int = 0


def reconcile_chunk_transcriptions(
    chunk_transcriptions: list[ChunkTranscription],
) -> tuple[TranscriptionResult, ReconciliationStats]:
    """Flatten chunk transcripts into one monotonic transcript."""
    accepted_segments: list[TranscriptSegment] = []
    dropped_duplicates = 0
    boundary_fixes = 0
    detected_language: str | None = None

    for chunk_transcription in sorted(chunk_transcriptions, key=lambda chunk: chunk.start_sec):
        if detected_language is None and chunk_transcription.detected_language:
            detected_language = chunk_transcription.detected_language

        for segment in chunk_transcription.segments:
            candidate = _candidate_segment(segment)
            if candidate is None:
                continue

            accepted, candidate, duplicate_drops, candidate_boundary_fixes = _accept_candidate(
                accepted_segments,
                candidate,
            )
            dropped_duplicates += duplicate_drops
            boundary_fixes += candidate_boundary_fixes
            if accepted and candidate is not None:
                accepted_segments.append(candidate)

    return (
        TranscriptionResult(detected_language=detected_language, segments=accepted_segments),
        ReconciliationStats(
            duplicate_segments_dropped=dropped_duplicates,
            boundary_fixes=boundary_fixes,
        ),
    )


def _candidate_segment(segment: TranscriptSegment) -> TranscriptSegment | None:
    cleaned_text = segment.text.strip()
    if not cleaned_text:
        return None

    return TranscriptSegment(
        id=segment.id,
        text=cleaned_text,
        start_sec=max(0.0, segment.start_sec),
        end_sec=max(segment.start_sec, segment.end_sec),
    )


def _accept_candidate(
    accepted_segments: list[TranscriptSegment],
    candidate: TranscriptSegment,
) -> tuple[bool, TranscriptSegment | None, int, int]:
    dropped_duplicates = 0
    boundary_fixes = 0

    if accepted_segments and _is_duplicate_segment(accepted_segments[-1], candidate):
        return False, None, 1, 0

    if not accepted_segments:
        return True, candidate, 0, 0

    previous = accepted_segments[-1]
    if candidate.start_sec < previous.end_sec:
        candidate = candidate.model_copy(update={"start_sec": previous.end_sec})
        boundary_fixes += 1
    if candidate.end_sec < candidate.start_sec:
        candidate = candidate.model_copy(update={"end_sec": candidate.start_sec})
        boundary_fixes += 1
    if candidate.end_sec <= candidate.start_sec:
        dropped_duplicates += 1
        return False, None, dropped_duplicates, boundary_fixes
    if _is_boundary_echo(previous, candidate):
        dropped_duplicates += 1
        return False, None, dropped_duplicates, boundary_fixes

    return True, candidate, dropped_duplicates, boundary_fixes


def _is_duplicate_segment(previous: TranscriptSegment, candidate: TranscriptSegment) -> bool:
    previous_text = _canonical_text(previous.text)
    candidate_text = _canonical_text(candidate.text)
    if not previous_text or not candidate_text:
        return False

    overlaps_in_time = (
        candidate.start_sec <= previous.end_sec and candidate.end_sec >= previous.start_sec
    )
    if previous_text == candidate_text and overlaps_in_time:
        return True

    if not overlaps_in_time:
        return False

    return previous_text in candidate_text or candidate_text in previous_text


def _is_boundary_echo(previous: TranscriptSegment, candidate: TranscriptSegment) -> bool:
    if (candidate.end_sec - candidate.start_sec) > _MIN_BOUNDARY_ECHO_SEC:
        return False

    previous_text = _canonical_text(previous.text)
    candidate_text = _canonical_text(candidate.text)
    if not previous_text or not candidate_text:
        return False

    return previous_text.endswith(candidate_text) or candidate_text in previous_text


def _canonical_text(text: str) -> str:
    normalized_text = _NON_WORD_PATTERN.sub(" ", text.casefold())
    return _WHITESPACE_PATTERN.sub(" ", normalized_text).strip()
