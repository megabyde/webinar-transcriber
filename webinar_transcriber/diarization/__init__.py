"""Local speaker-diarization helpers."""

from __future__ import annotations

from dataclasses import replace
from itertools import islice
from typing import TYPE_CHECKING

from .sherpa_diarizer import (
    DIARIZATION_MODEL,
    DiarizationProcessingError,
    SherpaOnnxDiarizer,
    normalize_speaker_labels,
)

if TYPE_CHECKING:
    from webinar_transcriber.models import SpeakerTurn, TranscriptSegment


__all__ = [
    "DIARIZATION_MODEL",
    "DiarizationProcessingError",
    "SherpaOnnxDiarizer",
    "assign_speakers",
    "normalize_speaker_labels",
]


def assign_speakers(
    segments: list[TranscriptSegment], turns: list[SpeakerTurn]
) -> list[TranscriptSegment]:
    """Return copies of transcript segments labeled by maximum speaker-turn overlap."""
    ordered_turns = sorted(turns, key=lambda turn: (turn.start_sec, turn.end_sec))
    indexed_segments = sorted(
        enumerate(segments), key=lambda item: (item[1].start_sec, item[1].end_sec)
    )
    assigned = list(segments)
    first_relevant_turn_index = 0
    for segment_index, segment in indexed_segments:
        while (
            first_relevant_turn_index < len(ordered_turns)
            and ordered_turns[first_relevant_turn_index].end_sec <= segment.start_sec
        ):
            first_relevant_turn_index += 1

        turn = _best_turn(segment, ordered_turns, start_index=first_relevant_turn_index)
        assigned[segment_index] = replace(
            segment, speaker=turn.speaker if turn is not None else None
        )
    return assigned


def _best_turn(
    segment: TranscriptSegment, turns: list[SpeakerTurn], *, start_index: int
) -> SpeakerTurn | None:
    best_turn: SpeakerTurn | None = None
    best_key: tuple[float, float] | None = None
    for turn in islice(turns, start_index, None):
        if turn.end_sec <= segment.start_sec:
            continue
        if turn.start_sec >= segment.end_sec:
            break
        overlap_sec = _overlap_sec(segment, turn)
        if overlap_sec <= 0:
            continue
        key = (overlap_sec, -_midpoint_distance_sec(segment, turn))
        if best_key is None or key > best_key:
            best_turn = turn
            best_key = key
    return best_turn


def _overlap_sec(segment: TranscriptSegment, turn: SpeakerTurn) -> float:
    return min(segment.end_sec, turn.end_sec) - max(segment.start_sec, turn.start_sec)


def _midpoint_distance_sec(segment: TranscriptSegment, turn: SpeakerTurn) -> float:
    return abs(segment.midpoint - turn.midpoint)
