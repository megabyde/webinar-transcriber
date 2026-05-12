"""Local speaker-diarization helpers."""

from __future__ import annotations

from dataclasses import replace

from webinar_transcriber.models import SpeakerTurn, TranscriptSegment

from .contracts import DiarizationProcessingError, Diarizer
from .sherpa_diarizer import SherpaOnnxDiarizer, normalize_speaker_labels

__all__ = [
    "DiarizationProcessingError",
    "Diarizer",
    "SherpaOnnxDiarizer",
    "SpeakerTurn",
    "assign_speakers",
    "normalize_speaker_labels",
]


def assign_speakers(
    segments: list[TranscriptSegment],
    turns: list[SpeakerTurn],
) -> list[TranscriptSegment]:
    """Return copies of transcript segments labeled by maximum speaker-turn overlap."""
    ordered_turns = sorted(turns, key=lambda turn: (turn.start_sec, turn.end_sec))
    assigned: list[TranscriptSegment] = []
    for segment in segments:
        turn = _best_turn(segment, ordered_turns)
        assigned.append(replace(segment, speaker=turn.speaker if turn is not None else None))
    return assigned


def _best_turn(segment: TranscriptSegment, turns: list[SpeakerTurn]) -> SpeakerTurn | None:
    best_turn: SpeakerTurn | None = None
    best_overlap = 0.0
    best_midpoint_distance = float("inf")
    for turn in turns:
        if turn.end_sec <= segment.start_sec:
            continue
        if turn.start_sec >= segment.end_sec:
            break
        overlap = min(segment.end_sec, turn.end_sec) - max(segment.start_sec, turn.start_sec)
        if overlap <= 0:
            continue
        midpoint_distance = abs(segment.midpoint - ((turn.start_sec + turn.end_sec) / 2.0))
        if overlap > best_overlap or (
            overlap == best_overlap and midpoint_distance < best_midpoint_distance
        ):
            best_turn = turn
            best_overlap = overlap
            best_midpoint_distance = midpoint_distance
    return best_turn
