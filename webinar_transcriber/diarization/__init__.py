"""Local speaker-diarization helpers."""

from __future__ import annotations

from dataclasses import replace

from webinar_transcriber.models import SpeakerTurn, TranscriptSegment

from .contracts import DiarizationProcessingError, Diarizer
from .sherpa_diarizer import DIARIZATION_MODEL, SherpaOnnxDiarizer, normalize_speaker_labels

__all__ = [
    "DIARIZATION_MODEL",
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
    candidates: list[SpeakerTurn] = []
    for turn in turns:
        if turn.end_sec <= segment.start_sec:
            continue
        if turn.start_sec >= segment.end_sec:
            break
        if _overlap_sec(segment, turn) > 0:
            candidates.append(turn)
    return max(
        candidates,
        key=lambda turn: (_overlap_sec(segment, turn), -_midpoint_distance_sec(segment, turn)),
        default=None,
    )


def _overlap_sec(segment: TranscriptSegment, turn: SpeakerTurn) -> float:
    return min(segment.end_sec, turn.end_sec) - max(segment.start_sec, turn.start_sec)


def _midpoint_distance_sec(segment: TranscriptSegment, turn: SpeakerTurn) -> float:
    return abs(segment.midpoint - turn.midpoint)
