"""Tests for transcript normalization helpers."""

from webinar_transcriber.models import TranscriptionResult, TranscriptSegment
from webinar_transcriber.transcript_processing import (
    MAX_SEGMENT_CHARS,
    normalize_transcription,
)


def test_normalize_transcription_drops_empty_segments_and_merges_short_adjacent_segments() -> None:
    transcription = TranscriptionResult(
        detected_language="ru",
        segments=[
            TranscriptSegment(id="segment-1", text="  ", start_sec=0.0, end_sec=0.4),
            TranscriptSegment(id="segment-2", text="Привет", start_sec=0.4, end_sec=1.0),
            TranscriptSegment(id="segment-3", text="всем.", start_sec=1.0, end_sec=1.6),
            TranscriptSegment(id="segment-4", text="", start_sec=1.6, end_sec=1.8),
            TranscriptSegment(id="segment-5", text="Это новая часть.", start_sec=3.0, end_sec=6.0),
        ],
    )

    normalized = normalize_transcription(transcription)

    assert normalized.detected_language == "ru"
    segment_texts = [segment.text for segment in normalized.segments]

    assert segment_texts == ["Привет всем.", "Это новая часть."]
    assert normalized.segments[0].start_sec == 0.4
    assert normalized.segments[0].end_sec == 1.6
    assert normalized.segments[1].start_sec == 3.0
    assert normalized.segments[1].end_sec == 6.0


def test_normalize_transcription_keeps_sentence_when_gap_is_large() -> None:
    transcription = TranscriptionResult(
        segments=[
            TranscriptSegment(id="segment-1", text="Короткая фраза.", start_sec=0.0, end_sec=1.0),
            TranscriptSegment(id="segment-2", text="Следующая фраза.", start_sec=2.5, end_sec=3.5),
        ]
    )

    normalized = normalize_transcription(transcription)

    segment_texts = [segment.text for segment in normalized.segments]

    assert segment_texts == ["Короткая фраза.", "Следующая фраза."]


def test_normalize_transcription_returns_empty_when_all_segments_are_blank() -> None:
    transcription = TranscriptionResult(
        detected_language="en",
        segments=[
            TranscriptSegment(id="segment-1", text=" ", start_sec=0.0, end_sec=0.5),
            TranscriptSegment(id="segment-2", text="", start_sec=0.5, end_sec=1.0),
        ],
    )

    normalized = normalize_transcription(transcription)

    assert normalized.detected_language == "en"
    assert normalized.segments == []


def test_normalize_transcription_splits_when_current_segment_is_already_too_long() -> None:
    transcription = TranscriptionResult(
        segments=[
            TranscriptSegment(id="segment-1", text="Длинный блок", start_sec=0.0, end_sec=16.0),
            TranscriptSegment(id="segment-2", text="Следующий блок", start_sec=16.1, end_sec=17.0),
        ]
    )

    normalized = normalize_transcription(transcription)

    assert [segment.text for segment in normalized.segments] == [
        "Длинный блок",
        "Следующий блок",
    ]


def test_normalize_transcription_splits_when_text_reaches_character_limit() -> None:
    transcription = TranscriptionResult(
        segments=[
            TranscriptSegment(
                id="segment-1",
                text="x" * MAX_SEGMENT_CHARS,
                start_sec=0.0,
                end_sec=1.0,
            ),
            TranscriptSegment(id="segment-2", text="tail", start_sec=1.1, end_sec=2.0),
        ]
    )

    normalized = normalize_transcription(transcription)

    assert [segment.text for segment in normalized.segments] == ["x" * MAX_SEGMENT_CHARS, "tail"]
