"""Tests for transcript coalescing."""

from webinar_transcriber.models import TranscriptionResult, TranscriptSegment
from webinar_transcriber.transcript.coalesce import MAX_SEGMENT_CHARS, coalesce_transcript


class TestCoalesceTranscript:
    def test_drops_empty_segments_and_merges_short_adjacent_segments(self) -> None:
        transcription = TranscriptionResult(
            detected_language="ru",
            segments=[
                TranscriptSegment(id="segment-1", text="  ", start_sec=0.0, end_sec=0.4),
                TranscriptSegment(id="segment-2", text="Привет", start_sec=0.4, end_sec=1.0),
                TranscriptSegment(id="segment-3", text="всем.", start_sec=1.0, end_sec=1.6),
                TranscriptSegment(id="segment-4", text="", start_sec=1.6, end_sec=1.8),
                TranscriptSegment(
                    id="segment-5", text="Это новая часть.", start_sec=3.0, end_sec=6.0
                ),
            ],
        )

        coalesced = coalesce_transcript(transcription)

        assert coalesced.detected_language == "ru"
        segment_texts = [segment.text for segment in coalesced.segments]

        assert segment_texts == ["Привет всем.", "Это новая часть."]
        assert coalesced.segments[0].start_sec == 0.4
        assert coalesced.segments[0].end_sec == 1.6
        assert coalesced.segments[1].start_sec == 3.0
        assert coalesced.segments[1].end_sec == 6.0

    def test_keeps_sentence_when_gap_is_large(self) -> None:
        transcription = TranscriptionResult(
            segments=[
                TranscriptSegment(
                    id="segment-1", text="Короткая фраза.", start_sec=0.0, end_sec=1.0
                ),
                TranscriptSegment(
                    id="segment-2", text="Следующая фраза.", start_sec=2.5, end_sec=3.5
                ),
            ]
        )

        coalesced = coalesce_transcript(transcription)

        segment_texts = [segment.text for segment in coalesced.segments]

        assert segment_texts == ["Короткая фраза.", "Следующая фраза."]

    def test_splits_when_speaker_changes_and_preserves_speaker(self) -> None:
        transcription = TranscriptionResult(
            segments=[
                TranscriptSegment(
                    id="segment-1", text="First speaker", start_sec=0.0, end_sec=1.0, speaker="S1"
                ),
                TranscriptSegment(
                    id="segment-2", text="still talking", start_sec=1.0, end_sec=2.0, speaker="S1"
                ),
                TranscriptSegment(
                    id="segment-3", text="Second speaker", start_sec=2.0, end_sec=3.0, speaker="S2"
                ),
            ]
        )

        coalesced = coalesce_transcript(transcription)

        assert [(segment.text, segment.speaker) for segment in coalesced.segments] == [
            ("First speaker still talking", "S1"),
            ("Second speaker", "S2"),
        ]

    def test_returns_empty_when_all_segments_are_blank(self) -> None:
        transcription = TranscriptionResult(
            detected_language="en",
            segments=[
                TranscriptSegment(id="segment-1", text=" ", start_sec=0.0, end_sec=0.5),
                TranscriptSegment(id="segment-2", text="", start_sec=0.5, end_sec=1.0),
            ],
        )

        coalesced = coalesce_transcript(transcription)

        assert coalesced.detected_language == "en"
        assert coalesced.segments == []

    def test_splits_when_current_segment_is_already_too_long(self) -> None:
        transcription = TranscriptionResult(
            segments=[
                TranscriptSegment(id="segment-1", text="Длинный блок", start_sec=0.0, end_sec=16.0),
                TranscriptSegment(
                    id="segment-2", text="Следующий блок", start_sec=16.1, end_sec=17.0
                ),
            ]
        )

        coalesced = coalesce_transcript(transcription)

        assert [segment.text for segment in coalesced.segments] == [
            "Длинный блок",
            "Следующий блок",
        ]

    def test_splits_when_text_reaches_character_limit(self) -> None:
        transcription = TranscriptionResult(
            segments=[
                TranscriptSegment(
                    id="segment-1", text="x" * MAX_SEGMENT_CHARS, start_sec=0.0, end_sec=1.0
                ),
                TranscriptSegment(id="segment-2", text="tail", start_sec=1.1, end_sec=2.0),
            ]
        )

        coalesced = coalesce_transcript(transcription)

        assert [segment.text for segment in coalesced.segments] == [
            "x" * MAX_SEGMENT_CHARS,
            "tail",
        ]

    def test_splits_after_non_ascii_sentence_terminator(self) -> None:
        transcription = TranscriptionResult(
            segments=[
                TranscriptSegment(
                    id="segment-1",
                    text="Opening thought\N{IDEOGRAPHIC FULL STOP}",
                    start_sec=0.0,
                    end_sec=3.0,
                ),
                TranscriptSegment(id="segment-2", text="Next topic", start_sec=3.0, end_sec=5.0),
            ]
        )

        coalesced = coalesce_transcript(transcription)

        assert [segment.text for segment in coalesced.segments] == [
            "Opening thought\N{IDEOGRAPHIC FULL STOP}",
            "Next topic",
        ]
