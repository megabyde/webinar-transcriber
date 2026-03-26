"""Tests for overlap reconciliation of chunk transcripts."""

from webinar_transcriber.models import ChunkTranscription, TranscriptSegment
from webinar_transcriber.reconciliation import reconcile_chunk_transcriptions


def test_reconcile_chunk_transcriptions_drops_overlapped_duplicates() -> None:
    transcription, stats = reconcile_chunk_transcriptions(
        [
            ChunkTranscription(
                chunk_id="chunk-1",
                start_sec=0.0,
                end_sec=10.0,
                detected_language="en",
                segments=[
                    TranscriptSegment(
                        id="c1s1",
                        text="Agenda review",
                        start_sec=0.0,
                        end_sec=2.0,
                    ),
                    TranscriptSegment(
                        id="c1s2",
                        text="Next topic",
                        start_sec=2.0,
                        end_sec=4.0,
                    ),
                ],
            ),
            ChunkTranscription(
                chunk_id="chunk-2",
                start_sec=3.0,
                end_sec=12.0,
                detected_language="en",
                segments=[
                    TranscriptSegment(
                        id="c2s1",
                        text="Next topic",
                        start_sec=3.0,
                        end_sec=4.5,
                    ),
                    TranscriptSegment(
                        id="c2s2",
                        text="Action items",
                        start_sec=4.5,
                        end_sec=6.0,
                    ),
                ],
            ),
        ]
    )

    assert [segment.text for segment in transcription.segments] == [
        "Agenda review",
        "Next topic",
        "Action items",
    ]
    assert stats.duplicate_segments_dropped == 1
    assert transcription.detected_language == "en"


def test_reconcile_chunk_transcriptions_clamps_non_monotonic_boundaries() -> None:
    transcription, stats = reconcile_chunk_transcriptions(
        [
            ChunkTranscription(
                chunk_id="chunk-1",
                start_sec=0.0,
                end_sec=8.0,
                segments=[
                    TranscriptSegment(
                        id="c1s1",
                        text="First",
                        start_sec=0.0,
                        end_sec=3.0,
                    ),
                    TranscriptSegment(
                        id="c1s2",
                        text="Second",
                        start_sec=2.5,
                        end_sec=4.0,
                    ),
                ],
            )
        ]
    )

    assert transcription.segments[1].start_sec == 3.0
    assert stats.boundary_fixes == 1


def test_reconcile_chunk_transcriptions_drops_zero_duration_boundary_echo() -> None:
    transcription, stats = reconcile_chunk_transcriptions(
        [
            ChunkTranscription(
                chunk_id="chunk-1",
                start_sec=0.0,
                end_sec=8.0,
                segments=[
                    TranscriptSegment(
                        id="c1s1",
                        text="This is the longer previous sentence.",
                        start_sec=0.0,
                        end_sec=3.0,
                    ),
                    TranscriptSegment(
                        id="c1s2",
                        text="previous sentence.",
                        start_sec=3.0,
                        end_sec=3.0,
                    ),
                    TranscriptSegment(
                        id="c1s3",
                        text="Next topic starts here.",
                        start_sec=3.0,
                        end_sec=5.0,
                    ),
                ],
            )
        ]
    )

    assert [segment.text for segment in transcription.segments] == [
        "This is the longer previous sentence.",
        "Next topic starts here.",
    ]
    assert stats.duplicate_segments_dropped == 1


def test_reconcile_chunk_transcriptions_drops_tiny_boundary_echo() -> None:
    transcription, stats = reconcile_chunk_transcriptions(
        [
            ChunkTranscription(
                chunk_id="chunk-1",
                start_sec=0.0,
                end_sec=8.0,
                segments=[
                    TranscriptSegment(
                        id="c1s1",
                        text="Да, но тема не прокатит.",
                        start_sec=0.0,
                        end_sec=2.0,
                    ),
                    TranscriptSegment(
                        id="c1s2",
                        text="тема не прокатит",
                        start_sec=2.0,
                        end_sec=2.03,
                    ),
                    TranscriptSegment(
                        id="c1s3",
                        text="Следующая мысль начинается здесь.",
                        start_sec=2.03,
                        end_sec=4.0,
                    ),
                ],
            )
        ]
    )

    assert [segment.text for segment in transcription.segments] == [
        "Да, но тема не прокатит.",
        "Следующая мысль начинается здесь.",
    ]
    assert stats.duplicate_segments_dropped == 1
