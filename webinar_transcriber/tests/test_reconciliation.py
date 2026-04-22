"""Tests for transcript reconciliation."""

from webinar_transcriber.models import DecodedWindow, InferenceWindow, TranscriptSegment
from webinar_transcriber.transcript.reconcile import reconcile_decoded_windows


class TestReconcileDecodedWindows:
    def test_returns_empty_transcript_for_empty_input(self) -> None:
        transcription, stats = reconcile_decoded_windows([])

        assert transcription.segments == []
        assert transcription.detected_language is None
        assert stats.boundary_fixes == 0

    def test_concatenates_non_overlapping_windows_in_order(self) -> None:
        transcription, stats = reconcile_decoded_windows([
            DecodedWindow(
                window=InferenceWindow(
                    window_id="window-2", region_index=0, start_sec=4.0, end_sec=6.0
                ),
                language="en",
                segments=[
                    TranscriptSegment(id="w2s1", text="Action items", start_sec=4.0, end_sec=6.0)
                ],
            ),
            DecodedWindow(
                window=InferenceWindow(
                    window_id="window-1", region_index=0, start_sec=0.0, end_sec=2.0
                ),
                language="en",
                segments=[
                    TranscriptSegment(id="w1s1", text="Agenda review", start_sec=0.0, end_sec=2.0)
                ],
            ),
        ])

        assert [segment.text for segment in transcription.segments] == [
            "Agenda review",
            "Action items",
        ]
        assert transcription.detected_language == "en"
        assert stats.boundary_fixes == 0

    def test_skips_blank_segments(self) -> None:
        transcription, stats = reconcile_decoded_windows([
            DecodedWindow(
                window=InferenceWindow(
                    window_id="window-1", region_index=0, start_sec=0.0, end_sec=2.0
                ),
                language="en",
                segments=[
                    TranscriptSegment(id="w1s1", text="   ", start_sec=0.0, end_sec=1.0),
                    TranscriptSegment(id="w1s2", text="Agenda review", start_sec=1.0, end_sec=2.0),
                ],
            )
        ])

        assert [segment.text for segment in transcription.segments] == ["Agenda review"]
        assert stats.boundary_fixes == 0

    def test_fixes_non_monotonic_boundaries_without_dropping_text(self) -> None:
        transcription, stats = reconcile_decoded_windows([
            DecodedWindow(
                window=InferenceWindow(
                    window_id="window-1", region_index=0, start_sec=0.0, end_sec=5.0
                ),
                segments=[
                    TranscriptSegment(id="w1s1", text="Agenda review", start_sec=0.0, end_sec=3.5)
                ],
            ),
            DecodedWindow(
                window=InferenceWindow(
                    window_id="window-2", region_index=0, start_sec=3.0, end_sec=7.0
                ),
                segments=[
                    TranscriptSegment(id="w2s1", text="Next topic", start_sec=3.0, end_sec=6.0)
                ],
            ),
        ])

        assert [segment.text for segment in transcription.segments] == [
            "Agenda review",
            "Next topic",
        ]
        assert transcription.segments[1].start_sec == 3.5
        assert transcription.segments[1].end_sec == 6.0
        assert stats.boundary_fixes == 1

    def test_skips_segment_when_fixing_produces_zero_duration(self) -> None:
        transcription, stats = reconcile_decoded_windows([
            DecodedWindow(
                window=InferenceWindow(
                    window_id="window-1", region_index=0, start_sec=0.0, end_sec=5.0
                ),
                segments=[
                    TranscriptSegment(id="w1s1", text="Agenda review", start_sec=0.0, end_sec=4.0)
                ],
            ),
            DecodedWindow(
                window=InferenceWindow(
                    window_id="window-2", region_index=0, start_sec=3.0, end_sec=7.0
                ),
                segments=[
                    TranscriptSegment(id="w2s1", text="Too short", start_sec=3.5, end_sec=3.8)
                ],
            ),
        ])

        assert [segment.text for segment in transcription.segments] == ["Agenda review"]
        assert stats.boundary_fixes == 2
