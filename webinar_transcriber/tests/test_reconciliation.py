"""Tests for transcript reconciliation."""

from webinar_transcriber.models import DecodedWindow, InferenceWindow, TranscriptSegment
from webinar_transcriber.transcript.reconcile import reconcile_decoded_windows


class TestReconcileDecodedWindows:
    def test_returns_empty_transcript_for_empty_input(self) -> None:
        transcription = reconcile_decoded_windows([])

        assert transcription.segments == []
        assert transcription.detected_language is None

    def test_concatenates_non_overlapping_windows_in_order(self) -> None:
        transcription = reconcile_decoded_windows([
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

    def test_skips_blank_segments(self) -> None:
        transcription = reconcile_decoded_windows([
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

    def test_fixes_non_monotonic_boundaries_without_dropping_text(self) -> None:
        transcription = reconcile_decoded_windows([
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

    def test_skips_segment_when_fixing_produces_zero_duration(self) -> None:
        transcription = reconcile_decoded_windows([
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

    def test_removes_duplicate_text_from_overlapping_window_boundary(self) -> None:
        transcription = reconcile_decoded_windows([
            DecodedWindow(
                window=InferenceWindow(
                    window_id="window-1", region_index=0, start_sec=0.0, end_sec=28.0
                ),
                segments=[
                    TranscriptSegment(
                        id="w1s1",
                        text="Let's discuss the next slide please",
                        start_sec=24.0,
                        end_sec=28.0,
                    )
                ],
            ),
            DecodedWindow(
                window=InferenceWindow(
                    window_id="window-2", region_index=0, start_sec=26.0, end_sec=54.0
                ),
                segments=[
                    TranscriptSegment(
                        id="w2s1",
                        text="next slide please and action items",
                        start_sec=26.0,
                        end_sec=30.0,
                    )
                ],
            ),
        ])

        assert [segment.text for segment in transcription.segments] == [
            "Let's discuss the next slide please",
            "and action items",
        ]
        assert transcription.segments[1].start_sec == 28.0
        assert transcription.segments[1].end_sec == 30.0

    def test_drops_fully_duplicated_overlap_segment(self) -> None:
        transcription = reconcile_decoded_windows([
            DecodedWindow(
                window=InferenceWindow(
                    window_id="window-1", region_index=0, start_sec=0.0, end_sec=28.0
                ),
                segments=[
                    TranscriptSegment(
                        id="w1s1", text="next slide please", start_sec=24.0, end_sec=28.0
                    )
                ],
            ),
            DecodedWindow(
                window=InferenceWindow(
                    window_id="window-2", region_index=0, start_sec=26.0, end_sec=54.0
                ),
                segments=[
                    TranscriptSegment(
                        id="w2s1", text="Next slide please", start_sec=26.0, end_sec=28.0
                    )
                ],
            ),
        ])

        assert [segment.text for segment in transcription.segments] == ["next slide please"]

    def test_sorts_window_ids_by_numeric_suffix_for_ties(self) -> None:
        transcription = reconcile_decoded_windows([
            DecodedWindow(
                window=InferenceWindow(
                    window_id="window-10", region_index=0, start_sec=0.0, end_sec=2.0
                ),
                segments=[
                    TranscriptSegment(id="w10s1", text="Window ten", start_sec=1.0, end_sec=2.0)
                ],
            ),
            DecodedWindow(
                window=InferenceWindow(
                    window_id="window-2", region_index=0, start_sec=0.0, end_sec=2.0
                ),
                segments=[
                    TranscriptSegment(id="w2s1", text="Window two", start_sec=0.0, end_sec=1.0)
                ],
            ),
        ])

        assert [segment.text for segment in transcription.segments] == ["Window two", "Window ten"]

    def test_accepts_non_numeric_window_ids(self) -> None:
        transcription = reconcile_decoded_windows([
            DecodedWindow(
                window=InferenceWindow(
                    window_id="intro", region_index=0, start_sec=0.0, end_sec=1.0
                ),
                segments=[
                    TranscriptSegment(id="intro-1", text="Intro", start_sec=0.0, end_sec=1.0)
                ],
            )
        ])

        assert [segment.text for segment in transcription.segments] == ["Intro"]
