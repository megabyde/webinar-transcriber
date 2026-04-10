"""Tests for overlap reconciliation of decoded inference windows."""

from webinar_transcriber.models import DecodedWindow, InferenceWindow, TranscriptSegment
from webinar_transcriber.reconciliation import reconcile_decoded_windows


class TestReconcileDecodedWindows:
    def test_drops_aligned_overlap_duplicates(self) -> None:
        transcription, stats = reconcile_decoded_windows([
            DecodedWindow(
                window=InferenceWindow(
                    window_id="window-1",
                    region_index=0,
                    start_sec=0.0,
                    end_sec=10.0,
                    overlap_sec=1.5,
                ),
                language="en",
                text="Agenda review Next topic",
                segments=[
                    TranscriptSegment(id="w1s1", text="Agenda review", start_sec=0.0, end_sec=2.0),
                    TranscriptSegment(id="w1s2", text="Next topic", start_sec=2.0, end_sec=4.0),
                ],
            ),
            DecodedWindow(
                window=InferenceWindow(
                    window_id="window-2",
                    region_index=0,
                    start_sec=3.0,
                    end_sec=12.0,
                    overlap_sec=1.0,
                ),
                language="en",
                text="Next topic Action items",
                segments=[
                    TranscriptSegment(id="w2s1", text="Next topic", start_sec=3.0, end_sec=4.5),
                    TranscriptSegment(id="w2s2", text="Action items", start_sec=4.5, end_sec=6.0),
                ],
            ),
        ])

        segment_texts = [segment.text for segment in transcription.segments]

        assert segment_texts == ["Agenda review", "Next topic", "Action items"]
        assert stats.duplicate_segments_dropped == 1
        assert transcription.detected_language == "en"

    def test_prefers_center_of_overlap(self) -> None:
        transcription, stats = reconcile_decoded_windows([
            DecodedWindow(
                window=InferenceWindow(
                    window_id="window-1",
                    region_index=0,
                    start_sec=0.0,
                    end_sec=6.0,
                    overlap_sec=2.0,
                ),
                text="please send the draft by Friday",
                segments=[
                    TranscriptSegment(
                        id="w1s1",
                        text="please send the draft by Friday",
                        start_sec=0.0,
                        end_sec=6.0,
                    )
                ],
            ),
            DecodedWindow(
                window=InferenceWindow(
                    window_id="window-2",
                    region_index=0,
                    start_sec=4.0,
                    end_sec=10.0,
                    overlap_sec=2.0,
                ),
                text="draft by Friday thanks everyone",
                segments=[
                    TranscriptSegment(
                        id="w2s1",
                        text="draft by Friday thanks everyone",
                        start_sec=4.0,
                        end_sec=8.0,
                    )
                ],
            ),
        ])

        assert "Friday" in " ".join(segment.text for segment in transcription.segments)
        assert "Friday Friday" not in " ".join(segment.text for segment in transcription.segments)
        assert stats.boundary_fixes >= 1

    def test_keeps_monotonic_segment_boundaries(self) -> None:
        transcription, _stats = reconcile_decoded_windows([
            DecodedWindow(
                window=InferenceWindow(
                    window_id="window-1",
                    region_index=0,
                    start_sec=0.0,
                    end_sec=5.0,
                    overlap_sec=1.5,
                ),
                text="intro next topic",
                segments=[
                    TranscriptSegment(id="w1s1", text="intro", start_sec=0.0, end_sec=2.0),
                    TranscriptSegment(id="w1s2", text="next topic", start_sec=2.0, end_sec=4.0),
                ],
            ),
            DecodedWindow(
                window=InferenceWindow(
                    window_id="window-2",
                    region_index=0,
                    start_sec=3.0,
                    end_sec=7.0,
                    overlap_sec=1.0,
                ),
                text="next topic wrap up",
                segments=[
                    TranscriptSegment(id="w2s1", text="next topic", start_sec=3.0, end_sec=4.0),
                    TranscriptSegment(id="w2s2", text="wrap up", start_sec=4.0, end_sec=6.0),
                ],
            ),
        ])

        segment_pairs = zip(transcription.segments, transcription.segments[1:], strict=False)

        assert all(current.start_sec >= previous.end_sec for previous, current in segment_pairs)

    def test_prefers_one_side_when_overlap_has_no_alignment(self) -> None:
        transcription, stats = reconcile_decoded_windows([
            DecodedWindow(
                window=InferenceWindow(
                    window_id="window-1",
                    region_index=0,
                    start_sec=9.37,
                    end_sec=29.37,
                    overlap_sec=2.0,
                ),
                text="Понятно, что, имея основной специализацией соблазнения девчонок,",
                segments=[
                    TranscriptSegment(
                        id="w1s1",
                        text="Понятно, что, имея основной специализацией соблазнения девчонок,",
                        start_sec=24.91,
                        end_sec=29.37,
                    )
                ],
            ),
            DecodedWindow(
                window=InferenceWindow(
                    window_id="window-2",
                    region_index=0,
                    start_sec=27.37,
                    end_sec=40.614,
                    overlap_sec=2.0,
                ),
                text="и начинаешь понимать, что секс-то важно.",
                segments=[
                    TranscriptSegment(
                        id="w2s1",
                        text="и начинаешь понимать, что секс-то важно.",
                        start_sec=27.37,
                        end_sec=40.51,
                    )
                ],
            ),
        ])

        segment_texts = [seg.text for seg in transcription.segments]
        expected_texts = [
            "Понятно, что, имея основной специализацией соблазнения девчонок,",
            "начинаешь понимать, что секс-то важно.",
        ]

        assert segment_texts == expected_texts
        assert stats.boundary_fixes >= 1
