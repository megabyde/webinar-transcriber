"""Tests for overlap reconciliation of decoded inference windows."""

from webinar_transcriber.models import DecodedWindow, InferenceWindow, TranscriptSegment
from webinar_transcriber.transcript.reconcile import (
    _align_overlap,
    _duplicate_token_indices,
    _group_match_runs,
    _mean_edge_distance,
    _segments_from_tokens,
    _TokenPiece,
    reconcile_decoded_windows,
)


class TestReconcileDecodedWindows:
    def test_returns_empty_transcript_for_empty_input(self) -> None:
        transcription, stats = reconcile_decoded_windows([])

        assert transcription.segments == []
        assert transcription.detected_language is None
        assert stats.duplicate_segments_dropped == 0
        assert stats.boundary_fixes == 0

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

    def test_skips_blank_windows_and_keeps_non_overlapping_segments(self) -> None:
        transcription, stats = reconcile_decoded_windows([
            DecodedWindow(
                window=InferenceWindow(
                    window_id="window-1",
                    region_index=0,
                    start_sec=0.0,
                    end_sec=2.0,
                ),
                language="en",
                text="   ",
                segments=[TranscriptSegment(id="w1s1", text="   ", start_sec=0.0, end_sec=2.0)],
            ),
            DecodedWindow(
                window=InferenceWindow(
                    window_id="window-2",
                    region_index=0,
                    start_sec=4.0,
                    end_sec=6.0,
                ),
                language="en",
                text="Agenda review",
                segments=[
                    TranscriptSegment(id="w2s1", text="Agenda review", start_sec=4.0, end_sec=6.0)
                ],
            ),
        ])

        assert [segment.text for segment in transcription.segments] == ["Agenda review"]
        assert stats == type(stats)()


class TestReconciliationInternals:
    def test_duplicate_token_indices_returns_empty_for_missing_or_non_overlapping_inputs(
        self,
    ) -> None:
        assert _duplicate_token_indices([], []) == (set(), set())

        previous_tokens = [
            _TokenPiece(
                text="alpha",
                normalized="alpha",
                start_sec=0.0,
                end_sec=1.0,
                window_order=0,
                segment_order=0,
                token_order=0,
                source_segment_id="prev-1",
            )
        ]
        current_tokens = [
            _TokenPiece(
                text="alpha",
                normalized="alpha",
                start_sec=2.0,
                end_sec=3.0,
                window_order=1,
                segment_order=0,
                token_order=0,
                source_segment_id="cur-1",
            )
        ]

        assert _duplicate_token_indices(previous_tokens, current_tokens) == (set(), set())

    def test_duplicate_token_indices_can_prefer_previous_overlap(self) -> None:
        previous_tokens = [
            _TokenPiece(
                text="alpha",
                normalized="alpha",
                start_sec=4.0,
                end_sec=4.3,
                window_order=0,
                segment_order=0,
                token_order=0,
                source_segment_id="prev-1",
            ),
            _TokenPiece(
                text="beta",
                normalized="beta",
                start_sec=4.3,
                end_sec=4.6,
                window_order=0,
                segment_order=0,
                token_order=1,
                source_segment_id="prev-1",
            ),
        ]
        current_tokens = [
            _TokenPiece(
                text="gamma",
                normalized="gamma",
                start_sec=4.2,
                end_sec=4.9,
                window_order=1,
                segment_order=0,
                token_order=0,
                source_segment_id="cur-1",
            ),
            _TokenPiece(
                text="delta",
                normalized="delta",
                start_sec=4.9,
                end_sec=5.8,
                window_order=1,
                segment_order=0,
                token_order=1,
                source_segment_id="cur-1",
            ),
        ]

        drop_previous, drop_current = _duplicate_token_indices(previous_tokens, current_tokens)

        assert drop_previous == {0, 1}
        assert drop_current == set()

    def test_duplicate_token_indices_can_drop_current_match_group(self) -> None:
        previous_tokens = [
            _TokenPiece(
                text="alpha",
                normalized="alpha",
                start_sec=0.0,
                end_sec=1.0,
                window_order=0,
                segment_order=0,
                token_order=0,
                source_segment_id="prev-1",
            ),
            _TokenPiece(
                text="beta",
                normalized="beta",
                start_sec=1.0,
                end_sec=2.0,
                window_order=0,
                segment_order=0,
                token_order=1,
                source_segment_id="prev-1",
            ),
        ]
        current_tokens = [
            _TokenPiece(
                text="alpha",
                normalized="alpha",
                start_sec=1.0,
                end_sec=1.2,
                window_order=1,
                segment_order=0,
                token_order=0,
                source_segment_id="cur-1",
            ),
            _TokenPiece(
                text="beta",
                normalized="beta",
                start_sec=1.2,
                end_sec=1.4,
                window_order=1,
                segment_order=0,
                token_order=1,
                source_segment_id="cur-1",
            ),
            _TokenPiece(
                text="gamma",
                normalized="gamma",
                start_sec=1.4,
                end_sec=1.8,
                window_order=1,
                segment_order=0,
                token_order=2,
                source_segment_id="cur-1",
            ),
        ]

        drop_previous, drop_current = _duplicate_token_indices(previous_tokens, current_tokens)

        assert drop_previous == set()
        assert drop_current == {0, 1}

    def test_group_match_runs_returns_empty_list_for_no_matches(self) -> None:
        assert _group_match_runs([]) == []

    def test_group_match_runs_splits_non_contiguous_matches(self) -> None:
        assert _group_match_runs([(0, 0), (2, 3)]) == [[(0, 0)], [(2, 3)]]

    def test_align_overlap_handles_empty_normalized_tokens_and_backtracks_left(self) -> None:
        assert (
            _align_overlap(
                [(0, _TokenPiece("", "", 0, 0, 0, 0, 0, "a"))],
                [(0, _TokenPiece("", "", 0, 0, 0, 0, 0, "b"))],
            )
            == []
        )

        previous_overlap = [
            (0, _TokenPiece("alpha", "alpha", 0, 1, 0, 0, 0, "a")),
            (1, _TokenPiece("beta", "beta", 1, 2, 0, 0, 1, "a")),
        ]
        current_overlap = [
            (0, _TokenPiece("beta", "beta", 0, 1, 1, 0, 0, "b")),
            (1, _TokenPiece("gamma", "gamma", 1, 2, 1, 0, 1, "b")),
        ]

        assert _align_overlap(previous_overlap, current_overlap) == [(1, 0)]

    def test_mean_edge_distance_returns_zero_for_empty_overlap(self) -> None:
        assert _mean_edge_distance([], window_start_sec=0.0, window_end_sec=1.0) == 0.0

    def test_segments_from_tokens_skips_blank_and_zero_duration_groups(self) -> None:
        segments, boundary_fixes = _segments_from_tokens([
            _TokenPiece(
                text=" ",
                normalized="",
                start_sec=0.0,
                end_sec=0.3,
                window_order=0,
                segment_order=0,
                token_order=0,
                source_segment_id="blank",
            ),
            _TokenPiece(
                text="Agenda",
                normalized="agenda",
                start_sec=0.0,
                end_sec=1.0,
                window_order=0,
                segment_order=1,
                token_order=0,
                source_segment_id="seg-1",
            ),
            _TokenPiece(
                text="Review",
                normalized="review",
                start_sec=0.5,
                end_sec=0.75,
                window_order=1,
                segment_order=0,
                token_order=0,
                source_segment_id="seg-2",
            ),
        ])

        assert [(segment.text, segment.start_sec, segment.end_sec) for segment in segments] == [
            ("Agenda", 0.0, 1.0)
        ]
        assert boundary_fixes == 2
