"""Tests for deterministic scene alignment."""

from webinar_transcriber.align import align_by_time
from webinar_transcriber.models import Scene, SlideFrame, TranscriptSegment


class TestAlignByTime:
    def test_assigns_segments_by_midpoint(self) -> None:
        blocks = align_by_time(
            transcript_segments=[
                TranscriptSegment(id="seg-1", text="Intro", start_sec=0.0, end_sec=0.8),
                TranscriptSegment(id="seg-2", text="Demo", start_sec=1.2, end_sec=1.8),
            ],
            scenes=[
                Scene(id="scene-1", start_sec=0.0, end_sec=1.0),
                Scene(id="scene-2", start_sec=1.0, end_sec=2.0),
            ],
            slide_frames=[
                SlideFrame(
                    id="frame-1",
                    scene_id="scene-1",
                    image_path="scene-1.png",
                    timestamp_sec=0.5,
                ),
                SlideFrame(
                    id="frame-2",
                    scene_id="scene-2",
                    image_path="scene-2.png",
                    timestamp_sec=1.5,
                ),
            ],
        )

        assert [block.transcript_text for block in blocks] == ["Intro", "Demo"]
        assert blocks[0].frame_id == "frame-1"
        assert blocks[1].frame_id == "frame-2"

    def test_returns_empty_list_when_there_are_no_scenes(self) -> None:
        blocks = align_by_time(
            transcript_segments=[
                TranscriptSegment(id="seg-1", text="Intro", start_sec=0.0, end_sec=0.8),
            ],
            scenes=[],
            slide_frames=[],
        )

        assert blocks == []

    def test_assigns_orphan_segments_to_nearest_blocks_and_warns(self) -> None:
        warnings: list[str] = []

        blocks = align_by_time(
            transcript_segments=[
                TranscriptSegment(id="seg-1", text="Before", start_sec=0.0, end_sec=0.2),
                TranscriptSegment(id="seg-2", text="Inside", start_sec=1.2, end_sec=1.6),
                TranscriptSegment(id="seg-3", text="After", start_sec=3.2, end_sec=3.6),
            ],
            scenes=[
                Scene(id="scene-1", start_sec=1.0, end_sec=2.0),
                Scene(id="scene-2", start_sec=2.0, end_sec=3.0),
            ],
            slide_frames=[],
            warnings=warnings,
        )

        assert [block.transcript_segment_ids for block in blocks] == [["seg-1", "seg-2"], ["seg-3"]]
        assert warnings == [
            (
                "Aligned 2 transcript segments to the nearest scene blocks because their "
                "midpoints fell outside all scene ranges."
            )
        ]

    def test_keeps_empty_blocks_for_scenes_without_matches(self) -> None:
        blocks = align_by_time(
            transcript_segments=[
                TranscriptSegment(id="seg-1", text="Demo", start_sec=1.2, end_sec=1.8),
            ],
            scenes=[
                Scene(id="scene-1", start_sec=0.0, end_sec=1.0),
                Scene(id="scene-2", start_sec=1.0, end_sec=2.0),
            ],
            slide_frames=[],
        )

        assert blocks[0].transcript_segment_ids == []
        assert blocks[0].transcript_text == ""
        assert blocks[1].transcript_segment_ids == ["seg-1"]
