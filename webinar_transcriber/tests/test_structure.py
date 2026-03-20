"""Tests for report structuring helpers."""

from webinar_transcriber.models import (
    AlignmentBlock,
    MediaAsset,
    MediaType,
    TranscriptionResult,
    TranscriptSegment,
)
from webinar_transcriber.structure import build_report


def test_build_report_uses_alignment_block_title_hint_and_warnings() -> None:
    report = build_report(
        MediaAsset(path="demo-file.mp4", media_type=MediaType.VIDEO, duration_sec=12.0),
        TranscriptionResult(
            detected_language="en",
            segments=[
                TranscriptSegment(
                    id="segment-1",
                    text="Agenda overview",
                    start_sec=0.0,
                    end_sec=2.0,
                )
            ],
        ),
        alignment_blocks=[
            AlignmentBlock(
                id="block-1",
                start_sec=0.0,
                end_sec=12.0,
                transcript_segment_ids=["segment-1"],
                transcript_text="Agenda overview",
                scene_id="scene-1",
                frame_id="frame-1",
                title_hint="Slide Title",
            )
        ],
        warnings=["low confidence"],
    )

    assert report.title == "Demo File"
    assert report.sections[0].title == "Slide Title"
    assert report.sections[0].frame_id == "frame-1"
    assert report.warnings == ["low confidence"]


def test_build_report_falls_back_for_empty_text_and_dedupes_summary() -> None:
    report = build_report(
        MediaAsset(path="", media_type=MediaType.AUDIO, duration_sec=10.0),
        TranscriptionResult(
            segments=[
                TranscriptSegment(id="segment-1", text="  ", start_sec=0.0, end_sec=1.0),
                TranscriptSegment(id="segment-2", text="Repeat me.", start_sec=1.0, end_sec=2.0),
                TranscriptSegment(id="segment-3", text="Repeat me.", start_sec=2.0, end_sec=3.0),
                TranscriptSegment(
                    id="segment-4", text="Please follow up.", start_sec=3.0, end_sec=4.0
                ),
            ],
        ),
    )

    assert report.title == "Transcription Report"
    assert report.sections[0].title == "Section 1"
    assert report.summary == ["Repeat me.", "Please follow up."]
    assert report.action_items == ["Please follow up."]
