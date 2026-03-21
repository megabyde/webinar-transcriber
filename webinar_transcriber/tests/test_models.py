"""Tests for typed pipeline models."""

from webinar_transcriber.models import (
    Diagnostics,
    MediaAsset,
    MediaType,
    ReportDocument,
    ReportSection,
    TranscriptSegment,
)


def test_media_asset_accepts_video_metadata() -> None:
    asset = MediaAsset(
        path="demo.mp4",
        media_type=MediaType.VIDEO,
        duration_sec=120.5,
        fps=30.0,
        width=1920,
        height=1080,
    )

    assert asset.media_type is MediaType.VIDEO
    assert asset.duration_sec == 120.5
    assert asset.width == 1920


def test_transcript_segment_accepts_timing_fields() -> None:
    segment = TranscriptSegment(
        id="seg-1",
        text="hello world",
        start_sec=0.0,
        end_sec=1.2,
    )

    assert segment.start_sec == 0.0
    assert segment.end_sec == 1.2


def test_report_document_defaults_optional_collections() -> None:
    report = ReportDocument(
        title="Demo",
        source_file="demo.wav",
        media_type=MediaType.AUDIO,
        sections=[
            ReportSection(
                id="section-1",
                title="Overview",
                start_sec=0.0,
                end_sec=10.0,
                transcript_text="Transcript body.",
            )
        ],
    )

    assert report.summary == []
    assert report.action_items == []
    assert report.sections[0].title == "Overview"


def test_diagnostics_defaults_empty_maps() -> None:
    diagnostics = Diagnostics()

    assert diagnostics.stage_durations_sec == {}
    assert diagnostics.item_counts == {}
    assert diagnostics.warnings == []
