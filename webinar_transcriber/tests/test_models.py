"""Tests for typed pipeline models."""

from webinar_transcriber.models import (
    Diagnostics,
    MediaAsset,
    MediaType,
    ReportDocument,
    ReportSection,
    TranscriptSegment,
    TranscriptWord,
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


def test_transcript_segment_preserves_nested_words() -> None:
    segment = TranscriptSegment(
        id="seg-1",
        text="hello world",
        start_sec=0.0,
        end_sec=1.2,
        words=[
            TranscriptWord(text="hello", start_sec=0.0, end_sec=0.5, confidence=0.9),
            TranscriptWord(text="world", start_sec=0.6, end_sec=1.2, confidence=0.95),
        ],
    )

    assert [word.text for word in segment.words] == ["hello", "world"]


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
