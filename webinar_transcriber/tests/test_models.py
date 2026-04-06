"""Tests for typed pipeline models."""

from webinar_transcriber.models import (
    AsrPipelineDiagnostics,
    DecodedWindow,
    Diagnostics,
    InferenceWindow,
    MediaAsset,
    MediaType,
    ReportDocument,
    ReportSection,
    SpeechRegion,
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
    assert segment.text == "hello world"


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
    assert report.sections[0].tldr is None


def test_diagnostics_defaults_empty_maps() -> None:
    diagnostics = Diagnostics()

    assert not diagnostics.llm_enabled
    assert diagnostics.llm_report_status == "disabled"
    assert diagnostics.llm_report_usage == {}
    assert diagnostics.stage_durations_sec == {}
    assert diagnostics.item_counts == {}
    assert diagnostics.asr_pipeline is None
    assert diagnostics.warnings == []


def test_asr_pipeline_support_models_accept_expected_fields() -> None:
    speech_region = SpeechRegion(start_sec=0.0, end_sec=1.5)
    window = InferenceWindow(
        window_id="window-1",
        region_index=0,
        start_sec=0.0,
        end_sec=1.5,
        overlap_sec=0.3,
    )
    decoded_window = DecodedWindow(
        window=window,
        input_prompt="hello",
        text="hello",
        segments=[
            TranscriptSegment(
                id="segment-1",
                text="hello",
                start_sec=0.0,
                end_sec=1.0,
            )
        ],
    )
    diagnostics = AsrPipelineDiagnostics(
        normalized_audio_duration_sec=120.0,
        vad_enabled=True,
        vad_region_count=5,
        carryover_enabled=True,
        window_count=4,
    )

    assert speech_region.end_sec == 1.5
    assert decoded_window.window.window_id == "window-1"
    assert decoded_window.input_prompt == "hello"
    assert diagnostics.carryover_enabled
    assert diagnostics.window_count == 4


def test_inference_window_is_ordered_by_timeline() -> None:
    windows = [
        InferenceWindow(window_id="window-3", region_index=1, start_sec=10.0, end_sec=12.0),
        InferenceWindow(window_id="window-2", region_index=0, start_sec=0.0, end_sec=8.0),
        InferenceWindow(window_id="window-1", region_index=0, start_sec=0.0, end_sec=6.0),
    ]

    window_ids = [window.window_id for window in sorted(windows)]

    assert window_ids == ["window-1", "window-2", "window-3"]
