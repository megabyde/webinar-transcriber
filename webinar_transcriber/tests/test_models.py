"""Tests for typed pipeline models."""

import pytest

from webinar_transcriber.models import (
    AudioAsset,
    Diagnostics,
    InferenceWindow,
    MediaType,
    ReportDocument,
    ReportSection,
    Scene,
    TranscriptionResult,
    TranscriptSegment,
    VideoAsset,
)


class TestCoreModels:
    def test_video_asset_accepts_video_metadata(self) -> None:
        asset = VideoAsset(path="demo.mp4", duration_sec=120.5, fps=30.0, width=1920, height=1080)

        assert asset.media_type is MediaType.VIDEO
        assert asset.duration_sec == 120.5
        assert asset.width == 1920

    def test_audio_asset_accepts_audio_metadata(self) -> None:
        asset = AudioAsset(path="demo.mp3", duration_sec=42.0, sample_rate=48_000, channels=2)

        assert asset.media_type is MediaType.AUDIO
        assert asset.sample_rate == 48_000
        assert asset.channels == 2

    def test_transcript_segment_accepts_timing_fields(self) -> None:
        segment = TranscriptSegment(id="seg-1", start_sec=0.0, end_sec=1.2, text="hello world")

        assert segment.id == "seg-1"
        assert segment.start_sec == 0.0
        assert segment.end_sec == 1.2
        assert segment.text == "hello world"
        assert segment.speaker is None

    def test_transcript_segment_exposes_midpoint(self) -> None:
        segment = TranscriptSegment(id="seg-1", text="hello world", start_sec=1.0, end_sec=2.2)

        assert segment.midpoint == 1.6
        assert segment.duration_sec == pytest.approx(1.2)
        next_seg = TranscriptSegment(id="seg-2", start_sec=3.0, end_sec=4.0, text="next")
        assert segment.gap_before(next_seg) == pytest.approx(0.8)

    def test_report_document_defaults_optional_collections(self) -> None:
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

    def test_scene_exposes_midpoint(self) -> None:
        scene = Scene(id="scene-1", start_sec=2.0, end_sec=5.0)

        assert scene.midpoint == 3.5

    def test_diagnostics_defaults_empty_maps(self) -> None:
        diagnostics = Diagnostics()

        assert diagnostics.llm is None
        assert diagnostics.stage_durations_sec == {}
        assert diagnostics.item_counts == {}
        assert diagnostics.asr_pipeline is None
        assert diagnostics.diarization is None
        assert diagnostics.warnings == []

    def test_transcript_json_compacts_empty_speaker_fields(self) -> None:
        transcription = TranscriptionResult(
            segments=[
                TranscriptSegment(id="segment-1", text="hello", start_sec=0.0, end_sec=1.0),
                TranscriptSegment(
                    id="segment-2", text="world", start_sec=1.0, end_sec=2.0, speaker="S1"
                ),
            ]
        )

        payload = transcription.to_json()

        segments = payload["segments"]
        assert isinstance(segments, list)
        first_segment = segments[0]
        second_segment = segments[1]
        assert isinstance(first_segment, dict)
        assert isinstance(second_segment, dict)
        assert "speaker" not in first_segment
        assert ("speaker", "S1") in second_segment.items()

    def test_inference_window_is_ordered_by_timeline(self) -> None:
        windows = [
            InferenceWindow(id="window-3", region_index=1, start_sec=10.0, end_sec=12.0),
            InferenceWindow(id="window-2", region_index=0, start_sec=0.0, end_sec=8.0),
            InferenceWindow(id="window-1", region_index=0, start_sec=0.0, end_sec=6.0),
        ]

        window_ids = [
            window.id
            for window in sorted(
                windows, key=lambda item: (item.start_sec, item.end_sec, item.region_index, item.id)
            )
        ]

        assert window_ids == ["window-1", "window-2", "window-3"]

    def test_transcript_segment_is_ordered_by_timeline(self) -> None:
        segments = [
            TranscriptSegment(id="seg-3", text="third", start_sec=2.0, end_sec=3.0),
            TranscriptSegment(id="seg-2", text="second", start_sec=0.0, end_sec=2.0),
            TranscriptSegment(id="seg-1", text="first", start_sec=0.0, end_sec=1.0),
        ]

        segment_ids = [
            segment.id
            for segment in sorted(
                segments, key=lambda item: (item.start_sec, item.end_sec, item.id)
            )
        ]

        assert segment_ids == ["seg-1", "seg-2", "seg-3"]
