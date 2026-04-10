"""Tests for run-directory path helpers."""

from datetime import datetime

import pytest

from webinar_transcriber.paths import (
    OutputDirectoryExistsError,
    build_run_layout,
    create_run_layout,
)


class TestRunLayout:
    def test_build_run_layout_uses_timestamp_and_slug(self, tmp_path) -> None:
        input_path = tmp_path / "Weekly Webinar!.mp4"
        input_path.write_text("demo", encoding="utf-8")

        layout = build_run_layout(
            input_path=input_path,
            now=datetime(2026, 3, 18, 20, 30, 45),
        )

        assert layout.run_dir.name == "20260318-203045-000000_weekly-webinar"
        assert layout.markdown_report_path.name == "report.md"
        assert layout.asr_dir.name == "asr"
        assert layout.speech_regions_path.name == "speech_regions.json"
        assert layout.decoded_windows_path.name == "decoded_windows.json"
        assert layout.subtitle_vtt_path.name == "transcript.vtt"
        assert layout.transcription_audio_path().name == "transcription-audio.wav"
        assert layout.transcription_audio_path("mp3").name == "transcription-audio.mp3"
        assert layout.frames_dir.name == "frames"

    def test_build_run_layout_rejects_existing_output_directory(self, tmp_path) -> None:
        existing_dir = tmp_path / "existing-run"
        existing_dir.mkdir()

        with pytest.raises(OutputDirectoryExistsError):
            build_run_layout(input_path=tmp_path / "demo.wav", output_dir=existing_dir)

    def test_create_run_layout_creates_directory(self, tmp_path) -> None:
        output_dir = tmp_path / "new-run"

        layout = create_run_layout(input_path=tmp_path / "demo.wav", output_dir=output_dir)

        assert layout.run_dir.exists()
        assert layout.json_report_path == output_dir / "report.json"
