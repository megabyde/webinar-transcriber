"""Tests for run-directory path helpers."""

from datetime import UTC, datetime

import pytest

from webinar_transcriber.paths import OutputDirectoryExistsError, create_run_layout

FIXED_RUN_TIME = datetime(2026, 3, 18, 20, 30, 45, tzinfo=UTC)


class TestRunLayout:
    def test_create_run_layout_uses_timestamp_and_slug(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        input_path = tmp_path / "Weekly Webinar!.mp4"
        input_path.write_text("demo", encoding="utf-8")

        layout = create_run_layout(input_path=input_path, now=FIXED_RUN_TIME)

        assert layout.run_dir.name == "20260318-203045-000000_weekly-webinar"
        assert layout.run_dir.exists()
        assert layout.markdown_report_path.name == "report.md"
        assert layout.asr_dir.name == "asr"
        assert layout.speech_regions_path.name == "speech_regions.json"
        assert layout.decoded_windows_path.name == "decoded_windows.json"
        assert layout.transcription_audio_path.name == "transcription-audio.mp3"
        assert layout.frames_dir.name == "frames"

    def test_create_run_layout_preserves_unicode_slug_text(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        input_path = tmp_path / "Вебинар-2026-итоги.mp4"
        input_path.write_text("demo", encoding="utf-8")

        layout = create_run_layout(input_path=input_path, now=FIXED_RUN_TIME)

        assert layout.run_dir.name == "20260318-203045-000000_вебинар-2026-итоги"

    def test_create_run_layout_falls_back_when_slug_has_no_words(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        input_path = tmp_path / "!!!.mp4"
        input_path.write_text("demo", encoding="utf-8")

        layout = create_run_layout(input_path=input_path, now=FIXED_RUN_TIME)

        assert layout.run_dir.name == "20260318-203045-000000_input"

    def test_create_run_layout_rejects_existing_output_directory(self, tmp_path) -> None:
        existing_dir = tmp_path / "existing-run"
        existing_dir.mkdir()

        with pytest.raises(OutputDirectoryExistsError):
            create_run_layout(input_path=tmp_path / "demo.wav", output_dir=existing_dir)

    def test_create_run_layout_creates_directory(self, tmp_path) -> None:
        output_dir = tmp_path / "new-run"

        layout = create_run_layout(input_path=tmp_path / "demo.wav", output_dir=output_dir)

        assert layout.run_dir.exists()
        assert layout.json_report_path == output_dir / "report.json"
