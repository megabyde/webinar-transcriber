"""Tests for media probing and shared media command helpers."""

import subprocess
from pathlib import Path

import pytest

from webinar_transcriber.media import (
    MediaProcessingError,
    _parse_frame_rate,
    probe_media,
    run_media_command,
)
from webinar_transcriber.models import AudioAsset, VideoAsset

FIXTURE_DIR = Path(__file__).parent / "fixtures"


class TestProbeMedia:
    def test_reads_audio_metadata(self) -> None:
        asset = probe_media(FIXTURE_DIR / "sample-audio.mp3")

        assert isinstance(asset, AudioAsset)
        assert asset.duration_sec > 0
        assert asset.sample_rate

    def test_treats_attached_cover_art_as_audio_only(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "webinar_transcriber.media.run_media_command",
            lambda *_args, **_kwargs: subprocess.CompletedProcess(
                ["ffprobe"],
                0,
                stdout="""
                {
                  "format": {"duration": "42.5"},
                  "streams": [
                    {
                      "codec_type": "audio",
                      "sample_rate": "48000",
                      "channels": 2,
                      "duration": "42.5",
                      "disposition": {"attached_pic": 0}
                    },
                    {
                      "codec_type": "video",
                      "width": 1280,
                      "height": 720,
                      "avg_frame_rate": "0/0",
                      "disposition": {"attached_pic": 1}
                    }
                  ]
                }
                """,
            ),
        )

        asset = probe_media(FIXTURE_DIR / "sample-audio.mp3")

        assert isinstance(asset, AudioAsset)
        assert asset.duration_sec == 42.5

    def test_reads_video_metadata(self) -> None:
        asset = probe_media(FIXTURE_DIR / "sample-video.mp4")

        assert isinstance(asset, VideoAsset)
        assert asset.fps is not None
        assert asset.width is not None
        assert asset.height is not None

    def test_raises_when_ffprobe_reports_no_streams(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "webinar_transcriber.media.run_media_command",
            lambda *_args, **_kwargs: subprocess.CompletedProcess(
                ["ffprobe"],
                0,
                stdout='{"format": {"duration": "1.0"}, "streams": []}',
            ),
        )

        with pytest.raises(MediaProcessingError, match="No audio or video stream found"):
            probe_media(FIXTURE_DIR / "sample-audio.mp3")


class TestRunMediaCommand:
    def test_wraps_timeout_with_media_processing_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def fake_run(*_args, **_kwargs):
            raise subprocess.TimeoutExpired(cmd=["ffprobe"], timeout=12.5)

        monkeypatch.setattr("webinar_transcriber.media.subprocess.run", fake_run)

        with pytest.raises(MediaProcessingError, match=r"ffprobe timed out after 300s\."):
            run_media_command("ffprobe")

    def test_raises_default_error_when_process_fails_without_stderr(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "webinar_transcriber.media.subprocess.run",
            lambda *_args, **_kwargs: subprocess.CompletedProcess(["ffprobe"], 1, stderr="   "),
        )

        with pytest.raises(MediaProcessingError, match="External media command failed"):
            run_media_command("ffprobe")


class TestParseFrameRate:
    @pytest.mark.parametrize(
        ("raw_value", "expected"),
        [
            (None, None),
            ("0/0", None),
            ("30000/1001", pytest.approx(29.97002997002997)),
        ],
    )
    def test_parses_optional_frame_rate(
        self,
        raw_value: str | None,
        expected: float | None,
    ) -> None:
        assert _parse_frame_rate(raw_value) == expected
