"""Tests for media probing and audio extraction."""

import subprocess
from pathlib import Path

import pytest

from webinar_transcriber.media import (
    MediaProcessingError,
    _parse_frame_rate,
    _run_command,
    extract_audio,
    probe_media,
)
from webinar_transcriber.models import MediaType

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def test_probe_media_reads_audio_metadata() -> None:
    asset = probe_media(FIXTURE_DIR / "sample-audio.mp3")

    assert asset.media_type is MediaType.AUDIO
    assert asset.duration_sec > 0
    assert asset.sample_rate


def test_probe_media_treats_attached_cover_art_as_audio_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "webinar_transcriber.media._run_command",
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

    assert asset.media_type is MediaType.AUDIO
    assert asset.duration_sec == 42.5
    assert asset.fps is None
    assert asset.width is None
    assert asset.height is None


def test_extract_audio_creates_wav(tmp_path) -> None:
    output_path = extract_audio(FIXTURE_DIR / "sample-audio.mp3", tmp_path / "audio.wav")

    assert output_path.exists()

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(output_path),
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert probe.returncode == 0
    assert "pcm_s16le" in probe.stdout


def test_run_command_wraps_timeout_with_media_processing_error(monkeypatch) -> None:
    def fake_run(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd=["ffprobe"], timeout=12.5)

    monkeypatch.setattr("webinar_transcriber.media.subprocess.run", fake_run)

    with pytest.raises(MediaProcessingError, match=r"ffprobe timed out after 300s\."):
        _run_command("ffprobe")


def test_run_command_raises_default_error_when_process_fails_without_stderr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "webinar_transcriber.media.subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(["ffprobe"], 1, stderr="   "),
    )

    with pytest.raises(MediaProcessingError, match="External media command failed"):
        _run_command("ffprobe")


def test_probe_media_reads_video_metadata() -> None:
    asset = probe_media(FIXTURE_DIR / "sample-video.mp4")

    assert asset.media_type is MediaType.VIDEO
    assert asset.fps is not None
    assert asset.width is not None
    assert asset.height is not None


def test_probe_media_raises_when_ffprobe_reports_no_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "webinar_transcriber.media._run_command",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            ["ffprobe"],
            0,
            stdout='{"format": {"duration": "1.0"}, "streams": []}',
        ),
    )

    with pytest.raises(MediaProcessingError, match="No audio or video stream found"):
        probe_media(FIXTURE_DIR / "sample-audio.mp3")


def test_parse_frame_rate_returns_none_for_missing_values() -> None:
    assert _parse_frame_rate(None) is None
    assert _parse_frame_rate("0/0") is None
