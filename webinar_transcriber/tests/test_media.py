"""Tests for media probing and audio extraction."""

import subprocess
from pathlib import Path

import pytest

from webinar_transcriber.media import MediaProcessingError, _run_command, extract_audio, probe_media
from webinar_transcriber.models import MediaType

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def test_probe_media_reads_audio_metadata() -> None:
    asset = probe_media(FIXTURE_DIR / "sample-audio.mp3")

    assert asset.media_type is MediaType.AUDIO
    assert asset.duration_sec > 0
    assert asset.sample_rate


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
