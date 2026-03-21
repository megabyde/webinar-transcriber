"""Tests for media probing and audio extraction."""

import subprocess
from pathlib import Path

from webinar_transcriber.media import extract_audio, prepared_transcription_audio, probe_media
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


def test_prepared_transcription_audio_reuses_audio_input() -> None:
    asset = probe_media(FIXTURE_DIR / "sample-audio.mp3")

    with prepared_transcription_audio(FIXTURE_DIR / "sample-audio.mp3", asset) as audio_path:
        assert audio_path == FIXTURE_DIR / "sample-audio.mp3"


def test_prepared_transcription_audio_cleans_up_temp_wav() -> None:
    asset = probe_media(FIXTURE_DIR / "sample-video.mp4")

    with prepared_transcription_audio(FIXTURE_DIR / "sample-video.mp4", asset) as audio_path:
        assert audio_path.exists()
        assert audio_path.suffix == ".wav"

    assert not audio_path.exists()
