"""Tests for transcription audio preparation, VAD, and chunk planning."""

from pathlib import Path

import numpy as np
import pytest

from webinar_transcriber.media import MediaProcessingError, probe_media
from webinar_transcriber.models import SpeechRegion
from webinar_transcriber.transcription_audio import (
    ChunkPlanSettings,
    VADSettings,
    _merge_nearby_regions,
    _silero_speech_timestamps,
    average_chunk_duration,
    detect_speech_regions,
    load_normalized_audio,
    normalized_audio_duration,
    plan_audio_chunks,
    prepared_transcription_audio,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def test_prepared_transcription_audio_normalizes_audio_input_to_temp_wav() -> None:
    asset = probe_media(FIXTURE_DIR / "sample-audio.mp3")

    with prepared_transcription_audio(FIXTURE_DIR / "sample-audio.mp3", asset) as audio_path:
        assert audio_path.exists()
        assert audio_path.suffix == ".wav"

    assert not audio_path.exists()


def test_prepared_transcription_audio_cleans_up_temp_wav() -> None:
    asset = probe_media(FIXTURE_DIR / "sample-video.mp4")

    with prepared_transcription_audio(FIXTURE_DIR / "sample-video.mp4", asset) as audio_path:
        assert audio_path.exists()
        assert audio_path.suffix == ".wav"

    assert not audio_path.exists()


def test_load_normalized_audio_returns_mono_float32_samples() -> None:
    asset = probe_media(FIXTURE_DIR / "sample-audio.mp3")

    with prepared_transcription_audio(FIXTURE_DIR / "sample-audio.mp3", asset) as audio_path:
        samples, sample_rate = load_normalized_audio(audio_path)

    assert sample_rate == 16_000
    assert samples.dtype == np.float32
    assert samples.ndim == 1
    assert samples.size > 0


def test_detect_speech_regions_falls_back_to_full_audio_without_silero(monkeypatch) -> None:
    monkeypatch.setattr(
        "webinar_transcriber.transcription_audio._silero_speech_timestamps",
        lambda *_args, **_kwargs: None,
    )

    samples = np.zeros(16_000, dtype=np.float32)
    regions, warnings = detect_speech_regions(
        samples,
        16_000,
        settings=VADSettings(enabled=True),
    )

    assert len(regions) == 1
    assert regions[0].start_sec == 0.0
    assert regions[0].end_sec == 1.0
    assert warnings


def test_detect_speech_regions_returns_empty_for_zero_duration() -> None:
    regions, warnings = detect_speech_regions(
        np.zeros(0, dtype=np.float32),
        16_000,
        settings=VADSettings(enabled=True),
    )

    assert regions == []
    assert warnings == []


def test_detect_speech_regions_returns_full_audio_when_vad_disabled() -> None:
    regions, warnings = detect_speech_regions(
        np.zeros(8_000, dtype=np.float32),
        16_000,
        settings=VADSettings(enabled=False),
    )

    assert len(regions) == 1
    assert regions[0].end_sec == 0.5
    assert warnings == []


def test_detect_speech_regions_drops_empty_timestamps(monkeypatch) -> None:
    monkeypatch.setattr(
        "webinar_transcriber.transcription_audio._silero_speech_timestamps",
        lambda *_args, **_kwargs: [{"start": 10, "end": 10}],
    )

    regions, warnings = detect_speech_regions(
        np.zeros(16_000, dtype=np.float32),
        16_000,
        settings=VADSettings(enabled=True),
    )

    assert regions == []
    assert warnings == []


def test_plan_audio_chunks_respects_overlap_and_max_duration() -> None:
    chunks = plan_audio_chunks(
        [SpeechRegion(start_sec=0.0, end_sec=45.0)],
        settings=ChunkPlanSettings(target_sec=20.0, max_sec=30.0, overlap_sec=1.5),
    )

    assert [chunk.id for chunk in chunks] == ["chunk-1", "chunk-2"]
    assert chunks[0].start_sec == 0.0
    assert chunks[0].end_sec == 20.0
    assert chunks[1].start_sec == 18.5
    assert chunks[1].end_sec == 45.0


def test_plan_audio_chunks_returns_empty_for_no_regions() -> None:
    assert plan_audio_chunks([], settings=ChunkPlanSettings()) == []


def test_average_chunk_duration_returns_none_for_empty_chunks() -> None:
    assert average_chunk_duration([]) is None


def test_normalized_audio_duration_returns_zero_for_invalid_sample_rate() -> None:
    assert normalized_audio_duration(np.zeros(16, dtype=np.float32), 0) == 0.0


def test_merge_nearby_regions_handles_empty_and_gaps() -> None:
    assert _merge_nearby_regions([], merge_gap_sec=0.5) == []
    merged = _merge_nearby_regions(
        [
            SpeechRegion(start_sec=0.0, end_sec=1.0),
            SpeechRegion(start_sec=1.2, end_sec=2.0),
            SpeechRegion(start_sec=3.0, end_sec=4.0),
        ],
        merge_gap_sec=0.3,
    )
    assert [(region.start_sec, region.end_sec) for region in merged] == [(0.0, 2.0), (3.0, 4.0)]


def test_silero_speech_timestamps_returns_none_when_module_missing(monkeypatch) -> None:
    def raise_import_error(_name: str):
        raise ImportError

    monkeypatch.setattr(
        "webinar_transcriber.transcription_audio.importlib.import_module", raise_import_error
    )

    assert (
        _silero_speech_timestamps(
            np.zeros(16_000, dtype=np.float32),
            sample_rate=16_000,
            settings=VADSettings(),
        )
        is None
    )


def test_load_normalized_audio_rejects_wrong_sample_rate(monkeypatch, tmp_path) -> None:
    asset = probe_media(FIXTURE_DIR / "sample-audio.mp3")

    with prepared_transcription_audio(FIXTURE_DIR / "sample-audio.mp3", asset) as audio_path:

        class FakeWave:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return None

            def getframerate(self):
                return 8_000

            def getnchannels(self):
                return 1

            def getsampwidth(self):
                return 2

            def getnframes(self):
                return 1

            def readframes(self, _count):
                return b"\x00\x00"

        monkeypatch.setattr(
            "webinar_transcriber.transcription_audio.wave.open",
            lambda *_args, **_kwargs: FakeWave(),
        )

        with pytest.raises(MediaProcessingError, match="Expected 16000 Hz"):
            load_normalized_audio(audio_path)
