"""Tests for transcription audio preparation and segmentation."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from webinar_transcriber.media import MediaProcessingError
from webinar_transcriber.models import SpeechRegion
from webinar_transcriber.segmentation import (
    _normalize_regions,
    _silero_speech_timestamps,
    detect_speech_regions,
    expand_speech_regions,
    normalized_audio_duration,
    repair_speech_regions,
)
from webinar_transcriber.transcription_audio import (
    load_normalized_audio,
    prepared_transcription_audio,
    preserve_transcription_audio,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def test_prepared_transcription_audio_normalizes_audio_input_to_temp_wav() -> None:
    with prepared_transcription_audio(FIXTURE_DIR / "sample-audio.mp3") as audio_path:
        assert audio_path.exists()
        assert audio_path.suffix == ".wav"

    assert not audio_path.exists()


def test_prepared_transcription_audio_cleans_up_temp_wav() -> None:
    with prepared_transcription_audio(FIXTURE_DIR / "sample-video.mp4") as audio_path:
        assert audio_path.exists()
        assert audio_path.suffix == ".wav"

    assert not audio_path.exists()


def test_preserve_transcription_audio_copies_wav_output(tmp_path: Path) -> None:
    with prepared_transcription_audio(FIXTURE_DIR / "sample-audio.mp3") as audio_path:
        kept_audio_path = preserve_transcription_audio(
            audio_path,
            tmp_path / "transcription-audio.wav",
        )

    assert kept_audio_path.exists()
    assert kept_audio_path.suffix == ".wav"
    assert kept_audio_path.read_bytes()[:4] == b"RIFF"


def test_preserve_transcription_audio_transcodes_mp3_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    with prepared_transcription_audio(FIXTURE_DIR / "sample-audio.mp3") as audio_path:
        expected_output = tmp_path / "transcription-audio.mp3"
        calls: list[tuple[Path, Path]] = []

        def fake_transcode(input_path: Path, output_path: Path) -> Path:
            calls.append((input_path, output_path))
            output_path.write_text("mp3", encoding="utf-8")
            return output_path

        monkeypatch.setattr(
            "webinar_transcriber.transcription_audio.transcode_audio_to_mp3",
            fake_transcode,
        )
        kept_audio_path = preserve_transcription_audio(
            audio_path,
            expected_output,
            audio_format="mp3",
        )

    assert calls == [(audio_path, expected_output)]
    assert kept_audio_path == expected_output
    assert kept_audio_path.read_text(encoding="utf-8") == "mp3"


def test_load_normalized_audio_returns_mono_float32_samples() -> None:
    with prepared_transcription_audio(FIXTURE_DIR / "sample-audio.mp3") as audio_path:
        samples, sample_rate = load_normalized_audio(audio_path)

    assert sample_rate == 16_000
    assert samples.dtype == np.float32
    assert samples.ndim == 1
    assert samples.size > 0


def test_detect_speech_regions_falls_back_to_full_audio_without_silero(monkeypatch) -> None:
    monkeypatch.setattr(
        "webinar_transcriber.segmentation._silero_speech_timestamps",
        lambda *_args, **_kwargs: None,
    )

    samples = np.zeros(16_000, dtype=np.float32)
    regions, warnings = detect_speech_regions(
        samples,
        16_000,
        enabled=True,
    )

    assert len(regions) == 1
    assert regions[0].start_sec == 0.0
    assert regions[0].end_sec == 1.0
    assert warnings


def test_detect_speech_regions_returns_empty_for_zero_duration() -> None:
    regions, warnings = detect_speech_regions(
        np.zeros(0, dtype=np.float32),
        16_000,
        enabled=True,
    )

    assert regions == []
    assert warnings == []


def test_detect_speech_regions_returns_full_audio_when_vad_disabled() -> None:
    regions, warnings = detect_speech_regions(
        np.zeros(8_000, dtype=np.float32),
        16_000,
        enabled=False,
    )

    assert len(regions) == 1
    assert regions[0].end_sec == 0.5
    assert warnings == []


def test_detect_speech_regions_drops_empty_timestamps(monkeypatch) -> None:
    monkeypatch.setattr(
        "webinar_transcriber.segmentation._silero_speech_timestamps",
        lambda *_args, **_kwargs: [{"start": 10, "end": 10}],
    )

    regions, warnings = detect_speech_regions(
        np.zeros(16_000, dtype=np.float32),
        16_000,
        enabled=True,
    )

    assert regions == []
    assert warnings == []


def test_normalized_audio_duration_returns_zero_for_invalid_sample_rate() -> None:
    assert normalized_audio_duration(np.zeros(16, dtype=np.float32), 0) == 0.0


def test_expand_speech_regions_clips_and_merges_overlaps() -> None:
    expanded = expand_speech_regions(
        [
            SpeechRegion(start_sec=0.0, end_sec=1.0),
            SpeechRegion(start_sec=1.15, end_sec=2.0),
            SpeechRegion(start_sec=3.0, end_sec=4.0),
        ],
        pad_ms=200,
        audio_duration_sec=4.0,
    )

    assert [(region.start_sec, region.end_sec) for region in expanded] == [(0.0, 2.2), (2.8, 4.0)]


def test_repair_speech_regions_merges_short_region_until_it_reaches_min_duration() -> None:
    repaired = repair_speech_regions([
        SpeechRegion(start_sec=0.0, end_sec=1.0),
        SpeechRegion(start_sec=1.4, end_sec=2.1),
        SpeechRegion(start_sec=2.5, end_sec=4.0),
    ])

    assert [(r.start_sec, r.end_sec) for r in repaired] == [(0.0, 4.0)]


def test_repair_speech_regions_prefers_the_nearest_neighbor() -> None:
    repaired = repair_speech_regions([
        SpeechRegion(start_sec=0.0, end_sec=5.0),
        SpeechRegion(start_sec=5.8, end_sec=6.8),
        SpeechRegion(start_sec=7.65, end_sec=20.0),
    ])

    assert [(r.start_sec, r.end_sec) for r in repaired] == [(0.0, 6.8), (7.65, 20.0)]


def test_repair_speech_regions_keeps_short_region_when_gaps_are_too_large() -> None:
    repaired = repair_speech_regions([
        SpeechRegion(start_sec=0.0, end_sec=12.0),
        SpeechRegion(start_sec=13.1, end_sec=13.9),
        SpeechRegion(start_sec=15.1, end_sec=28.0),
    ])

    expected_regions = [
        SpeechRegion(start_sec=0.0, end_sec=12.0),
        SpeechRegion(start_sec=13.1, end_sec=13.9),
        SpeechRegion(start_sec=15.1, end_sec=28.0),
    ]

    assert repaired == expected_regions


def test_normalize_regions_handles_empty_and_overlap() -> None:
    assert _normalize_regions([]) == []
    normalized = _normalize_regions([
        SpeechRegion(start_sec=0.0, end_sec=1.0),
        SpeechRegion(start_sec=0.9, end_sec=2.0),
    ])
    assert [(region.start_sec, region.end_sec) for region in normalized] == [(0.0, 2.0)]


def test_silero_speech_timestamps_returns_none_when_module_missing() -> None:
    with patch(
        "webinar_transcriber.segmentation.importlib.import_module",
        side_effect=ImportError,
    ):
        timestamps = _silero_speech_timestamps(
            np.zeros(16_000, dtype=np.float32),
            sample_rate=16_000,
            threshold=0.5,
            min_speech_duration_ms=250,
            min_silence_duration_ms=600,
            speech_pad_ms=30,
        )

    assert timestamps is None


def test_silero_speech_timestamps_uses_vad_iterator_and_reports_progress(monkeypatch) -> None:
    progress: list[tuple[float, int]] = []

    class FakeIterator:
        def __init__(
            self,
            _model,
            *,
            threshold,
            sampling_rate,
            min_silence_duration_ms,
            speech_pad_ms,
        ) -> None:
            assert threshold == 0.5
            assert sampling_rate == 16_000
            assert min_silence_duration_ms == 600
            assert speech_pad_ms == 30
            self._calls = 0

        def __call__(self, _chunk, *, return_seconds=False):
            assert return_seconds is False
            self._calls += 1
            if self._calls == 1:
                return {"start": 100}
            if self._calls == 3:
                return {"end": 900}
            return None

    class FakeSilero:
        @staticmethod
        def load_silero_vad():
            return object()

        VADIterator = FakeIterator

    class FakeTorch:
        @staticmethod
        def from_numpy(arr):
            return arr

    def fake_import_module(name: str):
        if name == "silero_vad":
            return FakeSilero
        if name == "torch":
            return FakeTorch
        raise ImportError(name)

    monkeypatch.setattr(
        "webinar_transcriber.segmentation.importlib.import_module",
        fake_import_module,
    )

    timestamps = _silero_speech_timestamps(
        np.zeros(1_600, dtype=np.float32),
        sample_rate=16_000,
        threshold=0.5,
        min_speech_duration_ms=10,
        min_silence_duration_ms=600,
        speech_pad_ms=30,
        progress_callback=lambda completed_sec, detected_count: progress.append((
            completed_sec,
            detected_count,
        )),
    )

    assert timestamps == [{"start": 100, "end": 900}]
    assert progress[-1][0] == pytest.approx(0.1)
    assert progress[-1][1] == 1


def test_silero_speech_timestamps_requires_normalized_16khz_audio(monkeypatch) -> None:
    class FakeSilero:
        @staticmethod
        def load_silero_vad():
            return object()

    class FakeTorch:
        @staticmethod
        def from_numpy(arr):
            return arr

    def fake_import_module(name: str):
        if name == "silero_vad":
            return FakeSilero
        if name == "torch":
            return FakeTorch
        raise ImportError(name)

    monkeypatch.setattr(
        "webinar_transcriber.segmentation.importlib.import_module",
        fake_import_module,
    )

    with pytest.raises(AssertionError, match="16000 Hz"):
        _silero_speech_timestamps(
            np.zeros(800, dtype=np.float32),
            sample_rate=8_000,
            threshold=0.5,
            min_speech_duration_ms=10,
            min_silence_duration_ms=600,
            speech_pad_ms=30,
        )


def test_silero_speech_timestamps_flushes_open_speech_at_end_of_stream(monkeypatch) -> None:
    class FakeIterator:
        def __init__(self, *_args, **_kwargs) -> None:
            self._calls = 0

        def __call__(self, _chunk, *, return_seconds=False):
            assert return_seconds is False
            self._calls += 1
            if self._calls == 1:
                return {"start": 200}
            return None

    class FakeSilero:
        @staticmethod
        def load_silero_vad():
            return object()

        VADIterator = FakeIterator

    class FakeTorch:
        @staticmethod
        def from_numpy(arr):
            return arr

    def fake_import_module(name: str):
        if name == "silero_vad":
            return FakeSilero
        if name == "torch":
            return FakeTorch
        raise ImportError(name)

    monkeypatch.setattr(
        "webinar_transcriber.segmentation.importlib.import_module",
        fake_import_module,
    )

    timestamps = _silero_speech_timestamps(
        np.zeros(1_600, dtype=np.float32),
        sample_rate=16_000,
        threshold=0.5,
        min_speech_duration_ms=10,
        min_silence_duration_ms=600,
        speech_pad_ms=30,
    )

    assert timestamps == [{"start": 200, "end": 1_600}]


def test_load_normalized_audio_rejects_wrong_sample_rate(monkeypatch, tmp_path) -> None:
    with prepared_transcription_audio(FIXTURE_DIR / "sample-audio.mp3") as audio_path:

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


@pytest.mark.parametrize(
    ("channels", "sample_width", "message"),
    [
        (2, 2, "Expected mono transcription audio"),
        (1, 1, "Expected 16-bit PCM"),
    ],
)
def test_load_normalized_audio_rejects_invalid_channel_or_sample_width(
    monkeypatch: pytest.MonkeyPatch,
    channels: int,
    sample_width: int,
    message: str,
) -> None:
    with prepared_transcription_audio(FIXTURE_DIR / "sample-audio.mp3") as audio_path:

        class FakeWave:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return None

            def getframerate(self):
                return 16_000

            def getnchannels(self):
                return channels

            def getsampwidth(self):
                return sample_width

            def getnframes(self):
                return 1

            def readframes(self, _count):
                return b"\x00\x00"

        monkeypatch.setattr(
            "webinar_transcriber.transcription_audio.wave.open",
            lambda *_args, **_kwargs: FakeWave(),
        )

        with pytest.raises(MediaProcessingError, match=message):
            load_normalized_audio(audio_path)
