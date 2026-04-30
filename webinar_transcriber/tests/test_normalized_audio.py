"""Tests for normalized-audio preparation and segmentation."""

import wave
from pathlib import Path
from typing import cast
from unittest.mock import patch

import av
import numpy as np
import pytest

from webinar_transcriber.media import MediaProcessingError
from webinar_transcriber.models import SpeechRegion
from webinar_transcriber.normalized_audio import (
    _mux_audio_frames,
    extract_audio,
    load_normalized_audio,
    prepared_transcription_audio,
    preserve_transcription_audio,
    transcode_audio_to_mp3,
)
from webinar_transcriber.segmentation import (
    VadSettings,
    _normalize_regions,
    _silero_speech_timestamps,
    detect_speech_regions,
    normalized_audio_duration,
)
from webinar_transcriber.tests.conftest import FakeContextContainer

FIXTURE_DIR = Path(__file__).parent / "fixtures"


class TestNormalizedAudio:
    def test_mux_audio_frames_handles_frame_list(self) -> None:
        frame = av.AudioFrame(format="s16", layout="mono", samples=1)
        encoded_frames: list[object | None] = []
        muxed_packets: list[str] = []

        class FakeOutputStream:
            def encode(self, audio_frame=None) -> list[str]:
                encoded_frames.append(audio_frame)
                return ["packet"] if audio_frame is not None else []

        class FakeOutputContainer:
            def mux(self, packet: str) -> None:
                muxed_packets.append(packet)

        _mux_audio_frames(
            cast("av.container.OutputContainer", FakeOutputContainer()),
            cast("av.audio.stream.AudioStream", FakeOutputStream()),
            None,
        )
        _mux_audio_frames(
            cast("av.container.OutputContainer", FakeOutputContainer()),
            cast("av.audio.stream.AudioStream", FakeOutputStream()),
            [frame],
        )

        assert encoded_frames == [frame]
        assert muxed_packets == ["packet"]

    def test_prepared_transcription_audio_normalizes_audio_input_to_temp_wav(self) -> None:
        with prepared_transcription_audio(FIXTURE_DIR / "sample-audio.mp3") as audio_path:
            assert audio_path.exists()
            assert audio_path.suffix == ".wav"

        assert not audio_path.exists()

    @pytest.mark.slow
    def test_extract_audio_creates_wav(self, tmp_path: Path) -> None:
        output_path = extract_audio(FIXTURE_DIR / "sample-audio.mp3", tmp_path / "audio.wav")

        assert output_path.exists()
        with wave.open(str(output_path), "rb") as wav_file:
            assert wav_file.getframerate() == 16_000
            assert wav_file.getnchannels() == 1
            assert wav_file.getsampwidth() == 2
            assert wav_file.getnframes() > 0

    def test_prepared_transcription_audio_cleans_up_temp_wav(self) -> None:
        with prepared_transcription_audio(FIXTURE_DIR / "sample-video.mp4") as audio_path:
            assert audio_path.exists()
            assert audio_path.suffix == ".wav"

        assert not audio_path.exists()

    def test_preserve_transcription_audio_copies_wav_output(self, tmp_path: Path) -> None:
        with prepared_transcription_audio(FIXTURE_DIR / "sample-audio.mp3") as audio_path:
            kept_audio_path = preserve_transcription_audio(
                audio_path, tmp_path / "transcription-audio.wav"
            )

        assert kept_audio_path.exists()
        assert kept_audio_path.suffix == ".wav"
        assert kept_audio_path.read_bytes()[:4] == b"RIFF"

    def test_preserve_transcription_audio_transcodes_mp3_output(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with prepared_transcription_audio(FIXTURE_DIR / "sample-audio.mp3") as audio_path:
            expected_output = tmp_path / "transcription-audio.mp3"
            calls: list[tuple[Path, Path]] = []

            def fake_transcode(input_path: Path, output_path: Path) -> Path:
                calls.append((input_path, output_path))
                output_path.write_text("mp3", encoding="utf-8")
                return output_path

            monkeypatch.setattr(
                "webinar_transcriber.normalized_audio.transcode_audio_to_mp3", fake_transcode
            )
            kept_audio_path = preserve_transcription_audio(
                audio_path, expected_output, audio_format="mp3"
            )

        assert calls == [(audio_path, expected_output)]
        assert kept_audio_path == expected_output
        assert kept_audio_path.read_text(encoding="utf-8") == "mp3"

    @pytest.mark.slow
    def test_transcode_audio_to_mp3_creates_real_mp3(self, tmp_path: Path) -> None:
        with prepared_transcription_audio(FIXTURE_DIR / "sample-audio.mp3") as audio_path:
            output_path = transcode_audio_to_mp3(audio_path, tmp_path / "audio.mp3")

        assert output_path.exists()
        container = av.open(str(output_path))
        try:
            audio_stream = next(stream for stream in container.streams if stream.type == "audio")
            assert "mp3" in audio_stream.codec_context.codec.name
            assert getattr(audio_stream.codec_context, "sample_rate", None) == 16_000
            assert getattr(audio_stream.codec_context, "channels", None) == 1
        finally:
            container.close()

    def test_extract_audio_wraps_open_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "webinar_transcriber.normalized_audio.av.open",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("bad open")),
        )

        with pytest.raises(MediaProcessingError, match=r"Could not open .*bad open"):
            extract_audio(FIXTURE_DIR / "sample-audio.mp3", tmp_path / "audio.wav")

    def test_extract_audio_rejects_inputs_without_audio_stream(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        class FakeContainer(FakeContextContainer):
            def __init__(self) -> None:
                self.streams = [type("Stream", (), {"type": "video"})()]

        monkeypatch.setattr(
            "webinar_transcriber.normalized_audio.av.open",
            lambda *_args, **_kwargs: FakeContainer(),
        )

        with pytest.raises(MediaProcessingError, match="No audio stream found"):
            extract_audio(FIXTURE_DIR / "sample-video.mp4", tmp_path / "audio.wav")

    def test_transcode_audio_to_mp3_wraps_open_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "webinar_transcriber.normalized_audio.av.open",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("bad open")),
        )

        with pytest.raises(MediaProcessingError, match=r"Could not open .*bad open"):
            transcode_audio_to_mp3(tmp_path / "input.wav", tmp_path / "audio.mp3")

    def test_extract_audio_raises_when_pyav_does_not_write_output(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        class FakeAudioStream:
            type = "audio"

        class FakeInputContainer(FakeContextContainer):
            def __init__(self) -> None:
                self.streams = [FakeAudioStream()]

            def decode(self, *_args, **_kwargs):
                return iter(())

        class FakeOutputStream:
            def __init__(self) -> None:
                self.layout = None

            def encode(self, _frame=None) -> list[str]:
                return []

        class FakeOutputContainer(FakeContextContainer):
            def add_stream(self, *_args, **_kwargs) -> FakeOutputStream:
                return FakeOutputStream()

        monkeypatch.setattr(
            "webinar_transcriber.normalized_audio.open_input_media_container",
            lambda *_args, **_kwargs: FakeInputContainer(),
        )
        monkeypatch.setattr(
            "webinar_transcriber.normalized_audio.open_output_media_container",
            lambda *_args, **_kwargs: FakeOutputContainer(),
        )

        with pytest.raises(MediaProcessingError, match=r"PyAV did not write .*audio.wav"):
            extract_audio(FIXTURE_DIR / "sample-audio.mp3", tmp_path / "audio.wav")

    def test_load_normalized_audio_returns_mono_float32_samples(self) -> None:
        with prepared_transcription_audio(FIXTURE_DIR / "sample-audio.mp3") as audio_path:
            samples, sample_rate = load_normalized_audio(audio_path)

        assert sample_rate == 16_000
        assert samples.dtype == np.float32
        assert samples.ndim == 1
        assert samples.size > 0

    def test_detect_speech_regions_falls_back_to_full_audio_without_silero(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            "webinar_transcriber.segmentation._silero_speech_timestamps",
            lambda *_args, **_kwargs: None,
        )

        samples = np.zeros(16_000, dtype=np.float32)
        regions, warnings = detect_speech_regions(samples, 16_000)

        assert len(regions) == 1
        assert regions[0].start_sec == 0.0
        assert regions[0].end_sec == 1.0
        assert warnings

    def test_detect_speech_regions_returns_empty_for_zero_duration(self) -> None:
        regions, warnings = detect_speech_regions(np.zeros(0, dtype=np.float32), 16_000)

        assert regions == []
        assert warnings == []

    def test_detect_speech_regions_returns_full_audio_when_vad_disabled(self) -> None:
        regions, warnings = detect_speech_regions(
            np.zeros(8_000, dtype=np.float32), 16_000, settings=VadSettings(enabled=False)
        )

        assert len(regions) == 1
        assert regions[0].end_sec == 0.5
        assert warnings == []

    def test_detect_speech_regions_drops_empty_timestamps(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "webinar_transcriber.segmentation._silero_speech_timestamps",
            lambda *_args, **_kwargs: [{"start": 10, "end": 10}],
        )

        regions, warnings = detect_speech_regions(np.zeros(16_000, dtype=np.float32), 16_000)

        assert regions == []
        assert warnings == []

    def test_normalized_audio_duration_returns_zero_for_invalid_sample_rate(self) -> None:
        assert normalized_audio_duration(np.zeros(16, dtype=np.float32), 0) == 0.0

    def test_normalize_regions_handles_empty_and_overlap(self) -> None:
        assert _normalize_regions([]) == []
        normalized = _normalize_regions([
            SpeechRegion(start_sec=0.0, end_sec=1.0),
            SpeechRegion(start_sec=0.9, end_sec=2.0),
        ])
        assert [(region.start_sec, region.end_sec) for region in normalized] == [(0.0, 2.0)]
        normalized = _normalize_regions([
            SpeechRegion(start_sec=0.0, end_sec=1.0),
            SpeechRegion(start_sec=1.5, end_sec=2.0),
        ])
        assert normalized == [
            SpeechRegion(start_sec=0.0, end_sec=1.0),
            SpeechRegion(start_sec=1.5, end_sec=2.0),
        ]

    def test_normalize_regions_keeps_disjoint_regions(self) -> None:
        normalized = _normalize_regions([
            SpeechRegion(start_sec=0.0, end_sec=1.0),
            SpeechRegion(start_sec=1.5, end_sec=2.0),
        ])

        assert normalized == [
            SpeechRegion(start_sec=0.0, end_sec=1.0),
            SpeechRegion(start_sec=1.5, end_sec=2.0),
        ]

    def test_normalize_regions_drops_non_positive_duration_regions(self) -> None:
        normalized = _normalize_regions([
            SpeechRegion(start_sec=1.0, end_sec=1.0),
            SpeechRegion(start_sec=2.0, end_sec=1.0),
            SpeechRegion(start_sec=3.0, end_sec=4.0),
        ])

        assert normalized == [SpeechRegion(start_sec=3.0, end_sec=4.0)]

    def test_silero_speech_timestamps_returns_none_when_module_missing(self) -> None:
        with patch(
            "webinar_transcriber.segmentation.importlib.import_module", side_effect=ImportError
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

    def test_silero_speech_timestamps_uses_get_speech_timestamps(
        self, monkeypatch, fake_silero_import_module
    ) -> None:
        def fake_get_speech_timestamps(
            samples,
            _model,
            *,
            threshold,
            sampling_rate,
            min_speech_duration_ms,
            min_silence_duration_ms,
            speech_pad_ms,
        ):
            assert len(samples) == 1_600
            assert threshold == 0.5
            assert sampling_rate == 16_000
            assert min_speech_duration_ms == 10
            assert min_silence_duration_ms == 600
            assert speech_pad_ms == 200
            return [{"start": 100, "end": 900}]

        monkeypatch.setattr(
            "webinar_transcriber.segmentation.importlib.import_module",
            fake_silero_import_module(get_speech_timestamps_fn=fake_get_speech_timestamps),
        )

        timestamps = _silero_speech_timestamps(
            np.zeros(1_600, dtype=np.float32),
            sample_rate=16_000,
            threshold=0.5,
            min_speech_duration_ms=10,
            min_silence_duration_ms=600,
            speech_pad_ms=200,
        )

        assert timestamps == [{"start": 100, "end": 900}]

    def test_silero_speech_timestamps_requires_normalized_16khz_audio(
        self, monkeypatch, fake_silero_import_module
    ) -> None:
        monkeypatch.setattr(
            "webinar_transcriber.segmentation.importlib.import_module", fake_silero_import_module()
        )

        with pytest.raises(ValueError, match="16000 Hz"):
            _silero_speech_timestamps(
                np.zeros(800, dtype=np.float32),
                sample_rate=8_000,
                threshold=0.5,
                min_speech_duration_ms=10,
                min_silence_duration_ms=600,
                speech_pad_ms=30,
            )

    def test_silero_speech_timestamps_returns_backend_timestamps_unchanged(
        self, monkeypatch, fake_silero_import_module
    ) -> None:
        def fake_get_speech_timestamps(*_args, **_kwargs):
            return [{"start": 200, "end": 1_600}]

        monkeypatch.setattr(
            "webinar_transcriber.segmentation.importlib.import_module",
            fake_silero_import_module(get_speech_timestamps_fn=fake_get_speech_timestamps),
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

    def test_load_normalized_audio_rejects_wrong_sample_rate(self, monkeypatch, tmp_path) -> None:
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
                "webinar_transcriber.normalized_audio.wave.open",
                lambda *_args, **_kwargs: FakeWave(),
            )

            with pytest.raises(MediaProcessingError, match="Expected 16000 Hz"):
                load_normalized_audio(audio_path)

    @pytest.mark.parametrize(
        ("channels", "sample_width", "message"),
        [(2, 2, "Expected mono transcription audio"), (1, 1, "Expected 16-bit PCM")],
    )
    def test_load_normalized_audio_rejects_invalid_channel_or_sample_width(
        self, monkeypatch: pytest.MonkeyPatch, channels: int, sample_width: int, message: str
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
                "webinar_transcriber.normalized_audio.wave.open",
                lambda *_args, **_kwargs: FakeWave(),
            )

            with pytest.raises(MediaProcessingError, match=message):
                load_normalized_audio(audio_path)
