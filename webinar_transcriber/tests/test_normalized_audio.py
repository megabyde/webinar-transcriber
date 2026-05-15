"""Tests for normalized-audio preparation and segmentation."""

import wave
from pathlib import Path
from typing import Self, cast
from unittest.mock import Mock, patch

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
    SHERPA_VAD_BUFFER_SIZE_SEC,
    _silero_speech_regions,
    detect_speech_regions,
    normalize_regions,
    normalized_audio_duration,
)
from webinar_transcriber.tests.conftest import FakeContextContainer

FIXTURE_DIR = Path(__file__).parent / "fixtures"


class FakeWave:
    def __init__(
        self, *, framerate: int = 16_000, channels: int = 1, sample_width: int = 2
    ) -> None:
        self._framerate = framerate
        self._channels = channels
        self._sample_width = sample_width

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def getframerate(self) -> int:
        return self._framerate

    def getnchannels(self) -> int:
        return self._channels

    def getsampwidth(self) -> int:
        return self._sample_width

    def getnframes(self) -> int:
        return 1

    def readframes(self, _count: int) -> bytes:
        return b"\x00\x00"


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

            def fake_transcode(
                input_path: Path, output_path: Path, *, progress_callback=None
            ) -> Path:
                del progress_callback
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
            Mock(side_effect=OSError("bad open")),
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
            Mock(side_effect=OSError("bad open")),
        )

        with pytest.raises(MediaProcessingError, match=r"Could not open .*bad open"):
            transcode_audio_to_mp3(tmp_path / "input.wav", tmp_path / "audio.mp3")

    def test_load_normalized_audio_returns_mono_float32_samples(self) -> None:
        with prepared_transcription_audio(FIXTURE_DIR / "sample-audio.mp3") as audio_path:
            samples, sample_rate = load_normalized_audio(audio_path)

        assert sample_rate == 16_000
        assert samples.dtype == np.float32
        assert samples.ndim == 1
        assert samples.size > 0

    def test_detect_speech_regions_falls_back_to_full_audio_without_vad_backend(
        self, monkeypatch
    ) -> None:
        progress_updates: list[tuple[float, int]] = []
        monkeypatch.setattr(
            "webinar_transcriber.segmentation._silero_speech_regions",
            lambda *_args, **_kwargs: None,
        )

        samples = np.zeros(16_000, dtype=np.float32)
        regions, warnings = detect_speech_regions(
            samples,
            threads=1,
            progress_callback=lambda sec, count: progress_updates.append((sec, count)),
        )

        assert len(regions) == 1
        assert regions[0].start_sec == 0.0
        assert regions[0].end_sec == 1.0
        assert warnings
        assert progress_updates == [(1.0, 1)]

    def test_detect_speech_regions_returns_empty_for_zero_duration(self) -> None:
        regions, warnings = detect_speech_regions(np.zeros(0, dtype=np.float32), threads=1)

        assert regions == []
        assert warnings == []

    def test_detect_speech_regions_drops_empty_regions(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "webinar_transcriber.segmentation._silero_speech_regions",
            lambda *_args, **_kwargs: [SpeechRegion(start_sec=10.0, end_sec=10.0)],
        )

        regions, warnings = detect_speech_regions(np.zeros(16_000, dtype=np.float32), threads=1)

        assert regions == []
        assert warnings == []

    def test_normalized_audio_duration_uses_normalized_sample_rate(self) -> None:
        assert normalized_audio_duration(np.zeros(8_000, dtype=np.float32)) == 0.5

    def test_normalize_regions_handles_empty_and_overlap(self) -> None:
        assert normalize_regions([]) == []
        normalized = normalize_regions([
            SpeechRegion(start_sec=0.0, end_sec=1.0),
            SpeechRegion(start_sec=0.9, end_sec=2.0),
        ])
        assert [(region.start_sec, region.end_sec) for region in normalized] == [(0.0, 2.0)]
        normalized = normalize_regions([
            SpeechRegion(start_sec=0.0, end_sec=1.0),
            SpeechRegion(start_sec=1.5, end_sec=2.0),
        ])
        assert normalized == [
            SpeechRegion(start_sec=0.0, end_sec=1.0),
            SpeechRegion(start_sec=1.5, end_sec=2.0),
        ]

    def test_normalize_regions_keeps_disjoint_regions(self) -> None:
        normalized = normalize_regions([
            SpeechRegion(start_sec=0.0, end_sec=1.0),
            SpeechRegion(start_sec=1.5, end_sec=2.0),
        ])

        assert normalized == [
            SpeechRegion(start_sec=0.0, end_sec=1.0),
            SpeechRegion(start_sec=1.5, end_sec=2.0),
        ]

    def test_normalize_regions_drops_non_positive_duration_regions(self) -> None:
        normalized = normalize_regions([
            SpeechRegion(start_sec=1.0, end_sec=1.0),
            SpeechRegion(start_sec=2.0, end_sec=1.0),
            SpeechRegion(start_sec=3.0, end_sec=4.0),
        ])

        assert normalized == [SpeechRegion(start_sec=3.0, end_sec=4.0)]

    def test_silero_speech_regions_returns_none_when_sherpa_module_missing(self) -> None:
        with patch(
            "webinar_transcriber.segmentation.importlib.import_module", side_effect=ImportError
        ):
            regions = _silero_speech_regions(
                np.zeros(16_000, dtype=np.float32),
                threads=1,
            )

        assert regions is None

    def test_silero_speech_regions_uses_sherpa_vad_settings(
        self, monkeypatch, fake_sherpa_import_module
    ) -> None:
        progress_updates: list[tuple[float, int]] = []
        segment = type("Segment", (), {"start": 200, "samples": np.zeros(400)})()
        fake_import_module, fake_sherpa = fake_sherpa_import_module(
            segments=[segment], window_size=512
        )

        monkeypatch.setattr(
            "webinar_transcriber.segmentation.importlib.import_module",
            fake_import_module,
        )

        regions = _silero_speech_regions(
            np.zeros(1_600, dtype=np.float32),
            threads=3,
            progress_callback=lambda sec, count: progress_updates.append((sec, count)),
        )

        detector = fake_sherpa.detectors[0]
        assert detector.config.silero_vad.model.endswith("silero_vad.onnx")
        assert detector.config.silero_vad.threshold == 0.45
        assert detector.config.silero_vad.min_speech_duration == 0.150
        assert detector.config.silero_vad.min_silence_duration == 0.500
        assert detector.config.sample_rate == 16_000
        assert detector.config.num_threads == 3
        assert detector.buffer_size_in_seconds == SHERPA_VAD_BUFFER_SIZE_SEC
        assert [len(samples) for samples in detector.accepted_waveforms] == [512, 512, 512, 64]
        assert detector.flushed
        assert regions == [SpeechRegion(start_sec=0.0, end_sec=0.1)]
        assert progress_updates == [(0.1, 1)]

    def test_silero_speech_regions_reports_progress_by_second(
        self, monkeypatch, fake_sherpa_import_module
    ) -> None:
        progress_updates: list[tuple[float, int]] = []
        segments = [
            type("Segment", (), {"start": 400, "samples": np.zeros(400)})(),
            type("Segment", (), {"start": 20_000, "samples": np.zeros(800)})(),
        ]
        fake_import_module, _fake_sherpa = fake_sherpa_import_module(
            segments=segments, window_size=8_000
        )
        monkeypatch.setattr(
            "webinar_transcriber.segmentation.importlib.import_module",
            fake_import_module,
        )

        regions = _silero_speech_regions(
            np.zeros(40_000, dtype=np.float32),
            threads=1,
            progress_callback=lambda sec, count: progress_updates.append((sec, count)),
        )

        assert regions is not None
        assert len(regions) == 2
        assert regions[0].start_sec == 0.0
        assert regions[0].end_sec == pytest.approx(0.4)
        assert regions[1].start_sec == pytest.approx(0.9)
        assert regions[1].end_sec == pytest.approx(1.65)
        assert progress_updates == [(1.0, 2), (2.0, 2), (2.5, 2)]

    def test_silero_speech_regions_returns_none_when_detector_construction_fails(
        self, monkeypatch, fake_sherpa_import_module
    ) -> None:
        fake_import_module, _fake_sherpa = fake_sherpa_import_module()
        monkeypatch.setattr(
            "webinar_transcriber.segmentation.importlib.import_module",
            fake_import_module,
        )
        monkeypatch.setattr(
            "webinar_transcriber.segmentation._silero_vad_model_path",
            Mock(side_effect=OSError("model unavailable")),
        )

        regions = _silero_speech_regions(
            np.zeros(1_600, dtype=np.float32),
            threads=1,
        )

        assert regions is None

    def test_silero_speech_regions_returns_none_for_invalid_window_size(
        self, monkeypatch, fake_sherpa_import_module
    ) -> None:
        fake_import_module, _fake_sherpa = fake_sherpa_import_module(window_size=0)
        monkeypatch.setattr(
            "webinar_transcriber.segmentation.importlib.import_module",
            fake_import_module,
        )

        regions = _silero_speech_regions(
            np.zeros(1_600, dtype=np.float32),
            threads=1,
        )

        assert regions is None

    def test_silero_speech_regions_pads_and_merges_sherpa_segments(
        self, monkeypatch, fake_sherpa_import_module
    ) -> None:
        segments = [
            type("Segment", (), {"start": 400, "samples": np.zeros(400)})(),
            type("Segment", (), {"start": 900, "samples": np.zeros(300)})(),
            type("Segment", (), {"start": 2_100, "samples": np.zeros(100)})(),
        ]
        fake_import_module, _fake_sherpa = fake_sherpa_import_module(
            segments=segments, window_size=1_024
        )

        monkeypatch.setattr(
            "webinar_transcriber.segmentation.importlib.import_module",
            fake_import_module,
        )

        regions = _silero_speech_regions(
            np.zeros(3_000, dtype=np.float32),
            threads=1,
        )

        assert regions is not None
        assert regions == [SpeechRegion(start_sec=0.0, end_sec=0.1875)]

    def test_silero_speech_regions_drops_empty_sherpa_segments(
        self, monkeypatch, fake_sherpa_import_module
    ) -> None:
        segment = type("Segment", (), {"start": 1_000, "samples": np.zeros(0)})()
        fake_import_module, _fake_sherpa = fake_sherpa_import_module(segments=[segment])

        monkeypatch.setattr(
            "webinar_transcriber.segmentation.importlib.import_module",
            fake_import_module,
        )

        regions = _silero_speech_regions(
            np.zeros(1_600, dtype=np.float32),
            threads=1,
        )

        assert regions == []

    @pytest.mark.slow
    def test_silero_speech_regions_detects_speech_with_real_sherpa_model(self) -> None:
        samples, _ = load_normalized_audio(FIXTURE_DIR / "speech-sample.wav")

        regions = _silero_speech_regions(
            samples,
            threads=1,
        )

        assert regions
        assert regions[0].start_sec == 0
        assert regions[0].end_sec > regions[0].start_sec

    @pytest.mark.parametrize(
        ("framerate", "channels", "sample_width", "message"),
        [
            (8_000, 1, 2, "Expected 16000 Hz"),
            (16_000, 2, 2, "Expected mono transcription audio"),
            (16_000, 1, 1, "Expected 16-bit PCM"),
        ],
    )
    def test_load_normalized_audio_rejects_invalid_wav_contract(
        self,
        monkeypatch: pytest.MonkeyPatch,
        framerate: int,
        channels: int,
        sample_width: int,
        message: str,
    ) -> None:
        with prepared_transcription_audio(FIXTURE_DIR / "sample-audio.mp3") as audio_path:
            monkeypatch.setattr(
                "webinar_transcriber.normalized_audio.wave.open",
                lambda *_args, **_kwargs: FakeWave(
                    framerate=framerate, channels=channels, sample_width=sample_width
                ),
            )

            with pytest.raises(MediaProcessingError, match=message):
                load_normalized_audio(audio_path)
