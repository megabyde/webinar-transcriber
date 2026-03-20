"""Tests for ASR backend adapters."""

from types import SimpleNamespace

import pytest

from webinar_transcriber.asr import (
    DEFAULT_FASTER_WHISPER_MODEL,
    DEFAULT_MLX_WHISPER_MODEL,
    FasterWhisperTranscriber,
    MlxWhisperTranscriber,
    WhisperTranscriber,
)
from webinar_transcriber.models import TranscriptionResult


class FakeWord:
    """Simple stand-in for a faster-whisper word object."""

    def __init__(self, word: str, start: float, end: float, probability: float) -> None:
        self.word = word
        self.start = start
        self.end = end
        self.probability = probability


class FakeSegment:
    """Simple stand-in for a faster-whisper segment object."""

    def __init__(self) -> None:
        self.text = " agenda review "
        self.start = 0.0
        self.end = 1.5
        self.words = [
            FakeWord(" agenda ", 0.0, 0.6, 0.91),
            FakeWord(" review ", 0.7, 1.5, 0.95),
        ]


class FakeInfo:
    """Simple stand-in for faster-whisper transcription metadata."""

    language = "en"


class FakeModel:
    """Simple stand-in for a WhisperModel instance."""

    def transcribe(self, audio_path: str, *, beam_size: int, vad_filter: bool):
        assert audio_path.endswith(".wav")
        assert beam_size == 5
        assert vad_filter is True
        return [FakeSegment()], FakeInfo()


def test_faster_whisper_transcriber_normalizes_model_output(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("webinar_transcriber.asr.WhisperModel", lambda *args, **kwargs: FakeModel())

    transcriber = FasterWhisperTranscriber(model_name=DEFAULT_FASTER_WHISPER_MODEL)
    progress_updates: list[float] = []
    result = transcriber.transcribe(
        tmp_path / "audio.wav",
        progress_callback=lambda completed_sec: progress_updates.append(completed_sec),
    )

    assert result.detected_language == "en"
    assert result.segments[0].text == "agenda review"
    assert [word.text for word in result.segments[0].words] == ["agenda", "review"]
    assert progress_updates == [1.5]
    assert transcriber.supports_live_progress is True


def test_default_model_and_compute_type_use_quality_focused_defaults(monkeypatch) -> None:
    captured: dict[str, str] = {}

    class RecordingModel:
        def __init__(self, model_name: str, *, device: str, compute_type: str) -> None:
            captured["model_name"] = model_name
            captured["device"] = device
            captured["compute_type"] = compute_type

    monkeypatch.setattr("webinar_transcriber.asr.WhisperModel", RecordingModel)

    FasterWhisperTranscriber(device="auto")

    assert captured["model_name"] == DEFAULT_FASTER_WHISPER_MODEL
    assert captured["compute_type"] == "int8"


def test_mlx_whisper_transcriber_normalizes_model_output(monkeypatch, tmp_path) -> None:
    def fake_transcribe(
        audio_path: str,
        *,
        path_or_hf_repo: str,
        word_timestamps: bool,
    ) -> dict[str, object]:
        assert audio_path.endswith(".wav")
        assert path_or_hf_repo == DEFAULT_MLX_WHISPER_MODEL
        assert word_timestamps is True
        return {
            "language": "en",
            "segments": [
                {
                    "text": " agenda review ",
                    "start": 0.0,
                    "end": 1.5,
                    "words": [
                        {
                            "word": " agenda ",
                            "start": 0.0,
                            "end": 0.6,
                            "probability": 0.91,
                        },
                        {
                            "word": " review ",
                            "start": 0.7,
                            "end": 1.5,
                            "probability": 0.95,
                        },
                    ],
                }
            ],
        }

    monkeypatch.setattr(
        "webinar_transcriber.asr._import_mlx_whisper",
        lambda: SimpleNamespace(transcribe=fake_transcribe),
    )

    transcriber = MlxWhisperTranscriber(model_name=DEFAULT_MLX_WHISPER_MODEL)
    progress_updates: list[float] = []
    result = transcriber.transcribe(
        tmp_path / "audio.wav",
        progress_callback=lambda completed_sec: progress_updates.append(completed_sec),
    )

    assert result.detected_language == "en"
    assert result.segments[0].text == "agenda review"
    assert [word.text for word in result.segments[0].words] == ["agenda", "review"]
    assert progress_updates == []
    assert transcriber.supports_live_progress is False


def test_mlx_whisper_transcriber_preloads_model(monkeypatch) -> None:
    captured: dict[str, str] = {}

    def fake_preload(model_name: str) -> None:
        captured["model_name"] = model_name

    monkeypatch.setattr("webinar_transcriber.asr._preload_mlx_model", fake_preload)

    transcriber = MlxWhisperTranscriber(model_name=DEFAULT_MLX_WHISPER_MODEL)
    transcriber.prepare_model()

    assert captured["model_name"] == DEFAULT_MLX_WHISPER_MODEL


def test_whisper_transcriber_prefers_mlx_when_available(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("webinar_transcriber.asr._mlx_backend_available", lambda: True)
    captured: dict[str, object] = {}

    class FakeMlxDelegate:
        def __init__(self, model_name: str) -> None:
            assert model_name == DEFAULT_MLX_WHISPER_MODEL

        @property
        def supports_live_progress(self) -> bool:
            return False

        def prepare_model(self) -> None:
            captured["prepared"] = True

        def transcribe(self, audio_path, *, progress_callback=None):
            captured["audio_path"] = audio_path
            captured["callback"] = progress_callback
            return TranscriptionResult(detected_language="en", segments=[])

    monkeypatch.setattr("webinar_transcriber.asr.MlxWhisperTranscriber", FakeMlxDelegate)

    transcriber = WhisperTranscriber()
    transcriber.prepare_model()
    transcriber.transcribe(tmp_path / "audio.wav")

    assert transcriber.backend == "mlx"
    assert transcriber.supports_live_progress is False
    assert captured["prepared"] is True
    assert str(captured["audio_path"]).endswith("audio.wav")


def test_whisper_transcriber_falls_back_to_faster_whisper(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("webinar_transcriber.asr._mlx_backend_available", lambda: False)
    captured: dict[str, object] = {}

    class FakeFasterDelegate:
        def __init__(self, model_name: str, *, device: str, compute_type: str | None) -> None:
            assert model_name == DEFAULT_FASTER_WHISPER_MODEL
            assert device == "auto"
            assert compute_type is None

        @property
        def supports_live_progress(self) -> bool:
            return True

        def prepare_model(self) -> None:
            captured["prepared"] = True

        def transcribe(self, audio_path, *, progress_callback=None):
            captured["audio_path"] = audio_path
            captured["callback"] = progress_callback
            return TranscriptionResult(detected_language="en", segments=[])

    monkeypatch.setattr("webinar_transcriber.asr.FasterWhisperTranscriber", FakeFasterDelegate)

    transcriber = WhisperTranscriber()
    transcriber.prepare_model()
    transcriber.transcribe(tmp_path / "audio.wav")

    assert transcriber.backend == "faster-whisper"
    assert transcriber.supports_live_progress is True
    assert captured["prepared"] is True
    assert str(captured["audio_path"]).endswith("audio.wav")


def test_explicit_mlx_backend_requires_availability(monkeypatch) -> None:
    monkeypatch.setattr("webinar_transcriber.asr._mlx_backend_available", lambda: False)

    with pytest.raises(RuntimeError, match="MLX Whisper requires"):
        WhisperTranscriber(backend="mlx")
