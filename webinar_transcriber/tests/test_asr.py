"""Tests for the faster-whisper adapter."""

from webinar_transcriber.asr import WhisperTranscriber


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


def test_whisper_transcriber_normalizes_model_output(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("webinar_transcriber.asr.WhisperModel", lambda *args, **kwargs: FakeModel())

    transcriber = WhisperTranscriber(model_name="tiny")
    result = transcriber.transcribe(tmp_path / "audio.wav")

    assert result.detected_language == "en"
    assert result.segments[0].text == "agenda review"
    assert [word.text for word in result.segments[0].words] == ["agenda", "review"]


def test_default_compute_type_prefers_float32_for_auto_device(monkeypatch) -> None:
    captured: dict[str, str] = {}

    class RecordingModel:
        def __init__(self, model_name: str, *, device: str, compute_type: str) -> None:
            captured["model_name"] = model_name
            captured["device"] = device
            captured["compute_type"] = compute_type

    monkeypatch.setattr("webinar_transcriber.asr.WhisperModel", RecordingModel)

    WhisperTranscriber(model_name="tiny", device="auto")

    assert captured["compute_type"] == "float32"
