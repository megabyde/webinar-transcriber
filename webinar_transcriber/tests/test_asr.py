"""Tests for the whisper.cpp ASR adapter."""

import numpy as np
import pytest

from webinar_transcriber.asr import (
    DEFAULT_WHISPER_CPP_MODEL,
    WhisperCppTranscriber,
    _device_name_from_system_info,
)
from webinar_transcriber.models import ChunkTranscription, TranscriptSegment


def test_default_model_uses_whisper_cpp_model_path() -> None:
    transcriber = WhisperCppTranscriber()

    assert transcriber.model_name == str(DEFAULT_WHISPER_CPP_MODEL)


def test_prepare_model_requires_model_file(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="model file does not exist") as error:
        WhisperCppTranscriber(model_name=str(tmp_path / "missing.bin")).prepare_model()
    message = str(error.value)
    assert "--asr-model" in message
    assert "README.md" in message


def test_prepare_model_missing_default_model_is_actionable(tmp_path, monkeypatch) -> None:
    default_model_path = tmp_path / "missing-default-model.bin"
    monkeypatch.setattr(
        "webinar_transcriber.asr.DEFAULT_WHISPER_CPP_MODEL",
        default_model_path,
    )
    with pytest.raises(RuntimeError, match="model file does not exist") as error:
        WhisperCppTranscriber(model_name=str(default_model_path)).prepare_model()
    message = str(error.value)
    assert "Download ggml-large-v3-turbo.bin" in message
    assert str(default_model_path) in message


def test_prepare_model_reads_runtime_details(monkeypatch, tmp_path) -> None:
    model_path = tmp_path / "model.bin"
    model_path.write_text("stub", encoding="utf-8")

    class FakeLibrary:
        def __init__(self, _library_path, log_path=None) -> None:
            self._library_path = _library_path
            self._log_path = log_path

        def create_session(self, _model_path):
            from webinar_transcriber.whispercpp import WhisperCppRuntimeDetails

            class FakeSession:
                runtime_details = WhisperCppRuntimeDetails(
                    library_path="/usr/local/lib/libwhisper.so",
                    system_info="METAL = 1",
                )

                def close(self) -> None:
                    return None

            return FakeSession()

    monkeypatch.setattr("webinar_transcriber.asr.WhisperCppLibrary", FakeLibrary)

    transcriber = WhisperCppTranscriber(model_name=str(model_path))
    transcriber.prepare_model()

    assert transcriber.device_name == "metal"
    assert transcriber.library_path == "/usr/local/lib/libwhisper.so"


def test_whisper_cpp_transcriber_uses_library_for_chunked_transcription(
    monkeypatch, tmp_path
) -> None:
    model_path = tmp_path / "model.bin"
    model_path.write_text("stub", encoding="utf-8")
    progress_updates: list[float] = []

    monkeypatch.setattr(
        "webinar_transcriber.asr.load_normalized_audio",
        lambda _path: (np.zeros(16_000, dtype=np.float32), 16_000),
    )

    class FakeLibrary:
        def __init__(self, _library_path, log_path=None) -> None:
            self._library_path = _library_path
            self._log_path = log_path

        def create_session(self, _model_path):
            from webinar_transcriber.whispercpp import WhisperCppRuntimeDetails

            class FakeSession:
                runtime_details = WhisperCppRuntimeDetails(
                    library_path="/usr/local/lib/libwhisper.so",
                    system_info="CPU = 1",
                )

                def transcribe_chunks(
                    self,
                    audio_samples,
                    chunks,
                    *,
                    threads,
                    initial_prompt=None,
                    progress_callback=None,
                ):
                    del audio_samples, initial_prompt
                    assert threads == 6
                    assert len(chunks) == 1
                    if progress_callback is not None:
                        progress_callback(chunks[0].end_sec)
                    return [
                        ChunkTranscription(
                            chunk_id=chunks[0].id,
                            start_sec=0.0,
                            end_sec=1.0,
                            detected_language="ru",
                            segments=[
                                TranscriptSegment(
                                    id="segment-1",
                                    text="agenda review",
                                    start_sec=0.0,
                                    end_sec=1.0,
                                )
                            ],
                        )
                    ]

                def close(self) -> None:
                    return None

            return FakeSession()

    monkeypatch.setattr("webinar_transcriber.asr.WhisperCppLibrary", FakeLibrary)

    transcriber = WhisperCppTranscriber(model_name=str(model_path), threads=6)
    result = transcriber.transcribe(
        tmp_path / "audio.wav",
        progress_callback=lambda completed_sec: progress_updates.append(completed_sec),
    )

    assert result.detected_language == "ru"
    assert [segment.text for segment in result.segments] == ["agenda review"]
    assert progress_updates == [1.0]
    assert transcriber.device_name == "cpu"


def test_device_name_from_system_info_prefers_enabled_backend() -> None:
    assert _device_name_from_system_info("CPU = 1 | METAL = 1") == "metal"
    assert (
        _device_name_from_system_info("WHISPER : MTL : EMBED_LIBRARY = 1 | CPU : NEON = 1 |")
        == "metal"
    )
    assert _device_name_from_system_info("CPU = 1") == "cpu"
