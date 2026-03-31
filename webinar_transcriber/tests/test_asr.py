"""Tests for the whisper.cpp ASR adapter."""

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from webinar_transcriber.asr import (
    DEFAULT_WHISPER_CPP_MODEL,
    PromptCarryoverSettings,
    WhisperCppTranscriber,
    _carryover_drop_reason,
    _device_name_from_system_info,
    _read_sysctl_int,
    _sanitize_prompt,
    build_prompt_carryover,
)
from webinar_transcriber.models import DecodedWindow, InferenceWindow, TranscriptSegment

if TYPE_CHECKING:
    from webinar_transcriber.whispercpp import WhisperCppSession


class TestWhisperCppTranscriber:
    def test_default_model_uses_whisper_cpp_model_path(self) -> None:
        transcriber = WhisperCppTranscriber()

        assert transcriber.model_name == str(DEFAULT_WHISPER_CPP_MODEL)

    def test_prepare_model_requires_model_file(self, tmp_path) -> None:
        with pytest.raises(RuntimeError, match="model file does not exist") as error:
            WhisperCppTranscriber(model_name=str(tmp_path / "missing.bin")).prepare_model()
        message = str(error.value)
        assert "--asr-model" in message
        assert "README.md" in message

    def test_prepare_model_missing_default_model_is_actionable(
        self,
        tmp_path,
        monkeypatch,
    ) -> None:
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

    def test_prepare_model_reads_runtime_details(self, monkeypatch, tmp_path) -> None:
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
        assert transcriber.system_info == "METAL = 1"
        assert transcriber.library_path == "/usr/local/lib/libwhisper.so"

    def test_whisper_cpp_transcriber_carries_input_prompt_into_decoded_windows(
        self,
        monkeypatch,
        tmp_path,
    ) -> None:
        model_path = tmp_path / "model.bin"
        model_path.write_text("stub", encoding="utf-8")
        progress_updates: list[float] = []

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

                    def decode_window(
                        self,
                        audio_samples,
                        window,
                        *,
                        threads,
                        prompt=None,
                        language_hint=None,
                    ):
                        del audio_samples, language_hint
                        assert threads == 6
                        return DecodedWindow(
                            window=window,
                            text=(
                                "agenda review"
                                if window.window_id == "window-1"
                                else f"{prompt} follow up"
                            ),
                            language="ru",
                            segments=[
                                TranscriptSegment(
                                    id="segment-1",
                                    text="agenda review",
                                    start_sec=0.0,
                                    end_sec=1.0,
                                )
                            ],
                        )

                    def close(self) -> None:
                        return None

                return FakeSession()

        monkeypatch.setattr("webinar_transcriber.asr.WhisperCppLibrary", FakeLibrary)

        transcriber = WhisperCppTranscriber(model_name=str(model_path), threads=6)
        decoded_windows = transcriber.transcribe_inference_windows(
            np.zeros(16_000, dtype=np.float32),
            [
                InferenceWindow(
                    window_id="window-1",
                    region_index=0,
                    start_sec=0.0,
                    end_sec=1.0,
                ),
                InferenceWindow(
                    window_id="window-2",
                    region_index=0,
                    start_sec=1.0,
                    end_sec=2.0,
                ),
            ],
            progress_callback=lambda completed_sec, _segment_count: progress_updates.append(
                completed_sec
            ),
        )

        assert [w.language for w in decoded_windows] == ["ru", "ru"]
        assert [w.text for w in decoded_windows] == ["agenda review", "agenda review follow up"]
        assert [w.input_prompt for w in decoded_windows] == [None, "agenda review"]
        assert progress_updates == [1.0, 2.0]
        assert transcriber.device_name == "cpu"

    def test_transcriber_destructor_swallows_close_failures(self) -> None:
        transcriber = object.__new__(WhisperCppTranscriber)

        def failing_close() -> None:
            raise RuntimeError("shutdown")

        transcriber.close = failing_close  # type: ignore[method-assign]

        transcriber.__del__()

    @pytest.mark.parametrize("stdout", ["abc", "0"])
    def test_read_sysctl_int_returns_none_for_invalid_or_nonpositive_values(
        self,
        monkeypatch: pytest.MonkeyPatch,
        stdout: str,
    ) -> None:
        monkeypatch.setattr(
            "webinar_transcriber.asr.subprocess.run",
            lambda *_args, **_kwargs: type("Result", (), {"stdout": stdout})(),
        )

        assert _read_sysctl_int("hw.physicalcpu") is None

    def test_transcriber_properties_are_safe_before_runtime_is_prepared(self) -> None:
        transcriber = WhisperCppTranscriber(threads=0)

        assert transcriber.threads == 1
        assert transcriber.system_info is None
        assert transcriber.library_path is None

    def test_transcribe_inference_windows_reuses_existing_session_without_prepare(self) -> None:
        class FakeSession:
            def decode_window(
                self,
                audio_samples,
                window,
                *,
                threads,
                prompt=None,
                language_hint=None,
            ):
                del audio_samples, threads, prompt, language_hint
                return DecodedWindow(
                    window=window,
                    text="agenda review",
                    language="en",
                    segments=[
                        TranscriptSegment(
                            id="segment-1",
                            text="agenda review",
                            start_sec=window.start_sec,
                            end_sec=window.end_sec,
                        )
                    ],
                )

        fake_session = cast("WhisperCppSession", FakeSession())

        class ReuseSessionTranscriber(WhisperCppTranscriber):
            def prepare_model(self) -> None:
                raise AssertionError("should not prepare")

            def _ensure_session(self):
                return fake_session

        transcriber = ReuseSessionTranscriber(model_name="stub.bin")

        decoded_windows = transcriber.transcribe_inference_windows(
            np.zeros(16_000, dtype=np.float32),
            [
                InferenceWindow(
                    window_id="window-1",
                    region_index=0,
                    start_sec=0.0,
                    end_sec=1.0,
                )
            ],
        )

        assert [window.text for window in decoded_windows] == ["agenda review"]


class TestPromptCarryover:
    def test_build_prompt_carryover_uses_last_sentences_and_token_budget(self) -> None:
        carryover = build_prompt_carryover(
            DecodedWindow(
                window=InferenceWindow(
                    window_id="window-2",
                    region_index=0,
                    start_sec=18.5,
                    end_sec=35.0,
                    overlap_sec=1.5,
                ),
                text="First sentence. Second sentence. Third sentence here.",
                segments=[],
            ),
            settings=PromptCarryoverSettings(max_sentences=2, max_tokens=4),
        )

        assert carryover == "sentence. Third sentence here."

    def test_build_prompt_carryover_drops_fallback_windows(self) -> None:
        carryover = build_prompt_carryover(
            DecodedWindow(
                window=InferenceWindow(
                    window_id="window-2",
                    region_index=0,
                    start_sec=18.5,
                    end_sec=35.0,
                    overlap_sec=1.5,
                ),
                text="This should not carry.",
                segments=[],
                fallback_used=True,
            ),
            settings=PromptCarryoverSettings(),
        )

        assert carryover is None

    def test_device_name_from_system_info_prefers_enabled_backend(self) -> None:
        system_info = "WHISPER : MTL : EMBED_LIBRARY = 1 | CPU : NEON = 1 |"

        assert _device_name_from_system_info("CPU = 1 | METAL = 1") == "metal"
        assert _device_name_from_system_info(system_info) == "metal"
        assert _device_name_from_system_info("CPU = 1") == "cpu"

    def test_carryover_drop_reason_covers_disabled_and_empty_windows(self) -> None:
        decoded_window = DecodedWindow(
            window=InferenceWindow(
                window_id="window-1",
                region_index=0,
                start_sec=0.0,
                end_sec=1.0,
            ),
            text="",
            segments=[],
        )

        disabled_reason = _carryover_drop_reason(
            decoded_window,
            settings=PromptCarryoverSettings(enabled=False),
        )
        empty_text_reason = _carryover_drop_reason(
            decoded_window,
            settings=PromptCarryoverSettings(),
        )

        assert disabled_reason == "carryover_disabled"
        assert empty_text_reason == "empty_text"

    def test_sanitize_prompt_drops_missing_and_noise_only_prompts(self) -> None:
        assert _sanitize_prompt(None, max_tokens=8) == ""
        assert _sanitize_prompt("(((", max_tokens=8) == ""
