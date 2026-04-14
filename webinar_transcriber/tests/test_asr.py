"""Tests for the whisper.cpp ASR adapter."""

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from webinar_transcriber.asr import (
    DEFAULT_WHISPER_CPP_MODEL_EXAMPLE,
    DEFAULT_WHISPER_CPP_MODEL_FILENAME,
    DEFAULT_WHISPER_CPP_MODEL_REPO,
    PromptCarryoverSettings,
    WhisperCppTranscriber,
    build_prompt_carryover,
)
from webinar_transcriber.asr.carryover import _carryover_drop_reason, _sanitize_prompt
from webinar_transcriber.asr.config import _read_sysctl_int, default_asr_threads
from webinar_transcriber.asr.transcriber import (
    _device_name_from_system_info,
    _download_default_whisper_cpp_model,
)
from webinar_transcriber.models import DecodedWindow, InferenceWindow, TranscriptSegment

if TYPE_CHECKING:
    from webinar_transcriber.whispercpp import WhisperCppSession


class TestWhisperCppTranscriber:
    def test_default_model_uses_hugging_face_reference(self) -> None:
        transcriber = WhisperCppTranscriber()

        assert (
            transcriber.model_name
            == f"{DEFAULT_WHISPER_CPP_MODEL_REPO}/{DEFAULT_WHISPER_CPP_MODEL_FILENAME}"
        )

    def test_prepare_model_requires_model_file(self, tmp_path) -> None:
        with pytest.raises(RuntimeError, match="model file does not exist") as error:
            WhisperCppTranscriber(model_name=str(tmp_path / "missing.bin")).prepare_model()
        message = str(error.value)
        assert "--asr-model" in message
        assert "README.md" in message

    def test_prepare_model_missing_default_model_is_actionable(self, tmp_path, monkeypatch) -> None:
        with pytest.raises(RuntimeError, match="model file does not exist") as error:
            WhisperCppTranscriber(
                model_name=str(tmp_path / "missing-default-model.bin")
            ).prepare_model()
        message = str(error.value)
        assert "Download a whisper.cpp model there" in message
        assert str(DEFAULT_WHISPER_CPP_MODEL_EXAMPLE) in message

    def test_prepare_model_downloads_default_model_into_hf_cache(
        self, tmp_path, monkeypatch
    ) -> None:
        cached_model_path = tmp_path / "hf-cache" / "ggml-large-v3-turbo.bin"
        cached_model_path.parent.mkdir(parents=True)
        cached_model_path.write_text("stub", encoding="utf-8")
        created_paths: list[str] = []

        class FakeLibrary:
            def __init__(self, _library_path, log_path=None) -> None:
                self._library_path = _library_path
                self._log_path = log_path

            def create_session(self, model_path):
                from webinar_transcriber.whispercpp import WhisperCppRuntimeDetails

                created_paths.append(str(model_path))

                class FakeSession:
                    runtime_details = WhisperCppRuntimeDetails(
                        library_path="/usr/local/lib/libwhisper.so", system_info="CPU = 1"
                    )

                    def close(self) -> None:
                        return None

                return FakeSession()

        monkeypatch.setattr(
            "webinar_transcriber.asr.transcriber._download_default_whisper_cpp_model",
            lambda: cached_model_path,
        )
        monkeypatch.setattr("webinar_transcriber.asr.transcriber.WhisperCppLibrary", FakeLibrary)

        transcriber = WhisperCppTranscriber()
        transcriber.prepare_model()

        assert created_paths == [str(cached_model_path)]
        assert transcriber.model_name == str(cached_model_path)

    def test_prepare_model_reports_default_download_failure(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "webinar_transcriber.asr.transcriber._download_default_whisper_cpp_model",
            lambda: (_ for _ in ()).throw(RuntimeError("download failed")),
        )

        with pytest.raises(RuntimeError, match="download failed") as error:
            WhisperCppTranscriber().prepare_model()

        message = str(error.value)
        assert "Hugging Face cache defaults" in message
        assert str(DEFAULT_WHISPER_CPP_MODEL_EXAMPLE) in message

    def test_prepare_model_does_not_download_explicit_model_path(
        self, tmp_path, monkeypatch
    ) -> None:
        explicit_model_path = tmp_path / "missing.bin"
        download_calls: list[None] = []
        monkeypatch.setattr(
            "webinar_transcriber.asr.transcriber._download_default_whisper_cpp_model",
            lambda: download_calls.append(None),  # type: ignore[return-value]
        )

        with pytest.raises(RuntimeError, match="model file does not exist"):
            WhisperCppTranscriber(model_name=str(explicit_model_path)).prepare_model()

        assert download_calls == []

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
                        library_path="/usr/local/lib/libwhisper.so", system_info="METAL = 1"
                    )

                    def close(self) -> None:
                        return None

                return FakeSession()

        monkeypatch.setattr("webinar_transcriber.asr.transcriber.WhisperCppLibrary", FakeLibrary)

        transcriber = WhisperCppTranscriber(model_name=str(model_path))
        transcriber.prepare_model()

        assert transcriber.device_name == "metal"
        assert transcriber.system_info == "METAL = 1"
        assert transcriber.library_path == "/usr/local/lib/libwhisper.so"

    def test_prepare_model_raises_when_model_resolution_returns_none(self, monkeypatch) -> None:
        monkeypatch.setattr(
            WhisperCppTranscriber,
            "_resolve_model_path",
            lambda self: None,  # type: ignore[return-value]
        )

        with pytest.raises(RuntimeError, match=r"returned no whisper\.cpp model path"):
            WhisperCppTranscriber().prepare_model()

    def test_transcriber_context_manager_closes_prepared_session(
        self, monkeypatch, tmp_path
    ) -> None:
        from webinar_transcriber.whispercpp import WhisperCppRuntimeDetails

        model_path = tmp_path / "model.bin"
        model_path.write_text("stub", encoding="utf-8")
        close_calls: list[str] = []

        class FakeSession:
            runtime_details = WhisperCppRuntimeDetails(
                library_path="/usr/local/lib/libwhisper.so", system_info="CPU = 1"
            )

            def close(self) -> None:
                close_calls.append("closed")

        class FakeLibrary:
            def __init__(self, _library_path, log_path=None) -> None:
                self._library_path = _library_path
                self._log_path = log_path

            def create_session(self, _model_path):
                return FakeSession()

        monkeypatch.setattr("webinar_transcriber.asr.transcriber.WhisperCppLibrary", FakeLibrary)

        with WhisperCppTranscriber(model_name=str(model_path)) as transcriber:
            transcriber.prepare_model()
            assert transcriber.system_info == "CPU = 1"

        assert close_calls == ["closed"]

    def test_whisper_cpp_transcriber_carries_input_prompt_into_decoded_windows(
        self, monkeypatch, tmp_path
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
                        library_path="/usr/local/lib/libwhisper.so", system_info="CPU = 1"
                    )

                    def decode_window(
                        self, audio_samples, window, *, threads, prompt=None, language_hint=None
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
                                    id="segment-1", text="agenda review", start_sec=0.0, end_sec=1.0
                                )
                            ],
                        )

                    def close(self) -> None:
                        return None

                return FakeSession()

        monkeypatch.setattr("webinar_transcriber.asr.transcriber.WhisperCppLibrary", FakeLibrary)

        transcriber = WhisperCppTranscriber(model_name=str(model_path), threads=6)
        decoded_windows = transcriber.transcribe_inference_windows(
            np.zeros(16_000, dtype=np.float32),
            [
                InferenceWindow(window_id="window-1", region_index=0, start_sec=0.0, end_sec=1.0),
                InferenceWindow(window_id="window-2", region_index=0, start_sec=1.0, end_sec=2.0),
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
        self, monkeypatch: pytest.MonkeyPatch, stdout: str
    ) -> None:
        monkeypatch.setattr(
            "webinar_transcriber.asr.config.subprocess.run",
            lambda *_args, **_kwargs: type("Result", (), {"stdout": stdout})(),
        )

        assert _read_sysctl_int("hw.physicalcpu") is None

    def test_read_sysctl_int_returns_none_on_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_run(*_args, **_kwargs):
            raise subprocess.TimeoutExpired(cmd=["sysctl"], timeout=1.0)

        monkeypatch.setattr("webinar_transcriber.asr.config.subprocess.run", fake_run)

        assert _read_sysctl_int("hw.physicalcpu") is None

    def test_default_asr_threads_uses_sysctl_priority_then_cpu_count(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        values = {
            "hw.perflevel0.physicalcpu": None,
            "hw.physicalcpu": 6,
        }
        monkeypatch.setattr(
            "webinar_transcriber.asr.config._read_sysctl_int", lambda name: values[name]
        )
        monkeypatch.setattr("webinar_transcriber.asr.config.os.cpu_count", lambda: 8)

        assert default_asr_threads() == 6

    def test_transcriber_properties_are_safe_before_runtime_is_prepared(self) -> None:
        transcriber = WhisperCppTranscriber(threads=0)

        assert transcriber.threads == 1
        assert transcriber.system_info is None
        assert transcriber.library_path is None

    def test_transcribe_inference_windows_reuses_existing_session_without_prepare(self) -> None:
        class FakeSession:
            def decode_window(
                self, audio_samples, window, *, threads, prompt=None, language_hint=None
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
            [InferenceWindow(window_id="window-1", region_index=0, start_sec=0.0, end_sec=1.0)],
        )

        assert [window.text for window in decoded_windows] == ["agenda review"]

    def test_ensure_session_raises_when_prepare_model_does_not_initialize_session(self) -> None:
        class BrokenTranscriber(WhisperCppTranscriber):
            def prepare_model(self) -> None:
                self._session = None

        with pytest.raises(RuntimeError, match="session was not initialized"):
            BrokenTranscriber(model_name="stub.bin")._ensure_session()

    def test_download_default_model_requires_huggingface_hub_dependency(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "webinar_transcriber.asr.transcriber.importlib.import_module",
            lambda _name: (_ for _ in ()).throw(ModuleNotFoundError("missing")),
        )

        with pytest.raises(RuntimeError, match="huggingface_hub is not installed"):
            WhisperCppTranscriber()._resolve_model_path()

    def test_download_default_model_wraps_backend_failures(self, monkeypatch) -> None:
        class FakeHub:
            @staticmethod
            def hf_hub_download(**_kwargs):
                raise RuntimeError("backend failed")

        monkeypatch.setattr(
            "webinar_transcriber.asr.transcriber.importlib.import_module",
            lambda _name: FakeHub,
        )

        with pytest.raises(
            RuntimeError, match=r"Automatic download of the default whisper\.cpp model"
        ):
            WhisperCppTranscriber()._resolve_model_path()

    def test_download_default_model_returns_downloaded_path(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "webinar_transcriber.asr.transcriber.importlib.import_module",
            lambda _name: type(
                "FakeHub", (), {"hf_hub_download": staticmethod(lambda **_kwargs: "/tmp/model.bin")}
            ),
        )

        assert _download_default_whisper_cpp_model() == Path("/tmp/model.bin")


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

    def test_build_prompt_carryover_drops_known_hallucination_phrases(self) -> None:
        carryover = build_prompt_carryover(
            DecodedWindow(
                window=InferenceWindow(
                    window_id="window-3",
                    region_index=0,
                    start_sec=35.0,
                    end_sec=52.0,
                    overlap_sec=1.5,
                ),
                text="Thank you for watching, please like and share.",
                segments=[],
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
                window_id="window-1", region_index=0, start_sec=0.0, end_sec=1.0
            ),
            text="",
            segments=[],
        )

        disabled_reason = _carryover_drop_reason(
            decoded_window, settings=PromptCarryoverSettings(enabled=False)
        )
        empty_text_reason = _carryover_drop_reason(
            decoded_window, settings=PromptCarryoverSettings()
        )

        assert disabled_reason == "carryover_disabled"
        assert empty_text_reason == "empty_text"

    def test_carryover_drop_reason_detects_repeated_hallucination_text(self) -> None:
        repeated_reason = _carryover_drop_reason(
            DecodedWindow(
                window=InferenceWindow(
                    window_id="window-4",
                    region_index=0,
                    start_sec=52.0,
                    end_sec=70.0,
                    overlap_sec=1.5,
                ),
                text="plan plan plan plan plan plan review review review review",
                segments=[],
            ),
            settings=PromptCarryoverSettings(),
        )

        assert repeated_reason == "hallucination_detected"

    def test_carryover_drop_reason_keeps_punctuation_only_text(self) -> None:
        reason = _carryover_drop_reason(
            DecodedWindow(
                window=InferenceWindow(
                    window_id="window-5",
                    region_index=0,
                    start_sec=0.0,
                    end_sec=1.0,
                ),
                text="(((",
                segments=[],
            ),
            settings=PromptCarryoverSettings(),
        )

        assert reason is None

    def test_sanitize_prompt_drops_missing_and_noise_only_prompts(self) -> None:
        assert _sanitize_prompt(None, max_tokens=8) == ""
        assert _sanitize_prompt("(((", max_tokens=8) == ""
