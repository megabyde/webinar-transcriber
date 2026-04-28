"""Tests for the whisper.cpp ASR adapter."""

import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from webinar_transcriber.asr import (
    DEFAULT_WHISPER_CPP_MODEL_EXAMPLE,
    DEFAULT_WHISPER_CPP_MODEL_FILENAME,
    ASRProcessingError,
    PromptCarryoverSettings,
    WhisperCppTranscriber,
    build_prompt_carryover,
)
from webinar_transcriber.asr.carryover import _sanitize_prompt
from webinar_transcriber.asr.config import (
    DEFAULT_WHISPER_ENTROPY_THOLD,
    DEFAULT_WHISPER_LOGPROB_THOLD,
    DEFAULT_WHISPER_NO_SPEECH_THOLD,
    default_asr_threads,
)
from webinar_transcriber.asr.transcriber import _device_name_from_system_info
from webinar_transcriber.models import DecodedWindow, InferenceWindow, TranscriptSegment


class FakeSegment:
    def __init__(self, t0: int, t1: int, text: str) -> None:
        self.t0 = t0
        self.t1 = t1
        self.text = text


class FakeModel:
    def __init__(self) -> None:
        self.init_calls: list[tuple[str, dict[str, object]]] = []
        self.transcribe_calls: list[tuple[np.ndarray, dict[str, str | None]]] = []
        self.auto_detect_calls: list[tuple[np.ndarray, int]] = []
        self.system_info_value = "CPU = 1"
        self.detected_language = "en"
        self.returned_segments = [[FakeSegment(0, 100, "agenda review")]]
        self.init_error: Exception | None = None
        self.transcribe_error: Exception | None = None

    def __call__(self, model_name: str, **kwargs) -> "FakeModel":
        self.init_calls.append((model_name, kwargs))
        if (init_error := self.init_error) is not None:
            raise init_error
        return self

    def system_info(self) -> str:
        return self.system_info_value

    def transcribe(self, audio_samples: np.ndarray, **kwargs: str | None) -> list[FakeSegment]:
        self.transcribe_calls.append((audio_samples.copy(), kwargs))
        if (transcribe_error := self.transcribe_error) is not None:
            raise transcribe_error
        call_index = len(self.transcribe_calls) - 1
        if call_index < len(self.returned_segments):
            return self.returned_segments[call_index]
        return []

    def auto_detect_language(
        self, audio_samples: np.ndarray, *, n_threads: int
    ) -> tuple[tuple[str, float], dict[str, float]]:
        self.auto_detect_calls.append((audio_samples.copy(), n_threads))
        return ((self.detected_language, 0.99), {self.detected_language: 0.99})


class FakeModelWithNullContext(FakeModel):
    def __call__(self, model_name: str, **kwargs) -> "FakeModelWithNullContext":
        super().__call__(model_name, **kwargs)
        self._ctx = None
        return self


def install_fake_pywhispercpp(
    monkeypatch: pytest.MonkeyPatch,
    *,
    model: FakeModel | None = None,
) -> FakeModel:
    fake_model = model or FakeModel()
    monkeypatch.setattr("webinar_transcriber.asr.transcriber._model_cls", lambda: fake_model)
    monkeypatch.setattr(
        "webinar_transcriber.asr.transcriber._disable_tqdm_progress",
        lambda: __import__("contextlib").nullcontext(),
    )
    return fake_model


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch) -> FakeModel:
    return install_fake_pywhispercpp(monkeypatch)


class TestWhisperCppTranscriber:
    def test_default_model_uses_builtin_identifier(self) -> None:
        transcriber = WhisperCppTranscriber()

        assert transcriber.model_name == DEFAULT_WHISPER_CPP_MODEL_FILENAME

    def test_prepare_model_requires_model_file(self, tmp_path) -> None:
        with pytest.raises(ASRProcessingError, match="model file does not exist") as error:
            WhisperCppTranscriber(model_name=str(tmp_path / "missing.bin")).prepare_model()

        message = str(error.value)
        assert "--asr-model" in message
        assert "README.md" in message

    def test_prepare_model_missing_default_model_is_actionable(self, tmp_path) -> None:
        with pytest.raises(ASRProcessingError, match="model file does not exist") as error:
            WhisperCppTranscriber(
                model_name=str(tmp_path / "missing-default-model.bin")
            ).prepare_model()

        message = str(error.value)
        assert "Download a whisper.cpp model there" in message
        assert DEFAULT_WHISPER_CPP_MODEL_EXAMPLE in message

    def test_prepare_model_uses_default_model_identifier(self, fake_model: FakeModel) -> None:
        transcriber = WhisperCppTranscriber()

        transcriber.prepare_model()

        assert fake_model.init_calls[0][0] == DEFAULT_WHISPER_CPP_MODEL_FILENAME
        assert transcriber.model_name == DEFAULT_WHISPER_CPP_MODEL_FILENAME

    def test_prepare_model_uses_explicit_model_identifier(self, fake_model: FakeModel) -> None:
        transcriber = WhisperCppTranscriber(model_name="base")

        transcriber.prepare_model()

        assert fake_model.init_calls[0][0] == "base"
        assert transcriber.model_name == "base"

    def test_prepare_model_disables_tqdm_progress_during_model_setup(
        self, monkeypatch, fake_model: FakeModel
    ) -> None:
        progress_context_calls: list[str] = []

        class _ProgressContext:
            def __enter__(self) -> None:
                progress_context_calls.append("enter")

            def __exit__(self, exc_type, exc, traceback) -> None:
                del exc_type, exc, traceback
                progress_context_calls.append("exit")

        def make_progress_context() -> _ProgressContext:
            return _ProgressContext()

        monkeypatch.setattr(
            "webinar_transcriber.asr.transcriber._disable_tqdm_progress",
            make_progress_context,
        )

        WhisperCppTranscriber().prepare_model()

        assert progress_context_calls == ["enter", "exit"]

    def test_prepare_model_passes_decode_settings(self, fake_model: FakeModel) -> None:
        WhisperCppTranscriber(threads=6).prepare_model()

        _, kwargs = fake_model.init_calls[0]
        assert kwargs["n_threads"] == 6
        assert kwargs["print_realtime"] is False
        assert kwargs["print_progress"] is False
        assert kwargs["no_context"] is True
        assert kwargs["split_on_word"] is True
        assert kwargs["entropy_thold"] == DEFAULT_WHISPER_ENTROPY_THOLD
        assert kwargs["logprob_thold"] == DEFAULT_WHISPER_LOGPROB_THOLD
        assert kwargs["no_speech_thold"] == DEFAULT_WHISPER_NO_SPEECH_THOLD

    def test_prepare_model_redirects_native_output_to_log(
        self, monkeypatch, tmp_path, capfd
    ) -> None:
        class NativeOutputModel(FakeModel):
            def __call__(self, model_name: str, **kwargs) -> "NativeOutputModel":
                os.write(1, b"native stdout\n")
                os.write(2, b"native stderr\n")
                super().__call__(model_name, **kwargs)
                return self

        native_model = NativeOutputModel()
        install_fake_pywhispercpp(monkeypatch, model=native_model)
        log_path = tmp_path / "whisper-cpp.log"

        transcriber = WhisperCppTranscriber()
        transcriber.set_log_path(log_path)
        transcriber.prepare_model()

        _, kwargs = native_model.init_calls[0]
        assert "redirect_whispercpp_logs_to" not in kwargs
        captured = capfd.readouterr()
        assert captured.out == ""
        assert captured.err == ""
        assert log_path.read_text(encoding="utf-8") == "native stdout\nnative stderr\n"

    def test_prepare_model_wraps_model_initialization_failures(self, fake_model: FakeModel) -> None:
        fake_model.init_error = RuntimeError("boom")

        with pytest.raises(ASRProcessingError, match=r"Could not prepare whisper\.cpp model"):
            WhisperCppTranscriber().prepare_model()

    def test_prepare_model_rejects_model_without_native_context(self, monkeypatch) -> None:
        install_fake_pywhispercpp(monkeypatch, model=FakeModelWithNullContext())

        with pytest.raises(ASRProcessingError, match=r"Could not prepare whisper\.cpp model"):
            WhisperCppTranscriber().prepare_model()

    def test_prepare_model_reads_runtime_details(self, fake_model: FakeModel, tmp_path) -> None:
        fake_model.system_info_value = "METAL = 1"
        model_path = tmp_path / "model.bin"
        model_path.write_text("stub", encoding="utf-8")

        transcriber = WhisperCppTranscriber(model_name=str(model_path))
        transcriber.prepare_model()

        assert transcriber.device_name == "metal"
        assert transcriber.system_info == "METAL = 1"

    def test_transcriber_context_manager_clears_prepared_model(
        self, fake_model: FakeModel, tmp_path
    ) -> None:
        model_path = tmp_path / "model.bin"
        model_path.write_text("stub", encoding="utf-8")

        with WhisperCppTranscriber(model_name=str(model_path)) as transcriber:
            transcriber.prepare_model()
            assert transcriber.system_info == "CPU = 1"

        assert transcriber.system_info is None

    def test_close_redirects_native_teardown_output_to_log(
        self, monkeypatch, tmp_path, capfd
    ) -> None:
        class NativeTeardownRuntime(FakeModel):
            def __del__(self) -> None:
                os.write(2, b"native teardown\n")

        class NativeTeardownModel(FakeModel):
            def __call__(self, model_name: str, **kwargs) -> NativeTeardownRuntime:
                super().__call__(model_name, **kwargs)
                return NativeTeardownRuntime()

        install_fake_pywhispercpp(monkeypatch, model=NativeTeardownModel())
        log_path = tmp_path / "whisper-cpp.log"
        transcriber = WhisperCppTranscriber()
        transcriber.set_log_path(log_path)
        transcriber.prepare_model()

        transcriber.close()

        captured = capfd.readouterr()
        assert captured.err == ""
        assert log_path.read_text(encoding="utf-8") == "native teardown\n"
        assert transcriber.system_info is None

    def test_redirect_native_output_falls_back_without_file_descriptors(
        self, monkeypatch, tmp_path
    ) -> None:
        from webinar_transcriber.asr import transcriber as transcriber_module

        log_path = tmp_path / "whisper-cpp.log"
        monkeypatch.setattr(transcriber_module, "_duplicated_output_fds", list)

        with transcriber_module._redirect_native_output(log_path):
            print("python stdout")

        assert log_path.read_text(encoding="utf-8") == "python stdout\n"

    def test_output_fds_skips_streams_without_file_descriptors(self, monkeypatch) -> None:
        from webinar_transcriber.asr import transcriber as transcriber_module

        class StreamWithoutFileDescriptor:
            def fileno(self) -> int:
                raise OSError

        stream = StreamWithoutFileDescriptor()
        monkeypatch.setattr(transcriber_module.sys, "stdout", stream)
        monkeypatch.setattr(transcriber_module.sys, "stderr", stream)

        assert transcriber_module._output_fds() == [1, 2]

    def test_duplicated_output_fds_skips_invalid_descriptors(self, monkeypatch) -> None:
        from webinar_transcriber.asr import transcriber as transcriber_module

        monkeypatch.setattr(transcriber_module, "_output_fds", lambda: [-1])

        assert transcriber_module._duplicated_output_fds() == []

    def test_set_log_path_updates_transcriber(self) -> None:
        transcriber = WhisperCppTranscriber()
        log_path = Path("/tmp/whisper-cpp.log")

        transcriber.set_log_path(log_path)

        assert transcriber._log_path == log_path

    def test_whisper_cpp_transcriber_carries_input_prompt_into_decoded_windows(
        self, fake_model: FakeModel
    ) -> None:
        fake_model.system_info_value = "CPU = 1 | CUDA = 1"
        fake_model.detected_language = "ru"
        fake_model.returned_segments = [
            [FakeSegment(0, 100, "agenda review")],
            [FakeSegment(0, 100, "agenda review follow up")],
        ]
        progress_updates: list[float] = []

        transcriber = WhisperCppTranscriber(threads=6)
        decoded_windows = transcriber.transcribe_inference_windows(
            np.zeros(32_000, dtype=np.float32),
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
        assert transcriber.device_name == "cuda"
        assert len(fake_model.auto_detect_calls) == 1
        np.testing.assert_array_equal(
            fake_model.auto_detect_calls[0][0], np.zeros(16_000, dtype=np.float32)
        )
        assert fake_model.auto_detect_calls[0][1] == 6
        assert fake_model.transcribe_calls[0][1] == {}
        assert fake_model.transcribe_calls[1][1] == {
            "initial_prompt": "agenda review",
            "language": "ru",
        }

    def test_transcribe_inference_windows_uses_forced_language(self, fake_model: FakeModel) -> None:
        fake_model.detected_language = "ru"

        decoded_windows = WhisperCppTranscriber(language="en").transcribe_inference_windows(
            np.zeros(16_000, dtype=np.float32),
            [InferenceWindow(window_id="window-1", region_index=0, start_sec=0.0, end_sec=1.0)],
        )

        assert [window.language for window in decoded_windows] == ["en"]
        assert fake_model.auto_detect_calls == []
        assert fake_model.transcribe_calls[0][1] == {"language": "en"}

    def test_model_cls_imports_pywhispercpp_model(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "webinar_transcriber.asr.transcriber.importlib.import_module",
            lambda _name: type("FakeModelModule", (), {"Model": FakeModel})(),
        )

        from webinar_transcriber.asr import transcriber as transcriber_module

        assert transcriber_module._model_cls() is FakeModel

    def test_disable_tqdm_progress_sets_env_var(self, monkeypatch) -> None:
        from webinar_transcriber.asr import transcriber as transcriber_module

        monkeypatch.delenv("TQDM_DISABLE", raising=False)

        with transcriber_module._disable_tqdm_progress():
            assert os.environ["TQDM_DISABLE"] == "1"

        assert "TQDM_DISABLE" not in os.environ

    def test_disable_tqdm_progress_restores_previous_env_value(self, monkeypatch) -> None:
        from webinar_transcriber.asr import transcriber as transcriber_module

        monkeypatch.setenv("TQDM_DISABLE", "0")

        with transcriber_module._disable_tqdm_progress():
            assert os.environ["TQDM_DISABLE"] == "1"

        assert os.environ["TQDM_DISABLE"] == "0"

    def test_disable_tqdm_progress_redirects_tqdm_stderr(self, monkeypatch, capsys) -> None:
        from tqdm import tqdm

        from webinar_transcriber.asr import transcriber as transcriber_module

        monkeypatch.delenv("TQDM_DISABLE", raising=False)

        with transcriber_module._disable_tqdm_progress():
            progress_bar = tqdm(total=1)
            progress_bar.close()

        assert capsys.readouterr().err == ""

    def test_transcribe_inference_windows_clips_segment_times_to_window(
        self, fake_model: FakeModel
    ) -> None:
        fake_model.returned_segments = [[FakeSegment(-50, 300, "agenda review")]]
        transcriber = WhisperCppTranscriber()

        decoded_window = transcriber.transcribe_inference_windows(
            np.zeros(32_000, dtype=np.float32),
            [InferenceWindow(window_id="window-1", region_index=0, start_sec=1.0, end_sec=2.0)],
        )[0]

        assert decoded_window.segments == [
            TranscriptSegment(
                id="window-1-segment-1",
                text="agenda review",
                start_sec=1.0,
                end_sec=2.0,
            )
        ]

    def test_transcribe_inference_windows_returns_empty_window_for_empty_audio(
        self, fake_model: FakeModel
    ) -> None:
        transcriber = WhisperCppTranscriber()

        decoded_window = transcriber.transcribe_inference_windows(
            np.zeros(1, dtype=np.float32),
            [InferenceWindow(window_id="window-1", region_index=0, start_sec=1.0, end_sec=2.0)],
        )[0]

        assert decoded_window.segments == []
        assert fake_model.transcribe_calls == []

    def test_transcribe_inference_windows_handles_model_without_segments(
        self, fake_model: FakeModel
    ) -> None:
        fake_model.returned_segments = []

        decoded_window = WhisperCppTranscriber().transcribe_inference_windows(
            np.zeros(16_000, dtype=np.float32),
            [InferenceWindow(window_id="window-1", region_index=0, start_sec=0.0, end_sec=1.0)],
        )[0]

        assert decoded_window.text == ""
        assert decoded_window.segments == []

    def test_transcribe_inference_windows_wraps_model_failures(self, fake_model: FakeModel) -> None:
        fake_model.transcribe_error = RuntimeError("boom")

        with pytest.raises(ASRProcessingError, match=r"whisper\.cpp inference failed for window-1"):
            WhisperCppTranscriber().transcribe_inference_windows(
                np.zeros(16_000, dtype=np.float32),
                [InferenceWindow(window_id="window-1", region_index=0, start_sec=0.0, end_sec=1.0)],
            )

    def test_default_asr_threads_uses_cpu_count(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("webinar_transcriber.asr.config.os.cpu_count", lambda: 8)

        assert default_asr_threads() == 8

    def test_default_asr_threads_falls_back_to_positive_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("webinar_transcriber.asr.config.os.cpu_count", lambda: None)

        assert default_asr_threads() == 4

    def test_transcriber_properties_are_safe_before_runtime_is_prepared(self) -> None:
        transcriber = WhisperCppTranscriber(threads=0)

        assert transcriber.threads == 1
        assert transcriber.system_info is None

    def test_transcribe_inference_windows_reuses_existing_model_without_prepare(
        self, fake_model: FakeModel
    ) -> None:
        fake_model("stub.bin")

        class ReuseModelTranscriber(WhisperCppTranscriber):
            def prepare_model(self) -> None:
                raise AssertionError("should not prepare")

        transcriber = ReuseModelTranscriber(model_name="stub.bin")
        transcriber._model = cast("Any", fake_model)

        decoded_windows = transcriber.transcribe_inference_windows(
            np.zeros(16_000, dtype=np.float32),
            [InferenceWindow(window_id="window-1", region_index=0, start_sec=0.0, end_sec=1.0)],
        )

        assert [window.text for window in decoded_windows] == ["agenda review"]

    def test_ensure_model_raises_when_prepare_model_does_not_initialize_model(self) -> None:
        class BrokenTranscriber(WhisperCppTranscriber):
            def prepare_model(self) -> None:
                self._model = None

        with pytest.raises(ASRProcessingError, match="model was not initialized"):
            BrokenTranscriber(model_name="stub.bin")._ensure_model()

    def test_resolve_model_name_raises_when_explicit_model_path_is_uninitialized(self) -> None:
        transcriber = WhisperCppTranscriber(model_name="stub.bin")
        transcriber._uses_default_model_name = False
        transcriber._configured_model_path = None

        with pytest.raises(ASRProcessingError, match=r"--asr-model path was not initialized"):
            transcriber._resolve_model_name()


class TestPromptCarryover:
    def test_build_prompt_carryover_uses_last_sentences_and_token_budget(self) -> None:
        carryover = build_prompt_carryover(
            DecodedWindow(
                window=InferenceWindow(
                    window_id="window-2",
                    region_index=0,
                    start_sec=18.5,
                    end_sec=35.0,
                ),
                text="First sentence. Second sentence. Third sentence here.",
                segments=[],
            ),
            settings=PromptCarryoverSettings(max_sentences=2, max_tokens=4),
        )

        assert carryover == "sentence. Third sentence here."

    def test_build_prompt_carryover_drops_empty_windows(self) -> None:
        carryover = build_prompt_carryover(
            DecodedWindow(
                window=InferenceWindow(
                    window_id="window-2",
                    region_index=0,
                    start_sec=18.5,
                    end_sec=35.0,
                ),
                text="",
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

    def test_build_prompt_carryover_drops_when_disabled(self) -> None:
        carryover = build_prompt_carryover(
            DecodedWindow(
                window=InferenceWindow(
                    window_id="window-1",
                    region_index=0,
                    start_sec=0.0,
                    end_sec=1.0,
                ),
                text="Carry this.",
                segments=[],
            ),
            settings=PromptCarryoverSettings(enabled=False),
        )

        assert carryover is None

    def test_build_prompt_carryover_drops_noise_only_text(self) -> None:
        carryover = build_prompt_carryover(
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

        assert carryover is None

    def test_sanitize_prompt_drops_missing_and_noise_only_prompts(self) -> None:
        assert _sanitize_prompt(None, max_tokens=8) == ""
        assert _sanitize_prompt("(((", max_tokens=8) == ""
