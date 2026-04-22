"""Tests for the whisper.cpp ASR adapter."""

import subprocess
from pathlib import Path
from typing import ClassVar, cast

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
from webinar_transcriber.asr.carryover import _carryover_drop_reason, _sanitize_prompt
from webinar_transcriber.asr.config import (
    DEFAULT_WHISPER_ENTROPY_THOLD,
    DEFAULT_WHISPER_LOGPROB_THOLD,
    DEFAULT_WHISPER_NO_SPEECH_THOLD,
    _read_sysctl_int,
    default_asr_threads,
)
from webinar_transcriber.asr.transcriber import _device_name_from_system_info, _PyWhisperModel
from webinar_transcriber.models import DecodedWindow, InferenceWindow, TranscriptSegment


class FakeSegment:
    def __init__(self, t0: int, t1: int, text: str) -> None:
        self.t0 = t0
        self.t1 = t1
        self.text = text


class FakeModel:
    init_calls: ClassVar[list[tuple[str, dict[str, object]]]] = []
    transcribe_calls: ClassVar[list[tuple[np.ndarray, dict[str, str | None]]]] = []
    auto_detect_calls: ClassVar[list[tuple[np.ndarray, int]]] = []
    system_info_value: ClassVar[str] = "CPU = 1"
    detected_language: ClassVar[str] = "en"
    returned_segments: ClassVar[list[list[FakeSegment]]] = [[FakeSegment(0, 100, "agenda review")]]
    init_error: ClassVar[Exception | None] = None
    transcribe_error: ClassVar[Exception | None] = None

    def __init__(self, model_name: str, **kwargs) -> None:
        type(self).init_calls.append((model_name, kwargs))
        if (init_error := type(self).init_error) is not None:
            raise init_error

    @classmethod
    def reset(cls) -> None:
        cls.init_calls = []
        cls.transcribe_calls = []
        cls.auto_detect_calls = []
        cls.system_info_value = "CPU = 1"
        cls.detected_language = "en"
        cls.returned_segments = [[FakeSegment(0, 100, "agenda review")]]
        cls.init_error = None
        cls.transcribe_error = None

    def system_info(self) -> str:
        return type(self).system_info_value

    def transcribe(self, audio_samples: np.ndarray, **kwargs: str | None) -> list[FakeSegment]:
        type(self).transcribe_calls.append((audio_samples.copy(), kwargs))
        if (transcribe_error := type(self).transcribe_error) is not None:
            raise transcribe_error
        call_index = len(type(self).transcribe_calls) - 1
        if call_index < len(type(self).returned_segments):
            return type(self).returned_segments[call_index]
        return []

    def auto_detect_language(
        self, audio_samples: np.ndarray, *, n_threads: int
    ) -> tuple[tuple[str, float], dict[str, float]]:
        type(self).auto_detect_calls.append((audio_samples.copy(), n_threads))
        return ((type(self).detected_language, 0.99), {type(self).detected_language: 0.99})


class FakeModelWithNullContext(FakeModel):
    def __init__(self, model_name: str, **kwargs) -> None:
        super().__init__(model_name, **kwargs)
        self._ctx = None


def install_fake_pywhispercpp(
    monkeypatch: pytest.MonkeyPatch,
    *,
    model_cls: type[FakeModel] = FakeModel,
) -> None:
    model_cls.reset()
    monkeypatch.setattr("webinar_transcriber.asr.transcriber._model_cls", lambda: model_cls)
    monkeypatch.setattr(
        "webinar_transcriber.asr.transcriber._suppress_pywhispercpp_download_progress",
        lambda: __import__("contextlib").nullcontext(),
    )


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

    def test_prepare_model_uses_default_model_identifier(self, monkeypatch) -> None:
        install_fake_pywhispercpp(monkeypatch)
        transcriber = WhisperCppTranscriber()

        transcriber.prepare_model()

        assert FakeModel.init_calls[0][0] == DEFAULT_WHISPER_CPP_MODEL_FILENAME
        assert transcriber.model_name == DEFAULT_WHISPER_CPP_MODEL_FILENAME

    def test_prepare_model_uses_explicit_model_identifier(self, monkeypatch) -> None:
        install_fake_pywhispercpp(monkeypatch)
        transcriber = WhisperCppTranscriber(model_name="base")

        transcriber.prepare_model()

        assert FakeModel.init_calls[0][0] == "base"
        assert transcriber.model_name == "base"

    def test_prepare_model_suppresses_pywhispercpp_download_progress(self, monkeypatch) -> None:
        install_fake_pywhispercpp(monkeypatch)
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
            "webinar_transcriber.asr.transcriber._suppress_pywhispercpp_download_progress",
            make_progress_context,
        )

        WhisperCppTranscriber().prepare_model()

        assert progress_context_calls == ["enter", "exit"]

    def test_prepare_model_passes_decode_settings(self, monkeypatch) -> None:
        install_fake_pywhispercpp(monkeypatch)

        WhisperCppTranscriber(threads=6).prepare_model()

        _, kwargs = FakeModel.init_calls[0]
        assert kwargs["n_threads"] == 6
        assert kwargs["print_realtime"] is False
        assert kwargs["print_progress"] is False
        assert kwargs["no_context"] is True
        assert kwargs["split_on_word"] is True
        assert kwargs["entropy_thold"] == DEFAULT_WHISPER_ENTROPY_THOLD
        assert kwargs["logprob_thold"] == DEFAULT_WHISPER_LOGPROB_THOLD
        assert kwargs["no_speech_thold"] == DEFAULT_WHISPER_NO_SPEECH_THOLD

    def test_prepare_model_passes_native_log_path(self, monkeypatch, tmp_path) -> None:
        install_fake_pywhispercpp(monkeypatch)
        log_path = tmp_path / "whisper-cpp.log"

        transcriber = WhisperCppTranscriber()
        transcriber.set_log_path(log_path)
        transcriber.prepare_model()

        _, kwargs = FakeModel.init_calls[0]
        assert kwargs["redirect_whispercpp_logs_to"] == str(log_path)

    def test_prepare_model_wraps_model_initialization_failures(self, monkeypatch) -> None:
        install_fake_pywhispercpp(monkeypatch)
        FakeModel.init_error = RuntimeError("boom")

        with pytest.raises(ASRProcessingError, match=r"Could not prepare whisper\.cpp model"):
            WhisperCppTranscriber().prepare_model()

    def test_prepare_model_rejects_model_without_native_context(self, monkeypatch) -> None:
        install_fake_pywhispercpp(monkeypatch, model_cls=FakeModelWithNullContext)

        with pytest.raises(ASRProcessingError, match=r"Could not prepare whisper\.cpp model"):
            WhisperCppTranscriber().prepare_model()

    def test_prepare_model_reads_runtime_details(self, monkeypatch, tmp_path) -> None:
        install_fake_pywhispercpp(monkeypatch)
        FakeModel.system_info_value = "METAL = 1"
        model_path = tmp_path / "model.bin"
        model_path.write_text("stub", encoding="utf-8")

        transcriber = WhisperCppTranscriber(model_name=str(model_path))
        transcriber.prepare_model()

        assert transcriber.device_name == "metal"
        assert transcriber.system_info == "METAL = 1"

    def test_transcriber_context_manager_clears_prepared_model(self, monkeypatch, tmp_path) -> None:
        install_fake_pywhispercpp(monkeypatch)
        model_path = tmp_path / "model.bin"
        model_path.write_text("stub", encoding="utf-8")

        with WhisperCppTranscriber(model_name=str(model_path)) as transcriber:
            transcriber.prepare_model()
            assert transcriber.system_info == "CPU = 1"

        assert transcriber.system_info is None

    def test_set_log_path_updates_transcriber(self) -> None:
        transcriber = WhisperCppTranscriber()
        log_path = Path("/tmp/whisper-cpp.log")

        transcriber.set_log_path(log_path)

        assert transcriber._log_path == log_path

    def test_whisper_cpp_transcriber_carries_input_prompt_into_decoded_windows(
        self, monkeypatch
    ) -> None:
        install_fake_pywhispercpp(monkeypatch)
        FakeModel.system_info_value = "CPU = 1 | CUDA = 1"
        FakeModel.detected_language = "ru"
        FakeModel.returned_segments = [
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
        assert len(FakeModel.auto_detect_calls) == 1
        np.testing.assert_array_equal(
            FakeModel.auto_detect_calls[0][0], np.zeros(16_000, dtype=np.float32)
        )
        assert FakeModel.auto_detect_calls[0][1] == 6
        assert FakeModel.transcribe_calls[0][1] == {}
        assert FakeModel.transcribe_calls[1][1] == {
            "initial_prompt": "agenda review",
            "language": "ru",
        }

    def test_model_cls_imports_pywhispercpp_model(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "webinar_transcriber.asr.transcriber.importlib.import_module",
            lambda _name: type("FakeModelModule", (), {"Model": FakeModel})(),
        )

        from webinar_transcriber.asr import transcriber as transcriber_module

        assert transcriber_module._model_cls() is FakeModel

    def test_suppress_pywhispercpp_download_progress_disables_tqdm(self, monkeypatch) -> None:
        class FakeUtilsModule:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def tqdm(self, *args, **kwargs):
                del args
                self.calls.append(kwargs)
                return kwargs

        fake_utils_module = FakeUtilsModule()
        monkeypatch.setattr(
            "webinar_transcriber.asr.transcriber.importlib.import_module",
            lambda _name: fake_utils_module,
        )

        from webinar_transcriber.asr import transcriber as transcriber_module

        with transcriber_module._suppress_pywhispercpp_download_progress():
            fake_utils_module.tqdm(total=1)

        assert fake_utils_module.calls == [{"disable": True, "total": 1}]

    def test_suppress_pywhispercpp_download_progress_targets_real_tqdm_attribute(
        self, monkeypatch
    ) -> None:
        fake_utils_module = type("FakeUtilsModule", (), {"tqdm": object()})()
        monkeypatch.setattr(
            "webinar_transcriber.asr.transcriber.importlib.import_module",
            lambda _name: fake_utils_module,
        )

        from webinar_transcriber.asr import transcriber as transcriber_module

        assert hasattr(
            transcriber_module.importlib.import_module("pywhispercpp.utils"),
            "tqdm",
        )

    def test_transcribe_inference_windows_clips_segment_times_to_window(self, monkeypatch) -> None:
        install_fake_pywhispercpp(monkeypatch)
        FakeModel.returned_segments = [[FakeSegment(-50, 300, "agenda review")]]
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
        self, monkeypatch
    ) -> None:
        install_fake_pywhispercpp(monkeypatch)
        transcriber = WhisperCppTranscriber()

        decoded_window = transcriber.transcribe_inference_windows(
            np.zeros(1, dtype=np.float32),
            [InferenceWindow(window_id="window-1", region_index=0, start_sec=1.0, end_sec=2.0)],
        )[0]

        assert decoded_window.segments == []
        assert FakeModel.transcribe_calls == []

    def test_transcribe_inference_windows_handles_model_without_segments(self, monkeypatch) -> None:
        install_fake_pywhispercpp(monkeypatch)
        FakeModel.returned_segments = []

        decoded_window = WhisperCppTranscriber().transcribe_inference_windows(
            np.zeros(16_000, dtype=np.float32),
            [InferenceWindow(window_id="window-1", region_index=0, start_sec=0.0, end_sec=1.0)],
        )[0]

        assert decoded_window.text == ""
        assert decoded_window.segments == []

    def test_transcribe_inference_windows_wraps_model_failures(self, monkeypatch) -> None:
        install_fake_pywhispercpp(monkeypatch)
        FakeModel.transcribe_error = RuntimeError("boom")

        with pytest.raises(ASRProcessingError, match=r"whisper\.cpp inference failed for window-1"):
            WhisperCppTranscriber().transcribe_inference_windows(
                np.zeros(16_000, dtype=np.float32),
                [InferenceWindow(window_id="window-1", region_index=0, start_sec=0.0, end_sec=1.0)],
            )

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

    def test_transcribe_inference_windows_reuses_existing_model_without_prepare(
        self, monkeypatch
    ) -> None:
        install_fake_pywhispercpp(monkeypatch)
        fake_model = FakeModel("stub.bin")

        class ReuseModelTranscriber(WhisperCppTranscriber):
            def prepare_model(self) -> None:
                raise AssertionError("should not prepare")

        transcriber = ReuseModelTranscriber(model_name="stub.bin")
        transcriber._model = cast("_PyWhisperModel", fake_model)

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
