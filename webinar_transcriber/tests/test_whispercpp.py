"""Tests for the low-level whisper.cpp ctypes wrapper."""

import ctypes
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from webinar_transcriber.models import InferenceWindow
from webinar_transcriber.whispercpp import (
    WhisperCppError,
    WhisperCppLibrary,
    WhisperCppSession,
    _decode_c_string,
    _encode_optional_text,
    _library_log_callback,
    _load_backend_plugins,
    _resolve_ggml_library_path,
    _set_log_sink_path,
    _system_info_supports_gpu,
    _WhisperContextParams,
    _WhisperFullParams,
    resolve_library_path,
)


class FakeFunction:
    """Callable helper that also accepts ctypes signature attributes."""

    def __init__(self, callback):
        self._callback = callback
        self.argtypes = None
        self.restype = None

    def __call__(self, *args, **kwargs):
        return self._callback(*args, **kwargs)


class FakeCDLL:
    """Tiny fake shared library for wrapper tests."""

    def __init__(self) -> None:
        self._full_params = _WhisperFullParams()
        self.context_params_seen: list[_WhisperContextParams] = []
        self._segment_texts = [b" agenda review ", b" next step "]
        self._segment_t0 = [0, 150]
        self._segment_t1 = [150, 300]

        self.whisper_context_default_params = FakeFunction(lambda: _WhisperContextParams())
        self.whisper_init_from_file_with_params = FakeFunction(self._init_from_file_with_params)
        self.whisper_init_state = FakeFunction(lambda *_args: 2)
        self.whisper_free = FakeFunction(lambda *_args: None)
        self.whisper_free_state = FakeFunction(lambda *_args: None)
        self.whisper_full_default_params_by_ref = FakeFunction(
            lambda *_args: ctypes.pointer(self._full_params)
        )
        self.whisper_free_params = FakeFunction(lambda *_args: None)
        self.whisper_full_with_state = FakeFunction(lambda *_args: 0)
        self.whisper_full_n_segments_from_state = FakeFunction(lambda *_args: 2)
        self.whisper_full_lang_id_from_state = FakeFunction(lambda *_args: 0)
        self.whisper_full_get_segment_t0_from_state = FakeFunction(
            lambda _state, index: self._segment_t0[index]
        )
        self.whisper_full_get_segment_t1_from_state = FakeFunction(
            lambda _state, index: self._segment_t1[index]
        )
        self.whisper_full_get_segment_text_from_state = FakeFunction(
            lambda _state, index: self._segment_texts[index]
        )
        self.whisper_lang_str = FakeFunction(lambda _lang_id: b"en")
        self.whisper_log_set = FakeFunction(lambda *_args: None)
        self.whisper_print_system_info = FakeFunction(lambda: b"METAL = 1")

    def _init_from_file_with_params(self, _model_path, context_params):
        self.context_params_seen.append(context_params)
        return 1


def test_resolve_library_path_prefers_env_var(monkeypatch, tmp_path) -> None:
    library_path = tmp_path / "libwhisper.dylib"
    library_path.write_text("stub", encoding="utf-8")
    monkeypatch.setenv("WHISPER_CPP_LIB", str(library_path))

    assert resolve_library_path() == library_path


def test_resolve_library_path_raises_for_missing_env_target(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("WHISPER_CPP_LIB", str(tmp_path / "missing.so"))

    with pytest.raises(WhisperCppError, match="WHISPER_CPP_LIB points to a missing"):
        resolve_library_path()


def test_resolve_library_path_uses_direct_argument_and_errors_when_missing(tmp_path) -> None:
    library_path = tmp_path / "libwhisper.dylib"
    library_path.write_text("stub", encoding="utf-8")

    assert resolve_library_path(library_path) == library_path

    with pytest.raises(WhisperCppError, match="shared library does not exist"):
        resolve_library_path(tmp_path / "missing.dylib")


def test_whisper_cpp_library_decodes_window_with_fake_cdll(monkeypatch, tmp_path) -> None:
    library_path = tmp_path / "libwhisper.dylib"
    library_path.write_text("stub", encoding="utf-8")
    model_path = tmp_path / "model.bin"
    model_path.write_text("model", encoding="utf-8")
    monkeypatch.setattr("webinar_transcriber.whispercpp._load_backend_plugins", lambda: None)
    monkeypatch.setattr(
        "webinar_transcriber.whispercpp.ctypes.CDLL",
        lambda _path, mode=0: FakeCDLL(),
    )

    library = WhisperCppLibrary(library_path)
    decoded_window = library.decode_window(
        model_path,
        np.zeros(16_000, dtype=np.float32),
        InferenceWindow(window_id="window-1", region_index=0, start_sec=0.0, end_sec=3.0),
        threads=4,
    )

    assert library.runtime_details().system_info == "METAL = 1"
    assert decoded_window.language == "en"
    assert [segment.text for segment in decoded_window.segments] == [
        "agenda review",
        "next step",
    ]
    assert [segment.end_sec for segment in decoded_window.segments] == [1.5, 3.0]


def test_whisper_cpp_library_disables_gpu_when_system_info_has_no_backend(
    monkeypatch, tmp_path
) -> None:
    library_path = tmp_path / "libwhisper.dylib"
    library_path.write_text("stub", encoding="utf-8")
    model_path = tmp_path / "model.bin"
    model_path.write_text("model", encoding="utf-8")
    fake_cdll = FakeCDLL()
    fake_cdll.whisper_print_system_info = FakeFunction(
        lambda: b"WHISPER : COREML = 0 | OPENVINO = 0 | "
    )
    monkeypatch.setattr("webinar_transcriber.whispercpp._load_backend_plugins", lambda: None)
    monkeypatch.setattr(
        "webinar_transcriber.whispercpp.ctypes.CDLL",
        lambda _path, mode=0: fake_cdll,
    )

    library = WhisperCppLibrary(library_path)
    library.decode_window(
        model_path,
        np.zeros(16_000, dtype=np.float32),
        InferenceWindow(window_id="window-1", region_index=0, start_sec=0.0, end_sec=3.0),
        threads=4,
    )

    assert fake_cdll.context_params_seen
    assert fake_cdll.context_params_seen[0].use_gpu is False
    assert fake_cdll.context_params_seen[0].flash_attn is False


def test_load_backend_plugins_loads_candidate_shared_objects(monkeypatch, tmp_path) -> None:
    plugin_path = tmp_path / "libggml-metal.so"
    plugin_path.write_text("stub", encoding="utf-8")
    ggml_library_path = tmp_path / "libggml.so"
    ggml_library_path.write_text("stub", encoding="utf-8")

    class FakeGgmlLibrary:
        def __init__(self) -> None:
            self.ggml_log_set = FakeFunction(lambda *_args: None)
            self.ggml_backend_load_all_from_path = Mock()

    fake_ggml_library = FakeGgmlLibrary()

    monkeypatch.setattr("webinar_transcriber.whispercpp._BACKEND_REGISTRATION_DONE", False)
    monkeypatch.setattr("webinar_transcriber.whispercpp._GGML_LIBRARY_HANDLE", None)
    monkeypatch.setattr(
        "webinar_transcriber.whispercpp._candidate_backend_plugin_paths",
        lambda: [plugin_path],
    )
    monkeypatch.setattr(
        "webinar_transcriber.whispercpp._resolve_ggml_library_path",
        lambda: ggml_library_path,
    )
    with patch(
        "webinar_transcriber.whispercpp.ctypes.CDLL",
        return_value=fake_ggml_library,
    ) as cdll_mock:
        _load_backend_plugins()

    cdll_mock.assert_called_once_with(
        str(ggml_library_path),
        mode=getattr(ctypes, "RTLD_GLOBAL", 0),
    )
    fake_ggml_library.ggml_backend_load_all_from_path.assert_called_once_with(
        str(plugin_path.parent).encode("utf-8")
    )


def test_load_backend_plugins_returns_when_no_ggml_library(monkeypatch) -> None:
    monkeypatch.setattr("webinar_transcriber.whispercpp._BACKEND_REGISTRATION_DONE", False)
    monkeypatch.setattr("webinar_transcriber.whispercpp._GGML_LIBRARY_HANDLE", None)
    monkeypatch.setattr(
        "webinar_transcriber.whispercpp._resolve_ggml_library_path",
        lambda: None,
    )

    _load_backend_plugins()

    from webinar_transcriber import whispercpp

    assert whispercpp._BACKEND_REGISTRATION_DONE is True
    assert whispercpp._GGML_LIBRARY_HANDLE is None


def test_resolve_ggml_library_path_uses_find_library(monkeypatch) -> None:
    monkeypatch.setattr(
        "webinar_transcriber.whispercpp.ctypes.util.find_library", lambda _name: "libggml.so"
    )
    monkeypatch.setattr("pathlib.Path.exists", lambda self: False)

    assert _resolve_ggml_library_path() == Path("libggml.so")


def test_system_info_helpers_and_encoding_helpers() -> None:
    assert _system_info_supports_gpu("WHISPER : MTL : EMBED_LIBRARY = 1 | CPU : NEON = 1 |")
    assert _system_info_supports_gpu("CUDA = 1")
    assert not _system_info_supports_gpu("WHISPER : COREML = 0 | OPENVINO = 0 | CPU : NEON = 1 |")
    assert _encode_optional_text(None) is None
    assert _encode_optional_text("") is None
    assert _encode_optional_text("ru") == b"ru"
    assert _decode_c_string(None) == ""
    assert _decode_c_string(b"\xfftest") == "\ufffdtest"


def test_library_log_callback_writes_to_log_file(tmp_path) -> None:
    log_path = tmp_path / "native.log"
    _set_log_sink_path(log_path)

    _library_log_callback(0, b"native line\n", None)

    assert "native line" in log_path.read_text(encoding="utf-8")
    _set_log_sink_path(None)


def test_whisper_cpp_library_raises_when_context_init_fails(monkeypatch, tmp_path) -> None:
    library_path = tmp_path / "libwhisper.dylib"
    library_path.write_text("stub", encoding="utf-8")
    model_path = tmp_path / "model.bin"
    model_path.write_text("model", encoding="utf-8")
    fake_cdll = FakeCDLL()
    fake_cdll.whisper_init_from_file_with_params = FakeFunction(lambda *_args: 0)
    monkeypatch.setattr("webinar_transcriber.whispercpp._load_backend_plugins", lambda: None)
    monkeypatch.setattr(
        "webinar_transcriber.whispercpp.ctypes.CDLL",
        lambda _path, mode=0: fake_cdll,
    )

    library = WhisperCppLibrary(library_path)

    with pytest.raises(WhisperCppError, match=r"Failed to initialize whisper\.cpp model"):
        library.decode_window(
            model_path,
            np.zeros(16_000, dtype=np.float32),
            InferenceWindow(window_id="window-1", region_index=0, start_sec=0.0, end_sec=1.0),
            threads=2,
        )


def test_whisper_cpp_library_raises_when_state_init_fails(monkeypatch, tmp_path) -> None:
    library_path = tmp_path / "libwhisper.dylib"
    library_path.write_text("stub", encoding="utf-8")
    model_path = tmp_path / "model.bin"
    model_path.write_text("model", encoding="utf-8")
    fake_cdll = FakeCDLL()
    fake_cdll.whisper_init_state = FakeFunction(lambda *_args: 0)
    monkeypatch.setattr("webinar_transcriber.whispercpp._load_backend_plugins", lambda: None)
    monkeypatch.setattr(
        "webinar_transcriber.whispercpp.ctypes.CDLL",
        lambda _path, mode=0: fake_cdll,
    )

    library = WhisperCppLibrary(library_path)

    with pytest.raises(WhisperCppError, match=r"Failed to initialize whisper\.cpp runtime state"):
        library.decode_window(
            model_path,
            np.zeros(16_000, dtype=np.float32),
            InferenceWindow(window_id="window-1", region_index=0, start_sec=0.0, end_sec=1.0),
            threads=2,
        )


def test_decode_window_handles_empty_input_and_inference_failure(monkeypatch, tmp_path) -> None:
    library_path = tmp_path / "libwhisper.dylib"
    library_path.write_text("stub", encoding="utf-8")
    fake_cdll = FakeCDLL()
    monkeypatch.setattr("webinar_transcriber.whispercpp._load_backend_plugins", lambda: None)
    monkeypatch.setattr(
        "webinar_transcriber.whispercpp.ctypes.CDLL",
        lambda _path, mode=0: fake_cdll,
    )
    library = WhisperCppLibrary(library_path)

    empty_window = library._decode_window(
        ctypes.c_void_p(1),
        ctypes.c_void_p(2),
        np.zeros(0, dtype=np.float32),
        InferenceWindow(window_id="window-1", region_index=0, start_sec=0.0, end_sec=1.0),
        threads=2,
        prompt=None,
        language_hint="ru",
    )
    assert empty_window.language == "ru"
    assert empty_window.segments == []

    fake_cdll.whisper_full_with_state = FakeFunction(lambda *_args: 1)
    with pytest.raises(WhisperCppError, match="inference failed for window-2"):
        library._decode_window(
            ctypes.c_void_p(1),
            ctypes.c_void_p(2),
            np.zeros(16_000, dtype=np.float32),
            InferenceWindow(window_id="window-2", region_index=0, start_sec=0.0, end_sec=1.0),
            threads=2,
            prompt=None,
            language_hint=None,
        )


def test_session_destructor_swallows_close_failures() -> None:
    session = object.__new__(WhisperCppSession)

    def failing_close() -> None:
        raise RuntimeError("shutdown")

    session.close = failing_close  # type: ignore[method-assign]

    session.__del__()
