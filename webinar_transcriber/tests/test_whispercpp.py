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
    resolve_library_path,
)
from webinar_transcriber.whispercpp.bindings import (
    _decode_c_string,
    _encode_optional_text,
    _system_info_supports_gpu,
    _WhisperContextParams,
    _WhisperFullParams,
)
from webinar_transcriber.whispercpp.library import (
    _candidate_backend_plugin_paths,
    _library_log_callback,
    _load_backend_plugins,
    _resolve_ggml_library_path,
    _set_log_sink_path,
)


class TestResolveLibraryPath:
    def test_resolve_library_path_prefers_env_var(self, monkeypatch, tmp_path) -> None:
        library_path = tmp_path / "libwhisper.dylib"
        library_path.write_text("stub", encoding="utf-8")
        monkeypatch.setenv("WHISPER_CPP_LIB", str(library_path))

        assert resolve_library_path() == library_path

    def test_resolve_library_path_raises_for_missing_env_target(
        self, monkeypatch, tmp_path
    ) -> None:
        monkeypatch.setenv("WHISPER_CPP_LIB", str(tmp_path / "missing.so"))

        with pytest.raises(WhisperCppError, match="WHISPER_CPP_LIB points to a missing"):
            resolve_library_path()

    def test_resolve_library_path_uses_direct_argument_and_errors_when_missing(
        self, tmp_path
    ) -> None:
        library_path = tmp_path / "libwhisper.dylib"
        library_path.write_text("stub", encoding="utf-8")

        assert resolve_library_path(library_path) == library_path

        with pytest.raises(WhisperCppError, match="shared library does not exist"):
            resolve_library_path(tmp_path / "missing.dylib")

    def test_resolve_library_path_uses_candidate_default_path(self, monkeypatch) -> None:
        monkeypatch.delenv("WHISPER_CPP_LIB", raising=False)
        monkeypatch.setattr(
            "pathlib.Path.exists",
            lambda self: self == Path("build/libwhisper.so"),
        )

        assert resolve_library_path() == Path("build/libwhisper.so")

    def test_resolve_library_path_uses_find_library_and_raises_when_not_found(
        self, monkeypatch
    ) -> None:
        monkeypatch.delenv("WHISPER_CPP_LIB", raising=False)
        monkeypatch.setattr("pathlib.Path.exists", lambda self: False)
        monkeypatch.setattr(
            "webinar_transcriber.whispercpp.library.ctypes.util.find_library",
            lambda _name: "libwhisper.so",
        )

        assert resolve_library_path() == Path("libwhisper.so")

        monkeypatch.setattr(
            "webinar_transcriber.whispercpp.library.ctypes.util.find_library",
            lambda _name: None,
        )

        with pytest.raises(WhisperCppError, match=r"Could not find a whisper\.cpp shared library"):
            resolve_library_path()


class TestWhisperCppLibrary:
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

            self.whisper_context_default_params = TestWhisperCppLibrary.FakeFunction(
                lambda: _WhisperContextParams()
            )
            self.whisper_init_from_file_with_params = TestWhisperCppLibrary.FakeFunction(
                self._init_from_file_with_params
            )
            self.whisper_init_state = TestWhisperCppLibrary.FakeFunction(lambda *_args: 2)
            self.whisper_free = TestWhisperCppLibrary.FakeFunction(lambda *_args: None)
            self.whisper_free_state = TestWhisperCppLibrary.FakeFunction(lambda *_args: None)
            self.whisper_full_default_params_by_ref = TestWhisperCppLibrary.FakeFunction(
                lambda *_args: ctypes.pointer(self._full_params)
            )
            self.whisper_free_params = TestWhisperCppLibrary.FakeFunction(lambda *_args: None)
            self.whisper_full_with_state = TestWhisperCppLibrary.FakeFunction(lambda *_args: 0)
            self.whisper_full_n_segments_from_state = TestWhisperCppLibrary.FakeFunction(
                lambda *_args: 2
            )
            self.whisper_full_lang_id_from_state = TestWhisperCppLibrary.FakeFunction(
                lambda *_args: 0
            )
            self.whisper_full_get_segment_t0_from_state = TestWhisperCppLibrary.FakeFunction(
                lambda _state, index: self._segment_t0[index]
            )
            self.whisper_full_get_segment_t1_from_state = TestWhisperCppLibrary.FakeFunction(
                lambda _state, index: self._segment_t1[index]
            )
            self.whisper_full_get_segment_text_from_state = TestWhisperCppLibrary.FakeFunction(
                lambda _state, index: self._segment_texts[index]
            )
            self.whisper_lang_str = TestWhisperCppLibrary.FakeFunction(lambda _lang_id: b"en")
            self.whisper_log_set = TestWhisperCppLibrary.FakeFunction(lambda *_args: None)
            self.whisper_print_system_info = TestWhisperCppLibrary.FakeFunction(
                lambda: b"METAL = 1"
            )

        def _init_from_file_with_params(self, _model_path, context_params):
            self.context_params_seen.append(context_params)
            return 1

    def test_whisper_cpp_library_decodes_window_with_fake_cdll(self, monkeypatch, tmp_path) -> None:
        library_path = tmp_path / "libwhisper.dylib"
        library_path.write_text("stub", encoding="utf-8")
        model_path = tmp_path / "model.bin"
        model_path.write_text("model", encoding="utf-8")
        monkeypatch.setattr(
            "webinar_transcriber.whispercpp.library._load_backend_plugins", lambda: None
        )
        monkeypatch.setattr(
            "webinar_transcriber.whispercpp.library.ctypes.CDLL",
            lambda _path, mode=0: self.FakeCDLL(),
        )

        library = WhisperCppLibrary(library_path)
        session = library.create_session(model_path)
        decoded_window = session.decode_window(
            np.zeros(16_000, dtype=np.float32),
            InferenceWindow(window_id="window-1", region_index=0, start_sec=0.0, end_sec=3.0),
            threads=4,
        )
        session.close()

        assert library.runtime_details().system_info == "METAL = 1"
        assert decoded_window.language == "en"
        segment_texts = [segment.text for segment in decoded_window.segments]

        assert segment_texts == ["agenda review", "next step"]
        assert [segment.end_sec for segment in decoded_window.segments] == [1.5, 3.0]

    def test_whisper_cpp_session_context_manager_closes_native_handles(
        self, monkeypatch, tmp_path
    ) -> None:
        library_path = tmp_path / "libwhisper.dylib"
        library_path.write_text("stub", encoding="utf-8")
        model_path = tmp_path / "model.bin"
        model_path.write_text("model", encoding="utf-8")
        fake_cdll = self.FakeCDLL()
        released_handles: list[tuple[str, int]] = []
        fake_cdll.whisper_free_state = self.FakeFunction(
            lambda state: released_handles.append(("state", state))
        )
        fake_cdll.whisper_free = self.FakeFunction(
            lambda context: released_handles.append(("context", context))
        )
        monkeypatch.setattr(
            "webinar_transcriber.whispercpp.library._load_backend_plugins", lambda: None
        )
        monkeypatch.setattr(
            "webinar_transcriber.whispercpp.library.ctypes.CDLL", lambda _path, mode=0: fake_cdll
        )

        library = WhisperCppLibrary(library_path)
        with library.create_session(model_path) as session:
            assert session.runtime_details.system_info == "METAL = 1"

        assert released_handles == [("state", 2), ("context", 1)]

    def test_whisper_cpp_library_disables_gpu_when_system_info_has_no_backend(
        self, monkeypatch, tmp_path
    ) -> None:
        library_path = tmp_path / "libwhisper.dylib"
        library_path.write_text("stub", encoding="utf-8")
        model_path = tmp_path / "model.bin"
        model_path.write_text("model", encoding="utf-8")
        fake_cdll = self.FakeCDLL()
        fake_cdll.whisper_print_system_info = self.FakeFunction(
            lambda: b"WHISPER : COREML = 0 | OPENVINO = 0 | "
        )
        monkeypatch.setattr(
            "webinar_transcriber.whispercpp.library._load_backend_plugins", lambda: None
        )
        monkeypatch.setattr(
            "webinar_transcriber.whispercpp.library.ctypes.CDLL", lambda _path, mode=0: fake_cdll
        )

        library = WhisperCppLibrary(library_path)
        session = library.create_session(model_path)
        session.close()

        assert fake_cdll.context_params_seen
        assert fake_cdll.context_params_seen[0].use_gpu is False
        assert fake_cdll.context_params_seen[0].flash_attn is False


class TestBackendPluginLoading:
    def test_load_backend_plugins_loads_candidate_shared_objects(
        self, monkeypatch, tmp_path
    ) -> None:
        plugin_path = tmp_path / "libggml-metal.so"
        plugin_path.write_text("stub", encoding="utf-8")
        ggml_library_path = tmp_path / "libggml.so"
        ggml_library_path.write_text("stub", encoding="utf-8")

        class FakeGgmlLibrary:
            def __init__(self) -> None:
                self.ggml_log_set = TestWhisperCppLibrary.FakeFunction(lambda *_args: None)
                self.ggml_backend_load_all_from_path = Mock()

        fake_ggml_library = FakeGgmlLibrary()

        monkeypatch.setattr(
            "webinar_transcriber.whispercpp.library._BACKEND_REGISTRATION_DONE", False
        )
        monkeypatch.setattr("webinar_transcriber.whispercpp.library._GGML_LIBRARY_HANDLE", None)
        monkeypatch.setattr(
            "webinar_transcriber.whispercpp.library._candidate_backend_plugin_paths",
            lambda: [plugin_path],
        )
        monkeypatch.setattr(
            "webinar_transcriber.whispercpp.library._resolve_ggml_library_path",
            lambda: ggml_library_path,
        )
        with patch(
            "webinar_transcriber.whispercpp.library.ctypes.CDLL", return_value=fake_ggml_library
        ) as cdll_mock:
            _load_backend_plugins()

        cdll_mock.assert_called_once_with(
            str(ggml_library_path), mode=getattr(ctypes, "RTLD_GLOBAL", 0)
        )
        fake_ggml_library.ggml_backend_load_all_from_path.assert_called_once_with(
            str(plugin_path.parent).encode("utf-8")
        )

    def test_load_backend_plugins_returns_when_no_ggml_library(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "webinar_transcriber.whispercpp.library._BACKEND_REGISTRATION_DONE", False
        )
        monkeypatch.setattr("webinar_transcriber.whispercpp.library._GGML_LIBRARY_HANDLE", None)
        monkeypatch.setattr(
            "webinar_transcriber.whispercpp.library._resolve_ggml_library_path", lambda: None
        )

        _load_backend_plugins()

        from webinar_transcriber.whispercpp import library

        assert library._BACKEND_REGISTRATION_DONE is True
        assert library._GGML_LIBRARY_HANDLE is None

    def test_resolve_ggml_library_path_uses_find_library(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "webinar_transcriber.whispercpp.library.ctypes.util.find_library",
            lambda _name: "libggml.so",
        )
        monkeypatch.setattr("pathlib.Path.exists", lambda self: False)

        assert _resolve_ggml_library_path() == Path("libggml.so")

    def test_candidate_backend_plugin_paths_covers_explicit_and_globbed_layouts(
        self, monkeypatch, tmp_path
    ) -> None:
        explicit_ggml_dir = tmp_path / "ggml"
        explicit_ggml_plugin = explicit_ggml_dir / "nested" / "libggml-metal.so"
        explicit_ggml_plugin.parent.mkdir(parents=True)
        explicit_ggml_plugin.write_text("stub", encoding="utf-8")

        explicit_root_dir = tmp_path / "plugins"
        globbed_plugin = explicit_root_dir / "version" / "libexec" / "libggml-cuda.so"
        globbed_plugin.parent.mkdir(parents=True)
        globbed_plugin.write_text("stub", encoding="utf-8")

        monkeypatch.setenv("GGML_BACKEND_LIB_DIR", str(explicit_ggml_dir))
        plugin_paths = _candidate_backend_plugin_paths()
        assert explicit_ggml_plugin in plugin_paths

        monkeypatch.setenv("GGML_BACKEND_LIB_DIR", str(explicit_root_dir))
        plugin_paths = _candidate_backend_plugin_paths()
        assert globbed_plugin in plugin_paths

    def test_resolve_ggml_library_path_uses_candidate_and_returns_none_when_absent(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            "pathlib.Path.exists",
            lambda self: self == Path("/usr/local/lib/libggml.so"),
        )
        assert _resolve_ggml_library_path() == Path("/usr/local/lib/libggml.so")

        monkeypatch.setattr("pathlib.Path.exists", lambda self: False)
        monkeypatch.setattr(
            "webinar_transcriber.whispercpp.library.ctypes.util.find_library",
            lambda _name: None,
        )
        assert _resolve_ggml_library_path() is None


class TestWhisperCppHelpers:
    @pytest.mark.parametrize(
        ("system_info", "expected"),
        [
            ("WHISPER : MTL : EMBED_LIBRARY = 1 | CPU : NEON = 1 |", True),
            ("CUDA = 1", True),
            ("WHISPER : COREML = 0 | OPENVINO = 0 | CPU : NEON = 1 |", False),
        ],
    )
    def test_system_info_supports_gpu(self, system_info: str, expected: bool) -> None:
        assert _system_info_supports_gpu(system_info) == expected

    @pytest.mark.parametrize(("text", "expected"), [(None, None), ("", None), ("ru", b"ru")])
    def test_encode_optional_text(self, text: str | None, expected: bytes | None) -> None:
        assert _encode_optional_text(text) == expected

    @pytest.mark.parametrize(("value", "expected"), [(None, ""), (b"\xfftest", "\ufffdtest")])
    def test_decode_c_string(self, value: bytes | None, expected: str) -> None:
        assert _decode_c_string(value) == expected

    def test_library_log_callback_writes_to_log_file(self, tmp_path) -> None:
        log_path = tmp_path / "native.log"
        _set_log_sink_path(log_path)

        _library_log_callback(0, b"native line\n", None)

        assert "native line" in log_path.read_text(encoding="utf-8")
        _set_log_sink_path(None)

    def test_library_log_callback_ignores_missing_sink_and_empty_text(self, tmp_path) -> None:
        _set_log_sink_path(None)
        _library_log_callback(0, b"ignored\n", None)

        log_path = tmp_path / "native.log"
        _set_log_sink_path(log_path)
        _library_log_callback(0, None, None)

        assert not log_path.exists()
        _set_log_sink_path(None)


class TestWhisperCppLibraryFailures:
    def test_whisper_cpp_library_raises_when_context_init_fails(
        self, monkeypatch, tmp_path
    ) -> None:
        library_path = tmp_path / "libwhisper.dylib"
        library_path.write_text("stub", encoding="utf-8")
        model_path = tmp_path / "model.bin"
        model_path.write_text("model", encoding="utf-8")
        fake_cdll = TestWhisperCppLibrary.FakeCDLL()
        fake_cdll.whisper_init_from_file_with_params = TestWhisperCppLibrary.FakeFunction(
            lambda *_args: 0
        )
        monkeypatch.setattr(
            "webinar_transcriber.whispercpp.library._load_backend_plugins", lambda: None
        )
        monkeypatch.setattr(
            "webinar_transcriber.whispercpp.library.ctypes.CDLL", lambda _path, mode=0: fake_cdll
        )

        library = WhisperCppLibrary(library_path)

        with pytest.raises(WhisperCppError, match=r"Failed to initialize whisper\.cpp model"):
            library.create_session(model_path)

    def test_whisper_cpp_library_raises_when_state_init_fails(self, monkeypatch, tmp_path) -> None:
        library_path = tmp_path / "libwhisper.dylib"
        library_path.write_text("stub", encoding="utf-8")
        model_path = tmp_path / "model.bin"
        model_path.write_text("model", encoding="utf-8")
        fake_cdll = TestWhisperCppLibrary.FakeCDLL()
        fake_cdll.whisper_init_state = TestWhisperCppLibrary.FakeFunction(lambda *_args: 0)
        monkeypatch.setattr(
            "webinar_transcriber.whispercpp.library._load_backend_plugins", lambda: None
        )
        monkeypatch.setattr(
            "webinar_transcriber.whispercpp.library.ctypes.CDLL", lambda _path, mode=0: fake_cdll
        )

        library = WhisperCppLibrary(library_path)

        with pytest.raises(
            WhisperCppError, match=r"Failed to initialize whisper\.cpp runtime state"
        ):
            library.create_session(model_path)

    def test_decode_window_handles_empty_input_and_inference_failure(
        self, monkeypatch, tmp_path
    ) -> None:
        library_path = tmp_path / "libwhisper.dylib"
        library_path.write_text("stub", encoding="utf-8")
        fake_cdll = TestWhisperCppLibrary.FakeCDLL()
        monkeypatch.setattr(
            "webinar_transcriber.whispercpp.library._load_backend_plugins", lambda: None
        )
        monkeypatch.setattr(
            "webinar_transcriber.whispercpp.library.ctypes.CDLL", lambda _path, mode=0: fake_cdll
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

        fake_cdll.whisper_full_with_state = TestWhisperCppLibrary.FakeFunction(lambda *_args: 1)
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


class TestWhisperCppSession:
    def test_session_destructor_swallows_close_failures(self) -> None:
        session = object.__new__(WhisperCppSession)

        def failing_close() -> None:
            raise RuntimeError("shutdown")

        session.close = failing_close  # type: ignore[method-assign]

        session.__del__()
