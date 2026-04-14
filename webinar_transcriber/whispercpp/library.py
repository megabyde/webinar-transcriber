"""Thin ctypes wrapper around the whisper.cpp C API."""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import threading
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Self

import numpy as np

from webinar_transcriber.models import DecodedWindow, InferenceWindow, TranscriptSegment
from webinar_transcriber.normalized_audio import sample_index_for_time

from . import bindings as _bindings

if TYPE_CHECKING:
    from types import TracebackType

_BACKEND_PLUGIN_GLOB: Final[str] = "libggml-*.so"
_BACKEND_REGISTRATION_DONE = False
_GGML_LIBRARY_HANDLE: ctypes.CDLL | None = None
_LOG_SINK_LOCK = threading.RLock()
_LOG_SINK_PATH: Path | None = None
GPU_BACKEND_PATTERN = _bindings.GPU_BACKEND_PATTERN
_GGML_LOG_CALLBACK_TYPE = _bindings._GGML_LOG_CALLBACK_TYPE
_TICKS_PER_SECOND = _bindings._TICKS_PER_SECOND
_WHISPER_SAMPLING_GREEDY = _bindings._WHISPER_SAMPLING_GREEDY
_WhisperContextParams = _bindings._WhisperContextParams
_WhisperFullParams = _bindings._WhisperFullParams
_configure_context_params = _bindings._configure_context_params
_configure_signatures = _bindings._configure_signatures
_decode_c_string = _bindings._decode_c_string
_encode_optional_text = _bindings._encode_optional_text
_system_info_supports_gpu = _bindings._system_info_supports_gpu


class WhisperCppError(RuntimeError):
    """Raised when the whisper.cpp runtime cannot be used successfully."""


@dataclass(frozen=True)
class WhisperCppRuntimeDetails:
    """Runtime metadata surfaced by the library wrapper."""

    library_path: str
    system_info: str


class WhisperCppSession:
    """Prepared whisper.cpp model/state pair reused across inference calls."""

    def __init__(
        self,
        library: WhisperCppLibrary,
        *,
        context: ctypes.c_void_p,
        state: ctypes.c_void_p,
        runtime_details: WhisperCppRuntimeDetails,
    ) -> None:
        self._library = library
        self._context = context
        self._state = state
        self._runtime_details = runtime_details
        self._closed = False

    @property
    def runtime_details(self) -> WhisperCppRuntimeDetails:
        return self._runtime_details

    def __enter__(self) -> Self:
        return self

    def decode_window(
        self,
        audio_samples: np.ndarray,
        window: InferenceWindow,
        *,
        threads: int,
        prompt: str | None = None,
        language_hint: str | None = None,
    ) -> DecodedWindow:
        return self._library._decode_window(
            self._context,
            self._state,
            np.ascontiguousarray(audio_samples, dtype=np.float32),
            window,
            threads=threads,
            prompt=prompt,
            language_hint=language_hint,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._library._lib.whisper_free_state(self._state)
        self._library._lib.whisper_free(self._context)
        self._closed = True

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown cleanup
        with suppress(Exception):
            self.close()


class WhisperCppLibrary:
    """A small object wrapper over the whisper.cpp shared library."""

    def __init__(self, library_path: Path | None = None, *, log_path: Path | None = None) -> None:
        resolved_path = resolve_library_path(library_path)
        self._library_path = resolved_path
        self._log_path = log_path
        _set_log_sink_path(self._log_path)
        _load_backend_plugins()
        self._lib = ctypes.CDLL(str(resolved_path))
        _configure_signatures(self._lib)
        _configure_library_logging(self._lib)

    @property
    def library_path(self) -> str:
        return str(self._library_path)

    def system_info(self) -> str:
        return _decode_c_string(self._lib.whisper_print_system_info())

    def runtime_details(self) -> WhisperCppRuntimeDetails:
        return WhisperCppRuntimeDetails(
            library_path=self.library_path, system_info=self.system_info()
        )

    def create_session(self, model_path: Path) -> WhisperCppSession:
        context_params = self._lib.whisper_context_default_params()
        runtime_details = self.runtime_details()
        _configure_context_params(context_params, system_info=runtime_details.system_info)
        _set_log_sink_path(self._log_path)
        context = self._lib.whisper_init_from_file_with_params(
            os.fspath(model_path).encode("utf-8"), context_params
        )
        if not context:
            raise WhisperCppError(f"Failed to initialize whisper.cpp model: {model_path}")

        state = self._lib.whisper_init_state(context)
        if not state:
            self._lib.whisper_free(context)
            raise WhisperCppError("Failed to initialize whisper.cpp runtime state.")

        return WhisperCppSession(
            self, context=context, state=state, runtime_details=runtime_details
        )

    def _decode_window(
        self,
        context: ctypes.c_void_p,
        state: ctypes.c_void_p,
        audio_samples: np.ndarray,
        window: InferenceWindow,
        *,
        threads: int,
        prompt: str | None,
        language_hint: str | None,
    ) -> DecodedWindow:
        start_index = sample_index_for_time(window.start_sec)
        end_index = min(len(audio_samples), sample_index_for_time(window.end_sec))
        window_samples = np.ascontiguousarray(
            audio_samples[start_index:end_index], dtype=np.float32
        )

        if window_samples.size == 0:
            return DecodedWindow(window=window, language=language_hint)

        params_ptr = self._lib.whisper_full_default_params_by_ref(_WHISPER_SAMPLING_GREEDY)
        try:
            params = params_ptr.contents
            params.n_threads = max(1, threads)
            params.no_context = True
            params.no_timestamps = False
            params.print_special = False
            params.print_progress = False
            params.print_realtime = False
            params.print_timestamps = False
            params.token_timestamps = False
            params.split_on_word = True
            params.max_len = 0
            params.max_tokens = 0
            params.single_segment = False
            params.tdrz_enable = False
            params.initial_prompt = _encode_optional_text(prompt)
            params.carry_initial_prompt = False
            params.language = _encode_optional_text(language_hint)
            params.detect_language = language_hint is None
            params.greedy.best_of = 1
            params.beam_search.patience = 1.0

            _set_log_sink_path(self._log_path)
            result = self._lib.whisper_full_with_state(
                context,
                state,
                params,
                window_samples.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                int(window_samples.size),
            )
            if result != 0:
                raise WhisperCppError(f"whisper.cpp inference failed for {window.window_id}.")

            detected_language = _decode_c_string(
                self._lib.whisper_lang_str(self._lib.whisper_full_lang_id_from_state(state))
            )
            segment_count = self._lib.whisper_full_n_segments_from_state(state)
            segments: list[TranscriptSegment] = []
            for segment_index in range(segment_count):
                start_sec = window.start_sec + (
                    self._lib.whisper_full_get_segment_t0_from_state(state, segment_index)
                    / _TICKS_PER_SECOND
                )
                end_sec = window.start_sec + (
                    self._lib.whisper_full_get_segment_t1_from_state(state, segment_index)
                    / _TICKS_PER_SECOND
                )
                text = _decode_c_string(
                    self._lib.whisper_full_get_segment_text_from_state(state, segment_index)
                ).strip()
                segments.append(
                    TranscriptSegment(
                        id=f"{window.window_id}-segment-{segment_index + 1}",
                        text=text,
                        start_sec=max(window.start_sec, start_sec),
                        end_sec=max(start_sec, end_sec),
                    )
                )

            return DecodedWindow(
                window=window,
                text=" ".join(segment.text for segment in segments if segment.text).strip(),
                segments=segments,
                fallback_used=False,
                language=detected_language,
            )
        finally:
            self._lib.whisper_free_params(params_ptr)


def resolve_library_path(library_path: Path | None = None) -> Path:
    """Return the shared-library path for whisper.cpp."""
    if library_path is not None:
        if library_path.exists():
            return library_path
        raise WhisperCppError(f"whisper.cpp shared library does not exist: {library_path}")

    explicit_path = os.environ.get("WHISPER_CPP_LIB")
    if explicit_path:
        explicit_library_path = Path(explicit_path).expanduser()
        if explicit_library_path.exists():
            return explicit_library_path
        raise WhisperCppError(
            "WHISPER_CPP_LIB points to a missing whisper.cpp shared library: "
            f"{explicit_library_path}"
        )

    candidate_paths = [
        Path("build/src/libwhisper.dylib"),
        Path("build/src/libwhisper.so"),
        Path("build/libwhisper.dylib"),
        Path("build/libwhisper.so"),
        Path("/opt/homebrew/lib/libwhisper.dylib"),
        Path("/usr/local/lib/libwhisper.dylib"),
        Path("/usr/local/lib/libwhisper.so"),
        Path("/usr/lib/libwhisper.so"),
        Path("/usr/lib/x86_64-linux-gnu/libwhisper.so"),
        Path("/usr/lib/aarch64-linux-gnu/libwhisper.so"),
    ]
    for candidate_path in candidate_paths:
        if candidate_path.exists():
            return candidate_path

    discovered_library = ctypes.util.find_library("whisper")
    if discovered_library:  # pragma: no cover - host linker discovery
        return Path(discovered_library)

    raise WhisperCppError(
        "Could not find a whisper.cpp shared library. "
        "Set WHISPER_CPP_LIB or build/install libwhisper."
    )


def _load_backend_plugins() -> None:
    global _BACKEND_REGISTRATION_DONE, _GGML_LIBRARY_HANDLE
    if _BACKEND_REGISTRATION_DONE:  # pragma: no cover - process-global native state
        return

    ggml_library_path = _resolve_ggml_library_path()
    if ggml_library_path is None:  # pragma: no cover - optional native dependency
        _BACKEND_REGISTRATION_DONE = True
        return

    _GGML_LIBRARY_HANDLE = ctypes.CDLL(
        str(ggml_library_path), mode=getattr(ctypes, "RTLD_GLOBAL", 0)
    )
    _configure_ggml_logging(_GGML_LIBRARY_HANDLE)
    _GGML_LIBRARY_HANDLE.ggml_backend_load_all_from_path.argtypes = [ctypes.c_char_p]
    _GGML_LIBRARY_HANDLE.ggml_backend_load_all_from_path.restype = None

    plugin_dirs = {plugin_path.parent for plugin_path in _candidate_backend_plugin_paths()}
    for plugin_dir in sorted(plugin_dirs):
        _GGML_LIBRARY_HANDLE.ggml_backend_load_all_from_path(os.fspath(plugin_dir).encode("utf-8"))
    _BACKEND_REGISTRATION_DONE = True


def _candidate_backend_plugin_paths() -> list[Path]:
    explicit_plugin_dir = os.environ.get("GGML_BACKEND_LIB_DIR")
    candidate_dirs: list[Path] = []
    if explicit_plugin_dir:
        candidate_dirs.append(Path(explicit_plugin_dir).expanduser())

    candidate_dirs.extend([
        Path("/opt/homebrew/Cellar/ggml"),
        Path("/usr/local/Cellar/ggml"),
        Path("/usr/local/lib/ggml"),
        Path("/usr/lib/ggml"),
        Path("/usr/local/libexec/ggml"),
        Path("/usr/libexec/ggml"),
    ])

    plugin_paths: list[Path] = []
    for candidate_dir in candidate_dirs:
        if not candidate_dir.exists():
            continue
        if candidate_dir.name == "ggml":
            plugin_paths.extend(sorted(candidate_dir.rglob(_BACKEND_PLUGIN_GLOB)))
        else:
            plugin_paths.extend(sorted(candidate_dir.glob("*/libexec/libggml-*.so")))
    return plugin_paths


def _resolve_ggml_library_path() -> Path | None:
    candidate_paths = [
        Path("/opt/homebrew/opt/ggml/lib/libggml.0.dylib"),
        Path("/opt/homebrew/lib/libggml.0.dylib"),
        Path("/usr/local/opt/ggml/lib/libggml.0.dylib"),
        Path("/usr/local/lib/libggml.so"),
        Path("/usr/lib/libggml.so"),
        Path("/usr/lib/x86_64-linux-gnu/libggml.so"),
        Path("/usr/lib/aarch64-linux-gnu/libggml.so"),
    ]
    for candidate_path in candidate_paths:
        if candidate_path.exists():
            return candidate_path

    discovered_library = ctypes.util.find_library("ggml")
    if discovered_library:  # pragma: no cover - host linker discovery
        return Path(discovered_library)
    return None


def _set_log_sink_path(log_path: Path | None) -> None:
    global _LOG_SINK_PATH
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    _LOG_SINK_PATH = log_path


def _append_log_message(text: bytes | None) -> None:
    if _LOG_SINK_PATH is None or not text:
        return
    with _LOG_SINK_LOCK, _LOG_SINK_PATH.open("ab") as log_file:
        log_file.write(text)


@_GGML_LOG_CALLBACK_TYPE
def _library_log_callback(_level: int, text: bytes | None, _user_data) -> None:
    _append_log_message(text)


def _configure_ggml_logging(library: ctypes.CDLL) -> None:
    library.ggml_log_set.argtypes = [_GGML_LOG_CALLBACK_TYPE, ctypes.c_void_p]
    library.ggml_log_set.restype = None
    library.ggml_log_set(_library_log_callback, None)


def _configure_library_logging(library: ctypes.CDLL) -> None:
    library.whisper_log_set(_library_log_callback, None)
