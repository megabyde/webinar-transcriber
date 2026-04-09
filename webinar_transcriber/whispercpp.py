"""Thin ctypes wrapper around the whisper.cpp C API."""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import re
import threading
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Self

import numpy as np

from webinar_transcriber.models import DecodedWindow, InferenceWindow, TranscriptSegment

if TYPE_CHECKING:
    from types import TracebackType

_WHISPER_SAMPLING_GREEDY: Final[int] = 0
_TICKS_PER_SECOND: Final[float] = 100.0
GPU_BACKEND_PATTERN: Final = re.compile(
    r"(?i)\b(metal|mtl|cuda|vulkan|coreml)\b[^|]*?(?:=|:)\s*(?:1|true)"
)
_BACKEND_PLUGIN_GLOB: Final[str] = "libggml-*.so"
_BACKEND_REGISTRATION_DONE = False
_GGML_LIBRARY_HANDLE: ctypes.CDLL | None = None
_LOG_SINK_LOCK = threading.RLock()
_LOG_SINK_PATH: Path | None = None
_GGML_LOG_CALLBACK_TYPE = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)


class WhisperCppError(RuntimeError):
    """Raised when the whisper.cpp runtime cannot be used successfully."""


class _WhisperAhead(ctypes.Structure):
    _fields_ = [
        ("n_text_layer", ctypes.c_int),
        ("n_head", ctypes.c_int),
    ]


class _WhisperAheads(ctypes.Structure):
    _fields_ = [
        ("n_heads", ctypes.c_size_t),
        ("heads", ctypes.POINTER(_WhisperAhead)),
    ]


class _WhisperContextParams(ctypes.Structure):
    _fields_ = [
        ("use_gpu", ctypes.c_bool),
        ("flash_attn", ctypes.c_bool),
        ("gpu_device", ctypes.c_int),
        ("dtw_token_timestamps", ctypes.c_bool),
        ("dtw_aheads_preset", ctypes.c_int),
        ("dtw_n_top", ctypes.c_int),
        ("dtw_aheads", _WhisperAheads),
        ("dtw_mem_size", ctypes.c_size_t),
    ]


class _WhisperGrammarElement(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("value", ctypes.c_uint32),
    ]


class _WhisperVadParams(ctypes.Structure):
    _fields_ = [
        ("threshold", ctypes.c_float),
        ("min_speech_duration_ms", ctypes.c_int),
        ("min_silence_duration_ms", ctypes.c_int),
        ("max_speech_duration_s", ctypes.c_float),
        ("speech_pad_ms", ctypes.c_int),
        ("samples_overlap", ctypes.c_float),
    ]


class _WhisperGreedyParams(ctypes.Structure):
    _fields_ = [("best_of", ctypes.c_int)]


class _WhisperBeamSearchParams(ctypes.Structure):
    _fields_ = [
        ("beam_size", ctypes.c_int),
        ("patience", ctypes.c_float),
    ]


class _WhisperFullParams(ctypes.Structure):
    _fields_ = [
        ("strategy", ctypes.c_int),
        ("n_threads", ctypes.c_int),
        ("n_max_text_ctx", ctypes.c_int),
        ("offset_ms", ctypes.c_int),
        ("duration_ms", ctypes.c_int),
        ("translate", ctypes.c_bool),
        ("no_context", ctypes.c_bool),
        ("no_timestamps", ctypes.c_bool),
        ("single_segment", ctypes.c_bool),
        ("print_special", ctypes.c_bool),
        ("print_progress", ctypes.c_bool),
        ("print_realtime", ctypes.c_bool),
        ("print_timestamps", ctypes.c_bool),
        ("token_timestamps", ctypes.c_bool),
        ("thold_pt", ctypes.c_float),
        ("thold_ptsum", ctypes.c_float),
        ("max_len", ctypes.c_int),
        ("split_on_word", ctypes.c_bool),
        ("max_tokens", ctypes.c_int),
        ("debug_mode", ctypes.c_bool),
        ("audio_ctx", ctypes.c_int),
        ("tdrz_enable", ctypes.c_bool),
        ("suppress_regex", ctypes.c_char_p),
        ("initial_prompt", ctypes.c_char_p),
        ("carry_initial_prompt", ctypes.c_bool),
        ("prompt_tokens", ctypes.POINTER(ctypes.c_int32)),
        ("prompt_n_tokens", ctypes.c_int),
        ("language", ctypes.c_char_p),
        ("detect_language", ctypes.c_bool),
        ("suppress_blank", ctypes.c_bool),
        ("suppress_nst", ctypes.c_bool),
        ("temperature", ctypes.c_float),
        ("max_initial_ts", ctypes.c_float),
        ("length_penalty", ctypes.c_float),
        ("temperature_inc", ctypes.c_float),
        ("entropy_thold", ctypes.c_float),
        ("logprob_thold", ctypes.c_float),
        ("no_speech_thold", ctypes.c_float),
        ("greedy", _WhisperGreedyParams),
        ("beam_search", _WhisperBeamSearchParams),
        ("new_segment_callback", ctypes.c_void_p),
        ("new_segment_callback_user_data", ctypes.c_void_p),
        ("progress_callback", ctypes.c_void_p),
        ("progress_callback_user_data", ctypes.c_void_p),
        ("encoder_begin_callback", ctypes.c_void_p),
        ("encoder_begin_callback_user_data", ctypes.c_void_p),
        ("abort_callback", ctypes.c_void_p),
        ("abort_callback_user_data", ctypes.c_void_p),
        ("logits_filter_callback", ctypes.c_void_p),
        ("logits_filter_callback_user_data", ctypes.c_void_p),
        ("grammar_rules", ctypes.POINTER(ctypes.POINTER(_WhisperGrammarElement))),
        ("n_grammar_rules", ctypes.c_size_t),
        ("i_start_rule", ctypes.c_size_t),
        ("grammar_penalty", ctypes.c_float),
        ("vad", ctypes.c_bool),
        ("vad_model_path", ctypes.c_char_p),
        ("vad_params", _WhisperVadParams),
    ]


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
        del exc_type, exc, traceback
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
            library_path=self.library_path,
            system_info=self.system_info(),
        )

    def create_session(self, model_path: Path) -> WhisperCppSession:
        context_params = self._lib.whisper_context_default_params()
        runtime_details = self.runtime_details()
        _configure_context_params(context_params, system_info=runtime_details.system_info)
        _set_log_sink_path(self._log_path)
        context = self._lib.whisper_init_from_file_with_params(
            os.fspath(model_path).encode("utf-8"),
            context_params,
        )
        if not context:
            raise WhisperCppError(f"Failed to initialize whisper.cpp model: {model_path}")

        state = self._lib.whisper_init_state(context)
        if not state:
            self._lib.whisper_free(context)
            raise WhisperCppError("Failed to initialize whisper.cpp runtime state.")

        return WhisperCppSession(
            self,
            context=context,
            state=state,
            runtime_details=runtime_details,
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
        start_index = max(0, round(window.start_sec * 16_000))
        end_index = min(len(audio_samples), round(window.end_sec * 16_000))
        window_samples = np.ascontiguousarray(
            audio_samples[start_index:end_index], dtype=np.float32
        )

        if window_samples.size == 0:
            return DecodedWindow(
                window=window,
                language=language_hint,
            )

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
        str(ggml_library_path),
        mode=getattr(ctypes, "RTLD_GLOBAL", 0),
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


def _configure_signatures(library: ctypes.CDLL) -> None:
    library.whisper_context_default_params.restype = _WhisperContextParams
    library.whisper_init_from_file_with_params.argtypes = [ctypes.c_char_p, _WhisperContextParams]
    library.whisper_init_from_file_with_params.restype = ctypes.c_void_p
    library.whisper_init_state.argtypes = [ctypes.c_void_p]
    library.whisper_init_state.restype = ctypes.c_void_p
    library.whisper_free.argtypes = [ctypes.c_void_p]
    library.whisper_free.restype = None
    library.whisper_free_state.argtypes = [ctypes.c_void_p]
    library.whisper_free_state.restype = None
    library.whisper_full_default_params_by_ref.argtypes = [ctypes.c_int]
    library.whisper_full_default_params_by_ref.restype = ctypes.POINTER(_WhisperFullParams)
    library.whisper_free_params.argtypes = [ctypes.POINTER(_WhisperFullParams)]
    library.whisper_free_params.restype = None
    library.whisper_full_with_state.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        _WhisperFullParams,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
    ]
    library.whisper_full_with_state.restype = ctypes.c_int
    library.whisper_full_n_segments_from_state.argtypes = [ctypes.c_void_p]
    library.whisper_full_n_segments_from_state.restype = ctypes.c_int
    library.whisper_full_lang_id_from_state.argtypes = [ctypes.c_void_p]
    library.whisper_full_lang_id_from_state.restype = ctypes.c_int
    library.whisper_full_get_segment_t0_from_state.argtypes = [ctypes.c_void_p, ctypes.c_int]
    library.whisper_full_get_segment_t0_from_state.restype = ctypes.c_int64
    library.whisper_full_get_segment_t1_from_state.argtypes = [ctypes.c_void_p, ctypes.c_int]
    library.whisper_full_get_segment_t1_from_state.restype = ctypes.c_int64
    library.whisper_full_get_segment_text_from_state.argtypes = [ctypes.c_void_p, ctypes.c_int]
    library.whisper_full_get_segment_text_from_state.restype = ctypes.c_char_p
    library.whisper_lang_str.argtypes = [ctypes.c_int]
    library.whisper_lang_str.restype = ctypes.c_char_p
    library.whisper_log_set.argtypes = [_GGML_LOG_CALLBACK_TYPE, ctypes.c_void_p]
    library.whisper_log_set.restype = None
    library.whisper_print_system_info.argtypes = []
    library.whisper_print_system_info.restype = ctypes.c_char_p


def _configure_context_params(
    context_params: _WhisperContextParams,
    *,
    system_info: str,
) -> None:
    use_gpu = _system_info_supports_gpu(system_info)
    context_params.use_gpu = use_gpu
    if not use_gpu:
        context_params.flash_attn = False
        context_params.gpu_device = 0


def _system_info_supports_gpu(system_info: str) -> bool:
    return GPU_BACKEND_PATTERN.search(system_info) is not None


def _encode_optional_text(value: str | None) -> bytes | None:
    if not value:
        return None
    return value.encode("utf-8")


def _decode_c_string(value: bytes | None) -> str:
    if not value:
        return ""
    return value.decode("utf-8", errors="replace")


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
