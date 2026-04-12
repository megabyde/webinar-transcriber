"""Public whisper.cpp facade."""

from .bindings import (
    _decode_c_string,
    _encode_optional_text,
    _system_info_supports_gpu,
    _WhisperContextParams,
    _WhisperFullParams,
)
from .library import (
    GPU_BACKEND_PATTERN,
    WhisperCppError,
    WhisperCppLibrary,
    WhisperCppRuntimeDetails,
    WhisperCppSession,
    _candidate_backend_plugin_paths,
    _library_log_callback,
    _load_backend_plugins,
    _resolve_ggml_library_path,
    _set_log_sink_path,
    resolve_library_path,
)

__all__ = [
    "GPU_BACKEND_PATTERN",
    "WhisperCppError",
    "WhisperCppLibrary",
    "WhisperCppRuntimeDetails",
    "WhisperCppSession",
    "_WhisperContextParams",
    "_WhisperFullParams",
    "_candidate_backend_plugin_paths",
    "_decode_c_string",
    "_encode_optional_text",
    "_library_log_callback",
    "_load_backend_plugins",
    "_resolve_ggml_library_path",
    "_set_log_sink_path",
    "_system_info_supports_gpu",
    "resolve_library_path",
]
