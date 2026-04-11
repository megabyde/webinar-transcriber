"""ctypes declarations and signature helpers for the whisper.cpp C API."""

from __future__ import annotations

import ctypes
import re
from typing import Final

_WHISPER_SAMPLING_GREEDY: Final[int] = 0
_TICKS_PER_SECOND: Final[float] = 100.0
GPU_BACKEND_PATTERN: Final = re.compile(
    r"(?i)\b(metal|mtl|cuda|vulkan|coreml)\b[^|]*?(?:=|:)\s*(?:1|true)"
)
_GGML_LOG_CALLBACK_TYPE = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)


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
