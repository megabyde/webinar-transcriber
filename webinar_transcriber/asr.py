"""ASR adapter built on top of the whisper.cpp C API."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from webinar_transcriber.models import AudioChunk, ChunkTranscription, TranscriptionResult
from webinar_transcriber.transcription_audio import load_normalized_audio
from webinar_transcriber.whispercpp import (
    WhisperCppLibrary,
    WhisperCppRuntimeDetails,
    WhisperCppSession,
)

if TYPE_CHECKING:
    from collections.abc import Callable


ASR_BACKEND_NAME = "whisper.cpp"
DEFAULT_WHISPER_CPP_MODEL = Path("models/whisper-cpp/ggml-large-v3-turbo.bin")
_ENABLED_BACKEND_PATTERN = re.compile(
    r"(?i)\b(metal|mtl|cuda|vulkan|coreml)\b[^|]*?(?:=|:)\s*(?:1|true)"
)


def _read_sysctl_int(name: str) -> int | None:
    try:
        result = subprocess.run(
            ["sysctl", "-n", name],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, PermissionError, subprocess.CalledProcessError):
        return None

    value = result.stdout.strip()
    if not value.isdigit():
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


def default_asr_threads() -> int:
    return (
        _read_sysctl_int("hw.perflevel0.physicalcpu")
        or _read_sysctl_int("hw.physicalcpu")
        or os.cpu_count()
        or 4
    )


DEFAULT_ASR_THREADS = default_asr_threads()


def _missing_model_error_message(model_path: Path) -> str:
    default_path = DEFAULT_WHISPER_CPP_MODEL
    if model_path == default_path:
        location_hint = f"Download ggml-large-v3-turbo.bin to {default_path}."
    else:
        location_hint = (
            f"Download a whisper.cpp model there, or use --asr-model {default_path}."
        )
    return (
        f"whisper.cpp model file does not exist: {model_path}. "
        f"{location_hint} See README.md for model download instructions."
    )


class Transcriber(Protocol):
    """Protocol for components that convert audio to transcript text."""

    @property
    def model_name(self) -> str:
        """Effective model name for diagnostics and logging."""

    @property
    def device_name(self) -> str:
        """Effective device name for diagnostics and logging."""

    def prepare_model(self) -> None:
        """Warm or validate model assets before transcription starts."""

    def transcribe(
        self,
        audio_path: Path,
        *,
        progress_callback: Callable[[float], None] | None = None,
    ) -> TranscriptionResult:
        """Return a normalized transcription for the provided audio file."""

    def close(self) -> None:
        """Release any prepared runtime resources."""


class WhisperCppTranscriber:
    """ASR implementation using the in-process whisper.cpp C API."""

    def __init__(
        self,
        model_name: str | None = None,
        *,
        threads: int = 4,
        initial_prompt: str | None = None,
        library_path: Path | None = None,
        log_path: Path | None = None,
    ) -> None:
        self._model_path = Path(model_name) if model_name else DEFAULT_WHISPER_CPP_MODEL
        self._threads = max(1, threads)
        self._initial_prompt = initial_prompt
        self._library_path = library_path
        self._log_path = log_path
        self._runtime_details: WhisperCppRuntimeDetails | None = None
        self._runtime: WhisperCppLibrary | None = None
        self._session: WhisperCppSession | None = None

    @property
    def model_name(self) -> str:
        return str(self._model_path)

    @property
    def device_name(self) -> str:
        if self._runtime_details is None:
            return "auto"
        return _device_name_from_system_info(self._runtime_details.system_info)

    @property
    def threads(self) -> int:
        return self._threads

    @property
    def system_info(self) -> str | None:
        if self._runtime_details is None:
            return None
        return self._runtime_details.system_info

    @property
    def library_path(self) -> str | None:
        if self._runtime_details is None:
            return None
        return self._runtime_details.library_path

    def set_log_path(self, log_path: Path) -> None:
        self._log_path = log_path

    def prepare_model(self) -> None:
        if not self._model_path.exists():
            raise RuntimeError(_missing_model_error_message(self._model_path))
        self.close()
        runtime = WhisperCppLibrary(self._library_path, log_path=self._log_path)
        session = runtime.create_session(self._model_path)
        self._runtime = runtime
        self._session = session
        self._runtime_details = session.runtime_details

    def transcribe(
        self,
        audio_path: Path,
        *,
        progress_callback: Callable[[float], None] | None = None,
    ) -> TranscriptionResult:
        audio_samples, sample_rate = load_normalized_audio(audio_path)
        duration_sec = len(audio_samples) / float(sample_rate) if sample_rate else 0.0
        chunk = AudioChunk(id="chunk-1", start_sec=0.0, end_sec=duration_sec)
        chunk_transcriptions = self.transcribe_audio_chunks(
            audio_samples,
            [chunk],
            progress_callback=progress_callback,
        )
        if chunk_transcriptions:
            return TranscriptionResult(
                detected_language=chunk_transcriptions[0].detected_language,
                segments=chunk_transcriptions[0].segments,
            )
        return TranscriptionResult()

    def transcribe_audio_chunks(
        self,
        audio_samples,
        chunks: list[AudioChunk],
        *,
        progress_callback: Callable[[float], None] | None = None,
    ) -> list[ChunkTranscription]:
        session = self._ensure_session()
        return session.transcribe_chunks(
            audio_samples,
            chunks,
            threads=self._threads,
            initial_prompt=self._initial_prompt,
            progress_callback=progress_callback,
        )

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None
        self._runtime = None

    def _ensure_session(self) -> WhisperCppSession:
        if self._session is None:
            self.prepare_model()
        assert self._session is not None
        return self._session

    def __del__(self) -> None:
        self.close()


WhisperTranscriber = WhisperCppTranscriber


def _device_name_from_system_info(system_info: str) -> str:
    match = _ENABLED_BACKEND_PATTERN.search(system_info)
    if match is not None:
        backend_name = match.group(1).lower()
        return "metal" if backend_name == "mtl" else backend_name
    return "cpu"
