"""Helpers for deterministic transcription audio preparation."""

from __future__ import annotations

import wave
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import av
import numpy as np

from webinar_transcriber.media import (
    MediaProcessingError,
    open_audio_input_container,
    open_output_media_container,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path

    from av.audio.frame import AudioFrame
    from av.audio.stream import AudioStream
    from av.container import OutputContainer


NORMALIZED_SAMPLE_RATE = 16_000
NORMALIZED_CHANNELS = 1
NORMALIZED_SAMPLE_WIDTH_BYTES = 2
NORMALIZED_AUDIO_CODEC = "pcm_s16le"


@dataclass(frozen=True, slots=True)
class _AudioOutputSpec:
    output_codec: str
    resample_format: str


_AUDIO_OUTPUT_SPECS = {
    "wav": _AudioOutputSpec(output_codec=NORMALIZED_AUDIO_CODEC, resample_format="s16"),
    "mp3": _AudioOutputSpec(output_codec="mp3", resample_format="fltp"),
}


def sample_index_for_time(time_sec: float) -> int:
    """Return the normalized-audio sample index for one timestamp."""
    return max(0, round(time_sec * NORMALIZED_SAMPLE_RATE))


def _mux_audio_frames(
    output_container: OutputContainer,
    output_stream: AudioStream,
    audio_frames: Iterable[AudioFrame] | None,
) -> None:
    if audio_frames is None:
        return
    for audio_frame in audio_frames:
        for packet in output_stream.encode(audio_frame):
            output_container.mux(packet)


def _transcode_audio_with_pyav(
    input_path: Path,
    output_path: Path,
    *,
    output_codec: str,
    resample_format: str,
    progress_callback: Callable[[float], None] | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        open_audio_input_container(input_path) as (input_container, input_stream),
        open_output_media_container(output_path) as output_container,
    ):
        output_stream = cast(
            "AudioStream", output_container.add_stream(output_codec, rate=NORMALIZED_SAMPLE_RATE)
        )  # av stubs type add_stream() as base Stream regardless of codec argument
        output_stream.layout = "mono"
        resampler = av.AudioResampler(
            format=resample_format, layout="mono", rate=NORMALIZED_SAMPLE_RATE
        )

        for decoded_frame in input_container.decode(input_stream):
            if progress_callback is not None and decoded_frame.time is not None:
                progress_callback(float(decoded_frame.time))
            _mux_audio_frames(output_container, output_stream, resampler.resample(decoded_frame))

        _mux_audio_frames(output_container, output_stream, resampler.resample(None))

        for packet in output_stream.encode(None):
            output_container.mux(packet)

    if not output_path.exists():  # pragma: no cover - PyAV defensive postcondition
        raise MediaProcessingError(f"PyAV did not write {output_path}.")
    return output_path


def write_transcription_audio(
    input_path: Path,
    output_path: Path,
    *,
    audio_format: str = "wav",
    progress_callback: Callable[[float], None] | None = None,
) -> Path:
    """Convert audio into a normalized transcription-audio format.

    Returns:
        Path: The written normalized audio path.
    """
    spec = _AUDIO_OUTPUT_SPECS[audio_format]
    return _transcode_audio_with_pyav(
        input_path,
        output_path,
        output_codec=spec.output_codec,
        resample_format=spec.resample_format,
        progress_callback=progress_callback,
    )


def preserve_transcription_audio(
    audio_path: Path, output_path: Path, *, progress_callback: Callable[[float], None] | None = None
) -> Path:
    """Persist prepared transcription audio as an MP3 run artifact.

    Returns:
        Path: The written artifact path.
    """
    return write_transcription_audio(
        audio_path, output_path, audio_format="mp3", progress_callback=progress_callback
    )


def load_normalized_audio(audio_path: Path) -> np.ndarray:
    """Return mono float32 PCM audio samples from a normalized WAV file.

    Returns:
        np.ndarray: The float32 PCM samples at the normalized sample rate.

    Raises:
        MediaProcessingError: If the WAV does not match the normalized audio contract.
    """
    with wave.open(str(audio_path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        raw_frames = wav_file.readframes(wav_file.getnframes())

    if sample_rate != NORMALIZED_SAMPLE_RATE:
        raise MediaProcessingError(
            f"Expected {NORMALIZED_SAMPLE_RATE} Hz transcription audio, got {sample_rate} Hz."
        )
    if channels != NORMALIZED_CHANNELS:
        raise MediaProcessingError(f"Expected mono transcription audio, got {channels} channels.")
    if sample_width != NORMALIZED_SAMPLE_WIDTH_BYTES:
        raise MediaProcessingError(
            f"Expected 16-bit PCM transcription audio, got {sample_width * 8}-bit."
        )

    return np.frombuffer(raw_frames, dtype=np.int16).astype(np.float32) / 32768.0
