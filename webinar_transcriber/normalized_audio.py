"""Helpers for deterministic transcription audio preparation."""

from __future__ import annotations

import tempfile
import wave
from contextlib import contextmanager
from pathlib import Path
from shutil import copy2
from typing import TYPE_CHECKING, cast

import av
import numpy as np

from webinar_transcriber.media import (
    MediaProcessingError,
    _required_audio_stream,
    open_input_media_container,
    open_output_media_container,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from av.audio.frame import AudioFrame
    from av.audio.stream import AudioStream
    from av.container import OutputContainer

NORMALIZED_SAMPLE_RATE = 16_000
NORMALIZED_CHANNELS = 1
NORMALIZED_SAMPLE_WIDTH_BYTES = 2
NORMALIZED_AUDIO_CODEC = "pcm_s16le"


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
    output_error_message: str,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open_input_media_container(
        input_path, error_message="Could not open {path}: {error}"
    ) as input_container:
        input_stream = _required_audio_stream(
            input_container, error_message=f"No audio stream found in {input_path}."
        )
        with open_output_media_container(
            output_path,
            error_message=output_error_message,
        ) as output_container:
            output_stream = cast(
                "AudioStream",
                output_container.add_stream(output_codec, rate=NORMALIZED_SAMPLE_RATE),
            )
            output_stream.layout = "mono"
            resampler = av.AudioResampler(
                format=resample_format, layout="mono", rate=NORMALIZED_SAMPLE_RATE
            )

            for decoded_frame in input_container.decode(input_stream):
                _mux_audio_frames(
                    output_container, output_stream, resampler.resample(decoded_frame)
                )

            _mux_audio_frames(output_container, output_stream, resampler.resample(None))

            for packet in output_stream.encode(None):
                output_container.mux(packet)

    if not output_path.exists():  # pragma: no cover - PyAV defensive postcondition
        raise MediaProcessingError(f"PyAV did not write {output_path}.")
    return output_path


def extract_audio(input_path: Path, output_path: Path) -> Path:
    """Convert the input media into a mono 16 kHz WAV file.

    Returns:
        Path: The written normalized WAV path.
    """
    return _transcode_audio_with_pyav(
        input_path,
        output_path,
        output_codec=NORMALIZED_AUDIO_CODEC,
        resample_format="s16",
        output_error_message=f"Could not normalize audio from {input_path}: {{error}}",
    )


def transcode_audio_to_mp3(input_path: Path, output_path: Path) -> Path:
    """Convert normalized transcription audio into an MP3 artifact.

    Returns:
        Path: The written MP3 artifact path.
    """
    return _transcode_audio_with_pyav(
        input_path,
        output_path,
        output_codec="mp3",
        resample_format="fltp",
        output_error_message=f"Could not transcode audio from {input_path}: {{error}}",
    )


@contextmanager
def prepared_transcription_audio(input_path: Path) -> Iterator[Path]:
    """Yield a normalized mono 16 kHz WAV file for transcription."""
    with tempfile.TemporaryDirectory(prefix="webinar-transcriber-audio-") as temp_dir:
        audio_path = Path(temp_dir) / f"{input_path.stem}.wav"
        extract_audio(input_path, audio_path)
        yield audio_path


def preserve_transcription_audio(
    audio_path: Path, output_path: Path, *, audio_format: str = "wav"
) -> Path:
    """Persist prepared transcription audio as a run artifact.

    Returns:
        Path: The written artifact path.

    Raises:
        ValueError: If the requested audio format is unsupported.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if audio_format == "wav":
        copy2(audio_path, output_path)
        return output_path
    if audio_format == "mp3":
        return transcode_audio_to_mp3(audio_path, output_path)
    raise ValueError(  # pragma: no cover - CLI validates the supported format choices
        f"Unsupported transcription audio format: {audio_format}"
    )


def load_normalized_audio(audio_path: Path) -> tuple[np.ndarray, int]:
    """Return mono float32 PCM audio samples from a normalized WAV file.

    Returns:
        tuple[np.ndarray, int]: The float32 PCM samples and their sample rate.

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

    samples = np.frombuffer(raw_frames, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, sample_rate
