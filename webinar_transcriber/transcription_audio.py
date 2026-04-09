"""Helpers for deterministic transcription audio preparation."""

from __future__ import annotations

import tempfile
import wave
from contextlib import contextmanager
from pathlib import Path
from shutil import copy2
from typing import TYPE_CHECKING

import numpy as np

from webinar_transcriber.media import MediaProcessingError, _run_command

if TYPE_CHECKING:
    from collections.abc import Iterator

NORMALIZED_SAMPLE_RATE = 16_000


def extract_audio(input_path: Path, output_path: Path) -> Path:
    """Convert the input media into a mono 16 kHz WAV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _run_command(
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    )
    return output_path


def transcode_audio_to_mp3(input_path: Path, output_path: Path) -> Path:
    """Convert normalized transcription audio into an MP3 artifact."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _run_command(
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-codec:a",
        "libmp3lame",
        "-q:a",
        "2",
        str(output_path),
    )
    return output_path


@contextmanager
def prepared_transcription_audio(input_path: Path) -> Iterator[Path]:
    """Yield a normalized mono 16 kHz WAV file for transcription."""
    with tempfile.TemporaryDirectory(prefix="webinar-transcriber-audio-") as temp_dir:
        audio_path = Path(temp_dir) / f"{input_path.stem}.wav"
        extract_audio(input_path, audio_path)
        yield audio_path


def preserve_transcription_audio(
    audio_path: Path,
    output_path: Path,
    *,
    audio_format: str = "wav",
) -> Path:
    """Persist prepared transcription audio as a run artifact."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if audio_format == "wav":
        copy2(audio_path, output_path)
        return output_path
    if audio_format == "mp3":
        return transcode_audio_to_mp3(audio_path, output_path)
    raise ValueError(f"Unsupported transcription audio format: {audio_format}")


def load_normalized_audio(audio_path: Path) -> tuple[np.ndarray, int]:
    """Return mono float32 PCM audio samples from a normalized WAV file."""
    with wave.open(str(audio_path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        raw_frames = wav_file.readframes(wav_file.getnframes())

    if sample_rate != NORMALIZED_SAMPLE_RATE:
        raise MediaProcessingError(
            f"Expected {NORMALIZED_SAMPLE_RATE} Hz transcription audio, got {sample_rate} Hz."
        )
    if channels != 1:
        raise MediaProcessingError(f"Expected mono transcription audio, got {channels} channels.")
    if sample_width != 2:
        raise MediaProcessingError(
            f"Expected 16-bit PCM transcription audio, got {sample_width * 8}-bit."
        )

    samples = np.frombuffer(raw_frames, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, sample_rate
