"""Helpers for deterministic transcription audio preparation."""

from __future__ import annotations

import tempfile
import wave
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from webinar_transcriber.media import MediaProcessingError, extract_audio

if TYPE_CHECKING:
    from collections.abc import Iterator

NORMALIZED_SAMPLE_RATE = 16_000


@contextmanager
def prepared_transcription_audio(input_path: Path) -> Iterator[Path]:
    """Yield a normalized mono 16 kHz WAV file for transcription."""
    with tempfile.TemporaryDirectory(prefix="webinar-transcriber-audio-") as temp_dir:
        audio_path = Path(temp_dir) / f"{input_path.stem}.wav"
        extract_audio(input_path, audio_path)
        yield audio_path


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
