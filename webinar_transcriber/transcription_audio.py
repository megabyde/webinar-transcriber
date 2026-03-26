"""Helpers for deterministic transcription audio preparation and chunk planning."""

from __future__ import annotations

import importlib
import tempfile
import wave
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from webinar_transcriber.media import MediaProcessingError, extract_audio
from webinar_transcriber.models import AudioChunk, MediaAsset, SpeechRegion

if TYPE_CHECKING:
    from collections.abc import Iterator

NORMALIZED_SAMPLE_RATE = 16_000
DEFAULT_VAD_THRESHOLD = 0.5
DEFAULT_MIN_SPEECH_DURATION_MS = 250
DEFAULT_MIN_SILENCE_DURATION_MS = 400
DEFAULT_SPEECH_PAD_MS = 150
DEFAULT_CHUNK_TARGET_SEC = 20.0
DEFAULT_CHUNK_MAX_SEC = 30.0
DEFAULT_CHUNK_OVERLAP_SEC = 1.5
DEFAULT_MERGE_GAP_SEC = 0.75


@dataclass(frozen=True)
class VADSettings:
    """Configuration for the optional VAD stage."""

    enabled: bool = True
    threshold: float = DEFAULT_VAD_THRESHOLD
    min_speech_duration_ms: int = DEFAULT_MIN_SPEECH_DURATION_MS
    min_silence_duration_ms: int = DEFAULT_MIN_SILENCE_DURATION_MS
    speech_pad_ms: int = DEFAULT_SPEECH_PAD_MS


@dataclass(frozen=True)
class ChunkPlanSettings:
    """Configuration for chunk planning."""

    target_sec: float = DEFAULT_CHUNK_TARGET_SEC
    max_sec: float = DEFAULT_CHUNK_MAX_SEC
    overlap_sec: float = DEFAULT_CHUNK_OVERLAP_SEC
    merge_gap_sec: float = DEFAULT_MERGE_GAP_SEC


@contextmanager
def prepared_transcription_audio(input_path: Path, _media_asset: MediaAsset) -> Iterator[Path]:
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


def detect_speech_regions(
    samples: np.ndarray,
    sample_rate: int,
    *,
    settings: VADSettings,
) -> tuple[list[SpeechRegion], list[str]]:
    """Return detected speech regions and any warnings emitted during detection."""
    duration_sec = len(samples) / float(sample_rate)
    if duration_sec <= 0:
        return [], []

    if not settings.enabled:
        return [SpeechRegion(start_sec=0.0, end_sec=duration_sec)], []

    timestamps = _silero_speech_timestamps(
        samples,
        sample_rate=sample_rate,
        settings=settings,
    )
    if timestamps is None:
        warning = "Silero VAD is unavailable; falling back to one full-audio speech region."
        return [SpeechRegion(start_sec=0.0, end_sec=duration_sec)], [warning]

    regions = [
        SpeechRegion(
            start_sec=float(timestamp["start"]) / sample_rate,
            end_sec=float(timestamp["end"]) / sample_rate,
        )
        for timestamp in timestamps
        if float(timestamp["end"]) > float(timestamp["start"])
    ]
    if not regions:
        return [], []
    return _merge_nearby_regions(regions, merge_gap_sec=settings.speech_pad_ms / 1000.0), []


def plan_audio_chunks(
    speech_regions: list[SpeechRegion],
    *,
    settings: ChunkPlanSettings,
) -> list[AudioChunk]:
    """Create bounded ASR chunks from speech regions."""
    if not speech_regions:
        return []

    merged_regions = _merge_nearby_regions(speech_regions, merge_gap_sec=settings.merge_gap_sec)
    chunks: list[AudioChunk] = []
    chunk_index = 1

    for region in merged_regions:
        chunk_start = region.start_sec
        while chunk_start < region.end_sec:
            chunk_end = min(chunk_start + settings.max_sec, region.end_sec)
            if chunk_end - chunk_start > settings.target_sec and chunk_end < region.end_sec:
                chunk_end = min(chunk_start + settings.target_sec, region.end_sec)
            chunks.append(
                AudioChunk(
                    id=f"chunk-{chunk_index}",
                    start_sec=max(0.0, chunk_start),
                    end_sec=max(chunk_start, chunk_end),
                )
            )
            chunk_index += 1
            if chunk_end >= region.end_sec:
                break
            chunk_start = max(region.start_sec, chunk_end - settings.overlap_sec)

    return chunks


def average_chunk_duration(chunks: list[AudioChunk]) -> float | None:
    """Return the average chunk duration in seconds."""
    if not chunks:
        return None
    return sum(chunk.end_sec - chunk.start_sec for chunk in chunks) / len(chunks)


def normalized_audio_duration(samples: np.ndarray, sample_rate: int) -> float:
    """Return the normalized audio duration in seconds."""
    if sample_rate <= 0:
        return 0.0
    return len(samples) / float(sample_rate)


def _merge_nearby_regions(
    regions: list[SpeechRegion],
    *,
    merge_gap_sec: float,
) -> list[SpeechRegion]:
    ordered_regions = sorted(regions, key=lambda region: region.start_sec)
    if not ordered_regions:
        return []

    merged: list[SpeechRegion] = [ordered_regions[0]]
    for region in ordered_regions[1:]:
        previous = merged[-1]
        if region.start_sec <= previous.end_sec + merge_gap_sec:
            merged[-1] = SpeechRegion(
                start_sec=previous.start_sec,
                end_sec=max(previous.end_sec, region.end_sec),
            )
        else:
            merged.append(region)
    return merged


def _silero_speech_timestamps(
    samples: np.ndarray,
    *,
    sample_rate: int,
    settings: VADSettings,
) -> list[dict[str, int]] | None:
    try:
        silero_vad = importlib.import_module("silero_vad")
        torch = importlib.import_module("torch")
    except ImportError:
        return None

    speech_model = silero_vad.load_silero_vad()
    audio_tensor = torch.from_numpy(samples)
    return silero_vad.get_speech_timestamps(
        audio_tensor,
        speech_model,
        sampling_rate=sample_rate,
        threshold=settings.threshold,
        min_speech_duration_ms=settings.min_speech_duration_ms,
        min_silence_duration_ms=settings.min_silence_duration_ms,
        speech_pad_ms=settings.speech_pad_ms,
        return_seconds=False,
    )
