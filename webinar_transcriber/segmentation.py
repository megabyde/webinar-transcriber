"""Speech-region detection and preparation for deterministic ASR decoding.

The policy lives here rather than in the transcriber wrapper so the ASR integration stays a thin
decode shim. That separation makes it easier to evolve segmentation heuristics without entangling
them with runtime-specific library details.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from importlib import resources
from typing import TYPE_CHECKING

from webinar_transcriber.models import SpeechRegion
from webinar_transcriber.normalized_audio import NORMALIZED_SAMPLE_RATE

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    import numpy as np

DEFAULT_VAD_THRESHOLD = 0.45
DEFAULT_MIN_SPEECH_DURATION_MS = 150
DEFAULT_MIN_SILENCE_DURATION_MS = 500
DEFAULT_SPEECH_REGION_PAD_MS = 350
SHERPA_VAD_BUFFER_SIZE_SEC = 10 * 60


@dataclass(frozen=True)
class VadSettings:
    """Configuration for VAD-driven speech region planning."""

    enabled: bool = True
    threshold: float = DEFAULT_VAD_THRESHOLD
    min_speech_duration_ms: int = DEFAULT_MIN_SPEECH_DURATION_MS
    min_silence_duration_ms: int = DEFAULT_MIN_SILENCE_DURATION_MS
    speech_region_pad_ms: int = DEFAULT_SPEECH_REGION_PAD_MS


DEFAULT_VAD_SETTINGS = VadSettings()


def detect_speech_regions(
    samples: np.ndarray,
    sample_rate: int,
    *,
    settings: VadSettings = DEFAULT_VAD_SETTINGS,
    progress_callback: Callable[[float, int], None] | None = None,
) -> tuple[list[SpeechRegion], list[str]]:
    """Return coarse Silero speech regions and any warnings emitted during detection."""
    duration_sec = len(samples) / float(sample_rate) if sample_rate > 0 else 0.0
    if duration_sec <= 0:
        return [], []

    if not settings.enabled:
        if progress_callback is not None:
            progress_callback(duration_sec, 1)
        return [SpeechRegion(start_sec=0.0, end_sec=duration_sec)], []

    timestamps = _silero_speech_timestamps(
        samples,
        sample_rate=sample_rate,
        threshold=settings.threshold,
        min_speech_duration_ms=settings.min_speech_duration_ms,
        min_silence_duration_ms=settings.min_silence_duration_ms,
        speech_pad_ms=settings.speech_region_pad_ms,
        progress_callback=progress_callback,
    )
    if timestamps is None:
        if progress_callback is not None:
            progress_callback(duration_sec, 1)
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
    return normalize_regions(regions), []


def normalized_audio_duration(samples: np.ndarray, sample_rate: int) -> float:
    """Return the normalized audio duration in seconds."""
    if sample_rate <= 0:
        return 0.0
    return len(samples) / float(sample_rate)


def normalize_regions(regions: list[SpeechRegion]) -> list[SpeechRegion]:
    """Return ordered, merged speech regions with empty ranges removed."""
    ordered_regions = sorted(regions, key=lambda region: (region.start_sec, region.end_sec))
    if not ordered_regions:
        return []

    merged: list[SpeechRegion] = []
    for region in ordered_regions:
        if region.end_sec <= region.start_sec:
            continue
        if not merged:
            merged.append(region)
            continue

        previous = merged[-1]
        if region.start_sec <= previous.end_sec:
            merged[-1] = SpeechRegion(
                start_sec=previous.start_sec, end_sec=max(previous.end_sec, region.end_sec)
            )
            continue
        merged.append(region)
    return merged


def _silero_speech_timestamps(
    samples: np.ndarray,
    *,
    sample_rate: int,
    threshold: float,
    min_speech_duration_ms: int,
    min_silence_duration_ms: int,
    speech_pad_ms: int,
    progress_callback: Callable[[float, int], None] | None = None,
) -> list[dict[str, int]] | None:
    if sample_rate != NORMALIZED_SAMPLE_RATE:
        raise ValueError("Silero VAD expects normalized 16000 Hz audio.")

    sherpa_onnx = _load_sherpa_onnx()
    if sherpa_onnx is None:
        return None

    try:
        config = sherpa_onnx.VadModelConfig()
        config.silero_vad.model = str(_silero_vad_model_path())
        config.silero_vad.threshold = threshold
        config.silero_vad.min_speech_duration = min_speech_duration_ms / 1000.0
        config.silero_vad.min_silence_duration = min_silence_duration_ms / 1000.0
        config.sample_rate = sample_rate
        detector = sherpa_onnx.VoiceActivityDetector(
            config, buffer_size_in_seconds=SHERPA_VAD_BUFFER_SIZE_SEC
        )
    except (AttributeError, OSError, RuntimeError, ValueError):
        return None

    window_size = int(config.silero_vad.window_size)
    if window_size <= 0:
        return None

    def collect_ready_segments() -> None:
        while not detector.empty():
            segment = detector.front
            start = int(segment.start)
            end = start + len(segment.samples)
            timestamps.append({"start": start, "end": end})
            detector.pop()

    timestamps: list[dict[str, int]] = []
    next_report_sample = sample_rate
    for start_index in range(0, len(samples), window_size):
        completed_samples = min(len(samples), start_index + window_size)
        detector.accept_waveform(samples[start_index:completed_samples])
        collect_ready_segments()
        if progress_callback is not None and (
            completed_samples >= next_report_sample or completed_samples == len(samples)
        ):
            progress_callback(completed_samples / sample_rate, len(timestamps))
            while next_report_sample <= completed_samples:
                next_report_sample += sample_rate
    detector.flush()
    collect_ready_segments()

    return _pad_and_merge_timestamps(
        timestamps,
        sample_count=len(samples),
        sample_rate=sample_rate,
        speech_pad_ms=speech_pad_ms,
    )


def _pad_and_merge_timestamps(
    timestamps: list[dict[str, int]], *, sample_count: int, sample_rate: int, speech_pad_ms: int
) -> list[dict[str, int]]:
    pad_samples = max(0, round(speech_pad_ms / 1000.0 * sample_rate))
    padded = sorted(
        (
            {
                "start": max(0, int(timestamp["start"]) - pad_samples),
                "end": min(sample_count, int(timestamp["end"]) + pad_samples),
            }
            for timestamp in timestamps
        ),
        key=lambda timestamp: (timestamp["start"], timestamp["end"]),
    )
    merged: list[dict[str, int]] = []
    for timestamp in padded:
        if timestamp["end"] <= timestamp["start"]:
            continue
        if merged and timestamp["start"] <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], timestamp["end"])
            continue
        merged.append(timestamp)
    return merged


def _silero_vad_model_path() -> resources.abc.Traversable:
    return resources.files("webinar_transcriber.assets").joinpath("silero_vad.onnx")


def _load_sherpa_onnx() -> ModuleType | None:
    try:
        return importlib.import_module("sherpa_onnx")
    except ImportError:
        return None
