"""Speech-region detection and preparation for deterministic ASR decoding.

The policy lives here rather than in the transcriber wrapper so the ASR integration stays a thin
decode shim. That separation makes it easier to evolve segmentation heuristics without entangling
them with runtime-specific library details.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

from webinar_transcriber.models import SpeechRegion
from webinar_transcriber.normalized_audio import NORMALIZED_SAMPLE_RATE

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    import numpy as np

DEFAULT_VAD_THRESHOLD = 0.5
DEFAULT_MIN_SPEECH_DURATION_MS = 250
DEFAULT_MIN_SILENCE_DURATION_MS = 600
DEFAULT_SPEECH_REGION_PAD_MS = 200
DEFAULT_MIN_REPAIRED_REGION_SEC = 3.0
DEFAULT_REPAIR_MAX_GAP_SEC = 0.9


@dataclass(frozen=True)
class VadSettings:
    """Configuration for VAD-driven speech region planning."""

    enabled: bool = True
    threshold: float = DEFAULT_VAD_THRESHOLD
    min_speech_duration_ms: int = DEFAULT_MIN_SPEECH_DURATION_MS
    min_silence_duration_ms: int = DEFAULT_MIN_SILENCE_DURATION_MS
    speech_region_pad_ms: int = DEFAULT_SPEECH_REGION_PAD_MS


def detect_speech_regions(
    samples: np.ndarray,
    sample_rate: int,
    *,
    threshold: float = DEFAULT_VAD_THRESHOLD,
    min_speech_duration_ms: int = DEFAULT_MIN_SPEECH_DURATION_MS,
    min_silence_duration_ms: int = DEFAULT_MIN_SILENCE_DURATION_MS,
    speech_pad_ms: int = DEFAULT_SPEECH_REGION_PAD_MS,
    progress_callback: Callable[[float, int], None] | None = None,
    enabled: bool = True,
) -> tuple[list[SpeechRegion], list[str]]:
    """Return coarse Silero speech regions and any warnings emitted during detection."""
    duration_sec = len(samples) / float(sample_rate) if sample_rate > 0 else 0.0
    if duration_sec <= 0:
        return [], []

    if not enabled:
        return [SpeechRegion(start_sec=0.0, end_sec=duration_sec)], []

    timestamps = _silero_speech_timestamps(
        samples,
        sample_rate=sample_rate,
        threshold=threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
        progress_callback=progress_callback,
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
    return _normalize_regions(regions), []


def repair_speech_regions(
    regions: list[SpeechRegion],
    *,
    min_region_sec: float = DEFAULT_MIN_REPAIRED_REGION_SEC,
    max_gap_sec: float = DEFAULT_REPAIR_MAX_GAP_SEC,
) -> list[SpeechRegion]:
    """Merge short speech regions across nearby pauses until they become usable ASR regions.

    Returns:
        list[SpeechRegion]: The repaired speech regions.
    """
    repaired = _normalize_regions(regions)
    min_duration = max(0.0, min_region_sec)
    gap_limit = max(0.0, max_gap_sec)
    if len(repaired) < 2 or min_duration <= 0:
        return repaired

    merged: list[SpeechRegion] = []
    current = repaired[0]
    next_index = 1

    while True:
        if (current.end_sec - current.start_sec) >= min_duration:
            merged.append(current)
        else:
            left_gap = current.start_sec - merged[-1].end_sec if merged else float("inf")
            right_gap = (
                repaired[next_index].start_sec - current.end_sec
                if next_index < len(repaired)
                else float("inf")
            )
            if min(left_gap, right_gap) > gap_limit:
                merged.append(current)
            elif left_gap <= right_gap:
                previous = merged.pop()
                current = SpeechRegion(start_sec=previous.start_sec, end_sec=current.end_sec)
                continue
            else:
                current = SpeechRegion(
                    start_sec=current.start_sec, end_sec=repaired[next_index].end_sec
                )
                next_index += 1
                continue

        if next_index >= len(repaired):
            return merged
        current = repaired[next_index]
        next_index += 1


def normalized_audio_duration(samples: np.ndarray, sample_rate: int) -> float:
    """Return the normalized audio duration in seconds."""
    if sample_rate <= 0:
        return 0.0
    return len(samples) / float(sample_rate)


def _normalize_regions(regions: list[SpeechRegion]) -> list[SpeechRegion]:
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
    modules = _load_silero_modules()
    if modules is None:
        return None
    silero_vad, torch = modules

    if sample_rate != NORMALIZED_SAMPLE_RATE:
        raise ValueError("Silero VAD expects normalized 16000 Hz audio.")

    speech_model = silero_vad.load_silero_vad()
    timestamps = silero_vad.get_speech_timestamps(
        torch.from_numpy(samples),
        speech_model,
        threshold=threshold,
        sampling_rate=sample_rate,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
    )
    if progress_callback is not None:
        progress_callback(len(samples) / sample_rate, len(timestamps))

    return timestamps


def _load_silero_modules() -> tuple[ModuleType, ModuleType] | None:
    try:
        silero_vad = importlib.import_module("silero_vad")
        torch = importlib.import_module("torch")
    except ImportError:
        return None
    return silero_vad, torch
