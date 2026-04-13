"""Speech-region detection and preparation for deterministic ASR decoding.

The policy lives here rather than in `whispercpp.py` so the native binding stays a thin decode
shim. That separation makes it easier to evolve segmentation heuristics without entangling them
with the low-level library wrapper.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from webinar_transcriber.models import SpeechRegion
from webinar_transcriber.normalized_audio import NORMALIZED_SAMPLE_RATE

if TYPE_CHECKING:
    from collections.abc import Callable

DEFAULT_VAD_THRESHOLD = 0.5
DEFAULT_MIN_SPEECH_DURATION_MS = 250
DEFAULT_MIN_SILENCE_DURATION_MS = 600
# Keep Silero padding low so VAD answers "where is speech?" and the explicit expansion stage owns
# most of the ASR boundary budget.
DEFAULT_SPEECH_PAD_MS = 30
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
    speech_pad_ms: int = DEFAULT_SPEECH_PAD_MS,
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


def expand_speech_regions(
    regions: list[SpeechRegion], *, pad_ms: int, audio_duration_sec: float
) -> list[SpeechRegion]:
    """Apply ASR-specific context padding and merge overlaps after clipping to audio bounds."""
    if not regions or audio_duration_sec <= 0:
        return []

    pad_sec = max(0.0, pad_ms / 1000.0)
    expanded = [
        SpeechRegion(
            start_sec=max(0.0, region.start_sec - pad_sec),
            end_sec=min(audio_duration_sec, region.end_sec + pad_sec),
        )
        for region in regions
        if region.end_sec > region.start_sec
    ]
    return _normalize_regions(expanded)


def repair_speech_regions(
    regions: list[SpeechRegion],
    *,
    min_region_sec: float = DEFAULT_MIN_REPAIRED_REGION_SEC,
    max_gap_sec: float = DEFAULT_REPAIR_MAX_GAP_SEC,
) -> list[SpeechRegion]:
    """Merge short speech regions across nearby pauses until they become usable ASR regions."""
    repaired = _normalize_regions(regions)
    min_duration = max(0.0, min_region_sec)
    gap_limit = max(0.0, max_gap_sec)
    if len(repaired) < 2 or min_duration <= 0:
        return repaired

    max_iterations = len(repaired)
    for _ in range(max_iterations):
        changed = False
        for i, region in enumerate(repaired):
            if (region.end_sec - region.start_sec) >= min_duration:
                continue

            left_gap = region.start_sec - repaired[i - 1].end_sec if i > 0 else float("inf")
            right_gap = (
                repaired[i + 1].start_sec - region.end_sec
                if i + 1 < len(repaired)
                else float("inf")
            )
            gap = min(left_gap, right_gap)
            if gap > gap_limit:
                continue

            if left_gap <= right_gap:
                repaired[i - 1 : i + 1] = [
                    SpeechRegion(start_sec=repaired[i - 1].start_sec, end_sec=region.end_sec)
                ]
            else:
                repaired[i : i + 2] = [
                    SpeechRegion(start_sec=region.start_sec, end_sec=repaired[i + 1].end_sec)
                ]
            changed = True
            break
        if not changed:
            break

    return repaired


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

    assert sample_rate == NORMALIZED_SAMPLE_RATE, "Silero VAD expects normalized 16000 Hz audio."

    speech_model = silero_vad.load_silero_vad()
    iterator = silero_vad.VADIterator(
        speech_model,
        threshold=threshold,
        sampling_rate=sample_rate,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
    )
    window_size = 512
    min_speech_samples = (sample_rate * min_speech_duration_ms) / 1000
    total_samples = len(samples)
    start_sample: int | None = None
    timestamps: list[dict[str, int]] = []

    for chunk_start in range(0, total_samples, window_size):
        chunk = _silero_chunk(samples, start=chunk_start, window_size=window_size)
        event = iterator(torch.from_numpy(chunk), return_seconds=False)
        start_sample = _consume_vad_event(
            timestamps,
            event=event,
            start_sample=start_sample,
            min_speech_samples=min_speech_samples,
        )
        _report_vad_progress(
            progress_callback,
            processed_samples=min(chunk_start + window_size, total_samples),
            total_samples=total_samples,
            sample_rate=sample_rate,
            detected_count=len(timestamps),
        )

    if start_sample is not None and (total_samples - start_sample) > min_speech_samples:
        timestamps.append({"start": start_sample, "end": total_samples})

    return timestamps


def _load_silero_modules():
    try:
        silero_vad = importlib.import_module("silero_vad")
        torch = importlib.import_module("torch")
    except ImportError:
        return None
    return silero_vad, torch


def _silero_chunk(samples: np.ndarray, *, start: int, window_size: int) -> np.ndarray:
    chunk = samples[start : start + window_size]
    if len(chunk) < window_size:
        chunk = np.pad(chunk, (0, window_size - len(chunk)))
    return chunk


def _consume_vad_event(
    timestamps: list[dict[str, int]],
    *,
    event: dict[str, int] | None,
    start_sample: int | None,
    min_speech_samples: float,
) -> int | None:
    if event is None:
        return start_sample
    if "start" in event:
        return int(event["start"])
    if "end" not in event or start_sample is None:
        return start_sample

    end_sample = int(event["end"])
    if end_sample > start_sample and (end_sample - start_sample) > min_speech_samples:
        timestamps.append({"start": start_sample, "end": end_sample})
    return None


def _report_vad_progress(
    progress_callback,
    *,
    processed_samples: int,
    total_samples: int,
    sample_rate: int,
    detected_count: int,
) -> None:
    if progress_callback is None:
        return
    progress_callback(min(total_samples, processed_samples) / sample_rate, detected_count)
