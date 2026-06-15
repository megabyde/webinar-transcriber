"""Speech-region detection and preparation for deterministic ASR decoding.

The policy lives here rather than in the transcriber wrapper so the ASR integration stays a thin
decode shim. That separation makes it easier to evolve segmentation heuristics without entangling
them with runtime-specific library details.
"""

from __future__ import annotations

from importlib import resources
from typing import TYPE_CHECKING

from webinar_transcriber._env import load_sherpa_onnx
from webinar_transcriber.models import SpeechRegion
from webinar_transcriber.normalized_audio import NORMALIZED_SAMPLE_RATE

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

    ProgressCallback = Callable[[float, int], None]

VAD_THRESHOLD = 0.45
MIN_SPEECH_DURATION_SEC = 0.150
MIN_SILENCE_DURATION_SEC = 0.500
SPEECH_REGION_PAD_SEC = 0.350
SHERPA_VAD_BUFFER_SIZE_SEC = 10 * 60


def detect_speech_regions(
    samples: np.ndarray, *, threads: int, progress_callback: ProgressCallback | None = None
) -> tuple[list[SpeechRegion], list[str]]:
    """Return coarse Silero speech regions and any warnings emitted during detection."""
    duration_sec = normalized_audio_duration(samples)
    if duration_sec <= 0:
        return [], []

    regions = _silero_speech_regions(samples, threads=threads, progress_callback=progress_callback)
    if regions is None:
        if progress_callback is not None:
            progress_callback(duration_sec, 1)
        warning = "Silero VAD is unavailable; falling back to one full-audio speech region."
        return [SpeechRegion(start_sec=0.0, end_sec=duration_sec)], [warning]

    return normalize_regions(regions), []


def normalized_audio_duration(samples: np.ndarray) -> float:
    """Return the normalized audio duration in seconds."""
    return len(samples) / NORMALIZED_SAMPLE_RATE


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


def _silero_speech_regions(
    samples: np.ndarray, *, threads: int, progress_callback: ProgressCallback | None = None
) -> list[SpeechRegion] | None:
    sherpa_onnx = load_sherpa_onnx()
    if sherpa_onnx is None:
        return None

    try:
        config = sherpa_onnx.VadModelConfig()
        config.silero_vad.model = str(_silero_vad_model_path())
        config.silero_vad.threshold = VAD_THRESHOLD
        config.silero_vad.min_speech_duration = MIN_SPEECH_DURATION_SEC
        config.silero_vad.min_silence_duration = MIN_SILENCE_DURATION_SEC
        config.sample_rate = NORMALIZED_SAMPLE_RATE
        config.num_threads = threads
        detector = sherpa_onnx.VoiceActivityDetector(
            config, buffer_size_in_seconds=SHERPA_VAD_BUFFER_SIZE_SEC
        )
    except (
        AttributeError,
        OSError,
        RuntimeError,
        ValueError,
    ):  # pragma: no cover - sherpa boundary
        return None

    window_size = int(config.silero_vad.window_size)
    if window_size <= 0:
        return None

    def collect_ready_segments() -> None:
        while not detector.empty():
            segment = detector.front
            start = int(segment.start)
            end = start + len(segment.samples)
            if end > start:
                regions.append(
                    SpeechRegion(
                        start_sec=start / NORMALIZED_SAMPLE_RATE,
                        end_sec=end / NORMALIZED_SAMPLE_RATE,
                    )
                )
            detector.pop()

    regions: list[SpeechRegion] = []
    next_report_sample = NORMALIZED_SAMPLE_RATE
    for start_index in range(0, len(samples), window_size):
        completed_samples = min(len(samples), start_index + window_size)
        detector.accept_waveform(samples[start_index:completed_samples])
        collect_ready_segments()
        if progress_callback is not None and (
            completed_samples >= next_report_sample or completed_samples == len(samples)
        ):
            progress_callback(completed_samples / NORMALIZED_SAMPLE_RATE, len(regions))
            next_report_sample = (
                completed_samples // NORMALIZED_SAMPLE_RATE + 1
            ) * NORMALIZED_SAMPLE_RATE
    detector.flush()
    collect_ready_segments()

    pad_samples = round(SPEECH_REGION_PAD_SEC * NORMALIZED_SAMPLE_RATE)
    pad_sec = pad_samples / NORMALIZED_SAMPLE_RATE
    audio_duration_sec = normalized_audio_duration(samples)
    padded_regions = [
        SpeechRegion(
            start_sec=max(0.0, region.start_sec - pad_sec),
            end_sec=min(audio_duration_sec, region.end_sec + pad_sec),
        )
        for region in regions
    ]
    return normalize_regions(padded_regions)


def _silero_vad_model_path() -> resources.abc.Traversable:
    return resources.files("webinar_transcriber.assets").joinpath("silero_vad.onnx")
