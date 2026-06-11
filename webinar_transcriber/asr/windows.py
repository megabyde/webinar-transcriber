"""Whisper inference-window planning over detected speech regions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from webinar_transcriber.models import InferenceWindow

if TYPE_CHECKING:
    from webinar_transcriber.models import SpeechRegion

INFERENCE_WINDOW_DURATION_SEC = 28.0
INFERENCE_WINDOW_OVERLAP_SEC = 2.0


def plan_inference_windows(speech_regions: list[SpeechRegion]) -> list[InferenceWindow]:
    """Plan bounded Whisper inference windows from speech regions.

    Returns:
        list[InferenceWindow]: Ordered windows with overlap inside long regions.
    """
    windows: list[InferenceWindow] = []

    for region_index, region in enumerate(speech_regions):
        if region.duration_sec <= 0:
            continue

        window_start_sec = region.start_sec
        while window_start_sec < region.end_sec:
            window_end_sec = min(region.end_sec, window_start_sec + INFERENCE_WINDOW_DURATION_SEC)
            windows.append(
                InferenceWindow(
                    id=f"window-{len(windows) + 1}",
                    region_index=region_index,
                    start_sec=window_start_sec,
                    end_sec=window_end_sec,
                )
            )
            if window_end_sec >= region.end_sec:
                break
            window_start_sec = max(window_start_sec, window_end_sec - INFERENCE_WINDOW_OVERLAP_SEC)

    return windows
