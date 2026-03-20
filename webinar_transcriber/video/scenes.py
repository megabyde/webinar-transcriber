"""Scene detection for webinar videos."""

from collections.abc import Iterable
from pathlib import Path

import av
import numpy as np

from webinar_transcriber.models import Scene

SAMPLE_INTERVAL_SEC = 1.0
MIN_SCENE_LENGTH_SEC = 3.0
DIFFERENCE_THRESHOLD = 12.0
TARGET_SAMPLE_WIDTH = 160
TARGET_SAMPLE_HEIGHT = 90


def detect_scenes(
    video_path: Path,
    *,
    min_scene_length_sec: float = MIN_SCENE_LENGTH_SEC,
) -> list[Scene]:
    """Detect slide changes by sampling the video once per second."""
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        duration_sec = (
            float(stream.duration) * float(stream.time_base)
            if stream.duration is not None and stream.time_base is not None
            else 0.0
        )
        sampled_frames: list[tuple[float, np.ndarray]] = []
        previous_time = 0.0
        next_sample_time = 0.0

        for frame in container.decode(stream):
            current_time = float(frame.time or 0.0)
            previous_time = current_time
            if current_time + 1e-6 < next_sample_time:
                continue

            sampled_frames.append((current_time, _frame_sample(frame)))
            next_sample_time = current_time + SAMPLE_INTERVAL_SEC

        if duration_sec <= 0.0:
            duration_sec = previous_time

    scene_starts = _detect_scene_start_times(
        sampled_frames,
        min_scene_length_sec=min_scene_length_sec,
    )
    if not scene_starts:
        scene_starts = [0.0]

    scene_bounds = list(zip(scene_starts, [*scene_starts[1:], duration_sec], strict=False))
    return [
        Scene(id=f"scene-{index}", start_sec=float(start), end_sec=float(end))
        for index, (start, end) in enumerate(scene_bounds, start=1)
    ]


def _detect_scene_start_times(
    sampled_frames: Iterable[tuple[float, np.ndarray]],
    *,
    difference_threshold: float = DIFFERENCE_THRESHOLD,
    min_scene_length_sec: float = MIN_SCENE_LENGTH_SEC,
) -> list[float]:
    samples = list(sampled_frames)
    if not samples:
        return []

    scene_starts = [float(samples[0][0])]
    accepted_frame = samples[0][1]

    for current_time, current_frame in samples[1:]:
        if (current_time - scene_starts[-1]) < min_scene_length_sec:
            continue

        difference = float(np.abs(current_frame - accepted_frame).mean())
        if difference >= difference_threshold:
            scene_starts.append(float(current_time))
            accepted_frame = current_frame

    return scene_starts


def _frame_sample(frame: av.VideoFrame) -> np.ndarray:
    grayscale = frame.to_ndarray(format="gray")
    step_y = max(grayscale.shape[0] // TARGET_SAMPLE_HEIGHT, 1)
    step_x = max(grayscale.shape[1] // TARGET_SAMPLE_WIDTH, 1)
    return grayscale[::step_y, ::step_x].astype(np.float32)
