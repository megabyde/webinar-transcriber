"""Scene detection for webinar videos."""

import math
import subprocess
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np

from webinar_transcriber.media import MediaProcessingError
from webinar_transcriber.models import Scene

SAMPLE_INTERVAL_SEC = 1.0
MIN_SCENE_LENGTH_SEC = 3.0
DIFFERENCE_THRESHOLD = 12.0
TARGET_SAMPLE_WIDTH = 160
TARGET_SAMPLE_HEIGHT = 90


def detect_scenes(
    video_path: Path,
    *,
    duration_sec: float | None = None,
    min_scene_length_sec: float = MIN_SCENE_LENGTH_SEC,
    progress_callback: Callable[[], None] | None = None,
) -> list[Scene]:
    """Detect slide changes by sampling the video once per second."""
    scene_starts: list[float] = []
    accepted_frame: np.ndarray | None = None
    last_sample_time = 0.0

    for current_time, current_frame in _iter_sampled_frames(video_path):
        last_sample_time = current_time
        if progress_callback is not None:
            progress_callback()

        if accepted_frame is None:
            scene_starts.append(current_time)
            accepted_frame = current_frame
            continue

        if (current_time - scene_starts[-1]) < min_scene_length_sec:
            continue

        difference = float(np.abs(current_frame - accepted_frame).mean())
        if difference >= DIFFERENCE_THRESHOLD:
            scene_starts.append(current_time)
            accepted_frame = current_frame

    if duration_sec is None:
        duration_sec = _estimate_sample_end_time(last_sample_time)

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


def estimate_sample_count(duration_sec: float) -> int:
    """Estimate how many one-second scene samples will be processed."""
    if duration_sec <= 0:
        return 1
    return max(1, math.ceil(duration_sec / SAMPLE_INTERVAL_SEC))


def _iter_sampled_frames(video_path: Path) -> Iterable[tuple[float, np.ndarray]]:
    frame_size = TARGET_SAMPLE_WIDTH * TARGET_SAMPLE_HEIGHT
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-vf",
            (
                f"fps={1 / SAMPLE_INTERVAL_SEC:g},"
                f"scale={TARGET_SAMPLE_WIDTH}:{TARGET_SAMPLE_HEIGHT},"
                "format=gray"
            ),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if process.stdout is None or process.stderr is None:
        process.kill()
        raise MediaProcessingError("Could not open ffmpeg pipes for scene detection.")

    sample_index = 0
    try:
        while True:
            chunk = process.stdout.read(frame_size)
            if len(chunk) == 0:
                break
            if len(chunk) != frame_size:
                process.kill()
                raise MediaProcessingError("Incomplete frame sample returned by ffmpeg.")

            frame = np.frombuffer(chunk, dtype=np.uint8).reshape((
                TARGET_SAMPLE_HEIGHT,
                TARGET_SAMPLE_WIDTH,
            ))
            yield sample_index * SAMPLE_INTERVAL_SEC, frame.astype(np.float32)
            sample_index += 1
    finally:
        stderr_output = process.stderr.read().strip()
        return_code = process.wait()

    if return_code != 0:
        raise MediaProcessingError(stderr_output or "ffmpeg scene sampling failed.")


def _estimate_sample_end_time(last_sample_time: float) -> float:
    if last_sample_time <= 0:
        return 0.0
    return last_sample_time + SAMPLE_INTERVAL_SEC
