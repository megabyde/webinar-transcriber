"""Scene detection for webinar videos."""

import math
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, cast

import imagehash
import numpy as np
from PIL import Image

from webinar_transcriber.media import MediaProcessingError, open_input_media_container
from webinar_transcriber.models import Scene

if TYPE_CHECKING:
    from av.video.frame import VideoFrame

SAMPLE_INTERVAL_SEC = 1.0
MIN_SCENE_LENGTH_SEC = 3.0
DIFFERENCE_THRESHOLD = 48
TARGET_SAMPLE_WIDTH = 256
TARGET_SAMPLE_HEIGHT = 144
PHASH_HASH_SIZE = 16


def detect_scenes(
    video_path: Path,
    *,
    duration_sec: float | None = None,
    min_scene_length_sec: float = MIN_SCENE_LENGTH_SEC,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[Scene]:
    """Detect slide changes by sampling the video once per second.

    Returns:
        list[Scene]: The detected scene list.
    """
    last_sample_time = 0.0

    def sampled_frames() -> Iterable[tuple[float, np.ndarray]]:
        nonlocal last_sample_time
        for current_time, current_frame in _iter_sampled_frames(video_path):
            last_sample_time = current_time
            yield current_time, current_frame

    scene_starts = _detect_scene_start_times(
        sampled_frames(),
        difference_threshold=DIFFERENCE_THRESHOLD,
        min_scene_length_sec=min_scene_length_sec,
        progress_callback=progress_callback,
    )

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
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[float]:
    scene_starts: list[float] = []
    accepted_hash: imagehash.ImageHash | None = None

    for sample_count, (current_time, current_frame) in enumerate(sampled_frames, start=1):
        current_hash = imagehash.phash(
            Image.fromarray(current_frame.astype(np.uint8)),
            hash_size=PHASH_HASH_SIZE,
        )
        if accepted_hash is None:
            scene_starts.append(float(current_time))
            accepted_hash = current_hash
            if progress_callback is not None:
                progress_callback(sample_count, len(scene_starts))
            continue

        if (current_time - scene_starts[-1]) < min_scene_length_sec:
            if progress_callback is not None:
                progress_callback(sample_count, len(scene_starts))
            continue

        difference = current_hash - accepted_hash
        if difference >= difference_threshold:
            scene_starts.append(float(current_time))
            accepted_hash = current_hash
        if progress_callback is not None:
            progress_callback(sample_count, len(scene_starts))

    return scene_starts


def estimate_sample_count(duration_sec: float) -> int:
    """Estimate how many one-second scene samples will be processed.

    Returns:
        int: The estimated number of sampled frames.
    """
    if duration_sec <= 0:
        return 1
    return max(1, math.ceil(duration_sec / SAMPLE_INTERVAL_SEC))


def _iter_sampled_frames(video_path: Path) -> Iterable[tuple[float, np.ndarray]]:
    with open_input_media_container(
        video_path,
        error_message="Could not open {path} with PyAV for scene detection: {error}",
    ) as input_container:
        video_stream = next(
            (stream for stream in input_container.streams if stream.type == "video"),
            None,
        )
        if video_stream is None:
            raise MediaProcessingError(f"No video stream found in {video_path}.")

        sample_index = 0
        next_sample_time = 0.0
        for decoded_frame in input_container.decode(video_stream):
            frame = cast("VideoFrame", decoded_frame)
            if frame.time is None or frame.time < next_sample_time:
                continue
            sampled_frame = frame.reformat(
                width=TARGET_SAMPLE_WIDTH,
                height=TARGET_SAMPLE_HEIGHT,
                format="gray",
            )
            yield next_sample_time, sampled_frame.to_ndarray()
            sample_index += 1
            next_sample_time = sample_index * SAMPLE_INTERVAL_SEC


def _estimate_sample_end_time(last_sample_time: float) -> float:
    if last_sample_time <= 0:
        return 0.0
    return last_sample_time + SAMPLE_INTERVAL_SEC
