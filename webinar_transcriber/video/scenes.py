"""Scene detection for webinar videos."""

import io
import math
import select
import subprocess
from collections.abc import Callable, Iterable
from pathlib import Path
from time import perf_counter
from typing import IO

import imagehash
import numpy as np
from PIL import Image

from webinar_transcriber.media import MEDIA_COMMAND_TIMEOUT_SEC, MediaProcessingError
from webinar_transcriber.models import Scene

SAMPLE_INTERVAL_SEC = 1.0
MIN_SCENE_LENGTH_SEC = 3.0
DIFFERENCE_THRESHOLD = 48
TARGET_SAMPLE_WIDTH = 256
TARGET_SAMPLE_HEIGHT = 144
PHASH_HASH_SIZE = 16
SCENE_SAMPLE_TIMEOUT_SEC = MEDIA_COMMAND_TIMEOUT_SEC


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
    frame_size = TARGET_SAMPLE_WIDTH * TARGET_SAMPLE_HEIGHT
    try:
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
    except OSError as error:
        raise MediaProcessingError(
            f"Could not start ffmpeg for scene detection: {error}"
        ) from error

    if process.stdout is None or process.stderr is None:
        process.kill()
        raise MediaProcessingError("Could not open ffmpeg pipes for scene detection.")

    start_time = perf_counter()
    partial_frame = bytearray()

    try:
        sample_index = 0
        while chunk := _read_stdout_chunk(
            process.stdout,
            size=frame_size - len(partial_frame),
            timeout_sec=_remaining_scene_sample_timeout(start_time),
        ):
            partial_frame.extend(chunk)
            if len(partial_frame) < frame_size:
                continue
            frame_bytes = bytes(partial_frame)
            frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((
                TARGET_SAMPLE_HEIGHT,
                TARGET_SAMPLE_WIDTH,
            ))
            yield sample_index * SAMPLE_INTERVAL_SEC, frame
            sample_index += 1
            partial_frame.clear()
        process.wait(timeout=_remaining_scene_sample_timeout(start_time))
    except subprocess.TimeoutExpired as error:
        process.kill()
        process.wait()
        raise MediaProcessingError(
            f"ffmpeg scene sampling timed out after {SCENE_SAMPLE_TIMEOUT_SEC:g}s."
        ) from error

    stderr_data = process.stderr.read()
    stderr_text = (
        stderr_data.decode(errors="replace").strip()
        if isinstance(stderr_data, bytes)
        else str(stderr_data).strip()
    )
    if process.returncode != 0:
        raise MediaProcessingError(stderr_text or "ffmpeg scene sampling failed.")

    if partial_frame:
        raise MediaProcessingError("Incomplete frame sample returned by ffmpeg.")


def _remaining_scene_sample_timeout(start_time: float) -> float:
    remaining = SCENE_SAMPLE_TIMEOUT_SEC - (perf_counter() - start_time)
    if remaining <= 0:
        raise subprocess.TimeoutExpired(cmd=["ffmpeg"], timeout=SCENE_SAMPLE_TIMEOUT_SEC)
    return remaining


def _read_stdout_chunk(stream: IO[bytes], *, size: int, timeout_sec: float) -> bytes:
    try:
        fileno = stream.fileno()
    except (AttributeError, io.UnsupportedOperation, OSError):
        return stream.read(size)
    ready, _, _ = select.select([fileno], [], [], timeout_sec)
    if not ready:
        raise subprocess.TimeoutExpired(cmd=["ffmpeg"], timeout=SCENE_SAMPLE_TIMEOUT_SEC)
    return stream.read(size)


def _estimate_sample_end_time(last_sample_time: float) -> float:
    if last_sample_time <= 0:
        return 0.0
    return last_sample_time + SAMPLE_INTERVAL_SEC
