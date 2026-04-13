"""Scene detection for webinar videos."""

import io
import math
import os
import select
import subprocess
from collections.abc import Callable, Iterable
from contextlib import suppress
from pathlib import Path
from time import monotonic

import numpy as np

from webinar_transcriber.media import MEDIA_COMMAND_TIMEOUT_SEC, MediaProcessingError
from webinar_transcriber.models import Scene

SAMPLE_INTERVAL_SEC = 1.0
MIN_SCENE_LENGTH_SEC = 3.0
DIFFERENCE_THRESHOLD = 12.0
TARGET_SAMPLE_WIDTH = 160
TARGET_SAMPLE_HEIGHT = 90
SCENE_SAMPLE_TIMEOUT_SEC = MEDIA_COMMAND_TIMEOUT_SEC


def detect_scenes(
    video_path: Path,
    *,
    duration_sec: float | None = None,
    min_scene_length_sec: float = MIN_SCENE_LENGTH_SEC,
    progress_callback: Callable[[int], None] | None = None,
) -> list[Scene]:
    """Detect slide changes by sampling the video once per second."""
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
    progress_callback: Callable[[int], None] | None = None,
) -> list[float]:
    scene_starts: list[float] = []
    accepted_frame: np.ndarray | None = None

    for current_time, current_frame in sampled_frames:
        if accepted_frame is None:
            scene_starts.append(float(current_time))
            accepted_frame = current_frame
            if progress_callback is not None:
                progress_callback(len(scene_starts))
            continue

        if (current_time - scene_starts[-1]) < min_scene_length_sec:
            if progress_callback is not None:
                progress_callback(len(scene_starts))
            continue

        difference = float(np.abs(current_frame - accepted_frame).mean())
        if difference >= difference_threshold:
            scene_starts.append(float(current_time))
            # Compare against the scene's anchor frame, not the previous sample,
            # so gradual visual drift does not trigger spurious scene breaks.
            accepted_frame = current_frame
        if progress_callback is not None:
            progress_callback(len(scene_starts))

    return scene_starts


def estimate_sample_count(duration_sec: float) -> int:
    """Estimate how many one-second scene samples will be processed."""
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

    sample_index = 0
    deadline = monotonic() + SCENE_SAMPLE_TIMEOUT_SEC
    failure: BaseException | None = None
    try:
        while True:
            chunk = _read_with_timeout(process, frame_size=frame_size, deadline=deadline)
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
    except BaseException as error:
        failure = error
        raise
    finally:
        stderr_output = process.stderr.read()
        stderr_text = (
            stderr_output.decode(errors="replace").strip()
            if isinstance(stderr_output, bytes)
            else str(stderr_output).strip()
        )
        try:
            return_code = process.wait(timeout=1.0)
        except TypeError:
            return_code = process.wait()
        except subprocess.TimeoutExpired:
            process.kill()
            return_code = process.wait()
        if return_code != 0 and failure is None:
            raise MediaProcessingError(stderr_text or "ffmpeg scene sampling failed.")


def _estimate_sample_end_time(last_sample_time: float) -> float:
    if last_sample_time <= 0:
        return 0.0
    return last_sample_time + SAMPLE_INTERVAL_SEC


def _read_with_timeout(
    process: subprocess.Popen[bytes], *, frame_size: int, deadline: float
) -> bytes:
    if process.stdout is None:
        raise MediaProcessingError("Could not open ffmpeg pipes for scene detection.")

    chunk = bytearray()
    while len(chunk) < frame_size:
        remaining_sec = deadline - monotonic()
        if remaining_sec <= 0:
            process.kill()
            with suppress(ProcessLookupError):
                process.wait()
            raise MediaProcessingError(
                f"ffmpeg scene sampling timed out after {SCENE_SAMPLE_TIMEOUT_SEC:g}s."
            )

        try:
            ready, _, _ = select.select([process.stdout], [], [], min(1.0, remaining_sec))
        except (AttributeError, io.UnsupportedOperation):
            data = process.stdout.read(frame_size - len(chunk))
            if len(data) == 0:
                return bytes(chunk)
            chunk.extend(data)
            continue
        if not ready:
            continue

        bytes_needed = frame_size - len(chunk)
        data = os.read(process.stdout.fileno(), bytes_needed)
        if len(data) == 0:
            return bytes(chunk)
        chunk.extend(data)

    return bytes(chunk)
