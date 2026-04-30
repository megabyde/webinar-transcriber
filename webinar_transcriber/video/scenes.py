"""Scene detection for webinar videos."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import av
from av.filter import Graph

from webinar_transcriber.media import (
    MediaProcessingError,
    _required_video_stream,
    open_input_media_container,
)
from webinar_transcriber.models import Scene

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from av.video.stream import VideoStream


MIN_SCENE_LENGTH_SEC = 2.0
SCENE_SCAN_FPS = 0.5
SCENE_SCORE_THRESHOLD = 0.05


@dataclass(frozen=True)
class SceneDetectionSettings:
    """Policy settings for sampled scene-change detection."""

    scan_fps: float = SCENE_SCAN_FPS
    score_threshold: float = SCENE_SCORE_THRESHOLD
    min_scene_length_sec: float = MIN_SCENE_LENGTH_SEC


DEFAULT_SCENE_DETECTION_SETTINGS = SceneDetectionSettings()


def detect_scenes(
    video_path: Path,
    *,
    duration_sec: float | None = None,
    settings: SceneDetectionSettings = DEFAULT_SCENE_DETECTION_SETTINGS,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[Scene]:
    """Detect slide changes with PyAV's native scene filter.

    Returns:
        list[Scene]: The detected scene list.
    """
    scene_starts, resolved_duration_sec = _select_scene_starts(
        video_path,
        settings=settings,
        progress_callback=progress_callback,
    )
    duration_sec = resolved_duration_sec if duration_sec is None else duration_sec
    scene_bounds = list(zip(scene_starts, [*scene_starts[1:], duration_sec], strict=False))
    return [
        Scene(id=f"scene-{index}", start_sec=float(start), end_sec=float(end))
        for index, (start, end) in enumerate(scene_bounds, start=1)
    ]


def _select_scene_starts(
    video_path: Path,
    *,
    settings: SceneDetectionSettings,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[list[float], float]:
    try:
        with open_input_media_container(
            video_path, error_message="Could not open {path} with PyAV for scene detection: {error}"
        ) as input_container:
            video_stream = _required_video_stream(
                input_container, error_message=f"No video stream found in {video_path}."
            )
            scene_filter = _build_scene_filter_graph(video_stream, settings=settings)
            scene_starts = [0.0]
            processed_sample_count = 0
            reported_scene_count = 0

            for decoded_frame in input_container.decode(video_stream):
                scene_filter.vpush(decoded_frame)
                while True:
                    try:
                        filtered_frame = scene_filter.vpull()
                    except av.BlockingIOError:
                        break
                    current_time = _frame_time_sec(filtered_frame)
                    if current_time is None:
                        continue
                    if (current_time - scene_starts[-1]) >= settings.min_scene_length_sec:
                        scene_starts.append(current_time)
                if progress_callback is not None:
                    processed_sample_count, reported_scene_count = _report_scene_progress(
                        progress_callback,
                        decoded_frame,
                        settings=settings,
                        processed_sample_count=processed_sample_count,
                        reported_scene_count=reported_scene_count,
                        scene_count=len(scene_starts),
                    )

            duration_sec = _video_duration_sec(input_container, video_stream)
            if progress_callback is not None:
                _report_final_scene_progress(
                    progress_callback,
                    duration_sec=duration_sec,
                    settings=settings,
                    processed_sample_count=processed_sample_count,
                    reported_scene_count=reported_scene_count,
                    scene_count=len(scene_starts),
                )
            return scene_starts, duration_sec
    except (OSError, av.FFmpegError) as error:
        raise MediaProcessingError(f"Could not detect scenes in {video_path}: {error}") from error


def _build_scene_filter_graph(
    video_stream: VideoStream, *, settings: SceneDetectionSettings
) -> Graph:
    graph = Graph()
    source = graph.add_buffer(template=video_stream)
    fps = graph.add("fps", fps=f"{settings.scan_fps:g}")
    select = graph.add("select", expr=f"gt(scene,{settings.score_threshold})")
    sink = graph.add("buffersink")
    source.link_to(fps)
    fps.link_to(select)
    select.link_to(sink)
    graph.configure()
    return graph


def _sample_count_for_frame(
    frame: object, *, settings: SceneDetectionSettings, fallback: int
) -> int:
    frame_time_sec = _frame_time_sec(frame)
    if frame_time_sec is None:
        return fallback
    return max(1, int(frame_time_sec * settings.scan_fps) + 1)


def _report_scene_progress(
    progress_callback: Callable[[int, int], None],
    frame: object,
    *,
    settings: SceneDetectionSettings,
    processed_sample_count: int,
    reported_scene_count: int,
    scene_count: int,
) -> tuple[int, int]:
    sample_count = _sample_count_for_frame(
        frame, settings=settings, fallback=processed_sample_count + 1
    )
    if sample_count <= processed_sample_count:
        return processed_sample_count, reported_scene_count
    progress_callback(sample_count, scene_count)
    return sample_count, scene_count


def _report_final_scene_progress(
    progress_callback: Callable[[int, int], None],
    *,
    duration_sec: float,
    settings: SceneDetectionSettings,
    processed_sample_count: int,
    reported_scene_count: int,
    scene_count: int,
) -> None:
    final_sample_count = max(
        processed_sample_count, estimated_scene_sample_count(duration_sec, settings=settings)
    )
    if final_sample_count != processed_sample_count or reported_scene_count != scene_count:
        progress_callback(final_sample_count, scene_count)


def estimated_scene_sample_count(
    duration_sec: float, *, settings: SceneDetectionSettings = DEFAULT_SCENE_DETECTION_SETTINGS
) -> int:
    """Return the number of scene-detection samples expected for a duration."""
    if duration_sec <= 0:
        return 1
    return max(1, math.ceil(duration_sec * settings.scan_fps))


def _frame_time_sec(frame: object) -> float | None:
    if (time := getattr(frame, "time", None)) is not None:
        return float(time)
    pts = getattr(frame, "pts", None)
    time_base = getattr(frame, "time_base", None)
    if pts is None or time_base is None:
        return None
    return float(pts * time_base)


def _video_duration_sec(input_container: object, video_stream: object) -> float:
    if (duration := getattr(input_container, "duration", None)) is not None:
        return float(duration / av.time_base)
    stream_duration = getattr(video_stream, "duration", None)
    stream_time_base = getattr(video_stream, "time_base", None)
    if stream_duration is None or stream_time_base is None:
        return 0.0
    return float(stream_duration * stream_time_base)
