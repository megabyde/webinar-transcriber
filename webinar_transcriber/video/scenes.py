"""Scene detection for webinar videos."""

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import av
from av.filter import Graph

from webinar_transcriber.media import (
    MediaProcessingError,
    _required_input_stream,
    open_input_media_container,
)
from webinar_transcriber.models import Scene

if TYPE_CHECKING:
    from av.video.stream import VideoStream


MIN_SCENE_LENGTH_SEC = 3.0
SCENE_SCORE_THRESHOLD = 0.008


def detect_scenes(
    video_path: Path,
    *,
    duration_sec: float | None = None,
    min_scene_length_sec: float = MIN_SCENE_LENGTH_SEC,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[Scene]:
    """Detect slide changes with PyAV's native scene filter.

    Returns:
        list[Scene]: The detected scene list.
    """
    scene_starts, resolved_duration_sec = _select_scene_starts(
        video_path,
        scene_score_threshold=SCENE_SCORE_THRESHOLD,
        min_scene_length_sec=min_scene_length_sec,
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
    scene_score_threshold: float,
    min_scene_length_sec: float,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[list[float], float]:
    try:
        with open_input_media_container(
            video_path,
            error_message="Could not open {path} with PyAV for scene detection: {error}",
        ) as input_container:
            video_stream = _required_input_stream(
                input_container,
                "video",
                error_message=f"No video stream found in {video_path}.",
            )
            scene_filter = _build_scene_filter_graph(
                video_stream,
                scene_score_threshold=scene_score_threshold,
            )
            scene_starts = [0.0]
            processed_frame_count = 0

            for decoded_frame in input_container.decode(video_stream):
                processed_frame_count += 1
                scene_filter.vpush(decoded_frame)
                while True:
                    try:
                        filtered_frame = scene_filter.vpull()
                    except av.BlockingIOError:
                        break
                    current_time = _frame_time_sec(filtered_frame)
                    if current_time is None:
                        continue
                    if (current_time - scene_starts[-1]) >= min_scene_length_sec:
                        scene_starts.append(current_time)
                if progress_callback is not None:
                    progress_callback(processed_frame_count, len(scene_starts))

            return scene_starts, _video_duration_sec(input_container, video_stream)
    except (OSError, av.FFmpegError) as error:
        raise MediaProcessingError(f"Could not detect scenes in {video_path}: {error}") from error


def _build_scene_filter_graph(
    video_stream: "VideoStream", *, scene_score_threshold: float
) -> Graph:
    graph = Graph()
    source = graph.add_buffer(template=video_stream)
    select = graph.add("select", expr=f"gt(scene,{scene_score_threshold})")
    sink = graph.add("buffersink")
    source.link_to(select)
    select.link_to(sink)
    graph.configure()
    return graph


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
