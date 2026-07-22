"""Scene detection and representative-frame capture for webinar videos."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING

from av.error import BlockingIOError as AvBlockingIOError
from av.error import FFmpegError
from av.filter import Graph

from webinar_transcriber.media import MediaProcessingError, open_video_input_container
from webinar_transcriber.models import Scene

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path
    from typing import Protocol

    from av.container import InputContainer
    from av.video.frame import VideoFrame
    from av.video.stream import VideoStream

    SceneProgressCallback = Callable[[float, int], None]
    FrameProgressCallback = Callable[[int], None]

    class SavableImage(Protocol):
        """PIL-compatible image that can be saved to a path."""

        def save(self, fp: Path, /) -> None: ...


MIN_SCENE_LENGTH_SEC = 2.0
SCENE_SCAN_FPS = 0.5
SCENE_SCORE_THRESHOLD = 0.05


def detect_scenes(
    video_path: Path,
    duration_sec: float,
    *,
    progress_callback: SceneProgressCallback | None = None,
) -> list[Scene]:
    """Detect slide-change boundaries as time-bounded scenes, without capturing frames.

    A low-FPS decode pass finds slide changes; the representative frame for each scene is captured
    separately by ``save_scene_frames``. Each returned scene has no ``image_path`` yet.

    Returns:
        list[Scene]: Detected scenes in timeline order.
    """
    scene_starts = _detect_scene_starts(
        video_path, duration_sec, progress_callback=progress_callback
    )
    scene_bounds = zip(scene_starts, [*scene_starts[1:], duration_sec], strict=False)

    scenes: list[Scene] = []
    for index, (start_sec, next_start) in enumerate(scene_bounds, start=1):
        # The final scene-change frame can sit past the probed duration. Clamp it to avoid a zero-
        # or negative-length scene.
        end_sec = min(next_start, duration_sec)
        if end_sec <= start_sec:
            continue
        scenes.append(Scene(id=f"scene-{index}", start_sec=start_sec, end_sec=end_sec))
    return scenes


def save_scene_frames(
    video_path: Path,
    scenes: list[Scene],
    frames_dir: Path,
    *,
    progress_callback: FrameProgressCallback | None = None,
) -> list[Scene]:
    """Save a representative frame from the middle of each scene.

    Seeking to a scene's midpoint lands on a settled slide rather than a slide-change or fade-in
    frame, so it needs no special-casing for the opening scene.

    Returns:
        list[Scene]: The scenes with ``image_path`` set to the saved frame.
    """
    if not scenes:
        return []

    frames_dir.mkdir(parents=True, exist_ok=True)
    captured: list[Scene] = []
    try:
        with open_video_input_container(video_path) as (input_container, video_stream):
            for scene in scenes:
                midpoint_sec = (scene.start_sec + scene.end_sec) / 2.0
                image = _decode_frame_at(input_container, video_stream, midpoint_sec)
                output_path = frames_dir / f"{scene.id}.png"
                image.save(output_path)
                captured.append(
                    replace(scene, image_path=output_path.relative_to(frames_dir.parent).as_posix())
                )
                if progress_callback is not None:
                    progress_callback(len(captured))
            return captured
    except (OSError, FFmpegError) as ex:  # pragma: no cover - defensive FFmpeg boundary
        raise MediaProcessingError(f"Could not save scene frames for {video_path}: {ex}") from ex


def _detect_scene_starts(
    video_path: Path,
    duration_sec: float,
    *,
    progress_callback: SceneProgressCallback | None = None,
) -> list[float]:
    """Return the start time of each detected scene; scene one starts at the top of the video."""
    try:
        with open_video_input_container(video_path) as (input_container, video_stream):
            scene_filter = _build_scene_filter_graph(video_stream)
            starts: list[float] = [0.0]
            processed_sample_count = 0
            reported_scene_count = 0

            for decoded_frame in input_container.decode(video_stream):
                scene_filter.vpush(decoded_frame)
                for current_time, _filtered_frame in _filtered_scene_frames(scene_filter):
                    if (current_time - starts[-1]) >= MIN_SCENE_LENGTH_SEC:
                        starts.append(current_time)
                if progress_callback is not None:
                    sample_count = _sample_count_for_frame(
                        decoded_frame, fallback=processed_sample_count + 1
                    )
                    if sample_count > processed_sample_count:
                        progress_callback(sample_count, len(starts))
                        processed_sample_count = sample_count
                        reported_scene_count = len(starts)

            if progress_callback is not None:
                final_sample_count = max(
                    processed_sample_count, estimated_scene_sample_count(duration_sec)
                )
                if (  # pragma: no cover - backend-dependent final progress reconciliation
                    final_sample_count != processed_sample_count
                    or reported_scene_count != len(starts)
                ):
                    progress_callback(final_sample_count, len(starts))
            return starts
    except (OSError, FFmpegError) as ex:  # pragma: no cover - defensive FFmpeg boundary
        raise MediaProcessingError(f"Could not detect scenes in {video_path}: {ex}") from ex


def _decode_frame_at(
    input_container: InputContainer, video_stream: VideoStream, target_sec: float
) -> SavableImage:
    """Seek to ``target_sec`` and return the first decoded frame at or after it."""
    if video_stream.time_base is not None:
        input_container.seek(int(target_sec / video_stream.time_base), stream=video_stream)
    fallback_image: SavableImage | None = None
    for frame in input_container.decode(video_stream):
        image = _frame_to_image(frame)
        frame_time = _frame_time_sec(frame)
        if frame_time is None or frame_time >= target_sec:
            return image
        fallback_image = image
    if fallback_image is None:  # pragma: no cover - a seekable stream yields at least one frame
        raise MediaProcessingError(f"No frame decoded near {target_sec:.2f}s.")
    return fallback_image


def _filtered_scene_frames(scene_filter: Graph) -> Iterator[tuple[float, VideoFrame]]:
    """Yield each buffered scene-change frame with its timestamp."""
    while True:
        try:
            filtered_frame = scene_filter.vpull()
        except AvBlockingIOError:
            return
        current_time = _frame_time_sec(filtered_frame)
        if current_time is not None:
            yield current_time, filtered_frame


def _frame_to_image(frame: VideoFrame) -> SavableImage:
    return frame.to_image().convert("RGB")


def _build_scene_filter_graph(video_stream: VideoStream) -> Graph:
    """Build the PyAV graph that downsamples to SCENE_SCAN_FPS and emits slide-change frames."""
    graph = Graph()
    source = graph.add_buffer(template=video_stream)
    fps = graph.add("fps", fps=f"{SCENE_SCAN_FPS:g}")
    select = graph.add("select", expr=f"gt(scene,{SCENE_SCORE_THRESHOLD})")
    sink = graph.add("buffersink")
    source.link_to(fps)
    fps.link_to(select)
    select.link_to(sink)
    graph.configure()
    return graph


def _sample_count_for_frame(frame: VideoFrame, *, fallback: int) -> int:
    frame_time_sec = _frame_time_sec(frame)
    if frame_time_sec is None:  # pragma: no cover - PyAV decoded frames normally carry time
        return fallback
    return max(1, int(frame_time_sec * SCENE_SCAN_FPS) + 1)


def estimated_scene_sample_count(duration_sec: float) -> int:
    """Return the number of scene-detection samples expected for a duration."""
    return max(1, math.ceil(duration_sec * SCENE_SCAN_FPS))


def _frame_time_sec(frame: VideoFrame) -> float | None:
    if frame.time is None:  # pragma: no cover - fallback for frames without direct time
        pts = frame.pts
        time_base = frame.time_base
        if pts is None or time_base is None:
            return None
        return float(pts * time_base)
    return float(frame.time)
