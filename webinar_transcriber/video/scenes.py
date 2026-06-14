"""Scene detection and representative-frame capture for webinar videos."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import av
from av.filter import Graph

from webinar_transcriber.media import MediaProcessingError, open_video_input_container
from webinar_transcriber.models import Scene

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path
    from typing import Protocol

    from av.video.frame import VideoFrame
    from av.video.stream import VideoStream

    ProgressCallback = Callable[[float, int], None]

    class SavableImage(Protocol):
        """The slice of a PIL image the frame writer needs: save to a path."""

        def save(self, fp: Path, /) -> None: ...

    SceneOpener = tuple[float, SavableImage]


MIN_SCENE_LENGTH_SEC = 2.0
SCENE_SCAN_FPS = 0.5
SCENE_SCORE_THRESHOLD = 0.05
SCENE_ONE_FRAME_OFFSET_SEC = 1.0


def detect_scenes(
    video_path: Path,
    frames_dir: Path,
    duration_sec: float,
    *,
    progress_callback: ProgressCallback | None = None,
) -> list[Scene]:
    """Detect slide changes and save the frame that opened each scene.

    PyAV's scene filter already yields the frame at each slide change, so the
    decode pass that finds scene boundaries saves those frames as the scenes'
    representative images instead of re-decoding near each boundary in a second
    pass. Scene one has no slide-change frame of its own, so it walks forward to the
    frame about a second in (SCENE_ONE_FRAME_OFFSET_SEC) to avoid a black or fade-in
    opening frame. Each returned scene carries the run-directory-relative path of its
    saved frame.

    Returns:
        list[Scene]: Detected scenes, each with its representative frame image path.
    """
    scene_openers = _select_scene_openers(
        video_path, duration_sec, progress_callback=progress_callback
    )
    scene_starts = [start_sec for start_sec, _image in scene_openers]
    scene_bounds = zip(scene_starts, [*scene_starts[1:], duration_sec], strict=False)

    frames_dir.mkdir(parents=True, exist_ok=True)
    scenes: list[Scene] = []
    for index, ((start_sec, next_start), (_start, opener_image)) in enumerate(
        zip(scene_bounds, scene_openers, strict=False), start=1
    ):
        # Clamp guards the case where the last scene-change frame sits past the probed
        # duration, which would otherwise produce a zero- or negative-length scene.
        end_sec = min(next_start, duration_sec)
        if end_sec <= start_sec:
            continue
        output_path = frames_dir / f"scene-{index}.png"
        opener_image.save(output_path)
        scenes.append(
            Scene(
                id=f"scene-{index}",
                start_sec=start_sec,
                end_sec=end_sec,
                image_path=output_path.relative_to(frames_dir.parent).as_posix(),
            )
        )
    return scenes


def _select_scene_openers(
    video_path: Path,
    duration_sec: float,
    *,
    progress_callback: ProgressCallback | None = None,
) -> list[SceneOpener]:
    """Return each scene start paired with the frame image that opened it."""
    try:
        with open_video_input_container(video_path) as (input_container, video_stream):
            scene_filter = _build_scene_filter_graph(video_stream)
            openers: list[SceneOpener] = []
            processed_sample_count = 0
            reported_scene_count = 0

            for decoded_frame in input_container.decode(video_stream):
                _seed_or_refine_scene_one(openers, decoded_frame)
                scene_filter.vpush(decoded_frame)
                for current_time, filtered_frame in _filtered_scene_frames(scene_filter):
                    if (current_time - openers[-1][0]) >= MIN_SCENE_LENGTH_SEC:
                        openers.append((current_time, _frame_to_image(filtered_frame)))
                if progress_callback is not None:
                    sample_count = _sample_count_for_frame(
                        decoded_frame, fallback=processed_sample_count + 1
                    )
                    if sample_count > processed_sample_count:
                        progress_callback(sample_count, len(openers))
                        processed_sample_count = sample_count
                        reported_scene_count = len(openers)

            if progress_callback is not None:
                final_sample_count = max(
                    processed_sample_count, estimated_scene_sample_count(duration_sec)
                )
                if (  # pragma: no cover - backend-dependent final progress reconciliation
                    final_sample_count != processed_sample_count
                    or reported_scene_count != len(openers)
                ):
                    progress_callback(final_sample_count, len(openers))
            return openers
    except (OSError, av.FFmpegError) as ex:  # pragma: no cover - defensive FFmpeg boundary
        raise MediaProcessingError(f"Could not detect scenes in {video_path}: {ex}") from ex


def _seed_or_refine_scene_one(openers: list[SceneOpener], decoded_frame: VideoFrame) -> None:
    """Seed scene one with the first frame, then walk it forward through the first second.

    Scene one has no slide-change frame, so its image advances with each decoded frame until
    one reaches ``SCENE_ONE_FRAME_OFFSET_SEC``, landing on real content instead of a black or
    fade-in opening frame. Refinement stops once the first real scene starts.
    """
    if not openers:
        openers.append((0.0, _frame_to_image(decoded_frame)))
    elif len(openers) == 1 and (_frame_time_sec(decoded_frame) or 0.0) < SCENE_ONE_FRAME_OFFSET_SEC:
        openers[0] = (0.0, _frame_to_image(decoded_frame))


def _filtered_scene_frames(scene_filter: Graph) -> Iterator[tuple[float, VideoFrame]]:
    """Yield (timestamp, frame) for each scene-change frame buffered in the filter graph."""
    while True:
        try:
            filtered_frame = scene_filter.vpull()
        except av.BlockingIOError:
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
