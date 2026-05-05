"""Representative frame extraction for detected scenes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import av
from PIL import ImageOps

from webinar_transcriber.media import (
    MediaProcessingError,
    open_video_input_container,
)
from webinar_transcriber.models import SlideFrame

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from av.container import InputContainer
    from av.video.stream import VideoStream

    from webinar_transcriber.models import Scene

REPRESENTATIVE_FRAME_OFFSET_SEC = 1.0


def extract_representative_frames(
    video_path: Path,
    scenes: list[Scene],
    frames_dir: Path,
    *,
    progress_callback: Callable[[], None] | None = None,
    warning_callback: Callable[[str], None] | None = None,
) -> list[SlideFrame]:
    """Extract one representative frame near the start of each scene.

    Returns:
        list[SlideFrame]: The successfully extracted slide frames.
    """
    frames_dir.mkdir(parents=True, exist_ok=True)
    frames: list[SlideFrame] = []
    unreported_scenes = list(scenes)

    try:
        with open_video_input_container(video_path) as (input_container, video_stream):
            for index, scene in enumerate(scenes, start=1):
                frame_timestamp_sec = min(
                    scene.end_sec, scene.start_sec + REPRESENTATIVE_FRAME_OFFSET_SEC
                )
                output_path = frames_dir / f"{scene.id}.png"
                extracted, failure_detail = _extract_frame_from_container(
                    input_container, video_stream, frame_timestamp_sec, output_path
                )
                if not extracted:
                    if warning_callback is not None:
                        warning_callback(
                            f"Frame extraction failed for {scene.id} at "
                            f"{frame_timestamp_sec:.1f}s: {failure_detail}"
                        )
                    if progress_callback is not None:
                        progress_callback()
                    unreported_scenes.pop(0)
                    continue

                if failure_detail is not None and warning_callback is not None:
                    warning_callback(
                        f"Frame extraction used nearest decoded frame for {scene.id} at "
                        f"{frame_timestamp_sec:.1f}s: {failure_detail}"
                    )
                frames.append(
                    SlideFrame(
                        id=f"frame-{index}",
                        scene_id=scene.id,
                        image_path=str(output_path),
                        timestamp_sec=frame_timestamp_sec,
                    )
                )
                if progress_callback is not None:
                    progress_callback()
                unreported_scenes.pop(0)
    except (MediaProcessingError, OSError, av.FFmpegError) as error:
        _report_frame_extraction_failures(
            unreported_scenes,
            error,
            progress_callback=progress_callback,
            warning_callback=warning_callback,
        )

    return frames


def _report_frame_extraction_failures(
    scenes: list[Scene],
    error: Exception,
    *,
    progress_callback: Callable[[], None] | None,
    warning_callback: Callable[[str], None] | None,
) -> None:
    for scene in scenes:
        frame_timestamp_sec = min(scene.end_sec, scene.start_sec + REPRESENTATIVE_FRAME_OFFSET_SEC)
        if warning_callback is not None:
            warning_callback(
                f"Frame extraction failed for {scene.id} at {frame_timestamp_sec:.1f}s: {error}"
            )
        if progress_callback is not None:
            progress_callback()


def _extract_frame_from_container(
    input_container: InputContainer,
    video_stream: VideoStream,
    timestamp_sec: float,
    output_path: Path,
) -> tuple[bool, str | None]:
    try:
        input_container.seek(int(max(timestamp_sec, 0.0) * av.time_base), backward=True)
        frame = None
        nearest_earlier_frame = None
        for decoded_frame in input_container.decode(video_stream):
            if decoded_frame.time is None:
                continue
            nearest_earlier_frame = decoded_frame
            if decoded_frame.time >= timestamp_sec:
                frame = decoded_frame
                break
        if frame is None:
            if nearest_earlier_frame is None:
                return False, f"PyAV did not decode a frame at {timestamp_sec:.3f}s"
            frame = nearest_earlier_frame
            fallback_detail = (
                f"PyAV decoded nearest frame before {timestamp_sec:.3f}s at {frame.time:.3f}s"
            )
        else:
            fallback_detail = None

        output_path.parent.mkdir(parents=True, exist_ok=True)
        ImageOps.exif_transpose(frame.to_image()).convert("RGB").save(output_path)
    except (OSError, av.FFmpegError) as error:  # pragma: no cover - PyAV/save defensive boundary
        return False, str(error)

    if not output_path.exists():  # pragma: no cover - PyAV/save defensive boundary
        return False, f"PyAV did not write {output_path}"
    return True, fallback_detail
