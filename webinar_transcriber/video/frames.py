"""Representative frame extraction for detected scenes."""

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, cast

import av
from PIL import Image, ImageOps

from webinar_transcriber.media import MediaProcessingError, open_input_media_container
from webinar_transcriber.models import Scene, SlideFrame

if TYPE_CHECKING:
    from av.video.frame import VideoFrame

REPRESENTATIVE_FRAME_OFFSET_SEC = 0.5


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

    for index, scene in enumerate(scenes, start=1):
        frame_timestamp_sec = min(scene.end_sec, scene.start_sec + REPRESENTATIVE_FRAME_OFFSET_SEC)
        output_path = frames_dir / f"{scene.id}.png"
        extracted, failure_detail = _extract_frame(video_path, frame_timestamp_sec, output_path)
        if not extracted:
            if warning_callback is not None:
                warning_callback(
                    f"Frame extraction failed for {scene.id} at {frame_timestamp_sec:.1f}s: "
                    f"{failure_detail}"
                )
            if progress_callback is not None:
                progress_callback()
            continue

        _normalize_extracted_frame(output_path)
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

    return frames


def _extract_frame(
    video_path: Path, timestamp_sec: float, output_path: Path
) -> tuple[bool, str | None]:
    try:
        with open_input_media_container(
            video_path,
            error_message="{error}",
        ) as input_container:
            video_stream = next(
                (stream for stream in input_container.streams if stream.type == "video"),
                None,
            )
            if video_stream is None:
                return False, f"No video stream found in {video_path}"

            input_container.seek(int(max(timestamp_sec, 0.0) * av.time_base), backward=True)
            frame = None
            for decoded_frame in input_container.decode(video_stream):
                current_frame = cast("VideoFrame", decoded_frame)
                if current_frame.time is None:
                    continue
                frame = current_frame
                if current_frame.time >= timestamp_sec:
                    break
            if frame is None:
                return False, f"PyAV did not decode a frame at {timestamp_sec:.3f}s"

            output_path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_image().convert("RGB").save(output_path)
    except (MediaProcessingError, OSError, av.FFmpegError) as error:
        return False, str(error)

    if not output_path.exists():
        return False, f"PyAV did not write {output_path}"
    return True, None


def _normalize_extracted_frame(output_path: Path) -> None:
    with Image.open(output_path) as image:
        normalized_image = ImageOps.exif_transpose(image).convert("RGB")
        normalized_image.save(output_path)
