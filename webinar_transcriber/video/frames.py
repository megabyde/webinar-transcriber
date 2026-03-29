"""Representative frame extraction for detected scenes."""

import subprocess
from collections.abc import Callable
from pathlib import Path

from PIL import Image, ImageOps

from webinar_transcriber.media import MediaProcessingError
from webinar_transcriber.models import Scene, SlideFrame

FRAME_EXTRACT_TIMEOUT_SEC = 300.0


def extract_representative_frames(
    video_path: Path,
    scenes: list[Scene],
    frames_dir: Path,
    *,
    progress_callback: Callable[[], None] | None = None,
) -> list[SlideFrame]:
    """Extract one representative frame near the midpoint of each scene."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    frames: list[SlideFrame] = []

    for index, scene in enumerate(scenes, start=1):
        midpoint_sec = (scene.start_sec + scene.end_sec) / 2
        output_path = frames_dir / f"{scene.id}.png"
        if not _extract_frame(video_path, midpoint_sec, output_path):
            if progress_callback is not None:
                progress_callback()
            continue

        _normalize_extracted_frame(output_path)
        frames.append(
            SlideFrame(
                id=f"frame-{index}",
                scene_id=scene.id,
                image_path=str(output_path),
                timestamp_sec=midpoint_sec,
            )
        )
        if progress_callback is not None:
            progress_callback()

    return frames


def _extract_frame(video_path: Path, timestamp_sec: float, output_path: Path) -> bool:
    try:
        result = subprocess.run(
            _frame_extract_command(video_path, timestamp_sec, output_path),
            capture_output=True,
            check=False,
            text=True,
            timeout=FRAME_EXTRACT_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired as error:
        raise MediaProcessingError(
            f"ffmpeg frame extraction timed out after {FRAME_EXTRACT_TIMEOUT_SEC:g}s."
        ) from error
    return result.returncode == 0 and output_path.exists()


def _frame_extract_command(video_path: Path, timestamp_sec: float, output_path: Path) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-noautorotate",
        "-ss",
        f"{timestamp_sec:.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        str(output_path),
    ]


def _normalize_extracted_frame(output_path: Path) -> None:
    with Image.open(output_path) as image:
        normalized_image = ImageOps.exif_transpose(image).convert("RGB")
        normalized_image.save(output_path)
