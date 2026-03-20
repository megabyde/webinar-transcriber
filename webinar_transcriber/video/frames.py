"""Representative frame extraction for detected scenes."""

import subprocess
from collections.abc import Callable
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image, ImageOps

from webinar_transcriber.models import Scene, SlideFrame


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
        image = Image.open(output_path).convert("RGB")
        grayscale = image.convert("L")
        sharpness = _sharpness_score(np.asarray(grayscale, dtype=np.float32))
        dedupe_hash = str(imagehash.phash(image))

        frames.append(
            SlideFrame(
                id=f"frame-{index}",
                scene_id=scene.id,
                image_path=str(output_path),
                timestamp_sec=midpoint_sec,
                sharpness_score=sharpness,
                dedupe_hash=dedupe_hash,
            )
        )
        if progress_callback is not None:
            progress_callback()

    return frames


def _extract_frame(video_path: Path, timestamp_sec: float, output_path: Path) -> bool:
    result = subprocess.run(
        _frame_extract_command(video_path, timestamp_sec, output_path),
        capture_output=True,
        check=False,
        text=True,
    )
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


def _sharpness_score(grayscale_pixels: np.ndarray) -> float:
    center = grayscale_pixels[1:-1, 1:-1]
    if center.size == 0:
        return 0.0

    laplacian = (
        (-4 * center)
        + grayscale_pixels[:-2, 1:-1]
        + grayscale_pixels[2:, 1:-1]
        + grayscale_pixels[1:-1, :-2]
        + grayscale_pixels[1:-1, 2:]
    )
    return float(laplacian.var())
