"""Representative frame extraction for detected scenes."""

from pathlib import Path

import cv2
import imagehash
from PIL import Image

from webinar_transcriber.models import Scene, SlideFrame


def extract_representative_frames(
    video_path: Path, scenes: list[Scene], frames_dir: Path
) -> list[SlideFrame]:
    """Extract one representative frame near the midpoint of each scene."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    frames: list[SlideFrame] = []

    try:
        for index, scene in enumerate(scenes, start=1):
            midpoint_sec = (scene.start_sec + scene.end_sec) / 2
            capture.set(cv2.CAP_PROP_POS_MSEC, midpoint_sec * 1000)
            success, frame = capture.read()
            if not success:
                continue

            output_path = frames_dir / f"{scene.id}.png"
            cv2.imwrite(str(output_path), frame)

            grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharpness = float(cv2.Laplacian(grayscale, cv2.CV_64F).var())
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            dedupe_hash = str(imagehash.phash(Image.fromarray(rgb_image)))

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
    finally:
        capture.release()

    return frames
