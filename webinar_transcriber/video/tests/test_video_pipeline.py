"""Tests for baseline video helpers."""

from pathlib import Path

import numpy as np
from PIL import Image

from webinar_transcriber.video import detect_scenes, extract_representative_frames
from webinar_transcriber.video.frames import _normalize_extracted_frame
from webinar_transcriber.video.scenes import _detect_scene_start_times

FIXTURE_DIR = Path(__file__).parents[2] / "tests" / "fixtures"


def test_detect_scenes_finds_multiple_segments() -> None:
    progress_ticks: list[int] = []
    scenes = detect_scenes(
        FIXTURE_DIR / "sample-video.mp4",
        min_scene_length_sec=1.0,
        progress_callback=lambda: progress_ticks.append(1),
    )

    assert len(scenes) >= 2
    assert scenes[0].start_sec == 0.0
    assert len(progress_ticks) == 2


def test_extract_representative_frames_creates_images(tmp_path) -> None:
    video_path = FIXTURE_DIR / "sample-video.mp4"
    scenes = detect_scenes(video_path, min_scene_length_sec=1.0)
    progress_ticks: list[int] = []

    frames = extract_representative_frames(
        video_path,
        scenes,
        tmp_path / "frames",
        progress_callback=lambda: progress_ticks.append(1),
    )

    assert len(frames) >= 2
    assert all(Path(frame.image_path).exists() for frame in frames)
    assert len(progress_ticks) == len(scenes)


def test_detect_scene_start_times_uses_last_accepted_frame() -> None:
    samples = [
        (0.0, np.zeros((2, 2), dtype=np.float32)),
        (1.0, np.full((2, 2), 6.0, dtype=np.float32)),
        (2.0, np.full((2, 2), 12.0, dtype=np.float32)),
        (3.0, np.full((2, 2), 18.0, dtype=np.float32)),
    ]

    scene_starts = _detect_scene_start_times(
        samples,
        difference_threshold=10.0,
        min_scene_length_sec=0.0,
    )

    assert scene_starts == [0.0, 2.0]


def test_detect_scene_start_times_respects_min_scene_length() -> None:
    samples = [
        (0.0, np.zeros((2, 2), dtype=np.float32)),
        (1.0, np.full((2, 2), 30.0, dtype=np.float32)),
        (4.0, np.full((2, 2), 30.0, dtype=np.float32)),
    ]

    scene_starts = _detect_scene_start_times(
        samples,
        difference_threshold=10.0,
        min_scene_length_sec=3.0,
    )

    assert scene_starts == [0.0, 4.0]


def test_normalize_extracted_frame_applies_exif_orientation(tmp_path) -> None:
    image_path = tmp_path / "rotated.jpg"
    image = Image.new("RGB", (4, 2), color="white")
    exif = Image.Exif()
    exif[274] = 6
    image.save(image_path, exif=exif)

    _normalize_extracted_frame(image_path)

    with Image.open(image_path) as normalized_image:
        assert normalized_image.size == (2, 4)
        assert normalized_image.getexif().get(274) is None
