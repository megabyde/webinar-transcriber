"""Tests for baseline video helpers."""

from pathlib import Path

from webinar_transcriber.video import detect_scenes, extract_representative_frames

FIXTURE_DIR = Path(__file__).parents[2] / "tests" / "fixtures"


def test_detect_scenes_finds_multiple_segments() -> None:
    scenes = detect_scenes(FIXTURE_DIR / "sample-video.mp4")

    assert len(scenes) >= 2
    assert scenes[0].start_sec == 0.0


def test_extract_representative_frames_creates_images(tmp_path) -> None:
    video_path = FIXTURE_DIR / "sample-video.mp4"
    scenes = detect_scenes(video_path)

    frames = extract_representative_frames(video_path, scenes, tmp_path / "frames")

    assert len(frames) >= 2
    assert all(Path(frame.image_path).exists() for frame in frames)
