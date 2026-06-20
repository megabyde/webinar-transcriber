"""Tests for baseline video helpers."""

from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest
from PIL import Image

from webinar_transcriber.models import Scene
from webinar_transcriber.video import detect_scenes, estimated_scene_sample_count, save_scene_frames

FIXTURE_DIR = Path(__file__).parent / "fixtures"
SAMPLE_VIDEO_PATH = FIXTURE_DIR / "sample-video.mp4"
SAMPLE_VIDEO_DURATION_SEC = 2.0


def _patch_scene_detection(
    monkeypatch: pytest.MonkeyPatch,
    *,
    scan_fps: float,
    min_scene_length_sec: float,
    score_threshold: float | None = None,
) -> None:
    monkeypatch.setattr("webinar_transcriber.video.scenes.SCENE_SCAN_FPS", scan_fps)
    monkeypatch.setattr(
        "webinar_transcriber.video.scenes.MIN_SCENE_LENGTH_SEC", min_scene_length_sec
    )
    if score_threshold is not None:
        monkeypatch.setattr(
            "webinar_transcriber.video.scenes.SCENE_SCORE_THRESHOLD", score_threshold
        )


def _write_synthetic_video(output_path: Path, frame_colors: list[int], *, fps: int = 10) -> None:
    with av.open(str(output_path), "w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width = 64
        stream.height = 64
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, fps)

        for index, color in enumerate(frame_colors):
            pixels = np.full((64, 64, 3), color, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = index
            frame.time_base = Fraction(1, fps)
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)  # pragma: no cover - encoder-dependent delayed packets


class TestDetectScenes:
    @pytest.mark.slow
    def test_detect_scenes_finds_multiple_segments(self, monkeypatch: pytest.MonkeyPatch) -> None:
        progress_updates: list[tuple[float, int]] = []

        def on_progress(sample_count: float, scene_count: int) -> None:
            progress_updates.append((sample_count, scene_count))

        _patch_scene_detection(
            monkeypatch, scan_fps=2.0, min_scene_length_sec=1.0, score_threshold=0.006
        )
        scenes = detect_scenes(
            SAMPLE_VIDEO_PATH, SAMPLE_VIDEO_DURATION_SEC, progress_callback=on_progress
        )

        assert len(scenes) >= 2
        assert scenes[0].start_sec == 0.0
        # Detection returns time bounds only; frames are captured separately.
        assert all(scene.image_path is None for scene in scenes)
        assert progress_updates
        assert progress_updates[-1][1] == len(scenes)

    @pytest.mark.slow
    def test_detect_scenes_respects_default_min_scene_length(self) -> None:
        scenes = detect_scenes(SAMPLE_VIDEO_PATH, SAMPLE_VIDEO_DURATION_SEC)

        assert scenes == [Scene(id="scene-1", start_sec=0.0, end_sec=2.0)]

    @pytest.mark.slow
    def test_detect_scenes_keeps_blank_video_as_one_scene(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        video_path = tmp_path / "blank.mp4"
        _write_synthetic_video(video_path, [0] * 10)

        _patch_scene_detection(monkeypatch, scan_fps=10.0, min_scene_length_sec=0.1)
        scenes = detect_scenes(video_path, 1.0)

        assert scenes == [Scene(id="scene-1", start_sec=0.0, end_sec=1.0)]

    @pytest.mark.slow
    def test_detect_scenes_finds_synthetic_color_scenes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        video_path = tmp_path / "alternating-color.mp4"
        progress_updates: list[tuple[float, int]] = []
        _write_synthetic_video(video_path, [0, 255] * 5)

        _patch_scene_detection(monkeypatch, scan_fps=10.0, min_scene_length_sec=0.1)
        scenes = detect_scenes(
            video_path,
            1.0,
            progress_callback=lambda sample_count, scene_count: progress_updates.append((
                sample_count,
                scene_count,
            )),
        )

        assert scenes == [
            Scene(id="scene-1", start_sec=0.0, end_sec=0.1),
            Scene(id="scene-2", start_sec=0.1, end_sec=1.0),
        ]
        assert progress_updates[-1] == (10, 2)

    def test_detect_scenes_drops_zero_duration_scene(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "webinar_transcriber.video.scenes._detect_scene_starts",
            lambda *_args, **_kwargs: [0.0],
        )

        assert detect_scenes(SAMPLE_VIDEO_PATH, 0.0) == []

    def test_detect_scenes_clamps_scene_bounds_to_duration(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "webinar_transcriber.video.scenes._detect_scene_starts",
            lambda *_args, **_kwargs: [0.0, 1.0, 3.0],
        )

        scenes = detect_scenes(SAMPLE_VIDEO_PATH, 2.0)

        assert scenes == [
            Scene(id="scene-1", start_sec=0.0, end_sec=1.0),
            Scene(id="scene-2", start_sec=1.0, end_sec=2.0),
        ]

    @pytest.mark.slow
    def test_detect_scenes_uses_provided_duration_for_scene_bounds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        video_path = tmp_path / "blank.mp4"
        _write_synthetic_video(video_path, [0] * 10)

        _patch_scene_detection(monkeypatch, scan_fps=10.0, min_scene_length_sec=0.1)
        scenes = detect_scenes(video_path, 2.0)

        assert scenes == [Scene(id="scene-1", start_sec=0.0, end_sec=2.0)]


class TestSaveSceneFrames:
    @pytest.mark.slow
    def test_save_scene_frames_writes_a_mid_scene_frame_per_scene(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        video_path = tmp_path / "alternating-color.mp4"
        _write_synthetic_video(video_path, [0, 255] * 5)
        _patch_scene_detection(monkeypatch, scan_fps=10.0, min_scene_length_sec=0.1)
        scenes = detect_scenes(video_path, 1.0)
        frames_dir = tmp_path / "frames"
        saved_progress: list[int] = []

        captured = save_scene_frames(
            video_path, scenes, frames_dir, progress_callback=saved_progress.append
        )

        assert [scene.image_path for scene in captured] == [
            f"frames/{scene.id}.png" for scene in scenes
        ]
        for scene in captured:
            with Image.open(tmp_path / "frames" / f"{scene.id}.png") as image:
                assert image.mode == "RGB"
                assert image.width > 0
        assert saved_progress == [1, 2]

    @pytest.mark.slow
    def test_save_scene_frames_falls_back_to_last_frame_past_video_end(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Frames cover 0.0-0.9s, but the probed duration stretches the lone scene to 2.0s, so its
        # 1.0s midpoint sits past the last frame and the capture falls back to the final frame.
        video_path = tmp_path / "blank.mp4"
        _write_synthetic_video(video_path, [0] * 10)
        _patch_scene_detection(monkeypatch, scan_fps=10.0, min_scene_length_sec=0.1)
        scenes = detect_scenes(video_path, 2.0)

        captured = save_scene_frames(video_path, scenes, tmp_path / "frames")

        assert [scene.image_path for scene in captured] == ["frames/scene-1.png"]
        assert (tmp_path / "frames" / "scene-1.png").exists()

    def test_save_scene_frames_with_no_scenes_writes_nothing(self, tmp_path: Path) -> None:
        assert save_scene_frames(SAMPLE_VIDEO_PATH, [], tmp_path / "frames") == []
        assert not (tmp_path / "frames").exists()


class TestEstimatedSceneSampleCount:
    def test_estimated_scene_sample_count_rounds_up(self) -> None:
        assert estimated_scene_sample_count(9.1) == 5

    def test_estimated_scene_sample_count_keeps_zero_duration_visible(self) -> None:
        assert estimated_scene_sample_count(0.0) == 1
