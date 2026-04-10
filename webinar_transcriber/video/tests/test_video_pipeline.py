"""Tests for baseline video helpers."""

import io
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from webinar_transcriber.media import MediaProcessingError
from webinar_transcriber.models import Scene
from webinar_transcriber.video import detect_scenes, extract_representative_frames
from webinar_transcriber.video.frames import (
    _extract_frame,
    _frame_extract_command,
    _normalize_extracted_frame,
)
from webinar_transcriber.video.scenes import (
    TARGET_SAMPLE_HEIGHT,
    TARGET_SAMPLE_WIDTH,
    _detect_scene_start_times,
    _estimate_sample_end_time,
    _iter_sampled_frames,
    estimate_sample_count,
)

FIXTURE_DIR = Path(__file__).parents[2] / "tests" / "fixtures"


class TestDetectScenes:
    def test_detect_scenes_finds_multiple_segments(self) -> None:
        scene_counts: list[int] = []
        scenes = detect_scenes(
            FIXTURE_DIR / "sample-video.mp4",
            min_scene_length_sec=1.0,
            progress_callback=lambda scene_count: scene_counts.append(scene_count),
        )

        assert len(scenes) >= 2
        assert scenes[0].start_sec == 0.0
        assert scene_counts == [1, 2]


class TestFrameExtraction:
    def test_extract_representative_frames_creates_images(self, tmp_path) -> None:
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

    def test_detect_scene_start_times_uses_last_accepted_frame(self) -> None:
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

    def test_detect_scene_start_times_respects_min_scene_length(self) -> None:
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

    def test_detect_scene_start_times_reports_running_scene_counts(self) -> None:
        scene_counts: list[int] = []
        samples = [
            (0.0, np.zeros((2, 2), dtype=np.float32)),
            (1.0, np.full((2, 2), 30.0, dtype=np.float32)),
            (4.0, np.full((2, 2), 30.0, dtype=np.float32)),
        ]

        scene_starts = _detect_scene_start_times(
            samples,
            difference_threshold=10.0,
            min_scene_length_sec=3.0,
            progress_callback=lambda scene_count: scene_counts.append(scene_count),
        )

        assert scene_starts == [0.0, 4.0]
        assert scene_counts == [1, 1, 2]

    def test_normalize_extracted_frame_applies_exif_orientation(self, tmp_path) -> None:
        image_path = tmp_path / "rotated.jpg"
        image = Image.new("RGB", (4, 2), color="white")
        exif = Image.Exif()
        exif[274] = 6
        image.save(image_path, exif=exif)

        _normalize_extracted_frame(image_path)

        with Image.open(image_path) as normalized_image:
            assert normalized_image.size == (2, 4)
            assert normalized_image.getexif().get(274) is None

    def test_frame_extract_command_disables_ffmpeg_autorotate(self, tmp_path) -> None:
        command = _frame_extract_command(
            FIXTURE_DIR / "sample-video.mp4",
            12.345,
            tmp_path / "scene-1.png",
        )

        assert command[:3] == ["ffmpeg", "-y", "-noautorotate"]
        assert "-i" in command

    def test_extract_frame_wraps_timeout_with_media_processing_error(
        self,
        tmp_path,
        monkeypatch,
    ) -> None:
        def fake_run(*_args, **_kwargs):
            raise subprocess.TimeoutExpired(cmd=["ffmpeg"], timeout=12.5)

        monkeypatch.setattr("webinar_transcriber.video.frames.subprocess.run", fake_run)

        with pytest.raises(
            MediaProcessingError,
            match=r"ffmpeg frame extraction timed out after 300s\.",
        ):
            _extract_frame(FIXTURE_DIR / "sample-video.mp4", 1.0, tmp_path / "scene-1.png")

    def test_extract_representative_frames_skips_failed_scene_but_still_reports_progress(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        progress_ticks: list[int] = []
        scenes = [
            Scene(id="scene-1", start_sec=0.0, end_sec=2.0),
            Scene(id="scene-2", start_sec=2.0, end_sec=4.0),
        ]

        def fake_extract_frame(_video_path: Path, _timestamp_sec: float, output_path: Path) -> bool:
            if output_path.stem == "scene-1":
                return False
            output_path.write_bytes(b"frame")
            return True

        monkeypatch.setattr(
            "webinar_transcriber.video.frames._extract_frame",
            fake_extract_frame,
        )
        monkeypatch.setattr(
            "webinar_transcriber.video.frames._normalize_extracted_frame",
            lambda _p: None,
        )

        frames = extract_representative_frames(
            FIXTURE_DIR / "sample-video.mp4",
            scenes,
            tmp_path / "frames",
            progress_callback=lambda: progress_ticks.append(1),
        )

        assert [frame.scene_id for frame in frames] == ["scene-2"]
        assert len(progress_ticks) == 2
        assert "Frame extraction failed for scene-1 at 1.0s" in caplog.text

    def test_extract_frame_returns_false_when_ffmpeg_does_not_write_output(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "webinar_transcriber.video.frames.subprocess.run",
            lambda *_args, **_kwargs: subprocess.CompletedProcess(["ffmpeg"], returncode=0),
        )

        extracted = _extract_frame(FIXTURE_DIR / "sample-video.mp4", 1.0, tmp_path / "scene-1.png")

        assert not extracted


class TestDetectScenesFallback:
    def test_detect_scenes_returns_single_zero_length_scene_when_no_samples(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "webinar_transcriber.video.scenes._iter_sampled_frames",
            lambda _path: iter(()),
        )

        scenes = detect_scenes(FIXTURE_DIR / "sample-video.mp4")

        assert scenes == [Scene(id="scene-1", start_sec=0.0, end_sec=0.0)]


class TestIterSampledFrames:
    def test_iter_sampled_frames_raises_when_ffmpeg_pipes_are_unavailable(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class FakeProcess:
            def __init__(self) -> None:
                self.stdout = None
                self.stderr = io.BytesIO()
                self.killed = False

            def kill(self) -> None:
                self.killed = True

        process = FakeProcess()
        monkeypatch.setattr(
            "webinar_transcriber.video.scenes.subprocess.Popen",
            lambda *_a, **_k: process,
        )

        with pytest.raises(MediaProcessingError, match="Could not open ffmpeg pipes"):
            list(_iter_sampled_frames(FIXTURE_DIR / "sample-video.mp4"))

        assert process.killed

    def test_iter_sampled_frames_raises_on_incomplete_frame_sample(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class FakeProcess:
            def __init__(self) -> None:
                self.stdout = io.BytesIO(b"\x00")
                self.stderr = io.BytesIO()
                self.killed = False

            def kill(self) -> None:
                self.killed = True

            def wait(self) -> int:
                return 0

        process = FakeProcess()
        monkeypatch.setattr(
            "webinar_transcriber.video.scenes.subprocess.Popen",
            lambda *_a, **_k: process,
        )

        with pytest.raises(
            MediaProcessingError, match="Incomplete frame sample returned by ffmpeg"
        ):
            list(_iter_sampled_frames(FIXTURE_DIR / "sample-video.mp4"))

        assert process.killed

    def test_iter_sampled_frames_raises_when_ffmpeg_exits_nonzero(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class FakeProcess:
            def __init__(self) -> None:
                self.stdout = io.BytesIO(b"")
                self.stderr = io.BytesIO(b"ffmpeg failed")

            def wait(self) -> int:
                return 1

        monkeypatch.setattr(
            "webinar_transcriber.video.scenes.subprocess.Popen",
            lambda *_a, **_k: FakeProcess(),
        )

        with pytest.raises(MediaProcessingError, match="ffmpeg failed"):
            list(_iter_sampled_frames(FIXTURE_DIR / "sample-video.mp4"))

    def test_iter_sampled_frames_raises_when_ffmpeg_exits_nonzero_after_yielding_frames(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        frame_size = TARGET_SAMPLE_WIDTH * TARGET_SAMPLE_HEIGHT

        class FakeProcess:
            def __init__(self) -> None:
                self.stdout = io.BytesIO(b"\x00" * frame_size)
                self.stderr = io.BytesIO(b"ffmpeg failed late")

            def wait(self) -> int:
                return 1

        monkeypatch.setattr(
            "webinar_transcriber.video.scenes.subprocess.Popen",
            lambda *_a, **_k: FakeProcess(),
        )

        iterator = iter(_iter_sampled_frames(FIXTURE_DIR / "sample-video.mp4"))
        current_time, frame = next(iterator)

        assert current_time == 0.0
        assert frame.shape == (TARGET_SAMPLE_HEIGHT, TARGET_SAMPLE_WIDTH)

        with pytest.raises(MediaProcessingError, match="ffmpeg failed late"):
            list(iterator)


class TestSceneSamplingHelpers:
    def test_scene_sampling_helpers_cover_zero_duration_and_trailing_sample(self) -> None:
        assert estimate_sample_count(0.0) == 1
        assert _estimate_sample_end_time(0.0) == 0.0
        assert _estimate_sample_end_time(4.0) == 5.0
