"""Tests for baseline video helpers."""

import io
import subprocess
from pathlib import Path
from typing import IO, cast

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
    SCENE_SAMPLE_TIMEOUT_SEC,
    TARGET_SAMPLE_HEIGHT,
    TARGET_SAMPLE_WIDTH,
    _detect_scene_start_times,
    _estimate_sample_end_time,
    _iter_sampled_frames,
    _read_stdout_chunk,
    _remaining_scene_sample_timeout,
    estimate_sample_count,
)

FIXTURE_DIR = Path(__file__).parents[2] / "tests" / "fixtures"
SAMPLE_VIDEO_PATH = FIXTURE_DIR / "sample-video.mp4"


def _frame(fill: int, *, invert_quadrants: bool = False) -> np.ndarray:
    frame = np.full((32, 32), fill, dtype=np.uint8)
    if invert_quadrants:
        frame[:16, :16] = 255 - fill
        frame[16:, 16:] = 255 - fill
    return frame


def _scene(index: int, start_sec: float, end_sec: float) -> Scene:
    return Scene(id=f"scene-{index}", start_sec=start_sec, end_sec=end_sec)


def _install_fake_phashes(monkeypatch: pytest.MonkeyPatch, distances: list[int]) -> None:
    class FakeHash:
        def __init__(self, key: int) -> None:
            self.key = key

        def __sub__(self, other: object) -> int:
            assert isinstance(other, FakeHash)
            return abs(self.key - other.key)

    keys = [0]
    for distance in distances:
        keys.append(keys[-1] + distance)

    fake_hashes = iter(FakeHash(key) for key in keys)
    monkeypatch.setattr(
        "webinar_transcriber.video.scenes.imagehash.phash",
        lambda *_args, **_kwargs: next(fake_hashes),
    )


class TestDetectScenes:
    @pytest.mark.slow
    def test_detect_scenes_finds_multiple_segments(self) -> None:
        progress_updates: list[tuple[int, int]] = []

        def on_progress(sample_count: int, scene_count: int) -> None:
            progress_updates.append((sample_count, scene_count))

        scenes = detect_scenes(
            SAMPLE_VIDEO_PATH,
            min_scene_length_sec=1.0,
            progress_callback=on_progress,
        )

        assert len(scenes) >= 2
        assert scenes[0].start_sec == 0.0
        assert progress_updates
        assert progress_updates[-1][0] == estimate_sample_count(scenes[-1].end_sec)
        assert progress_updates[-1][1] == len(scenes)


class TestFrameExtraction:
    @pytest.mark.slow
    def test_extract_frame_writes_real_image(self, tmp_path: Path) -> None:
        output_path = tmp_path / "scene-1.png"

        extracted, failure_detail = _extract_frame(SAMPLE_VIDEO_PATH, 1.0, output_path)

        assert extracted
        assert failure_detail is None
        assert output_path.exists()
        with Image.open(output_path) as image:
            assert image.mode == "RGB"
            assert image.width > 0
            assert image.height > 0

    @pytest.mark.slow
    def test_extract_representative_frames_creates_real_images(self, tmp_path: Path) -> None:
        scenes = [_scene(1, 0.0, 1.0), _scene(2, 1.0, 2.0)]

        frames = extract_representative_frames(
            SAMPLE_VIDEO_PATH,
            scenes,
            tmp_path / "frames",
        )

        assert [frame.scene_id for frame in frames] == ["scene-1", "scene-2"]
        assert [frame.timestamp_sec for frame in frames] == [0.5, 1.5]
        assert all(Path(frame.image_path).exists() for frame in frames)
        for frame in frames:
            with Image.open(frame.image_path) as image:
                assert image.mode == "RGB"
                assert image.width > 0
                assert image.height > 0

    def test_extract_representative_frames_creates_images(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        scenes = [_scene(1, 0.0, 2.0), _scene(2, 2.0, 4.0)]
        progress_ticks: list[int] = []

        def fake_extract_frame(
            _video_path: Path, _timestamp_sec: float, output_path: Path
        ) -> tuple[bool, str | None]:
            output_path.write_bytes(b"frame")
            return True, None

        monkeypatch.setattr("webinar_transcriber.video.frames._extract_frame", fake_extract_frame)
        monkeypatch.setattr(
            "webinar_transcriber.video.frames._normalize_extracted_frame", lambda _p: None
        )

        frames = extract_representative_frames(
            SAMPLE_VIDEO_PATH,
            scenes,
            tmp_path / "frames",
            progress_callback=lambda: progress_ticks.append(1),
        )

        assert len(frames) >= 2
        assert all(Path(frame.image_path).exists() for frame in frames)
        assert len(progress_ticks) == len(scenes)

    def test_detect_scene_start_times_uses_last_accepted_frame(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        samples = [
            (0.0, _frame(0)),
            (1.0, _frame(0)),
            (2.0, _frame(32, invert_quadrants=True)),
            (3.0, _frame(32, invert_quadrants=True)),
        ]

        _install_fake_phashes(monkeypatch, [0, 10, 0])

        scene_starts = _detect_scene_start_times(
            samples, difference_threshold=1, min_scene_length_sec=0.0
        )

        assert scene_starts == [0.0, 2.0]

    def test_detect_scene_start_times_respects_min_scene_length(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        samples = [
            (0.0, _frame(0)),
            (1.0, _frame(32, invert_quadrants=True)),
            (4.0, _frame(32, invert_quadrants=True)),
        ]

        _install_fake_phashes(monkeypatch, [10, 0])

        scene_starts = _detect_scene_start_times(
            samples, difference_threshold=1, min_scene_length_sec=3.0
        )

        assert scene_starts == [0.0, 4.0]

    def test_detect_scene_start_times_reports_sample_and_scene_progress(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        progress_updates: list[tuple[int, int]] = []
        samples = [
            (0.0, _frame(0)),
            (1.0, _frame(32, invert_quadrants=True)),
            (4.0, _frame(32, invert_quadrants=True)),
        ]

        _install_fake_phashes(monkeypatch, [10, 0])

        def on_progress(sample_count: int, scene_count: int) -> None:
            progress_updates.append((sample_count, scene_count))

        scene_starts = _detect_scene_start_times(
            samples,
            difference_threshold=1,
            min_scene_length_sec=3.0,
            progress_callback=on_progress,
        )

        assert scene_starts == [0.0, 4.0]
        assert progress_updates == [(1, 1), (2, 1), (3, 2)]

    def test_detect_scene_start_times_default_threshold_ignores_minor_variants(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        samples = [
            (0.0, _frame(0)),
            (4.0, _frame(32, invert_quadrants=True)),
        ]

        _install_fake_phashes(monkeypatch, [30])

        scene_starts = _detect_scene_start_times(samples, min_scene_length_sec=0.0)

        assert scene_starts == [0.0]

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
        command = _frame_extract_command(SAMPLE_VIDEO_PATH, 12.345, tmp_path / "scene-1.png")

        assert command[:3] == ["ffmpeg", "-y", "-noautorotate"]
        assert "-i" in command

    def test_extract_frame_wraps_timeout_with_media_processing_error(
        self, tmp_path, monkeypatch
    ) -> None:
        def fake_run(*_args, **_kwargs):
            raise subprocess.TimeoutExpired(cmd=["ffmpeg"], timeout=12.5)

        monkeypatch.setattr("webinar_transcriber.video.frames.subprocess.run", fake_run)

        with pytest.raises(
            MediaProcessingError, match=r"ffmpeg frame extraction timed out after 300s\."
        ):
            _extract_frame(SAMPLE_VIDEO_PATH, 1.0, tmp_path / "scene-1.png")

    def test_extract_representative_frames_skips_failed_scene_but_still_reports_progress(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        progress_ticks: list[int] = []
        warnings: list[str] = []
        scenes = [_scene(1, 0.0, 2.0), _scene(2, 2.0, 4.0)]

        def fake_extract_frame(
            _video_path: Path, _timestamp_sec: float, output_path: Path
        ) -> tuple[bool, str | None]:
            if output_path.stem == "scene-1":
                return False, f"ffmpeg did not write {output_path}"
            output_path.write_bytes(b"frame")
            return True, None

        monkeypatch.setattr("webinar_transcriber.video.frames._extract_frame", fake_extract_frame)
        monkeypatch.setattr(
            "webinar_transcriber.video.frames._normalize_extracted_frame", lambda _p: None
        )

        frames = extract_representative_frames(
            SAMPLE_VIDEO_PATH,
            scenes,
            tmp_path / "frames",
            progress_callback=lambda: progress_ticks.append(1),
            warning_callback=warnings.append,
        )

        assert [frame.scene_id for frame in frames] == ["scene-2"]
        assert len(progress_ticks) == 2
        assert warnings == [
            f"Frame extraction failed for scene-1 at 0.5s: "
            f"ffmpeg did not write {tmp_path / 'frames' / 'scene-1.png'}"
        ]

    def test_extract_representative_frames_uses_first_stable_frame(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured_timestamps: list[float] = []

        def fake_extract_frame(
            _video_path: Path, timestamp_sec: float, output_path: Path
        ) -> tuple[bool, str | None]:
            captured_timestamps.append(timestamp_sec)
            output_path.write_bytes(b"frame")
            return True, None

        monkeypatch.setattr("webinar_transcriber.video.frames._extract_frame", fake_extract_frame)
        monkeypatch.setattr(
            "webinar_transcriber.video.frames._normalize_extracted_frame", lambda _p: None
        )

        frames = extract_representative_frames(
            SAMPLE_VIDEO_PATH,
            [_scene(1, 2.0, 5.0)],
            tmp_path / "frames",
        )

        assert captured_timestamps == [2.5]
        assert frames[0].timestamp_sec == 2.5

    def test_extract_frame_returns_false_when_ffmpeg_does_not_write_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "webinar_transcriber.video.frames.subprocess.run",
            lambda *_args, **_kwargs: subprocess.CompletedProcess(["ffmpeg"], returncode=0),
        )

        extracted, failure_detail = _extract_frame(SAMPLE_VIDEO_PATH, 1.0, tmp_path / "scene-1.png")

        assert not extracted
        assert failure_detail == f"ffmpeg did not write {tmp_path / 'scene-1.png'}"

    def test_extract_frame_returns_true_when_ffmpeg_writes_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        output_path = tmp_path / "scene-1.png"

        def fake_run(*_args, **_kwargs) -> subprocess.CompletedProcess[str]:
            output_path.write_bytes(b"frame")
            return subprocess.CompletedProcess(["ffmpeg"], returncode=0)

        monkeypatch.setattr("webinar_transcriber.video.frames.subprocess.run", fake_run)

        extracted, failure_detail = _extract_frame(SAMPLE_VIDEO_PATH, 1.0, output_path)

        assert extracted
        assert failure_detail is None

    def test_extract_frame_returns_stderr_when_ffmpeg_exits_nonzero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "webinar_transcriber.video.frames.subprocess.run",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                subprocess.CalledProcessError(1, ["ffmpeg"], stderr="decode failed")
            ),
        )

        extracted, failure_detail = _extract_frame(SAMPLE_VIDEO_PATH, 1.0, tmp_path / "scene-1.png")

        assert not extracted
        assert failure_detail == "decode failed"


class TestDetectScenesFallback:
    def test_detect_scenes_returns_single_zero_length_scene_when_no_samples(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "webinar_transcriber.video.scenes._iter_sampled_frames", lambda _path: iter(())
        )

        scenes = detect_scenes(SAMPLE_VIDEO_PATH)

        assert scenes == [Scene(id="scene-1", start_sec=0.0, end_sec=0.0)]


class TestIterSampledFrames:
    def test_iter_sampled_frames_wraps_ffmpeg_startup_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "webinar_transcriber.video.scenes.subprocess.Popen",
            lambda *_a, **_k: (_ for _ in ()).throw(FileNotFoundError("ffmpeg missing")),
        )

        with pytest.raises(
            MediaProcessingError, match="Could not start ffmpeg for scene detection: ffmpeg missing"
        ):
            list(_iter_sampled_frames(SAMPLE_VIDEO_PATH))

    def test_iter_sampled_frames_raises_when_ffmpeg_pipes_are_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakeProcess:
            def __init__(self) -> None:
                self.stdout = None
                self.stderr = io.BytesIO()

            def kill(self) -> None:
                return None

        process = FakeProcess()
        monkeypatch.setattr(
            "webinar_transcriber.video.scenes.subprocess.Popen", lambda *_a, **_k: process
        )

        with pytest.raises(MediaProcessingError, match="Could not open ffmpeg pipes"):
            list(_iter_sampled_frames(SAMPLE_VIDEO_PATH))

    def test_iter_sampled_frames_raises_on_incomplete_frame_sample(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakeProcess:
            def __init__(self) -> None:
                self.stdout = io.BytesIO(b"\x00")
                self.stderr = io.BytesIO()
                self.returncode = 0

            def wait(self, timeout: float | None = None) -> int:
                del timeout
                return self.returncode

        process = FakeProcess()
        monkeypatch.setattr(
            "webinar_transcriber.video.scenes.subprocess.Popen", lambda *_a, **_k: process
        )

        with pytest.raises(
            MediaProcessingError, match="Incomplete frame sample returned by ffmpeg"
        ):
            list(_iter_sampled_frames(SAMPLE_VIDEO_PATH))

    def test_iter_sampled_frames_raises_when_ffmpeg_exits_nonzero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakeProcess:
            def __init__(self) -> None:
                self.stdout = io.BytesIO()
                self.stderr = io.BytesIO(b"ffmpeg failed")
                self.returncode = 1

            def wait(self, timeout: float | None = None) -> int:
                del timeout
                return self.returncode

        monkeypatch.setattr(
            "webinar_transcriber.video.scenes.subprocess.Popen", lambda *_a, **_k: FakeProcess()
        )

        with pytest.raises(MediaProcessingError) as exc_info:
            list(_iter_sampled_frames(SAMPLE_VIDEO_PATH))

        assert str(exc_info.value) == "ffmpeg failed"

    def test_iter_sampled_frames_raises_when_ffmpeg_exits_nonzero_even_with_frame_output(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        frame_size = TARGET_SAMPLE_WIDTH * TARGET_SAMPLE_HEIGHT

        class FakeProcess:
            def __init__(self) -> None:
                self.stdout = io.BytesIO(b"\x00" * frame_size)
                self.stderr = io.BytesIO(b"ffmpeg failed late")
                self.returncode = 1

            def wait(self, timeout: float | None = None) -> int:
                del timeout
                return self.returncode

        monkeypatch.setattr(
            "webinar_transcriber.video.scenes.subprocess.Popen", lambda *_a, **_k: FakeProcess()
        )

        with pytest.raises(MediaProcessingError) as exc_info:
            list(_iter_sampled_frames(SAMPLE_VIDEO_PATH))

        assert str(exc_info.value) == "ffmpeg failed late"

    def test_iter_sampled_frames_prefers_ffmpeg_failure_when_ffmpeg_exits_nonzero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakeProcess:
            def __init__(self) -> None:
                self.stdout = io.BytesIO(b"\x00")
                self.stderr = io.BytesIO(b"ffmpeg failed")
                self.returncode = 1

            def wait(self, timeout: float | None = None) -> int:
                del timeout
                return self.returncode

        monkeypatch.setattr(
            "webinar_transcriber.video.scenes.subprocess.Popen", lambda *_a, **_k: FakeProcess()
        )

        with pytest.raises(MediaProcessingError, match="ffmpeg failed"):
            list(_iter_sampled_frames(SAMPLE_VIDEO_PATH))

    def test_iter_sampled_frames_kills_process_when_ffmpeg_times_out(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakeProcess:
            def __init__(self) -> None:
                self.killed = False

            stdout = io.BytesIO()
            stderr = io.BytesIO()
            returncode = 0

            def kill(self) -> None:
                self.killed = True

            def wait(self, timeout: float | None = None) -> int:
                del timeout
                return self.returncode

        process = FakeProcess()
        monkeypatch.setattr(
            "webinar_transcriber.video.scenes.subprocess.Popen", lambda *_a, **_k: process
        )
        monkeypatch.setattr(
            "webinar_transcriber.video.scenes._read_stdout_chunk",
            lambda *_a, **_k: (_ for _ in ()).throw(
                subprocess.TimeoutExpired(cmd=["ffmpeg"], timeout=SCENE_SAMPLE_TIMEOUT_SEC)
            ),
        )

        with pytest.raises(
            MediaProcessingError, match=r"ffmpeg scene sampling timed out after 300s\."
        ):
            list(_iter_sampled_frames(FIXTURE_DIR / "sample-video.mp4"))

        assert process.killed

    def test_iter_sampled_frames_yields_frames_before_waiting_for_completion(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        frame_size = TARGET_SAMPLE_WIDTH * TARGET_SAMPLE_HEIGHT

        class FakeProcess:
            def __init__(self) -> None:
                self.stdout = io.BytesIO(b"\x00" * frame_size)
                self.stderr = io.BytesIO()
                self.returncode = 0
                self.wait_called = False

            def wait(self, timeout: float | None = None) -> int:
                del timeout
                self.wait_called = True
                return self.returncode

        process = FakeProcess()
        monkeypatch.setattr(
            "webinar_transcriber.video.scenes.subprocess.Popen", lambda *_a, **_k: process
        )

        frames = iter(_iter_sampled_frames(SAMPLE_VIDEO_PATH))
        sample_time, frame = next(frames)

        assert sample_time == 0.0
        assert frame.shape == (TARGET_SAMPLE_HEIGHT, TARGET_SAMPLE_WIDTH)
        assert not process.wait_called
        with pytest.raises(StopIteration):
            next(frames)
        assert process.wait_called


class TestSceneSamplingHelpers:
    def test_scene_sampling_helpers_cover_zero_duration_and_trailing_sample(self) -> None:
        assert estimate_sample_count(0.0) == 1
        assert _estimate_sample_end_time(0.0) == 0.0
        assert _estimate_sample_end_time(4.0) == 5.0

    def test_remaining_scene_sample_timeout_raises_once_budget_is_exhausted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("webinar_transcriber.video.scenes.perf_counter", lambda: 10.0)

        with pytest.raises(subprocess.TimeoutExpired):
            _remaining_scene_sample_timeout(10.0 - SCENE_SAMPLE_TIMEOUT_SEC - 1.0)

    def test_read_stdout_chunk_raises_when_pipe_is_not_ready(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakeStream:
            def fileno(self) -> int:
                return 123

            def read(self, _size: int) -> bytes:
                raise AssertionError("read should not run when select reports no readiness")

        monkeypatch.setattr(
            "webinar_transcriber.video.scenes.select.select",
            lambda *_args, **_kwargs: ([], [], []),
        )

        with pytest.raises(subprocess.TimeoutExpired):
            _read_stdout_chunk(cast("IO[bytes]", FakeStream()), size=1, timeout_sec=1.0)

    def test_read_stdout_chunk_reads_when_pipe_is_ready(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakeStream:
            def fileno(self) -> int:
                return 123

            def read(self, size: int) -> bytes:
                assert size == 2
                return b"ok"

        monkeypatch.setattr(
            "webinar_transcriber.video.scenes.select.select",
            lambda *_args, **_kwargs: ([123], [], []),
        )

        assert _read_stdout_chunk(cast("IO[bytes]", FakeStream()), size=2, timeout_sec=1.0) == b"ok"
