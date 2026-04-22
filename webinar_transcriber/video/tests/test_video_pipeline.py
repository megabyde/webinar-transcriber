"""Tests for baseline video helpers."""

from pathlib import Path
from typing import cast

import numpy as np
import pytest
from PIL import Image

from webinar_transcriber.media import MediaProcessingError
from webinar_transcriber.models import Scene
from webinar_transcriber.tests.conftest import FakeContextContainer
from webinar_transcriber.video import detect_scenes, extract_representative_frames
from webinar_transcriber.video.frames import _extract_frame, _normalize_extracted_frame
from webinar_transcriber.video.scenes import (
    _detect_scene_start_times,
    _estimate_sample_end_time,
    _iter_sampled_frames,
    estimate_sample_count,
)

FIXTURE_DIR = Path(__file__).parents[2] / "tests" / "fixtures"
SAMPLE_VIDEO_PATH = FIXTURE_DIR / "sample-video.mp4"


def _scene(index: int, start_sec: float, end_sec: float) -> Scene:
    return Scene(id=f"scene-{index}", start_sec=start_sec, end_sec=end_sec)


class _FakeHash:
    def __init__(self, key: int) -> None:
        self.key = key

    def __sub__(self, other: object) -> int:
        assert isinstance(other, _FakeHash)
        return abs(self.key - other.key)


def _install_fake_phashes(monkeypatch: pytest.MonkeyPatch, distances: list[int]) -> None:
    keys = [0]
    for distance in distances:
        keys.append(keys[-1] + distance)

    fake_hashes = iter(_FakeHash(key) for key in keys)
    monkeypatch.setattr(
        "webinar_transcriber.video.scenes.imagehash.phash",
        lambda *_args, **_kwargs: next(fake_hashes),
    )


def _frame(fill: int, *, invert_quadrants: bool = False) -> np.ndarray:
    frame = np.full((32, 32), fill, dtype=np.uint8)
    if invert_quadrants:
        frame[:16, :16] = 255 - fill
        frame[16:, 16:] = 255 - fill
    return frame


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

    def test_extract_frame_returns_false_when_open_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "webinar_transcriber.video.frames.av.open",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("bad open")),
        )

        extracted, failure_detail = _extract_frame(SAMPLE_VIDEO_PATH, 1.0, tmp_path / "scene-1.png")

        assert not extracted
        assert failure_detail == "bad open"

    def test_extract_frame_returns_false_when_no_video_stream(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakeContainer(FakeContextContainer):
            def __init__(self) -> None:
                self.streams = [cast("object", type("AudioStream", (), {"type": "audio"})())]

        monkeypatch.setattr(
            "webinar_transcriber.video.frames.av.open", lambda *_args, **_kwargs: FakeContainer()
        )

        extracted, failure_detail = _extract_frame(SAMPLE_VIDEO_PATH, 1.0, tmp_path / "scene-1.png")

        assert not extracted
        assert failure_detail == f"No video stream found in {SAMPLE_VIDEO_PATH}"

    def test_extract_frame_skips_frames_without_time(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        output_path = tmp_path / "scene-1.png"

        class FakeImage:
            def convert(self, _mode: str) -> "FakeImage":
                return self

            def save(self, path: Path) -> None:
                path.write_bytes(b"frame")

        class FakeFrame:
            def __init__(self, time: float | None) -> None:
                self.time = time

            def to_image(self) -> FakeImage:
                return FakeImage()

        class FakeVideoStream:
            type = "video"
            index = 0

        class FakeContainer(FakeContextContainer):
            def __init__(self) -> None:
                self.streams = [FakeVideoStream()]

            def seek(self, *_args, **_kwargs) -> None:
                return None

            def decode(self, *_args, **_kwargs):
                return iter([FakeFrame(None), FakeFrame(1.0)])

        monkeypatch.setattr(
            "webinar_transcriber.video.frames.av.open", lambda *_args, **_kwargs: FakeContainer()
        )

        extracted, failure_detail = _extract_frame(SAMPLE_VIDEO_PATH, 1.0, output_path)

        assert extracted
        assert failure_detail is None
        assert output_path.exists()

    def test_extract_frame_returns_false_when_no_frame_is_decoded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakeVideoStream:
            type = "video"
            index = 0

        class FakeContainer(FakeContextContainer):
            def __init__(self) -> None:
                self.streams = [FakeVideoStream()]

            def seek(self, *_args, **_kwargs) -> None:
                return None

            def decode(self, *_args, **_kwargs):
                return iter([cast("object", type("Frame", (), {"time": None})())])

        monkeypatch.setattr(
            "webinar_transcriber.video.frames.av.open", lambda *_args, **_kwargs: FakeContainer()
        )

        extracted, failure_detail = _extract_frame(SAMPLE_VIDEO_PATH, 1.0, tmp_path / "scene-1.png")

        assert not extracted
        assert failure_detail == "PyAV did not decode a frame at 1.000s"

    def test_extract_frame_returns_false_when_save_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakeImage:
            def convert(self, _mode: str) -> "FakeImage":
                return self

            def save(self, _path: Path) -> None:
                raise OSError("save failed")

        class FakeFrame:
            time = 1.0

            def to_image(self) -> FakeImage:
                return FakeImage()

        class FakeVideoStream:
            type = "video"
            index = 0

        class FakeContainer(FakeContextContainer):
            def __init__(self) -> None:
                self.streams = [FakeVideoStream()]

            def seek(self, *_args, **_kwargs) -> None:
                return None

            def decode(self, *_args, **_kwargs):
                return iter([FakeFrame()])

        monkeypatch.setattr(
            "webinar_transcriber.video.frames.av.open", lambda *_args, **_kwargs: FakeContainer()
        )

        extracted, failure_detail = _extract_frame(SAMPLE_VIDEO_PATH, 1.0, tmp_path / "scene-1.png")

        assert not extracted
        assert failure_detail == "save failed"

    def test_extract_frame_returns_false_when_output_is_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        output_path = tmp_path / "scene-1.png"

        class FakeImage:
            def convert(self, _mode: str) -> "FakeImage":
                return self

            def save(self, _path: Path) -> None:
                return None

        class FakeFrame:
            time = 1.0

            def to_image(self) -> FakeImage:
                return FakeImage()

        class FakeVideoStream:
            type = "video"
            index = 0

        class FakeContainer(FakeContextContainer):
            def __init__(self) -> None:
                self.streams = [FakeVideoStream()]

            def seek(self, *_args, **_kwargs) -> None:
                return None

            def decode(self, *_args, **_kwargs):
                return iter([FakeFrame()])

        monkeypatch.setattr(
            "webinar_transcriber.video.frames.av.open", lambda *_args, **_kwargs: FakeContainer()
        )

        extracted, failure_detail = _extract_frame(SAMPLE_VIDEO_PATH, 1.0, output_path)

        assert not extracted
        assert failure_detail == f"PyAV did not write {output_path}"


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
    def test_iter_sampled_frames_wraps_open_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "webinar_transcriber.video.scenes.open_input_media_container",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(MediaProcessingError("bad open")),
        )

        with pytest.raises(
            MediaProcessingError,
            match="bad open",
        ):
            list(_iter_sampled_frames(SAMPLE_VIDEO_PATH))

    def test_iter_sampled_frames_raises_when_no_video_stream(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakeContainer(FakeContextContainer):
            def __init__(self) -> None:
                self.streams = [cast("object", type("AudioStream", (), {"type": "audio"})())]

        monkeypatch.setattr(
            "webinar_transcriber.video.scenes.open_input_media_container",
            lambda *_args, **_kwargs: FakeContainer(),
        )

        with pytest.raises(
            MediaProcessingError, match=f"No video stream found in {SAMPLE_VIDEO_PATH}."
        ):
            list(_iter_sampled_frames(SAMPLE_VIDEO_PATH))


class TestSceneSamplingHelpers:
    def test_scene_sampling_helpers_cover_zero_duration_and_trailing_sample(self) -> None:
        assert estimate_sample_count(0.0) == 1
        assert _estimate_sample_end_time(0.0) == 0.0
        assert _estimate_sample_end_time(4.0) == 5.0
