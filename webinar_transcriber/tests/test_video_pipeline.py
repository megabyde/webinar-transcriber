"""Tests for baseline video helpers."""

from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest
from PIL import Image

from webinar_transcriber.models import Scene
from webinar_transcriber.tests.conftest import FakeContextContainer
from webinar_transcriber.video import (
    SceneDetectionSettings,
    detect_scenes,
    estimated_scene_sample_count,
    extract_representative_frames,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures"
SAMPLE_VIDEO_PATH = FIXTURE_DIR / "sample-video.mp4"


def _scene(index: int, start_sec: float, end_sec: float) -> Scene:
    return Scene(id=f"scene-{index}", start_sec=start_sec, end_sec=end_sec)


class FakeImage:
    def __init__(self, *, save_output: bool = True) -> None:
        self.save_output = save_output

    def load(self) -> None:
        return None

    def getexif(self) -> dict[int, int]:
        return {}

    def copy(self) -> "FakeImage":
        return self

    def convert(self, _mode: str) -> "FakeImage":
        return self

    def save(self, path: Path) -> None:
        if self.save_output:
            path.write_bytes(b"frame")


class FakeFrame:
    def __init__(
        self,
        time: float | None,
        *,
        pts: int | None = None,
        time_base: float | None = None,
        save_output: bool = True,
        image: Image.Image | None = None,
    ) -> None:
        self.time = time
        self.pts = pts
        self.time_base = time_base
        self.save_output = save_output
        self.image = image

    def to_image(self) -> Image.Image | FakeImage:
        if self.image is not None:
            return self.image
        return FakeImage(save_output=self.save_output)


class FakeVideoStream:
    type = "video"
    index = 0


class FakeFrameContainer(FakeContextContainer):
    def __init__(self, frame_batches: list[list[FakeFrame]] | None = None) -> None:
        self.streams = [FakeVideoStream()]
        self.frame_batches = list(frame_batches or [])
        self.seek_offsets: list[int] = []
        self.decode_calls = 0

    def seek(self, offset: int, *_args, **_kwargs) -> None:
        self.seek_offsets.append(offset)

    def decode(self, *_args, **_kwargs):
        self.decode_calls += 1
        if self.frame_batches:
            return iter(self.frame_batches.pop(0))
        return iter([FakeFrame(999.0)])


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
    def test_detect_scenes_finds_multiple_segments(self) -> None:
        progress_updates: list[tuple[int, int]] = []

        def on_progress(sample_count: int, scene_count: int) -> None:
            progress_updates.append((sample_count, scene_count))

        scenes = detect_scenes(
            SAMPLE_VIDEO_PATH,
            settings=SceneDetectionSettings(
                scan_fps=2.0, score_threshold=0.006, min_scene_length_sec=1.0
            ),
            progress_callback=on_progress,
        )

        assert len(scenes) >= 2
        assert scenes[0].start_sec == 0.0
        assert progress_updates
        assert progress_updates[-1][1] == len(scenes)

    @pytest.mark.slow
    def test_detect_scenes_respects_default_min_scene_length(self) -> None:
        scenes = detect_scenes(SAMPLE_VIDEO_PATH)

        assert scenes == [Scene(id="scene-1", start_sec=0.0, end_sec=2.0)]

    @pytest.mark.slow
    def test_detect_scenes_keeps_blank_video_as_one_scene(self, tmp_path: Path) -> None:
        video_path = tmp_path / "blank.mp4"
        _write_synthetic_video(video_path, [0] * 10)

        scenes = detect_scenes(
            video_path,
            settings=SceneDetectionSettings(
                scan_fps=10.0, score_threshold=0.05, min_scene_length_sec=0.1
            ),
        )

        assert scenes == [Scene(id="scene-1", start_sec=0.0, end_sec=1.0)]

    @pytest.mark.slow
    def test_detect_scenes_finds_synthetic_alternating_color_scene(self, tmp_path: Path) -> None:
        video_path = tmp_path / "alternating-color.mp4"
        progress_updates: list[tuple[int, int]] = []
        _write_synthetic_video(video_path, [0, 255] * 5)

        scenes = detect_scenes(
            video_path,
            settings=SceneDetectionSettings(
                scan_fps=10.0, score_threshold=0.05, min_scene_length_sec=0.1
            ),
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


class TestFrameExtraction:
    @pytest.mark.slow
    def test_extract_representative_frames_creates_real_images(self, tmp_path: Path) -> None:
        scenes = [_scene(1, 0.0, 1.0), _scene(2, 1.0, 2.0)]

        frames = extract_representative_frames(SAMPLE_VIDEO_PATH, scenes, tmp_path / "frames")

        assert [frame.scene_id for frame in frames] == ["scene-1", "scene-2"]
        assert [frame.timestamp_sec for frame in frames] == [1.0, 2.0]
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
        open_calls: list[Path] = []
        container = FakeFrameContainer()

        def fake_open(path: str, **_kwargs) -> FakeFrameContainer:
            open_calls.append(Path(path))
            return container

        monkeypatch.setattr("webinar_transcriber.video.frames.av.open", fake_open)

        frames = extract_representative_frames(
            SAMPLE_VIDEO_PATH,
            scenes,
            tmp_path / "frames",
            progress_callback=lambda: progress_ticks.append(1),
        )

        assert len(frames) >= 2
        assert all(Path(frame.image_path).exists() for frame in frames)
        assert len(progress_ticks) == len(scenes)
        assert open_calls == [SAMPLE_VIDEO_PATH]
        assert container.decode_calls == len(scenes)

    def test_extract_representative_frames_applies_exif_orientation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        image = Image.new("RGB", (4, 2), color="white")
        exif = Image.Exif()
        exif[274] = 6
        image.info["exif"] = exif.tobytes()
        container = FakeFrameContainer([[FakeFrame(1.0, image=image)]])

        monkeypatch.setattr(
            "webinar_transcriber.video.frames.av.open", lambda *_args, **_kwargs: container
        )

        frames = extract_representative_frames(
            SAMPLE_VIDEO_PATH, [_scene(1, 0.0, 2.0)], tmp_path / "frames"
        )

        with Image.open(frames[0].image_path) as normalized_image:
            assert normalized_image.size == (2, 4)
            assert normalized_image.getexif().get(274) is None

    def test_extract_representative_frames_skips_failed_scene_but_still_reports_progress(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        progress_ticks: list[int] = []
        warnings: list[str] = []
        scenes = [_scene(1, 0.0, 2.0), _scene(2, 2.0, 4.0)]
        container = FakeFrameContainer([
            [FakeFrame(None)],
            [FakeFrame(3.0)],
        ])

        monkeypatch.setattr(
            "webinar_transcriber.video.frames.av.open", lambda *_args, **_kwargs: container
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
            "Frame extraction failed for scene-1 at 1.0s: PyAV did not decode a frame at 1.000s"
        ]

    def test_extract_representative_frames_warns_when_only_early_frames_decode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        warnings: list[str] = []
        container = FakeFrameContainer([[FakeFrame(0.9)]])

        monkeypatch.setattr(
            "webinar_transcriber.video.frames.av.open", lambda *_args, **_kwargs: container
        )

        frames = extract_representative_frames(
            SAMPLE_VIDEO_PATH,
            [_scene(1, 0.0, 2.0)],
            tmp_path / "frames",
            warning_callback=warnings.append,
        )

        assert [frame.scene_id for frame in frames] == ["scene-1"]
        assert warnings == [
            "Frame extraction used nearest decoded frame for scene-1 at 1.0s: "
            "PyAV decoded nearest frame before 1.000s at 0.900s"
        ]

    def test_extract_representative_frames_reports_only_unprocessed_scenes_after_late_save_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        progress_ticks: list[int] = []
        warnings: list[str] = []
        scenes = [_scene(1, 0.0, 2.0), _scene(2, 2.0, 4.0)]
        container = FakeFrameContainer([
            [FakeFrame(1.0)],
            [FakeFrame(3.0, save_output=False)],
        ])

        monkeypatch.setattr(
            "webinar_transcriber.video.frames.av.open", lambda *_args, **_kwargs: container
        )

        frames = extract_representative_frames(
            SAMPLE_VIDEO_PATH,
            scenes,
            tmp_path / "frames",
            progress_callback=lambda: progress_ticks.append(1),
            warning_callback=warnings.append,
        )

        assert [frame.scene_id for frame in frames] == ["scene-1"]
        assert progress_ticks == [1, 1]
        assert warnings == [
            f"Frame extraction failed for scene-2 at 3.0s: PyAV did not write "
            f"{tmp_path / 'frames' / 'scene-2.png'}"
        ]

    def test_extract_representative_frames_uses_first_stable_frame(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        container = FakeFrameContainer([[FakeFrame(3.0)]])

        monkeypatch.setattr(
            "webinar_transcriber.video.frames.av.open", lambda *_args, **_kwargs: container
        )

        frames = extract_representative_frames(
            SAMPLE_VIDEO_PATH, [_scene(1, 2.0, 5.0)], tmp_path / "frames"
        )

        assert container.seek_offsets == [int(3.0 * av.time_base)]
        assert frames[0].timestamp_sec == 3.0

    def test_extract_representative_frames_warns_each_scene_when_open_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        progress_ticks: list[int] = []
        warnings: list[str] = []
        scenes = [_scene(1, 0.0, 2.0), _scene(2, 2.0, 4.0)]
        monkeypatch.setattr(
            "webinar_transcriber.video.frames.av.open",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("bad open")),
        )

        frames = extract_representative_frames(
            SAMPLE_VIDEO_PATH,
            scenes,
            tmp_path / "frames",
            progress_callback=lambda: progress_ticks.append(1),
            warning_callback=warnings.append,
        )

        assert frames == []
        assert progress_ticks == [1, 1]
        open_error = f"Could not open {SAMPLE_VIDEO_PATH} with PyAV: bad open"
        assert warnings == [
            f"Frame extraction failed for scene-1 at 1.0s: {open_error}",
            f"Frame extraction failed for scene-2 at 3.0s: {open_error}",
        ]


class TestDetectScenesFallback:
    def test_estimated_scene_sample_count_uses_detection_settings(self) -> None:
        sample_count = estimated_scene_sample_count(
            9.1, settings=SceneDetectionSettings(scan_fps=0.5)
        )

        assert sample_count == 5

    def test_estimated_scene_sample_count_keeps_zero_duration_visible(self) -> None:
        assert estimated_scene_sample_count(0.0) == 1

    @pytest.mark.slow
    def test_detect_scenes_prefers_explicit_duration(self, tmp_path: Path) -> None:
        video_path = tmp_path / "blank.mp4"
        _write_synthetic_video(video_path, [0] * 10)

        scenes = detect_scenes(
            video_path,
            duration_sec=2.0,
            settings=SceneDetectionSettings(
                scan_fps=10.0, score_threshold=0.05, min_scene_length_sec=0.1
            ),
        )

        assert scenes == [Scene(id="scene-1", start_sec=0.0, end_sec=2.0)]
