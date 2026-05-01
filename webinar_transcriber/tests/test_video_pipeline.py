"""Tests for baseline video helpers."""

from pathlib import Path
from typing import cast

import av
import pytest
from PIL import Image

from webinar_transcriber.media import MediaProcessingError
from webinar_transcriber.models import Scene
from webinar_transcriber.tests.conftest import FakeContextContainer
from webinar_transcriber.video import (
    SceneDetectionSettings,
    detect_scenes,
    estimated_scene_sample_count,
    extract_representative_frames,
)
from webinar_transcriber.video.frames import _normalize_extracted_frame
from webinar_transcriber.video.scenes import _select_scene_starts

FIXTURE_DIR = Path(__file__).parent / "fixtures"
SAMPLE_VIDEO_PATH = FIXTURE_DIR / "sample-video.mp4"


def _scene(index: int, start_sec: float, end_sec: float) -> Scene:
    return Scene(id=f"scene-{index}", start_sec=start_sec, end_sec=end_sec)


class FakeImage:
    def __init__(self, *, save_output: bool = True) -> None:
        self.save_output = save_output

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
    ) -> None:
        self.time = time
        self.pts = pts
        self.time_base = time_base
        self.save_output = save_output

    def to_image(self) -> FakeImage:
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
        assert open_calls == [SAMPLE_VIDEO_PATH]
        assert container.decode_calls == len(scenes)

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
        container = FakeFrameContainer([
            [FakeFrame(None)],
            [FakeFrame(3.0)],
        ])

        monkeypatch.setattr(
            "webinar_transcriber.video.frames.av.open", lambda *_args, **_kwargs: container
        )
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
        monkeypatch.setattr(
            "webinar_transcriber.video.frames._normalize_extracted_frame", lambda _p: None
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

    def test_extract_representative_frames_reports_only_unprocessed_scenes_after_late_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        progress_ticks: list[int] = []
        warnings: list[str] = []
        scenes = [_scene(1, 0.0, 2.0), _scene(2, 2.0, 4.0)]
        container = FakeFrameContainer([
            [FakeFrame(1.0)],
            [FakeFrame(3.0)],
        ])

        def normalize_or_fail(path: Path) -> None:
            if path.name == "scene-2.png":
                raise OSError("normalize failed")

        monkeypatch.setattr(
            "webinar_transcriber.video.frames.av.open", lambda *_args, **_kwargs: container
        )
        monkeypatch.setattr(
            "webinar_transcriber.video.frames._normalize_extracted_frame", normalize_or_fail
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
        assert warnings == ["Frame extraction failed for scene-2 at 3.0s: normalize failed"]

    def test_extract_representative_frames_uses_first_stable_frame(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        container = FakeFrameContainer([[FakeFrame(3.0)]])

        monkeypatch.setattr(
            "webinar_transcriber.video.frames.av.open", lambda *_args, **_kwargs: container
        )
        monkeypatch.setattr(
            "webinar_transcriber.video.frames._normalize_extracted_frame", lambda _p: None
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
        assert warnings == [
            "Frame extraction failed for scene-1 at 1.0s: bad open",
            "Frame extraction failed for scene-2 at 3.0s: bad open",
        ]


class TestDetectScenesFallback:
    def test_estimated_scene_sample_count_uses_detection_settings(self) -> None:
        sample_count = estimated_scene_sample_count(
            9.1, settings=SceneDetectionSettings(scan_fps=0.5)
        )

        assert sample_count == 5

    def test_detect_scenes_returns_single_zero_length_scene_when_no_frames(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "webinar_transcriber.video.scenes._select_scene_starts",
            lambda *_args, **_kwargs: ([0.0], 0.0),
        )

        scenes = detect_scenes(SAMPLE_VIDEO_PATH)

        assert scenes == [Scene(id="scene-1", start_sec=0.0, end_sec=0.0)]

    def test_detect_scenes_prefers_explicit_duration(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "webinar_transcriber.video.scenes._select_scene_starts",
            lambda *_args, **_kwargs: ([0.0, 1.0], 9.0),
        )

        scenes = detect_scenes(SAMPLE_VIDEO_PATH, duration_sec=2.0)

        assert [(scene.start_sec, scene.end_sec) for scene in scenes] == [(0.0, 1.0), (1.0, 2.0)]


class TestSelectSceneStarts:
    def test_select_scene_starts_wraps_open_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "webinar_transcriber.video.scenes.open_input_media_container",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(MediaProcessingError("bad open")),
        )

        with pytest.raises(MediaProcessingError, match="bad open"):
            _select_scene_starts(
                SAMPLE_VIDEO_PATH,
                settings=SceneDetectionSettings(score_threshold=0.3, min_scene_length_sec=3.0),
            )

    def test_select_scene_starts_raises_when_no_video_stream(
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
            _select_scene_starts(
                SAMPLE_VIDEO_PATH,
                settings=SceneDetectionSettings(score_threshold=0.3, min_scene_length_sec=3.0),
            )

    def test_select_scene_starts_reports_progress(self, monkeypatch: pytest.MonkeyPatch) -> None:
        progress_updates: list[tuple[int, int]] = []

        class FakeFrame:
            def __init__(
                self, time: float | None, *, pts: int | None = None, time_base: float | None = None
            ) -> None:
                self.time = time
                self.pts = pts
                self.time_base = time_base

        class FakeVideoStream:
            type = "video"
            duration = 50
            time_base = 0.1

        class FakeGraph:
            def __init__(self) -> None:
                self._queued_frames = [
                    [],
                    [FakeFrame(None), FakeFrame(None, pts=10, time_base=0.1)],
                    [FakeFrame(4.0)],
                ]
                self._active_outputs: list[FakeFrame] = []

            def vpush(self, _frame: object) -> None:
                self._active_outputs = self._queued_frames.pop(0)

            def vpull(self) -> FakeFrame:
                if not self._active_outputs:
                    raise av.BlockingIOError(11, "again")
                return self._active_outputs.pop(0)

        class FakeContainer(FakeContextContainer):
            duration = None

            def __init__(self) -> None:
                self.streams = [FakeVideoStream()]

            def decode(self, *_args, **_kwargs):
                return iter([FakeFrame(None), FakeFrame(None), FakeFrame(None)])

        monkeypatch.setattr(
            "webinar_transcriber.video.scenes.open_input_media_container",
            lambda *_args, **_kwargs: FakeContainer(),
        )
        monkeypatch.setattr(
            "webinar_transcriber.video.scenes._build_scene_filter_graph",
            lambda *_args, **_kwargs: FakeGraph(),
        )

        scene_starts, duration_sec = _select_scene_starts(
            SAMPLE_VIDEO_PATH,
            settings=SceneDetectionSettings(
                scan_fps=2.0, score_threshold=0.3, min_scene_length_sec=3.0
            ),
            progress_callback=lambda sample_count, scene_count: progress_updates.append((
                sample_count,
                scene_count,
            )),
        )

        assert scene_starts == [0.0, 4.0]
        assert duration_sec == 5.0
        assert progress_updates == [(1, 1), (2, 1), (3, 2), (10, 2)]

    def test_select_scene_starts_returns_zero_duration_without_container_or_stream_duration(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        progress_updates: list[tuple[int, int]] = []

        class FakeVideoStream:
            type = "video"
            duration = None
            time_base = None

        class FakeGraph:
            def vpush(self, _frame: object) -> None:
                return None

            def vpull(self) -> object:
                raise av.BlockingIOError(11, "again")

        class FakeContainer(FakeContextContainer):
            duration = None

            def __init__(self) -> None:
                self.streams = [FakeVideoStream()]

            def decode(self, *_args, **_kwargs):
                return iter([FakeFrame(None)])

        monkeypatch.setattr(
            "webinar_transcriber.video.scenes.open_input_media_container",
            lambda *_args, **_kwargs: FakeContainer(),
        )
        monkeypatch.setattr(
            "webinar_transcriber.video.scenes._build_scene_filter_graph",
            lambda *_args, **_kwargs: FakeGraph(),
        )

        scene_starts, duration_sec = _select_scene_starts(
            SAMPLE_VIDEO_PATH,
            settings=SceneDetectionSettings(score_threshold=0.3, min_scene_length_sec=3.0),
            progress_callback=lambda sample_count, scene_count: progress_updates.append((
                sample_count,
                scene_count,
            )),
        )

        assert scene_starts == [0.0]
        assert duration_sec == 0.0
        assert progress_updates == [(1, 1)]

    def test_select_scene_starts_wraps_filter_errors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeVideoStream:
            type = "video"
            duration = None
            time_base = None

        class FakeGraph:
            def vpush(self, _frame: object) -> None:
                raise OSError("filter failed")

        class FakeContainer(FakeContextContainer):
            duration = None

            def __init__(self) -> None:
                self.streams = [FakeVideoStream()]

            def decode(self, *_args, **_kwargs):
                return iter([FakeFrame(None)])

        monkeypatch.setattr(
            "webinar_transcriber.video.scenes.open_input_media_container",
            lambda *_args, **_kwargs: FakeContainer(),
        )
        monkeypatch.setattr(
            "webinar_transcriber.video.scenes._build_scene_filter_graph",
            lambda *_args, **_kwargs: FakeGraph(),
        )

        with pytest.raises(
            MediaProcessingError,
            match=f"Could not detect scenes in {SAMPLE_VIDEO_PATH}: filter failed",
        ):
            _select_scene_starts(
                SAMPLE_VIDEO_PATH,
                settings=SceneDetectionSettings(score_threshold=0.3, min_scene_length_sec=3.0),
            )
