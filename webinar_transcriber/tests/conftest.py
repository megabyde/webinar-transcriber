"""Shared test helpers for webinar_transcriber/tests."""

import importlib
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Self

import numpy as np
import pytest
from PIL import Image
from rich.console import Console

from webinar_transcriber.asr import WhisperCppTranscriber
from webinar_transcriber.models import (
    AudioAsset,
    DecodedWindow,
    Diagnostics,
    InferenceWindow,
    MediaType,
    ReportDocument,
    Scene,
    SpeechRegion,
    TranscriptionResult,
    TranscriptSegment,
    VideoAsset,
)
from webinar_transcriber.paths import RunLayout
from webinar_transcriber.processor import ProcessArtifacts
from webinar_transcriber.ui import StageReporter

FIXTURE_DIR = Path(__file__).parent / "fixtures"
ORIGINAL_IMPORT_MODULE = importlib.import_module


def fake_import_module(
    modules: dict[str, object], *, fallback: bool = False
) -> Callable[[str], object]:
    def import_module(name: str) -> object:
        if name in modules:
            return modules[name]
        if fallback:
            return ORIGINAL_IMPORT_MODULE(name)
        raise ImportError(name)

    return import_module


@dataclass
class _RecordingHandle:
    """Test handle that captures update() calls on a determinate stage."""

    key: str
    label: str
    total: float | None
    detail: str | None
    progress_log: list[tuple[str, str, float, str | None]]
    completed: float = 0.0
    started_at: float = field(default_factory=perf_counter)

    def elapsed_sec(self) -> float:
        return perf_counter() - self.started_at

    def update(
        self,
        *,
        advance: float | None = None,
        completed: float | None = None,
        detail: str | None = None,
    ) -> None:
        if completed is not None:
            advance = max(0.0, completed - self.completed)
        applied_advance = advance if advance is not None and advance > 0 else 0.0
        if applied_advance > 0:
            self.completed += applied_advance
        if detail is not None:
            self.detail = detail
        if self.total is not None:
            self.progress_log.append(("advance", self.key, applied_advance, detail))


class RecordingStageReporter(StageReporter):
    """Test reporter that records observable pipeline events."""

    def __init__(self) -> None:
        super().__init__(console=Console(quiet=True))
        self.started: list[tuple[str, str]] = []
        self.finished: list[tuple[str, str]] = []
        self.progress: list[tuple[str, str, float, str | None]] = []
        self.warnings: list[str] = []
        self.completed: ProcessArtifacts | None = None

    @contextmanager
    def track(self, key, label, *, total=None, detail=None):
        self.started.append((key, label))
        if total is not None:
            self.progress.append(("start", key, total, detail))
        handle = _RecordingHandle(
            key=key, label=label, total=total, detail=detail, progress_log=self.progress
        )
        completed_normally = False
        try:
            yield handle
            completed_normally = True
        finally:
            if completed_normally:
                self.finished.append((key, handle.detail or ""))

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def complete_run(self, artifacts: ProcessArtifacts) -> None:
        self.completed = artifacts


def process_artifacts(input_path: Path, run_dir: Path) -> ProcessArtifacts:
    return ProcessArtifacts(
        layout=RunLayout(run_dir=run_dir),
        media_asset=VideoAsset(path=str(input_path), duration_sec=1.0),
        transcription=TranscriptionResult(detected_language="en"),
        report=ReportDocument(
            title="Demo", source_file=str(input_path), media_type=MediaType.VIDEO
        ),
        diagnostics=Diagnostics(),
    )


class FakeContextContainer:
    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class FakeSherpaVoiceActivityDetector:
    def __init__(
        self,
        config: SimpleNamespace,
        *,
        buffer_size_in_seconds: int,
        segments: list[object],
    ) -> None:
        self.config = config
        self.buffer_size_in_seconds = buffer_size_in_seconds
        self.accepted_waveforms: list[object] = []
        self.flushed = False
        self.segments = list(segments)

    def accept_waveform(self, samples) -> None:
        self.accepted_waveforms.append(samples)

    def flush(self) -> None:
        self.flushed = True

    def empty(self) -> bool:
        return not self.segments

    @property
    def front(self):
        return self.segments[0]

    def pop(self) -> None:
        self.segments.pop(0)


class FakeSherpaModule:
    def __init__(self, *, segments: list[object] | None, window_size: int) -> None:
        self._segments = list(segments or [])
        self._window_size = window_size
        self.detectors: list[FakeSherpaVoiceActivityDetector] = []

    def VadModelConfig(self) -> SimpleNamespace:  # noqa: N802
        # Production sets every config field it reads; only window_size needs a fake value.
        return SimpleNamespace(silero_vad=SimpleNamespace(window_size=self._window_size))

    def VoiceActivityDetector(  # noqa: N802
        self, config: SimpleNamespace, *, buffer_size_in_seconds: int
    ) -> FakeSherpaVoiceActivityDetector:
        detector = FakeSherpaVoiceActivityDetector(
            config, buffer_size_in_seconds=buffer_size_in_seconds, segments=self._segments
        )
        self.detectors.append(detector)
        return detector


@pytest.fixture
def fake_sherpa_import_module() -> Callable[..., tuple[Callable[[str], object], object]]:
    def build(*, segments: list[object] | None = None, window_size: int = 512):
        fake_sherpa = FakeSherpaModule(segments=segments, window_size=window_size)
        return fake_import_module({"sherpa_onnx": fake_sherpa}, fallback=True), fake_sherpa

    return build


@dataclass
class PipelineRuntime:
    media_asset: AudioAsset | VideoAsset
    speech_regions: list[SpeechRegion]
    audio_samples: np.ndarray
    sample_rate: int = 16_000
    vad_warnings: list[str] | None = None


class FakeTranscriber(WhisperCppTranscriber):
    """Stable whisper-style test double for deterministic transcripts."""

    def __init__(
        self,
        *,
        detected_language: str = "en",
        segments: list[TranscriptSegment] | None = None,
        window_segments: list[list[TranscriptSegment]] | None = None,
    ) -> None:
        super().__init__(model_name="test-model", threads=4)
        self._detected_language = detected_language
        self.windows_seen: list[InferenceWindow] = []
        # window_segments gives distinct segments per decode window; segments repeats one list for
        # every window (the common case).
        self._window_segments = window_segments
        self._segments = segments or [
            TranscriptSegment(
                id="segment-1",
                text="Agenda review and project status update.",
                start_sec=0.0,
                end_sec=3.0,
            ),
            TranscriptSegment(
                id="segment-2",
                text="Next step please send the draft by Friday.",
                start_sec=3.0,
                end_sec=6.0,
            ),
        ]

    @property
    def device_name(self) -> str:
        return "cpu"

    @property
    def system_info(self) -> str:
        return "CPU = 1"

    def prepare_model(self) -> None:
        return None

    def transcribe_inference_windows(
        self,
        audio_samples: np.ndarray,
        windows: list[InferenceWindow],
        *,
        progress_callback: Callable[[int, int], None] | None = None,
        warning_callback: Callable[[str], None] | None = None,
    ) -> list[DecodedWindow]:
        del audio_samples, warning_callback
        self.windows_seen = list(windows)
        per_window = self._window_segments or [self._segments for _ in windows]
        if progress_callback is not None:
            for index, segments in enumerate(per_window, start=1):
                progress_callback(index, len(segments))
        return [
            DecodedWindow(
                window=window,
                text=" ".join(segment.text for segment in segments),
                detected_language=self._detected_language,
                segments=list(segments),
            )
            for window, segments in zip(windows, per_window, strict=False)
        ]


def install_pipeline_runtime(monkeypatch: pytest.MonkeyPatch, runtime: PipelineRuntime) -> None:
    """Patch expensive processor seams while letting the real pipeline run."""

    def fake_write_transcription_audio(
        _input_path: Path, output_path: Path, *, progress_callback=None, **_kwargs
    ) -> Path:
        if progress_callback is not None:
            progress_callback(runtime.media_asset.duration_sec)
        output_path.write_bytes(b"RIFFstub")
        return output_path

    def fake_detect_speech_regions(*_args, progress_callback=None, **_kwargs):
        if progress_callback is not None:
            progress_callback(
                len(runtime.audio_samples) / runtime.sample_rate, len(runtime.speech_regions)
            )
        return runtime.speech_regions, list(runtime.vad_warnings or [])

    monkeypatch.setattr(
        "webinar_transcriber.processor.write_transcription_audio",
        fake_write_transcription_audio,
    )
    monkeypatch.setattr(
        "webinar_transcriber.processor.probe_media", lambda _input_path: runtime.media_asset
    )
    monkeypatch.setattr(
        "webinar_transcriber.processor.load_normalized_audio",
        lambda _path: runtime.audio_samples,
    )
    monkeypatch.setattr(
        "webinar_transcriber.processor.detect_speech_regions", fake_detect_speech_regions
    )


def install_video_scene_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    scenes: list[Scene],
) -> None:
    def fake_detect_scenes(
        _input_path: Path, _duration_sec: float, *, progress_callback=None
    ) -> list[Scene]:
        for index, _scene in enumerate(scenes, start=1):
            if progress_callback is not None:
                progress_callback(index, index)
        return scenes

    def fake_save_scene_frames(
        _input_path: Path, in_scenes: list[Scene], frames_dir: Path, *, progress_callback=None
    ) -> list[Scene]:
        frames_dir.mkdir(parents=True, exist_ok=True)
        for index, scene in enumerate(in_scenes, start=1):
            if scene.image_path is not None:
                Image.new("RGB", (8, 8), color="white").save(
                    frames_dir / Path(scene.image_path).name
                )
            if progress_callback is not None:
                progress_callback(index)
        return in_scenes

    monkeypatch.setattr("webinar_transcriber.processor.detect_scenes", fake_detect_scenes)
    monkeypatch.setattr("webinar_transcriber.processor.save_scene_frames", fake_save_scene_frames)


def audio_runtime(
    *,
    duration_sec: float = 6.0,
    speech_regions: list[SpeechRegion] | None = None,
    vad_warnings: list[str] | None = None,
) -> PipelineRuntime:
    return PipelineRuntime(
        media_asset=AudioAsset(
            path=str(FIXTURE_DIR / "sample-audio.mp3"),
            duration_sec=duration_sec,
            sample_rate=44_100,
            channels=2,
        ),
        speech_regions=speech_regions or [SpeechRegion(start_sec=0.0, end_sec=duration_sec)],
        audio_samples=np.zeros(16_000, dtype=np.float32),
        vad_warnings=vad_warnings,
    )


def video_runtime() -> PipelineRuntime:
    return PipelineRuntime(
        media_asset=VideoAsset(
            path=str(FIXTURE_DIR / "sample-video.mp4"),
            duration_sec=2.0,
            sample_rate=48_000,
            channels=2,
            fps=25.0,
            width=320,
            height=240,
        ),
        speech_regions=[SpeechRegion(start_sec=0.0, end_sec=1.8)],
        audio_samples=np.zeros(32_000, dtype=np.float32),
    )
