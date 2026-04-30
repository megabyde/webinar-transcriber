"""Shared test helpers for webinar_transcriber/tests."""

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
import pytest

from webinar_transcriber.asr import WhisperCppTranscriber
from webinar_transcriber.models import (
    AudioAsset,
    DecodedWindow,
    InferenceWindow,
    SpeechRegion,
    TranscriptSegment,
    VideoAsset,
)
from webinar_transcriber.processor import ProcessArtifacts
from webinar_transcriber.reporter import BaseStageReporter

FIXTURE_DIR = Path(__file__).parent / "fixtures"


class FakeTorch:
    @staticmethod
    def from_numpy(arr):
        return arr


class FakeContextContainer:
    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


@pytest.fixture
def fake_silero_import_module() -> Callable[..., Callable[[str], object]]:
    def build(
        *,
        iterator_cls: type | None = None,
        get_speech_timestamps_fn: Callable[..., object] | None = None,
    ) -> Callable[[str], object]:
        class FakeSilero:
            VADIterator: type | None = None
            get_speech_timestamps: Callable[..., object] | None = None

            @staticmethod
            def load_silero_vad():
                return object()

        if iterator_cls is not None:  # pragma: no cover - fixture option for narrow VAD tests
            FakeSilero.VADIterator = iterator_cls
        if get_speech_timestamps_fn is not None:
            FakeSilero.get_speech_timestamps = staticmethod(get_speech_timestamps_fn)

        def fake_import_module(name: str) -> object:
            if name == "silero_vad":
                return FakeSilero
            if name == "torch":
                return FakeTorch
            raise ImportError(name)  # pragma: no cover - mirrors importlib for unexpected modules

        return fake_import_module

    return build


@dataclass
class PipelineRuntime:
    media_asset: AudioAsset | VideoAsset
    speech_regions: list[SpeechRegion]
    audio_samples: np.ndarray
    sample_rate: int = 16_000
    vad_warnings: list[str] | None = None


class RecordingReporter(BaseStageReporter):
    """Small reporter fake that records observable pipeline notifications."""

    def __init__(self) -> None:
        self.started: list[tuple[str, str]] = []
        self.finished: list[tuple[str, str]] = []
        self.progress: list[tuple[str, str, float, str | None]] = []
        self.warnings: list[str] = []
        self.completed: ProcessArtifacts | None = None

    def stage_started(self, stage_key: str, label: str) -> None:
        self.started.append((stage_key, label))

    def progress_started(
        self,
        stage_key: str,
        label: str,
        *,
        total: float,
        count_label: str | None = None,
        count_multiplier: float = 1.0,
        rate_label: str | None = None,
        rate_multiplier: float = 1.0,
        detail: str | None = None,
    ) -> None:
        del count_label, count_multiplier, rate_label, rate_multiplier
        self.started.append((stage_key, label))
        self.progress.append(("start", stage_key, total, detail))

    def progress_advanced(
        self, stage_key: str, *, advance: float = 1.0, detail: str | None = None
    ) -> None:
        self.progress.append(("advance", stage_key, advance, detail))

    def stage_finished(self, stage_key: str, label: str, *, detail: str | None = None) -> None:
        del label
        self.finished.append((stage_key, detail or ""))

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def complete_run(self, artifacts: ProcessArtifacts) -> None:
        self.completed = artifacts


class FakeTranscriber(WhisperCppTranscriber):
    """Stable whisper-style test double for deterministic transcripts."""

    def __init__(
        self, *, detected_language: str = "en", segments: list[TranscriptSegment] | None = None
    ) -> None:
        super().__init__(model_name="test-model")
        self._detected_language = detected_language
        self.language_hints: list[str | None] = []
        self.windows_seen: list[InferenceWindow] = []
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
        language: str | None = None,
        progress_callback: Callable[[float, int], None] | None = None,
    ) -> list[DecodedWindow]:
        del audio_samples
        self.language_hints.append(language)
        self.windows_seen = list(windows)
        assert progress_callback is not None
        for window in windows:
            progress_callback(window.end_sec, len(self._segments))
        return [
            DecodedWindow(
                window=window,
                text=" ".join(segment.text for segment in self._segments),
                language=self._detected_language,
                segments=list(self._segments),
            )
            for window in windows
        ]


def install_pipeline_runtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, input_path: Path, runtime: PipelineRuntime
) -> None:
    """Patch expensive processor seams while letting the real pipeline run."""
    audio_path = tmp_path / f"{input_path.stem}.wav"

    @contextmanager
    def fake_prepared_transcription_audio(_input_path: Path):
        audio_path.write_bytes(b"RIFFstub")
        try:
            yield audio_path
        finally:
            audio_path.unlink(missing_ok=True)

    def fake_detect_speech_regions(*_args, **_kwargs):
        return runtime.speech_regions, list(runtime.vad_warnings or [])

    monkeypatch.setattr(
        "webinar_transcriber.processor.prepared_transcription_audio",
        fake_prepared_transcription_audio,
    )
    monkeypatch.setattr(
        "webinar_transcriber.processor.media_runtime.probe_media",
        lambda _input_path: runtime.media_asset,
    )
    monkeypatch.setattr(
        "webinar_transcriber.processor.asr.load_normalized_audio",
        lambda _path: (runtime.audio_samples, runtime.sample_rate),
    )
    monkeypatch.setattr(
        "webinar_transcriber.processor.asr.detect_speech_regions", fake_detect_speech_regions
    )


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
