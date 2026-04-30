"""Behavior-focused tests for the processor pipeline."""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from webinar_transcriber.asr import WhisperCppTranscriber
from webinar_transcriber.diagnostics import write_run_diagnostics
from webinar_transcriber.llm.contracts import (
    LLMConfigurationError,
    LLMProcessingError,
    LLMProcessor,
    LLMReportMetadataResult,
    LLMReportPolishPlan,
    LLMSectionPolishResult,
)
from webinar_transcriber.models import (
    AsrPipelineDiagnostics,
    AudioAsset,
    DecodedWindow,
    InferenceWindow,
    ReportDocument,
    Scene,
    SlideFrame,
    SpeechRegion,
    TranscriptSegment,
    VideoAsset,
)
from webinar_transcriber.paths import RunLayout
from webinar_transcriber.processor import ProcessArtifacts, RunContext, process_input
from webinar_transcriber.processor.llm import resolve_llm_processor
from webinar_transcriber.processor.llm_types import LLMRuntimeState
from webinar_transcriber.processor.support import (
    asr_runtime_detail,
    stage,
    window_transcription_stage_detail,
)
from webinar_transcriber.reporter import BaseStageReporter
from webinar_transcriber.segmentation import VadSettings

if TYPE_CHECKING:
    from collections.abc import Callable

FIXTURE_DIR = Path(__file__).parent / "fixtures"
EXPECTED_LLM_WARNING = (
    "Section polish response returned an empty transcript text for section-1; kept original text."
)
EXPECTED_LLM_USAGE = {"input_tokens": 20, "output_tokens": 6, "total_tokens": 26}
EXPECTED_LLM_SECTION_TEXT = (
    "Agenda review, and project status update.\n\nNext step: please send the draft by Friday."
)


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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


class TestProcessInput:
    def test_processes_audio_into_reports_and_diagnostics(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        input_path = FIXTURE_DIR / "sample-audio.mp3"
        reporter = RecordingReporter()
        transcriber = FakeTranscriber()
        runtime = audio_runtime()
        install_pipeline_runtime(monkeypatch, tmp_path, input_path=input_path, runtime=runtime)

        artifacts = process_input(
            input_path,
            output_dir=tmp_path / "run",
            language="en",
            transcriber=transcriber,
            reporter=reporter,
        )

        assert transcriber.language_hints == ["en"]
        assert reporter.completed is artifacts
        assert artifacts.report.detected_language == "en"
        assert artifacts.report.summary == []
        assert artifacts.report.action_items == []
        assert len(artifacts.report.sections) == 1
        assert (
            "Next step please send the draft by Friday."
            in artifacts.report.sections[0].transcript_text
        )

        assert artifacts.layout.metadata_path.exists()
        assert artifacts.layout.transcript_path.exists()
        assert artifacts.layout.subtitle_vtt_path.exists()
        assert artifacts.layout.markdown_report_path.exists()
        assert artifacts.layout.docx_report_path.stat().st_size > 0
        assert artifacts.layout.json_report_path.exists()
        assert artifacts.layout.speech_regions_path.exists()
        assert artifacts.layout.decoded_windows_path.exists()
        assert not artifacts.layout.transcription_audio_path().exists()

        markdown = artifacts.layout.markdown_report_path.read_text(encoding="utf-8")
        vtt = artifacts.layout.subtitle_vtt_path.read_text(encoding="utf-8")
        diagnostics_payload = read_json(artifacts.layout.diagnostics_path)

        assert "# Sample Audio" in markdown
        assert "Agenda review and project status update." in markdown
        assert "WEBVTT" in vtt
        assert "00:00:00.000 --> 00:00:03.000" in vtt
        assert diagnostics_payload["asr_backend"] == "whisper.cpp"
        assert diagnostics_payload["asr_model"] == "test-model"
        assert diagnostics_payload["item_counts"]["windows"] == 1
        assert diagnostics_payload["item_counts"]["report_sections"] == 1
        assert artifacts.diagnostics.asr_pipeline is not None
        assert artifacts.diagnostics.asr_pipeline.system_info == "CPU = 1"

    def test_uses_vad_regions_as_decode_windows(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        input_path = FIXTURE_DIR / "sample-audio.mp3"
        transcriber = FakeTranscriber()
        regions = [
            SpeechRegion(start_sec=0.0, end_sec=1.2),
            SpeechRegion(start_sec=2.1, end_sec=3.0),
        ]
        install_pipeline_runtime(
            monkeypatch,
            tmp_path,
            input_path=input_path,
            runtime=audio_runtime(duration_sec=3.0, speech_regions=regions),
        )

        artifacts = process_input(
            input_path, output_dir=tmp_path / "windowed-run", transcriber=transcriber
        )

        decoded_windows_payload = read_json(artifacts.layout.decoded_windows_path)
        assert [window["window"] for window in decoded_windows_payload["decoded_windows"]] == [
            {"window_id": "window-1", "region_index": 0, "start_sec": 0.0, "end_sec": 1.2},
            {"window_id": "window-2", "region_index": 1, "start_sec": 2.1, "end_sec": 3.0},
        ]
        assert transcriber.windows_seen == [
            InferenceWindow(window_id="window-1", region_index=0, start_sec=0.0, end_sec=1.2),
            InferenceWindow(window_id="window-2", region_index=1, start_sec=2.1, end_sec=3.0),
        ]
        assert artifacts.diagnostics.item_counts["vad_regions"] == 2
        assert artifacts.diagnostics.item_counts["windows"] == 2
        assert artifacts.diagnostics.asr_pipeline is not None
        assert artifacts.diagnostics.asr_pipeline.average_window_duration_sec == pytest.approx(1.05)

    def test_normalizes_transcript_before_report_generation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        input_path = FIXTURE_DIR / "sample-audio.mp3"
        install_pipeline_runtime(
            monkeypatch, tmp_path, input_path=input_path, runtime=audio_runtime()
        )
        transcriber = FakeTranscriber(
            detected_language="ru",
            segments=[
                TranscriptSegment(id="segment-1", text="", start_sec=0.0, end_sec=0.1),
                TranscriptSegment(id="segment-2", text="Привет", start_sec=0.1, end_sec=0.8),
                TranscriptSegment(id="segment-3", text="всем.", start_sec=0.8, end_sec=1.4),
                TranscriptSegment(
                    id="segment-4", text="Новая тема начинается позже.", start_sec=3.0, end_sec=6.0
                ),
            ],
        )

        artifacts = process_input(input_path, output_dir=tmp_path / "run", transcriber=transcriber)

        raw_payload = read_json(artifacts.layout.transcript_path)
        assert [segment["text"] for segment in raw_payload["segments"]] == [
            "Привет",
            "всем.",
            "Новая тема начинается позже.",
        ]
        assert artifacts.diagnostics.item_counts["normalized_transcript_segments"] == 2
        assert artifacts.report.sections[0].transcript_text.startswith("Привет всем.")

    def test_keeps_normalized_audio_artifact(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        input_path = FIXTURE_DIR / "sample-audio.mp3"
        install_pipeline_runtime(
            monkeypatch, tmp_path, input_path=input_path, runtime=audio_runtime()
        )

        artifacts = process_input(
            input_path,
            output_dir=tmp_path / "run",
            transcriber=FakeTranscriber(),
            keep_audio=True,
            kept_audio_format="wav",
        )

        kept_audio_path = artifacts.layout.transcription_audio_path()
        assert kept_audio_path.exists()
        assert kept_audio_path.read_bytes()[:4] == b"RIFF"

    @pytest.mark.slow
    def test_keeps_normalized_audio_artifact_with_real_media_prep(self, tmp_path: Path) -> None:
        artifacts = process_input(
            FIXTURE_DIR / "sample-audio.mp3",
            output_dir=tmp_path / "real-run",
            transcriber=FakeTranscriber(),
            keep_audio=True,
            kept_audio_format="wav",
            vad=VadSettings(enabled=False),
        )

        kept_audio_path = artifacts.layout.transcription_audio_path()
        assert kept_audio_path.exists()
        assert kept_audio_path.read_bytes()[:4] == b"RIFF"
        assert artifacts.media_asset.path.endswith("sample-audio.mp3")
        assert artifacts.media_asset.duration_sec > 0

    @pytest.mark.slow
    def test_video_artifact_contract_with_real_media_pipeline(self, tmp_path: Path) -> None:
        artifacts = process_input(
            FIXTURE_DIR / "sample-video.mp4",
            output_dir=tmp_path / "real-video-run",
            transcriber=FakeTranscriber(
                segments=[
                    TranscriptSegment(
                        id="segment-1",
                        text="Opening slide.",
                        start_sec=0.0,
                        end_sec=0.8,
                    ),
                    TranscriptSegment(
                        id="segment-2",
                        text="Closing slide.",
                        start_sec=0.8,
                        end_sec=1.8,
                    ),
                ]
            ),
            vad=VadSettings(enabled=False),
        )

        assert artifacts.layout.metadata_path.exists()
        assert artifacts.layout.scenes_path.exists()
        assert artifacts.layout.frames_dir.exists()
        assert artifacts.layout.markdown_report_path.exists()
        assert artifacts.layout.docx_report_path.stat().st_size > 0
        assert artifacts.layout.json_report_path.exists()
        assert artifacts.layout.subtitle_vtt_path.exists()
        assert artifacts.diagnostics.item_counts["scenes"] >= 1
        assert artifacts.diagnostics.item_counts["frames"] >= 1
        assert artifacts.report.sections
        assert any(section.image_path for section in artifacts.report.sections)

    def test_does_not_close_caller_supplied_transcriber(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        input_path = FIXTURE_DIR / "sample-audio.mp3"
        install_pipeline_runtime(
            monkeypatch, tmp_path, input_path=input_path, runtime=audio_runtime()
        )

        class CloseTrackingTranscriber(FakeTranscriber):
            def __init__(self) -> None:
                super().__init__()
                self.close_calls = 0

            def close(self) -> None:  # pragma: no cover - the test asserts this is not called
                self.close_calls += 1

        transcriber = CloseTrackingTranscriber()

        process_input(input_path, output_dir=tmp_path / "run", transcriber=transcriber)

        assert transcriber.close_calls == 0

    def test_closes_owned_transcriber_on_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class CloseTrackingTranscriber(FakeTranscriber):
            def __init__(self) -> None:
                super().__init__()
                self.close_calls = 0

            def close(self) -> None:
                self.close_calls += 1

        transcriber = CloseTrackingTranscriber()
        monkeypatch.setattr(
            "webinar_transcriber.asr.WhisperCppTranscriber", lambda *args, **kwargs: transcriber
        )
        monkeypatch.setattr(
            "webinar_transcriber.processor.processor_asr.run_asr_pipeline",
            lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("asr failed")),
        )

        with pytest.raises(RuntimeError, match="asr failed"):
            process_input(FIXTURE_DIR / "sample-audio.mp3", output_dir=tmp_path / "run")

        assert transcriber.close_calls == 1

    def test_persists_intermediate_artifacts_on_failure(self, tmp_path: Path) -> None:
        class OneWindowTranscriber(FakeTranscriber):
            def transcribe_inference_windows(
                self,
                audio_samples: np.ndarray,
                windows: list[InferenceWindow],
                *,
                language: str | None = None,
                progress_callback: Callable[[float, int], None] | None = None,
            ) -> list[DecodedWindow]:
                del audio_samples, language
                if progress_callback is not None:
                    progress_callback(windows[0].end_sec, 1)
                return [
                    DecodedWindow(
                        window=windows[0],
                        text="Agenda review and project status update.",
                        language="en",
                        segments=[
                            TranscriptSegment(
                                id="segment-1",
                                text="Agenda review and project status update.",
                                start_sec=0.0,
                                end_sec=3.0,
                            )
                        ],
                    )
                ]

        output_dir = tmp_path / "failed-run"
        with (
            patch("webinar_transcriber.structure.build_report", side_effect=RuntimeError("boom")),
            patch(
                "webinar_transcriber.processor.asr.load_normalized_audio",
                return_value=(np.zeros(16_000, dtype=np.float32), 16_000),
            ),
            patch(
                "webinar_transcriber.processor.asr.detect_speech_regions",
                return_value=([SpeechRegion(start_sec=0.0, end_sec=3.0)], []),
            ),
            pytest.raises(RuntimeError, match="boom"),
        ):
            process_input(
                FIXTURE_DIR / "sample-audio.mp3",
                output_dir=output_dir,
                transcriber=OneWindowTranscriber(),
            )

        assert (output_dir / "metadata.json").exists()
        assert (output_dir / "transcript.json").exists()
        assert (output_dir / "asr" / "speech_regions.json").exists()
        assert (output_dir / "asr" / "decoded_windows.json").exists()
        diagnostics_payload = read_json(output_dir / "diagnostics.json")
        assert diagnostics_payload["status"] == "failed"
        assert diagnostics_payload["failed_stage"] == "structure"
        assert diagnostics_payload["error"] == "boom"
        assert not (output_dir / "report.json").exists()

    def test_forwards_vad_warnings_to_report_and_reporter(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        input_path = FIXTURE_DIR / "sample-audio.mp3"
        reporter = RecordingReporter()
        install_pipeline_runtime(
            monkeypatch,
            tmp_path,
            input_path=input_path,
            runtime=audio_runtime(vad_warnings=["Silero warning"]),
        )

        artifacts = process_input(
            input_path,
            output_dir=tmp_path / "warning-run",
            transcriber=FakeTranscriber(),
            reporter=reporter,
        )

        assert reporter.warnings == ["Silero warning"]
        assert artifacts.report.warnings == ["Silero warning"]
        assert artifacts.diagnostics.warnings == ["Silero warning"]

    def test_writes_video_scene_artifacts_and_frame_links(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        input_path = FIXTURE_DIR / "sample-video.mp4"
        scenes = [
            Scene(id="scene-1", start_sec=0.0, end_sec=0.9),
            Scene(id="scene-2", start_sec=0.9, end_sec=1.8),
        ]
        frames = [
            SlideFrame(
                id="frame-1",
                scene_id="scene-1",
                image_path=str(tmp_path / "frame-1.png"),
                timestamp_sec=0.5,
            ),
            SlideFrame(
                id="frame-2",
                scene_id="scene-2",
                image_path=str(tmp_path / "frame-2.png"),
                timestamp_sec=1.4,
            ),
        ]
        for frame in frames:
            Image.new("RGB", (8, 8), color="white").save(frame.image_path)
        install_pipeline_runtime(
            monkeypatch, tmp_path, input_path=input_path, runtime=video_runtime()
        )

        def fake_detect_scenes(*_args, progress_callback=None, **_kwargs) -> list[Scene]:
            assert progress_callback is not None
            progress_callback(1, 1)
            progress_callback(2, 2)
            return scenes

        def fake_extract_frames(*_args, progress_callback=None, warning_callback=None, **_kwargs):
            assert progress_callback is not None
            assert warning_callback is not None
            progress_callback()
            progress_callback()
            return frames

        monkeypatch.setattr("webinar_transcriber.video.detect_scenes", fake_detect_scenes)
        monkeypatch.setattr(
            "webinar_transcriber.video.extract_representative_frames", fake_extract_frames
        )

        artifacts = process_input(
            input_path,
            output_dir=tmp_path / "video-run",
            transcriber=FakeTranscriber(
                segments=[
                    TranscriptSegment(
                        id="segment-1", text="Intro slide.", start_sec=0.0, end_sec=0.8
                    ),
                    TranscriptSegment(
                        id="segment-2", text="Second slide.", start_sec=1.0, end_sec=1.7
                    ),
                ]
            ),
        )

        scenes_payload = read_json(artifacts.layout.scenes_path)
        report_payload = read_json(artifacts.layout.json_report_path)

        assert scenes_payload["scenes"] == [
            {"id": "scene-1", "start_sec": 0.0, "end_sec": 0.9},
            {"id": "scene-2", "start_sec": 0.9, "end_sec": 1.8},
        ]
        assert artifacts.diagnostics.item_counts["scenes"] == 2
        assert artifacts.diagnostics.item_counts["frames"] == 2
        assert artifacts.report.sections[0].image_path == frames[0].image_path
        assert report_payload["sections"][0]["image_path"] == frames[0].image_path

    def test_frame_extraction_warnings_reach_report_and_diagnostics(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        input_path = FIXTURE_DIR / "sample-video.mp4"
        reporter = RecordingReporter()
        install_pipeline_runtime(
            monkeypatch, tmp_path, input_path=input_path, runtime=video_runtime()
        )
        monkeypatch.setattr(
            "webinar_transcriber.video.detect_scenes",
            lambda *_args, **_kwargs: [Scene(id="scene-1", start_sec=0.0, end_sec=1.8)],
        )

        def fake_extract_frames(*_args, warning_callback=None, **_kwargs):
            assert warning_callback is not None
            warning_callback("Frame extraction failed")
            return []

        monkeypatch.setattr(
            "webinar_transcriber.video.extract_representative_frames", fake_extract_frames
        )

        artifacts = process_input(
            input_path,
            output_dir=tmp_path / "video-warning-run",
            transcriber=FakeTranscriber(),
            reporter=reporter,
        )

        assert reporter.warnings == ["Frame extraction failed"]
        assert artifacts.report.warnings == ["Frame extraction failed"]
        assert artifacts.diagnostics.warnings == ["Frame extraction failed"]

    def test_docx_export_warnings_reach_report_json_and_returned_artifacts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        input_path = FIXTURE_DIR / "sample-video.mp4"
        missing_image_path = tmp_path / "missing-frame.png"
        install_pipeline_runtime(
            monkeypatch, tmp_path, input_path=input_path, runtime=video_runtime()
        )
        monkeypatch.setattr(
            "webinar_transcriber.video.detect_scenes",
            lambda *_args, **_kwargs: [Scene(id="scene-1", start_sec=0.0, end_sec=1.8)],
        )
        monkeypatch.setattr(
            "webinar_transcriber.video.extract_representative_frames",
            lambda *_args, **_kwargs: [
                SlideFrame(
                    id="frame-1",
                    scene_id="scene-1",
                    image_path=str(missing_image_path),
                    timestamp_sec=1.0,
                )
            ],
        )

        artifacts = process_input(
            input_path,
            output_dir=tmp_path / "docx-warning-run",
            transcriber=FakeTranscriber(),
        )

        expected_warning = f"Section image does not exist: {missing_image_path}"
        report_payload = read_json(artifacts.layout.json_report_path)
        diagnostics_payload = read_json(artifacts.layout.diagnostics_path)
        assert artifacts.report.warnings == [expected_warning]
        assert report_payload["warnings"] == [expected_warning]
        assert diagnostics_payload["warnings"] == [expected_warning]


class TestProcessorSupport:
    def test_write_run_diagnostics_returns_none_without_layout(self) -> None:
        ctx = RunContext(reporter=BaseStageReporter())

        diagnostics = write_run_diagnostics(
            ctx,
            status="failed",
            failed_stage="prepare_run_dir",
            error="boom",
            asr_model="test-model",
            llm_enabled=False,
        )

        assert diagnostics is None

    def test_write_run_diagnostics_can_suppress_write_errors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ctx = RunContext(
            reporter=BaseStageReporter(),
            asr_pipeline=AsrPipelineDiagnostics(vad_enabled=True, threads=1),
            layout=RunLayout(run_dir=tmp_path),
        )

        def fail_write_text(self, *args, **kwargs):
            del self, args, kwargs
            raise OSError("readonly")

        monkeypatch.setattr(Path, "write_text", fail_write_text)

        diagnostics = write_run_diagnostics(
            ctx,
            status="failed",
            failed_stage="prepare_run_dir",
            error="boom",
            asr_model="test-model",
            llm_enabled=False,
            suppress_errors=True,
        )

        assert diagnostics is not None
        assert diagnostics.error == "boom"
        with pytest.raises(OSError, match="readonly"):
            write_run_diagnostics(
                ctx,
                status="failed",
                failed_stage="prepare_run_dir",
                error="boom",
                asr_model="test-model",
                llm_enabled=False,
            )

    def test_stage_records_timing_on_failure_without_finish_event(self) -> None:
        reporter = RecordingReporter()
        ctx = RunContext(reporter=reporter)

        with pytest.raises(RuntimeError, match="boom"), stage(ctx, "probe_media", "Probing media"):
            raise RuntimeError("boom")

        assert "probe_media" in ctx.stage_timings
        assert reporter.started == [("probe_media", "Probing media")]
        assert reporter.finished == []

    def test_window_transcription_stage_detail_reports_rtf(self) -> None:
        detail = window_transcription_stage_detail(
            window_count=1, total_duration_sec=12.5, elapsed_sec=2.5
        )

        assert detail == "1 window | RTF 5x"

    def test_asr_runtime_detail_uses_model_name_verbatim(self) -> None:
        transcriber = cast(
            "WhisperCppTranscriber",
            SimpleNamespace(model_name="/tmp/models/local-model.bin", device_name="cpu"),
        )

        assert asr_runtime_detail(transcriber) == "/tmp/models/local-model.bin | cpu"

    def test_transcriber_device_name_is_auto_before_runtime_is_prepared(self) -> None:
        assert WhisperCppTranscriber().device_name == "auto"

    def test_resolve_llm_processor_uses_environment_processor_details(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reporter = RecordingReporter()
        warnings: list[str] = []
        fake_processor = cast(
            "LLMProcessor", SimpleNamespace(provider_name="openai", model_name="gpt-test")
        )

        resolved_processor, runtime = resolve_llm_processor(
            enable_llm=True,
            llm_processor=fake_processor,
            reporter=reporter,
            warnings=warnings,
            llm_runtime=LLMRuntimeState(),
        )

        assert resolved_processor is fake_processor
        assert runtime.provider_name == "openai"
        assert runtime.model_name == "gpt-test"

        env_processor = cast(
            "LLMProcessor", SimpleNamespace(provider_name="anthropic", model_name="claude-test")
        )
        monkeypatch.setattr(
            "webinar_transcriber.processor.llm.build_llm_processor_from_env", lambda: env_processor
        )

        resolved_processor, runtime = resolve_llm_processor(
            enable_llm=True,
            llm_processor=None,
            reporter=reporter,
            warnings=warnings,
            llm_runtime=LLMRuntimeState(),
        )

        assert resolved_processor is env_processor
        assert runtime.provider_name == "anthropic"
        assert runtime.model_name == "claude-test"

        monkeypatch.setattr(
            "webinar_transcriber.processor.llm.build_llm_processor_from_env",
            lambda: (_ for _ in ()).throw(LLMConfigurationError("missing env")),
        )

        resolved_processor, runtime = resolve_llm_processor(
            enable_llm=True,
            llm_processor=None,
            reporter=reporter,
            warnings=warnings,
            llm_runtime=LLMRuntimeState(),
        )

        assert resolved_processor is None
        assert runtime.report_status == "fallback"
        assert warnings == ["missing env"]
        assert reporter.warnings == ["missing env"]


@pytest.fixture
def llm_success_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ProcessArtifacts, RecordingReporter]:
    input_path = FIXTURE_DIR / "sample-audio.mp3"
    install_pipeline_runtime(monkeypatch, tmp_path, input_path=input_path, runtime=audio_runtime())
    reporter = RecordingReporter()

    class FakeLLMProcessor:
        provider_name = "openai"
        model_name = "test-llm-model"

        def report_polish_plan(self, report: ReportDocument) -> LLMReportPolishPlan:
            return LLMReportPolishPlan(section_count=len(report.sections), worker_count=1)

        def polish_report_sections_with_progress(
            self, report, *, progress_callback: Callable[[int], None] | None = None
        ) -> LLMSectionPolishResult:
            assert progress_callback is not None
            progress_callback(len(report.sections))
            return LLMSectionPolishResult(
                section_tldrs={"section-1": "Updated section TL;DR."},
                section_transcripts={"section-1": EXPECTED_LLM_SECTION_TEXT},
                usage={"input_tokens": 8, "output_tokens": 3, "total_tokens": 11},
                warnings=[EXPECTED_LLM_WARNING],
            )

        def polish_report_metadata(
            self, report: ReportDocument, *, section_transcripts: dict[str, str]
        ) -> LLMReportMetadataResult:
            assert section_transcripts == {"section-1": EXPECTED_LLM_SECTION_TEXT}
            return LLMReportMetadataResult(
                summary=["Refined summary point."],
                action_items=["Send the draft by Friday."],
                section_titles={"section-1": "Refined Section Title"},
                usage={"input_tokens": 12, "output_tokens": 3, "total_tokens": 15},
            )

    artifacts = process_input(
        input_path,
        output_dir=tmp_path / "llm-run",
        transcriber=FakeTranscriber(),
        enable_llm=True,
        llm_processor=cast("LLMProcessor", FakeLLMProcessor()),
        reporter=reporter,
    )
    return artifacts, reporter


class TestProcessInputLlm:
    def test_applies_llm_outputs_to_report_and_diagnostics(
        self, llm_success_result: tuple[ProcessArtifacts, RecordingReporter]
    ) -> None:
        artifacts, reporter = llm_success_result

        assert artifacts.report.summary == ["Refined summary point."]
        assert artifacts.report.action_items == ["Send the draft by Friday."]
        assert artifacts.report.sections[0].title == "Refined Section Title"
        assert artifacts.report.sections[0].tldr == "Updated section TL;DR."
        assert artifacts.report.sections[0].transcript_text == EXPECTED_LLM_SECTION_TEXT
        assert artifacts.report.warnings == [EXPECTED_LLM_WARNING]
        assert reporter.warnings == [EXPECTED_LLM_WARNING]
        assert artifacts.diagnostics.llm_enabled
        assert artifacts.diagnostics.llm_model == "test-llm-model"
        assert artifacts.diagnostics.llm_report_status == "applied"
        assert artifacts.diagnostics.llm_report_usage == EXPECTED_LLM_USAGE
        assert "llm_report_sections" in artifacts.diagnostics.stage_durations_sec
        assert "llm_report_metadata" in artifacts.diagnostics.stage_durations_sec

        diagnostics_payload = read_json(artifacts.layout.diagnostics_path)
        assert diagnostics_payload["llm_report_usage"] == EXPECTED_LLM_USAGE
        assert diagnostics_payload["warnings"] == [EXPECTED_LLM_WARNING]

    def test_reports_all_sections_in_llm_progress(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        input_path = FIXTURE_DIR / "sample-audio.mp3"
        install_pipeline_runtime(
            monkeypatch,
            tmp_path,
            input_path=input_path,
            runtime=audio_runtime(
                duration_sec=10.0,
                speech_regions=[
                    SpeechRegion(start_sec=0.0, end_sec=2.5),
                    SpeechRegion(start_sec=8.0, end_sec=10.0),
                ],
            ),
        )
        reporter = RecordingReporter()

        class TwoSectionTranscriber(FakeTranscriber):
            def transcribe_inference_windows(
                self,
                audio_samples: np.ndarray,
                windows: list[InferenceWindow],
                *,
                language: str | None = None,
                progress_callback: Callable[[float, int], None] | None = None,
            ) -> list[DecodedWindow]:
                del audio_samples, language
                segments = [
                    [
                        TranscriptSegment(
                            id="segment-1", text="Opening topic.", start_sec=0.0, end_sec=1.5
                        )
                    ],
                    [
                        TranscriptSegment(
                            id="segment-2",
                            text="Later topic starts now.",
                            start_sec=8.0,
                            end_sec=9.5,
                        )
                    ],
                ]
                if progress_callback is not None:
                    progress_callback(windows[-1].end_sec, 2)
                return [
                    DecodedWindow(
                        window=window,
                        text=segments[index][0].text,
                        segments=segments[index],
                        language="en",
                    )
                    for index, window in enumerate(windows)
                ]

        class TwoSectionLLMProcessor:
            provider_name = "openai"
            model_name = "test-llm-model"

            def report_polish_plan(self, report: ReportDocument) -> LLMReportPolishPlan:
                return LLMReportPolishPlan(section_count=len(report.sections), worker_count=2)

            def polish_report_sections_with_progress(
                self, report, *, progress_callback: Callable[[int], None] | None = None
            ) -> LLMSectionPolishResult:
                assert progress_callback is not None
                progress_callback(1)
                progress_callback(1)
                return LLMSectionPolishResult(
                    section_tldrs={
                        section.id: f"{section.title} recap" for section in report.sections
                    },
                    section_transcripts={
                        section.id: section.transcript_text for section in report.sections
                    },
                    usage={"total_tokens": 2},
                )

            def polish_report_metadata(
                self, report: ReportDocument, *, section_transcripts: dict[str, str]
            ) -> LLMReportMetadataResult:
                return LLMReportMetadataResult(
                    summary=[],
                    action_items=[],
                    section_titles={report.sections[0].id: "Renamed first section"},
                    usage={"total_tokens": 3},
                )

        artifacts = process_input(
            input_path,
            output_dir=tmp_path / "llm-two-section-run",
            transcriber=TwoSectionTranscriber(),
            enable_llm=True,
            llm_processor=cast("LLMProcessor", TwoSectionLLMProcessor()),
            reporter=reporter,
        )

        assert len(artifacts.report.sections) == 2
        assert artifacts.report.sections[0].title == "Renamed first section"
        assert artifacts.diagnostics.llm_report_usage == {"total_tokens": 5}
        assert ("start", "llm_report_sections", 2.0, None) in reporter.progress
        assert [
            event
            for event in reporter.progress
            if event[0] == "advance" and event[1] == "llm_report_sections"
        ] == [
            ("advance", "llm_report_sections", 1.0, None),
            ("advance", "llm_report_sections", 1.0, None),
        ]

    def test_warns_when_llm_configuration_is_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        input_path = FIXTURE_DIR / "sample-audio.mp3"
        install_pipeline_runtime(
            monkeypatch, tmp_path, input_path=input_path, runtime=audio_runtime()
        )
        reporter = RecordingReporter()
        monkeypatch.setattr(
            "webinar_transcriber.processor.llm.build_llm_processor_from_env",
            lambda: (_ for _ in ()).throw(LLMConfigurationError("missing llm config")),
        )

        artifacts = process_input(
            input_path,
            output_dir=tmp_path / "missing-llm-run",
            transcriber=FakeTranscriber(),
            enable_llm=True,
            reporter=reporter,
        )

        assert reporter.warnings == ["missing llm config"]
        assert artifacts.report.warnings == ["missing llm config"]
        assert artifacts.diagnostics.llm_enabled
        assert artifacts.diagnostics.llm_report_status == "fallback"
        assert artifacts.diagnostics.llm_model is None

    def test_falls_back_when_section_polish_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        input_path = FIXTURE_DIR / "sample-audio.mp3"
        install_pipeline_runtime(
            monkeypatch, tmp_path, input_path=input_path, runtime=audio_runtime()
        )
        reporter = RecordingReporter()

        class FailingSectionLLMProcessor:
            provider_name = "openai"
            model_name = "test-llm-model"

            def report_polish_plan(self, report: ReportDocument) -> LLMReportPolishPlan:
                return LLMReportPolishPlan(section_count=len(report.sections), worker_count=1)

            def polish_report_sections_with_progress(
                self, report, *, progress_callback: Callable[[int], None] | None = None
            ) -> LLMSectionPolishResult:
                del report, progress_callback
                raise LLMProcessingError("section polish failed")

            def polish_report_metadata(
                self, report: ReportDocument, *, section_transcripts: dict[str, str]
            ) -> LLMReportMetadataResult:
                raise AssertionError("metadata polish should not run")

        artifacts = process_input(
            input_path,
            output_dir=tmp_path / "section-fallback-run",
            transcriber=FakeTranscriber(),
            enable_llm=True,
            llm_processor=cast("LLMProcessor", FailingSectionLLMProcessor()),
            reporter=reporter,
        )

        assert reporter.warnings == ["section polish failed"]
        assert artifacts.diagnostics.llm_report_status == "fallback"
        assert artifacts.report.summary == []
        assert artifacts.report.action_items == []

    def test_metadata_polish_failure_keeps_section_timing_and_falls_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        input_path = FIXTURE_DIR / "sample-audio.mp3"
        install_pipeline_runtime(
            monkeypatch, tmp_path, input_path=input_path, runtime=audio_runtime()
        )
        reporter = RecordingReporter()

        class MetadataFailingLLMProcessor:
            provider_name = "openai"
            model_name = "test-llm-model"

            def report_polish_plan(self, report: ReportDocument) -> LLMReportPolishPlan:
                return LLMReportPolishPlan(section_count=len(report.sections), worker_count=1)

            def polish_report_sections_with_progress(
                self, report, *, progress_callback: Callable[[int], None] | None = None
            ) -> LLMSectionPolishResult:
                assert progress_callback is not None
                progress_callback(len(report.sections))
                return LLMSectionPolishResult(
                    section_tldrs={},
                    section_transcripts={
                        section.id: section.transcript_text for section in report.sections
                    },
                    usage={"total_tokens": 11},
                )

            def polish_report_metadata(
                self, report: ReportDocument, *, section_transcripts: dict[str, str]
            ) -> LLMReportMetadataResult:
                del report, section_transcripts
                raise LLMProcessingError("metadata failed")

        artifacts = process_input(
            input_path,
            output_dir=tmp_path / "metadata-fallback-run",
            transcriber=FakeTranscriber(),
            enable_llm=True,
            llm_processor=cast("LLMProcessor", MetadataFailingLLMProcessor()),
            reporter=reporter,
        )

        assert reporter.warnings == ["metadata failed"]
        assert artifacts.diagnostics.llm_report_status == "fallback"
        assert artifacts.diagnostics.llm_report_usage == {}
        assert "llm_report_sections" in artifacts.diagnostics.stage_durations_sec
        assert "llm_report_metadata" in artifacts.diagnostics.stage_durations_sec
