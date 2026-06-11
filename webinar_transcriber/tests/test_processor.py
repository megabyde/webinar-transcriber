"""Behavior-focused tests for the processor pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock, patch

import numpy as np
import pytest
from rich.console import Console

from webinar_transcriber.diagnostics import write_run_diagnostics
from webinar_transcriber.llm import (
    LlmProcessingError,
    LlmReportMetadataResult,
    LlmReportPolishPlan,
    LlmSectionPolishResult,
)
from webinar_transcriber.models import (
    DecodedWindow,
    InferenceWindow,
    ReportDocument,
    Scene,
    SceneFrame,
    SpeakerTurn,
    SpeechRegion,
    TranscriptSegment,
)
from webinar_transcriber.paths import RunLayout
from webinar_transcriber.processor import ProcessArtifacts, RunContext, plan_inference_windows
from webinar_transcriber.processor import process_input as _process_input
from webinar_transcriber.tests.conftest import (
    FakeTranscriber,
    RecordingStageReporter,
    audio_runtime,
    install_pipeline_runtime,
    install_video_scene_runtime,
    video_runtime,
)
from webinar_transcriber.ui import StageReporter

if TYPE_CHECKING:
    from collections.abc import Callable

    from webinar_transcriber.diarization import SherpaOnnxDiarizer
    from webinar_transcriber.llm.processor import InstructorLLMProcessor

FIXTURE_DIR = Path(__file__).parent / "fixtures"
EXPECTED_LLM_WARNING = (
    "Section polish response returned an empty transcript text for section-1; kept original text."
)
EXPECTED_LLM_SECTION_TEXT = (
    "Agenda review, and project status update.\n\nNext step: please send the draft by Friday."
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def silent_reporter() -> StageReporter:
    return StageReporter(console=Console(quiet=True))


def process_input(
    *args: Any,
    threads: int = 4,
    asr_model: str | None = None,
    language: str | None = None,
    keep_audio: bool = False,
    llm_processor: InstructorLLMProcessor | None = None,
    diarize_speakers: int | None = None,
    diarizer: SherpaOnnxDiarizer | None = None,
    **kwargs: Any,
) -> ProcessArtifacts:
    return _process_input(
        *args,
        threads=threads,
        asr_model=asr_model,
        language=language,
        keep_audio=keep_audio,
        llm_processor=llm_processor,
        diarizer=diarizer,
        diarization_speaker_count=diarize_speakers,
        **kwargs,
    )


class ConfigurableLLMProcessor:
    provider_name = "openai"
    model_name = "test-llm-model"

    def __init__(
        self,
        *,
        section_result: LlmSectionPolishResult | Callable[[ReportDocument], LlmSectionPolishResult],
        metadata_result: (
            LlmReportMetadataResult
            | Callable[[ReportDocument, dict[str, str]], LlmReportMetadataResult]
            | None
        ) = None,
        section_error: LlmProcessingError | None = None,
        metadata_error: LlmProcessingError | None = None,
        section_progress: list[int] | None = None,
        worker_count: int = 1,
    ) -> None:
        if isinstance(section_result, LlmSectionPolishResult):
            self._section_result = lambda _report: section_result
        else:
            self._section_result = section_result
        metadata_result = metadata_result or LlmReportMetadataResult(
            summary=[], action_items=[], section_titles={}
        )
        if isinstance(metadata_result, LlmReportMetadataResult):
            self._metadata_result = lambda _report, _section_transcripts: metadata_result
        else:
            self._metadata_result = metadata_result
        self._section_error = section_error
        self._metadata_error = metadata_error
        self._section_progress = section_progress
        self._worker_count = worker_count

    def report_polish_plan(self, report: ReportDocument) -> LlmReportPolishPlan:
        return LlmReportPolishPlan(
            section_count=len(report.sections), worker_count=self._worker_count
        )

    def polish_report_sections_with_progress(
        self, report: ReportDocument, *, progress_callback: Callable[[int], None] | None = None
    ) -> LlmSectionPolishResult:
        if self._section_error is not None:
            raise self._section_error
        if progress_callback is not None:
            for completed_count in self._section_progress or [len(report.sections)]:
                progress_callback(completed_count)
        return self._section_result(report)

    def polish_report_metadata(
        self, report: ReportDocument, *, section_transcripts: dict[str, str]
    ) -> LlmReportMetadataResult:
        if self._metadata_error is not None:
            raise self._metadata_error
        return self._metadata_result(report, section_transcripts)


class FakeDiarizer:
    system_info = "fake sherpa"

    def __init__(self, turns: list[SpeakerTurn]) -> None:
        self.turns = turns
        self.calls: list[int] = []
        self.prepare_calls: list[int | None] = []

    def prepare(self, *, speaker_count: int | None) -> None:
        self.prepare_calls.append(speaker_count)

    def diarize(
        self, samples: np.ndarray, *, progress_callback: Callable[[int, int], None] | None = None
    ) -> list[SpeakerTurn]:
        self.calls.append(len(samples))
        if progress_callback is not None:
            progress_callback(1, 0)
            progress_callback(1, 2)
            progress_callback(2, 2)
        return self.turns


def run_basic_audio_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ProcessArtifacts, RecordingStageReporter, FakeTranscriber]:
    input_path = FIXTURE_DIR / "sample-audio.mp3"
    reporter = RecordingStageReporter()
    transcriber = FakeTranscriber()
    install_pipeline_runtime(monkeypatch, tmp_path, input_path=input_path, runtime=audio_runtime())

    artifacts = process_input(
        input_path,
        output_dir=tmp_path / "run",
        language="en",
        transcriber=transcriber,
        reporter=reporter,
    )
    return artifacts, reporter, transcriber


class TestProcessInput:
    def test_processes_audio_into_report_sections(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        artifacts, reporter, transcriber = run_basic_audio_pipeline(tmp_path, monkeypatch)

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

    def test_processes_audio_writes_expected_artifacts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        artifacts, _, _ = run_basic_audio_pipeline(tmp_path, monkeypatch)

        assert artifacts.layout.metadata_path.exists()
        assert artifacts.layout.transcript_path.exists()
        assert artifacts.layout.markdown_report_path.exists()
        assert artifacts.layout.docx_report_path.stat().st_size > 0
        assert artifacts.layout.json_report_path.exists()
        assert artifacts.layout.speech_regions_path.exists()
        assert artifacts.layout.decoded_windows_path.exists()
        assert not artifacts.layout.transcription_audio_path().exists()
        assert not artifacts.layout.diarization_path.exists()

        markdown = artifacts.layout.markdown_report_path.read_text(encoding="utf-8")
        assert "# Sample Audio" in markdown
        assert "Agenda review and project status update." in markdown

    def test_processes_audio_writes_metadata_and_diagnostics(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        artifacts, _, _ = run_basic_audio_pipeline(tmp_path, monkeypatch)

        diagnostics_payload = read_json(artifacts.layout.diagnostics_path)
        metadata_payload = read_json(artifacts.layout.metadata_path)

        assert metadata_payload["media_type"] == "audio"
        assert metadata_payload["duration_sec"] == 6.0
        assert diagnostics_payload["llm"] is None
        assert diagnostics_payload["asr_pipeline"]["backend"] == "whisper.cpp"
        assert diagnostics_payload["asr_pipeline"]["model"] == "test-model"
        assert diagnostics_payload["item_counts"]["windows"] == 1
        assert diagnostics_payload["item_counts"]["report_sections"] == 1
        assert artifacts.diagnostics.asr_pipeline is not None
        assert artifacts.diagnostics.asr_pipeline.system_info == "CPU = 1"
        assert artifacts.diagnostics.diarization is None

    def test_processes_audio_reports_progress(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _, reporter, _ = run_basic_audio_pipeline(tmp_path, monkeypatch)

        assert ("start", "prepare_transcription_audio", 6.0, None) in reporter.progress
        assert ("advance", "prepare_transcription_audio", 6.0, None) in reporter.progress
        assert ("start", "vad", 1.0, "0 regions") in reporter.progress
        assert ("advance", "vad", 1.0, "1 region") in reporter.progress
        vad_finished = [detail for stage, detail in reporter.finished if stage == "vad"]
        assert len(vad_finished) == 1
        assert vad_finished[0].startswith("1 region | RTF ")
        assert ("start", "transcribe", 1.0, "0 segments") in reporter.progress
        assert ("advance", "transcribe", 1.0, "2 segments") in reporter.progress

    def test_diarizes_transcript_and_report_when_enabled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        input_path = FIXTURE_DIR / "sample-audio.mp3"
        transcriber = FakeTranscriber()
        reporter = RecordingStageReporter()
        diarizer = FakeDiarizer([
            SpeakerTurn(start_sec=0.0, end_sec=3.2, speaker="S1"),
            SpeakerTurn(start_sec=3.2, end_sec=6.0, speaker="S2"),
        ])
        install_pipeline_runtime(
            monkeypatch, tmp_path, input_path=input_path, runtime=audio_runtime(duration_sec=6.0)
        )

        artifacts = process_input(
            input_path,
            output_dir=tmp_path / "diarized-run",
            transcriber=transcriber,
            diarize_speakers=4,
            diarizer=diarizer,  # type: ignore
            reporter=reporter,
        )

        transcript_payload = read_json(artifacts.layout.transcript_path)
        diarization_payload = read_json(artifacts.layout.diarization_path)

        assert diarizer.calls == [16_000]
        assert diarizer.prepare_calls == [4]
        assert [segment["speaker"] for segment in transcript_payload["segments"]] == ["S1", "S2"]
        assert diarization_payload == [
            {"start_sec": 0.0, "end_sec": 3.2, "speaker": "S1"},
            {"start_sec": 3.2, "end_sec": 6.0, "speaker": "S2"},
        ]
        assert (
            "**S1:** Agenda review and project status update."
            in artifacts.report.sections[0].transcript_text
        )
        markdown = artifacts.layout.markdown_report_path.read_text(encoding="utf-8")
        assert "**S2:** Next step please send the draft by Friday." in markdown
        assert artifacts.diagnostics.diarization is not None
        assert artifacts.diagnostics.diarization.speaker_count == 2
        assert artifacts.diagnostics.diarization.turn_count == 2
        assert ("start", "diarize", 100.0, "preparing model") in reporter.progress
        assert ("advance", "diarize", 1.0, "analyzing audio") in reporter.progress
        assert ("advance", "diarize", 64.0, "embedding speakers") in reporter.progress
        diarize_progress = [
            detail
            for action, stage_key, _, detail in reporter.progress
            if action == "advance" and stage_key == "diarize"
        ]
        assert any(
            detail and detail.startswith("2 speakers | 2 turns | RTF ")
            for detail in diarize_progress
        )
        diarize_finished = [
            detail for stage_key, detail in reporter.finished if stage_key == "diarize"
        ]
        assert len(diarize_finished) == 1
        assert diarize_finished[0].startswith("2 speakers | 2 turns | RTF ")

    def test_prepares_diarization_model_before_progress(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        input_path = FIXTURE_DIR / "sample-audio.mp3"
        transcriber = FakeTranscriber()
        reporter = RecordingStageReporter()

        class PreparingDiarizer:
            system_info = "fake sherpa"

            def __init__(self) -> None:
                self.prepare_calls: list[int | None] = []

            def prepare(self, *, speaker_count: int | None) -> None:
                self.prepare_calls.append(speaker_count)

            def diarize(
                self,
                samples: np.ndarray,
                *,
                progress_callback: Callable[[int, int], None] | None = None,
            ) -> list[SpeakerTurn]:
                del samples
                if progress_callback is not None:
                    progress_callback(2, 2)
                return [SpeakerTurn(start_sec=0.0, end_sec=2.0, speaker="S1")]

        diarizer = PreparingDiarizer()
        install_pipeline_runtime(
            monkeypatch, tmp_path, input_path=input_path, runtime=audio_runtime(duration_sec=6.0)
        )

        process_input(
            input_path,
            output_dir=tmp_path / "diarized-default-run",
            transcriber=transcriber,
            diarize_speakers=1,
            diarizer=diarizer,  # type: ignore
            reporter=reporter,
        )

        assert diarizer.prepare_calls == [1]
        assert any(key == "diarize" for key, _ in reporter.finished)

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
        assert [window["window"] for window in decoded_windows_payload] == [
            {"id": "window-1", "region_index": 0, "start_sec": 0.0, "end_sec": 1.2},
            {"id": "window-2", "region_index": 1, "start_sec": 2.1, "end_sec": 3.0},
        ]
        assert transcriber.windows_seen == [
            InferenceWindow(id="window-1", region_index=0, start_sec=0.0, end_sec=1.2),
            InferenceWindow(id="window-2", region_index=1, start_sec=2.1, end_sec=3.0),
        ]
        assert artifacts.diagnostics.item_counts["vad_regions"] == 2
        assert artifacts.diagnostics.item_counts["windows"] == 2
        assert artifacts.diagnostics.asr_pipeline is not None
        assert artifacts.diagnostics.asr_pipeline.average_window_duration_sec == pytest.approx(1.05)

    def test_splits_long_speech_region_into_overlapping_inference_windows(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        input_path = FIXTURE_DIR / "sample-audio.mp3"
        transcriber = FakeTranscriber()
        install_pipeline_runtime(
            monkeypatch,
            tmp_path,
            input_path=input_path,
            runtime=audio_runtime(
                duration_sec=65.0, speech_regions=[SpeechRegion(start_sec=0.0, end_sec=65.0)]
            ),
        )

        artifacts = process_input(
            input_path, output_dir=tmp_path / "long-window-run", transcriber=transcriber
        )

        assert transcriber.windows_seen == [
            InferenceWindow(id="window-1", region_index=0, start_sec=0.0, end_sec=28.0),
            InferenceWindow(id="window-2", region_index=0, start_sec=26.0, end_sec=54.0),
            InferenceWindow(id="window-3", region_index=0, start_sec=52.0, end_sec=65.0),
        ]
        assert artifacts.diagnostics.item_counts["vad_regions"] == 1
        assert artifacts.diagnostics.item_counts["windows"] == 3
        assert artifacts.diagnostics.asr_pipeline is not None
        assert artifacts.diagnostics.asr_pipeline.average_window_duration_sec == pytest.approx(23.0)

    def test_inference_window_planning_skips_empty_regions(self) -> None:
        windows = plan_inference_windows([
            SpeechRegion(start_sec=1.0, end_sec=1.0),
            SpeechRegion(start_sec=2.0, end_sec=1.5),
            SpeechRegion(start_sec=3.0, end_sec=4.0),
        ])

        assert windows == [
            InferenceWindow(id="window-1", region_index=2, start_sec=3.0, end_sec=4.0)
        ]

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
        preserve_calls: list[tuple[Path, Path]] = []

        def fake_preserve_audio(audio_path: Path, output_path: Path):
            preserve_calls.append((audio_path, output_path))
            output_path.write_text("mp3", encoding="utf-8")
            return output_path

        monkeypatch.setattr(
            "webinar_transcriber.processor.preserve_transcription_audio", fake_preserve_audio
        )

        artifacts = process_input(
            input_path, output_dir=tmp_path / "run", transcriber=FakeTranscriber(), keep_audio=True
        )

        kept_audio_path = artifacts.layout.transcription_audio_path()
        assert kept_audio_path.exists()
        assert kept_audio_path.suffix == ".mp3"
        assert preserve_calls == [(tmp_path / "sample-audio.wav", kept_audio_path)]

    @pytest.mark.slow
    def test_keeps_normalized_audio_artifact_with_real_media_prep(self, tmp_path: Path) -> None:
        artifacts = process_input(
            FIXTURE_DIR / "sample-audio.mp3",
            output_dir=tmp_path / "real-run",
            transcriber=FakeTranscriber(),
            keep_audio=True,
        )

        kept_audio_path = artifacts.layout.transcription_audio_path()
        assert kept_audio_path.exists()
        assert kept_audio_path.suffix == ".mp3"
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
                        id="segment-1", text="Opening slide.", start_sec=0.0, end_sec=0.8
                    ),
                    TranscriptSegment(
                        id="segment-2", text="Closing slide.", start_sec=0.8, end_sec=1.8
                    ),
                ]
            ),
        )

        assert artifacts.layout.metadata_path.exists()
        assert artifacts.layout.scenes_path.exists()
        assert artifacts.layout.frames_dir.exists()
        assert artifacts.layout.markdown_report_path.exists()
        assert artifacts.layout.docx_report_path.stat().st_size > 0
        assert artifacts.layout.json_report_path.exists()
        assert artifacts.diagnostics.item_counts["scenes"] >= 1
        assert artifacts.diagnostics.item_counts["frames"] >= 1
        assert artifacts.report.sections
        assert any(section.image_path for section in artifacts.report.sections)

    def test_closes_caller_supplied_transcriber(
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

            def close(self) -> None:
                self.close_calls += 1

        transcriber = CloseTrackingTranscriber()

        process_input(input_path, output_dir=tmp_path / "run", transcriber=transcriber)

        assert transcriber.close_calls == 1

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
            "webinar_transcriber.processor.WhisperCppTranscriber",
            lambda *args, **kwargs: transcriber,
        )
        monkeypatch.setattr(
            "webinar_transcriber.processor._run_asr_pipeline",
            Mock(side_effect=RuntimeError("asr failed")),
        )

        with pytest.raises(RuntimeError, match="asr failed"):
            process_input(FIXTURE_DIR / "sample-audio.mp3", output_dir=tmp_path / "run")

        assert transcriber.close_calls == 1

    def test_passes_threads_to_owned_transcriber(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        input_path = FIXTURE_DIR / "sample-audio.mp3"
        install_pipeline_runtime(
            monkeypatch, tmp_path, input_path=input_path, runtime=audio_runtime()
        )
        transcriber = FakeTranscriber()
        transcriber_kwargs: dict[str, object] = {}

        def transcriber_factory(*_args: object, **kwargs: object) -> FakeTranscriber:
            transcriber_kwargs.update(kwargs)
            return transcriber

        monkeypatch.setattr(
            "webinar_transcriber.processor.WhisperCppTranscriber", transcriber_factory
        )

        process_input(input_path, output_dir=tmp_path / "run", threads=3)

        assert transcriber_kwargs["threads"] == 3

    def test_persists_intermediate_artifacts_on_failure(self, tmp_path: Path) -> None:
        class OneWindowTranscriber(FakeTranscriber):
            def transcribe_inference_windows(
                self,
                audio_samples: np.ndarray,
                windows: list[InferenceWindow],
                *,
                language: str | None = None,
                progress_callback: Callable[[int, int], None] | None = None,
                warning_callback: Callable[[str], None] | None = None,
            ) -> list[DecodedWindow]:
                del audio_samples, language, warning_callback
                if progress_callback is not None:
                    progress_callback(1, 1)
                return [
                    DecodedWindow(
                        window=windows[0],
                        text="Agenda review and project status update.",
                        detected_language="en",
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
            patch("webinar_transcriber.processor.build_report", side_effect=RuntimeError("boom")),
            patch(
                "webinar_transcriber.processor.load_normalized_audio",
                return_value=np.zeros(16_000, dtype=np.float32),
            ),
            patch(
                "webinar_transcriber.processor.detect_speech_regions",
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
        assert diagnostics_payload["item_counts"]["transcript_segments"] == 1
        assert diagnostics_payload["item_counts"]["windows"] == 1
        assert diagnostics_payload["error"] == "boom"
        assert not (output_dir / "report.json").exists()

    def test_forwards_vad_warnings_to_report_and_reporter(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        input_path = FIXTURE_DIR / "sample-audio.mp3"
        reporter = RecordingStageReporter()
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
        run_dir = tmp_path / "video-run"
        frames_dir = run_dir / "frames"
        scenes = [
            Scene(id="scene-1", start_sec=0.0, end_sec=0.9),
            Scene(id="scene-2", start_sec=0.9, end_sec=1.8),
        ]
        frames = [
            SceneFrame(
                id="frame-1",
                scene_id="scene-1",
                image_path=str(frames_dir / "scene-1.png"),
                timestamp_sec=0.5,
            ),
            SceneFrame(
                id="frame-2",
                scene_id="scene-2",
                image_path=str(frames_dir / "scene-2.png"),
                timestamp_sec=1.4,
            ),
        ]
        install_pipeline_runtime(
            monkeypatch, tmp_path, input_path=input_path, runtime=video_runtime()
        )
        install_video_scene_runtime(monkeypatch, scenes=scenes, frames=frames)

        artifacts = process_input(
            input_path,
            output_dir=run_dir,
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

        assert scenes_payload == [
            {"id": "scene-1", "start_sec": 0.0, "end_sec": 0.9},
            {"id": "scene-2", "start_sec": 0.9, "end_sec": 1.8},
        ]
        assert artifacts.diagnostics.item_counts["scenes"] == 2
        assert artifacts.diagnostics.item_counts["frames"] == 2
        assert artifacts.report.sections[0].image_path == "frames/scene-1.png"
        assert report_payload["sections"][0]["image_path"] == "frames/scene-1.png"

    def test_frame_extraction_warnings_reach_report_and_diagnostics(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        input_path = FIXTURE_DIR / "sample-video.mp4"
        reporter = RecordingStageReporter()
        install_pipeline_runtime(
            monkeypatch, tmp_path, input_path=input_path, runtime=video_runtime()
        )
        install_video_scene_runtime(
            monkeypatch,
            scenes=[Scene(id="scene-1", start_sec=0.0, end_sec=1.8)],
            frame_warning="Frame extraction failed",
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
            "webinar_transcriber.processor.detect_scenes",
            lambda *_args, **_kwargs: [Scene(id="scene-1", start_sec=0.0, end_sec=1.8)],
        )
        monkeypatch.setattr(
            "webinar_transcriber.processor.extract_representative_frames",
            lambda *_args, **_kwargs: [
                SceneFrame(
                    id="frame-1",
                    scene_id="scene-1",
                    image_path=str(missing_image_path),
                    timestamp_sec=1.0,
                )
            ],
        )

        artifacts = process_input(
            input_path, output_dir=tmp_path / "docx-warning-run", transcriber=FakeTranscriber()
        )

        expected_warning = f"Section image does not exist: {missing_image_path}"
        report_payload = read_json(artifacts.layout.json_report_path)
        diagnostics_payload = read_json(artifacts.layout.diagnostics_path)
        assert artifacts.report.warnings == [expected_warning]
        assert report_payload["warnings"] == [expected_warning]
        assert diagnostics_payload["warnings"] == [expected_warning]


class TestProcessorSupport:
    def test_write_run_diagnostics_can_suppress_write_errors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        layout = RunLayout(run_dir=tmp_path)
        ctx = RunContext(reporter=silent_reporter())

        def fail_write_text(self, *args, **kwargs):
            del self, args, kwargs
            raise OSError("readonly")

        monkeypatch.setattr(Path, "write_text", fail_write_text)

        diagnostics = write_run_diagnostics(
            layout,
            ctx,
            status="failed",
            failed_stage="probe_media",
            error="boom",
            suppress_errors=True,
        )

        assert diagnostics.error == "boom"
        assert diagnostics.asr_pipeline is None
        assert diagnostics.item_counts["vad_regions"] == 0
        assert diagnostics.item_counts["windows"] == 0
        with pytest.raises(OSError, match="readonly"):
            write_run_diagnostics(
                layout,
                ctx,
                status="failed",
                failed_stage="probe_media",
                error="boom",
            )

    def test_stage_records_timing_on_failure_without_finish_event(self) -> None:
        reporter = RecordingStageReporter()
        ctx = RunContext(reporter=reporter)

        with pytest.raises(RuntimeError, match="boom"), ctx.stage("probe_media", "Probing media"):
            raise RuntimeError("boom")

        assert "probe_media" in ctx.stage_timings
        assert reporter.started == [("probe_media", "Probing media")]
        assert reporter.finished == []

    def test_transcriber_device_name_is_auto_before_runtime_is_prepared(self) -> None:
        from webinar_transcriber.asr import WhisperCppTranscriber

        assert WhisperCppTranscriber(threads=4).device_name == "auto"


@pytest.fixture
def llm_success_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ProcessArtifacts, RecordingStageReporter]:
    input_path = FIXTURE_DIR / "sample-audio.mp3"
    install_pipeline_runtime(monkeypatch, tmp_path, input_path=input_path, runtime=audio_runtime())
    reporter = RecordingStageReporter()

    def polished_metadata(
        _report: ReportDocument, section_transcripts: dict[str, str]
    ) -> LlmReportMetadataResult:
        assert section_transcripts == {"section-1": EXPECTED_LLM_SECTION_TEXT}
        return LlmReportMetadataResult(
            summary=["Refined summary point."],
            action_items=["Send the draft by Friday."],
            section_titles={"section-1": "Refined Section Title"},
        )

    artifacts = process_input(
        input_path,
        output_dir=tmp_path / "llm-run",
        transcriber=FakeTranscriber(),
        llm_processor=ConfigurableLLMProcessor(  # type: ignore
            section_result=LlmSectionPolishResult(
                section_tldrs={"section-1": "Updated section TL;DR."},
                section_transcripts={"section-1": EXPECTED_LLM_SECTION_TEXT},
                response_metadata=[
                    {"stage": "section_polish", "section_id": "section-1", "finish_reason": "stop"}
                ],
                warnings=[EXPECTED_LLM_WARNING],
            ),
            metadata_result=polished_metadata,
        ),
        reporter=reporter,
    )
    return artifacts, reporter


class TestProcessInputLlm:
    def test_applies_llm_outputs_to_report_and_diagnostics(
        self, llm_success_result: tuple[ProcessArtifacts, RecordingStageReporter]
    ) -> None:
        artifacts, reporter = llm_success_result

        assert artifacts.report.summary == ["Refined summary point."]
        assert artifacts.report.action_items == ["Send the draft by Friday."]
        assert artifacts.report.sections[0].title == "Refined Section Title"
        assert artifacts.report.sections[0].tldr == "Updated section TL;DR."
        assert artifacts.report.sections[0].transcript_text == EXPECTED_LLM_SECTION_TEXT
        assert artifacts.report.warnings == [EXPECTED_LLM_WARNING]
        assert reporter.warnings == [EXPECTED_LLM_WARNING]
        assert artifacts.diagnostics.llm is not None
        assert artifacts.diagnostics.llm.model == "test-llm-model"
        assert artifacts.diagnostics.llm.report_status == "applied"
        assert artifacts.diagnostics.llm.response_metadata == [
            {"stage": "section_polish", "section_id": "section-1", "finish_reason": "stop"}
        ]
        assert "llm_report_sections" in artifacts.diagnostics.stage_durations_sec
        assert "llm_report_metadata" in artifacts.diagnostics.stage_durations_sec

        diagnostics_payload = read_json(artifacts.layout.diagnostics_path)
        assert diagnostics_payload["llm"]["response_metadata"] == [
            {"stage": "section_polish", "section_id": "section-1", "finish_reason": "stop"}
        ]
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
        reporter = RecordingStageReporter()

        class TwoSectionTranscriber(FakeTranscriber):
            def transcribe_inference_windows(
                self,
                audio_samples: np.ndarray,
                windows: list[InferenceWindow],
                *,
                language: str | None = None,
                progress_callback: Callable[[int, int], None] | None = None,
                warning_callback: Callable[[str], None] | None = None,
            ) -> list[DecodedWindow]:
                del audio_samples, language, warning_callback
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
                    progress_callback(len(windows), 2)
                return [
                    DecodedWindow(
                        window=window,
                        text=segments[index][0].text,
                        segments=segments[index],
                        detected_language="en",
                    )
                    for index, window in enumerate(windows)
                ]

        def section_polish(report: ReportDocument) -> LlmSectionPolishResult:
            return LlmSectionPolishResult(
                section_tldrs={section.id: f"{section.title} recap" for section in report.sections},
                section_transcripts={
                    section.id: section.transcript_text for section in report.sections
                },
            )

        def metadata_polish(
            report: ReportDocument, _section_transcripts: dict[str, str]
        ) -> LlmReportMetadataResult:
            return LlmReportMetadataResult(
                summary=[],
                action_items=[],
                section_titles={report.sections[0].id: "Renamed first section"},
            )

        artifacts = process_input(
            input_path,
            output_dir=tmp_path / "llm-two-section-run",
            transcriber=TwoSectionTranscriber(),
            llm_processor=ConfigurableLLMProcessor(  # type: ignore
                section_result=section_polish,
                metadata_result=metadata_polish,
                section_progress=[1, 1],
                worker_count=2,
            ),
            reporter=reporter,
        )

        assert len(artifacts.report.sections) == 2
        assert artifacts.report.sections[0].title == "Renamed first section"
        section_starts = [
            event
            for event in reporter.progress
            if event[0] == "start" and event[1] == "llm_report_sections"
        ]
        assert len(section_starts) == 1
        assert section_starts[0][2] == 2.0
        non_zero_advances = [
            event
            for event in reporter.progress
            if event[0] == "advance" and event[1] == "llm_report_sections" and event[2] > 0
        ]
        assert non_zero_advances == [
            ("advance", "llm_report_sections", 1.0, None),
            ("advance", "llm_report_sections", 1.0, None),
        ]

    def test_falls_back_when_section_polish_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        input_path = FIXTURE_DIR / "sample-audio.mp3"
        install_pipeline_runtime(
            monkeypatch, tmp_path, input_path=input_path, runtime=audio_runtime()
        )
        reporter = RecordingStageReporter()

        artifacts = process_input(
            input_path,
            output_dir=tmp_path / "section-fallback-run",
            transcriber=FakeTranscriber(),
            llm_processor=ConfigurableLLMProcessor(  # type: ignore
                section_result=LlmSectionPolishResult(section_tldrs={}, section_transcripts={}),
                section_error=LlmProcessingError("section polish failed"),
            ),
            reporter=reporter,
        )

        assert reporter.warnings == ["section polish failed"]
        assert artifacts.diagnostics.llm is not None
        assert artifacts.diagnostics.llm.report_status == "fallback"
        assert artifacts.report.summary == []
        assert artifacts.report.action_items == []

    def test_metadata_polish_failure_keeps_section_timing_and_falls_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        input_path = FIXTURE_DIR / "sample-audio.mp3"
        install_pipeline_runtime(
            monkeypatch, tmp_path, input_path=input_path, runtime=audio_runtime()
        )
        reporter = RecordingStageReporter()

        def section_polish(report: ReportDocument) -> LlmSectionPolishResult:
            return LlmSectionPolishResult(
                section_tldrs={},
                section_transcripts={
                    section.id: section.transcript_text for section in report.sections
                },
            )

        artifacts = process_input(
            input_path,
            output_dir=tmp_path / "metadata-fallback-run",
            transcriber=FakeTranscriber(),
            llm_processor=ConfigurableLLMProcessor(  # type: ignore
                section_result=section_polish, metadata_error=LlmProcessingError("metadata failed")
            ),
            reporter=reporter,
        )

        assert reporter.warnings == ["metadata failed"]
        assert artifacts.diagnostics.llm is not None
        assert artifacts.diagnostics.llm.report_status == "fallback"
        assert "llm_report_sections" in artifacts.diagnostics.stage_durations_sec
        assert "llm_report_metadata" in artifacts.diagnostics.stage_durations_sec
