"""Tests for the end-to-end processor flow."""

import json
from collections.abc import Callable
from pathlib import Path

from docx import Document

from webinar_transcriber.models import TranscriptionResult, TranscriptSegment
from webinar_transcriber.processor import process_input
from webinar_transcriber.ui import NullStageReporter

FIXTURE_DIR = Path(__file__).parent / "fixtures"


class FakeTranscriber:
    """Stable test double for deterministic transcripts."""

    backend_name = "test-backend"
    model_name = "test-model"
    supports_live_progress = True
    uses_native_progress = False

    def prepare_model(self) -> None:
        """No-op test hook for the ASR preparation stage."""

    def transcribe(
        self,
        audio_path: Path,
        *,
        progress_callback: Callable[[float], None] | None = None,
    ) -> TranscriptionResult:
        assert audio_path.exists()
        if progress_callback is not None:
            progress_callback(0.75)
            progress_callback(1.5)
        return TranscriptionResult(
            detected_language="en",
            segments=[
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
            ],
        )


class RecordingReporter(NullStageReporter):
    """Collect stage updates for assertions."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str, str]] = []
        self.warnings: list[str] = []
        self.progress_events: list[tuple[str, str, float]] = []

    def begin_run(self, input_path: Path, *, output_format: str) -> None:
        self.events.append(("begin", input_path.name, output_format))

    def stage_started(self, stage_key: str, label: str) -> None:
        self.events.append(("start", stage_key, label))

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
    ) -> None:
        self.progress_events.append(("start", stage_key, total))

    def progress_advanced(self, stage_key: str, *, advance: float = 1.0) -> None:
        self.progress_events.append(("advance", stage_key, advance))

    def stage_finished(self, stage_key: str, label: str, *, detail: str | None = None) -> None:
        self.events.append(("finish", stage_key, detail or ""))

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def complete_run(self, artifacts) -> None:
        self.events.append(
            ("complete", artifacts.layout.run_dir.name, artifacts.report.source_file)
        )


def test_process_input_writes_reports_and_metadata(tmp_path) -> None:
    reporter = RecordingReporter()
    artifacts = process_input(
        FIXTURE_DIR / "sample-audio.mp3",
        output_dir=tmp_path / "run",
        transcriber=FakeTranscriber(),
        reporter=reporter,
    )

    assert artifacts.layout.metadata_path.exists()
    assert artifacts.layout.transcript_path.exists()
    assert artifacts.layout.markdown_report_path.exists()
    assert artifacts.layout.docx_report_path.exists()
    assert artifacts.layout.json_report_path.exists()
    assert not (artifacts.layout.run_dir / "audio.wav").exists()
    assert artifacts.report.detected_language == "en"
    assert artifacts.report.action_items == ["Next step please send the draft by Friday."]
    assert len(artifacts.report.sections) == 1
    assert (
        "Next step please send the draft by Friday." in artifacts.report.sections[0].transcript_text
    )
    assert artifacts.diagnostics.asr_backend == "test-backend"
    assert artifacts.diagnostics.asr_model == "test-model"
    diagnostics_payload = json.loads(artifacts.layout.diagnostics_path.read_text(encoding="utf-8"))

    markdown = artifacts.layout.markdown_report_path.read_text(encoding="utf-8")
    assert "# Sample Audio" in markdown
    assert "Agenda review and project status update." in markdown
    assert ("start", "extract_audio", "Preparing audio") in reporter.events
    assert ("finish", "extract_audio", "sample-audio.mp3") in reporter.events
    assert ("start", "probe_media", "Probing media") in reporter.events
    assert ("start", "prepare_asr", "Preparing ASR model") in reporter.events
    assert (
        "finish",
        "prepare_asr",
        "test-backend | test-model",
    ) in reporter.events
    assert any(event[0] == "complete" for event in reporter.events)
    assert any(event == ("start", "transcribe", 1.5) for event in reporter.progress_events)
    assert any(
        event[0] == "advance" and event[1] == "transcribe" for event in reporter.progress_events
    )
    assert diagnostics_payload["asr_backend"] == "test-backend"
    assert diagnostics_payload["asr_model"] == "test-model"

    document = Document(str(artifacts.layout.docx_report_path))
    assert "Sample Audio" in "\n".join(paragraph.text for paragraph in document.paragraphs)


def test_process_input_writes_video_scene_artifacts(tmp_path) -> None:
    reporter = RecordingReporter()
    transcription_audio_path: Path | None = None

    class VideoTranscriber(FakeTranscriber):
        def transcribe(
            self,
            audio_path: Path,
            *,
            progress_callback: Callable[[float], None] | None = None,
        ) -> TranscriptionResult:
            nonlocal transcription_audio_path
            transcription_audio_path = audio_path
            assert audio_path.exists()
            if progress_callback is not None:
                progress_callback(0.9)
                progress_callback(1.8)
            return TranscriptionResult(
                detected_language="en",
                segments=[
                    TranscriptSegment(
                        id="segment-1",
                        text="Agenda overview and open questions.",
                        start_sec=0.0,
                        end_sec=0.9,
                    ),
                    TranscriptSegment(
                        id="segment-2",
                        text="Timeline planning and next step review.",
                        start_sec=0.9,
                        end_sec=1.8,
                    ),
                ],
            )

    artifacts = process_input(
        FIXTURE_DIR / "sample-video.mp4",
        output_dir=tmp_path / "video-run",
        transcriber=VideoTranscriber(),
        reporter=reporter,
    )

    assert artifacts.media_asset.media_type.value == "video"
    assert not (artifacts.layout.run_dir / "audio.wav").exists()
    assert transcription_audio_path is not None
    assert not transcription_audio_path.exists()
    assert artifacts.layout.scenes_path.exists()
    assert artifacts.layout.frames_dir.exists()
    assert any(artifacts.layout.frames_dir.iterdir())
    assert artifacts.report.sections
    assert artifacts.report.sections[0].image_path
    assert any(event == ("start", "detect_scenes", 2) for event in reporter.progress_events)
    assert any(
        event[0] == "advance" and event[1] == "detect_scenes" for event in reporter.progress_events
    )
    assert any(
        event == ("start", "extract_frames", artifacts.diagnostics.item_counts["scenes"])
        for event in reporter.progress_events
    )


def test_process_input_uses_spinner_for_non_streaming_transcriber(tmp_path) -> None:
    reporter = RecordingReporter()

    class BlockingTranscriber:
        backend_name = "blocking-backend"
        model_name = "blocking-model"
        supports_live_progress = False
        uses_native_progress = False

        def __init__(self) -> None:
            self.prepared = False

        def prepare_model(self) -> None:
            self.prepared = True

        def transcribe(
            self,
            audio_path: Path,
            *,
            progress_callback: Callable[[float], None] | None = None,
        ) -> TranscriptionResult:
            assert audio_path.exists()
            assert progress_callback is None
            return TranscriptionResult(
                detected_language="en",
                segments=[
                    TranscriptSegment(
                        id="segment-1",
                        text="Agenda review and project status update.",
                        start_sec=0.0,
                        end_sec=1.5,
                    )
                ],
            )

    transcriber = BlockingTranscriber()
    artifacts = process_input(
        FIXTURE_DIR / "sample-audio.mp3",
        output_dir=tmp_path / "blocking-run",
        transcriber=transcriber,
        reporter=reporter,
    )

    assert artifacts.report.detected_language == "en"
    assert transcriber.prepared is True
    assert ("start", "prepare_asr", "Preparing ASR model") in reporter.events
    assert ("start", "transcribe", "Transcribing audio") in reporter.events
    assert not any(event[1] == "transcribe" for event in reporter.progress_events)


def test_process_input_allows_native_transcriber_progress(tmp_path) -> None:
    reporter = RecordingReporter()

    class NativeProgressTranscriber:
        backend_name = "native-backend"
        model_name = "native-model"
        supports_live_progress = False
        uses_native_progress = True

        def __init__(self) -> None:
            self.prepared = False

        def prepare_model(self) -> None:
            self.prepared = True

        def transcribe(
            self,
            audio_path: Path,
            *,
            progress_callback: Callable[[float], None] | None = None,
        ) -> TranscriptionResult:
            assert audio_path.exists()
            assert progress_callback is None
            return TranscriptionResult(
                detected_language="en",
                segments=[
                    TranscriptSegment(
                        id="segment-1",
                        text="Agenda review and project status update.",
                        start_sec=0.0,
                        end_sec=1.5,
                    )
                ],
            )

    transcriber = NativeProgressTranscriber()
    artifacts = process_input(
        FIXTURE_DIR / "sample-audio.mp3",
        output_dir=tmp_path / "native-progress-run",
        transcriber=transcriber,
        reporter=reporter,
    )

    assert artifacts.report.detected_language == "en"
    assert transcriber.prepared is True
    assert ("start", "prepare_asr", "Preparing ASR model") in reporter.events
    assert ("start", "transcribe", "Transcribing audio") not in reporter.events
    assert any(event[0] == "finish" and event[1] == "transcribe" for event in reporter.events)
    assert not any(event[1] == "transcribe" for event in reporter.progress_events)
