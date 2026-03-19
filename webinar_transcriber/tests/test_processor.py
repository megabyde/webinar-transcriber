"""Tests for the end-to-end processor flow."""

from pathlib import Path

from docx import Document

from webinar_transcriber.models import TranscriptionResult, TranscriptSegment
from webinar_transcriber.processor import process_input
from webinar_transcriber.ui import NullStageReporter

FIXTURE_DIR = Path(__file__).parent / "fixtures"


class FakeTranscriber:
    """Stable test double for deterministic transcripts."""

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        assert audio_path.exists()
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

    def begin_run(self, input_path: Path, *, ocr_enabled: bool, output_format: str) -> None:
        self.events.append(("begin", input_path.name, output_format))

    def stage_started(self, stage_key: str, label: str) -> None:
        self.events.append(("start", stage_key, label))

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
    assert artifacts.report.detected_language == "en"
    assert artifacts.report.action_items == ["Next step please send the draft by Friday."]

    markdown = artifacts.layout.markdown_report_path.read_text(encoding="utf-8")
    assert "# Sample Audio" in markdown
    assert "Agenda review and project status update." in markdown
    assert ("start", "probe_media", "Probing media") in reporter.events
    assert any(event[0] == "complete" for event in reporter.events)

    document = Document(str(artifacts.layout.docx_report_path))
    assert "Sample Audio" in "\n".join(paragraph.text for paragraph in document.paragraphs)


def test_process_input_writes_video_scene_artifacts(tmp_path) -> None:
    class VideoTranscriber(FakeTranscriber):
        def transcribe(self, audio_path: Path) -> TranscriptionResult:
            assert audio_path.exists()
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
    )

    assert artifacts.media_asset.media_type.value == "video"
    assert artifacts.layout.scenes_path.exists()
    assert artifacts.layout.frames_dir.exists()
    assert any(artifacts.layout.frames_dir.iterdir())
    assert artifacts.report.sections
    assert artifacts.report.sections[0].image_path


def test_process_input_writes_ocr_results_for_video(tmp_path) -> None:
    class VideoTranscriber(FakeTranscriber):
        def transcribe(self, audio_path: Path) -> TranscriptionResult:
            assert audio_path.exists()
            return TranscriptionResult(
                detected_language="en",
                segments=[
                    TranscriptSegment(
                        id="segment-1",
                        text="Agenda overview and key points.",
                        start_sec=0.0,
                        end_sec=0.9,
                    ),
                    TranscriptSegment(
                        id="segment-2",
                        text="Timeline planning and milestones.",
                        start_sec=0.9,
                        end_sec=1.8,
                    ),
                ],
            )

    artifacts = process_input(
        FIXTURE_DIR / "sample-video.mp4",
        output_dir=tmp_path / "video-ocr-run",
        ocr_enabled=True,
        transcriber=VideoTranscriber(),
    )

    assert artifacts.layout.ocr_path.exists()
    assert artifacts.report.ocr_enabled is True
    assert any(section.title for section in artifacts.report.sections)


def test_process_input_reports_audio_ocr_warning(tmp_path) -> None:
    reporter = RecordingReporter()

    artifacts = process_input(
        FIXTURE_DIR / "sample-audio.mp3",
        output_dir=tmp_path / "audio-ocr-run",
        ocr_enabled=True,
        transcriber=FakeTranscriber(),
        reporter=reporter,
    )

    assert artifacts.report.ocr_enabled is False
    assert reporter.warnings == ["OCR was requested for audio-only input and has been ignored."]
