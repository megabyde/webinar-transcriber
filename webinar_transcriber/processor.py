"""High-level processing orchestration."""

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from webinar_transcriber.asr import Transcriber, WhisperTranscriber
from webinar_transcriber.export import (
    write_docx_report,
    write_json_report,
    write_markdown_report,
)
from webinar_transcriber.media import extract_audio, probe_media
from webinar_transcriber.models import Diagnostics, MediaAsset, ReportDocument, TranscriptionResult
from webinar_transcriber.paths import RunLayout, create_run_layout
from webinar_transcriber.structure import build_report


@dataclass(frozen=True)
class ProcessArtifacts:
    """Runtime artifacts returned from a processing run."""

    layout: RunLayout
    media_asset: MediaAsset
    transcription: TranscriptionResult
    report: ReportDocument
    diagnostics: Diagnostics


def process_input(
    input_path: Path,
    *,
    output_dir: Path | None = None,
    output_format: str = "all",
    ocr_enabled: bool = False,
    transcriber: Transcriber | None = None,
) -> ProcessArtifacts:
    """Process a single audio or video file into report artifacts."""
    stage_timings: dict[str, float] = {}
    warnings: list[str] = []

    start = perf_counter()
    layout = create_run_layout(input_path=input_path, output_dir=output_dir)
    stage_timings["prepare_run_dir"] = perf_counter() - start

    start = perf_counter()
    media_asset = probe_media(input_path)
    stage_timings["probe_media"] = perf_counter() - start

    if media_asset.media_type.value == "audio" and ocr_enabled:
        warnings.append("OCR was requested for audio-only input and has been ignored.")
        ocr_enabled = False

    audio_path = layout.run_dir / "audio.wav"
    start = perf_counter()
    extract_audio(input_path, audio_path)
    stage_timings["extract_audio"] = perf_counter() - start

    start = perf_counter()
    active_transcriber = transcriber or WhisperTranscriber()
    transcription = active_transcriber.transcribe(audio_path)
    stage_timings["transcribe"] = perf_counter() - start

    start = perf_counter()
    report = build_report(
        media_asset,
        transcription,
        ocr_enabled=ocr_enabled,
        warnings=warnings,
    )
    stage_timings["structure"] = perf_counter() - start

    _write_json(
        layout.metadata_path,
        {
            "media": media_asset.model_dump(mode="json"),
        },
    )
    _write_json(layout.transcript_path, transcription.model_dump(mode="json"))

    start = perf_counter()
    _write_requested_reports(report, layout, output_format)
    stage_timings["export"] = perf_counter() - start

    diagnostics = Diagnostics(
        stage_durations_sec={key: round(value, 6) for key, value in stage_timings.items()},
        item_counts={
            "transcript_segments": len(transcription.segments),
            "report_sections": len(report.sections),
        },
        warnings=warnings,
    )
    _write_json(layout.diagnostics_path, diagnostics.model_dump(mode="json"))

    return ProcessArtifacts(
        layout=layout,
        media_asset=media_asset,
        transcription=transcription,
        report=report,
        diagnostics=diagnostics,
    )


def _write_requested_reports(report: ReportDocument, layout: RunLayout, output_format: str) -> None:
    formats = {"md", "docx", "json"} if output_format == "all" else {output_format}

    if "md" in formats:
        write_markdown_report(report, layout.markdown_report_path)
    if "docx" in formats:
        write_docx_report(report, layout.docx_report_path)

    write_json_report(report, layout.json_report_path)


def _write_json(output_path: Path, payload: dict[str, object]) -> None:
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
