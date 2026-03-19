"""High-level processing orchestration."""

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from webinar_transcriber.align import align_by_time, align_with_ocr
from webinar_transcriber.asr import Transcriber, WhisperTranscriber
from webinar_transcriber.export import (
    write_docx_report,
    write_json_report,
    write_markdown_report,
)
from webinar_transcriber.media import extract_audio, probe_media
from webinar_transcriber.models import (
    AlignmentBlock,
    Diagnostics,
    MediaAsset,
    OcrResult,
    ReportDocument,
    SlideFrame,
    TranscriptionResult,
)
from webinar_transcriber.ocr import extract_ocr_results
from webinar_transcriber.paths import RunLayout, create_run_layout
from webinar_transcriber.structure import build_report
from webinar_transcriber.video import detect_scenes, extract_representative_frames


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
    alignment_blocks: list[AlignmentBlock] | None = None
    slide_frames: list[SlideFrame] = []
    ocr_results: list[OcrResult] = []

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

    if media_asset.media_type.value == "video":
        start = perf_counter()
        scenes = detect_scenes(input_path)
        stage_timings["detect_scenes"] = perf_counter() - start
        _write_json(
            layout.scenes_path, {"scenes": [scene.model_dump(mode="json") for scene in scenes]}
        )

        start = perf_counter()
        slide_frames = extract_representative_frames(input_path, scenes, layout.frames_dir)
        stage_timings["extract_frames"] = perf_counter() - start
        if ocr_enabled:
            start = perf_counter()
            ocr_results = extract_ocr_results(
                slide_frames,
                detected_language=transcription.detected_language,
            )
            stage_timings["ocr"] = perf_counter() - start
            _write_json(
                layout.ocr_path,
                {"results": [result.model_dump(mode="json") for result in ocr_results]},
            )
            alignment_blocks = align_with_ocr(
                transcription.segments, scenes, slide_frames, ocr_results
            )
            if not any((result.text or "").strip() for result in ocr_results):
                warnings.append("OCR did not extract readable text; alignment stayed time-based.")
        else:
            alignment_blocks = align_by_time(transcription.segments, scenes, slide_frames)
    else:
        scenes = []

    start = perf_counter()
    report = build_report(
        media_asset,
        transcription,
        ocr_enabled=ocr_enabled,
        alignment_blocks=alignment_blocks,
        warnings=warnings,
    )
    stage_timings["structure"] = perf_counter() - start

    if slide_frames:
        frame_by_id = {frame.id: frame for frame in slide_frames}
        for section in report.sections:
            if section.frame_id and section.frame_id in frame_by_id:
                section.image_path = frame_by_id[section.frame_id].image_path

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
            "scenes": len(scenes),
            "frames": len(slide_frames),
            "ocr_results": len(ocr_results),
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
