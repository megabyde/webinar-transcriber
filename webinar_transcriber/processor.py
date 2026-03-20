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
from webinar_transcriber.ui import NullStageReporter
from webinar_transcriber.video import (
    detect_scenes,
    estimate_sample_count,
    extract_representative_frames,
)


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
    reporter: NullStageReporter | None = None,
) -> ProcessArtifacts:
    """Process a single audio or video file into report artifacts."""
    active_reporter = reporter or NullStageReporter()
    stage_timings: dict[str, float] = {}
    warnings: list[str] = []
    alignment_blocks: list[AlignmentBlock] | None = None
    slide_frames: list[SlideFrame] = []
    ocr_results: list[OcrResult] = []

    active_reporter.begin_run(input_path, ocr_enabled=ocr_enabled, output_format=output_format)

    active_reporter.stage_started("prepare_run_dir", "Preparing run directory")
    start = perf_counter()
    layout = create_run_layout(input_path=input_path, output_dir=output_dir)
    stage_timings["prepare_run_dir"] = perf_counter() - start
    active_reporter.stage_finished(
        "prepare_run_dir", "Preparing run directory", detail=str(layout.run_dir)
    )

    active_reporter.stage_started("probe_media", "Probing media")
    start = perf_counter()
    media_asset = probe_media(input_path)
    stage_timings["probe_media"] = perf_counter() - start
    active_reporter.stage_finished(
        "probe_media",
        "Probing media",
        detail=f"{media_asset.media_type.value}, {media_asset.duration_sec:.1f}s",
    )

    if media_asset.media_type.value == "audio" and ocr_enabled:
        warning = "OCR was requested for audio-only input and has been ignored."
        warnings.append(warning)
        active_reporter.warn(warning)
        ocr_enabled = False

    audio_path = layout.run_dir / "audio.wav"
    active_reporter.stage_started("extract_audio", "Extracting audio")
    start = perf_counter()
    extract_audio(input_path, audio_path)
    stage_timings["extract_audio"] = perf_counter() - start
    active_reporter.stage_finished("extract_audio", "Extracting audio", detail=str(audio_path.name))

    active_reporter.progress_started(
        "transcribe",
        "Transcribing audio",
        total=media_asset.duration_sec,
    )
    start = perf_counter()
    active_transcriber = transcriber or WhisperTranscriber()
    transcription = _transcribe_with_progress(
        active_transcriber,
        audio_path,
        total_duration_sec=media_asset.duration_sec,
        reporter=active_reporter,
    )
    stage_timings["transcribe"] = perf_counter() - start
    active_reporter.stage_finished(
        "transcribe",
        "Transcribing audio",
        detail=f"{len(transcription.segments)} segments",
    )

    if media_asset.media_type.value == "video":
        active_reporter.progress_started(
            "detect_scenes",
            "Detecting scenes",
            total=estimate_sample_count(media_asset.duration_sec),
        )
        start = perf_counter()
        scenes = detect_scenes(
            input_path,
            duration_sec=media_asset.duration_sec,
            progress_callback=lambda: active_reporter.progress_advanced("detect_scenes"),
        )
        stage_timings["detect_scenes"] = perf_counter() - start
        _write_json(
            layout.scenes_path, {"scenes": [scene.model_dump(mode="json") for scene in scenes]}
        )
        active_reporter.stage_finished(
            "detect_scenes",
            "Detecting scenes",
            detail=f"{len(scenes)} scenes",
        )

        active_reporter.progress_started(
            "extract_frames",
            "Extracting slide frames",
            total=len(scenes),
        )
        start = perf_counter()
        slide_frames = extract_representative_frames(
            input_path,
            scenes,
            layout.frames_dir,
            progress_callback=lambda: active_reporter.progress_advanced("extract_frames"),
        )
        stage_timings["extract_frames"] = perf_counter() - start
        active_reporter.stage_finished(
            "extract_frames",
            "Extracting slide frames",
            detail=f"{len(slide_frames)} frames",
        )
        if ocr_enabled:
            active_reporter.progress_started(
                "ocr",
                "Running OCR",
                total=len(slide_frames),
            )
            start = perf_counter()
            ocr_results = extract_ocr_results(
                slide_frames,
                detected_language=transcription.detected_language,
                progress_callback=lambda: active_reporter.progress_advanced("ocr"),
            )
            stage_timings["ocr"] = perf_counter() - start
            _write_json(
                layout.ocr_path,
                {"results": [result.model_dump(mode="json") for result in ocr_results]},
            )
            alignment_blocks = align_with_ocr(
                transcription.segments, scenes, slide_frames, ocr_results
            )
            active_reporter.stage_finished(
                "ocr",
                "Running OCR",
                detail=f"{len(ocr_results)} OCR results",
            )
            if not any((result.text or "").strip() for result in ocr_results):
                warning = "OCR did not extract readable text; alignment stayed time-based."
                warnings.append(warning)
                active_reporter.warn(warning)
        else:
            alignment_blocks = align_by_time(transcription.segments, scenes, slide_frames)
    else:
        scenes = []

    active_reporter.stage_started("structure", "Structuring report")
    start = perf_counter()
    report = build_report(
        media_asset,
        transcription,
        ocr_enabled=ocr_enabled,
        alignment_blocks=alignment_blocks,
        warnings=warnings,
    )
    stage_timings["structure"] = perf_counter() - start
    active_reporter.stage_finished(
        "structure",
        "Structuring report",
        detail=f"{len(report.sections)} sections",
    )

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

    active_reporter.stage_started("export", "Writing reports")
    start = perf_counter()
    _write_requested_reports(report, layout, output_format)
    stage_timings["export"] = perf_counter() - start
    active_reporter.stage_finished("export", "Writing reports", detail=output_format)

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

    artifacts = ProcessArtifacts(
        layout=layout,
        media_asset=media_asset,
        transcription=transcription,
        report=report,
        diagnostics=diagnostics,
    )
    active_reporter.complete_run(artifacts)
    return artifacts


def _write_requested_reports(report: ReportDocument, layout: RunLayout, output_format: str) -> None:
    formats = {"md", "docx", "json"} if output_format == "all" else {output_format}

    if "md" in formats:
        write_markdown_report(report, layout.markdown_report_path)
    if "docx" in formats:
        write_docx_report(report, layout.docx_report_path)

    write_json_report(report, layout.json_report_path)


def _write_json(output_path: Path, payload: dict[str, object]) -> None:
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _transcribe_with_progress(
    transcriber: Transcriber,
    audio_path: Path,
    *,
    total_duration_sec: float,
    reporter: NullStageReporter,
) -> TranscriptionResult:
    transcribed_until_sec = 0.0

    def _update_transcription_progress(completed_sec: float) -> None:
        nonlocal transcribed_until_sec
        advance = max(0.0, completed_sec - transcribed_until_sec)
        if advance > 0:
            reporter.progress_advanced("transcribe", advance=advance)
            transcribed_until_sec = completed_sec

    transcription = transcriber.transcribe(
        audio_path,
        progress_callback=_update_transcription_progress,
    )
    remaining_progress = max(0.0, total_duration_sec - transcribed_until_sec)
    if remaining_progress > 0:
        reporter.progress_advanced("transcribe", advance=remaining_progress)
    return transcription
