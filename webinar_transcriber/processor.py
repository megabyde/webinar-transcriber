"""High-level processing orchestration."""

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from webinar_transcriber.align import align_by_time
from webinar_transcriber.asr import Transcriber, WhisperTranscriber
from webinar_transcriber.export import (
    write_docx_report,
    write_json_report,
    write_markdown_report,
)
from webinar_transcriber.llm import (
    LLMConfigurationError,
    LLMProcessingError,
    LLMProcessor,
    build_llm_processor_from_env,
)
from webinar_transcriber.media import prepared_transcription_audio, probe_media
from webinar_transcriber.models import (
    AlignmentBlock,
    Diagnostics,
    MediaAsset,
    ReportDocument,
    SlideFrame,
    TranscriptionResult,
)
from webinar_transcriber.paths import RunLayout, create_run_layout
from webinar_transcriber.structure import build_report
from webinar_transcriber.transcript_processing import normalize_transcription
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


@dataclass
class LLMRuntimeState:
    """Observed state for the optional LLM report stage."""

    provider_name: str | None = None
    model_name: str | None = None
    report_status: str = "disabled"
    report_latency_sec: float | None = None
    report_usage: dict[str, int] | None = None


def process_input(
    input_path: Path,
    *,
    output_dir: Path | None = None,
    output_format: str = "all",
    asr_backend: str = "auto",
    asr_model: str | None = None,
    enable_llm: bool = False,
    transcriber: Transcriber | None = None,
    llm_processor: LLMProcessor | None = None,
    reporter: NullStageReporter | None = None,
) -> ProcessArtifacts:
    """Process a single audio or video file into report artifacts."""
    active_reporter = reporter or NullStageReporter()
    stage_timings: dict[str, float] = {}
    warnings: list[str] = []
    alignment_blocks: list[AlignmentBlock] | None = None
    slide_frames: list[SlideFrame] = []
    llm_runtime = LLMRuntimeState()

    active_transcriber = transcriber or WhisperTranscriber(
        model_name=asr_model,
        backend=asr_backend,
    )

    active_reporter.begin_run(input_path, output_format=output_format)

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
    _write_json(
        layout.metadata_path,
        {
            "media": media_asset.model_dump(mode="json"),
        },
    )
    active_reporter.stage_finished(
        "probe_media",
        "Probing media",
        detail=f"{media_asset.media_type.value}, {media_asset.duration_sec:.1f}s",
    )

    active_reporter.stage_started("extract_audio", "Preparing audio")
    start = perf_counter()
    with prepared_transcription_audio(input_path, media_asset) as audio_path:
        stage_timings["extract_audio"] = perf_counter() - start
        active_reporter.stage_finished(
            "extract_audio",
            "Preparing audio",
            detail=str(audio_path.name),
        )

        active_reporter.stage_started("prepare_asr", "Preparing ASR model")
        start = perf_counter()
        active_transcriber.prepare_model()
        stage_timings["prepare_asr"] = perf_counter() - start
        active_reporter.stage_finished(
            "prepare_asr",
            "Preparing ASR model",
            detail=_asr_runtime_detail(active_transcriber),
        )

        start = perf_counter()
        transcription = _run_transcription_stage(
            active_transcriber,
            audio_path,
            total_duration_sec=media_asset.duration_sec,
            reporter=active_reporter,
        )
        stage_timings["transcribe"] = perf_counter() - start
        _write_json(layout.transcript_path, _transcription_payload(transcription))
        active_reporter.stage_finished(
            "transcribe",
            "Transcribing audio",
            detail=f"{len(transcription.segments)} segments",
        )
        normalized_transcription = normalize_transcription(transcription)

        llm_enhancer, llm_runtime = _resolve_llm_processor(
            enable_llm=enable_llm,
            llm_processor=llm_processor,
            reporter=active_reporter,
            warnings=warnings,
            llm_runtime=llm_runtime,
        )
        report_transcription = normalized_transcription

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
        alignment_blocks = align_by_time(report_transcription.segments, scenes, slide_frames)
    else:
        scenes = []

    active_reporter.stage_started("structure", "Structuring report")
    start = perf_counter()
    report = build_report(
        media_asset,
        report_transcription,
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

    report, llm_runtime = _maybe_polish_report(
        report,
        llm_processor=llm_enhancer,
        reporter=active_reporter,
        warnings=warnings,
        stage_timings=stage_timings,
        llm_runtime=llm_runtime,
    )
    report.warnings = list(warnings)

    active_reporter.stage_started("export", "Writing reports")
    start = perf_counter()
    _write_requested_reports(report, layout, output_format)
    stage_timings["export"] = perf_counter() - start
    active_reporter.stage_finished("export", "Writing reports", detail=output_format)

    diagnostics = Diagnostics(
        asr_backend=active_transcriber.backend_name,
        asr_model=active_transcriber.model_name,
        llm_enabled=enable_llm,
        llm_model=llm_runtime.model_name,
        llm_transcript_status="disabled",
        llm_report_status=llm_runtime.report_status,
        llm_transcript_latency_sec=None,
        llm_report_latency_sec=llm_runtime.report_latency_sec,
        llm_transcript_usage={},
        llm_report_usage=llm_runtime.report_usage or {},
        stage_durations_sec={key: round(value, 6) for key, value in stage_timings.items()},
        item_counts={
            "transcript_segments": len(transcription.segments),
            "normalized_transcript_segments": len(normalized_transcription.segments),
            "polished_transcript_segments": 0,
            "report_sections": len(report.sections),
            "scenes": len(scenes),
            "frames": len(slide_frames),
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


def _resolve_llm_processor(
    *,
    enable_llm: bool,
    llm_processor: LLMProcessor | None,
    reporter: NullStageReporter,
    warnings: list[str],
    llm_runtime: LLMRuntimeState,
) -> tuple[LLMProcessor | None, LLMRuntimeState]:
    if not enable_llm:
        return None, llm_runtime

    if llm_processor is not None:
        llm_runtime.provider_name = llm_processor.provider_name
        llm_runtime.model_name = llm_processor.model_name
        return llm_processor, llm_runtime

    try:
        resolved_processor = build_llm_processor_from_env()
    except LLMConfigurationError as error:
        warnings.append(str(error))
        reporter.warn(str(error))
        llm_runtime.report_status = "fallback"
        return None, llm_runtime

    llm_runtime.provider_name = resolved_processor.provider_name
    llm_runtime.model_name = resolved_processor.model_name
    return resolved_processor, llm_runtime


def _maybe_polish_report(
    report: ReportDocument,
    *,
    llm_processor: LLMProcessor | None,
    reporter: NullStageReporter,
    warnings: list[str],
    stage_timings: dict[str, float],
    llm_runtime: LLMRuntimeState,
) -> tuple[ReportDocument, LLMRuntimeState]:
    if llm_processor is None:
        return report, llm_runtime

    polish_plan = llm_processor.report_polish_plan(report)
    label = _llm_stage_label(
        "Polishing report with LLM",
        llm_runtime,
        detail=_llm_report_plan_label_detail(polish_plan),
    )
    reporter.progress_started(
        "llm_report",
        label,
        total=max(float(polish_plan.section_count), 1.0),
        count_label="sections",
    )
    start = perf_counter()
    try:
        polish_result = llm_processor.polish_report_with_progress(
            report,
            progress_callback=lambda advance: reporter.progress_advanced(
                "llm_report",
                advance=float(advance),
            ),
        )
    except LLMProcessingError as error:
        report_latency_sec = perf_counter() - start
        stage_timings["llm_report"] = report_latency_sec
        warnings.append(str(error))
        reporter.warn(str(error))
        reporter.stage_finished(
            "llm_report",
            label,
            detail=_llm_fallback_detail(llm_runtime),
        )
        llm_runtime.report_status = "fallback"
        llm_runtime.report_latency_sec = report_latency_sec
        return report, llm_runtime

    report_latency_sec = perf_counter() - start
    stage_timings["llm_report"] = report_latency_sec
    for warning in polish_result.warnings:
        warnings.append(warning)
        reporter.warn(warning)
    report.summary = polish_result.summary
    report.action_items = polish_result.action_items
    for section in report.sections:
        section.title = polish_result.section_titles.get(section.id, section.title)
        section.transcript_text = polish_result.section_transcripts.get(
            section.id,
            section.transcript_text,
        )
    reporter.stage_finished(
        "llm_report",
        label,
        detail=_llm_report_detail(
            llm_runtime,
            section_count=len(polish_result.section_transcripts),
            title_count=len(polish_result.section_titles),
            summary_count=len(polish_result.summary),
            action_item_count=len(polish_result.action_items),
            usage=polish_result.usage,
        ),
    )
    llm_runtime.report_status = "applied"
    llm_runtime.report_latency_sec = report_latency_sec
    llm_runtime.report_usage = polish_result.usage
    return report, llm_runtime


def _write_requested_reports(report: ReportDocument, layout: RunLayout, output_format: str) -> None:
    formats = {"md", "docx", "json"} if output_format == "all" else {output_format}

    if "md" in formats:
        write_markdown_report(report, layout.markdown_report_path)
    if "docx" in formats:
        write_docx_report(report, layout.docx_report_path)

    write_json_report(report, layout.json_report_path)


def _write_json(output_path: Path, payload: dict[str, object]) -> None:
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _transcription_payload(transcription: TranscriptionResult) -> dict[str, object]:
    return transcription.model_dump(mode="json")


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


def _run_transcription_stage(
    transcriber: Transcriber,
    audio_path: Path,
    *,
    total_duration_sec: float,
    reporter: NullStageReporter,
) -> TranscriptionResult:
    if transcriber.supports_live_progress:
        reporter.progress_started(
            "transcribe",
            "Transcribing audio",
            total=total_duration_sec,
            count_label="frames",
            count_multiplier=100.0,
            rate_label="frames/s",
            rate_multiplier=100.0,
        )
        return _transcribe_with_progress(
            transcriber,
            audio_path,
            total_duration_sec=total_duration_sec,
            reporter=reporter,
        )

    if transcriber.uses_native_progress:
        reporter.stage_timing_started("transcribe", "Transcribing audio")
        return transcriber.transcribe(audio_path)

    reporter.stage_started("transcribe", "Transcribing audio")
    return transcriber.transcribe(audio_path)


def _asr_runtime_detail(transcriber: Transcriber) -> str:
    return f"{transcriber.backend_name} | {transcriber.model_name}"


def _llm_stage_label(
    base_label: str,
    llm_runtime: LLMRuntimeState,
    *,
    detail: str | None = None,
) -> str:
    runtime_detail = _llm_runtime_detail(llm_runtime)
    parenthetical = " | ".join(part for part in (runtime_detail, detail) if part)
    return f"{base_label} ({parenthetical})" if parenthetical else base_label


def _llm_runtime_detail(llm_runtime: LLMRuntimeState) -> str:
    parts = [
        value
        for value in (
            llm_runtime.provider_name,
            llm_runtime.model_name,
        )
        if value
    ]
    return " | ".join(parts)


def _llm_report_detail(
    llm_runtime: LLMRuntimeState,
    *,
    section_count: int,
    title_count: int,
    summary_count: int,
    action_item_count: int,
    usage: dict[str, int],
) -> str:
    parts = [
        _optional_count_detail(summary_count, singular="summary bullet", plural="summary bullets"),
        _optional_count_detail(
            action_item_count,
            singular="action item",
            plural="action items",
        ),
        _title_update_detail(title_count=title_count, section_count=section_count),
        _token_usage_detail(usage),
    ]
    return " | ".join(part for part in parts if part)


def _llm_fallback_detail(llm_runtime: LLMRuntimeState) -> str:
    runtime_detail = _llm_runtime_detail(llm_runtime)
    return " | ".join(part for part in (runtime_detail, "fallback") if part)


def _llm_report_plan_label_detail(plan) -> str:
    worker_label = "worker" if plan.worker_count == 1 else "workers"
    return f"{plan.worker_count} {worker_label}"


def _token_usage_detail(usage: dict[str, int]) -> str:
    total_tokens = usage.get("total_tokens")
    if total_tokens is None:
        return ""
    token_label = "token" if total_tokens == 1 else "tokens"
    return f"{total_tokens} {token_label}"


def _optional_count_detail(count: int, *, singular: str, plural: str) -> str:
    if count <= 0:
        return ""
    label = singular if count == 1 else plural
    return f"{count} {label}"


def _title_update_detail(*, title_count: int, section_count: int) -> str:
    if title_count <= 0 or title_count == section_count:
        return ""
    label = "title updated" if title_count == 1 else "titles updated"
    return f"{title_count} {label}"
