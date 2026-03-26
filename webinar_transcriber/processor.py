"""High-level processing orchestration."""

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from webinar_transcriber.align import align_by_time
from webinar_transcriber.asr import ASR_BACKEND_NAME, Transcriber, WhisperCppTranscriber
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
from webinar_transcriber.media import probe_media
from webinar_transcriber.models import (
    AlignmentBlock,
    AsrPipelineDiagnostics,
    Diagnostics,
    MediaAsset,
    ReportDocument,
    SlideFrame,
    TranscriptionResult,
)
from webinar_transcriber.paths import RunLayout, create_run_layout
from webinar_transcriber.reconciliation import reconcile_chunk_transcriptions
from webinar_transcriber.structure import build_report
from webinar_transcriber.transcript_processing import normalize_transcription
from webinar_transcriber.transcription_audio import (
    ChunkPlanSettings,
    VADSettings,
    average_chunk_duration,
    detect_speech_regions,
    load_normalized_audio,
    normalized_audio_duration,
    plan_audio_chunks,
    prepared_transcription_audio,
)
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
    asr_model: str | None = None,
    vad_enabled: bool = True,
    chunk_target_sec: float = 20.0,
    chunk_max_sec: float = 30.0,
    chunk_overlap_sec: float = 1.5,
    asr_threads: int = 4,
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
    asr_pipeline = AsrPipelineDiagnostics(vad_enabled=vad_enabled, threads=asr_threads)

    active_transcriber = transcriber or WhisperCppTranscriber(
        model_name=asr_model,
        threads=asr_threads,
    )

    active_reporter.begin_run(input_path, output_format=output_format)

    active_reporter.stage_started("prepare_run_dir", "Preparing run directory")
    start = perf_counter()
    layout = create_run_layout(input_path=input_path, output_dir=output_dir)
    stage_timings["prepare_run_dir"] = perf_counter() - start
    active_reporter.stage_finished(
        "prepare_run_dir", "Preparing run directory", detail=str(layout.run_dir)
    )
    _configure_asr_logging(active_transcriber, layout)

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

    active_reporter.stage_started("prepare_transcription_audio", "Preparing audio")
    start = perf_counter()
    with prepared_transcription_audio(input_path, media_asset) as audio_path:
        stage_timings["prepare_transcription_audio"] = perf_counter() - start
        active_reporter.stage_finished(
            "prepare_transcription_audio",
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

        if isinstance(active_transcriber, WhisperCppTranscriber):
            active_reporter.stage_started("vad", "Detecting speech regions")
            start = perf_counter()
            audio_samples, sample_rate = load_normalized_audio(audio_path)
            asr_pipeline.normalized_audio_duration_sec = normalized_audio_duration(
                audio_samples, sample_rate
            )
            speech_regions, vad_warnings = detect_speech_regions(
                audio_samples,
                sample_rate,
                settings=VADSettings(enabled=vad_enabled),
            )
            stage_timings["vad"] = perf_counter() - start
            asr_pipeline.vad_region_count = len(speech_regions)
            active_reporter.stage_finished(
                "vad",
                "Detecting speech regions",
                detail=_count_label(len(speech_regions), singular="region"),
            )
            for warning in vad_warnings:
                warnings.append(warning)
                active_reporter.warn(warning)

            active_reporter.stage_started("plan_chunks", "Planning chunks")
            start = perf_counter()
            chunks = plan_audio_chunks(
                speech_regions,
                settings=ChunkPlanSettings(
                    target_sec=chunk_target_sec,
                    max_sec=chunk_max_sec,
                    overlap_sec=chunk_overlap_sec,
                ),
            )
            stage_timings["plan_chunks"] = perf_counter() - start
            asr_pipeline.chunk_count = len(chunks)
            asr_pipeline.average_chunk_duration_sec = average_chunk_duration(chunks)
            asr_pipeline.overlap_duration_sec = chunk_overlap_sec
            active_reporter.stage_finished(
                "plan_chunks",
                "Planning chunks",
                detail=_count_label(len(chunks), singular="chunk"),
            )

            active_reporter.progress_started(
                "transcribe",
                "Transcribing audio",
                total=media_asset.duration_sec,
                count_label="frames",
                count_multiplier=100.0,
                rate_label="frames/s",
                rate_multiplier=100.0,
            )
            transcribed_until_sec = 0.0

            def _on_chunk_completed(completed_sec: float) -> None:
                nonlocal transcribed_until_sec
                advance = max(0.0, completed_sec - transcribed_until_sec)
                if advance > 0:
                    active_reporter.progress_advanced("transcribe", advance=advance)
                    transcribed_until_sec = completed_sec

            start = perf_counter()
            chunk_transcriptions = active_transcriber.transcribe_audio_chunks(
                audio_samples,
                chunks,
                progress_callback=_on_chunk_completed,
            )
            remaining_progress = max(0.0, media_asset.duration_sec - transcribed_until_sec)
            if remaining_progress > 0:
                active_reporter.progress_advanced("transcribe", advance=remaining_progress)
            stage_timings["transcribe"] = perf_counter() - start
            active_reporter.stage_finished(
                "transcribe",
                "Transcribing audio",
                detail=_chunk_transcription_stage_detail(
                    chunk_count=len(chunks),
                    total_duration_sec=media_asset.duration_sec,
                    elapsed_sec=stage_timings["transcribe"],
                ),
            )

            active_reporter.stage_started("reconcile", "Reconciling transcript chunks")
            start = perf_counter()
            transcription, reconciliation_stats = reconcile_chunk_transcriptions(
                chunk_transcriptions
            )
            stage_timings["reconcile"] = perf_counter() - start
            asr_pipeline.reconciliation_duplicate_segments_dropped = (
                reconciliation_stats.duplicate_segments_dropped
            )
            asr_pipeline.reconciliation_boundary_fixes = reconciliation_stats.boundary_fixes
            asr_pipeline.system_info = active_transcriber.system_info
            active_reporter.stage_finished(
                "reconcile",
                "Reconciling transcript chunks",
                detail=f"{len(transcription.segments)} segments",
            )
        else:
            start = perf_counter()
            transcription = _run_transcription_stage(
                active_transcriber,
                audio_path,
                total_duration_sec=media_asset.duration_sec,
                reporter=active_reporter,
            )
            stage_timings["transcribe"] = perf_counter() - start
            active_reporter.stage_finished(
                "transcribe",
                "Transcribing audio",
                detail=_transcription_stage_detail(
                    transcription,
                    total_duration_sec=media_asset.duration_sec,
                    elapsed_sec=stage_timings["transcribe"],
                ),
            )

        _write_json(layout.transcript_path, _transcription_payload(transcription))
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
            detail=_count_label(len(scenes), singular="scene"),
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
            detail=_count_label(len(slide_frames), singular="frame"),
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
        detail=_count_label(len(report.sections), singular="section"),
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
        asr_backend=ASR_BACKEND_NAME,
        asr_model=active_transcriber.model_name,
        llm_enabled=enable_llm,
        llm_model=llm_runtime.model_name,
        llm_report_status=llm_runtime.report_status,
        llm_report_latency_sec=llm_runtime.report_latency_sec,
        llm_report_usage=llm_runtime.report_usage or {},
        stage_durations_sec={key: round(value, 6) for key, value in stage_timings.items()},
        item_counts={
            "transcript_segments": len(transcription.segments),
            "normalized_transcript_segments": len(normalized_transcription.segments),
            "vad_regions": asr_pipeline.vad_region_count,
            "chunks": asr_pipeline.chunk_count,
            "report_sections": len(report.sections),
            "scenes": len(scenes),
            "frames": len(slide_frames),
        },
        asr_pipeline=asr_pipeline,
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
    section_label = _llm_stage_label(
        "Polishing section text with LLM",
        llm_runtime,
        detail=_llm_report_plan_label_detail(polish_plan),
    )
    summary_label = _llm_stage_label("Polishing report summary with LLM", llm_runtime)
    reporter.progress_started(
        "llm_report_sections",
        section_label,
        total=max(float(polish_plan.section_count), 1.0),
        count_label="sections",
    )
    start = perf_counter()
    try:
        section_result = llm_processor.polish_report_sections_with_progress(
            report,
            progress_callback=lambda advance: reporter.progress_advanced(
                "llm_report_sections",
                advance=float(advance),
            ),
        )
    except LLMProcessingError as error:
        report_latency_sec = perf_counter() - start
        stage_timings["llm_report"] = report_latency_sec
        warnings.append(str(error))
        reporter.warn(str(error))
        reporter.stage_finished(
            "llm_report_sections",
            section_label,
            detail=_llm_fallback_detail(llm_runtime),
        )
        llm_runtime.report_status = "fallback"
        llm_runtime.report_latency_sec = report_latency_sec
        return report, llm_runtime

    for warning in section_result.warnings:
        warnings.append(warning)
        reporter.warn(warning)
    reporter.stage_finished(
        "llm_report_sections",
        section_label,
        detail=_count_label(len(section_result.section_transcripts), singular="section"),
    )
    reporter.stage_started("llm_report", summary_label)

    try:
        metadata_result = llm_processor.polish_report_metadata(
            report,
            section_transcripts=section_result.section_transcripts,
        )
    except LLMProcessingError as error:
        report_latency_sec = perf_counter() - start
        stage_timings["llm_report"] = report_latency_sec
        warnings.append(str(error))
        reporter.warn(str(error))
        reporter.stage_finished(
            "llm_report",
            summary_label,
            detail=_llm_fallback_detail(llm_runtime),
        )
        llm_runtime.report_status = "fallback"
        llm_runtime.report_latency_sec = report_latency_sec
        return report, llm_runtime

    report_latency_sec = perf_counter() - start
    stage_timings["llm_report"] = report_latency_sec
    usage = _merged_usage(section_result.usage, metadata_result.usage)
    report.summary = metadata_result.summary
    report.action_items = metadata_result.action_items
    for section in report.sections:
        section.title = metadata_result.section_titles.get(section.id, section.title)
        section.transcript_text = section_result.section_transcripts.get(
            section.id,
            section.transcript_text,
        )
    reporter.stage_finished(
        "llm_report",
        summary_label,
        detail=_llm_report_detail(
            llm_runtime,
            section_count=len(section_result.section_transcripts),
            title_count=len(metadata_result.section_titles),
            summary_count=len(metadata_result.summary),
            action_item_count=len(metadata_result.action_items),
            usage=usage,
        ),
    )
    llm_runtime.report_status = "applied"
    llm_runtime.report_latency_sec = report_latency_sec
    llm_runtime.report_usage = usage
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


def _count_label(count: int, *, singular: str, plural: str | None = None) -> str:
    resolved_plural = plural or f"{singular}s"
    label = singular if count == 1 else resolved_plural
    return f"{count} {label}"


def _configure_asr_logging(transcriber: Transcriber, layout: RunLayout) -> None:
    if isinstance(transcriber, WhisperCppTranscriber):
        transcriber.set_log_path(layout.run_dir / "whisper-cpp.log")


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


def _asr_runtime_detail(transcriber: Transcriber) -> str:
    return f"{transcriber.model_name} | {transcriber.device_name}"


def _transcription_stage_detail(
    transcription: TranscriptionResult,
    *,
    total_duration_sec: float,
    elapsed_sec: float,
) -> str:
    segment_count = len(transcription.segments)
    segment_label = "segment" if segment_count == 1 else "segments"
    details = [f"{segment_count} {segment_label}"]
    average_frames_per_second = _average_frames_per_second(
        total_duration_sec=total_duration_sec,
        elapsed_sec=elapsed_sec,
    )
    if average_frames_per_second is not None:
        details.append(f"avg {average_frames_per_second:.0f} frames/s")
    return " | ".join(details)


def _chunk_transcription_stage_detail(
    *,
    chunk_count: int,
    total_duration_sec: float,
    elapsed_sec: float,
) -> str:
    chunk_label = "chunk" if chunk_count == 1 else "chunks"
    details = [f"{chunk_count} {chunk_label}"]
    average_frames_per_second = _average_frames_per_second(
        total_duration_sec=total_duration_sec,
        elapsed_sec=elapsed_sec,
    )
    if average_frames_per_second is not None:
        details.append(f"avg {average_frames_per_second:.0f} frames/s")
    return " | ".join(details)


def _average_frames_per_second(*, total_duration_sec: float, elapsed_sec: float) -> float | None:
    if total_duration_sec <= 0 or elapsed_sec <= 0:
        return None
    return (total_duration_sec * 100.0) / elapsed_sec


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


def _merged_usage(*usage_dicts: dict[str, int]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for usage in usage_dicts:
        for key, value in usage.items():
            merged[key] = merged.get(key, 0) + value
    return merged


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
