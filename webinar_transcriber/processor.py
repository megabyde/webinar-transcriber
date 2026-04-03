"""High-level processing orchestration."""

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from webinar_transcriber.align import align_by_time
from webinar_transcriber.asr import (
    ASR_BACKEND_NAME,
    DEFAULT_ASR_THREADS,
    DEFAULT_CARRYOVER_MAX_SENTENCES,
    DEFAULT_CARRYOVER_MAX_TOKENS,
    PromptCarryoverSettings,
    WhisperCppTranscriber,
)
from webinar_transcriber.export import (
    write_docx_report,
    write_json_report,
    write_markdown_report,
    write_vtt_subtitles,
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
    InferenceWindow,
    MediaAsset,
    ReportDocument,
    SlideFrame,
    TranscriptionResult,
)
from webinar_transcriber.paths import RunLayout, create_run_layout
from webinar_transcriber.reconciliation import reconcile_decoded_windows
from webinar_transcriber.segmentation import (
    DEFAULT_MIN_SILENCE_DURATION_MS,
    DEFAULT_MIN_SPEECH_DURATION_MS,
    DEFAULT_SPEECH_PAD_MS,
    DEFAULT_SPEECH_REGION_PAD_MS,
    DEFAULT_VAD_THRESHOLD,
    detect_speech_regions,
    expand_speech_regions,
    normalized_audio_duration,
    repair_speech_regions,
)
from webinar_transcriber.structure import build_report
from webinar_transcriber.transcript_processing import normalize_transcription
from webinar_transcriber.transcription_audio import (
    load_normalized_audio,
    prepared_transcription_audio,
    preserve_transcription_audio,
)
from webinar_transcriber.ui import NullStageReporter
from webinar_transcriber.usage import merge_usage
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


@dataclass(frozen=True)
class _StageTimer:
    stage_timings: dict[str, float]
    key: str
    start_sec: float

    def finish(self) -> float:
        elapsed_sec = perf_counter() - self.start_sec
        self.stage_timings[self.key] = elapsed_sec
        return elapsed_sec


@dataclass(frozen=True)
class _AsrPipelineResult:
    transcription: TranscriptionResult
    normalized_transcription: TranscriptionResult


def process_input(
    input_path: Path,
    *,
    output_dir: Path | None = None,
    output_format: str = "all",
    asr_model: str | None = None,
    vad_enabled: bool = True,
    vad_threshold: float = DEFAULT_VAD_THRESHOLD,
    min_speech_duration_ms: int = DEFAULT_MIN_SPEECH_DURATION_MS,
    min_silence_duration_ms: int = DEFAULT_MIN_SILENCE_DURATION_MS,
    speech_region_pad_ms: int = DEFAULT_SPEECH_REGION_PAD_MS,
    carryover_enabled: bool = True,
    carryover_max_sentences: int = DEFAULT_CARRYOVER_MAX_SENTENCES,
    carryover_max_tokens: int = DEFAULT_CARRYOVER_MAX_TOKENS,
    asr_threads: int = DEFAULT_ASR_THREADS,
    keep_audio: bool = False,
    kept_audio_format: str = "wav",
    enable_llm: bool = False,
    transcriber: WhisperCppTranscriber | None = None,
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
    asr_pipeline.carryover_enabled = carryover_enabled

    active_transcriber = transcriber or WhisperCppTranscriber(
        model_name=asr_model,
        threads=asr_threads,
        carryover_settings=PromptCarryoverSettings(
            enabled=carryover_enabled,
            max_sentences=carryover_max_sentences,
            max_tokens=carryover_max_tokens,
        ),
    )
    active_reporter.begin_run(input_path, output_format=output_format)

    active_reporter.stage_started("prepare_run_dir", "Preparing run directory")
    timer = _start_stage_timer(stage_timings, "prepare_run_dir")
    layout = create_run_layout(input_path=input_path, output_dir=output_dir)
    timer.finish()
    active_reporter.stage_finished(
        "prepare_run_dir", "Preparing run directory", detail=str(layout.run_dir)
    )
    _configure_asr_logging(active_transcriber, layout)

    active_reporter.stage_started("probe_media", "Probing media")
    timer = _start_stage_timer(stage_timings, "probe_media")
    media_asset = probe_media(input_path)
    timer.finish()
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
    timer = _start_stage_timer(stage_timings, "prepare_transcription_audio")
    with prepared_transcription_audio(input_path) as audio_path:
        timer.finish()
        active_reporter.stage_finished(
            "prepare_transcription_audio",
            "Preparing audio",
            detail=str(audio_path.name),
        )
        asr_result = _run_asr_pipeline(
            audio_path=audio_path,
            media_asset=media_asset,
            transcriber=active_transcriber,
            layout=layout,
            reporter=active_reporter,
            stage_timings=stage_timings,
            warnings=warnings,
            asr_pipeline=asr_pipeline,
            vad_enabled=vad_enabled,
            vad_threshold=vad_threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_region_pad_ms=speech_region_pad_ms,
        )
        transcription = asr_result.transcription
        normalized_transcription = asr_result.normalized_transcription

        llm_enhancer, llm_runtime = _resolve_llm_processor(
            enable_llm=enable_llm,
            llm_processor=llm_processor,
            reporter=active_reporter,
            warnings=warnings,
            llm_runtime=llm_runtime,
        )
        report_transcription = normalized_transcription
        if keep_audio:
            preserved_audio_path = layout.transcription_audio_path(kept_audio_format)
            preserve_transcription_audio(
                audio_path,
                preserved_audio_path,
                audio_format=kept_audio_format,
            )

    if media_asset.media_type.value == "video":
        active_reporter.progress_started(
            "detect_scenes",
            "Detecting scenes",
            total=estimate_sample_count(media_asset.duration_sec),
            count_label="s",
            detail="0 scenes",
        )
        timer = _start_stage_timer(stage_timings, "detect_scenes")
        scenes = detect_scenes(
            input_path,
            duration_sec=media_asset.duration_sec,
            progress_callback=lambda scene_count: active_reporter.progress_advanced(
                "detect_scenes",
                detail=_count_label(scene_count, singular="scene"),
            ),
        )
        timer.finish()
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
        timer = _start_stage_timer(stage_timings, "extract_frames")
        slide_frames = extract_representative_frames(
            input_path,
            scenes,
            layout.frames_dir,
            progress_callback=lambda: active_reporter.progress_advanced("extract_frames"),
        )
        timer.finish()
        active_reporter.stage_finished(
            "extract_frames",
            "Extracting slide frames",
            detail=_count_label(len(slide_frames), singular="frame"),
        )
        alignment_blocks = align_by_time(report_transcription.segments, scenes, slide_frames)
    else:
        scenes = []

    structure_total = max(
        (
            len(alignment_blocks)
            if alignment_blocks is not None
            else len(report_transcription.segments)
        ),
        1,
    )
    structure_count_label = "blocks" if alignment_blocks is not None else "segments"
    active_reporter.progress_started(
        "structure",
        "Structuring report",
        total=float(structure_total),
        count_label=structure_count_label,
        detail="0 sections",
    )
    on_structure_progress, finish_structure_progress = _progress_updater(
        active_reporter,
        stage_key="structure",
    )
    timer = _start_stage_timer(stage_timings, "structure")
    report = build_report(
        media_asset,
        report_transcription,
        alignment_blocks=alignment_blocks,
        warnings=warnings,
        progress_callback=lambda completed_count, section_count: on_structure_progress(
            float(completed_count),
            detail=_count_label(section_count, singular="section"),
        ),
    )
    finish_structure_progress(
        float(structure_total),
        detail=_count_label(len(report.sections), singular="section"),
    )
    timer.finish()
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

    active_reporter.stage_started("export", "Writing artifacts")
    timer = _start_stage_timer(stage_timings, "export")
    _write_requested_artifacts(report, normalized_transcription, layout, output_format)
    timer.finish()
    active_reporter.stage_finished("export", "Writing artifacts", detail=output_format)

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
            "windows": asr_pipeline.window_count,
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


def _run_asr_pipeline(
    *,
    audio_path: Path,
    media_asset: MediaAsset,
    transcriber: WhisperCppTranscriber,
    layout: RunLayout,
    reporter: NullStageReporter,
    stage_timings: dict[str, float],
    warnings: list[str],
    asr_pipeline: AsrPipelineDiagnostics,
    vad_enabled: bool,
    vad_threshold: float,
    min_speech_duration_ms: int,
    min_silence_duration_ms: int,
    speech_region_pad_ms: int,
) -> _AsrPipelineResult:
    reporter.stage_started("prepare_asr", "Preparing ASR model")
    timer = _start_stage_timer(stage_timings, "prepare_asr")
    transcriber.prepare_model()
    timer.finish()
    reporter.stage_finished(
        "prepare_asr",
        "Preparing ASR model",
        detail=_asr_runtime_detail(transcriber),
    )

    audio_samples, sample_rate = load_normalized_audio(audio_path)
    asr_pipeline.normalized_audio_duration_sec = normalized_audio_duration(
        audio_samples,
        sample_rate,
    )
    reporter.progress_started(
        "vad",
        "Detecting speech regions",
        total=media_asset.duration_sec,
        count_label="s",
    )
    on_vad_progress, finish_vad_progress = _progress_updater(
        reporter,
        stage_key="vad",
    )

    timer = _start_stage_timer(stage_timings, "vad")
    speech_regions, vad_warnings = detect_speech_regions(
        audio_samples,
        sample_rate,
        enabled=vad_enabled,
        threshold=vad_threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=DEFAULT_SPEECH_PAD_MS,
        progress_callback=lambda completed_sec, detected_count: on_vad_progress(
            completed_sec,
            detail=_count_label(detected_count, singular="region"),
        ),
    )
    finish_vad_progress(
        media_asset.duration_sec,
        detail=_count_label(len(speech_regions), singular="region"),
    )
    timer.finish()
    asr_pipeline.vad_region_count = len(speech_regions)
    reporter.stage_finished(
        "vad",
        "Detecting speech regions",
        detail=_count_label(len(speech_regions), singular="region"),
    )
    for warning in vad_warnings:
        warnings.append(warning)
        reporter.warn(warning)
    _write_json(
        layout.speech_regions_path,
        {"speech_regions": [region.model_dump(mode="json") for region in speech_regions]},
    )

    reporter.stage_started("prepare_speech_regions", "Preparing speech regions")
    timer = _start_stage_timer(stage_timings, "prepare_speech_regions")
    repaired_regions = repair_speech_regions(speech_regions)
    expanded_regions = expand_speech_regions(
        repaired_regions,
        pad_ms=speech_region_pad_ms,
        audio_duration_sec=asr_pipeline.normalized_audio_duration_sec or 0.0,
    )
    windows = [
        InferenceWindow(
            window_id=f"window-{i + 1}",
            region_index=i,
            start_sec=region.start_sec,
            end_sec=region.end_sec,
            overlap_sec=0.0,
        )
        for i, region in enumerate(expanded_regions)
        if region.end_sec > region.start_sec
    ]
    _write_json(
        layout.expanded_regions_path,
        {"expanded_regions": [region.model_dump(mode="json") for region in expanded_regions]},
    )
    timer.finish()
    asr_pipeline.window_count = len(windows)
    asr_pipeline.average_window_duration_sec = (
        sum(w.end_sec - w.start_sec for w in windows) / len(windows) if windows else None
    )
    reporter.stage_finished(
        "prepare_speech_regions",
        "Preparing speech regions",
        detail=(
            f"{len(speech_regions)} -> {len(expanded_regions)} regions"
            f" | {_count_label(len(windows), singular='window')}"
        ),
    )

    reporter.progress_started(
        "transcribe",
        "Transcribing audio",
        total=media_asset.duration_sec,
        count_label="s",
        detail="0 segments",
    )
    on_window_completed, finish_transcribe_progress = _progress_updater(
        reporter,
        stage_key="transcribe",
    )

    timer = _start_stage_timer(stage_timings, "transcribe")
    decoded_windows = transcriber.transcribe_inference_windows(
        audio_samples,
        windows,
        progress_callback=lambda completed_sec, segment_count: on_window_completed(
            completed_sec,
            detail=_count_label(segment_count, singular="segment"),
        ),
    )
    _write_json(
        layout.decoded_windows_path,
        {"decoded_windows": [window.model_dump(mode="json") for window in decoded_windows]},
    )
    finish_transcribe_progress(media_asset.duration_sec)
    transcribe_elapsed_sec = timer.finish()
    reporter.stage_finished(
        "transcribe",
        "Transcribing audio",
        detail=_window_transcription_stage_detail(
            window_count=len(windows),
            total_duration_sec=media_asset.duration_sec,
            elapsed_sec=transcribe_elapsed_sec,
        ),
    )

    reporter.stage_started("reconcile", "Reconciling transcript windows")
    timer = _start_stage_timer(stage_timings, "reconcile")
    transcription, reconciliation_stats = reconcile_decoded_windows(decoded_windows)
    timer.finish()
    asr_pipeline.reconciliation_duplicate_segments_dropped = (
        reconciliation_stats.duplicate_segments_dropped
    )
    asr_pipeline.reconciliation_boundary_fixes = reconciliation_stats.boundary_fixes
    asr_pipeline.system_info = transcriber.system_info
    reporter.stage_finished(
        "reconcile",
        "Reconciling transcript windows",
        detail=f"{len(transcription.segments)} segments",
    )

    _write_json(layout.transcript_path, _transcription_payload(transcription))
    normalized_transcription = normalize_transcription(transcription)
    return _AsrPipelineResult(
        transcription=transcription,
        normalized_transcription=normalized_transcription,
    )


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
    timer = _start_stage_timer(stage_timings, "llm_report")
    try:
        section_result = llm_processor.polish_report_sections_with_progress(
            report,
            progress_callback=lambda advance: reporter.progress_advanced(
                "llm_report_sections",
                advance=float(advance),
            ),
        )
    except LLMProcessingError as error:
        report_latency_sec = timer.finish()
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
        report_latency_sec = timer.finish()
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

    report_latency_sec = timer.finish()
    usage = merge_usage(section_result.usage, metadata_result.usage)
    report.summary = metadata_result.summary
    report.action_items = metadata_result.action_items
    for section in report.sections:
        section.title = metadata_result.section_titles.get(section.id, section.title)
        section.tldr = section_result.section_tldrs.get(section.id, section.tldr)
        section.transcript_text = section_result.section_transcripts.get(
            section.id,
            section.transcript_text,
        )
    reporter.stage_finished(
        "llm_report",
        summary_label,
        detail=_llm_report_detail(
            section_count=len(section_result.section_transcripts),
            tldr_count=len(section_result.section_tldrs),
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


def _start_stage_timer(stage_timings: dict[str, float], key: str) -> _StageTimer:
    return _StageTimer(stage_timings=stage_timings, key=key, start_sec=perf_counter())


def _write_requested_artifacts(
    report: ReportDocument,
    transcription: TranscriptionResult,
    layout: RunLayout,
    output_format: str,
) -> None:
    formats = {"md", "docx", "json"} if output_format == "all" else {output_format}

    if "md" in formats:
        write_markdown_report(report, layout.markdown_report_path)
    if "docx" in formats:
        write_docx_report(report, layout.docx_report_path)

    write_json_report(report, layout.json_report_path)
    write_vtt_subtitles(transcription, layout.subtitle_vtt_path)


def _write_json(output_path: Path, payload: dict[str, object]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _progress_updater(
    reporter: NullStageReporter,
    *,
    stage_key: str,
):
    completed = 0.0

    def update(next_completed: float, *, detail: str | None = None) -> None:
        nonlocal completed
        advance = max(0.0, next_completed - completed)
        if advance > 0:
            reporter.progress_advanced(stage_key, advance=advance, detail=detail)
            completed = next_completed

    def finish(total: float, *, detail: str | None = None) -> None:
        update(total, detail=detail)

    return update, finish


def _count_label(count: int, *, singular: str, plural: str | None = None) -> str:
    resolved_plural = plural or f"{singular}s"
    label = singular if count == 1 else resolved_plural
    return f"{count} {label}"


def _configure_asr_logging(transcriber: WhisperCppTranscriber, layout: RunLayout) -> None:
    transcriber.set_log_path(layout.run_dir / "whisper-cpp.log")


def _transcription_payload(transcription: TranscriptionResult) -> dict[str, object]:
    return transcription.model_dump(mode="json")


def _asr_model_label(model_name: str) -> str:
    path = Path(model_name)
    if path.is_absolute():
        if repo_label := _hf_cache_repo_label(path):
            return f"{repo_label}/{path.name} (HF cache)"
        return path.name
    return model_name


def _hf_cache_repo_label(path: Path) -> str | None:
    for part in path.parts:
        if part.startswith("models--"):
            return part.removeprefix("models--").replace("--", "/")
    return None


def _asr_runtime_detail(transcriber: WhisperCppTranscriber) -> str:
    model_label = _asr_model_label(transcriber.model_name)
    return f"{model_label} | {transcriber.device_name}"


def _window_transcription_stage_detail(
    *,
    window_count: int,
    total_duration_sec: float,
    elapsed_sec: float,
) -> str:
    window_label = "window" if window_count == 1 else "windows"
    details = [f"{window_count} {window_label}"]
    if total_duration_sec > 0 and elapsed_sec > 0:
        details.append(f"RTF {elapsed_sec / total_duration_sec:.2f}")
    return " | ".join(details)


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
    *,
    section_count: int,
    tldr_count: int,
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
        _optional_count_detail(tldr_count, singular="TL;DR", plural="TL;DRs"),
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
