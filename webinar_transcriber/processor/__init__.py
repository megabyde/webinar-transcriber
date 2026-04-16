"""High-level processing orchestration."""

from __future__ import annotations

from contextlib import nullcontext, suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import webinar_transcriber.asr as asr_runtime
import webinar_transcriber.export as export_runtime
import webinar_transcriber.media as media_runtime
import webinar_transcriber.structure as structure_runtime
import webinar_transcriber.video as video_runtime
from webinar_transcriber.align import align_by_time
from webinar_transcriber.asr import PromptCarryoverSettings, default_asr_threads
from webinar_transcriber.labels import count_label
from webinar_transcriber.models import (
    AlignmentBlock,
    Diagnostics,
    MediaAsset,
    ReportDocument,
    Scene,
    SlideFrame,
    TranscriptionResult,
    VideoAsset,
)
from webinar_transcriber.normalized_audio import (
    prepared_transcription_audio,
    preserve_transcription_audio,
)
from webinar_transcriber.paths import RunLayout, create_run_layout
from webinar_transcriber.reporter import NullStageReporter, StageReporter
from webinar_transcriber.segmentation import VadSettings

from . import asr as processor_asr
from .llm import LLMRuntimeState, maybe_polish_report, resolve_llm_processor
from .support import (
    build_diagnostics,
    configure_asr_logging,
    progress_updater,
    start_stage_timer,
    write_json,
    write_model_json,
)

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Literal

    from webinar_transcriber.asr import WhisperCppTranscriber
    from webinar_transcriber.llm import LLMProcessor


@dataclass(frozen=True)
class ProcessArtifacts:
    """Runtime artifacts returned from a processing run."""

    layout: RunLayout
    media_asset: MediaAsset
    transcription: TranscriptionResult
    report: ReportDocument
    diagnostics: Diagnostics


@dataclass
class _RunContext:
    """Mutable state for one processing run."""

    reporter: StageReporter
    asr_pipeline: _AsrPipelineState
    stage_timings: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    current_stage: str | None = None
    layout: RunLayout | None = None
    media_asset: MediaAsset | None = None
    alignment_blocks: list[AlignmentBlock] | None = None
    scenes: list[Scene] = field(default_factory=list)
    slide_frames: list[SlideFrame] = field(default_factory=list)
    transcription: TranscriptionResult | None = None
    normalized_transcription: TranscriptionResult | None = None
    report: ReportDocument | None = None
    llm_runtime: LLMRuntimeState = field(default_factory=LLMRuntimeState)


DEFAULT_VAD_SETTINGS = VadSettings()
DEFAULT_PROMPT_CARRYOVER_SETTINGS = PromptCarryoverSettings()


@dataclass
class _AsrPipelineState:
    """Mutable ASR diagnostics state accumulated during processing."""

    vad_enabled: bool
    threads: int
    normalized_audio_duration_sec: float | None = None
    vad_region_count: int = 0
    carryover_enabled: bool = False
    window_count: int = 0
    average_window_duration_sec: float | None = None
    reconciliation_duplicate_segments_dropped: int = 0
    reconciliation_boundary_fixes: int = 0
    system_info: str | None = None


def _build_run_diagnostics(
    ctx: _RunContext,
    *,
    status: Literal["succeeded", "failed"],
    asr_model: str | None,
    llm_enabled: bool,
    failed_stage: str | None = None,
    error: str | None = None,
) -> Diagnostics:
    """Build one diagnostics payload from the accumulated run state."""
    return build_diagnostics(
        status=status,
        failed_stage=failed_stage,
        error=error,
        asr_model=asr_model,
        llm_enabled=llm_enabled,
        llm_model=ctx.llm_runtime.model_name,
        llm_report_status=ctx.llm_runtime.report_status,
        llm_report_latency_sec=ctx.llm_runtime.report_latency_sec,
        llm_report_usage=ctx.llm_runtime.report_usage,
        stage_timings=ctx.stage_timings,
        asr_pipeline=ctx.asr_pipeline,
        transcript_segment_count=len(ctx.transcription.segments) if ctx.transcription else 0,
        normalized_transcript_segment_count=(
            len(ctx.normalized_transcription.segments) if ctx.normalized_transcription else 0
        ),
        report_section_count=len(ctx.report.sections) if ctx.report else 0,
        scene_count=len(ctx.scenes),
        frame_count=len(ctx.slide_frames),
        warnings=ctx.warnings,
    )


def _write_run_diagnostics(
    ctx: _RunContext,
    *,
    status: Literal["succeeded", "failed"],
    asr_model: str | None,
    llm_enabled: bool,
    failed_stage: str | None = None,
    error: str | None = None,
    suppress_errors: bool = False,
) -> Diagnostics | None:
    """Write diagnostics when a run layout exists and return the payload."""
    if ctx.layout is None:
        return None
    diagnostics = _build_run_diagnostics(
        ctx,
        status=status,
        asr_model=asr_model,
        llm_enabled=llm_enabled,
        failed_stage=failed_stage,
        error=error,
    )
    error_scope = suppress(Exception) if suppress_errors else nullcontext()
    with error_scope:
        write_model_json(ctx.layout.diagnostics_path, diagnostics)
    return diagnostics


def process_input(
    input_path: Path,
    *,
    output_dir: Path | None = None,
    asr_model: str | None = None,
    vad: VadSettings = DEFAULT_VAD_SETTINGS,
    carryover: PromptCarryoverSettings = DEFAULT_PROMPT_CARRYOVER_SETTINGS,
    asr_threads: int | None = None,
    keep_audio: bool = False,
    kept_audio_format: str = "wav",
    enable_llm: bool = False,
    transcriber: WhisperCppTranscriber | None = None,
    llm_processor: LLMProcessor | None = None,
    reporter: StageReporter | None = None,
) -> ProcessArtifacts:
    """Process a single audio or video file into report artifacts."""
    active_reporter = reporter or NullStageReporter()
    asr_threads = asr_threads or default_asr_threads()
    asr_pipeline = _AsrPipelineState(vad_enabled=vad.enabled, threads=asr_threads)
    asr_pipeline.carryover_enabled = carryover.enabled
    ctx = _RunContext(reporter=active_reporter, asr_pipeline=asr_pipeline)

    active_transcriber = transcriber or asr_runtime.WhisperCppTranscriber(
        model_name=asr_model, threads=asr_threads, carryover_settings=carryover
    )
    transcriber_scope = (
        active_transcriber if transcriber is None else nullcontext(active_transcriber)
    )
    with transcriber_scope as active_transcriber:

        def record_warning(message: str) -> None:
            ctx.warnings.append(message)
            ctx.reporter.warn(message)

        ctx.reporter.begin_run(input_path)
        try:
            ctx.current_stage = "prepare_run_dir"
            ctx.reporter.stage_started("prepare_run_dir", "Preparing run directory")
            timer = start_stage_timer(ctx.stage_timings, "prepare_run_dir")
            ctx.layout = create_run_layout(input_path=input_path, output_dir=output_dir)
            timer.finish()
            ctx.reporter.stage_finished(
                "prepare_run_dir", "Preparing run directory", detail=str(ctx.layout.run_dir)
            )
            configure_asr_logging(active_transcriber, ctx.layout)

            ctx.current_stage = "probe_media"
            ctx.reporter.stage_started("probe_media", "Probing media")
            timer = start_stage_timer(ctx.stage_timings, "probe_media")
            ctx.media_asset = media_runtime.probe_media(input_path)
            timer.finish()
            write_json(ctx.layout.metadata_path, {"media": ctx.media_asset.model_dump(mode="json")})
            ctx.reporter.stage_finished(
                "probe_media",
                "Probing media",
                detail=f"{ctx.media_asset.media_type.value}, {ctx.media_asset.duration_sec:.1f}s",
            )

            ctx.current_stage = "prepare_transcription_audio"
            ctx.reporter.stage_started("prepare_transcription_audio", "Preparing audio")
            timer = start_stage_timer(ctx.stage_timings, "prepare_transcription_audio")
            with prepared_transcription_audio(input_path) as audio_path:
                timer.finish()
                ctx.reporter.stage_finished(
                    "prepare_transcription_audio", "Preparing audio", detail=str(audio_path.name)
                )
                ctx.current_stage = "asr"
                asr_result = processor_asr.run_asr_pipeline(
                    audio_path=audio_path,
                    media_asset=ctx.media_asset,
                    transcriber=active_transcriber,
                    layout=ctx.layout,
                    reporter=ctx.reporter,
                    stage_timings=ctx.stage_timings,
                    warnings=ctx.warnings,
                    asr_pipeline=ctx.asr_pipeline,
                    vad=vad,
                )
                ctx.transcription = asr_result.transcription
                ctx.normalized_transcription = asr_result.normalized_transcription

                llm_enhancer, ctx.llm_runtime = resolve_llm_processor(
                    enable_llm=enable_llm,
                    llm_processor=llm_processor,
                    reporter=ctx.reporter,
                    warnings=ctx.warnings,
                    llm_runtime=ctx.llm_runtime,
                )
                if keep_audio:
                    preserved_audio_path = ctx.layout.transcription_audio_path(kept_audio_format)
                    preserve_transcription_audio(
                        audio_path, preserved_audio_path, audio_format=kept_audio_format
                    )
            # Subsequent video, structure, and export stages intentionally run after temp audio
            # cleanup.

            if isinstance(ctx.media_asset, VideoAsset):
                ctx.current_stage = "detect_scenes"
                detect_scene_total = video_runtime.estimate_sample_count(
                    ctx.media_asset.duration_sec
                )
                ctx.reporter.progress_started(
                    "detect_scenes",
                    "Detecting scenes",
                    total=detect_scene_total,
                    count_label="s",
                    detail="0 scenes",
                )
                on_detect_scenes_progress, finish_detect_scenes_progress = progress_updater(
                    ctx.reporter, stage_key="detect_scenes"
                )
                timer = start_stage_timer(ctx.stage_timings, "detect_scenes")
                ctx.scenes = video_runtime.detect_scenes(
                    input_path,
                    duration_sec=ctx.media_asset.duration_sec,
                    progress_callback=lambda scene_count: on_detect_scenes_progress(
                        scene_count,
                        detail=count_label(scene_count, "scene"),
                    ),
                )
                finish_detect_scenes_progress(
                    detect_scene_total, detail=count_label(len(ctx.scenes), "scene")
                )
                timer.finish()
                write_json(
                    ctx.layout.scenes_path,
                    {"scenes": [scene.model_dump(mode="json") for scene in ctx.scenes]},
                )
                ctx.reporter.stage_finished(
                    "detect_scenes",
                    "Detecting scenes",
                    detail=count_label(len(ctx.scenes), "scene"),
                )

                ctx.current_stage = "extract_frames"
                ctx.reporter.progress_started(
                    "extract_frames", "Extracting slide frames", total=len(ctx.scenes)
                )
                timer = start_stage_timer(ctx.stage_timings, "extract_frames")
                ctx.slide_frames = video_runtime.extract_representative_frames(
                    input_path,
                    ctx.scenes,
                    ctx.layout.frames_dir,
                    progress_callback=lambda: ctx.reporter.progress_advanced("extract_frames"),
                    warning_callback=record_warning,
                )
                timer.finish()
                ctx.reporter.stage_finished(
                    "extract_frames",
                    "Extracting slide frames",
                    detail=count_label(len(ctx.slide_frames), "frame"),
                )
                ctx.alignment_blocks = align_by_time(
                    ctx.normalized_transcription.segments,
                    ctx.scenes,
                    ctx.slide_frames,
                    warnings=ctx.warnings,
                )

            structure_total = max(
                (
                    len(ctx.alignment_blocks)
                    if ctx.alignment_blocks is not None
                    else len(ctx.normalized_transcription.segments)
                ),
                1,
            )
            structure_count_label = "blocks" if ctx.alignment_blocks is not None else "segments"
            ctx.current_stage = "structure"
            ctx.reporter.progress_started(
                "structure",
                "Structuring report",
                total=float(structure_total),
                count_label=structure_count_label,
                detail="0 sections",
            )
            on_structure_progress, finish_structure_progress = progress_updater(
                ctx.reporter, stage_key="structure"
            )

            timer = start_stage_timer(ctx.stage_timings, "structure")
            ctx.report = structure_runtime.build_report(
                ctx.media_asset,
                ctx.normalized_transcription,
                alignment_blocks=ctx.alignment_blocks,
                warnings=ctx.warnings,
                progress_callback=lambda completed_count, section_count: on_structure_progress(
                    float(completed_count), detail=count_label(section_count, "section")
                ),
            )
            finish_structure_progress(
                float(structure_total), detail=count_label(len(ctx.report.sections), "section")
            )
            timer.finish()
            ctx.reporter.stage_finished(
                "structure",
                "Structuring report",
                detail=count_label(len(ctx.report.sections), "section"),
            )

            if ctx.slide_frames:
                frame_by_id = {frame.id: frame for frame in ctx.slide_frames}
                ctx.report = ctx.report.model_copy(
                    update={
                        "sections": [
                            section.model_copy(
                                update={
                                    "image_path": frame_by_id[section.frame_id].image_path,
                                }
                            )
                            if section.frame_id and section.frame_id in frame_by_id
                            else section
                            for section in ctx.report.sections
                        ]
                    }
                )

            ctx.current_stage = "llm_report"
            ctx.report, ctx.llm_runtime = maybe_polish_report(
                ctx.report,
                llm_processor=llm_enhancer,
                reporter=ctx.reporter,
                warnings=ctx.warnings,
                stage_timings=ctx.stage_timings,
                llm_runtime=ctx.llm_runtime,
            )
            ctx.report = ctx.report.model_copy(update={"warnings": list(ctx.warnings)})

            ctx.current_stage = "export"
            ctx.reporter.stage_started("export", "Writing artifacts")
            timer = start_stage_timer(ctx.stage_timings, "export")
            export_runtime.write_markdown_report(ctx.report, ctx.layout.markdown_report_path)
            export_runtime.write_docx_report(
                ctx.report,
                ctx.layout.docx_report_path,
                warning_callback=record_warning,
            )
            export_runtime.write_json_report(ctx.report, ctx.layout.json_report_path)
            export_runtime.write_vtt_subtitles(
                ctx.normalized_transcription, ctx.layout.subtitle_vtt_path
            )
            timer.finish()
            ctx.reporter.stage_finished("export", "Writing artifacts")

            diagnostics = _write_run_diagnostics(
                ctx,
                status="succeeded",
                asr_model=active_transcriber.model_name,
                llm_enabled=enable_llm,
            )
            diagnostics = cast("Diagnostics", diagnostics)

            artifacts = ProcessArtifacts(
                layout=ctx.layout,
                media_asset=ctx.media_asset,
                transcription=ctx.transcription,
                report=ctx.report,
                diagnostics=diagnostics,
            )
            ctx.reporter.complete_run(artifacts)
            return artifacts
        except Exception as ex:
            _write_run_diagnostics(
                ctx,
                status="failed",
                failed_stage=ctx.current_stage,
                error=str(ex),
                asr_model=active_transcriber.model_name,
                llm_enabled=enable_llm,
                suppress_errors=True,
            )
            raise
