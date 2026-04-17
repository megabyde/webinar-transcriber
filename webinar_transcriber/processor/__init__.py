"""High-level processing orchestration."""

from __future__ import annotations

from contextlib import ExitStack, nullcontext, suppress
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
from webinar_transcriber.reporter import NullStageReporter
from webinar_transcriber.segmentation import VadSettings

from . import asr as processor_asr
from .llm import LLMRuntimeState, maybe_polish_report, resolve_llm_processor
from .support import (
    build_diagnostics,
    configure_asr_logging,
    progress_stage,
    stage,
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

    reporter: NullStageReporter
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
    """Build one diagnostics payload from the accumulated run state.

    Returns:
        Diagnostics: The assembled run diagnostics payload.
    """
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
    """Write diagnostics when a run layout exists and return the payload.

    Returns:
        Diagnostics | None: The written diagnostics payload, if a run layout exists.
    """
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
    reporter: NullStageReporter | None = None,
) -> ProcessArtifacts:
    """Process a single audio or video file into report artifacts.

    Returns:
        ProcessArtifacts: The completed processing artifacts.
    """
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
            with stage(ctx, "prepare_run_dir", "Preparing run directory") as st:
                layout = create_run_layout(input_path=input_path, output_dir=output_dir)
                ctx.layout = layout
                st.detail = str(layout.run_dir)
            configure_asr_logging(active_transcriber, layout)

            with stage(ctx, "probe_media", "Probing media") as st:
                media_asset = media_runtime.probe_media(input_path)
                ctx.media_asset = media_asset
                write_json(layout.metadata_path, {"media": media_asset.model_dump(mode="json")})
                st.detail = f"{media_asset.media_type.value}, {media_asset.duration_sec:.1f}s"

            with ExitStack() as audio_scope:
                with stage(ctx, "prepare_transcription_audio", "Preparing audio") as st:
                    audio_path = audio_scope.enter_context(prepared_transcription_audio(input_path))
                    st.detail = str(audio_path.name)
                asr_result = processor_asr.run_asr_pipeline(
                    audio_path=audio_path,
                    media_asset=media_asset,
                    transcriber=active_transcriber,
                    layout=layout,
                    ctx=ctx,
                    warnings=ctx.warnings,
                    asr_pipeline=ctx.asr_pipeline,
                    vad=vad,
                )
                transcription = asr_result.transcription
                normalized_transcription = asr_result.normalized_transcription
                ctx.transcription = transcription
                ctx.normalized_transcription = normalized_transcription

                llm_enhancer, ctx.llm_runtime = resolve_llm_processor(
                    enable_llm=enable_llm,
                    llm_processor=llm_processor,
                    reporter=ctx.reporter,
                    warnings=ctx.warnings,
                    llm_runtime=ctx.llm_runtime,
                )
                if keep_audio:
                    preserved_audio_path = layout.transcription_audio_path(kept_audio_format)
                    preserve_transcription_audio(
                        audio_path, preserved_audio_path, audio_format=kept_audio_format
                    )
            # Subsequent video, structure, and export stages intentionally run after temp audio
            # cleanup.

            if isinstance(media_asset, VideoAsset):
                detect_scene_total = video_runtime.estimate_sample_count(media_asset.duration_sec)
                with progress_stage(
                    ctx,
                    "detect_scenes",
                    "Detecting scenes",
                    total=detect_scene_total,
                    count_label="s",
                    detail="0 scenes",
                ) as st:
                    ctx.scenes = video_runtime.detect_scenes(
                        input_path,
                        duration_sec=media_asset.duration_sec,
                        progress_callback=lambda scene_count: st.advance_to(
                            scene_count,
                            detail=count_label(scene_count, "scene"),
                        ),
                    )
                    st.finish_progress(
                        detect_scene_total, detail=count_label(len(ctx.scenes), "scene")
                    )
                write_json(
                    layout.scenes_path,
                    {"scenes": [scene.model_dump(mode="json") for scene in ctx.scenes]},
                )
                with progress_stage(
                    ctx,
                    "extract_frames",
                    "Extracting slide frames",
                    total=float(len(ctx.scenes)),
                ) as st:
                    ctx.slide_frames = video_runtime.extract_representative_frames(
                        input_path,
                        ctx.scenes,
                        layout.frames_dir,
                        progress_callback=st.advance,
                        warning_callback=record_warning,
                    )
                    st.detail = count_label(len(ctx.slide_frames), "frame")
                ctx.alignment_blocks = align_by_time(
                    normalized_transcription.segments,
                    ctx.scenes,
                    ctx.slide_frames,
                )

            structure_total = max(
                (
                    len(ctx.alignment_blocks)
                    if ctx.alignment_blocks is not None
                    else len(normalized_transcription.segments)
                ),
                1,
            )
            structure_count_label = "blocks" if ctx.alignment_blocks is not None else "segments"
            with progress_stage(
                ctx,
                "structure",
                "Structuring report",
                total=float(structure_total),
                count_label=structure_count_label,
                detail="0 sections",
            ) as st:
                report = structure_runtime.build_report(
                    media_asset,
                    normalized_transcription,
                    alignment_blocks=ctx.alignment_blocks,
                    warnings=ctx.warnings,
                    progress_callback=lambda completed_count, section_count: st.advance_to(
                        float(completed_count), detail=count_label(section_count, "section")
                    ),
                )
                st.finish_progress(
                    float(structure_total), detail=count_label(len(report.sections), "section")
                )
            ctx.report = report

            if ctx.slide_frames:
                frame_by_id = {frame.id: frame for frame in ctx.slide_frames}
                report = report.model_copy(
                    update={
                        "sections": [
                            section.model_copy(
                                update={
                                    "image_path": frame_by_id[section.frame_id].image_path,
                                }
                            )
                            if section.frame_id and section.frame_id in frame_by_id
                            else section
                            for section in report.sections
                        ]
                    }
                )
                ctx.report = report

            report, ctx.llm_runtime = maybe_polish_report(
                report,
                llm_processor=llm_enhancer,
                ctx=ctx,
                warnings=ctx.warnings,
                llm_runtime=ctx.llm_runtime,
            )
            report = report.model_copy(update={"warnings": list(ctx.warnings)})
            ctx.report = report

            with stage(ctx, "export", "Writing artifacts"):
                export_runtime.write_markdown_report(report, layout.markdown_report_path)
                export_runtime.write_docx_report(
                    report,
                    layout.docx_report_path,
                    warning_callback=record_warning,
                )
                export_runtime.write_json_report(report, layout.json_report_path)
                export_runtime.write_vtt_subtitles(
                    normalized_transcription, layout.subtitle_vtt_path
                )

            diagnostics = _write_run_diagnostics(
                ctx,
                status="succeeded",
                asr_model=active_transcriber.model_name,
                llm_enabled=enable_llm,
            )
            diagnostics = cast("Diagnostics", diagnostics)

            artifacts = ProcessArtifacts(
                layout=layout,
                media_asset=media_asset,
                transcription=transcription,
                report=report,
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
