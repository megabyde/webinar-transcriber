"""High-level processing orchestration."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING

from webinar_transcriber.align import align_by_time
from webinar_transcriber.asr import (
    DEFAULT_ASR_THREADS,
    PromptCarryoverSettings,
    WhisperCppTranscriber,
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
    VideoAsset,
)
from webinar_transcriber.paths import RunLayout, create_run_layout
from webinar_transcriber.reporter import NullStageReporter, StageReporter
from webinar_transcriber.segmentation import VadSettings
from webinar_transcriber.structure import build_report
from webinar_transcriber.transcription_audio import (
    prepared_transcription_audio,
    preserve_transcription_audio,
)
from webinar_transcriber.video import (
    detect_scenes,
    estimate_sample_count,
    extract_representative_frames,
)

from . import support
from .asr import run_asr_pipeline
from .llm import (
    LLMRuntimeState,
    maybe_polish_report,
    resolve_llm_processor,
)
from .support import (
    build_diagnostics,
    configure_asr_logging,
    count_label,
    progress_updater,
    start_stage_timer,
    write_json,
    write_requested_artifacts,
)

if TYPE_CHECKING:
    from pathlib import Path

    from webinar_transcriber.llm import LLMProcessor


@dataclass(frozen=True)
class ProcessArtifacts:
    """Runtime artifacts returned from a processing run."""

    layout: RunLayout
    media_asset: MediaAsset
    transcription: TranscriptionResult
    report: ReportDocument
    diagnostics: Diagnostics


DEFAULT_VAD_SETTINGS = VadSettings()
DEFAULT_PROMPT_CARRYOVER_SETTINGS = PromptCarryoverSettings()
_asr_runtime_detail = support.asr_runtime_detail
_window_transcription_stage_detail = support.window_transcription_stage_detail


def process_input(
    input_path: Path,
    *,
    output_dir: Path | None = None,
    output_format: str = "all",
    asr_model: str | None = None,
    vad: VadSettings = DEFAULT_VAD_SETTINGS,
    carryover: PromptCarryoverSettings = DEFAULT_PROMPT_CARRYOVER_SETTINGS,
    asr_threads: int = DEFAULT_ASR_THREADS,
    keep_audio: bool = False,
    kept_audio_format: str = "wav",
    enable_llm: bool = False,
    transcriber: WhisperCppTranscriber | None = None,
    llm_processor: LLMProcessor | None = None,
    reporter: StageReporter | None = None,
) -> ProcessArtifacts:
    """Process a single audio or video file into report artifacts."""
    active_reporter = reporter or NullStageReporter()
    stage_timings: dict[str, float] = {}
    warnings: list[str] = []
    alignment_blocks: list[AlignmentBlock] | None = None
    slide_frames: list[SlideFrame] = []
    llm_runtime = LLMRuntimeState()
    asr_pipeline = AsrPipelineDiagnostics(vad_enabled=vad.enabled, threads=asr_threads)
    asr_pipeline.carryover_enabled = carryover.enabled

    active_transcriber = transcriber or WhisperCppTranscriber(
        model_name=asr_model,
        threads=asr_threads,
        carryover_settings=carryover,
    )
    transcriber_scope = (
        active_transcriber if transcriber is None else nullcontext(active_transcriber)
    )
    with transcriber_scope as active_transcriber:
        active_reporter.begin_run(input_path, output_format=output_format)

        active_reporter.stage_started("prepare_run_dir", "Preparing run directory")
        timer = start_stage_timer(stage_timings, "prepare_run_dir")
        layout = create_run_layout(input_path=input_path, output_dir=output_dir)
        timer.finish()
        active_reporter.stage_finished(
            "prepare_run_dir",
            "Preparing run directory",
            detail=str(layout.run_dir),
        )
        configure_asr_logging(active_transcriber, layout)

        active_reporter.stage_started("probe_media", "Probing media")
        timer = start_stage_timer(stage_timings, "probe_media")
        media_asset = probe_media(input_path)
        timer.finish()
        write_json(layout.metadata_path, {"media": media_asset.model_dump(mode="json")})
        active_reporter.stage_finished(
            "probe_media",
            "Probing media",
            detail=f"{media_asset.media_type.value}, {media_asset.duration_sec:.1f}s",
        )

        active_reporter.stage_started("prepare_transcription_audio", "Preparing audio")
        timer = start_stage_timer(stage_timings, "prepare_transcription_audio")
        with prepared_transcription_audio(input_path) as audio_path:
            timer.finish()
            active_reporter.stage_finished(
                "prepare_transcription_audio",
                "Preparing audio",
                detail=str(audio_path.name),
            )
            asr_result = run_asr_pipeline(
                audio_path=audio_path,
                media_asset=media_asset,
                transcriber=active_transcriber,
                layout=layout,
                reporter=active_reporter,
                stage_timings=stage_timings,
                warnings=warnings,
                asr_pipeline=asr_pipeline,
                vad=vad,
            )
            transcription = asr_result.transcription
            normalized_transcription = asr_result.normalized_transcription

            llm_enhancer, llm_runtime = resolve_llm_processor(
                enable_llm=enable_llm,
                llm_processor=llm_processor,
                reporter=active_reporter,
                warnings=warnings,
                llm_runtime=llm_runtime,
            )
            if keep_audio:
                preserved_audio_path = layout.transcription_audio_path(kept_audio_format)
                preserve_transcription_audio(
                    audio_path,
                    preserved_audio_path,
                    audio_format=kept_audio_format,
                )
        # Subsequent video, structure, and export stages intentionally run after temp audio cleanup.

        if isinstance(media_asset, VideoAsset):
            active_reporter.progress_started(
                "detect_scenes",
                "Detecting scenes",
                total=estimate_sample_count(media_asset.duration_sec),
                count_label="s",
                detail="0 scenes",
            )
            timer = start_stage_timer(stage_timings, "detect_scenes")
            scenes = detect_scenes(
                input_path,
                duration_sec=media_asset.duration_sec,
                progress_callback=lambda scene_count: active_reporter.progress_advanced(
                    "detect_scenes",
                    detail=count_label(scene_count, singular="scene"),
                ),
            )
            timer.finish()
            write_json(
                layout.scenes_path,
                {"scenes": [scene.model_dump(mode="json") for scene in scenes]},
            )
            active_reporter.stage_finished(
                "detect_scenes",
                "Detecting scenes",
                detail=count_label(len(scenes), singular="scene"),
            )

            active_reporter.progress_started(
                "extract_frames",
                "Extracting slide frames",
                total=len(scenes),
            )
            timer = start_stage_timer(stage_timings, "extract_frames")
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
                detail=count_label(len(slide_frames), singular="frame"),
            )
            alignment_blocks = align_by_time(
                normalized_transcription.segments,
                scenes,
                slide_frames,
                warnings=warnings,
            )
        else:
            scenes = []

        structure_total = max(
            (
                len(alignment_blocks)
                if alignment_blocks is not None
                else len(normalized_transcription.segments)
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
        on_structure_progress, finish_structure_progress = progress_updater(
            active_reporter,
            stage_key="structure",
        )

        timer = start_stage_timer(stage_timings, "structure")
        report = build_report(
            media_asset,
            normalized_transcription,
            alignment_blocks=alignment_blocks,
            warnings=warnings,
            progress_callback=lambda completed_count, section_count: on_structure_progress(
                float(completed_count),
                detail=count_label(section_count, singular="section"),
            ),
        )
        finish_structure_progress(
            float(structure_total),
            detail=count_label(len(report.sections), singular="section"),
        )
        timer.finish()
        active_reporter.stage_finished(
            "structure",
            "Structuring report",
            detail=count_label(len(report.sections), singular="section"),
        )

        if slide_frames:
            frame_by_id = {frame.id: frame for frame in slide_frames}
            for section in report.sections:
                if section.frame_id and section.frame_id in frame_by_id:
                    section.image_path = frame_by_id[section.frame_id].image_path

        report, llm_runtime = maybe_polish_report(
            report,
            llm_processor=llm_enhancer,
            reporter=active_reporter,
            warnings=warnings,
            stage_timings=stage_timings,
            llm_runtime=llm_runtime,
        )
        report.warnings = list(warnings)

        active_reporter.stage_started("export", "Writing artifacts")
        timer = start_stage_timer(stage_timings, "export")
        write_requested_artifacts(report, normalized_transcription, layout, output_format)
        timer.finish()
        active_reporter.stage_finished("export", "Writing artifacts", detail=output_format)

        diagnostics = build_diagnostics(
            asr_model=active_transcriber.model_name,
            llm_enabled=enable_llm,
            llm_model=llm_runtime.model_name,
            llm_report_status=llm_runtime.report_status,
            llm_report_latency_sec=llm_runtime.report_latency_sec,
            llm_report_usage=llm_runtime.report_usage,
            stage_timings=stage_timings,
            asr_pipeline=asr_pipeline,
            transcript_segment_count=len(transcription.segments),
            normalized_transcript_segment_count=len(normalized_transcription.segments),
            report_section_count=len(report.sections),
            scene_count=len(scenes),
            frame_count=len(slide_frames),
            warnings=warnings,
        )
        write_json(layout.diagnostics_path, diagnostics.model_dump(mode="json"))

        artifacts = ProcessArtifacts(
            layout=layout,
            media_asset=media_asset,
            transcription=transcription,
            report=report,
            diagnostics=diagnostics,
        )
        active_reporter.complete_run(artifacts)
        return artifacts
