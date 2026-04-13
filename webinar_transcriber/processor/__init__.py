"""High-level processing orchestration."""

from __future__ import annotations

from contextlib import nullcontext, suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING

import webinar_transcriber.asr as asr_runtime
import webinar_transcriber.media as media_runtime
import webinar_transcriber.structure as structure_runtime
import webinar_transcriber.video as video_runtime
from webinar_transcriber.align import align_by_time
from webinar_transcriber.asr import DEFAULT_ASR_THREADS, PromptCarryoverSettings
from webinar_transcriber.labels import count_label
from webinar_transcriber.models import (
    AlignmentBlock,
    AsrPipelineDiagnostics,
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
    write_requested_artifacts,
)

if TYPE_CHECKING:
    from pathlib import Path

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


@dataclass(frozen=True)
class FrameExtractionArtifacts:
    """Runtime artifacts returned from a frame-extraction run."""

    layout: RunLayout
    media_asset: VideoAsset
    scenes: list[Scene]
    slide_frames: list[SlideFrame]


DEFAULT_VAD_SETTINGS = VadSettings()
DEFAULT_PROMPT_CARRYOVER_SETTINGS = PromptCarryoverSettings()


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
    current_stage: str | None = None
    layout: RunLayout | None = None
    media_asset: MediaAsset | None = None
    alignment_blocks: list[AlignmentBlock] | None = None
    scenes: list[Scene] = []
    slide_frames: list[SlideFrame] = []
    transcription: TranscriptionResult | None = None
    normalized_transcription: TranscriptionResult | None = None
    report: ReportDocument | None = None
    llm_runtime = LLMRuntimeState()
    asr_pipeline = AsrPipelineDiagnostics(vad_enabled=vad.enabled, threads=asr_threads)
    asr_pipeline.carryover_enabled = carryover.enabled

    active_transcriber = transcriber or asr_runtime.WhisperCppTranscriber(
        model_name=asr_model, threads=asr_threads, carryover_settings=carryover
    )
    transcriber_scope = (
        active_transcriber if transcriber is None else nullcontext(active_transcriber)
    )
    with transcriber_scope as active_transcriber:
        active_reporter.begin_run(input_path, output_format=output_format)
        try:
            current_stage = "prepare_run_dir"
            active_reporter.stage_started("prepare_run_dir", "Preparing run directory")
            timer = start_stage_timer(stage_timings, "prepare_run_dir")
            layout = create_run_layout(input_path=input_path, output_dir=output_dir)
            timer.finish()
            active_reporter.stage_finished(
                "prepare_run_dir", "Preparing run directory", detail=str(layout.run_dir)
            )
            configure_asr_logging(active_transcriber, layout)

            current_stage = "probe_media"
            active_reporter.stage_started("probe_media", "Probing media")
            timer = start_stage_timer(stage_timings, "probe_media")
            media_asset = media_runtime.probe_media(input_path)
            timer.finish()
            write_json(layout.metadata_path, {"media": media_asset.model_dump(mode="json")})
            active_reporter.stage_finished(
                "probe_media",
                "Probing media",
                detail=f"{media_asset.media_type.value}, {media_asset.duration_sec:.1f}s",
            )

            current_stage = "prepare_transcription_audio"
            active_reporter.stage_started("prepare_transcription_audio", "Preparing audio")
            timer = start_stage_timer(stage_timings, "prepare_transcription_audio")
            with prepared_transcription_audio(input_path) as audio_path:
                timer.finish()
                active_reporter.stage_finished(
                    "prepare_transcription_audio", "Preparing audio", detail=str(audio_path.name)
                )
                current_stage = "asr"
                asr_result = processor_asr.run_asr_pipeline(
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
                        audio_path, preserved_audio_path, audio_format=kept_audio_format
                    )
            # Subsequent video, structure, and export stages intentionally run after temp audio
            # cleanup.

            if isinstance(media_asset, VideoAsset):
                current_stage = "detect_scenes"
                active_reporter.progress_started(
                    "detect_scenes",
                    "Detecting scenes",
                    total=video_runtime.estimate_sample_count(media_asset.duration_sec),
                    count_label="s",
                    detail="0 scenes",
                )
                timer = start_stage_timer(stage_timings, "detect_scenes")
                scenes = video_runtime.detect_scenes(
                    input_path,
                    duration_sec=media_asset.duration_sec,
                    progress_callback=lambda scene_count: active_reporter.progress_advanced(
                        "detect_scenes", detail=count_label(scene_count, "scene")
                    ),
                )
                timer.finish()
                write_json(
                    layout.scenes_path,
                    {"scenes": [scene.model_dump(mode="json") for scene in scenes]},
                )
                active_reporter.stage_finished(
                    "detect_scenes", "Detecting scenes", detail=count_label(len(scenes), "scene")
                )

                current_stage = "extract_frames"
                active_reporter.progress_started(
                    "extract_frames", "Extracting slide frames", total=len(scenes)
                )
                timer = start_stage_timer(stage_timings, "extract_frames")
                slide_frames = video_runtime.extract_representative_frames(
                    input_path,
                    scenes,
                    layout.frames_dir,
                    progress_callback=lambda: active_reporter.progress_advanced("extract_frames"),
                )
                timer.finish()
                active_reporter.stage_finished(
                    "extract_frames",
                    "Extracting slide frames",
                    detail=count_label(len(slide_frames), "frame"),
                )
                alignment_blocks = align_by_time(
                    normalized_transcription.segments, scenes, slide_frames, warnings=warnings
                )

            structure_total = max(
                (
                    len(alignment_blocks)
                    if alignment_blocks is not None
                    else len(normalized_transcription.segments)
                ),
                1,
            )
            structure_count_label = "blocks" if alignment_blocks is not None else "segments"
            current_stage = "structure"
            active_reporter.progress_started(
                "structure",
                "Structuring report",
                total=float(structure_total),
                count_label=structure_count_label,
                detail="0 sections",
            )
            on_structure_progress, finish_structure_progress = progress_updater(
                active_reporter, stage_key="structure"
            )

            timer = start_stage_timer(stage_timings, "structure")
            report = structure_runtime.build_report(
                media_asset,
                normalized_transcription,
                alignment_blocks=alignment_blocks,
                warnings=warnings,
                progress_callback=lambda completed_count, section_count: on_structure_progress(
                    float(completed_count), detail=count_label(section_count, "section")
                ),
            )
            finish_structure_progress(
                float(structure_total), detail=count_label(len(report.sections), "section")
            )
            timer.finish()
            active_reporter.stage_finished(
                "structure",
                "Structuring report",
                detail=count_label(len(report.sections), "section"),
            )

            if slide_frames:
                frame_by_id = {frame.id: frame for frame in slide_frames}
                for section in report.sections:
                    if section.frame_id and section.frame_id in frame_by_id:
                        section.image_path = frame_by_id[section.frame_id].image_path

            current_stage = "llm_report"
            report, llm_runtime = maybe_polish_report(
                report,
                llm_processor=llm_enhancer,
                reporter=active_reporter,
                warnings=warnings,
                stage_timings=stage_timings,
                llm_runtime=llm_runtime,
            )
            report.warnings = list(warnings)

            current_stage = "export"
            active_reporter.stage_started("export", "Writing artifacts")
            timer = start_stage_timer(stage_timings, "export")
            write_requested_artifacts(report, normalized_transcription, layout, output_format)
            timer.finish()
            active_reporter.stage_finished("export", "Writing artifacts", detail=output_format)

            diagnostics = build_diagnostics(
                status="succeeded",
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
        except Exception as error:
            if layout is not None:
                diagnostics = build_diagnostics(
                    status="failed",
                    failed_stage=current_stage,
                    error=str(error),
                    asr_model=active_transcriber.model_name,
                    llm_enabled=enable_llm,
                    llm_model=llm_runtime.model_name,
                    llm_report_status=llm_runtime.report_status,
                    llm_report_latency_sec=llm_runtime.report_latency_sec,
                    llm_report_usage=llm_runtime.report_usage,
                    stage_timings=stage_timings,
                    asr_pipeline=asr_pipeline,
                    transcript_segment_count=len(transcription.segments) if transcription else 0,
                    normalized_transcript_segment_count=(
                        len(normalized_transcription.segments)
                        if normalized_transcription is not None
                        else 0
                    ),
                    report_section_count=len(report.sections) if report is not None else 0,
                    scene_count=len(scenes),
                    frame_count=len(slide_frames),
                    warnings=warnings,
                )
                with suppress(Exception):
                    write_json(layout.diagnostics_path, diagnostics.model_dump(mode="json"))
            raise


def extract_frames_input(
    input_path: Path, *, output_dir: Path | None = None, reporter: StageReporter | None = None
) -> FrameExtractionArtifacts:
    """Detect scenes and extract representative frames from a video input."""
    active_reporter = reporter or NullStageReporter()

    active_reporter.begin_run(input_path, output_format="frames")
    active_reporter.stage_started("prepare_run_dir", "Preparing run directory")
    layout = create_run_layout(input_path=input_path, output_dir=output_dir)
    active_reporter.stage_finished(
        "prepare_run_dir", "Preparing run directory", detail=str(layout.run_dir)
    )

    active_reporter.stage_started("probe_media", "Probing media")
    media_asset = media_runtime.probe_media(input_path)
    active_reporter.stage_finished(
        "probe_media",
        "Probing media",
        detail=f"{media_asset.media_type.value}, {media_asset.duration_sec:.1f}s",
    )
    if not isinstance(media_asset, VideoAsset):
        raise media_runtime.MediaProcessingError(
            "Frame extraction is only supported for video input."
        )

    active_reporter.progress_started(
        "detect_scenes",
        "Detecting scenes",
        total=video_runtime.estimate_sample_count(media_asset.duration_sec),
        count_label="s",
        detail="0 scenes",
    )
    scenes = video_runtime.detect_scenes(
        input_path,
        duration_sec=media_asset.duration_sec,
        progress_callback=lambda scene_count: active_reporter.progress_advanced(
            "detect_scenes", detail=count_label(scene_count, "scene")
        ),
    )
    active_reporter.stage_finished(
        "detect_scenes", "Detecting scenes", detail=count_label(len(scenes), "scene")
    )

    active_reporter.progress_started("extract_frames", "Extracting slide frames", total=len(scenes))
    slide_frames = video_runtime.extract_representative_frames(
        input_path,
        scenes,
        layout.frames_dir,
        progress_callback=lambda: active_reporter.progress_advanced("extract_frames"),
    )
    active_reporter.stage_finished(
        "extract_frames", "Extracting slide frames", detail=count_label(len(slide_frames), "frame")
    )
    write_json(layout.scenes_path, {"scenes": [scene.model_dump(mode="json") for scene in scenes]})
    return FrameExtractionArtifacts(
        layout=layout, media_asset=media_asset, scenes=scenes, slide_frames=slide_frames
    )
