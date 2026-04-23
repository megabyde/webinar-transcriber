"""Private report-phase orchestration helpers."""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import TYPE_CHECKING

import webinar_transcriber.export as export_runtime
import webinar_transcriber.structure as structure_runtime
import webinar_transcriber.video as video_runtime
from webinar_transcriber.align import align_by_time
from webinar_transcriber.models import VideoAsset

from .llm import maybe_polish_report, resolve_llm_processor
from .support import count_label, progress_stage, stage, write_json
from .types import ReportPhaseResult

if TYPE_CHECKING:
    from pathlib import Path

    from webinar_transcriber.llm import LLMProcessor
    from webinar_transcriber.models import MediaAsset, TranscriptionResult
    from webinar_transcriber.paths import RunLayout

    from .types import RunContext


def run_report_phase(
    *,
    input_path: Path,
    layout: RunLayout,
    media_asset: MediaAsset,
    normalized_transcription: TranscriptionResult,
    enable_llm: bool,
    llm_processor: LLMProcessor | None,
    ctx: RunContext,
) -> ReportPhaseResult:
    """Run the video, structure, optional LLM, and export half of the pipeline.

    Returns:
        ReportPhaseResult: The report artifacts and optional video alignment data.
    """
    llm_enhancer, ctx.llm_runtime = resolve_llm_processor(
        enable_llm=enable_llm,
        llm_processor=llm_processor,
        reporter=ctx.reporter,
        warnings=ctx.warnings,
        llm_runtime=ctx.llm_runtime,
    )

    scenes = list(ctx.scenes)
    slide_frames = list(ctx.slide_frames)
    alignment_blocks = ctx.alignment_blocks

    def record_warning(message: str) -> None:
        ctx.warnings.append(message)
        ctx.reporter.warn(message)

    if isinstance(media_asset, VideoAsset):
        detect_scene_total = _estimated_scene_frame_count(media_asset)
        if detect_scene_total is None:
            with stage(ctx, "detect_scenes", "Detecting scenes") as st:
                scenes = video_runtime.detect_scenes(
                    input_path,
                    duration_sec=media_asset.duration_sec,
                )
                st.detail = count_label(len(scenes), "scene")
        else:
            with progress_stage(
                ctx,
                "detect_scenes",
                "Detecting scenes",
                total=float(detect_scene_total),
                count_label="frames",
                detail="0 scenes",
            ) as st:
                scenes = video_runtime.detect_scenes(
                    input_path,
                    duration_sec=media_asset.duration_sec,
                    progress_callback=lambda frame_count, scene_count: st.advance_to(
                        float(frame_count),
                        detail=count_label(scene_count, "scene"),
                    ),
                )
                st.finish_progress(
                    float(detect_scene_total), detail=count_label(len(scenes), "scene")
                )
        write_json(layout.scenes_path, {"scenes": [asdict(scene) for scene in scenes]})

        with progress_stage(
            ctx,
            "extract_frames",
            "Extracting slide frames",
            total=float(len(scenes)),
        ) as st:
            slide_frames = video_runtime.extract_representative_frames(
                input_path,
                scenes,
                layout.frames_dir,
                progress_callback=st.advance,
                warning_callback=record_warning,
            )
            st.detail = count_label(len(slide_frames), "frame")

        alignment_blocks = align_by_time(
            normalized_transcription.segments,
            scenes,
            slide_frames,
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
            alignment_blocks=alignment_blocks,
            warnings=ctx.warnings,
            progress_callback=lambda completed_count, section_count: st.advance_to(
                float(completed_count),
                detail=count_label(section_count, "section"),
            ),
        )
        st.finish_progress(
            float(structure_total),
            detail=count_label(len(report.sections), "section"),
        )

    if slide_frames:
        frame_by_id = {frame.id: frame for frame in slide_frames}
        report = report.model_copy(
            update={
                "sections": [
                    section.model_copy(
                        update={"image_path": frame_by_id[section.frame_id].image_path}
                    )
                    if section.frame_id and section.frame_id in frame_by_id
                    else section
                    for section in report.sections
                ]
            }
        )

    report, ctx.llm_runtime = maybe_polish_report(
        report,
        llm_processor=llm_enhancer,
        ctx=ctx,
        warnings=ctx.warnings,
        llm_runtime=ctx.llm_runtime,
    )
    report = report.model_copy(update={"warnings": list(ctx.warnings)})

    with stage(ctx, "export", "Writing artifacts"):
        export_runtime.write_markdown_report(report, layout.markdown_report_path)
        export_runtime.write_docx_report(
            report,
            layout.docx_report_path,
            warning_callback=record_warning,
        )
        export_runtime.write_json_report(report, layout.json_report_path)
        export_runtime.write_vtt_subtitles(normalized_transcription, layout.subtitle_vtt_path)

    return ReportPhaseResult(
        report=report,
        alignment_blocks=alignment_blocks,
        scenes=scenes,
        slide_frames=slide_frames,
    )


def _estimated_scene_frame_count(media_asset: VideoAsset) -> int | None:
    fps = media_asset.fps
    if fps is None or fps <= 0 or media_asset.duration_sec <= 0:
        return None
    return max(1, math.ceil(media_asset.duration_sec * fps))
