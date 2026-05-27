"""Private report-phase orchestration helpers."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING

import webinar_transcriber.export as export_runtime
import webinar_transcriber.structure as structure_runtime
import webinar_transcriber.video as video_runtime
from webinar_transcriber.models import VideoAsset

from .llm import maybe_polish_report
from .support import count_label, counting_progress, progress_stage, stage, write_json

if TYPE_CHECKING:
    from webinar_transcriber.llm.contracts import LLMProcessor
    from webinar_transcriber.models import (
        AlignmentBlock,
        MediaAsset,
        ReportDocument,
        Scene,
        SceneFrame,
        TranscriptionResult,
    )
    from webinar_transcriber.paths import RunLayout

    from .types import RunContext


def run_report_phase(
    *,
    input_path: Path,
    layout: RunLayout,
    media_asset: MediaAsset,
    normalized_transcription: TranscriptionResult,
    llm_processor: LLMProcessor | None,
    ctx: RunContext,
) -> tuple[ReportDocument, list[Scene], list[SceneFrame]]:
    """Run the video, structure, optional LLM, and export half of the pipeline.

    Returns:
        tuple: The report artifact and video diagnostics inputs.
    """
    scenes: list[Scene] = []
    scene_frames: list[SceneFrame] = []
    alignment_blocks: list[AlignmentBlock] | None = None
    transcript_segments = normalized_transcription.segments

    if isinstance(media_asset, VideoAsset):
        detect_scene_total = video_runtime.estimated_scene_sample_count(media_asset.duration_sec)
        with progress_stage(
            ctx,
            "detect_scenes",
            "Detecting scenes",
            total=float(detect_scene_total),
            count_label="samples",
            detail="0 scenes",
        ) as st:
            scenes = video_runtime.detect_scenes(
                input_path,
                duration_sec=media_asset.duration_sec,
                progress_callback=counting_progress(st, "scene"),
            )
            st.advance_to(float(detect_scene_total), detail=count_label(len(scenes), "scene"))
        write_json(layout.scenes_path, {"scenes": [asdict(scene) for scene in scenes]})

        with progress_stage(
            ctx, "extract_frames", "Extracting scene frames", total=float(len(scenes))
        ) as st:
            scene_frames = video_runtime.extract_representative_frames(
                input_path,
                scenes,
                layout.frames_dir,
                progress_callback=counting_progress(st, "frame"),
                warning_callback=ctx.record_warning,
            )
            st.set_detail(count_label(len(scene_frames), "frame"))

        alignment_blocks = structure_runtime.align_by_time(
            transcript_segments, scenes, scene_frames
        )

    if alignment_blocks is not None:
        structure_item_count = len(alignment_blocks)
        structure_count_label = "blocks"
    else:
        structure_item_count = len(transcript_segments)
        structure_count_label = "segments"
    structure_total = max(structure_item_count, 1)
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
            progress_callback=counting_progress(st, "section"),
        )
        st.advance_to(float(structure_total), detail=count_label(len(report.sections), "section"))

    if scene_frames:
        frame_by_id = {frame.id: frame for frame in scene_frames}
        sections = []
        for section in report.sections:
            frame_id = section.frame_id
            frame = frame_by_id.get(frame_id) if frame_id else None
            sections.append(
                replace(section, image_path=_report_image_path(frame, layout.run_dir))
                if frame
                else section
            )
        report = replace(report, sections=sections)

    report = maybe_polish_report(
        report,
        llm_processor=llm_processor,
        ctx=ctx,
        llm_runtime=ctx.llm_runtime,
    )
    report = replace(report, warnings=list(ctx.warnings))

    with stage(ctx, "export", "Writing artifacts"):
        export_runtime.write_markdown_report(report, layout.markdown_report_path)
        export_runtime.write_docx_report(
            report, layout.docx_report_path, warning_callback=ctx.record_warning
        )
        report = replace(report, warnings=list(ctx.warnings))
        export_runtime.write_json_report(report, layout.json_report_path)

    return report, scenes, scene_frames


def _report_image_path(frame: SceneFrame, run_dir: Path) -> str:
    image_path = Path(frame.image_path)
    try:
        return image_path.relative_to(run_dir).as_posix()
    except ValueError:
        try:
            return image_path.resolve().relative_to(run_dir.resolve()).as_posix()
        except ValueError:
            return frame.image_path
