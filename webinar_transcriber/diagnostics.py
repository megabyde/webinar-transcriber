"""Run-diagnostics assembly and persistence helpers."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Literal

from webinar_transcriber.models import AsrPipelineDiagnostics, Diagnostics, LlmDiagnostics
from webinar_transcriber.processor.stages import write_json

if TYPE_CHECKING:
    from collections.abc import Sequence

    from webinar_transcriber.models import (
        DiarizationDiagnostics,
        ReportDocument,
        Scene,
        SceneFrame,
        TranscriptionResult,
    )
    from webinar_transcriber.paths import RunLayout
    from webinar_transcriber.processor import RunContext


def write_run_diagnostics(
    layout: RunLayout,
    ctx: RunContext,
    *,
    llm: LlmDiagnostics,
    status: Literal["succeeded", "failed"],
    transcription: TranscriptionResult | None = None,
    normalized_transcription: TranscriptionResult | None = None,
    asr_pipeline: AsrPipelineDiagnostics | None = None,
    diarization: DiarizationDiagnostics | None = None,
    report: ReportDocument | None = None,
    scenes: Sequence[Scene] = (),
    scene_frames: Sequence[SceneFrame] = (),
    failed_stage: str | None = None,
    error: str | None = None,
    suppress_errors: bool = False,
) -> Diagnostics:
    """Write diagnostics for the current processor context."""
    vad_region_count = asr_pipeline.vad_region_count if asr_pipeline is not None else 0
    window_count = asr_pipeline.window_count if asr_pipeline is not None else 0
    diagnostics = Diagnostics(
        status=status,
        failed_stage=failed_stage,
        error=error,
        llm=llm,
        stage_durations_sec={key: round(value, 6) for key, value in ctx.stage_timings.items()},
        item_counts={
            "transcript_segments": len(transcription.segments) if transcription else 0,
            "normalized_transcript_segments": (
                len(normalized_transcription.segments) if normalized_transcription else 0
            ),
            "vad_regions": vad_region_count,
            "windows": window_count,
            "report_sections": len(report.sections) if report else 0,
            "scenes": len(scenes),
            "frames": len(scene_frames),
        },
        asr_pipeline=asr_pipeline,
        diarization=diarization,
        warnings=ctx.warnings,
    )
    try:
        write_json(layout.diagnostics_path, asdict(diagnostics))
    except Exception:  # pragma: no cover - best-effort failed-run diagnostics
        if not suppress_errors:
            raise
    return diagnostics
