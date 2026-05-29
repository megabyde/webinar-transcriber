"""Run-diagnostics assembly and persistence helpers."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Literal

from webinar_transcriber.models import AsrPipelineDiagnostics, Diagnostics, LlmDiagnostics
from webinar_transcriber.processor.support import write_json

if TYPE_CHECKING:
    from collections.abc import Sequence

    from webinar_transcriber.models import (
        DiarizationDiagnostics,
        ReportDocument,
        Scene,
        SceneFrame,
        TranscriptionResult,
    )
    from webinar_transcriber.processor.types import RunContext


def build_diagnostics(
    ctx: RunContext,
    *,
    llm_enabled: bool,
    transcription: TranscriptionResult | None = None,
    normalized_transcription: TranscriptionResult | None = None,
    asr_pipeline: AsrPipelineDiagnostics | None = None,
    diarization: DiarizationDiagnostics | None = None,
    report: ReportDocument | None = None,
    scenes: Sequence[Scene] = (),
    scene_frames: Sequence[SceneFrame] = (),
    status: Literal["succeeded", "failed"] = "succeeded",
    failed_stage: str | None = None,
    error: str | None = None,
) -> Diagnostics:
    """Build the final diagnostics payload for one processing run.

    Returns:
        Diagnostics: The final diagnostics payload.
    """
    vad_region_count = asr_pipeline.vad_region_count if asr_pipeline is not None else 0
    window_count = asr_pipeline.window_count if asr_pipeline is not None else 0
    return Diagnostics(
        status=status,
        failed_stage=failed_stage,
        error=error,
        llm=LlmDiagnostics(
            enabled=llm_enabled,
            model=ctx.llm_runtime.model_name,
            report_status=ctx.llm_runtime.report_status,
            report_latency_sec=ctx.llm_runtime.report_latency_sec,
            response_metadata=ctx.llm_runtime.response_metadata,
        ),
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


def write_run_diagnostics(
    ctx: RunContext,
    *,
    status: Literal["succeeded", "failed"],
    llm_enabled: bool,
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
) -> Diagnostics | None:
    """Write diagnostics for the current processor context."""
    if ctx.layout is None:
        return None

    diagnostics = build_diagnostics(
        ctx,
        llm_enabled=llm_enabled,
        transcription=transcription,
        normalized_transcription=normalized_transcription,
        asr_pipeline=asr_pipeline,
        diarization=diarization,
        report=report,
        scenes=scenes,
        scene_frames=scene_frames,
        status=status,
        failed_stage=failed_stage,
        error=error,
    )
    try:
        write_json(ctx.layout.diagnostics_path, asdict(diagnostics))
    except Exception:  # pragma: no cover - best-effort failed-run diagnostics
        if not suppress_errors:
            raise
    return diagnostics
